"""Scan-to-map matching: small brute-force search of (dx, dy, dyaw) + z offset against the known map cells."""

from __future__ import annotations

import math

import torch

from .config import MatchParams


class ScanToMapMatcher:
    def __init__(self, emap, params: MatchParams):
        self.map = emap  # ElevationMap (read only)
        self.p = params
        dev = emap.device
        sx, sy, sa = int(params.xy_steps), int(params.xy_steps), int(params.yaw_steps)
        assert sx % 2 == 1 and sa % 2 == 1, "match xy_steps and yaw_steps must be odd"
        ox = torch.linspace(-float(params.xy_range), float(params.xy_range), sx, device=dev)
        oy = torch.linspace(-float(params.xy_range), float(params.xy_range), sy, device=dev)
        yr = math.radians(float(params.yaw_range_deg))
        oa = torch.linspace(-yr, yr, sa, device=dev)
        gx, gy, ga = torch.meshgrid(ox, oy, oa, indexing="ij")
        self.offs = torch.stack([gx, gy, ga], dim=-1).reshape(-1, 3)
        self.shape = (sx, sy, sa)
        self.zero_idx = (sx // 2) * sy * sa + (sy // 2) * sa + sa // 2
        self._idx = None
        self._idx_r = -1

    def _sub_idx(self, num_rays: int) -> torch.Tensor:
        if self._idx is None or self._idx_r != num_rays:
            m = min(num_rays, int(self.p.max_points))
            self._idx = torch.linspace(0, num_rays - 1, m, device=self.map.device).round().long()
            self._idx_r = num_rays
        return self._idx

    @torch.no_grad()
    def _chunk(self, scan: torch.Tensor, pose: torch.Tensor, ids: torch.Tensor):
        """Score K candidate offsets of the scan against the known map cells (chi-square-like score).

        scan [n, M, 4] body frame, pose [n, 4] (delayed est pose), ids [n] env ids.
        Returns correction [n, 4] = (dx, dy, dz, dyaw) in the start frame and a bool mask "xy/yaw accepted".
        """
        pp, m = self.p, self.map
        n, M, _ = scan.shape
        offs = self.offs
        yaw = pose[:, 3, None] + offs[None, :, 2]  # [n,K]
        c = torch.cos(yaw)[..., None]
        s = torch.sin(yaw)[..., None]
        bx = scan[:, None, :, 0]
        by = scan[:, None, :, 1]
        x = pose[:, 0, None, None] + offs[None, :, 0, None] + c * bx - s * by  # [n,K,M]
        y = pose[:, 1, None, None] + offs[None, :, 1, None] + s * bx + c * by
        origin = m.origin[ids]
        ix = torch.floor((x - origin[:, None, None, 0]) / m.cs).long()
        iy = torch.floor((y - origin[:, None, None, 1]) / m.cs).long()
        inb = (ix >= 0) & (ix < m.Wc) & (iy >= 0) & (iy < m.Hc)
        gidx = (ids[:, None, None] * m.WH + ix.clamp(0, m.Wc - 1) * m.Hc + iy.clamp(0, m.Hc - 1)).reshape(-1)
        K = offs.shape[0]
        m_h = m.raw_map.view(-1)[gidx].view(n, K, M)
        m_v = m.variance.view(-1)[gidx].view(n, K, M)
        z = scan[:, None, :, 2] + (pose[:, 2] + m.base_z0[ids])[:, None, None]
        valid = scan[:, None, :, 3] > 0.5
        known = inb & valid & (m_v < float(pp.var_thresh))

        w = known.float() / (m_v + m.meas_var)
        mu = (w * (z - m_h)).sum(-1) / w.sum(-1).clamp(min=1e-9)  # per-candidate mean height error (z drift)
        resid = z - m_h - mu[..., None]  # z is decoupled from the xy/yaw search
        clip = float(pp.err_clip)
        err = (resid**2 / (m_v + m.meas_var)).clamp(max=clip)
        nk = known.sum(-1)  # [n,K]
        score = (err * known).sum(-1) / nk.clamp(min=1)
        score = torch.where(nk >= int(pp.min_points), score, torch.full_like(score, clip))

        sx, sy, sa = self.shape
        S = score.reshape(n, sx, sy, sa)
        best_score, best = score.min(dim=1)
        zero_score = score[:, self.zero_idx]
        nk_best = nk.gather(1, best[:, None]).squeeze(1)

        # per-axis observability: the score profile along an axis (min over the others) must not be flat
        thr = float(pp.axis_obs_thresh)
        prof_x = S.amin(dim=(2, 3))
        prof_y = S.amin(dim=(1, 3))
        prof_a = S.amin(dim=(1, 2))
        obs_x = (prof_x.amax(-1) - prof_x.amin(-1)) > thr
        obs_y = (prof_y.amax(-1) - prof_y.amin(-1)) > thr
        obs_a = (prof_a.amax(-1) - prof_a.amin(-1)) > thr

        good = (
            (nk_best >= int(pp.min_points))
            & (best_score < clip * 0.999)
            & ((zero_score - best_score) > float(pp.improve_thresh))
        )
        off = offs[best]  # [n,3]
        gain = float(pp.gain)
        dx = gain * off[:, 0] * obs_x.float() * good.float()
        dy = gain * off[:, 1] * obs_y.float() * good.float()
        dyaw = gain * off[:, 2] * obs_a.float() * good.float()

        # z: mean height error at the chosen candidate if the xy match is trusted, else at zero offset
        mu_best = mu.gather(1, best[:, None]).squeeze(1)
        mu_zero = mu[:, self.zero_idx]
        nk_zero = nk[:, self.zero_idx]
        mu_sel = torch.where(good, mu_best, mu_zero)
        z_ok = torch.where(good, nk_best, nk_zero) >= int(pp.min_points)
        dz = -float(pp.gain_z) * mu_sel * z_ok.float()

        corr = torch.stack([dx, dy, dz, dyaw], dim=-1)
        accepted = good & (obs_x | obs_y | obs_a)
        return corr, accepted

    @torch.no_grad()
    def match(self, scan: torch.Tensor, pose: torch.Tensor):
        """scan [N, R, 4] (body frame), pose [N, 4] -> corr [N, 4], accepted [N]."""
        N = self.map.N
        sub = scan[:, self._sub_idx(scan.shape[1])]
        corr = torch.zeros(N, 4, device=self.map.device)
        accepted = torch.zeros(N, dtype=torch.bool, device=self.map.device)
        chunk = int(self.p.chunk)
        for start in range(0, N, chunk):
            ids = self.map.env_ids[start : start + chunk]
            c, a = self._chunk(sub[ids], pose[ids], ids)
            corr[ids] = c
            accepted[ids] = a
        return corr, accepted