"""Rolling start-frame elevation map (per-cell Kalman filter) + ego-crop denoiser.

Deployment usage::

    mapper = MapPerception(N, device, params, base_z0, q_base_lidar, t_base_lidar, train=False)
    mapper.load_checkpoint_state(ckpt)
    scan = mapper.build_scan(points_lidar, *odom.roll_pitch)       # points_lidar [N,R,3]: hits rel. to the scanner
    pred_h, conf = mapper.update(scan, odom, lag=None)             # -> mapper.obs [N, 2, n, n]

Everything that needs ground truth (training targets, fog-of-war) is injected through the optional ``supervisor``
object of ``MapPerception.update`` (sim only, see ``sim_support.SimMapSupervisor``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PerceptionParams, MapParams
from .math_utils import quat_apply
from .scan_matching import ScanToMapMatcher
from .training import map_losses, map_metrics


# ====================================================================== networks
class EgoMapDenoiser(nn.Module):
    """Spatial denoiser / completer working on an ego-centric crop of the fused map.

    Input  [B, 4, n, n]: (height relative to the robot * observed, normalised log-variance, observed mask,
                          1/(1+var) confidence).
    Output [B, 2, n, n]: (height residual, log-variance of the residual). ``pred_height = input_height + residual``
                          (zero-initialised residual head -> identity on the Kalman estimate at start).
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32, groups: int = 8):
        super().__init__()

        def block(cin, cout, stride=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
                nn.GroupNorm(min(groups, cout), cout),
                nn.ELU(),
            )

        b = base_channels
        self.enc1 = block(in_channels, b)
        self.enc2 = block(b, 2 * b, stride=2)
        self.enc3 = block(2 * b, 4 * b, stride=2)
        self.bottleneck = nn.Sequential(block(4 * b, 4 * b, dilation=2), block(4 * b, 4 * b, dilation=4))
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = block(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = block(2 * b, b)
        self.head = nn.Conv2d(b, 2, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0, math.log(0.05**2)]))

    @staticmethod
    def _match(up: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != ref.shape[-2:]:
            up = F.interpolate(up, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = self.dec2(torch.cat([self._match(self.up2(b), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        return self.head(d1)


class MapDenoiser:
    """EgoMapDenoiser + optimiser; inference or one online training step per call."""

    def __init__(self, device, p: MapParams, train: bool = False, lr: float = 1e-3):
        self.p = p
        self.train_enabled = bool(train)
        self.model = EgoMapDenoiser(in_channels=4, base_channels=32).to(device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

    @torch.no_grad()
    def infer(self, x: torch.Tensor):
        self.model.eval()
        out = self.model(x)
        return x[:, 0] + out[:, 0], out[:, 1].clamp(-10.0, 4.0)

    def train_step(self, crops: dict):
        """crops: x, observed, seen, inb, target. Returns pred_h, logvar (detached) and a log dict."""
        with torch.inference_mode(False), torch.enable_grad():
            self.model.train()
            c = {k: v.detach().clone() for k, v in crops.items()}
            out = self.model(c["x"])
            pred_h = c["x"][:, 0] + out[:, 0]
            logvar = out[:, 1].clamp(-10.0, 4.0)
            loss, lh, le, ln = map_losses(pred_h, logvar, c, float(self.p.w_complete), float(self.p.w_unseen))
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.eval()
        log = {"map/loss_total": loss.detach(), "map/loss_height": lh.detach(),
               "map/loss_edge": le.detach(), "map/loss_nll": ln.detach()}
        return pred_h.detach(), logvar.detach(), log


# ====================================================================== scan building
class ScanBuilder:
    """Lidar points -> gravity-aligned body-frame scan [N, R, 4] = (x, y, z_rel, valid)."""

    def __init__(self, q_base_lidar: torch.Tensor, t_base_lidar: torch.Tensor, max_range: float):
        self.q = q_base_lidar  # [N,4] rotation scanner -> base (w,x,y,z)
        self.t = t_base_lidar  # [N,3] scanner origin in the base frame
        self.max_range = float(max_range)

    def build(self, points_lidar: torch.Tensor, roll: torch.Tensor, pitch: torch.Tensor, noise_std: float = 0.0):
        """points_lidar [N,R,3]: hit relative to the scanner origin, expressed in the scanner frame."""
        n, r, _ = points_lidar.shape
        q = self.q[:, None, :].expand(n, r, 4)
        p_b = quat_apply(q.reshape(-1, 4), points_lidar.reshape(-1, 3)).reshape(n, r, 3) + self.t[:, None, :]

        cr, sr = torch.cos(roll)[:, None], torch.sin(roll)[:, None]
        cp, sp = torch.cos(pitch)[:, None], torch.sin(pitch)[:, None]
        x_b, y_b, z_b = p_b.unbind(-1)
        p_g = torch.stack(
            [
                cp * x_b + sp * sr * y_b + sp * cr * z_b,
                cr * y_b - sr * z_b,
                -sp * x_b + cp * sr * y_b + cp * cr * z_b,
            ],
            dim=-1,
        )
        x_b, y_b, z_rel = p_g.unbind(-1)
        valid = torch.isfinite(p_g).all(dim=-1) & ((x_b * x_b + y_b * y_b) <= self.max_range**2)
        if noise_std > 0.0:
            z_rel = z_rel + torch.randn_like(z_rel) * noise_std
        scan = torch.stack([x_b, y_b, z_rel, valid.float()], dim=-1)
        return torch.where(valid[..., None], scan, torch.zeros_like(scan))


# ====================================================================== rolling map
def shift_grid_(tensor: torch.Tensor, env_id: int, sx: int, sy: int, fill) -> None:
    """Shift a [N,Wc,Hc] tensor of one env in place; new cell (i,j) reads old cell (i+sx, j+sy)."""
    Wc, Hc = tensor.shape[1:3]
    if abs(sx) >= Wc or abs(sy) >= Hc:
        tensor[env_id].fill_(fill)
        return
    if sx != 0:
        tensor[env_id] = torch.roll(tensor[env_id], shifts=-sx, dims=0)
        if sx > 0:
            tensor[env_id, -sx:, :] = fill
        else:
            tensor[env_id, :-sx, :] = fill
    if sy != 0:
        tensor[env_id] = torch.roll(tensor[env_id], shifts=-sy, dims=1)
        if sy > 0:
            tensor[env_id, :, -sy:] = fill
        else:
            tensor[env_id, :, :-sy] = fill


class ElevationMap:
    """Per-env rolling map expressed in the (fixed) start frame; only storage origin / indices move."""

    def __init__(self, num_envs: int, device, p: MapParams, base_z0: torch.Tensor):
        self.N, self.device, self.p = num_envs, device, p
        self.cs = float(p.cell_size)
        self.Wc = int(round(float(p.width) / self.cs))
        self.Hc = int(round(float(p.height) / self.cs))
        self.Wm, self.Hm = self.Wc * self.cs, self.Hc * self.cs
        self.WH = self.Wc * self.Hc
        self.crop_n = int(p.crop_cells)
        self.meas_var = float(p.meas_std) ** 2
        self.base_z0 = base_z0  # [N] nominal base height above the ground at spawn
        self.env_ids = torch.arange(num_envs, device=device)

        self.origin = torch.empty(num_envs, 2, device=device)
        self.raw_map = torch.zeros(num_envs, self.Wc, self.Hc, device=device)  # Kalman height, tile-relative z
        self.variance = torch.full_like(self.raw_map, 1.0e6)
        self._fuse_sum = torch.zeros(num_envs * self.WH, device=device)
        self._fuse_cnt = torch.zeros(num_envs * self.WH, device=device)
        self.reset(self.env_ids)
        self.last_log: dict = {}

    def reset(self, ids: torch.Tensor) -> None:
        self.origin[ids, 0] = -0.5 * self.Wm
        self.origin[ids, 1] = -0.5 * self.Hm
        self.raw_map[ids] = 0.0
        self.variance[ids] = 1.0e6

    @torch.no_grad()
    def maybe_shift_window(self, pose_est: torch.Tensor) -> torch.Tensor:
        """Move the window of every env whose robot approaches a border. Returns the shifts [N,2] (long, cells)."""
        scan_margin = int(math.ceil(float(self.p.scan_max_range) / self.cs))
        crop_margin = self.crop_n // 2 + 2
        mx = min(max(scan_margin, crop_margin), max(1, self.Wc // 3))
        my = min(max(scan_margin, crop_margin), max(1, self.Hc // 3))

        ix = torch.floor((pose_est[:, 0] - self.origin[:, 0]) / self.cs).long()
        iy = torch.floor((pose_est[:, 1] - self.origin[:, 1]) / self.cs).long()
        need_x = (ix < mx) | (ix >= self.Wc - mx)
        need_y = (iy < my) | (iy >= self.Hc - my)
        need = need_x | need_y

        shifts = torch.zeros(self.N, 2, dtype=torch.long, device=self.device)
        for e in torch.where(need)[0].tolist():
            sx = int(ix[e].item() - self.Wc // 2) if need_x[e] else 0
            sy = int(iy[e].item() - self.Hc // 2) if need_y[e] else 0
            if sx == 0 and sy == 0:
                continue
            shift_grid_(self.raw_map, e, sx, sy, 0.0)
            shift_grid_(self.variance, e, sx, sy, 1.0e6)
            self.origin[e, 0] += sx * self.cs
            self.origin[e, 1] += sy * self.cs
            s, t = e * self.WH, (e + 1) * self.WH
            self._fuse_sum[s:t] = 0.0
            self._fuse_cnt[s:t] = 0.0
            shifts[e, 0], shifts[e, 1] = sx, sy

        self.last_log = {
            "map/shift_rate": need.float().mean(),
            "map/mean_origin_x": self.origin[:, 0].mean(),
            "map/mean_origin_y": self.origin[:, 1].mean(),
        }
        return shifts

    def scan_to_cells(self, scan: torch.Tensor, pose: torch.Tensor):
        """Place body-frame points into the map with ``pose`` (start frame). Returns (env, flat cell idx, height)."""
        valid = scan[..., 3] > 0.5
        c = torch.cos(pose[:, 3])[:, None]
        s = torch.sin(pose[:, 3])[:, None]
        x = pose[:, 0, None] + c * scan[..., 0] - s * scan[..., 1]
        y = pose[:, 1, None] + s * scan[..., 0] + c * scan[..., 1]
        ix = torch.floor((x - self.origin[:, 0, None]) / self.cs).long()
        iy = torch.floor((y - self.origin[:, 1, None]) / self.cs).long()
        ok = valid & (ix >= 0) & (ix < self.Wc) & (iy >= 0) & (iy < self.Hc)
        h = scan[..., 2] + (pose[:, 2] + self.base_z0)[:, None]
        env = self.env_ids[:, None].expand_as(ok)
        gflat = env * self.WH + ix * self.Hc + iy
        return env[ok], gflat[ok], h[ok]

    def fuse(self, env: torch.Tensor, gflat: torch.Tensor, meas: torch.Tensor, drift_dist: torch.Tensor) -> None:
        """Vectorised per-cell Kalman update. Hits falling in the same cell are averaged first."""
        if gflat.numel() == 0:
            return
        self._fuse_sum.scatter_add_(0, gflat, meas)
        self._fuse_cnt.scatter_add_(0, gflat, torch.ones_like(meas))
        cnt = self._fuse_cnt[gflat]
        avg = self._fuse_sum[gflat] / cnt
        prior_var = self.variance.view(-1)[gflat]
        meas_var = (self.meas_var + (float(self.p.drift_var_gain) * drift_dist[env]) ** 2) / cnt
        gain = prior_var / (prior_var + meas_var)
        prior_mean = self.raw_map.view(-1)[gflat]
        # duplicated indices carry identical values, so the (non-deterministic) write order is irrelevant
        self.raw_map.view(-1)[gflat] = prior_mean + gain * (avg - prior_mean)
        self.variance.view(-1)[gflat] = ((1.0 - gain) * prior_var).clamp_min(float(self.p.min_variance))
        self._fuse_sum[gflat] = 0.0
        self._fuse_cnt[gflat] = 0.0

    # ----- ego crops -----------------------------------------------------------------------
    def ego_grid(self, pose: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """Normalised grid_sample coordinates of an ego crop centred on ``pose`` (rotated by its yaw)."""
        n = self.crop_n
        u = (torch.arange(n, device=self.device) - n / 2 + 0.5) * self.cs
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        c = torch.cos(pose[:, 3])[:, None, None]
        s = torch.sin(pose[:, 3])[:, None, None]
        x = pose[:, 0, None, None] + c * uu - s * vv
        y = pose[:, 1, None, None] + s * uu + c * vv
        gx = 2.0 * (x - origin[:, 0, None, None]) / self.Wm - 1.0
        gy = 2.0 * (y - origin[:, 1, None, None]) / self.Hm - 1.0
        return torch.stack([gy, gx], dim=-1)  # grid_sample: last dim indexes y first

    @staticmethod
    def crop(maps: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(maps, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    @torch.no_grad()
    def input_crop(self, pose_est: torch.Tensor, ids: torch.Tensor | None = None) -> dict:
        """Denoiser input at the estimated pose. Returns {x [B,4,n,n], observed [B,n,n]}."""
        sel = (lambda t: t) if ids is None else (lambda t: t[ids])
        est, z0, origin = sel(pose_est), sel(self.base_z0), sel(self.origin)
        g = self.ego_grid(est, origin)
        h = self.crop(sel(self.raw_map).unsqueeze(1), g)[:, 0]
        var = self.crop(sel(self.variance).unsqueeze(1), g)[:, 0]
        var = torch.where(var > 0.0, var, torch.full_like(var, 1.0e6))  # zero padding -> unobserved
        observed = var < float(self.p.observed_var_thresh)
        z_est = (est[:, 2] + z0)[:, None, None]
        rel = torch.where(observed, h, torch.zeros_like(h)) - z_est  # unobserved: flat-ground prior
        logv = (torch.log(var.clamp(1e-6, 1e6)).clamp(-8.0, 14.0) + 5.0) / 5.0
        conf_in = 1.0 / (1.0 + var)
        return {"x": torch.stack([rel, logv, observed.float(), conf_in], dim=1), "observed": observed}


# ====================================================================== orchestrator
class MapPerception:
    """ElevationMap + ScanToMapMatcher + MapDenoiser, run in the order of the original pipeline."""

    def __init__(self, num_envs, device, params: PerceptionParams, base_z0, q_base_lidar, t_base_lidar, train=False):
        self.N, self.device, self.params = num_envs, device, params
        self.map = ElevationMap(num_envs, device, params.map, base_z0)
        self.matcher = ScanToMapMatcher(self.map, params.match)
        self.denoiser = MapDenoiser(device, params.map, train=train, lr=params.train.map_lr)
        self.scan_builder = ScanBuilder(q_base_lidar, t_base_lidar, params.map.scan_max_range)
        n = self.map.crop_n
        self.obs = torch.zeros(num_envs, 2, n, n, device=device)  # planner obs: [height rel. to robot, confidence]
        self._step = 0
        self.last_log: dict = {}

    def reset(self, ids: torch.Tensor) -> None:
        self.map.reset(ids)
        self.obs[ids] = 0.0

    def checkpoint_state(self) -> dict:
        return {"map": self.denoiser.model.state_dict()}

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.denoiser.model.load_state_dict(ckpt["map"])

    def build_scan(self, points_lidar, roll, pitch, noise_std: float = 0.0) -> torch.Tensor:
        return self.scan_builder.build(points_lidar, roll, pitch, noise_std)

    @torch.no_grad()
    def update(self, scan: torch.Tensor, odom, lag: torch.Tensor | None = None, supervisor=None):
        """Steps: shift window -> scan matching (corrects ``odom``) -> Kalman fusion -> ego crop -> denoiser.

        scan   [N,R,4]  body-frame scan, already delayed if sensor latency is emulated
        lag    [N]      control steps between scan capture and now (index into the pose history) or None
        supervisor      sim-only object with ``on_shift(shifts)``, ``on_scan(scan, emap, lag)`` and
                        ``targets(emap)``; enables fog-of-war bookkeeping, metrics and online training.
        """
        p = self.params
        log: dict = {}
        shifts = self.map.maybe_shift_window(odom.pose_est)
        log.update(self.map.last_log)
        if supervisor is not None:
            supervisor.on_shift(shifts)

        nav_pose = odom.pose_at_lag(lag)
        if p.match.enabled and self._step % max(1, int(p.match.period)) == 0:
            corr, accepted = self.matcher.match(scan, nav_pose)
            odom.apply_correction(corr, accepted)
            nav_pose = odom.pose_at_lag(lag)
            log.update({
                "match/accept_rate": accepted.float().mean(),
                "match/mean_abs_dxy": corr[:, :2].abs().sum(-1).mean(),
                "match/mean_abs_dz": corr[:, 2].abs().mean(),
            })

        env, gflat, h = self.map.scan_to_cells(scan, nav_pose)
        self.map.fuse(env, gflat, h, odom.drift_dist)
        if supervisor is not None:
            supervisor.on_scan(scan, self.map, lag)

        crops = self.map.input_crop(odom.pose_est)
        if supervisor is not None:
            crops.update(supervisor.targets(self.map))
            if self.denoiser.train_enabled:
                pred_h, logvar, tlog = self.denoiser.train_step(crops)
                log.update(tlog)
            else:
                pred_h, logvar = self.denoiser.infer(crops["x"])
            log.update(map_metrics(crops, pred_h))
        else:
            pred_h, logvar = self.denoiser.infer(crops["x"])

        ref = float(p.map.conf_ref_std)
        conf = 1.0 / (1.0 + torch.exp(logvar) / ref**2)
        self.obs[:, 0] = pred_h
        self.obs[:, 1] = conf
        self._step += 1
        self.last_log = log
        return pred_h, conf