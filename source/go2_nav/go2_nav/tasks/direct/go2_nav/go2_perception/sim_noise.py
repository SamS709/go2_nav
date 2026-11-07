"""Sensor randomisation used in simulation only (never needed on the robot). Pure torch.

Shared by the locomotion-training env and the navigation env so both policies see the same sensor model.
"""

from __future__ import annotations

import torch

from .height_grid import HeightGrid


def misalign_xz(points: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
    """Rotate points [N,R,3] in the x-z plane by angle_deg [N] (called ``yaw`` misalignment in the env cfgs)."""
    a = torch.deg2rad(angle_deg).unsqueeze(-1)
    c, s = torch.cos(a), torch.sin(a)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return torch.stack((x * c + z * s, y, -x * s + z * c), dim=-1)


def misalign_yz(points: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
    """Rotate points [N,R,3] in the y-z plane by angle_deg [N] (called ``roll`` misalignment in the env cfgs)."""
    a = torch.deg2rad(angle_deg).unsqueeze(-1)
    c, s = torch.cos(a), torch.sin(a)
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return torch.stack((x, y * c - z * s, y * s + z * c), dim=-1)


class CellDropout:
    """Gaussian-weighted random cell dropout shared by all envs; one distribution per grid.

    ``nav_grid`` is optional (the locomotion-training env only has one grid). ``n_zeros`` is defined for
    ``loc_grid`` and scaled by the cell count for ``nav_grid``.
    """

    def __init__(self, loc_grid: HeightGrid, nav_grid: HeightGrid | None, sigma: float, n_zeros: int,
                 max_reset_freq: int, device):
        self.device = device
        self.sigma = sigma
        self.max_reset_freq = int(max_reset_freq)
        self.prob_loc = self._gaussian_prob(loc_grid.x_cells, loc_grid.y_cells)
        self.n_loc = int(n_zeros)
        self.prob_nav, self.n_nav = None, 0
        if nav_grid is not None:
            self.prob_nav = self._gaussian_prob(nav_grid.x_cells, nav_grid.y_cells)
            self.n_nav = max(1, int(round(n_zeros * nav_grid.num_cells / max(1, loc_grid.num_cells))))
        self._resample()
        self.same_count = 0
        self.reset_freq = int(torch.randint(1, self.max_reset_freq + 1, (1,), device=device).item())

    def _gaussian_prob(self, h: int, w: int) -> torch.Tensor:
        y = torch.arange(h, device=self.device, dtype=torch.float32)
        x = torch.arange(w, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        expo = ((xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2) / (2 * self.sigma**2)
        prob = torch.exp(expo - expo.max())  # same distribution as exp(expo), without float overflow
        return (prob / prob.sum()).flatten()

    def _resample(self) -> None:
        self.idx_loc = torch.multinomial(self.prob_loc, self.n_loc, replacement=True)
        if self.prob_nav is not None:
            self.idx_nav = torch.multinomial(self.prob_nav, self.n_nav, replacement=True)

    def apply(self, flat_height: torch.Tensor, locomotion: bool = True):
        """flat_height [N, cells] -> (height with dropped cells zeroed, keep mask)."""
        self.same_count += 1
        if self.same_count == self.reset_freq:
            self.reset_freq = int(torch.randint(1, self.max_reset_freq + 1, (1,), device=self.device).item())
            self.same_count = 0
            self._resample()
        idx = self.idx_loc if locomotion else self.idx_nav
        out = flat_height.clone()
        keep = torch.ones_like(flat_height, dtype=torch.bool)
        out[:, idx] = 0.0
        keep[:, idx] = False
        return out, keep


class LidarMisalignment:
    """Per-env lidar mounting error: x-z rotation ("yaw"), y-z rotation ("roll") and a 3D offset of the ray hits.

    Applied to rays in the lidar frame BEFORE gridding. Resampled at every reset.
    """

    def __init__(self, num_envs: int, max_rot: float, max_offset: float, device):
        self.max_rot, self.max_offset, self.device = float(max_rot), float(max_offset), device
        self.rots_yaw = torch.empty(num_envs, device=device)
        self.rots_roll = torch.empty(num_envs, device=device)
        self.offsets = torch.empty(num_envs, 3, device=device)
        self.resample(torch.arange(num_envs, device=device))

    def resample(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        self.rots_yaw[env_ids] = torch.empty(n, device=self.device).uniform_(-self.max_rot, self.max_rot)
        self.rots_roll[env_ids] = torch.empty(n, device=self.device).uniform_(-self.max_rot, self.max_rot)
        self.offsets[env_ids] = torch.empty(n, 3, device=self.device).uniform_(-self.max_offset, self.max_offset)

    def apply(self, rays: torch.Tensor) -> torch.Tensor:
        rays = misalign_xz(rays, self.rots_yaw)
        rays = misalign_yz(rays, self.rots_roll)
        return rays + self.offsets.unsqueeze(1)


def sample_lag(n: int, delay_length: int, device) -> torch.Tensor:
    """Sensor lag in control steps, uniform in [0, delay_length] (inclusive). Single convention for all envs."""
    return torch.randint(0, max(0, int(delay_length)) + 1, (n,), device=device)


def set_buffer_lag(buffer, lag: torch.Tensor, batch_ids: torch.Tensor | None = None) -> None:
    """Set the time lag of an Isaac Lab DelayBuffer (its lags are int32)."""
    buffer.set_time_lag(lag.to(torch.int32), batch_ids=batch_ids)


def height_image(grid: HeightGrid, rays: torch.Tensor, dropout: CellDropout, noise: float, apply_dropout: bool,
                 locomotion: bool = True):
    """rays (already misaligned if desired) -> (height [N,X,Y], valid [N,X,Y]) with sim noise and cell dropout."""
    height, hit = grid.compute(rays, noise=noise)
    valid = hit
    if apply_dropout:
        height, keep = dropout.apply(height, locomotion)
        valid = hit & keep
    return grid.to_image(height), grid.to_image(valid)