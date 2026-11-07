"""Simulation-only helpers: the ground-truth supervisor of the map pipeline (CellDropout moved to go2_perception.sim_noise).

Nothing in here is needed on the robot.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .go2_perception.mapping import ElevationMap, shift_grid_
from .go2_perception.sim_noise import CellDropout  # noqa: F401  (re-exported)


class SimMapSupervisor:
    """Ground truth for the map pipeline: true heightfield, fog of war, true pose history, training targets.

    Plugged into ``MapPerception.update(..., supervisor=...)``. Reads the sim (terrain, start poses) through ``env``.
    """

    def __init__(self, env, emap: ElevationMap):
        self.env, self.map, self.device = env, emap, env.device
        N = env.num_envs
        self.true_map = torch.zeros(N, emap.Wc, emap.Hc, device=self.device)  # simulator heightfield, start frame
        self.seen = torch.zeros(N, emap.Wc, emap.Hc, dtype=torch.bool, device=self.device)  # GT fog of war
        self.pose_true_hist = torch.zeros(N, env.pp.odom.pose_hist_len, 4, device=self.device)
        self.true_pose = torch.zeros(N, 4, device=self.device)
        self._ready = False

    # ---------------------------------------------------------------- lifecycle
    def reset(self, ids: torch.Tensor) -> None:
        """Call after the map was reset and the start poses were resampled."""
        self.seen[ids] = False
        self.pose_true_hist[ids] = 0.0
        self.update_true_map(ids)

    def push_true_pose(self, pose: torch.Tensor) -> None:
        self.true_pose = pose
        self.pose_true_hist = torch.roll(self.pose_true_hist, 1, dims=1)
        self.pose_true_hist[:, 0] = pose

    def _gather(self, lag: torch.Tensor | None) -> torch.Tensor:
        if lag is None:
            return self.pose_true_hist[:, 0]
        return self.pose_true_hist[self.env.all_env_ids, lag]

    # ---------------------------------------------------------------- hooks used by MapPerception.update
    def on_shift(self, shifts: torch.Tensor) -> None:
        moved = torch.where((shifts != 0).any(dim=1))[0]
        if moved.numel() == 0:
            return
        for e in moved.tolist():
            shift_grid_(self.seen, e, int(shifts[e, 0]), int(shifts[e, 1]), False)
        self.update_true_map(moved)  # supervision must match the new cell positions

    def on_scan(self, scan: torch.Tensor, emap: ElevationMap, lag) -> None:
        _, gflat, _ = emap.scan_to_cells(scan, self._gather(lag))
        self.seen.view(-1)[gflat] = True

    @torch.no_grad()
    def targets(self, emap: ElevationMap, ids: torch.Tensor | None = None) -> dict:
        """{seen, inb, target} of the ego crop around the TRUE pose (pose pushed this step)."""
        sel = (lambda t: t) if ids is None else (lambda t: t[ids])
        tru, z0, origin = sel(self.true_pose), sel(emap.base_z0), sel(emap.origin)
        g = emap.ego_grid(tru, origin)
        z_true = (tru[:, 2] + z0)[:, None, None]
        target = emap.crop(sel(self.true_map).unsqueeze(1), g)[:, 0] - z_true
        seen = emap.crop(sel(self.seen).float().unsqueeze(1), g)[:, 0] > 0.5
        inb = (g.abs() <= 1.0).all(dim=-1)
        return {"seen": seen, "inb": inb, "target": target}

    def teacher_map(self, true_pose: torch.Tensor) -> torch.Tensor:
        """True map, ego crop at the TRUE pose, heights relative to the true base z. [N,1,n,n]"""
        emap = self.map
        crop = emap.crop(self.true_map.unsqueeze(1), emap.ego_grid(true_pose, emap.origin))
        return crop - (true_pose[:, 2] + emap.base_z0)[:, None, None, None]

    # ---------------------------------------------------------------- true heightfield
    def precompute_heightfield(self) -> None:
        if self._ready:
            return
        from isaaclab.utils.warp import raycast_mesh

        env = self.env
        prim_path = env._height_scanner.cfg.mesh_prim_paths[0]
        warp_mesh = env._height_scanner.meshes[prim_path]
        tile_w, tile_l = env.cfg.terrain.terrain_generator.size

        # Bounds from ALL terrain tiles, so curriculum changes never move an env outside the heightfield.
        origins = getattr(env._terrain, "terrain_origins", None)
        if origins is None:
            origins = env._terrain.env_origins
        origins = origins.reshape(-1, 3)
        margin = 0.5 * max(tile_w, tile_l) + 2.0
        x_min = origins[:, 0].min().item() - margin
        x_max = origins[:, 0].max().item() + margin
        y_min = origins[:, 1].min().item() - margin
        y_max = origins[:, 1].max().item() + margin

        res = float(env.cfg.map_cell_size)
        gx = max(2, int(math.ceil((x_max - x_min) / res)))
        gy = max(2, int(math.ceil((y_max - y_min) / res)))
        xs = x_min + (torch.arange(gx, device=self.device, dtype=torch.float32) + 0.5) * res
        ys = y_min + (torch.arange(gy, device=self.device, dtype=torch.float32) + 0.5) * res
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        ray_starts = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), torch.full_like(grid_x.reshape(-1), 100.0)], dim=-1
        ).unsqueeze(0)
        ray_dirs = torch.zeros_like(ray_starts)
        ray_dirs[..., 2] = -1.0
        hits, *_ = raycast_mesh(ray_starts, ray_dirs, mesh=warp_mesh)
        heights = hits[0, :, 2]
        heights = torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))

        self._hf = heights.reshape(gx, gy)
        self._hf_res, self._hf_x_min, self._hf_y_min = res, x_min, y_min
        self._ready = True

    def update_true_map(self, env_ids: torch.Tensor) -> None:
        """Sample the ground-truth terrain at the current rolling-map cell positions."""
        env, emap = self.env, self.map
        n = env_ids.numel()
        gx, gy = self._hf.shape
        res = self._hf_res
        origin = emap.origin[env_ids]

        ix = torch.arange(emap.Wc, device=self.device, dtype=torch.float32)
        iy = torch.arange(emap.Hc, device=self.device, dtype=torch.float32)
        lx, ly = torch.meshgrid(ix, iy, indexing="ij")
        local_x = origin[:, 0, None, None] + (lx[None] + 0.5) * emap.cs  # start-frame cell centres
        local_y = origin[:, 1, None, None] + (ly[None] + 0.5) * emap.cs

        yaw = env.start_yaw_w[env_ids]
        cy, sy = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
        world_x = cy * local_x - sy * local_y + env.start_pos_w[env_ids, 0, None, None]
        world_y = sy * local_x + cy * local_y + env.start_pos_w[env_ids, 1, None, None]

        gx_norm = (world_x - self._hf_x_min) / (gx * res) * 2.0 - 1.0
        gy_norm = (world_y - self._hf_y_min) / (gy * res) * 2.0 - 1.0
        grid = torch.stack([gy_norm, gx_norm], dim=-1)
        field = self._hf[None, None].expand(n, 1, gx, gy)
        sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=False)
        self.true_map[env_ids] = sampled.squeeze(1) - env._terrain.env_origins[env_ids, 2][:, None, None]