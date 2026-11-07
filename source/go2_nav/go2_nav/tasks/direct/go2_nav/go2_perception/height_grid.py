"""Ray hits (lidar frame) -> 2D height grid, policy convention (shared by locomotion, nav-student and teacher)."""

from __future__ import annotations

import torch

from .math_utils import quat_apply, quat_conjugate


def rays_in_lidar_frame(ray_hits_w: torch.Tensor, lidar_pos_w: torch.Tensor, lidar_quat_w: torch.Tensor) -> torch.Tensor:
    """World-frame ray hits [N,R,3] -> policy-convention rays in the lidar frame: -(hit rel. to scanner)."""
    n, r, _ = ray_hits_w.shape
    rel_w = -ray_hits_w + lidar_pos_w.unsqueeze(1)
    q = quat_conjugate(lidar_quat_w).unsqueeze(1).expand(n, r, 4)
    return quat_apply(q, rel_w)


class HeightGrid:
    def __init__(self, x_range, y_range, cell_size: float, desired_base_height: float, device):
        self.device = device
        self.inv_cell = 1.0 / float(cell_size)
        self.x_min, self.y_min = float(x_range[0]), float(y_range[0])
        self.x_cells = max(1, int((float(x_range[1]) - float(x_range[0])) * self.inv_cell))
        self.y_cells = max(1, int((float(y_range[1]) - float(y_range[0])) * self.inv_cell))
        self.num_cells = self.x_cells * self.y_cells
        self.desired_base_height = float(desired_base_height)

    def compute(self, rays_lidar: torch.Tensor, noise: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """rays_lidar [N, R, 3] (= -(hit in lidar frame), policy convention).

        Returns the FLAT height map [N, cells] and the hit mask [N, cells]. Cell dropout (sim only) is applied by the
        caller on the flat map, then ``to_image`` reshapes.
        """
        n, r, _ = rays_lidar.shape
        zeros = lambda dt: torch.zeros((n, self.num_cells), dtype=dt, device=self.device)

        rays_flat = rays_lidar.reshape(-1, 3)
        env_ids = torch.arange(n, device=self.device).unsqueeze(1).expand(n, r).reshape(-1)
        valid = torch.isfinite(rays_flat).all(dim=1)
        if not torch.any(valid):
            return zeros(torch.float32), zeros(torch.bool)
        rays_valid, env_ids = rays_flat[valid], env_ids[valid]

        xi = torch.floor((rays_valid[:, 0] - self.x_min) * self.inv_cell).long()
        yi = torch.floor((rays_valid[:, 1] - self.y_min) * self.inv_cell).long()
        inb = (xi >= 0) & (xi < self.x_cells) & (yi >= 0) & (yi < self.y_cells)
        if not torch.any(inb):
            return zeros(torch.float32), zeros(torch.bool)
        xi, yi, env_ids, z = xi[inb], yi[inb], env_ids[inb], rays_valid[inb, 2]

        flat_idx = env_ids * self.num_cells + xi * self.y_cells + yi
        hm = torch.full((n * self.num_cells,), -torch.inf, device=self.device)
        hm.scatter_reduce_(0, flat_idx, z, reduce="amax", include_self=True)
        hm = hm.reshape(n, self.num_cells) + self.desired_base_height
        hit = torch.isfinite(hm)  # capture BEFORE the where() destroys it
        hm = torch.where(hit, -hm, torch.zeros_like(hm))
        if noise > 0.0:
            hm = hm + (2.0 * torch.rand_like(hm) - 1.0) * noise
        return hm, hit

    def to_image(self, flat: torch.Tensor) -> torch.Tensor:
        """[N, cells] -> [N, x_cells, y_cells], flipped along x (policy convention)."""
        return flat.view(flat.shape[0], self.x_cells, self.y_cells).flip(dims=[1])