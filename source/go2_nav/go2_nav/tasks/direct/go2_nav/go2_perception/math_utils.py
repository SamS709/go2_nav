"""Small torch math helpers (quaternions are (w, x, y, z), same convention as Isaac Lab)."""

from __future__ import annotations

import torch


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle to [-pi, pi]."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def quat_to_roll_pitch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = q.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v [..., 3] by q [..., 4]."""
    w = q[..., 0:1]
    xyz = q[..., 1:]
    t = torch.cross(xyz, v, dim=-1) * 2.0
    return v + w * t + torch.cross(xyz, t, dim=-1)