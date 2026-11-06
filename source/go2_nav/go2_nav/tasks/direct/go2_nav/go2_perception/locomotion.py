"""Frozen low-level locomotion policy (TorchScript) and its observation / action bookkeeping."""

from __future__ import annotations

import os

import torch


class LocomotionController:
    """Wraps the TorchScript locomotion policy.

    Per control step:
        obs = loc.build_obs(proprio, cmd_vel)       # proprio [N,30] = (ang_vel, gravity, joint_pos_rel, joint_vel)
        loc.act(obs, height_grid)                   # height_grid [N, 2, X, Y] = (height, valid)
        targets = loc.joint_targets(default_joint_pos)
    """

    def __init__(self, policy_path: str, num_envs: int, num_joints: int, action_scale: float, device, hidden_dim: int = 128):
        self.device = device
        self.num_envs = num_envs
        self.action_scale = float(action_scale)

        path = os.path.expanduser(policy_path)
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Could not find locomotion policy at '{path}' (expected a TorchScript file).")
        self.policy = torch.jit.load(path, map_location=device)
        self.policy.eval()
        with torch.no_grad():
            self.policy.hidden_state = torch.zeros(
                1, num_envs, hidden_dim, dtype=self.policy.hidden_state.dtype, device=self.policy.hidden_state.device
            )

        self.actions = torch.zeros(num_envs, num_joints, device=device)
        self.prev_actions = torch.zeros_like(self.actions)

    def reset(self, env_ids: torch.Tensor) -> None:
        with torch.no_grad():
            self.policy.hidden_state[:, env_ids, :] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

    def build_obs(self, proprio: torch.Tensor, cmd_vel: torch.Tensor) -> torch.Tensor:
        """Uses the PREVIOUS action (call before ``act``)."""
        return torch.cat([proprio, cmd_vel, self.actions], dim=-1)

    def act(self, obs: torch.Tensor, height_grid: torch.Tensor) -> torch.Tensor:
        self.prev_actions.copy_(self.actions)
        with torch.inference_mode():
            out = self.policy(obs, [height_grid])
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.actions.copy_(out.to(self.device))
        return self.actions

    def joint_targets(self, default_joint_pos: torch.Tensor) -> torch.Tensor:
        return default_joint_pos + self.action_scale * self.actions