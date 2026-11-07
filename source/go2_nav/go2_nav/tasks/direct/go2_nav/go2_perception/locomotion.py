"""Frozen low-level locomotion policy (TorchScript) and its observation / action bookkeeping."""

from __future__ import annotations

import os

import torch

# Observation noise of the locomotion policy (ang vel, projected gravity, joint pos, joint vel). Sim only.
PROPRIO_NOISE = (0.1, 0.05, 0.01, 0.1)


def build_proprio(ang_vel, gravity, joint_pos_rel, joint_vel, noise_scale: float = 0.0) -> torch.Tensor:
    """Policy proprio [N,30]. ``noise_scale`` = 0 on the robot, 1.0 (randomize) in simulation."""
    parts = []
    for x, a in zip((ang_vel, gravity, joint_pos_rel, joint_vel), PROPRIO_NOISE):
        if noise_scale > 0.0:
            x = x + (2.0 * torch.rand_like(x) - 1.0) * a * noise_scale
        parts.append(x)
    return torch.cat(parts, dim=-1)


def build_loc_obs(proprio: torch.Tensor, cmd_vel: torch.Tensor, last_actions: torch.Tensor) -> torch.Tensor:
    """Layout of the locomotion actor observation: [proprio(30), cmd(3), last action(12)]."""
    return torch.cat([proprio, cmd_vel, last_actions], dim=-1)


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
        return build_loc_obs(proprio, cmd_vel, self.actions)

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