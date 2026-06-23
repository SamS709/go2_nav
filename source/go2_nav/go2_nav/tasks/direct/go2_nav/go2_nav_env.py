# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, sample_uniform

from .go2_nav_env_cfg import Go2NavEnvCfg


class Go2NavEnv(DirectRLEnv):
    cfg: Go2NavEnvCfg

    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = int(self.cfg.action_space)
        self._num_joints = int(self.robot.data.default_joint_pos.shape[-1])

        self.actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)

        self.predicted_odom = torch.zeros(self.num_envs, 9, device=self.device)
        self.predicted_pos_history = torch.zeros(
            self.num_envs, int(self.cfg.planner_history_len), 3, device=self.device
        )

        self.planner_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_planner_cmd = torch.zeros_like(self.planner_cmd)

        self.low_level_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self.prev_low_level_actions = torch.zeros_like(self.low_level_actions)
        self.low_level_joint_targets = self.robot.data.default_joint_pos.clone()

        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self.prev_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._lidar_x_cells = max(
            1, int((float(self.cfg.x_range[1]) - float(self.cfg.x_range[0])) / float(self.cfg.lidar_cell_size))
        )
        self._lidar_y_cells = max(
            1, int((float(self.cfg.lidar_y_range[1]) - float(self.cfg.lidar_y_range[0])) / float(self.cfg.lidar_cell_size))
        )
        self._lidar_num_cells = self._lidar_x_cells * self._lidar_y_cells
        self._latest_lidar_obs = torch.zeros(self.num_envs, self._lidar_num_cells, device=self.device)

        self._lidar_x_cells_loc = max(
            1,
            int(
                (float(self.cfg.x_range[1]) - float(self.cfg.x_range[0]))
                / float(self.cfg.res)
            ),
        )
        self._lidar_y_cells_loc = max(
            1,
            int(
                (float(self.cfg.y_range[1]) - float(self.cfg.y_range[0]))
                / float(self.cfg.res)
            ),
        )
        self._lidar_num_cells_loc = self._lidar_x_cells_loc * self._lidar_y_cells_loc

        self._finite_warn_counter = 0
        self.locomotion_policy = self._load_locomotion_policy(self.cfg.locomotion_policy_path)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self._height_scanner = self.cfg.height_scanner.class_type(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        x_cells = max(1, int((self.cfg.x_range[1] - self.cfg.x_range[0]) / self.cfg.res))
        y_cells = max(1, int((self.cfg.y_range[1] - self.cfg.y_range[0]) / self.cfg.res))
        self._create_gaussian_heightmap(x_cells, y_cells)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        

    def _load_locomotion_policy(self, policy_path: str):
        resolved_path = os.path.expanduser(policy_path)
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(os.getcwd(), resolved_path)

        if not os.path.isfile(resolved_path):
            if self.cfg.require_locomotion_policy:
                raise FileNotFoundError(
                    f"Could not find locomotion policy at '{resolved_path}'. "
                    "Set Go2NavEnvCfg.locomotion_policy_path to a valid TorchScript file."
                )
            print(
                f"[WARN] Locomotion policy file not found at '{resolved_path}'. "
                "Falling back to a simple command-to-joint heuristic controller."
            )
            return None

        policy = torch.jit.load(resolved_path, map_location=self.device)
        policy.eval()
        return policy


    def _create_gaussian_heightmap(self, h, w):
        y = torch.arange(h, device=self.device, dtype=torch.float32)
        x = torch.arange(w, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Center of the grid
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        
        # Compute 2D Gaussian
        gaussian_dist = torch.exp(((xx - cx)**2 + (yy - cy)**2) / (2 * self.cfg.sigma**2))
        
        # Normalize to create probability distribution
        gaussian_prob = gaussian_dist / gaussian_dist.sum()
        self.gaussian_prob_heightmap  = gaussian_prob.flatten()
        self.sampled_indices = torch.multinomial(self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True)
        self.same_zeros_count = 0
        self.reset_zeros_freq = int(torch.randint(1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device).item())        
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions)

        self.predicted_odom.copy_(self.actions[:, :9])
        self._update_predicted_position_history(self.actions[:, :3])

        cmd_limits = torch.tensor(self.cfg.locomotion_cmd_limits, device=self.device, dtype=self.actions.dtype)
        self.prev_planner_cmd.copy_(self.planner_cmd)
        self.planner_cmd.copy_(torch.tanh(self.actions[:, 9:12]) * cmd_limits.unsqueeze(0))

        
        proprio_obs_loc, height_data_loc = self._build_locomotion_observations(self.planner_cmd)

        self.prev_low_level_actions.copy_(self.low_level_actions)
        self.low_level_actions.copy_(self._query_locomotion_policy(proprio_obs_loc, height_data_loc))

        self.low_level_joint_targets = (
            self.robot.data.default_joint_pos + self.cfg.locomotion_action_scale * self.low_level_actions
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.low_level_joint_targets)

    def _get_observations(self) -> dict:
        lidar_obs = self._compute_lidar_height_map(locomotion=False)
        height_scan = lidar_obs.reshape(self.num_envs, 1, self._lidar_x_cells, self._lidar_y_cells)

        student_proprio = self.predicted_pos_history.reshape(self.num_envs, -1)

        true_odom = self._get_true_odom()
        goal_delta = self.goal_pos_w - self.robot.data.root_pos_w
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self.robot.data.root_quat_w))
        goal_heading = torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)

        teacher_proprio = torch.cat(
            [student_proprio, true_odom, goal_delta, goal_heading, self.planner_cmd],
            dim=-1,
        )

        student_proprio = self._sanitize_tensor(student_proprio, "student_proprio", clamp_abs=100.0)
        teacher_proprio = self._sanitize_tensor(teacher_proprio, "teacher_proprio", clamp_abs=100.0)
        student_height_scan = self._sanitize_tensor(height_scan, "student_height_scan", clamp_abs=10.0)
        teacher_height_scan = self._sanitize_tensor(height_scan, "teacher_height_scan", clamp_abs=10.0)

        return {
            "student_proprio": student_proprio,
            "student_height_scan": student_height_scan,
            "teacher_proprio": teacher_proprio,
            "teacher_height_scan": teacher_height_scan,
        }

    def _get_rewards(self) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        goal_delta_xy = self.goal_pos_w[:, :2] - root_pos_w[:, :2]
        goal_distance = torch.linalg.norm(goal_delta_xy, dim=-1)
        goal_progress = self.prev_goal_distance - goal_distance

        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self.robot.data.root_quat_w))

        rew_goal_distance = self.cfg.rew_scale_goal_distance * torch.exp(
            -goal_distance / max(1e-4, float(self.cfg.goal_distance_sigma))
        )
        rew_goal_progress = self.cfg.rew_scale_goal_progress * goal_progress
        rew_goal_orientation = self.cfg.rew_scale_goal_orientation * torch.exp(
            -torch.square(yaw_error) / max(1e-4, float(self.cfg.goal_orientation_sigma))
        )

        rew_time_penalty = self.cfg.rew_scale_time_penalty * torch.ones_like(goal_distance)

        true_odom = self._get_true_odom()
        odom_error = torch.mean(torch.square(self.predicted_odom - true_odom), dim=-1)
        rew_odom_prediction = self.cfg.rew_scale_odom_prediction * torch.exp(
            -odom_error / max(1e-4, float(self.cfg.odom_prediction_scale))
        )

        cmd_rate = torch.sum(torch.square(self.planner_cmd - self.prev_planner_cmd), dim=-1)
        cmd_mag = torch.sum(torch.square(self.planner_cmd), dim=-1)
        rew_cmd_smoothness = self.cfg.rew_scale_cmd_smoothness * cmd_rate
        rew_cmd_magnitude = self.cfg.rew_scale_cmd_magnitude * cmd_mag

        upright = torch.square(torch.clamp(-self.robot.data.projected_gravity_b[:, 2], min=0.0, max=1.0))
        rew_upright = self.cfg.rew_scale_upright * upright

        base_too_low = root_pos_w[:, 2] < self.cfg.min_base_height
        tipped = self.robot.data.projected_gravity_b[:, 2] > self.cfg.max_projected_gravity_z
        failed = base_too_low | tipped
        rew_termination = self.cfg.rew_scale_terminated * failed.float()
        rew_goal_bonus = self.cfg.rew_scale_goal_bonus * self.goal_reached.float()

        total_reward = (
            rew_goal_distance
            + rew_goal_progress
            + rew_goal_orientation
            + rew_time_penalty
            + rew_goal_bonus
            + rew_odom_prediction
            + rew_cmd_smoothness
            + rew_cmd_magnitude
            + rew_upright
            + rew_termination
        )

        self.prev_goal_distance.copy_(goal_distance.detach())
        return self._sanitize_tensor(total_reward, "total_reward", clamp_abs=100.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        base_too_low = self.robot.data.root_pos_w[:, 2] < self.cfg.min_base_height
        tipped = self.robot.data.projected_gravity_b[:, 2] > self.cfg.max_projected_gravity_z
        goal_distance = torch.linalg.norm(self.goal_pos_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1)
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self.robot.data.root_quat_w)).abs()
        self.goal_reached = (goal_distance < self.cfg.goal_reached_distance) & (yaw_error < self.cfg.goal_reached_yaw)

        terminated = base_too_low | tipped | self.goal_reached
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.robot.reset(env_ids)
        move_up = None
        move_down = None
        terrain_generator = self.cfg.terrain.terrain_generator
        if terrain_generator is not None and terrain_generator.curriculum:
            reached_goal = self.goal_reached[env_ids_tensor]
            move_up = self.reset_time_outs[env_ids_tensor] | reached_goal
            move_down = self.reset_terminated[env_ids_tensor] & ~move_up

        super()._reset_idx(env_ids_tensor)

        if move_up is not None and move_down is not None:
            self._terrain.update_env_origins(env_ids_tensor, move_up, move_down)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids_tensor].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids_tensor]

        self.actions[env_ids_tensor] = 0.0
        self.prev_actions[env_ids_tensor] = 0.0
        self.predicted_odom[env_ids_tensor] = 0.0
        self.predicted_pos_history[env_ids_tensor] = 0.0
        self.planner_cmd[env_ids_tensor] = 0.0
        self.prev_planner_cmd[env_ids_tensor] = 0.0
        self.low_level_actions[env_ids_tensor] = 0.0
        self.prev_low_level_actions[env_ids_tensor] = 0.0
        self.goal_reached[env_ids_tensor] = False

        self._update_goals(env_ids_tensor)
        self.prev_goal_distance[env_ids_tensor] = torch.linalg.norm(
            self.goal_pos_w[env_ids_tensor, :2] - default_root_state[:, :2], dim=-1
        )

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_tensor)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

    def _update_predicted_position_history(self, predicted_xyz: torch.Tensor) -> None:
        self.predicted_pos_history = torch.roll(self.predicted_pos_history, shifts=-1, dims=1)
        self.predicted_pos_history[:, -1, :] = predicted_xyz

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str, clamp_abs: float | None = None) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            self._finite_warn_counter += 1
            if self._finite_warn_counter <= 5 or self._finite_warn_counter % 500 == 0:
                print(f"[WARN] Non-finite values detected in {name}. Applying nan_to_num safeguard.")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            tensor = torch.clamp(tensor, min=-clamp_abs, max=clamp_abs)
        return tensor

    def _fit_feature_dim(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(tensor.shape[-1])
        if current_dim == target_dim:
            return tensor
        if current_dim > target_dim:
            return tensor[:, :target_dim]
        padding = torch.zeros(tensor.shape[0], target_dim - current_dim, device=tensor.device, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=-1)

    def _fallback_locomotion_action(self) -> torch.Tensor:
        action = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for joint_idx in range(self._num_joints):
            action[:, joint_idx] = self.planner_cmd[:, joint_idx % 3]
        return torch.tanh(action)

    def _query_locomotion_policy(self, locomotion_obs: torch.Tensor, height_data_loc: torch.Tensor) -> torch.Tensor:
        if self.locomotion_policy is None:
            return self._fallback_locomotion_action()

        with torch.inference_mode():
            actions = self.locomotion_policy(locomotion_obs, [height_data_loc])

        if isinstance(actions, (tuple, list)):
            actions = actions[0]
        
        actions = actions.to(self.device)

        return torch.tanh(actions)

    def _process_heightmap(self, height_map):
        sampled_indices = torch.multinomial(self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True)
        height_map_actor = height_map.clone()
        height_map_actor[:, sampled_indices] = 0.0
        return height_map_actor
    
    def _build_locomotion_observations(self, cmd_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        base_ang_vel = self.robot.data.root_ang_vel_b
        projected_gravity = self.robot.data.projected_gravity_b
        joint_pos_rel = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_vel = self.robot.data.joint_vel
        
        x_cells = max(1, int((float(self.cfg.x_range[1]) - float(self.cfg.x_range[0])) / float(self.cfg.res)))
        y_cells = max(1, int((float(self.cfg.y_range[1]) - float(self.cfg.y_range[0])) / float(self.cfg.res)))
        height_data = self._compute_height_data_from_cloud(randomize=self.cfg.randomize)
        height_data = height_data.view(self.num_envs, x_cells, y_cells).flip(dims=[1]).unsqueeze(1)
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # cell_size_m = float(self.cfg.res)
        # inv_cell_size = 1.0 / cell_size_m
        # x_min, x_max = float(self.cfg.x_range[0]), float(self.cfg.x_range[1])
        # y_min, y_max = float(self.cfg.y_range[0]), float(self.cfg.y_range[1])
        # print(height_data_student.reshape(int((x_max - x_min)*inv_cell_size),int((y_max - y_min)*inv_cell_size)))
        
        # print(height_data.reshape(self.num_envs, 15, 10).flip(1,2))            
            
        
        mock_cmd = torch.tensor([0.75, 0.0, 0.0], device=self.device, dtype=base_ang_vel.dtype).repeat(
            self.num_envs, 1
        )

        proprio_loc = torch.cat( 
            [
                base_ang_vel
                + (2.0 * torch.rand_like(base_ang_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(projected_gravity) - 1.0) * float(0.05) * self.cfg.randomize,
                mock_cmd,
                joint_pos_rel
                + (2.0 * torch.rand_like(joint_pos_rel) - 1.0) * float(0.01) * self.cfg.randomize,
                joint_vel + (2.0 * torch.rand_like(joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                self.low_level_actions,
            ],
            dim=-1,
        )
        return self._sanitize_tensor(proprio_loc, "proprio_loc", clamp_abs=50.0), self._sanitize_tensor(height_data, "height_data_loc", clamp_abs=10.0)

    def _apply_offset(self, height_map):
        if not hasattr(self, "_offsets"):
            return height_map
        offset_shape = (self._offsets.shape[0],) + (1,) * (height_map.ndim - 1)
        return height_map + self._offsets.view(offset_shape)
    
    def _apply_yaw_rotation(self, points: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(self._rots).unsqueeze(-1)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        x_coord = points[..., 0]
        y_coord = points[..., 1]
        z_coord = points[..., 2]
        rotated_x = x_coord * cos_angles + z_coord * sin_angles
        rotated_z = -x_coord * sin_angles + z_coord * cos_angles
        return torch.stack((rotated_x, y_coord, rotated_z), dim=-1)
    
    def _zero_heightmap_cells(self, height_map):
        self.same_zeros_count += 1
        if self.same_zeros_count == self.reset_zeros_freq:
            self.reset_zeros_freq = int(torch.randint(1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device).item())            
            self.same_zeros_count = 0   
            self.sampled_indices = torch.multinomial(self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True)
        height_map_actor = height_map.clone()
        height_map_actor[:, self.sampled_indices] = 0.0
        return height_map_actor
    
    def _compute_height_data_from_cloud(self, randomize: bool = False):
        """Compute flattened heightmap in lidar frame using cfg x/y bounds and cell size."""
        data = self._height_scanner.data
        ray_hits_w = data.ray_hits_w
        lidar_pos_w = data.pos_w
        lidar_quat_w = data.quat_w
        num_envs, num_rays, _ = ray_hits_w.shape
        rays_rel_w = - ray_hits_w + lidar_pos_w.unsqueeze(1)
        rays_lidar = quat_apply(
            quat_conjugate(lidar_quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
            rays_rel_w.reshape(-1, 3),
        ).reshape(num_envs, num_rays, 3)
        # rays_lidar = rays_rel_w
        if randomize and hasattr(self, "_rots"):
            rays_lidar = self._apply_yaw_rotation(rays_lidar)

        cell_size_m = float(self.cfg.res)
        inv_cell_size = 1.0 / cell_size_m
        x_min, x_max = float(self.cfg.x_range[0]), float(self.cfg.x_range[1])
        y_min, y_max = float(self.cfg.y_range[0]), float(self.cfg.y_range[1])
        x_cells = max(1, int((x_max - x_min) / cell_size_m))
        y_cells = max(1, int((y_max - y_min) / cell_size_m))
        num_cells = x_cells * y_cells

        rays_flat = rays_lidar.reshape(-1, 3)
        env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(num_envs, num_rays).reshape(-1)

        valid = torch.isfinite(rays_flat).all(dim=1)
        if not torch.any(valid):
            return torch.zeros((num_envs, num_cells), device=self.device)

        rays_valid = rays_flat[valid]
        env_ids = env_ids[valid]

        x_idx = torch.floor((rays_valid[:, 0] - x_min) * inv_cell_size).long()
        y_idx = torch.floor((rays_valid[:, 1] - y_min) * inv_cell_size).long()
        in_bounds = (x_idx >= 0) & (x_idx < x_cells) & (y_idx >= 0) & (y_idx < y_cells)
        if not torch.any(in_bounds):
            return torch.zeros((num_envs, num_cells), device=self.device)

        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        env_ids = env_ids[in_bounds]
        z_vals = rays_valid[in_bounds, 2]

        # Flatten env and cell indexing into one reduce op.
        flat_idx = env_ids * num_cells + x_idx * y_cells + y_idx
        height_map = torch.full((num_envs * num_cells,), -torch.inf, device=self.device)
        height_map.scatter_reduce_(0, flat_idx, z_vals, reduce="amax", include_self=True)
        height_map = torch.where(torch.isfinite(height_map), -height_map, torch.zeros_like(height_map))
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        
        # print(height_map + self.cfg.desired_base_height)
        height_map = height_map.reshape(num_envs, num_cells) - self.cfg.desired_base_height
        if randomize:
            height_map = self._apply_offset(height_map)
            height_map += (2.0 * torch.rand_like(height_map) - 1.0) * float(0.01)
            height_map = self._zero_heightmap_cells(height_map)            
            
        # Keep ordering consistent with lidar_debug flow.
        return height_map
    
    def _compute_lidar_height_map(self, locomotion) -> torch.Tensor:
        data = self._height_scanner.data
        ray_hits_w = data.ray_hits_w
        lidar_pos_w = data.pos_w
        lidar_quat_w = data.quat_w

        num_envs, num_rays, _ = ray_hits_w.shape
        rays_rel_w = ray_hits_w - lidar_pos_w.unsqueeze(1)
        rays_lidar = quat_apply(
            quat_conjugate(lidar_quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
            rays_rel_w.reshape(-1, 3),
        ).reshape(num_envs, num_rays, 3)
        
        if locomotion:
            cell_size = float(self.cfg.res)
            inv_cell_size = 1.0 / cell_size
            x_min, x_max = float(self.cfg.x_range[0]), float(self.cfg.x_range[1])
            y_min, y_max = float(self.cfg.y_range[0]), float(self.cfg.y_range[1])
            x_cells = self._lidar_x_cells_loc
            y_cells = self._lidar_y_cells_loc
            num_cells = x_cells * y_cells

        else:
            cell_size = float(self.cfg.lidar_cell_size)
            inv_cell_size = 1.0 / cell_size
            x_min, x_max = float(self.cfg.x_range[0]), float(self.cfg.x_range[1])
            y_min, y_max = float(self.cfg.y_range[0]), float(self.cfg.y_range[1])
            x_cells = self._lidar_x_cells
            y_cells = self._lidar_y_cells
            num_cells = x_cells * y_cells

        rays_flat = rays_lidar.reshape(-1, 3)
        env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1).expand(num_envs, num_rays).reshape(-1)

        valid = torch.isfinite(rays_flat).all(dim=1)
        if not torch.any(valid):
            return torch.zeros((num_envs, num_cells), device=self.device)

        rays_valid = rays_flat[valid]
        env_ids = env_ids[valid]

        x_idx = torch.floor((rays_valid[:, 0] - x_min) * inv_cell_size).long()
        y_idx = torch.floor((rays_valid[:, 1] - y_min) * inv_cell_size).long()
        in_bounds = (x_idx >= 0) & (x_idx < x_cells) & (y_idx >= 0) & (y_idx < y_cells)
        if not torch.any(in_bounds):
            return torch.zeros((num_envs, num_cells), device=self.device)

        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        env_ids = env_ids[in_bounds]
        z_vals = rays_valid[in_bounds, 2]

        flat_idx = env_ids * num_cells + x_idx * y_cells + y_idx
        height_map = torch.full((num_envs * num_cells,), -torch.inf, device=self.device)
        height_map.scatter_reduce_(0, flat_idx, z_vals, reduce="amax", include_self=True)
        height_map = torch.where(torch.isfinite(height_map), -height_map, torch.zeros_like(height_map))
        height_map = height_map.reshape(num_envs, num_cells)

        return self._sanitize_tensor(height_map, "lidar_obs", clamp_abs=10.0)

    def _get_true_odom(self) -> torch.Tensor:
        root_lin_vel = getattr(self.robot.data, "root_lin_vel_w", self.robot.data.root_lin_vel_b)
        root_ang_vel = getattr(self.robot.data, "root_ang_vel_w", self.robot.data.root_ang_vel_b)
        return torch.cat(
            [
                self.robot.data.root_pos_w,
                root_lin_vel,
                root_ang_vel[:, [0, 2, 1]],
            ],
            dim=-1,
        )

    def _update_goals(self, env_ids: torch.Tensor) -> None:
        terrain_generator = self.cfg.terrain.terrain_generator
        if terrain_generator is None or "maze" not in terrain_generator.sub_terrains:
            self.goal_pos_w[env_ids] = self._terrain.env_origins[env_ids]
            self.goal_pos_w[env_ids, 0] += 2.0
            self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
            return

        maze_cfg = terrain_generator.sub_terrains["maze"]

        cell_size = max(1e-4, float(maze_cfg.cell_size))
        width_scale = max(1.0, float(maze_cfg.maze_width_scale))
        max_cols = max(1, int(round(width_scale)))
        max_rows = max(2, int(max(maze_cfg.maze_height_range[0], maze_cfg.maze_height_range[1])))

        diff_low = float(terrain_generator.difficulty_range[0])
        diff_high = float(terrain_generator.difficulty_range[1])
        num_terrain_rows = max(1, int(terrain_generator.num_rows))

        if hasattr(self._terrain, "terrain_levels"):
            levels = self._terrain.terrain_levels[env_ids].to(dtype=torch.float32)
        else:
            levels = torch.zeros(len(env_ids), device=self.device, dtype=torch.float32)

        if num_terrain_rows > 1:
            level_frac = levels / float(num_terrain_rows - 1)
        else:
            level_frac = torch.zeros_like(levels)
        level_frac = torch.clamp(level_frac, 0.0, 1.0)

        difficulty = diff_low + level_frac * (diff_high - diff_low)
        maze_cols = torch.clamp((difficulty * width_scale).to(torch.long), min=1, max=max_cols)

        terrain_size_x = float(terrain_generator.size[0])
        terrain_size_y = float(terrain_generator.size[1])

        maze_cols_f = maze_cols.to(dtype=torch.float32)
        maze_rows_f = torch.full_like(maze_cols_f, float(max_rows))

        offset_x = 0.5 * (terrain_size_x - maze_cols_f * cell_size)
        offset_y = 0.5 * (terrain_size_y - maze_rows_f * cell_size)

        goal_local_x = offset_x + (maze_cols_f - 0.5) * cell_size
        goal_local_y = offset_y + (maze_rows_f - 0.5) * cell_size

        self.goal_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + goal_local_x
        self.goal_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + goal_local_y
        self.goal_pos_w[env_ids, 2] = self._terrain.env_origins[env_ids, 2]
        self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
        qw = quat_wxyz[:, 0]
        qx = quat_wxyz[:, 1]
        qy = quat_wxyz[:, 2]
        qz = quat_wxyz[:, 3]
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return torch.atan2(siny, cosy)