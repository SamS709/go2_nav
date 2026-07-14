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
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, sample_uniform
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporterCfg ,TerrainImporter


from .go2_nav_env_cfg import Go2NavEnvCfg
from go2_nav.maze_terrain_cfg import MeshMazeTerrainCfg, MAZE_REGISTRY
from .utils import env_ids_to_terrain_coords

class Go2NavEnv(DirectRLEnv):
    cfg: Go2NavEnvCfg

    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.tick_count_env: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.period_hist_mid_term = max(1, int(round(1 / self.cfg.freq_pos_mid_term)))
        self.period_hist_long_term = max(1, int(round(1 / self.cfg.freq_pos_long_term)))
        self.all_env_ids: torch.Tensor = torch.arange(self.num_envs, device=self.device)
        self._base_id_sensor, _base_name = self._contact_sensor.find_bodies("base")
        self._feet_ids_sensor, _feet_name = self._contact_sensor.find_bodies(".*_foot")
        self._thigh_ids_sensor, _thigh_name = self._contact_sensor.find_bodies(".*_thigh")
        self._hip_ids_sensor, _hip_name = self._contact_sensor.find_bodies(".*_hip")
        self._calf_ids_sensor, _calf_name = self._contact_sensor.find_bodies(".*_calf")
        
        self._base_id, _ = self._robot.find_bodies("base")
        self._feet_ids, _ = self._robot.find_bodies(".*_foot")
        self._thigh_ids, _ = self._robot.find_bodies(".*_thigh")
        self._hip_ids, _ = self._robot.find_bodies(".*_hip")
        self._calf_ids, _ = self._robot.find_bodies(".*_calf")
        
        self._undesired_contact_body_ids_sensor = self._thigh_ids_sensor + self._hip_ids_sensor + self._base_id_sensor
        self._body_contact_info_teacher_sensor = self._base_id_sensor + self._thigh_ids_sensor + self._calf_ids_sensor
        self._finite_warn_counter = 0

        # init output tensors
        # [x, y, z, vx, vy, vz, wx, wy, wz, vxd, vyd, vzd]
        # [^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^][^^^^^^^^^^^^^]
        # [              ODOM             ][   VEL CMDS  ]
        
        MAZE_REGISTRY.set_dims(self.cfg.NUM_ROWS, self.cfg.NUM_COLS, self.cfg.MAX_MAZE_ROWS, self.cfg.MAX_MAZE_COLS,self.device)
        self.maze_registery = MAZE_REGISTRY
        terrain_coords: torch.Tensor = env_ids_to_terrain_coords(self.all_env_ids, self._terrain)
        self.mazes: torch.Tensor = self.maze_registery.get_mazes_terrain_coords(terrain_coords).clone()

        self.pred_odom: torch.Tensor = torch.zeros(self.num_envs, 9, device=self.device)
        self.cmd_vel: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)

        action_dim = int(self.cfg.action_space)
        self._num_joints = int(self._robot.data.default_joint_pos.shape[-1])


        self.predicted_odom: torch.Tensor = torch.zeros(self.num_envs, 9, device=self.device)
        self.pred_pos_hist_mid_term = torch.zeros(
            self.num_envs, int(self.cfg.length_mid_term), 3, device=self.device
        )
        self.pred_pos_hist_long_term = torch.zeros(
            self.num_envs, int(self.cfg.length_long_term), 3, device=self.device
        )

        self.cmd_vel: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_cmd_vel: torch.Tensor = torch.zeros_like(self.cmd_vel)

        self.loc_actions: torch.Tensor = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self.prev_loc_actions: torch.Tensor = torch.zeros_like(self.loc_actions)
        self.low_level_joint_targets = self._robot.data.default_joint_pos.clone()

        self._init_goals_and_starts()
        

        self._finite_warn_counter = 0
        self.locomotion_policy = self._load_locomotion_policy(self.cfg.locomotion_policy_path)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "rew_goal_distance",
                "rew_goal_progress",
                "rew_goal_orientation",
                "pen_goal_penalty",
                "rew_goal_bonus",
                "rew_odom_pred",
                "pen_cmd_rate",
                "pen_cmd_bounds",
                "pen_undesired_contacts",
            ]
        }
    
    def _load_locomotion_policy(self, policy_path: str):
        resolved_path = os.path.expanduser(policy_path)
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(os.getcwd(), resolved_path)

        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"Could not find locomotion policy at '{resolved_path}'. "
                "Set Go2NavEnvCfg.locomotion_policy_path to a valid TorchScript file."
            )
        policy = torch.jit.load(resolved_path, map_location=self.device)
        policy.eval()
        with torch.no_grad():
            policy.hidden_state = torch.zeros(
                1, self.num_envs, 128,
                dtype=policy.hidden_state.dtype,
                device=policy.hidden_state.device,
            )
        return policy
        
    def _init_goals_and_starts(self):
        all_env_ids: torch.Tensor = torch.arange(self.num_envs, device=self.device)
        #goals
        self.goal_pos_w: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_yaw_w: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.goal_quat_w: torch.Tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self.prev_goal_distance: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.num_goal_maze_marker: int = self.maze_registery.num_terrain_types        
        self._update_goals(all_env_ids, center=False)
        # starts
        self.start_pos_w: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.start_yaw_w: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.start_quat_w: torch.Tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self._update_starts(all_env_ids, center=False)
        
        
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self._height_scanner = self.cfg.height_scanner.class_type(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._setup_lidar_values()
        self._create_gaussian_heightmap(self.loc_x_cells, self.loc_y_cells)
        
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self.start_markers = VisualizationMarkers(self.cfg.start_marker_cfg)
        self.env_markers = VisualizationMarkers(self.cfg.env_marker_cfg)
        self.vis_envs = 3
        
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
        
    def _setup_lidar_values(self):
        self.nav_inv_cell_size = 1.0 / float(self.cfg.nav_cell_size)
        self.nav_x_cells = max(
            1, int((float(self.cfg.nav_x_range[1]) - float(self.cfg.nav_x_range[0])) * self.nav_inv_cell_size )
        )
        self.nav_y_cells = max(
            1, int((float(self.cfg.nav_y_range[1]) - float(self.cfg.nav_y_range[0])) * self.nav_inv_cell_size )
        )
        self.nav_num_cells = self.nav_x_cells * self.nav_y_cells
        self._latest_lidar_obs = torch.zeros(self.num_envs, self.nav_num_cells, device=self.device)
        
        self.loc_inv_cell_size = 1.0 / float(self.cfg.loc_cell_size)
        self.loc_x_cells = max(
            1,
            int(
                (float(self.cfg.loc_x_range[1]) - float(self.cfg.loc_x_range[0]))
                * self.loc_inv_cell_size
            ),
        )
        self.loc_y_cells = max(
            1,
            int(
                (float(self.cfg.loc_y_range[1]) - float(self.cfg.loc_y_range[0]))
                * self.loc_inv_cell_size
            ),
        )
        self.loc_num_cells = self.loc_x_cells * self.loc_y_cells
        



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
        # [x, y, z, vx, vy, vz, wx, wy, wz, vxd, vyd, vzd]

        self.pred_odom.copy_(actions[:, :9])
        self.cmd_vel.copy_(actions[:, 9:12])
        self._update_predicted_position_history(self.pred_odom[:, :3])

        # cmd_limits = torch.tensor(self.cfg.locomotion_cmd_limits, device=self.device, dtype=self.actions.dtype)
        self.prev_cmd_vel.copy_(self.cmd_vel)
        
        proprio_obs_loc, height_data_loc = self._build_locomotion_observations()

        self.prev_loc_actions.copy_(self.loc_actions)
        # print(height_data_loc[0])
        self.loc_actions.copy_(self._query_locomotion_policy(proprio_obs_loc, height_data_loc))
        self.low_level_joint_targets = (
            self._robot.data.default_joint_pos + self.cfg.locomotion_action_scale * self.loc_actions
        )
        self.tick_count_env += 1
        

    def _update_predicted_position_history(self, predicted_xyz: torch.Tensor) -> None:
        
        """
        updating predicted_pos_buffer, filled by the back: the most recent are at the ens of the buffer.
        """
        
        
        # mid_term history
        should_update_mid = (self.tick_count_env % self.period_hist_mid_term == 0) 
        rolled = torch.roll(self.pred_pos_hist_mid_term, shifts=-1, dims=1)
        rolled[:, -1, :] = predicted_xyz
        mask = should_update_mid[:, None, None]
        self.pred_pos_hist_mid_term = torch.where(mask, rolled, self.pred_pos_hist_mid_term)
        
        # long_term history
        should_update_long = (self.tick_count_env % self.period_hist_long_term == 0) 
        rolled = torch.roll(self.pred_pos_hist_long_term, shifts=-1, dims=1)
        rolled[:, -1, :] = predicted_xyz
        mask = should_update_long[:, None, None]
        self.pred_pos_hist_long_term = torch.where(mask, rolled, self.pred_pos_hist_long_term)
        

    def _query_locomotion_policy(self, locomotion_obs: torch.Tensor, height_data_loc: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            actions = self.locomotion_policy(locomotion_obs, [height_data_loc])

        if isinstance(actions, (tuple, list)):
            actions = actions[0]
        
        actions = actions.to(self.device)

        return actions

    def _build_locomotion_observations(self) -> tuple[torch.Tensor, torch.Tensor]: 
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel
        
        # height_data = self._compute_height_data_from_cloud(randomize=self.cfg.randomize)
        height_data = self._compute_height_data_from_cloud(randomize=True)
        height_data = height_data.view(self.num_envs, self.loc_x_cells, self.loc_y_cells).flip(dims=[1]).unsqueeze(1)
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # cell_size_m = float(self.cfg.loc_cell_size)
        # inv_cell_size = 1.0 / cell_size_m
        # x_min, x_max = float(self.cfg.loc_x_range[0]), float(self.cfg.loc_x_range[1])
        # y_min, y_max = float(self.cfg.loc_y_range[0]), float(self.cfg.loc_y_range[1])
        # print(height_data_student.reshape(int((x_max - x_min)*inv_cell_size),int((y_max - y_min)*inv_cell_size)))
        
        # print(height_data.reshape(self.num_envs, 15, 10).flip(1,2))            
            
        # mock_cmd = torch.tensor([0.75, 0.0, 0.0], device=self.device, dtype=base_ang_vel.dtype).repeat(
        #     self.num_envs, 1
        # )

        proprio_loc = torch.cat( 
            [
                base_ang_vel
                + (2.0 * torch.rand_like(base_ang_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(projected_gravity) - 1.0) * float(0.05) * self.cfg.randomize,
                # mock_cmd,
                self.cmd_vel,
                joint_pos_rel
                + (2.0 * torch.rand_like(joint_pos_rel) - 1.0) * float(0.01) * self.cfg.randomize,
                joint_vel + (2.0 * torch.rand_like(joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                self.loc_actions,
            ],
            dim=-1,
        )
        return self._sanitize_tensor(proprio_loc, "proprio_loc", clamp_abs=50.0), self._sanitize_tensor(height_data, "height_data_loc", clamp_abs=10.0)



    def _apply_action(self) -> None:
        # self._robot.set_joint_position_target(self._robot.data.default_joint_pos)
        self._robot.set_joint_position_target(self.low_level_joint_targets)

    def _get_observations(self) -> dict:
        height_data_teacher = self._compute_height_data_from_cloud(randomize=False, locomotion=False)
        height_data_student = self._compute_height_data_from_cloud(randomize=self.cfg.randomize, locomotion=False)
        height_data_teacher = height_data_teacher.view(self.num_envs, self.nav_x_cells, self.nav_y_cells).flip(dims=[1]).unsqueeze(1)
        height_data_student = height_data_student.view(self.num_envs, self.nav_x_cells, self.nav_y_cells).flip(dims=[1]).unsqueeze(1)
        goal_xy_s = self._get_goal_pos_s()
        goal_yaw_s = self._get_goal_yaw_s()
        pred_pos_hist_mid_term = self.pred_pos_hist_mid_term.reshape(self.num_envs, -1)
        pred_pos_hist_long_term = self.pred_pos_hist_long_term.reshape(self.num_envs, -1)
        student_proprio = torch.cat([goal_xy_s, goal_yaw_s, pred_pos_hist_mid_term, pred_pos_hist_long_term], dim=-1)

        true_odom = self._get_true_odom_r()
        goal_delta = self.goal_pos_w - self._robot.data.root_pos_w
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w))
        goal_heading = torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)

        
        terrain_type = self.mazes[:,:,:,0].long()
        maze_encoded = torch.nn.functional.one_hot(terrain_type, self.maze_registery.num_terrain_types)
        maze_remaining = self.mazes[:,:,:,1:] 
        maze_encoded_flat = maze_encoded.reshape(self.num_envs, -1)
        maze_remaining_flat = maze_remaining.reshape(self.num_envs, -1)
        maze_features = torch.cat([maze_encoded_flat, maze_remaining_flat], dim=-1)

        teacher_proprio = torch.cat(
            [student_proprio.clone(), true_odom, goal_delta, goal_heading, self.cmd_vel, maze_features],
            dim=-1,
        )

        student_proprio = self._sanitize_tensor(student_proprio, "student_proprio", clamp_abs=100.0)
        teacher_proprio = self._sanitize_tensor(teacher_proprio, "teacher_proprio", clamp_abs=100.0)
        student_height_scan = self._sanitize_tensor(height_data_teacher, "student_height_scan", clamp_abs=10.0)
        teacher_height_scan = self._sanitize_tensor(height_data_student, "teacher_height_scan", clamp_abs=10.0)

        return {
            "student_proprio": student_proprio,
            "student_height_scan": student_height_scan,
            "teacher_proprio": teacher_proprio,
            "teacher_height_scan": teacher_height_scan,
        }
        
                
    
    def log_infos(self):
        mazes_array = self.maze_registery.as_list()
        self.env_markers.visualize(
            translations=self._robot.data.root_pos_w[self.vis_envs].unsqueeze(0),
            orientations=self._robot.data.root_quat_w[self.vis_envs].unsqueeze(0),
        )
        terrain_coords = env_ids_to_terrain_coords(self.all_env_ids, self._terrain)
        # print(terrain_coords[n])
        # print(self.mazes[n, :, :, 0])
        # num_rows=self.maze_registery.get_num_rows(terrain_coords)
        # print(num_rows[n])
        # maze_encoded = torch.nn.functional.one_hot(self.mazes[:,:,:,0].long(), self.maze_registery.num_terrain_types + 1)
        # print(self.pred_pos_hist_mid_term[self.vis_envs][-1])
        # print(self.pred_pos_hist_long_term[self.vis_envs][-1])

    def _get_rewards(self) -> torch.Tensor:

        self.log_infos()
        
        root_pos_w = self._robot.data.root_pos_w
        goal_delta_xy = self.goal_pos_w[:, :2] - root_pos_w[:, :2]
        goal_distance = torch.linalg.norm(goal_delta_xy, dim=-1)
        

        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w))

        rew_goal_distance = torch.exp(
            -goal_distance / max(1e-4, float(self.cfg.goal_distance_sigma))
        )
        self.prev_goal_distance.copy_(goal_distance.detach()) # store the prev goal dist
        
        #penalize orientation only when next to the goal
        rew_goal_orientation = torch.where(goal_distance < 0.3, torch.exp(
            -torch.square(yaw_error) / max(1e-4, float(self.cfg.goal_orientation_sigma))
        ), 0.0)

        rew_goal_progress = self.prev_goal_distance - goal_distance
        
        pen_time_penalty = torch.ones_like(goal_distance)

        true_odom = self._get_true_odom_r()
        odom_error = torch.mean(torch.square(self.predicted_odom - true_odom), dim=-1)
        rew_odom_prediction = torch.exp(
            -odom_error / max(1e-4, float(self.cfg.odom_prediction_scale))
        )

        pen_cmd_rate = torch.sum(torch.square(self.cmd_vel - self.prev_cmd_vel), dim=-1)
        
        cmd_bounds = torch.sum(torch.where(self.cmd_vel >= 1.0, torch.abs(self.cmd_vel), 0.0), dim=-1)
        pen_cmd_bounds = cmd_bounds + torch.sum(torch.where(self.cmd_vel <= 0.2, torch.exp(-10.0*torch.abs(self.cmd_vel)), 0.0), dim=-1)

        rew_goal_bonus = self.goal_reached.float()

        # undesired contacts
        is_contact = (
            torch.max(torch.norm(self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids_sensor], dim=-1), dim=1)[0] > 1.0
        )
        pen_undesired_contacts = torch.sum(is_contact, dim=1)
        
        rewards = {
            "rew_goal_distance": self.cfg.rew_scale_goal_distance * rew_goal_distance * self.step_dt,
            "rew_goal_orientation": self.cfg.rew_scale_goal_orientation * rew_goal_orientation * self.step_dt,
            "rew_goal_progress": self.cfg.rew_scale_goal_progress * rew_goal_progress * self.step_dt,
            "pen_goal_penalty": self.cfg.rew_scale_time_penalty * pen_time_penalty * self.step_dt,
            "rew_odom_pred": self.cfg.rew_scale_odom_prediction * rew_odom_prediction * self.step_dt,
            "pen_cmd_rate": self.cfg.rew_scale_cmd_rate * pen_cmd_rate * self.step_dt,
            "pen_cmd_bounds": self.cfg.rew_scale_cmd_bounds * pen_cmd_bounds * self.step_dt,
            "rew_goal_bonus": self.cfg.rew_scale_goal_bonus * rew_goal_bonus * self.step_dt,
            "pen_undesired_contacts": self.cfg.rew_scale_undesired_contacts * pen_undesired_contacts * self.step_dt,

        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = self._sanitize_tensor(reward, "reward", clamp_abs=100.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        goal_distance = torch.linalg.norm(self.goal_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=-1)
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w)).abs()
        self.goal_reached = (goal_distance < self.cfg.goal_reached_distance) & (yaw_error < self.cfg.goal_reached_yaw)
        
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id_sensor], dim=-1), dim=1)[0] > 1.0, dim=1)
        
        terminated = time_out | died_base | self.goal_reached 
        
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._robot.reset(env_ids)
        move_up = None
        move_down = None
        with torch.no_grad():
            self.locomotion_policy.hidden_state[:, env_ids_tensor, :] = 0.0
        terrain_generator = self.cfg.terrain.terrain_generator
        if terrain_generator is not None and terrain_generator.curriculum:
            reached_goal = self.goal_reached[env_ids_tensor]
            move_up = self.reset_time_outs[env_ids_tensor] | reached_goal
            move_down = self.reset_terminated[env_ids_tensor] & ~move_up

        super()._reset_idx(env_ids_tensor)
        
        self.tick_count_env[env_ids_tensor] = 0

        if move_up is not None and move_down is not None:
            self._terrain.update_env_origins(env_ids_tensor, move_up, move_down)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        default_root_state = self._robot.data.default_root_state[env_ids_tensor].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids_tensor]

        self.predicted_odom[env_ids_tensor] = 0.0
        self.pred_pos_hist_mid_term[env_ids_tensor] = 0.0
        self.pred_pos_hist_long_term[env_ids_tensor] = 0.0
        self.cmd_vel[env_ids_tensor] = 0.0
        self.prev_cmd_vel[env_ids_tensor] = 0.0
        self.loc_actions[env_ids_tensor] = 0.0
        self.prev_loc_actions[env_ids_tensor] = 0.0
        self.goal_reached[env_ids_tensor] = False

        self._update_goals(env_ids_tensor, False)
        self._update_starts(env_ids_tensor, False)
        
        root_state = torch.cat([self.start_pos_w[env_ids], self.start_quat_w[env_ids]], dim=-1)
        
        self.prev_goal_distance[env_ids_tensor] = torch.linalg.norm(
            self.goal_pos_w[env_ids_tensor, :2] - default_root_state[:, :2], dim=-1
        )

        self._robot.write_root_pose_to_sim(root_state, env_ids_tensor)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids_tensor] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids_tensor]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids_tensor]).item()
        self.extras["log"].update(extras)


    
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
    
    def _compute_height_data_from_cloud(self, randomize: bool = False, locomotion: bool = True):
        """Compute flattened heightmap in lidar frame using cfg x/y bounds and cell size."""
        # choose the desired output whether this is the loc or nav lidar
        
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

        if locomotion:
            x_cells = self.loc_x_cells
            y_cells = self.loc_y_cells
            inv_cell_size = self.loc_inv_cell_size            
            x_min = float(self.cfg.loc_x_range[0])
            y_min = float(self.cfg.loc_y_range[0])
            num_cells = self.loc_num_cells            
        else:
            x_cells = self.nav_x_cells
            y_cells = self.nav_y_cells
            inv_cell_size = self.nav_inv_cell_size
            x_min = float(self.cfg.nav_x_range[0])
            y_min = float(self.cfg.nav_y_range[0])
            num_cells = self.nav_num_cells

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
    
    def _get_true_odom_r(self) -> torch.Tensor:
        root_lin_vel = self._robot.data.root_lin_vel_b
        root_ang_vel = self._robot.data.root_ang_vel_b
        xyz_r = self._robot.data.root_pos_w - self.start_pos_w
        return torch.cat(
            [
                xyz_r,
                root_lin_vel,
                root_ang_vel
            ],
            dim=-1,
        )

    def _get_goal_pos_s(self) -> torch.Tensor:
        """Goal position in start frame (2D, x forward, y left)."""
        delta_w = self.goal_pos_w - self.start_pos_w  # (num_envs, 3)

        cos_yaw = torch.cos(-self.start_yaw_w)  # (num_envs,)
        sin_yaw = torch.sin(-self.start_yaw_w)

        x_r = cos_yaw * delta_w[:, 0] - sin_yaw * delta_w[:, 1]
        y_r = sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]

        return torch.stack([x_r, y_r], dim=-1)  # (num_envs, 2) — drop z, navigation is 2D

    def _get_goal_yaw_s(self) -> torch.Tensor:
        """Goal heading relative to start heading, wrapped to [-pi, pi]."""
        delta_yaw = self.goal_yaw_w - self.start_yaw_w  # (num_envs,)
        # Wrap to [-pi, pi]
        return torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw)).unsqueeze(-1)  # (num_envs,)

    def _update_goals(self, env_ids: torch.Tensor, center: bool = False) -> None:
        """Update goal positions for the given envs.

        If ``center`` is True, the goal is placed at the center of each env's
        maze. Otherwise, the goal is sampled uniformly at random over the actual
        n-bounds extent of that env's maze.

        Per-env maze row count follows the curriculum: terrain levels 0 and 1 both
        produce a 1-row maze, and level k (k >= 2) produces a k-row maze, clamped to
        `maze_max_rows`. Column count is always the fixed `maze_max_cols`.
        """
        terrain_generator = self.cfg.terrain.terrain_generator
        maze_cfg = terrain_generator.sub_terrains["maze"]
        cell_size = float(maze_cfg.cell_size)

        max_cols = max(1, int(round(maze_cfg.maze_max_cols)))
        

        # Per-env curriculum level (0-indexed). Levels 0 & 1 -> 1 row; level k (k>=2) -> k rows.
        terrain_coords = env_ids_to_terrain_coords(env_ids, self._terrain)
        maze_rows = self.maze_registery.get_num_rows(terrain_coords)
        maze_cols = torch.full_like(maze_rows, max_cols)
        
    
        maze_rows_f = maze_rows.to(dtype=torch.float32)
        maze_cols_f = maze_cols.to(dtype=torch.float32)

        # Per-env maze extent (rows -> world x, cols -> world y), centered on the tile.
        x_extent = maze_rows_f * cell_size
        y_extent = maze_cols_f * cell_size
        
        if center:
            # Maze is always centered on its tile, regardless of row/col count, so the
            # goal is simply the tile center -- no sampling needed.
            goal_local_x = torch.zeros_like(x_extent)
            goal_local_y = torch.zeros_like(y_extent)
        else:
            # Sample a uniformly random in-bounds point within the maze's footprint.
            x_min = -0.5 * x_extent
            x_min = torch.where(maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size)
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size)

            u_x = torch.rand(len(env_ids), device=self.device)
            u_y = torch.rand(len(env_ids), device=self.device)
            goal_local_x = x_min + u_x * (x_max - x_min)
            goal_local_y = y_min + u_y * (y_max - y_min)
            goal_local_x = torch.where(goal_local_x >= 0.0, (goal_local_x // cell_size) * cell_size, (goal_local_x // cell_size + 1) * cell_size)
            goal_local_y = torch.where(goal_local_y >= 0.0, (goal_local_y // cell_size) * cell_size, (goal_local_y // cell_size + 1) * cell_size)
            

        env_origins = self._terrain.env_origins[env_ids]
        
        self.goal_pos_w[env_ids, 0] = env_origins[:, 0] + goal_local_x
        self.goal_pos_w[env_ids, 1] = env_origins[:, 1] + goal_local_y
        self.goal_pos_w[env_ids, 2] = env_origins[:, 2]

        self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self.goal_quat_w[:, 0] = torch.cos(self.goal_yaw_w / 2)  # w
        self.goal_quat_w[:, 1] = 0.0                             # x
        self.goal_quat_w[:, 2] = 0.0                             # y
        self.goal_quat_w[:, 3] = torch.sin(self.goal_yaw_w / 2)  # z

        if type(self.vis_envs) == int:
            self.goal_markers.visualize(
                translations=self.goal_pos_w[self.vis_envs].unsqueeze(0),
                orientations=self.goal_quat_w[self.vis_envs].unsqueeze(0),
            )
        else:
            self.goal_markers.visualize(
                translations=self.goal_pos_w[self.vis_envs],
                orientations=self.goal_quat_w[self.vis_envs],
            )
        
    def _update_starts(self, env_ids: torch.Tensor, center: bool = False) -> None:
        """Update start positions for the given envs.

        If ``center`` is True, the start is placed at the center of each env's
        maze. Otherwise, the start is sampled uniformly at random over the actual
        n-bounds extent of that env's maze.

        Per-env maze row count follows the curriculum: terrain levels 0 and 1 both
        produce a 1-row maze, and level k (k >= 2) produces a k-row maze, clamped to
        `maze_max_rows`. Column count is always the fixed `maze_max_cols`.
        """
        terrain_generator = self.cfg.terrain.terrain_generator
        maze_cfg = terrain_generator.sub_terrains["maze"]
        cell_size = float(maze_cfg.cell_size)
        
        
        max_cols = max(1, int(round(maze_cfg.maze_max_cols)))
        # Per-env curriculum level (0-indexed). Levels 0 & 1 -> 1 row; level k (k>=2) -> k rows.
        terrain_coords = env_ids_to_terrain_coords(env_ids, self._terrain)
        maze_rows = self.maze_registery.get_num_rows(terrain_coords)
        maze_cols = torch.full_like(maze_rows, max_cols)

        maze_rows_f = maze_rows.to(dtype=torch.float32)
        maze_cols_f = maze_cols.to(dtype=torch.float32)

        # Per-env maze extent (rows -> world x, cols -> world y), centered on the tile.
        x_extent = maze_rows_f * cell_size
        y_extent = maze_cols_f * cell_size
        
        if center:
            # Maze is always centered on its tile, regardless of row/col count, so the
            # start is simply the tile center -- no sampling needed.
            start_local_x = torch.zeros_like(x_extent)
            start_local_y = torch.zeros_like(y_extent)
        else:
            # Sample a uniformly random in-bounds point within the maze's footprint.
            # Bounds are [tile_center - extent/2, tile_center + extent/2] per axis.
            # Sample a uniformly random in-bounds point within the maze's footprint.
            x_min = -0.5 * x_extent
            x_min = torch.where(maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size)
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size)

            u_x = torch.rand(len(env_ids), device=self.device)
            u_y = torch.rand(len(env_ids), device=self.device)
            start_local_x = x_min + u_x * (x_max - x_min)
            start_local_y = y_min + u_y * (y_max - y_min)
            start_local_x = torch.where(start_local_x >= 0.0, (start_local_x // cell_size) * cell_size, (start_local_x // cell_size + 1) * cell_size)
            start_local_y = torch.where(start_local_y >= 0.0, (start_local_y // cell_size) * cell_size, (start_local_y // cell_size + 1) * cell_size)
            

        env_origins = self._terrain.env_origins[env_ids]
        
        
        self.start_pos_w[env_ids, 0] = env_origins[:, 0] + start_local_x
        self.start_pos_w[env_ids, 1] = env_origins[:, 1] + start_local_y
        self.start_pos_w[env_ids, 2] = env_origins[:, 2] + self._robot.data.default_root_state[env_ids, 2].clone()

        self.start_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self.start_quat_w[:, 0] = torch.cos(self.start_yaw_w / 2)  # w
        self.start_quat_w[:, 1] = 0.0                             # x
        self.start_quat_w[:, 2] = 0.0                             # y
        self.start_quat_w[:, 3] = torch.sin(self.start_yaw_w / 2)  # z

        if type(self.vis_envs) == int:
            self.start_markers.visualize(
                translations=self.start_pos_w[self.vis_envs].unsqueeze(0),
                orientations=self.start_quat_w[self.vis_envs].unsqueeze(0),
            )
        else:
            self.start_markers.visualize(
                translations=self.start_pos_w[self.vis_envs],
                orientations=self.start_quat_w[self.vis_envs],
            )
        
    def _sanitize_tensor(self, tensor: torch.Tensor, name: str, clamp_abs: float | None = None) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            self._finite_warn_counter += 1
            if self._finite_warn_counter <= 5 or self._finite_warn_counter % 500 == 0:
                print(f"[WARN] Non-finite values detected in {name}. Applying nan_to_num safeguard.")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            tensor = torch.clamp(tensor, min=-clamp_abs, max=clamp_abs)
        return tensor

    @staticmethod
    def _fit_feature_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = int(tensor.shape[-1])
        if current_dim == target_dim:
            return tensor
        if current_dim > target_dim:
            return tensor[:, :target_dim]
        padding = torch.zeros(tensor.shape[0], target_dim - current_dim, device=tensor.device, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=-1)
        
    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        """Normalize angle to [-pi, pi]"""
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