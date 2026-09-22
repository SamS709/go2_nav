# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from tensordict import TensorDict
import os

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, sample_uniform
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.terrains import TerrainImporterCfg, TerrainImporter


from .go2_nav_env_cfg import Go2NavEnvCfg
from go2_nav.maze_terrain_cfg import MeshMazeTerrainCfg, MAZE_REGISTRY
from .utils import env_ids_to_terrain_coords
from .networks.cnn_rnn_model import CNNRNNNewModel
from rsl_rl.models.rnn_model import RNNModel


class Go2NavEnv(DirectRLEnv):
    cfg: Go2NavEnvCfg

    ######################################################
    # INIT FUNCTIONS
    ######################################################

    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._map_plot_counter = {}  # env_idx -> next file index
        self.tick_count_env: torch.Tensor = torch.zeros(
            self.num_envs, device=self.device
        )
        self.period_hist_short_term = max(
            1, int(round(1 / self.cfg.freq_pos_short_term))
        )
        self.period_hist_long_term = max(1, int(round(1 / self.cfg.freq_pos_long_term)))
        self.all_env_ids: torch.Tensor = torch.arange(self.num_envs, device=self.device)

        self._base_id_sensor, _base_name = self._contact_sensor.find_bodies("base")
        self._feet_ids_sensor, _feet_name = self._contact_sensor.find_bodies(".*_foot")
        self._thigh_ids_sensor, _thigh_name = self._contact_sensor.find_bodies(
            ".*_thigh"
        )
        self._hip_ids_sensor, _hip_name = self._contact_sensor.find_bodies(".*_hip")
        self._calf_ids_sensor, _calf_name = self._contact_sensor.find_bodies(".*_calf")

        self._base_id, _ = self._robot.find_bodies("base")
        self._feet_ids, _ = self._robot.find_bodies(".*_foot")
        self._thigh_ids, _ = self._robot.find_bodies(".*_thigh")
        self._hip_ids, _ = self._robot.find_bodies(".*_hip")
        self._calf_ids, _ = self._robot.find_bodies(".*_calf")

        self._undesired_contact_body_ids_sensor = (
            self._thigh_ids_sensor + self._hip_ids_sensor + self._base_id_sensor
        )
        self._body_contact_info_teacher_sensor = (
            self._base_id_sensor + self._thigh_ids_sensor + self._calf_ids_sensor
        )

        # init output tensors
        # [x, y, z, vx, vy, vz, wx, wy, wz, vxd, vyd, vzd]
        # [^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^][^^^^^^^^^^^^^]
        # [              ODOM             ][   VEL CMDS  ]

        MAZE_REGISTRY.set_dims(
            self.cfg.NUM_ROWS,
            self.cfg.NUM_COLS,
            self.cfg.MAX_MAZE_ROWS,
            self.cfg.MAX_MAZE_COLS,
            self.device,
        )
        self.maze_registery = MAZE_REGISTRY

        self.odom_obs_proprio_hist = torch.zeros(
            self.num_envs, int(self.cfg.length_odom_hist), 45, device=self.device
        )
        # The odometry model predicts 12 values: pose, linear velocity, and angular velocity.
        self.pred_odom: torch.Tensor = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self.cmd_vel: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)

        action_dim = int(self.cfg.action_space)
        self._num_joints = int(self._robot.data.default_joint_pos.shape[-1])

        self.pred_pos_hist_short_term = torch.zeros(
            self.num_envs, int(self.cfg.length_short_term), 3, device=self.device
        )
        self.pred_pos_hist_long_term = torch.zeros(
            self.num_envs, int(self.cfg.length_long_term), 3, device=self.device
        )

        self.cmd_vel: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_cmd_vel: torch.Tensor = torch.zeros_like(self.cmd_vel)

        self.loc_actions: torch.Tensor = torch.zeros(
            self.num_envs, self._num_joints, device=self.device
        )
        self.prev_loc_actions: torch.Tensor = torch.zeros_like(self.loc_actions)
        self.low_level_joint_targets = self._robot.data.default_joint_pos.clone()

        if self.cfg.delay == True:
            self._proprio_delay_buffer: DelayBuffer = DelayBuffer(
                history_length=self.cfg.delay_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            if self.cfg.delay_length > 0:
                self._proprio_delay_buffer.set_time_lag(
                    torch.randint(
                        low=0,
                        high=self.cfg.history_length,
                        size=(self.num_envs,),
                        device=self.device,
                    )
                )
            self._grid_delay_buffer: DelayBuffer = DelayBuffer(
                history_length=self.cfg.delay_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            if self.cfg.delay_length > 0:
                self._grid_delay_buffer.set_time_lag(
                    torch.randint(
                        low=0,
                        high=self.cfg.delay_length,
                        size=(self.num_envs,),
                        device=self.device,
                    )
                )

        self._init_goals_and_starts()

        self._finite_warn_counter = 0

        self.locomotion_policy = self._load_locomotion_policy()
        self.odom_model = self._load_odom_model()
        self.odom_optimizer = torch.optim.Adam(self.odom_model.parameters(), lr=1e-3)
        self.map_model = self._load_map_model()
        self.map_optimizer = torch.optim.Adam(self.map_model.parameters(), lr=1e-3)

        self.terrain_success_rate = torch.zeros(
            self.cfg.terrain.terrain_generator.num_rows,
            self.cfg.terrain.terrain_generator.num_cols,
            device=self.device,
        )
        self.terrain_episode_count = torch.zeros_like(self.terrain_success_rate)
        self._stall_counter = torch.zeros(self.num_envs, device=self.device)
        # Logging

        # Creating the global map that gets filled from its center according to the current pred odom and the heightmap
        map_w_cells = int(self.cfg.map_width / self.cfg.map_cell_size)
        map_h_cells = int(self.cfg.map_height / self.cfg.map_cell_size)
        self.pred_map = torch.zeros(self.num_envs, map_w_cells, map_h_cells, device=self.device)
        self.true_map = torch.zeros_like(self.pred_map)
        self.pred_map_confidence = torch.zeros_like(self.pred_map)
        self.map_revealed = torch.zeros(self.num_envs, map_w_cells, map_h_cells, dtype=torch.bool, device=self.device)
        local_x = (torch.arange(map_w_cells, device=self.device) - map_w_cells / 2 + 0.5) * self.cfg.map_cell_size
        local_y = (torch.arange(map_h_cells, device=self.device) - map_h_cells / 2 + 0.5) * self.cfg.map_cell_size
        lx, ly = torch.meshgrid(local_x, local_y, indexing="ij")
        self._map_local_pts = torch.stack([lx, ly], dim=-1).reshape(-1, 2)  # (cells, 2), env-independent
        print(f"Global map initialized with shape: {self.pred_map.shape}")
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
                "pen_stagnation",
            ]
        }

    def _init_goals_and_starts(self):
        # goals
        self.goal_pos_w: torch.Tensor = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.goal_yaw_w: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.goal_quat_w: torch.Tensor = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.prev_goal_distance: torch.Tensor = torch.zeros(
            self.num_envs, device=self.device
        )
        self.goal_reached: torch.Tensor = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.num_goal_maze_marker: int = self.maze_registery.num_terrain_types
        self._update_goals(self.all_env_ids, center=False)
        # starts
        self.start_pos_w: torch.Tensor = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self.start_yaw_w: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.start_quat_w: torch.Tensor = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._update_starts(self.all_env_ids, center=False)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self._height_scanner = self.cfg.height_scanner.class_type(
            self.cfg.height_scanner
        )
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._rots_yaw = torch.empty(self.num_envs, device=self.device)
        self._rots_roll = torch.empty(self.num_envs, device=self.device)
        self._offsets = torch.empty(self.num_envs, device=self.device)
        self._rots_yaw.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
        self._rots_roll.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
        self._offsets.uniform_(-self.cfg.max_offset, self.cfg.max_offset)
        self._setup_lidar_values()
        self._create_gaussian_heightmap(self.loc_x_cells, self.loc_y_cells)

        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self.start_markers = VisualizationMarkers(self.cfg.start_marker_cfg)
        self.env_markers = VisualizationMarkers(self.cfg.env_marker_cfg)
        self.vis_envs = 6

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=10000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._heightfield_ready = False

    def _setup_lidar_values(self):
        self.nav_inv_cell_size = 1.0 / float(self.cfg.nav_cell_size)
        self.nav_x_cells = max(
            1,
            int(
                (float(self.cfg.nav_x_range[1]) - float(self.cfg.nav_x_range[0]))
                * self.nav_inv_cell_size
            ),
        )
        self.nav_y_cells = max(
            1,
            int(
                (float(self.cfg.nav_y_range[1]) - float(self.cfg.nav_y_range[0]))
                * self.nav_inv_cell_size
            ),
        )
        self.nav_num_cells = self.nav_x_cells * self.nav_y_cells
        self._latest_lidar_obs = torch.zeros(
            self.num_envs, self.nav_num_cells, device=self.device
        )

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
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Center of the grid
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0

        # Compute 2D Gaussian
        gaussian_dist = torch.exp(
            ((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.cfg.sigma**2)
        )

        # Normalize to create probability distribution
        gaussian_prob = gaussian_dist / gaussian_dist.sum()
        self.gaussian_prob_heightmap = gaussian_prob.flatten()
        self.sampled_indices = torch.multinomial(
            self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True
        )
        self.same_zeros_count = 0
        self.reset_zeros_freq = int(
            torch.randint(
                1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device
            ).item()
        )
        
        
    

    #################################################################
    # MAP MODEL UTILS
    #################################################################

    def _load_map_model(self):
        obs = TensorDict(
            {
                "height_data": torch.randn(
                    self.num_envs, 1, self.nav_x_cells, self.nav_y_cells
                ),  # 2D: B, C, H, W
                "odom_data": torch.randn(self.num_envs, 12),  # 1D: B, D
            }
        )

        # 2. Define groups exactly as they appear in obs
        obs_groups = {
            "policy": [
                "height_data",
                "odom_data",
            ]  # Order matters for indexing if needed
        }

        # 3. Define CNN config for the 'height_map' group
        cnn_cfg = {
            "height_data": {
                "output_channels": [16, 32],
                "kernel_size": [3, 3],
                "stride": [2, 2],
                "activation": "relu",
                "max_pool": False,
                "global_pool": "avg",
            }
        }
        
        # 4. Initialize the model
        model = CNNRNNNewModel(
            obs=obs,
            obs_groups=obs_groups,
            activation="identity",
            obs_normalization=True,
            obs_set="policy",
            output_dim=2 * int(self.cfg.map_width / self.cfg.map_cell_size) * int(self.cfg.map_height / self.cfg.map_cell_size),
            hidden_dims=(128, 64),
            cnn_cfg=cnn_cfg,
            rnn_hidden_dim=64,
            rnn_num_layers=1,
            rnn_type="gru",
        )
        model = model.to(self.device)
        model.eval()
        model.reset()
        # 5. Test: forward pass
        with torch.no_grad():
            obs_dict = TensorDict(
                {"height_data": obs["height_data"].to(self.device), "odom_data": obs["odom_data"].to(self.device)},
                batch_size=obs["odom_data"].shape[:1],
                device=self.device,
            )
            model(obs_dict)
        print("Map model")
        print(model)  
        return model
    

    def _build_map_observations(self) -> tuple[torch.Tensor, torch.Tensor]:

        height_data = self._compute_height_data_from_cloud(
            randomize=self.cfg.randomize, locomotion=False
        )
        height_data = (
            height_data.view(self.num_envs, self.nav_x_cells, self.nav_y_cells)
            .flip(dims=[1])
            .unsqueeze(1)
        )
        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # print(height_data[self.vis_envs])

        # cell_size_m = float(self.cfg.loc_cell_size)
        # inv_cell_size = 1.0 / cell_size_m
        # x_min, x_max = float(self.cfg.loc_x_range[0]), float(self.cfg.loc_x_range[1])
        # y_min, y_max = float(self.cfg.loc_y_range[0]), float(self.cfg.loc_y_range[1])
        # print(height_data_student.reshape(int((x_max - x_min)*inv_cell_size),int((y_max - y_min)*inv_cell_size)))

        # print(height_data.reshape(self.num_envs, 15, 10).flip(1,2))

        proprio_loc = self.pred_odom.clone()
        return self._sanitize_tensor(
            proprio_loc, "obs_map_odom", clamp_abs=1000.0
        ), self._sanitize_tensor(height_data, "obs_map_height", clamp_abs=100.0)
        
    def _update_true_map(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        gx, gy = self._global_heightfield.shape
        res = self._global_heightfield_res
        x_min = self._global_heightfield_x_min
        y_min = self._global_heightfield_y_min
        map_w_cells, map_h_cells = self.true_map.shape[1], self.true_map.shape[2]

        local_x = (torch.arange(map_w_cells, device=self.device) - map_w_cells / 2 + 0.5) * self.cfg.map_cell_size
        local_y = (torch.arange(map_h_cells, device=self.device) - map_h_cells / 2 + 0.5) * self.cfg.map_cell_size
        lx, ly = torch.meshgrid(local_x, local_y, indexing="ij")
        local_pts = torch.stack([lx, ly], dim=-1).reshape(-1, 2)

        yaw = self.start_yaw_w[env_ids]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        wx = cos_y[:, None] * local_pts[None, :, 0] - sin_y[:, None] * local_pts[None, :, 1]
        wy = sin_y[:, None] * local_pts[None, :, 0] + cos_y[:, None] * local_pts[None, :, 1]
        world_x = wx + self.start_pos_w[env_ids, 0:1]
        world_y = wy + self.start_pos_w[env_ids, 1:2]

        # normalize using the SAME bounds the heightfield was actually built with
        gx_norm = (world_x - x_min) / (gx * res) * 2 - 1
        gy_norm = (world_y - y_min) / (gy * res) * 2 - 1
        sample_grid = torch.stack([gy_norm, gx_norm], dim=-1).reshape(n, map_w_cells, map_h_cells, 2)
        field = self._global_heightfield.unsqueeze(0).unsqueeze(0).expand(n, 1, gx, gy)

        sampled = F.grid_sample(field, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        self.true_map[env_ids] = sampled.squeeze(1)

    def _get_true_map_s(self) -> torch.Tensor:
        return self.true_map.clone()

    def _train_map_step(self, map_obs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        with torch.inference_mode(False), torch.enable_grad():
            odom_data, height_data = map_obs
            odom_data, height_data = odom_data.clone(), height_data.clone()

            map_w_cells, map_h_cells = self.true_map.shape[1], self.true_map.shape[2]
            map_cells = map_w_cells * map_h_cells

            target = self._get_true_map_s()               # (N, W, H)
            mask = self.map_revealed.to(target.dtype)      # (N, W, H) -- doubles as the confidence label

            obs_dict = TensorDict(
                {"height_data": height_data, "odom_data": odom_data},
                batch_size=odom_data.shape[:1],
            )

            self.map_model.train()
            raw_out: torch.Tensor = self.map_model(obs_dict)
            if isinstance(raw_out, (tuple, list)):
                raw_out = raw_out[0]
            raw_out = raw_out.to(self.device)

            height_pred = raw_out[:, :map_cells].reshape(self.num_envs, map_w_cells, map_h_cells)
            conf_logits = raw_out[:, map_cells:].reshape(self.num_envs, map_w_cells, map_h_cells)

            # 1. Height regression -- ONLY on revealed cells, same as before.
            per_elem_height_loss = F.smooth_l1_loss(height_pred, target, reduction="none")
            height_loss = (per_elem_height_loss * mask).sum() / mask.sum().clamp(min=1.0)

            # 2. Confidence classification -- supervised EVERYWHERE, since the
            # label (revealed or not) is known for every cell regardless.
            confidence_loss = F.binary_cross_entropy_with_logits(conf_logits, mask)

            loss = height_loss + 0.5 * confidence_loss

            self.map_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.map_model.parameters(), max_norm=1.0)
            self.map_optimizer.step()
            self.map_model.detach_hidden_state()
            self.map_model.eval()

            self.extras.setdefault("log", {})
            self.extras["log"].update({
                "map/loss_total": loss.detach(),
                "map/loss_height": height_loss.detach(),
                "map/loss_confidence": confidence_loss.detach(),
            })

            pred_confidence = torch.sigmoid(conf_logits).detach()
            return height_pred.detach(), pred_confidence
    
    def _precompute_global_heightfield(self):
        if self._heightfield_ready:
            return
        from isaaclab.utils.warp import raycast_mesh

        prim_path = self._height_scanner.cfg.mesh_prim_paths[0]
        warp_mesh = self._height_scanner.meshes[prim_path]

        gen_cfg = self.cfg.terrain.terrain_generator
        tile_w, tile_l = gen_cfg.size

        # Derive world bounds from env_origins rather than assuming the tile
        # grid starts at (0,0) -- terrain generators typically center the
        # whole grid on the world origin, so a (0,0) anchor only covers one
        # quadrant of the real terrain.
        origins = self._terrain.env_origins  # (num_envs, 3)
        margin = 1.5 * max(tile_w, tile_l)
        x_min = origins[:, 0].min().item() - margin
        x_max = origins[:, 0].max().item() + margin
        y_min = origins[:, 1].min().item() - margin
        y_max = origins[:, 1].max().item() + margin

        res = self.cfg.map_cell_size
        gx = max(2, int((x_max - x_min) / res))
        gy = max(2, int((y_max - y_min) / res))

        xs = torch.linspace(x_min, x_max, gx, device=self.device)
        ys = torch.linspace(y_min, y_max, gy, device=self.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        ray_starts = torch.stack(
            [grid_x.reshape(-1), grid_y.reshape(-1), torch.full_like(grid_x.reshape(-1), 100.0)], dim=-1
        ).unsqueeze(0)
        ray_dirs = torch.zeros_like(ray_starts)
        ray_dirs[..., 2] = -1.0

        hits, *_ = raycast_mesh(ray_starts, ray_dirs, mesh=warp_mesh)
        heights = hits[0, :, 2]
        heights = torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))

        self._global_heightfield = heights.reshape(gx, gy)
        self._global_heightfield_res = res
        self._global_heightfield_x_min = x_min
        self._global_heightfield_y_min = y_min
        self._heightfield_ready = True    
    
    def _reveal_around(self, env_ids: torch.Tensor, robot_xy_w: torch.Tensor, radius: float = None) -> None:
        radius = self.cfg.map_lidar_radius  
        n = env_ids.numel()
        map_w_cells, map_h_cells = self.true_map.shape[1], self.true_map.shape[2]

        yaw = self.start_yaw_w[env_ids]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        wx = cos_y[:, None] * self._map_local_pts[None, :, 0] - sin_y[:, None] * self._map_local_pts[None, :, 1]
        wy = sin_y[:, None] * self._map_local_pts[None, :, 0] + cos_y[:, None] * self._map_local_pts[None, :, 1]
        cell_world_x = wx + self.start_pos_w[env_ids, 0:1]
        cell_world_y = wy + self.start_pos_w[env_ids, 1:2]

        dist = torch.sqrt((cell_world_x - robot_xy_w[:, 0:1]) ** 2 + (cell_world_y - robot_xy_w[:, 1:2]) ** 2)
        newly_revealed = (dist <= radius).reshape(n, map_w_cells, map_h_cells)

        self.map_revealed[env_ids] |= newly_revealed

    #################################################################
    # ODOM MODEL UTILS
    #################################################################

    def _load_odom_model(self):
        obs = TensorDict(
            {
                "proprio": torch.randn(
                    self.num_envs,
                    45 * self.cfg.length_odom_hist + self.pred_odom.shape[-1],
                )  # 1D: B, D
            }
        )

        # 2. Define groups exactly as they appear in obs
        obs_groups = {"policy": ["proprio"]}  # Order matters for indexing if needed
        # 4. Initialize the model
        model = RNNModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="policy",
            output_dim=12,  # [x,y,z,r,p,y,vx,vy,vz,wx,wy,wz]
            hidden_dims=(128, 128),
            activation="identity",
            obs_normalization=True,
            rnn_num_layers=1,
            rnn_hidden_dim=64,
            rnn_type="gru",
        )
        # Move before the warm-up forward pass so the recurrent hidden state
        # is initialized on the same device as the model parameters.
        model = model.to(self.device)
        model.eval()
        model.reset()
        with torch.no_grad():
            output = model(
                TensorDict(
                    {"proprio": obs["proprio"].to(self.device)},
                    batch_size=[self.num_envs],
                )
            )
        print("Odom model")
        print(model) 
        return model

    def _update_odom_obs_history(self, odom_obs: torch.Tensor) -> None:
        """
        updating odom_obs_hist, filled by the back: the most recent are at the end of the buffer.
        """
        self.odom_obs_proprio_hist = torch.roll(self.odom_obs_proprio_hist, -1, 1)
        self.odom_obs_proprio_hist[:, -1, :] = odom_obs

    def _update_predicted_position_history(self, predicted_xyz: torch.Tensor) -> None:
        """
        updating predicted_pos_buffer, filled by the back: the most recent are at the end of the buffer.
        """

        # short_term history
        should_update_mid = self.tick_count_env % self.period_hist_short_term == 0
        rolled = torch.roll(self.pred_pos_hist_short_term, shifts=-1, dims=1)
        rolled[:, -1, :] = predicted_xyz
        mask = should_update_mid[:, None, None]
        self.pred_pos_hist_short_term = torch.where(
            mask, rolled, self.pred_pos_hist_short_term
        )

        # long_term history
        should_update_long = self.tick_count_env % self.period_hist_long_term == 0
        rolled = torch.roll(self.pred_pos_hist_long_term, shifts=-1, dims=1)
        rolled[:, -1, :] = predicted_xyz
        mask = should_update_long[:, None, None]
        self.pred_pos_hist_long_term = torch.where(
            mask, rolled, self.pred_pos_hist_long_term
        )

    def _get_true_odom_s(self) -> torch.Tensor:
        """Returns the odometry of the robot in its spawn frame."""
        root_pos_w = self._robot.data.root_pos_w.clone()
        root_quat_w = self._robot.data.root_quat_w.clone()
        root_lin_vel = self._robot.data.root_lin_vel_b.clone()
        root_ang_vel = self._robot.data.root_ang_vel_b.clone()

        delta_w = root_pos_w - self.start_pos_w  # (num_envs, 3)

        cos_yaw = torch.cos(-self.start_yaw_w)
        sin_yaw = torch.sin(-self.start_yaw_w)

        x_s = cos_yaw * delta_w[:, 0] - sin_yaw * delta_w[:, 1]
        y_s = sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        z_s = delta_w[:, 2]

        xyz_s = torch.stack([x_s, y_s, z_s], dim=-1)

        # Current orientation in world frame
        w, x, y, z = root_quat_w.unbind(dim=-1)

        roll_w = torch.atan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )

        pitch_w = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))

        yaw_w = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        # Orientation relative to spawn frame
        roll_s = roll_w
        pitch_s = pitch_w
        yaw_s = torch.atan2(
            torch.sin(yaw_w - self.start_yaw_w),
            torch.cos(yaw_w - self.start_yaw_w),
        )

        rpy_s = torch.stack([roll_s, pitch_s, yaw_s], dim=-1)

        return torch.cat([xyz_s, rpy_s, root_lin_vel, root_ang_vel], dim=-1).clone()

    def _train_odom_step(self, odom_obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(False), torch.enable_grad():

            odom_obs = odom_obs.clone()
            target = self._get_true_odom_s()

            obs_dict = TensorDict({"proprio": odom_obs}, batch_size=odom_obs.shape[:1])

            self.odom_model.train()
            pred_odom: torch.Tensor = self.odom_model(obs_dict)
            if isinstance(pred_odom, (tuple, list)):
                pred_odom = pred_odom[0]
            pred_odom = pred_odom.to(self.device)

            pos_loss = F.mse_loss(pred_odom[:, 0:3], target[:, 0:3])
            ori_loss = F.mse_loss(pred_odom[:, 3:6], target[:, 3:6])
            lin_vel_loss = F.mse_loss(pred_odom[:, 6:9], target[:, 6:9])
            ang_vel_loss = F.mse_loss(pred_odom[:, 9:12], target[:, 9:12])
            loss = pos_loss + ori_loss + 0.5 * lin_vel_loss + 0.5 * ang_vel_loss

            self.odom_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.odom_model.parameters(), max_norm=1.0)
            self.odom_optimizer.step()
            self.odom_model.detach_hidden_state()
            self.odom_model.eval()

            self.extras.setdefault("log", {})
            self.extras["log"].update(
                {
                    "odom/loss_total": loss.detach(),
                    "odom/loss_pos": pos_loss.detach(),
                    "odom/loss_ori": ori_loss.detach(),
                    "odom/loss_lin_vel": lin_vel_loss.detach(),
                    "odom/loss_ang_vel": ang_vel_loss.detach(),
                }
            )
            return pred_odom.detach()

    def _build_odom_observations(self) -> torch.Tensor:
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        mock_cmd = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=base_ang_vel.dtype
        ).repeat(self.num_envs, 1)
        # odom obs shape = 3 + 3 + 3 + 12 + 12 + 12 = 45
        odom_obs = torch.cat(
            [
                base_ang_vel
                + (2.0 * torch.rand_like(base_ang_vel) - 1.0)
                * float(0.1)
                * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(projected_gravity) - 1.0)
                * float(0.05)
                * self.cfg.randomize,
                mock_cmd,
                # self.cmd_vel,
                joint_pos_rel
                + (2.0 * torch.rand_like(joint_pos_rel) - 1.0)
                * float(0.01)
                * self.cfg.randomize,
                joint_vel
                + (2.0 * torch.rand_like(joint_vel) - 1.0)
                * float(0.1)
                * self.cfg.randomize,
                self.loc_actions,
            ],
            dim=-1,
        )

        self._update_odom_obs_history(odom_obs)
        odom_hist_flat = self.odom_obs_proprio_hist.reshape(self.num_envs, -1)
        return torch.cat([odom_hist_flat, self.pred_odom], dim=-1)

    #################################################################
    # LOCOMOTION MODEL UTILS
    #################################################################

    def _load_locomotion_policy(self):
        resolved_path = os.path.expanduser(self.cfg.locomotion_policy_path)
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
                1,
                self.num_envs,
                128,
                dtype=policy.hidden_state.dtype,
                device=policy.hidden_state.device,
            )
        return policy

    def _query_locomotion_policy(
        self, locomotion_obs: torch.Tensor, height_data_loc: torch.Tensor
    ) -> torch.Tensor:
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
        height_data = self._compute_height_data_from_cloud(
            randomize=self.cfg.randomize, locomotion=True
        )
        height_data = (
            height_data.view(self.num_envs, self.loc_x_cells, self.loc_y_cells)
            .flip(dims=[1])
            .unsqueeze(1)
        )

        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # print(height_data[self.vis_envs])

        # cell_size_m = float(self.cfg.loc_cell_size)
        # inv_cell_size = 1.0 / cell_size_m
        # x_min, x_max = float(self.cfg.loc_x_range[0]), float(self.cfg.loc_x_range[1])
        # y_min, y_max = float(self.cfg.loc_y_range[0]), float(self.cfg.loc_y_range[1])
        # print(height_data_student.reshape(int((x_max - x_min)*inv_cell_size),int((y_max - y_min)*inv_cell_size)))

        # print(height_data.reshape(self.num_envs, 15, 10).flip(1,2))

        mock_cmd = torch.tensor(
            [0.0, 0.0, 0.0], device=self.device, dtype=base_ang_vel.dtype
        ).repeat(self.num_envs, 1)

        proprio_loc = torch.cat(
            [
                base_ang_vel
                + (2.0 * torch.rand_like(base_ang_vel) - 1.0)
                * float(0.1)
                * self.cfg.randomize,
                projected_gravity
                + (2.0 * torch.rand_like(projected_gravity) - 1.0)
                * float(0.05)
                * self.cfg.randomize,
                mock_cmd,
                # self.cmd_vel,
                joint_pos_rel
                + (2.0 * torch.rand_like(joint_pos_rel) - 1.0)
                * float(0.01)
                * self.cfg.randomize,
                joint_vel
                + (2.0 * torch.rand_like(joint_vel) - 1.0)
                * float(0.1)
                * self.cfg.randomize,
                self.loc_actions,
            ],
            dim=-1,
        )
        return self._sanitize_tensor(
            proprio_loc, "proprio_loc", clamp_abs=50.0
        ), self._sanitize_tensor(height_data, "height_data_loc", clamp_abs=10.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions = [vxd, vyd, wzd]
        self.cmd_vel.copy_(actions[:, :3])
        self.prev_cmd_vel.copy_(self.cmd_vel)

        # 1. Locomotion policy
        # cmd_limits = torch.tensor(self.cfg.locomotion_cmd_limits, device=self.device, dtype=self.actions.dtype)

        proprio_obs_loc, height_data_loc = self._build_locomotion_observations()

        self.prev_loc_actions.copy_(self.loc_actions)
        self.loc_actions.copy_(
            self._query_locomotion_policy(proprio_obs_loc, height_data_loc)
        )
        self.low_level_joint_targets = (
            self._robot.data.default_joint_pos
            + self.cfg.locomotion_action_scale * self.loc_actions
        )

        # 2. Odom model
        odom_obs = self._build_odom_observations()
        self.pred_odom.copy_(self._train_odom_step(odom_obs))
        self._update_predicted_position_history(self.pred_odom[:, :3])
        
        # 2. Map model
        self._reveal_around(
            torch.arange(self.num_envs, device=self.device),
            self._robot.data.root_pos_w[:, :2],
        )
        map_obs = self._build_map_observations()
        pred_h, pred_c = self._train_map_step(map_obs)
        self.pred_map.copy_(pred_h)
        self.pred_map_confidence.copy_(pred_c)
        self._update_predicted_position_history(self.pred_odom[:, :3])
        self.tick_count_env += 1



    def _apply_action(self) -> None:
        # self._robot.set_joint_position_target(self._robot.data.default_joint_pos)
        self._robot.set_joint_position_target(self.low_level_joint_targets)

    def _get_observations(self) -> dict:
        height_data_teacher = self._compute_height_data_from_cloud(
            randomize=False, locomotion=False
        )
        height_data_student = self._compute_height_data_from_cloud(
            randomize=self.cfg.randomize, locomotion=False
        )
        height_data_teacher = (
            height_data_teacher.view(self.num_envs, self.nav_x_cells, self.nav_y_cells)
            .flip(dims=[1])
            .unsqueeze(1)
        )
        height_data_student = (
            height_data_student.view(self.num_envs, self.nav_x_cells, self.nav_y_cells)
            .flip(dims=[1])
            .unsqueeze(1)
        )

        # torch.set_printoptions(precision=2, linewidth=1000, sci_mode=False)
        # print(height_data_teacher[self.vis_envs][0,20:,25:45])

        goal_xy_s = self._get_goal_pos_s()
        goal_yaw_s = self._get_goal_yaw_s()
        pred_pos_hist_short_term = self.pred_pos_hist_short_term.reshape(
            self.num_envs, -1
        )
        pred_pos_hist_long_term = self.pred_pos_hist_long_term.reshape(
            self.num_envs, -1
        )
        student_proprio = torch.cat(
            [
                goal_xy_s,
                goal_yaw_s,
                pred_pos_hist_short_term,
                pred_pos_hist_long_term,
                self.prev_cmd_vel,
            ],
            dim=-1,
        )

        true_odom = self._get_true_odom_s()
        goal_delta = self.goal_pos_w - self._robot.data.root_pos_w
        yaw_error = self._wrap_to_pi(
            self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w)
        )
        goal_heading = torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)

        terrain_coords: torch.Tensor = env_ids_to_terrain_coords(
            self.all_env_ids, self._terrain
        )
        mazes: torch.Tensor = self.maze_registery.get_mazes_terrain_coords(
            terrain_coords
        ).clone()
        terrain_type = mazes[:, :, :, 0].long()
        maze_encoded = torch.nn.functional.one_hot(
            terrain_type, self.maze_registery.num_terrain_types
        )
        maze_remaining = mazes[:, :, :, 1:]
        maze_encoded_flat = maze_encoded.reshape(self.num_envs, -1)
        maze_remaining_flat = maze_remaining.reshape(self.num_envs, -1)
        maze_features = torch.cat([maze_encoded_flat, maze_remaining_flat], dim=-1)

        teacher_proprio = torch.cat(
            [
                student_proprio.clone(),
                true_odom,
                goal_delta,
                goal_heading,
                maze_features,
            ],
            dim=-1,
        )

        student_proprio = self._sanitize_tensor(
            student_proprio, "student_proprio", clamp_abs=100.0
        )
        teacher_proprio = self._sanitize_tensor(
            teacher_proprio, "teacher_proprio", clamp_abs=100.0
        )
        student_height_scan = self._sanitize_tensor(
            height_data_teacher, "student_height_scan", clamp_abs=10.0
        )
        teacher_height_scan = self._sanitize_tensor(
            height_data_student, "teacher_height_scan", clamp_abs=10.0
        )
        if self.cfg.delay:
            teacher_proprio = self._proprio_delay_buffer.compute(teacher_proprio)
            teacher_height_scan = self._grid_delay_buffer.compute(teacher_height_scan)
            student_proprio = self._proprio_delay_buffer.compute(student_proprio)
            student_height_scan = self._grid_delay_buffer.compute(student_height_scan)

        return {
            "student_proprio": student_proprio,
            "student_height_scan": student_height_scan,
            "teacher_proprio": teacher_proprio,
            "teacher_height_scan": teacher_height_scan,
        }

    def log_infos(self):
        self.env_markers.visualize(
            translations=self._robot.data.root_pos_w[self.vis_envs].unsqueeze(0),
            orientations=self._robot.data.root_quat_w[self.vis_envs].unsqueeze(0),
        )
        terrain_coords: torch.Tensor = env_ids_to_terrain_coords(
            torch.tensor([self.vis_envs], device=self.device), self._terrain
        )
        mazes: torch.Tensor = self.maze_registery.get_mazes_terrain_coords(
            terrain_coords
        ).clone()
        # print(terrain_coords)
        # print(mazes.shape)
        # print(mazes[: , :, :, 0]) # type

        # print(mazes[: , :, :, 3]) # north
        # print(mazes[: , :, :, 1]) # south
        # print(mazes[: , :, :, 2]) # east
        # print(mazes[: , :, :, 4]) # west

        # num_rows=self.maze_registery.get_num_rows(terrain_coords)
        # print(num_rows[n])
        # maze_encoded = torch.nn.functional.one_hot(self.mazes[:,:,:,0].long(), self.maze_registery.num_terrain_types + 1)
        # print(self.pred_pos_hist_short_term[self.vis_envs][-1])
        # print(self.pred_pos_hist_long_term[self.vis_envs][-1])

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.vis:
            self.log_infos()

        root_pos_w = self._robot.data.root_pos_w
        goal_delta_xy = self.goal_pos_w[:, :2] - root_pos_w[:, :2]
        goal_distance = torch.linalg.norm(goal_delta_xy, dim=-1)

        robot_yaw = self._quat_to_yaw(self._robot.data.root_quat_w)
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - robot_yaw)

        # --- goal distance (unchanged) ---
        rew_goal_distance = torch.exp(
            -goal_distance / max(1e-4, float(self.cfg.goal_distance_sigma))
        )

        # --- goal orientation: smooth gate instead of hard cutoff at 0.3 m ---
        # was: torch.where(goal_distance < 0.3, exp(...), 0.0)  <- discontinuous, encourages
        # boundary-hovering. Replaced with a smooth weight that fades in as the robot
        # approaches, controlled by cfg.goal_orientation_gate_sigma (new field).
        orientation_gate = torch.exp(
            -goal_distance / max(1e-4, float(self.cfg.goal_orientation_gate_sigma))
        )
        rew_goal_orientation = orientation_gate * torch.exp(
            -torch.square(yaw_error) / max(1e-4, float(self.cfg.goal_orientation_sigma))
        )

        # --- goal progress: clamp against reset-induced spikes ---
        # If prev_goal_distance wasn't correctly reset to the *new* episode's goal_distance
        # on env reset, a stale value causes a one-step teleport spike here. Clamp as
        # insurance regardless; ALSO verify _reset_idx() sets self.prev_goal_distance
        # from the fresh goal_distance before the first reward is computed.
        raw_progress = self.prev_goal_distance - goal_distance
        rew_goal_progress = torch.clamp(
            raw_progress, -self.cfg.goal_progress_clip, self.cfg.goal_progress_clip
        )
        self.prev_goal_distance.copy_(
            goal_distance.detach()
        )  # store the prev goal dist

        pen_time_penalty = torch.ones_like(goal_distance)

        true_odom = self._get_true_odom_s()
        odom_error = torch.mean(torch.square(self.pred_odom - true_odom), dim=-1)
        rew_odom_prediction = torch.exp(
            -odom_error / max(1e-4, float(self.cfg.odom_prediction_sigma))
        )
        # NOTE: still recommend moving this to a supervised aux loss on the prediction
        # head instead of shaping it through PPO's reward, if pred_odom is produced
        # by a differentiable submodule you control. Left as-is here since that's a
        # training-loop change, not a reward-function change.

        pen_cmd_rate = torch.sum(torch.square(self.cmd_vel - self.prev_cmd_vel), dim=-1)

        cmd_bounds_high = torch.sum(
            torch.where(self.cmd_vel >= 1.0, torch.abs(self.cmd_vel), 0.0), dim=-1
        )
        # Low-end penalty: gated off near the goal so it doesn't fight rew_goal_orientation,
        # which needs small, precise commands right at the goal. Reuses orientation_gate.
        low_cmd_penalty = torch.sum(
            torch.where(
                self.cmd_vel <= 0.2,
                torch.exp(-torch.abs(self.cmd_vel) / self.cfg.cmd_vel_sigma),
                0.0,
            ),
            dim=-1,
        )
        pen_cmd_bounds = cmd_bounds_high + (1.0 - orientation_gate) * low_cmd_penalty

        rew_goal_bonus = self.goal_reached.float()
        # Confirm episodes actually terminate/truncate when goal_reached fires elsewhere
        # in the step loop — otherwise this can pay out indefinitely while lingering at goal.

        # --- undesired contacts (unchanged; graded version noted below if you want it) ---
        contact_forces = torch.norm(
            self._contact_sensor.data.net_forces_w_history[
                :, :, self._undesired_contact_body_ids_sensor
            ],
            dim=-1,
        )
        is_contact = torch.max(contact_forces, dim=1)[0] > 1.0
        pen_undesired_contacts = torch.sum(is_contact, dim=1)

        # --- NEW: stagnation penalty ---
        # Penalizes sustained lack of progress toward the goal (stuck against geometry,
        # spinning in place, oscillating). Requires a persistent per-env counter buffer,
        # e.g. self._stall_counter = torch.zeros(num_envs, device=self.device), created
        # in __init__ and reset to 0 in _reset_idx().
        making_progress = raw_progress > self.cfg.stagnation_progress_eps
        self._stall_counter = torch.where(
            making_progress,
            0.0,
            self._stall_counter + 1,
        )
        pen_stagnation = (self._stall_counter >= self.cfg.stagnation_window).float()

        rewards = {
            "rew_goal_distance": self.cfg.rew_scale_goal_distance
            * rew_goal_distance
            * self.step_dt,
            "rew_goal_orientation": self.cfg.rew_scale_goal_orientation
            * rew_goal_orientation
            * self.step_dt,
            "rew_goal_progress": self.cfg.rew_scale_goal_progress
            * rew_goal_progress
            * self.step_dt,
            "pen_goal_penalty": self.cfg.rew_scale_time_penalty
            * pen_time_penalty
            * self.step_dt,
            "rew_odom_pred": self.cfg.rew_scale_odom_prediction
            * rew_odom_prediction
            * self.step_dt,
            "pen_cmd_rate": self.cfg.rew_scale_cmd_rate * pen_cmd_rate * self.step_dt,
            "pen_cmd_bounds": self.cfg.rew_scale_cmd_bounds
            * pen_cmd_bounds
            * self.step_dt,
            "rew_goal_bonus": self.cfg.rew_scale_goal_bonus
            * rew_goal_bonus
            * self.step_dt,
            "pen_undesired_contacts": self.cfg.rew_scale_undesired_contacts
            * pen_undesired_contacts
            * self.step_dt,
            "pen_stagnation": self.cfg.rew_scale_stagnation
            * pen_stagnation
            * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = self._sanitize_tensor(reward, "reward", clamp_abs=100.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        goal_distance = torch.linalg.norm(
            self.goal_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=-1
        )
        yaw_error = self._wrap_to_pi(
            self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w)
        ).abs()
        self.goal_reached = (goal_distance < self.cfg.goal_reached_distance) & (
            yaw_error < self.cfg.goal_reached_yaw
        )

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_base = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._base_id_sensor], dim=-1),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )

        terminated = died_base | self.goal_reached

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        
        self._precompute_global_heightfield()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._plot_map_s(env_ids_tensor, save_dir="/mnt/D/dev/robotics/nvidia/isaaclab/go2_nav/map_debug")
        self._robot.reset(env_ids)
        move_up = None
        move_down = None
        with torch.no_grad():
            self.locomotion_policy.hidden_state[:, env_ids_tensor, :] = 0.0
        terrain_generator = self.cfg.terrain.terrain_generator
        if terrain_generator is not None and terrain_generator.curriculum:
            reached_goal = self.goal_reached[env_ids_tensor]
            terrain_ids = env_ids_to_terrain_coords(env_ids_tensor, self._terrain)
            died = self.reset_terminated[env_ids_tensor] & ~reached_goal
            rows = terrain_ids[:, 0].long()
            cols = terrain_ids[:, 1].long()

            self.terrain_episode_count[rows, cols] += 1
            self.terrain_success_rate[rows, cols] += (
                reached_goal.float() - self.terrain_success_rate[rows, cols]
            ) / self.terrain_episode_count[rows, cols].clamp(min=1)

            cell_success = self.terrain_success_rate[rows, cols]
            move_up = cell_success > 0.8
            move_down = died & (cell_success < 0.2)

        super()._reset_idx(env_ids_tensor)
        

        odom_reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        odom_reset_mask[env_ids_tensor] = True
        with torch.no_grad():
            self.odom_model.reset(odom_reset_mask)
            
        map_reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        map_reset_mask[env_ids_tensor] = True
        with torch.no_grad():
            self.map_model.reset(map_reset_mask)

        self.tick_count_env[env_ids_tensor] = 0

        if move_up is not None and move_down is not None:
            self._terrain.update_env_origins(env_ids_tensor, move_up, move_down)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        default_root_state = self._robot.data.default_root_state[env_ids_tensor].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids_tensor]

        self.pred_odom[env_ids_tensor] = 0.0
        self.pred_pos_hist_short_term[env_ids_tensor] = 0.0
        self.pred_pos_hist_long_term[env_ids_tensor] = 0.0
        self.cmd_vel[env_ids_tensor] = 0.0
        self.prev_cmd_vel[env_ids_tensor] = 0.0
        self.loc_actions[env_ids_tensor] = 0.0
        self.prev_loc_actions[env_ids_tensor] = 0.0
        self.goal_reached[env_ids_tensor] = False
        self._stall_counter[env_ids_tensor] = 0.0


        self._update_goals(env_ids_tensor, False)
        self._update_starts(env_ids_tensor, False)
        self._update_true_map(env_ids_tensor)
        self.map_revealed[env_ids_tensor] = False
        self._reveal_around(env_ids_tensor, self.start_pos_w[env_ids_tensor, :2])
        if hasattr(self, "_rots"):
            num_resets = env_ids_tensor.numel()
            self._rots_yaw[env_ids_tensor] = torch.empty(
                num_resets, device=self.device
            ).uniform_(-self.cfg.max_rot, self.cfg.max_rot)
            self._rots_roll[env_ids_tensor] = torch.empty(
                num_resets, device=self.device
            ).uniform_(-self.cfg.max_rot, self.cfg.max_rot)
            self._offsets[env_ids_tensor] = torch.empty(
                num_resets, device=self.device
            ).uniform_(-self.cfg.max_offset, self.cfg.max_offset)

        if self.cfg.delay == True:
            self._proprio_delay_buffer.reset(env_ids_tensor.tolist())
            self._proprio_delay_buffer.set_time_lag(
                torch.randint(
                    low=0,
                    high=self.cfg.delay_length,
                    size=(self.num_envs,),
                    device=self.device,
                )
            )
            self._grid_delay_buffer.reset(env_ids_tensor.tolist())
            self._grid_delay_buffer.set_time_lag(
                torch.randint(
                    low=0,
                    high=self.cfg.delay_length,
                    size=(self.num_envs,),
                    device=self.device,
                )
            )

        root_state = torch.cat(
            [self.start_pos_w[env_ids], self.start_quat_w[env_ids]], dim=-1
        )

        self.prev_goal_distance[env_ids_tensor] = torch.linalg.norm(
            self.goal_pos_w[env_ids_tensor, :2] - default_root_state[:, :2], dim=-1
        )

        self._robot.write_root_pose_to_sim(root_state, env_ids_tensor)
        self._robot.write_root_velocity_to_sim(
            default_root_state[:, 7:], env_ids_tensor
        )
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids_tensor] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids_tensor]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids_tensor]
        ).item()
        self.extras["log"].update(extras)

    def _apply_offset(self, height_map):
        if not hasattr(self, "_offsets"):
            return height_map
        offset_shape = (self._offsets.shape[0],) + (1,) * (height_map.ndim - 1)
        return height_map + self._offsets.view(offset_shape)

    def _apply_yaw_rotation(self, points: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(self._rots_yaw).unsqueeze(-1)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        x_coord = points[..., 0]
        y_coord = points[..., 1]
        z_coord = points[..., 2]
        rotated_x = x_coord * cos_angles + z_coord * sin_angles
        rotated_z = -x_coord * sin_angles + z_coord * cos_angles
        return torch.stack((rotated_x, y_coord, rotated_z), dim=-1)

    def _apply_roll_rotation(self, points: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(self._rots).unsqueeze(-1)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        x_coord = points[..., 0]
        y_coord = points[..., 1]
        z_coord = points[..., 2]
        rotated_y = y_coord * cos_angles - z_coord * sin_angles
        rotated_z = y_coord * sin_angles + z_coord * cos_angles
        return torch.stack((x_coord, rotated_y, rotated_z), dim=-1)

    def _zero_heightmap_cells(self, height_map):
        self.same_zeros_count += 1
        if self.same_zeros_count == self.reset_zeros_freq:
            self.reset_zeros_freq = int(
                torch.randint(
                    1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device
                ).item()
            )
            self.same_zeros_count = 0
            self.sampled_indices = torch.multinomial(
                self.gaussian_prob_heightmap, self.cfg.n_zeros, replacement=True
            )
        height_map_actor = height_map.clone()
        height_map_actor[:, self.sampled_indices] = 0.0
        return height_map_actor

    def _compute_height_data_from_cloud(
        self, randomize: bool = False, locomotion: bool = True
    ):
        """Compute flattened heightmap in lidar frame using cfg x/y bounds and cell size."""
        # choose the desired output whether this is the loc or nav lidar

        data = self._height_scanner.data
        ray_hits_w = data.ray_hits_w
        lidar_pos_w = data.pos_w
        lidar_quat_w = data.quat_w
        num_envs, num_rays, _ = ray_hits_w.shape
        rays_rel_w = -ray_hits_w + lidar_pos_w.unsqueeze(1)
        rays_lidar = quat_apply(
            quat_conjugate(lidar_quat_w)
            .unsqueeze(1)
            .expand(num_envs, num_rays, 4)
            .reshape(-1, 4),
            rays_rel_w.reshape(-1, 3),
        ).reshape(num_envs, num_rays, 3)
        # rays_lidar = rays_rel_w
        if randomize and hasattr(self, "_rots"):
            rays_lidar = self._apply_yaw_rotation(rays_lidar)
            rays_lidar = self._apply_roll_rotation(rays_lidar)

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
        env_ids = (
            torch.arange(num_envs, device=self.device)
            .unsqueeze(1)
            .expand(num_envs, num_rays)
            .reshape(-1)
        )

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
        height_map.scatter_reduce_(
            0, flat_idx, z_vals, reduce="amax", include_self=True
        )
        height_map += self.cfg.desired_base_height_loc
        height_map = torch.where(
            torch.isfinite(height_map), -height_map, torch.zeros_like(height_map)
        )

        height_map = height_map.reshape(num_envs, num_cells)
        if randomize:
            height_map = self._apply_offset(height_map)
            height_map += (2.0 * torch.rand_like(height_map) - 1.0) * float(0.01)
            height_map = self._zero_heightmap_cells(height_map)

        # Keep ordering consistent with lidar_debug flow.
        return height_map

    def _get_goal_pos_s(self) -> torch.Tensor:
        """Goal position in start frame (2D, x forward, y left)."""
        delta_w = self.goal_pos_w - self.start_pos_w  # (num_envs, 3)

        cos_yaw = torch.cos(-self.start_yaw_w)  # (num_envs,)
        sin_yaw = torch.sin(-self.start_yaw_w)

        x_r = cos_yaw * delta_w[:, 0] - sin_yaw * delta_w[:, 1]
        y_r = sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]

        return torch.stack(
            [x_r, y_r], dim=-1
        )  # (num_envs, 2) — drop z, navigation is 2D

    def _get_goal_yaw_s(self) -> torch.Tensor:
        """Goal heading relative to start heading, wrapped to [-pi, pi]."""
        delta_yaw = self.goal_yaw_w - self.start_yaw_w  # (num_envs,)
        # Wrap to [-pi, pi]
        return torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw)).unsqueeze(
            -1
        )  # (num_envs,)

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
            x_min = torch.where(
                maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size
            )
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(
                maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size
            )

            u_x = torch.rand(len(env_ids), device=self.device)
            u_y = torch.rand(len(env_ids), device=self.device)
            goal_local_x = x_min + u_x * (x_max - x_min)
            goal_local_y = y_min + u_y * (y_max - y_min)
            goal_local_x = torch.where(
                goal_local_x >= 0.0,
                (goal_local_x // cell_size) * cell_size,
                (goal_local_x // cell_size + 1) * cell_size,
            )
            goal_local_y = torch.where(
                goal_local_y >= 0.0,
                (goal_local_y // cell_size) * cell_size,
                (goal_local_y // cell_size + 1) * cell_size,
            )

        env_origins = self._terrain.env_origins[env_ids]

        self.goal_pos_w[env_ids, 0] = env_origins[:, 0] + goal_local_x
        self.goal_pos_w[env_ids, 1] = env_origins[:, 1] + goal_local_y
        self.goal_pos_w[env_ids, 2] = env_origins[:, 2]

        self.goal_yaw_w[env_ids] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(-math.pi, math.pi)
        self.goal_quat_w[:, 0] = torch.cos(self.goal_yaw_w / 2)  # w
        self.goal_quat_w[:, 1] = 0.0  # x
        self.goal_quat_w[:, 2] = 0.0  # y
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
            x_min = torch.where(
                maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size
            )
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(
                maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size
            )

            u_x = torch.rand(len(env_ids), device=self.device)
            u_y = torch.rand(len(env_ids), device=self.device)
            start_local_x = x_min + u_x * (x_max - x_min)
            start_local_y = y_min + u_y * (y_max - y_min)
            start_local_x = torch.where(
                start_local_x >= 0.0,
                (start_local_x // cell_size) * cell_size,
                (start_local_x // cell_size + 1) * cell_size,
            )
            start_local_y = torch.where(
                start_local_y >= 0.0,
                (start_local_y // cell_size) * cell_size,
                (start_local_y // cell_size + 1) * cell_size,
            )

        env_origins = self._terrain.env_origins[env_ids]

        self.start_pos_w[env_ids, 0] = env_origins[:, 0] + start_local_x
        self.start_pos_w[env_ids, 1] = env_origins[:, 1] + start_local_y
        self.start_pos_w[env_ids, 2] = (
            env_origins[:, 2] + self._robot.data.default_root_state[env_ids, 2].clone()
        )

        self.start_yaw_w[env_ids] = torch.empty(
            len(env_ids), device=self.device
        ).uniform_(-math.pi, math.pi)
        self.start_quat_w[:, 0] = torch.cos(self.start_yaw_w / 2)  # w
        self.start_quat_w[:, 1] = 0.0  # x
        self.start_quat_w[:, 2] = 0.0  # y
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
    
    def _plot_map_s(self, vis_envs: torch.Tensor, save_dir: str = None) -> None:
        """Save a heatmap image of the global heightfield, true_map (with
        fog-of-war), pred_map, and the model's learned confidence, for each env
        index in vis_envs. Files are saved under save_dir/env{i}/plot{j}.png,
        with {j} auto-incrementing per env so repeated calls never overwrite a
        previous save.

        Args:
            vis_envs: 1D tensor (or int/0-dim tensor) of env indices to plot,
                e.g. torch.tensor([1, 3, 4]).
        """
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "map_debug")

        if torch.is_tensor(vis_envs):
            env_ids = vis_envs.reshape(-1).tolist()
        else:
            env_ids = [int(vis_envs)]

        for env_idx in env_ids:
            env_idx = int(env_idx)
            env_dir = os.path.join(save_dir, f"env{env_idx}")
            os.makedirs(env_dir, exist_ok=True)

            next_idx = self._map_plot_counter.get(env_idx, 1)
            while os.path.exists(os.path.join(env_dir, f"plot{next_idx}.png")):
                next_idx += 1
            self._map_plot_counter[env_idx] = next_idx + 1
            out_path = os.path.join(env_dir, f"plot{next_idx}.png")

            revealed = None
            if hasattr(self, "map_revealed"):
                revealed = self.map_revealed[env_idx].detach().cpu().numpy()

            panels = []
            if getattr(self, "_heightfield_ready", False):
                panels.append(dict(
                    data=self._global_heightfield.detach().cpu().numpy(),
                    title="global heightfield (all terrain)", cmap="terrain",
                ))

            if hasattr(self, "true_map"):
                panels.append(dict(
                    data=self.true_map[env_idx].detach().cpu().numpy(),
                    title=f"true_map[env {env_idx}] (dim = unrevealed)", cmap="terrain",
                    dim_overlay=(~revealed if revealed is not None else None),
                    contour=revealed,
                ))

            if hasattr(self, "pred_map"):
                panels.append(dict(
                    data=self.pred_map[env_idx].detach().cpu().numpy(),
                    title=f"pred_map[env {env_idx}]", cmap="terrain",
                    contour=revealed,
                ))

            if hasattr(self, "pred_map_confidence"):
                conf = self.pred_map_confidence[env_idx].detach().cpu().numpy()
                panels.append(dict(
                    data=conf, title=f"pred confidence[env {env_idx}]",
                    cmap="viridis", vmin=0.0, vmax=1.0, contour=revealed,
                ))
                h = self.pred_map[env_idx].detach().cpu().numpy()
                panels.append(dict(
                    data=h, title=f"pred_map x confidence[env {env_idx}]", cmap="terrain",
                    alpha_by=conf, contour=revealed,
                ))

            if not panels:
                print("[_plot_map_s] Nothing to plot yet (heightfield/true_map/pred_map not populated).")
                return

            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
            if len(panels) == 1:
                axes = [axes]

            for ax, p in zip(axes, panels):
                data = p["data"]
                kwargs = {}
                if "vmin" in p:
                    kwargs["vmin"], kwargs["vmax"] = p["vmin"], p["vmax"]
                im = ax.imshow(data.T, origin="lower", cmap=p["cmap"], aspect="equal", **kwargs)

                if p.get("alpha_by") is not None:
                    fade = np.ones((*p["alpha_by"].T.shape, 4))
                    fade[..., :3] = 1.0
                    fade[..., 3] = 1.0 - p["alpha_by"].T
                    ax.imshow(fade, origin="lower", aspect="equal")

                if p.get("dim_overlay") is not None:
                    dim = np.zeros((*p["dim_overlay"].T.shape, 4))
                    dim[..., 3] = np.where(p["dim_overlay"].T, 0.65, 0.0)
                    ax.imshow(dim, origin="lower", aspect="equal")

                if p.get("contour") is not None:
                    ax.contour(p["contour"].T.astype(float), levels=[0.5], colors="cyan", linewidths=1.0)

                ax.set_title(p["title"])
                ax.set_xlabel("x cells")
                ax.set_ylabel("y cells")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if getattr(self, "_heightfield_ready", False):
                res = self._global_heightfield_res
                x_min = self._global_heightfield_x_min
                y_min = self._global_heightfield_y_min
                root_pos = self._robot.data.root_pos_w[env_idx, :2].detach().cpu()
                gx = (float(root_pos[0]) - x_min) / res
                gy = (float(root_pos[1]) - y_min) / res
                axes[0].plot(gx, gy, "r+", markersize=15, markeredgewidth=2)

            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"[_plot_map_s] Saved {out_path}")
    
    

    def _sanitize_tensor(
        self, tensor: torch.Tensor, name: str, clamp_abs: float | None = None
    ) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            self._finite_warn_counter += 1
            if self._finite_warn_counter <= 5 or self._finite_warn_counter % 500 == 0:
                print(
                    f"[WARN] Non-finite values detected in {name}. Applying nan_to_num safeguard."
                )
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
        padding = torch.zeros(
            tensor.shape[0],
            target_dim - current_dim,
            device=tensor.device,
            dtype=tensor.dtype,
        )
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
