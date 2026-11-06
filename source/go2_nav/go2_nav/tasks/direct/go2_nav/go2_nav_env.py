# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 navigation environment (simulation shell).

This file only contains what is specific to the simulator: scene, sensor emulation (ray hits, noise, dropout, delays),
ground truth, rewards, terminations, curriculum, goals / starts, maze path tables and debugging plots.

The deployable logic lives in the Isaac-free package ``go2_perception``:

    self.loc     LocomotionController  frozen TorchScript policy + obs/action bookkeeping
    self.odom    OdometryEstimator     GRU attitude/velocity + dead-reckoning + pose / position histories
    self.mapper  MapPerception         rolling Kalman map, scan-to-map matching, ego-crop denoiser

and ``sim_support.SimMapSupervisor`` (``self.sim_gt``) provides the ground truth used only for supervision
(training targets, fog of war, teacher map).

Frames: world | start frame (origin = spawn, x = spawn heading; map, goal and histories live here) |
est pose (``self.odom.pose_est`` = [x, y, z, yaw]) | body (yaw-aligned) | ego crop (map window rotated by the yaw).

Per control step:
 1. locomotion policy on the (noisy, delayed) proprioception + height grid
 2. odometry network -> dead-reckoning
 3. map: build scan -> (delay) -> shift window -> scan matching -> Kalman fusion -> ego crop -> denoiser
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.buffers import DelayBuffer

from .go2_perception import HeightGrid, LocomotionController, MapPerception, OdometryEstimator, PerceptionParams
from .go2_perception.math_utils import quat_to_roll_pitch, quat_to_yaw, wrap_to_pi

from .go2_nav_env_cfg import Go2NavEnvCfg
from go2_nav.maze_terrain_cfg import MAZE_REGISTRY
from .utils import env_ids_to_terrain_coords
from .networks.cnn_rnn_model import CNNRNNSeqModel  # noqa: F401  (kept: may be needed for model registration)
from .sim_support import CellDropout, SimMapSupervisor


class Go2NavEnv(DirectRLEnv):
    cfg: Go2NavEnvCfg

    ######################################################
    # INIT FUNCTIONS
    ######################################################

    def __init__(self, cfg: Go2NavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        N = self.num_envs
        self._map_plot_counter = {}  # env_idx -> next file index
        self._ray_cache = {}
        self.pp = PerceptionParams.from_cfg(self.cfg)
        self._init_lidar_extrinsics()
        self.all_env_ids: torch.Tensor = torch.arange(N, device=self.device)

        self._base_id_sensor, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids_sensor, _ = self._contact_sensor.find_bodies(".*_foot")
        self._thigh_ids_sensor, _ = self._contact_sensor.find_bodies(".*_thigh")
        self._hip_ids_sensor, _ = self._contact_sensor.find_bodies(".*_hip")
        self._calf_ids_sensor, _ = self._contact_sensor.find_bodies(".*_calf")

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

        MAZE_REGISTRY.set_dims(
            self.cfg.NUM_ROWS, self.cfg.NUM_COLS, self.cfg.MAX_MAZE_ROWS, self.cfg.MAX_MAZE_COLS, self.device
        )
        self.maze_registery = MAZE_REGISTRY
        self._build_maze_path_tables()

        self.cmd_vel: torch.Tensor = torch.zeros(N, 3, device=self.device)
        self.prev_cmd_vel: torch.Tensor = torch.zeros_like(self.cmd_vel)
        self._num_joints = int(self._robot.data.default_joint_pos.shape[-1])

        # nominal base height above ground (tile-relative z of the base at spawn)
        self._base_z0 = self._robot.data.default_root_state[:, 2].clone()

        # ---------------- sensor delay buffers (sim emulation of latency) ----------------
        self._proprio_lag = self._sample_lag(N)
        self._height_lag = self._sample_lag(N)  # shared by every height-sensor buffer and by the pose histories
        if self.cfg.delay:
            mk = lambda: DelayBuffer(history_length=self.cfg.delay_length, batch_size=N, device=self.device)
            self._proprio_delay_buffer = mk()
            self._loc_heightmap_delay_buffer = mk()
            self._nav_heightmap_delay_buffer = mk()
            self._map_scan_delay_buffer = mk()
            self._proprio_delay_buffer.set_time_lag(self._proprio_lag.clone())
            for buf in self._height_delay_buffers:
                buf.set_time_lag(self._height_lag.clone())

        self._init_goals_and_starts()
        self._finite_warn_counter = 0

        # ---------------- deployable modules ----------------
        train = bool(self.pp.train.train_perception)
        self.loc = LocomotionController(
            self.cfg.locomotion_policy_path, N, self._num_joints, self.cfg.locomotion_action_scale, self.device
        )
        self.odom = OdometryEstimator(N, self.device, self.pp.odom, self.step_dt, train=train, lr=self.pp.train.odom_lr)
        self.mapper = MapPerception(
            N, self.device, self.pp, self._base_z0, self._q_base_lidar, self._t_base_lidar, train=train
        )
        self.sim_gt = SimMapSupervisor(self, self.mapper.map)  # sim only
        self._maybe_load_perception(self.pp.train.checkpoint)

        self.nav_heightmap = torch.zeros(N, 2, self.nav_grid.x_cells, self.nav_grid.y_cells, device=self.device)

        self.terrain_success_rate = torch.zeros(
            self.cfg.terrain.terrain_generator.num_rows,
            self.cfg.terrain.terrain_generator.num_cols,
            device=self.device,
        )
        self.terrain_episode_count = torch.zeros_like(self.terrain_success_rate)
        self._stall_counter = torch.zeros(N, device=self.device)

        m = self.mapper.map
        print(f"Global map initialized with shape: {tuple(m.raw_map.shape)}, ego crop: {m.crop_n}x{m.crop_n}")
        self._episode_sums = {
            key: torch.zeros(N, dtype=torch.float, device=self.device)
            for key in [
                "rew_goal_distance",
                "rew_goal_progress",
                "pen_goal_orientation",
                "pen_goal_penalty",
                "rew_goal_bonus",
                "pen_cmd_rate",
                "pen_cmd_bounds",
                "pen_undesired_contacts",
                "pen_stagnation",
            ]
        }

    @property
    def _height_delay_buffers(self):
        """Buffers fed by the height sensor (same physical sensor -> same lag)."""
        return (self._nav_heightmap_delay_buffer, self._loc_heightmap_delay_buffer, self._map_scan_delay_buffer)

    def _init_lidar_extrinsics(self) -> None:
        base_pos_w = self._robot.data.root_pos_w
        base_quat_w = self._robot.data.root_quat_w
        lidar_pos_w = self._height_scanner.data.pos_w
        lidar_quat_w = self._height_scanner.data.quat_w
        q_w_base = quat_conjugate(base_quat_w)
        # Rotation mapping scanner-frame vectors to base-frame vectors, and scanner origin in the base frame.
        self._q_base_lidar = quat_mul(q_w_base, lidar_quat_w)
        self._t_base_lidar = quat_apply(q_w_base, lidar_pos_w - base_pos_w)

    def _sample_lag(self, n: int) -> torch.Tensor:
        if int(self.cfg.delay_length) <= 0:
            return torch.zeros(n, dtype=torch.long, device=self.device)
        return torch.randint(0, int(self.cfg.delay_length), (n,), device=self.device)

    def _init_goals_and_starts(self):
        # goals
        self.goal_pos_w: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_yaw_w: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.goal_quat_w: torch.Tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self.prev_path_distance: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.num_goal_maze_marker: int = self.maze_registery.num_terrain_types
        self._update_goals(self.all_env_ids, center=False)
        # starts
        self.start_pos_w: torch.Tensor = torch.zeros(self.num_envs, 3, device=self.device)
        self.start_yaw_w: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.start_quat_w: torch.Tensor = torch.zeros(self.num_envs, 4, device=self.device)
        self._update_starts(self.all_env_ids, center=False)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        self._height_scanner = self.cfg.height_scanner.class_type(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self._rots_yaw = torch.empty(self.num_envs, device=self.device)
        self._rots_roll = torch.empty(self.num_envs, device=self.device)
        self._offsets = torch.empty(self.num_envs, 3, device=self.device)
        self._rots_yaw.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
        self._rots_roll.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
        self._offsets.uniform_(-self.cfg.max_offset, self.cfg.max_offset)

        # height grids (policy convention) and sim-only cell dropout
        self.nav_grid = HeightGrid(
            self.cfg.nav_x_range, self.cfg.nav_y_range, self.cfg.nav_cell_size, self.cfg.desired_base_height, self.device
        )
        self.loc_grid = HeightGrid(
            self.cfg.loc_x_range, self.cfg.loc_y_range, self.cfg.loc_cell_size, self.cfg.desired_base_height, self.device
        )
        self.dropout = CellDropout(
            self.loc_grid, self.nav_grid, self.cfg.sigma, self.cfg.n_zeros, self.cfg.max_reset_zeros_freq, self.device
        )

        if self.cfg.vis:
            self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self.start_markers = VisualizationMarkers(self.cfg.start_marker_cfg)
            self.env_markers = VisualizationMarkers(self.cfg.env_marker_cfg)
        if self.cfg.vis or self.cfg.plot:
            self.vis_envs: torch.Tensor = torch.tensor([6])

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=10000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    #################################################################
    # PERCEPTION CHECKPOINTS
    #################################################################

    def save_perception_models(self, path: str) -> None:
        """Odom model + input normaliser + map model (same file format as before; load it on the robot too)."""
        torch.save(
            {**self.odom.checkpoint_state(), **self.mapper.checkpoint_state()}, os.path.expanduser(path)
        )

    def _maybe_load_perception(self, path) -> None:
        if not path:
            return
        ckpt = torch.load(os.path.expanduser(path), map_location=self.device)
        self.odom.load_checkpoint_state(ckpt)
        self.mapper.load_checkpoint_state(ckpt)
        print(f"[Go2NavEnv] Loaded perception checkpoint from {path} (train_perception={self.pp.train.train_perception})")

    #################################################################
    # SIM GROUND TRUTH
    #################################################################

    def _true_pose_s(self) -> torch.Tensor:
        """Ground-truth [x, y, z, yaw] in the start frame."""
        delta_w = self._robot.data.root_pos_w - self.start_pos_w
        c = torch.cos(self.start_yaw_w)
        s = torch.sin(self.start_yaw_w)
        x_s = c * delta_w[:, 0] + s * delta_w[:, 1]
        y_s = -s * delta_w[:, 0] + c * delta_w[:, 1]
        z_s = delta_w[:, 2]
        yaw = wrap_to_pi(quat_to_yaw(self._robot.data.root_quat_w) - self.start_yaw_w)
        return torch.stack([x_s, y_s, z_s, yaw], dim=-1)

    def _get_true_odom_s(self) -> torch.Tensor:
        """Ground-truth [x, y, z, roll, pitch, yaw, v_b(3), w_b(3)] in the spawn frame."""
        pose = self._true_pose_s()
        roll_w, pitch_w = quat_to_roll_pitch(self._robot.data.root_quat_w)
        rpy_s = torch.stack([roll_w, pitch_w, pose[:, 3]], dim=-1)
        return torch.cat(
            [pose[:, :3], rpy_s, self._robot.data.root_lin_vel_b, self._robot.data.root_ang_vel_b], dim=-1
        ).clone()

    def _odom_target(self) -> torch.Tensor:
        t = self._get_true_odom_s()
        return torch.cat([t[:, 3:5], t[:, 6:9], t[:, 9:12]], dim=-1)  # roll, pitch, v_b, w_b

    def _log(self, d: dict) -> None:
        self.extras.setdefault("log", {}).update(d)

    def _log_pose_metrics(self) -> None:
        true = self._true_pose_s()
        est, dr = self.odom.pose_est, self.odom.pose_dr
        self._log(
            {
                "odom/ate_xy_corrected": torch.linalg.norm(est[:, :2] - true[:, :2], dim=-1).mean(),
                "odom/ate_xy_dead_reckoning": torch.linalg.norm(dr[:, :2] - true[:, :2], dim=-1).mean(),
                "odom/yaw_err_corrected": wrap_to_pi(est[:, 3] - true[:, 3]).abs().mean(),
                "odom/yaw_err_dead_reckoning": wrap_to_pi(dr[:, 3] - true[:, 3]).abs().mean(),
                "odom/z_err_corrected": (est[:, 2] - true[:, 2]).abs().mean(),
            }
        )

    #################################################################
    # SENSOR EMULATION (rays -> height grids / map scan)
    #################################################################

    def _apply_offsets(self, points: torch.Tensor) -> torch.Tensor:
        return points + self._offsets.unsqueeze(1)

    def _apply_yaw_rotation(self, points: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(self._rots_yaw).unsqueeze(-1)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        return torch.stack((x * cos_a + z * sin_a, y, -x * sin_a + z * cos_a), dim=-1)

    def _apply_roll_rotation(self, points: torch.Tensor) -> torch.Tensor:
        angles = torch.deg2rad(self._rots_roll).unsqueeze(-1)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        return torch.stack((x, y * cos_a - z * sin_a, y * sin_a + z * cos_a), dim=-1)

    def _rays_in_lidar_frame(self, randomize: bool) -> torch.Tensor:
        """Ray hits in the lidar frame (policy convention = -(hit rel. to scanner)), cached per step."""
        key = bool(randomize)
        if key in self._ray_cache:
            return self._ray_cache[key]
        data = self._height_scanner.data
        num_envs, num_rays, _ = data.ray_hits_w.shape
        rays_rel_w = -data.ray_hits_w + data.pos_w.unsqueeze(1)
        rays_lidar = quat_apply(
            quat_conjugate(data.quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
            rays_rel_w.reshape(-1, 3),
        ).reshape(num_envs, num_rays, 3)
        if randomize:
            rays_lidar = self._apply_yaw_rotation(rays_lidar)
            rays_lidar = self._apply_roll_rotation(rays_lidar)
            rays_lidar = self._apply_offsets(rays_lidar)
        self._ray_cache[key] = rays_lidar
        return rays_lidar

    def _height_image(self, grid: HeightGrid, locomotion: bool, randomize: bool, apply_dropout: bool):
        """Height grid [N, X, Y] + valid mask, with the sim-only noise / dropout."""
        rays = self._rays_in_lidar_frame(randomize)
        height, hit = grid.compute(rays, noise=0.01 if randomize else 0.0)
        valid = hit
        if apply_dropout:
            height, keep = self.dropout.apply(height, locomotion)
            valid = hit & keep
        return grid.to_image(height), grid.to_image(valid)

    def _build_nav_heightmap(self) -> torch.Tensor:
        """Policy-convention navigation grid (student_height_scan)."""
        height, valid = self._height_image(
            self.nav_grid, locomotion=False, randomize=self.cfg.randomize, apply_dropout=self.cfg.randomize
        )
        height = self._sanitize_tensor(height, "height_data_nav", clamp_abs=10.0)
        nav_heightmap = torch.stack([height, valid], dim=1)
        if self.cfg.delay:
            nav_heightmap = self._nav_heightmap_delay_buffer.compute(nav_heightmap)
        self.nav_heightmap = nav_heightmap
        return nav_heightmap

    def _update_map_perception(self) -> None:
        # student navigation grid (policy convention)
        self._build_nav_heightmap()

        # emulated body-frame scan (uses the ESTIMATED roll/pitch), delayed like the other height buffers
        rays = self._rays_in_lidar_frame(self.cfg.randomize)
        noise = float(self.pp.map.scan_noise) if self.cfg.randomize else 0.0
        scan = self.mapper.build_scan(-rays, *self.odom.roll_pitch, noise_std=noise)
        if self.cfg.delay:
            scan = self._map_scan_delay_buffer.compute(scan)

        self.sim_gt.push_true_pose(self._true_pose_s())
        lag = self._height_lag if self.cfg.delay else None
        self.mapper.update(scan, self.odom, lag=lag, supervisor=self.sim_gt)
        self._log(self.mapper.last_log)

    #################################################################
    # LOCOMOTION OBSERVATIONS
    #################################################################

    def _build_locomotion_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        height, valid = self._height_image(
            self.loc_grid, locomotion=True, randomize=self.cfg.randomize, apply_dropout=self.cfg.randomize
        )
        height = self._sanitize_tensor(height, "height_data_loc", clamp_abs=10.0)
        grid = torch.stack([height, valid.float()], dim=1)
        if self.cfg.delay:
            grid = self._loc_heightmap_delay_buffer.compute(grid)

        r = float(self.cfg.randomize)
        noisy = lambda x, a: x + (2.0 * torch.rand_like(x) - 1.0) * a * r
        proprio_delayed = torch.cat(
            [noisy(base_ang_vel, 0.1), noisy(projected_gravity, 0.05), noisy(joint_pos_rel, 0.01), noisy(joint_vel, 0.1)],
            dim=-1,
        )
        proprio_delayed = self._sanitize_tensor(proprio_delayed, "proprio_delayed", clamp_abs=50.0)
        if self.cfg.delay:
            proprio_delayed = self._proprio_delay_buffer.compute(proprio_delayed)
        self.proprio_delayed = proprio_delayed

        proprio_loc = self.loc.build_obs(proprio_delayed, self.cmd_vel)
        return self._sanitize_tensor(proprio_loc, "proprio_loc", clamp_abs=50.0), grid

    #################################################################
    # MAZE PATH TABLES
    #################################################################

    def _build_maze_path_tables(self, env_ids: torch.Tensor | None = None) -> None:
        """Build or incrementally update shared shortest-path tables."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            initial_build = not hasattr(self, "_maze_terrain_coords")
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
            initial_build = not hasattr(self, "_maze_terrain_coords")

        if env_ids.numel() == 0:
            return

        terrain_coords = env_ids_to_terrain_coords(env_ids, self._terrain)
        if initial_build:
            changed_ids = env_ids
        else:
            changed = torch.any(terrain_coords != self._maze_terrain_coords[env_ids], dim=1)
            changed_ids = env_ids[changed]
            if changed_ids.numel() == 0:
                return
            terrain_coords = env_ids_to_terrain_coords(changed_ids, self._terrain)

        layouts = self.maze_registery.get_mazes_terrain_coords(terrain_coords).detach().cpu()
        maze_rows = self.maze_registery.get_num_rows(terrain_coords).detach().cpu()
        maze_cols = torch.full_like(maze_rows, int(self.cfg.MAX_MAZE_COLS))
        max_cols = int(self.cfg.MAX_MAZE_COLS)
        max_cells = int(self.cfg.MAX_MAZE_ROWS) * max_cols

        if initial_build:
            self._maze_terrain_coords = torch.empty((self.num_envs, 2), dtype=terrain_coords.dtype, device=self.device)
            self._maze_layouts = torch.empty((self.num_envs, *layouts.shape[1:]), dtype=layouts.dtype)
            self._maze_rows = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
            self._maze_cols = torch.empty_like(self._maze_rows)
            self._maze_path_table_ids = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
            self._maze_path_distances = torch.empty((0, max_cells, max_cells), device=self.device)
            self._maze_table_ids_by_key: dict[tuple[int, int, bytes], int] = {}

        self._maze_terrain_coords[changed_ids] = terrain_coords
        self._maze_layouts[changed_ids.cpu()] = layouts
        self._maze_rows[changed_ids] = maze_rows.to(self.device)
        self._maze_cols[changed_ids] = maze_cols.to(self.device)

        new_tables = []
        for local_idx, env_idx in enumerate(changed_ids.tolist()):
            rows = int(maze_rows[local_idx])
            cols = int(maze_cols[local_idx])
            active_layout = layouts[local_idx, :rows, :cols].to(torch.int8).numpy()
            key = (rows, cols, active_layout.tobytes())
            table_id = self._maze_table_ids_by_key.get(key)
            if table_id is None:
                table_id = len(self._maze_table_ids_by_key)
                self._maze_table_ids_by_key[key] = table_id
                table = self._compute_maze_distance_table(active_layout, rows, cols)
                padded_table = torch.full((max_cells, max_cells), float("inf"))
                padded_table[: rows * cols, : rows * cols] = table
                new_tables.append(padded_table)
            self._maze_path_table_ids[env_idx] = table_id

        if new_tables:
            self._maze_path_distances = torch.cat(
                [self._maze_path_distances, torch.stack(new_tables).to(self.device)], dim=0
            )

    @staticmethod
    def _compute_maze_distance_table(layout, rows: int, cols: int) -> torch.Tensor:
        """Return all-pairs shortest cell distances for a recorded maze layout."""
        import numpy as np

        cell_count = rows * cols
        distances = np.full((cell_count, cell_count), np.inf, dtype=np.float32)
        for source in range(cell_count):
            source_row, source_col = divmod(source, cols)
            distances[source, source] = 0.0
            queue = [(source_row, source_col)]
            head = 0
            while head < len(queue):
                row, col = queue[head]
                head += 1
                current = row * cols + col
                neighbors = (
                    (-1, 0, 3, 1),  # recorded row - 1: current north/south wall channels
                    (1, 0, 1, 3),
                    (0, -1, 4, 2),
                    (0, 1, 2, 4),
                )
                for row_delta, col_delta, current_wall, neighbor_wall in neighbors:
                    next_row = row + row_delta
                    next_col = col + col_delta
                    if not (0 <= next_row < rows and 0 <= next_col < cols):
                        continue
                    if layout[row, col, current_wall] or layout[next_row, next_col, neighbor_wall]:
                        continue
                    next_cell = next_row * cols + next_col
                    if np.isinf(distances[source, next_cell]):
                        distances[source, next_cell] = distances[source, current] + 1.0
                        queue.append((next_row, next_col))
        return torch.from_numpy(distances)

    def _world_to_maze_cells(self, positions_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert world XY positions to recorded maze row and column indices."""
        local = positions_w[:, :2] - self._terrain.env_origins[:, :2]
        cell_size = float(self.cfg.terrain.terrain_generator.sub_terrains["maze"].cell_size)
        spawn_rows = self._maze_rows // 2
        spawn_cols = self._maze_cols // 2
        original_rows = spawn_rows + torch.round(local[:, 0] / cell_size).long()
        original_cols = spawn_cols - torch.round(local[:, 1] / cell_size).long()
        recorded_rows = self._maze_rows - 1 - original_rows
        recorded_cols = original_cols
        recorded_rows = torch.maximum(torch.zeros_like(recorded_rows), recorded_rows)
        recorded_rows = torch.minimum(recorded_rows, self._maze_rows - 1)
        recorded_cols = torch.maximum(torch.zeros_like(recorded_cols), recorded_cols)
        recorded_cols = torch.minimum(recorded_cols, self._maze_cols - 1)
        return recorded_rows, recorded_cols

    def _maze_cell_centers_world(self, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        """Return world XY centers for recorded maze cells."""
        cell_size = float(self.cfg.terrain.terrain_generator.sub_terrains["maze"].cell_size)
        original_rows = self._maze_rows - 1 - rows
        original_cols = cols
        spawn_rows = self._maze_rows // 2
        spawn_cols = self._maze_cols // 2
        local_x = (original_rows - spawn_rows).to(torch.float32) * cell_size
        local_y = -(original_cols - spawn_cols).to(torch.float32) * cell_size
        return self._terrain.env_origins[:, :2] + torch.stack((local_x, local_y), dim=-1)

    def _compute_maze_path_distance(self, positions_w: torch.Tensor) -> torch.Tensor:
        """Compute maze distance (meters) using progress along the next path corridor."""
        robot_rows, robot_cols = self._world_to_maze_cells(positions_w)
        goal_rows, goal_cols = self._world_to_maze_cells(self.goal_pos_w)
        max_cols = int(self.cfg.MAX_MAZE_COLS)
        max_rows = int(self.cfg.MAX_MAZE_ROWS)
        robot_flat = robot_rows * max_cols + robot_cols
        goal_flat = goal_rows * max_cols + goal_cols
        cell_distance = self._maze_path_distances[self._maze_path_table_ids, robot_flat, goal_flat]
        cell_size = float(self.cfg.terrain.terrain_generator.sub_terrains["maze"].cell_size)

        robot_centers = self._maze_cell_centers_world(robot_rows, robot_cols)

        # Select the neighboring cell that is one hop closer to the goal.
        best_next_dist = torch.full_like(cell_distance, float("inf"))
        next_rows = robot_rows.clone()
        next_cols = robot_cols.clone()
        for row_delta, col_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate_rows = robot_rows + row_delta
            candidate_cols = robot_cols + col_delta
            valid = (
                (candidate_rows >= 0)
                & (candidate_rows < max_rows)
                & (candidate_cols >= 0)
                & (candidate_cols < max_cols)
            )
            nbr_rows = candidate_rows.clamp(0, max_rows - 1)
            nbr_cols = candidate_cols.clamp(0, max_cols - 1)
            nbr_flat = nbr_rows * max_cols + nbr_cols

            step_dist = self._maze_path_distances[self._maze_path_table_ids, robot_flat, nbr_flat]
            nbr_to_goal = self._maze_path_distances[self._maze_path_table_ids, nbr_flat, goal_flat]
            connected = valid & (step_dist == 1)
            candidate_dist = torch.where(connected, nbr_to_goal, torch.full_like(nbr_to_goal, float("inf")))

            improve = candidate_dist < best_next_dist
            best_next_dist = torch.where(improve, candidate_dist, best_next_dist)
            next_rows = torch.where(improve, nbr_rows, next_rows)
            next_cols = torch.where(improve, nbr_cols, next_cols)

        next_centers = self._maze_cell_centers_world(next_rows, next_cols)
        direction = next_centers - robot_centers
        direction_norm = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
        unit_direction = direction / direction_norm
        forward_progress = ((positions_w[:, :2] - robot_centers) * unit_direction).sum(dim=-1)
        forward_progress = forward_progress.clamp(min=-cell_size, max=cell_size)
        distance = cell_distance * cell_size - forward_progress
        # direct Euclidean distance in METERS
        euclidean = torch.linalg.norm(self.goal_pos_w[:, :2] - positions_w[:, :2], dim=-1)
        same_cell = (robot_rows == goal_rows) & (robot_cols == goal_cols)
        distance = torch.where(same_cell, euclidean, distance)
        return torch.where(torch.isfinite(distance), distance, euclidean)

    def _shortest_maze_path_cells(self, env_idx: int, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Return a shortest path in recorded cell coordinates for plotting."""
        layout = self._maze_layouts[env_idx].numpy()
        rows = int(self._maze_rows[env_idx])
        cols = int(self._maze_cols[env_idx])
        queue = [start]
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        head = 0
        while head < len(queue):
            row, col = queue[head]
            head += 1
            if (row, col) == goal:
                break
            for row_delta, col_delta, current_wall, neighbor_wall in ((-1, 0, 3, 1), (1, 0, 1, 3), (0, -1, 4, 2), (0, 1, 2, 4)):
                next_cell = (row + row_delta, col + col_delta)
                if not (0 <= next_cell[0] < rows and 0 <= next_cell[1] < cols):
                    continue
                if layout[row, col, current_wall] or layout[next_cell[0], next_cell[1], neighbor_wall] or next_cell in previous:
                    continue
                previous[next_cell] = (row, col)
                queue.append(next_cell)
        if goal not in previous:
            return [start]
        path = []
        cell = goal
        while cell is not None:
            path.append(cell)
            cell = previous[cell]
        return list(reversed(path))

    #################################################################
    # DirectRLEnv API
    #################################################################

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions = [vxd, vyd, wzd]. prev_cmd_vel must be stored BEFORE overwriting cmd_vel.
        self._ray_cache.clear()
        self.prev_cmd_vel.copy_(self.cmd_vel)
        self.cmd_vel.copy_(actions)

        # 1. Locomotion policy
        proprio_obs_loc, height_grid_loc = self._build_locomotion_observations()
        self.loc.act(proprio_obs_loc, height_grid_loc)

        # 2. Odometry: network predicts attitude + body velocities, the pose is integrated outside the network
        target = self._odom_target() if self.odom.train_enabled else None
        self.odom.step(self.proprio_delayed, self.cmd_vel, self.loc.actions, target)
        self._log(self.odom.last_log)

        # 3. Map: scan, scan-to-map correction, Kalman fusion, ego-crop denoising
        self._update_map_perception()
        self.odom.pos_hist.push(self.odom.pose_est[:, :3])  # after the correction
        self._log_pose_metrics()

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.loc.joint_targets(self._robot.data.default_joint_pos))

    def _get_observations(self) -> dict:
        self._ray_cache.clear()
        height_t, valid_t = self._height_image(self.nav_grid, locomotion=False, randomize=False, apply_dropout=False)
        height_t = self._sanitize_tensor(height_t, "height_data_teacher", clamp_abs=10.0)
        height_data_teacher = torch.stack([height_t, valid_t], dim=1)

        # ---- goal expressed in the ESTIMATED robot frame (goal is given in the start frame) ----
        pose = self.odom.pose_est
        c = torch.cos(pose[:, 3])
        s = torch.sin(pose[:, 3])
        d = self._get_goal_pos_s() - pose[:, :2]
        goal_rel_xy = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)
        goal_rel_yaw = wrap_to_pi(self._get_goal_yaw_s().squeeze(-1) - pose[:, 3]).unsqueeze(-1)

        hist_short, hist_long = self.odom.pos_hist.relative_to(pose)
        vel = self.odom.student_vel  # predicted [vx_b, vy_b, wz_b]

        student_proprio = torch.cat([goal_rel_xy, goal_rel_yaw, hist_short, hist_long, vel, self.cmd_vel], dim=-1)

        true_odom = self._get_true_odom_s()
        goal_delta = self.goal_pos_w - self._robot.data.root_pos_w
        yaw_error = wrap_to_pi(self.goal_yaw_w - quat_to_yaw(self._robot.data.root_quat_w))
        goal_heading = torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)

        terrain_coords: torch.Tensor = env_ids_to_terrain_coords(self.all_env_ids, self._terrain)
        mazes: torch.Tensor = self.maze_registery.get_mazes_terrain_coords(terrain_coords).clone()
        terrain_type = mazes[:, :, :, 0].long()
        maze_encoded = torch.nn.functional.one_hot(terrain_type, self.maze_registery.num_terrain_types)
        maze_remaining = mazes[:, :, :, 1:]
        maze_features = torch.cat(
            [maze_encoded.reshape(self.num_envs, -1), maze_remaining.reshape(self.num_envs, -1)], dim=-1
        )

        teacher_proprio = torch.cat(
            [student_proprio.clone(), true_odom, goal_delta, goal_heading, maze_features], dim=-1
        )

        student_proprio = self._sanitize_tensor(student_proprio, "student_proprio", clamp_abs=100.0)
        teacher_proprio = self._sanitize_tensor(teacher_proprio, "teacher_proprio", clamp_abs=100.0)

        teacher_map = self.sim_gt.teacher_map(self._true_pose_s())

        return {
            "student_proprio": student_proprio,
            "student_height_scan": self.nav_heightmap.clone(),
            "student_map": self.mapper.obs.clone(),
            "teacher_proprio": teacher_proprio,
            "teacher_height_scan": height_data_teacher,
            "teacher_map": teacher_map,
        }

    def log_infos(self):
        if self.cfg.vis:
            self.env_markers.visualize(
                translations=self._robot.data.root_pos_w[self.vis_envs],
                orientations=self._robot.data.root_quat_w[self.vis_envs],
            )

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.vis:
            self.log_infos()

        # Path distance reward: shortest maze path from the robot to the goal
        root_pos_w = self._robot.data.root_pos_w
        path_distance = self._compute_maze_path_distance(root_pos_w)
        rew_goal_distance = torch.exp(-path_distance / max(1e-4, float(self.cfg.goal_distance_sigma)))

        # Yaw orientation penalty (only when the robot is close to the goal)
        robot_yaw = quat_to_yaw(self._robot.data.root_quat_w)
        yaw_error = wrap_to_pi(self.goal_yaw_w - robot_yaw)
        pen_goal_orientation = torch.where(
            path_distance < 0.1,
            (0.1 - path_distance) * -torch.square(yaw_error) / float(self.cfg.goal_orientation_sigma),
            0.0,
        )

        # Robot progress to goal
        raw_progress = self.prev_path_distance - path_distance
        rew_goal_progress = torch.clamp(raw_progress, -self.cfg.goal_progress_clip, self.cfg.goal_progress_clip)
        self.prev_path_distance.copy_(path_distance.detach())

        # Time penalty
        pen_time_penalty = torch.ones_like(path_distance)

        # Violent commands penalty
        pen_cmd_rate = torch.sum(torch.square(self.cmd_vel - self.prev_cmd_vel), dim=-1)

        # Commands bounds
        # 1. High commands (higher than what the locomotion policy has learned)
        cmd_bounds_high = torch.sum(
            torch.where(torch.abs(self.cmd_vel) >= 1.0, torch.abs(self.cmd_vel), 0.0), dim=-1
        )
        # 2. Low translational command while still far from the goal
        cmd_mag = torch.linalg.norm(self.cmd_vel[:, :2], dim=-1)
        low_cmd_penalty = torch.where(
            torch.logical_and(cmd_mag <= 0.2, path_distance > 0.1),
            torch.exp(-cmd_mag / self.cfg.cmd_vel_sigma),
            0.0,
        )
        pen_cmd_bounds = cmd_bounds_high + low_cmd_penalty

        # Reward for reaching the goal
        rew_goal_bonus = self.goal_reached.float()

        # Undesired contacts penalty
        contact_forces = torch.norm(
            self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids_sensor],
            dim=-1,
        )
        is_contact = torch.max(contact_forces, dim=1)[0] > 1.0
        pen_undesired_contacts = torch.sum(is_contact, dim=1)

        # Stagnation penalty
        making_progress = raw_progress > self.cfg.stagnation_progress_eps
        self._stall_counter = torch.where(making_progress, 0.0, self._stall_counter + 1)
        pen_stagnation = (self._stall_counter >= self.cfg.stagnation_window).float()

        rewards = {
            "rew_goal_distance": self.cfg.rew_scale_goal_distance * rew_goal_distance * self.step_dt,
            "pen_goal_orientation": self.cfg.rew_scale_goal_orientation * pen_goal_orientation * self.step_dt,
            "rew_goal_progress": self.cfg.rew_scale_goal_progress * rew_goal_progress * self.step_dt,
            "pen_goal_penalty": self.cfg.rew_scale_time_penalty * pen_time_penalty * self.step_dt,
            "pen_cmd_rate": self.cfg.rew_scale_cmd_rate * pen_cmd_rate * self.step_dt,
            "pen_cmd_bounds": self.cfg.rew_scale_cmd_bounds * pen_cmd_bounds * self.step_dt,
            "rew_goal_bonus": self.cfg.rew_scale_goal_bonus * rew_goal_bonus * self.step_dt,
            "pen_undesired_contacts": self.cfg.rew_scale_undesired_contacts * pen_undesired_contacts * self.step_dt,
            "pen_stagnation": self.cfg.rew_scale_stagnation * pen_stagnation * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        reward = self._sanitize_tensor(reward, "reward", clamp_abs=1000.0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        goal_distance = torch.linalg.norm(self.goal_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=-1)
        yaw_error = wrap_to_pi(self.goal_yaw_w - quat_to_yaw(self._robot.data.root_quat_w)).abs()
        self.goal_reached = (goal_distance < self.cfg.goal_reached_distance) & (yaw_error < self.cfg.goal_reached_yaw)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_base = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id_sensor], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )

        terminated = died_base | self.goal_reached
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        self.sim_gt.precompute_heightfield()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids is not None and len(env_ids) != self.num_envs and self.cfg.plot:
            plot = False
            for env_id in env_ids:
                cond = env_id in self.vis_envs.tolist()
                if cond:
                    plot = True
            if plot:
                self._plot_map_s(self.vis_envs, save_dir="/mnt/D/dev/robotics/nvidia/isaaclab/go2_nav/map_debug")
        self._robot.reset(env_ids)
        move_up = None
        move_down = None
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

        if move_up is not None and move_down is not None:
            self._terrain.update_env_origins(env_ids_tensor, move_up, move_down)
            # Curriculum changes terrain_levels -> maze layouts may have changed.
            self._build_maze_path_tables(env_ids_tensor)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        default_root_state = self._robot.data.default_root_state[env_ids_tensor].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids_tensor]

        # ---- deployable modules ----
        self.loc.reset(env_ids_tensor)
        self.odom.reset(env_ids_tensor, randomize=self.cfg.randomize)
        self.cmd_vel[env_ids_tensor] = 0.0
        self.prev_cmd_vel[env_ids_tensor] = 0.0
        self.goal_reached[env_ids_tensor] = False
        self._stall_counter[env_ids_tensor] = 0.0

        self._update_goals(env_ids_tensor, False)
        self._update_starts(env_ids_tensor, False)

        # the previous episode's map must not leak into the new one; GT after the start poses changed
        self.mapper.reset(env_ids_tensor)
        self.sim_gt.reset(env_ids_tensor)
        self.nav_heightmap[env_ids_tensor] = 0.0

        # randomisation of the scan misalignment, per env
        num_resets = env_ids_tensor.numel()
        self._rots_yaw[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.max_rot, self.cfg.max_rot
        )
        self._rots_roll[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.max_rot, self.cfg.max_rot
        )
        self._offsets[env_ids_tensor] = torch.empty(num_resets, 3, device=self.device).uniform_(
            -self.cfg.max_offset, self.cfg.max_offset
        )

        # ---- delay buffers: only the envs being reset get a new lag (others keep theirs) ----
        if self.cfg.delay:
            ids_list = env_ids_tensor.tolist()
            self._proprio_delay_buffer.reset(ids_list)
            for buf in self._height_delay_buffers:
                buf.reset(ids_list)
            self._proprio_lag[env_ids_tensor] = self._sample_lag(num_resets)
            # height-sensor buffers (nav, loc, map scan) and the pose histories share one lag: same physical sensor
            self._height_lag[env_ids_tensor] = self._sample_lag(num_resets)
            self._proprio_delay_buffer.set_time_lag(self._proprio_lag.clone())
            for buf in self._height_delay_buffers:
                buf.set_time_lag(self._height_lag.clone())

        root_state = torch.cat([self.start_pos_w[env_ids], self.start_quat_w[env_ids]], dim=-1)

        self.prev_path_distance[env_ids_tensor] = self._compute_maze_path_distance(self.start_pos_w)[env_ids_tensor]

        self._robot.write_root_pose_to_sim(root_state, env_ids_tensor)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

        # ---- logging (keep the per-step odom/map/match metrics, they are not episodic) ----
        kept = {k: v for k, v in self.extras.get("log", {}).items() if k.startswith(("odom/", "map/", "match/"))}
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids_tensor] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(kept)
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids_tensor]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids_tensor]).item()
        self.extras["log"].update(extras)

    #################################################################
    # GOALS / STARTS
    #################################################################

    def _get_goal_pos_s(self) -> torch.Tensor:
        """Goal position in the start frame (2D, x forward, y left)."""
        delta_w = self.goal_pos_w - self.start_pos_w  # (num_envs, 3)
        cos_yaw = torch.cos(-self.start_yaw_w)
        sin_yaw = torch.sin(-self.start_yaw_w)
        x_r = cos_yaw * delta_w[:, 0] - sin_yaw * delta_w[:, 1]
        y_r = sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        return torch.stack([x_r, y_r], dim=-1)

    def _get_goal_yaw_s(self) -> torch.Tensor:
        """Goal heading relative to start heading, wrapped to [-pi, pi]. Shape (num_envs, 1)."""
        delta_yaw = self.goal_yaw_w - self.start_yaw_w
        return torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw)).unsqueeze(-1)

    def _sample_cell_positions(self, env_ids: torch.Tensor, center: bool):
        """Random maze-cell-aligned (x, y) offsets (env frame) for goals / starts, plus the env origins."""
        maze_cfg = self.cfg.terrain.terrain_generator.sub_terrains["maze"]
        cell_size = float(maze_cfg.cell_size)
        max_cols = max(1, int(round(maze_cfg.maze_max_cols)))

        terrain_coords = env_ids_to_terrain_coords(env_ids, self._terrain)
        maze_rows = self.maze_registery.get_num_rows(terrain_coords)
        maze_cols = torch.full_like(maze_rows, max_cols)
        maze_rows_f = maze_rows.to(dtype=torch.float32)
        maze_cols_f = maze_cols.to(dtype=torch.float32)
        x_extent = maze_rows_f * cell_size
        y_extent = maze_cols_f * cell_size

        if center:
            local_x = torch.zeros_like(x_extent)
            local_y = torch.zeros_like(y_extent)
        else:
            x_min = torch.where(maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size)
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size)
            u_x = torch.rand(len(env_ids), device=self.device)
            u_y = torch.rand(len(env_ids), device=self.device)
            local_x = x_min + u_x * (x_max - x_min)
            local_y = y_min + u_y * (y_max - y_min)
            local_x = torch.where(local_x >= 0.0, (local_x // cell_size) * cell_size, (local_x // cell_size + 1) * cell_size)
            local_y = torch.where(local_y >= 0.0, (local_y // cell_size) * cell_size, (local_y // cell_size + 1) * cell_size)
        return local_x, local_y, self._terrain.env_origins[env_ids]

    @staticmethod
    def _yaw_to_quat_(quat: torch.Tensor, yaw: torch.Tensor) -> None:
        quat[:, 0] = torch.cos(yaw / 2)  # w
        quat[:, 1] = 0.0
        quat[:, 2] = 0.0
        quat[:, 3] = torch.sin(yaw / 2)  # z

    def _visualize(self, markers, pos, quat) -> None:
        if type(self.vis_envs) == int:
            markers.visualize(translations=pos[self.vis_envs].unsqueeze(0), orientations=quat[self.vis_envs].unsqueeze(0))
        else:
            markers.visualize(translations=pos[self.vis_envs], orientations=quat[self.vis_envs])

    def _update_goals(self, env_ids: torch.Tensor, center: bool = False) -> None:
        """Goal positions: maze centre if ``center`` else uniform over the env's maze (cell-aligned)."""
        local_x, local_y, env_origins = self._sample_cell_positions(env_ids, center)
        self.goal_pos_w[env_ids, 0] = env_origins[:, 0] + local_x
        self.goal_pos_w[env_ids, 1] = env_origins[:, 1] + local_y
        self.goal_pos_w[env_ids, 2] = env_origins[:, 2]
        self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self._yaw_to_quat_(self.goal_quat_w, self.goal_yaw_w)
        if self.cfg.vis:
            self._visualize(self.goal_markers, self.goal_pos_w, self.goal_quat_w)

    def _update_starts(self, env_ids: torch.Tensor, center: bool = False) -> None:
        """Start positions (same sampling rules as goals)."""
        local_x, local_y, env_origins = self._sample_cell_positions(env_ids, center)
        self.start_pos_w[env_ids, 0] = env_origins[:, 0] + local_x
        self.start_pos_w[env_ids, 1] = env_origins[:, 1] + local_y
        self.start_pos_w[env_ids, 2] = env_origins[:, 2] + self._robot.data.default_root_state[env_ids, 2].clone()
        self.start_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self._yaw_to_quat_(self.start_quat_w, self.start_yaw_w)
        if self.cfg.vis:
            self._visualize(self.start_markers, self.start_pos_w, self.start_quat_w)

    #################################################################
    # DEBUG PLOT
    #################################################################

    def _plot_map_s(self, vis_envs: torch.Tensor, save_dir: str = None) -> None:
        """Save a debug image per env in vis_envs under save_dir/env{i}/plot{j}.png (maps + ego crops)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "map_debug")
        env_ids = vis_envs.reshape(-1).tolist() if torch.is_tensor(vis_envs) else [int(vis_envs)]
        emap = self.mapper.map

        for env_idx in env_ids:
            env_idx = int(env_idx)
            env_dir = os.path.join(save_dir, f"env{env_idx}")
            os.makedirs(env_dir, exist_ok=True)
            next_idx = self._map_plot_counter.get(env_idx, 1)
            while os.path.exists(os.path.join(env_dir, f"plot{next_idx}.png")):
                next_idx += 1
            self._map_plot_counter[env_idx] = next_idx + 1
            out_path = os.path.join(env_dir, f"plot{next_idx}.png")

            ids = torch.tensor([env_idx], device=self.device)
            crops = {**emap.input_crop(self.odom.pose_est, ids), **self.sim_gt.targets(emap, ids)}
            true_pose = self._true_pose_s()[env_idx].detach().cpu().numpy()
            est_pose = self.odom.pose_est[env_idx].detach().cpu().numpy()
            seen = self.sim_gt.seen[env_idx].detach().cpu().numpy()
            var = emap.variance[env_idx].detach().cpu().numpy()
            raw = emap.raw_map[env_idx].detach().cpu().numpy()
            observed = var < float(self.pp.map.observed_var_thresh)
            obs = self.mapper.obs

            panels = [
                dict(data=self.sim_gt.true_map[env_idx].detach().cpu().numpy(), title="true_map (dim = never seen)",
                     cmap="terrain", dim=~seen, full=True),
                dict(data=np.where(observed, raw, np.nan), title="fused raw_map (observed cells)", cmap="terrain", full=True),
                dict(data=crops["x"][0, 0].cpu().numpy(), title="ego input height (rel. to robot)", cmap="terrain"),
                dict(data=obs[env_idx, 0].detach().cpu().numpy(), title="ego predicted height", cmap="terrain"),
                dict(data=crops["target"][0].cpu().numpy(), title="ego target height", cmap="terrain"),
                dict(data=obs[env_idx, 1].detach().cpu().numpy(), title="planner confidence",
                     cmap="viridis", vmin=0.0, vmax=1.0),
            ]

            fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
            for ax, p in zip(axes, panels):
                kwargs = {}
                if "vmin" in p:
                    kwargs["vmin"], kwargs["vmax"] = p["vmin"], p["vmax"]
                im = ax.imshow(p["data"].T, origin="lower", cmap=p["cmap"], aspect="equal", **kwargs)
                if p.get("dim") is not None:
                    dim = np.zeros((*p["dim"].T.shape, 4))
                    dim[..., 3] = np.where(p["dim"].T, 0.65, 0.0)
                    ax.imshow(dim, origin="lower", aspect="equal")
                if p.get("full"):
                    origin = emap.origin[env_idx].cpu().numpy()
                    ax.plot((true_pose[0] - origin[0]) / emap.cs - 0.5, (true_pose[1] - origin[1]) / emap.cs - 0.5,
                            "bx", markersize=10, markeredgewidth=2)
                    ax.plot((est_pose[0] - origin[0]) / emap.cs - 0.5, (est_pose[1] - origin[1]) / emap.cs - 0.5,
                            "r+", markersize=12, markeredgewidth=2)
                ax.set_title(p["title"], fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"[_plot_map_s] Saved {out_path}")

    #################################################################
    # UTILS
    #################################################################

    def _sanitize_tensor(self, tensor: torch.Tensor, name: str, clamp_abs: float | None = None) -> torch.Tensor:
        if not torch.isfinite(tensor).all():
            self._finite_warn_counter += 1
            if self._finite_warn_counter <= 5 or self._finite_warn_counter % 500 == 0:
                print(f"[WARN] Non-finite values detected in {name}. Applying nan_to_num safeguard.")
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp_abs is not None:
            tensor = torch.clamp(tensor, min=-clamp_abs, max=clamp_abs)
        return tensor