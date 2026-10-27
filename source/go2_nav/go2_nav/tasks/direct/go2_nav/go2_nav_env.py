# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 navigation environment.

Frames used in this file
------------------------
* world        : Isaac Sim world frame.
* start frame  : frame of the robot at the beginning of the episode (origin = spawn position, x = spawn heading).
                 The persistent map, the goal and the position histories live in this frame.
* true pose    : ground-truth pose in the start frame (``_true_pose_s``). Used ONLY for supervision, for the
                 teacher observations and to emulate sensors (never fed to the student).
* est pose     : ``self.pose_est`` = [x, y, z, yaw] in the start frame, produced by dead-reckoning the
                 odometry network's body velocities and nudged by the scan-to-map correction.
* body frame   : yaw-aligned frame attached to the robot (x forward, y left).
* ego crop     : map window centred on the robot and rotated by its yaw (what the planner sees).

New perception pipeline (per control step)
------------------------------------------
1. odom_model predicts [roll, pitch, v_body(3), w_body(3)] from delayed proprioception.
2. Dead-reckoning outside the network gives ``pose_est`` (and ``pose_dr`` = same without corrections, for metrics).
3. The height scanner's ray hits are expressed in the body frame (ideal sensor), delayed, and placed in the
   start-frame map with the pose that was valid at capture time.
4. Optional scan-to-map matching nudges ``pose_est`` (complementary-filter style loop closure).
5. A per-cell Kalman filter fuses the placed points into ``raw_map`` / ``_map_variance``.
6. An ego-centric crop of the fused map is denoised/completed by ``EgoMapDenoiser`` and given to the planner.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

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
from .networks.cnn_rnn_model import CNNRNNSeqModel  # noqa: F401  (kept: may be needed for model registration)
from rsl_rl.models.rnn_model import RNNModel


######################################################
# SMALL HELPER MODULES
######################################################


class EgoMapDenoiser(nn.Module):
    """Spatial denoiser / completer working on an ego-centric crop of the fused map.

    Input  [B, 4, n, n]: (height relative to the robot * observed, normalised log-variance, observed mask,
                          1/(1+var) confidence).
    Output [B, 2, n, n]: (height residual, log-variance of the residual). The environment builds
                          ``pred_height = input_height + residual`` (residual head, zero-initialised, so the
                          network starts as the identity on the Kalman estimate).

    U-Net with two stride-2 downsamplings, a dilated bottleneck (dilation 2 and 4; the receptive field covers
    roughly the whole 64x64 crop) and GroupNorm. No recurrence, no odometry, no geometry.
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32, groups: int = 8):
        super().__init__()

        def block(cin, cout, stride=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation),
                nn.GroupNorm(min(groups, cout), cout),
                nn.ELU(),
            )

        b = base_channels
        self.enc1 = block(in_channels, b)
        self.enc2 = block(b, 2 * b, stride=2)
        self.enc3 = block(2 * b, 4 * b, stride=2)
        self.bottleneck = nn.Sequential(block(4 * b, 4 * b, dilation=2), block(4 * b, 4 * b, dilation=4))
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = block(4 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = block(2 * b, b)
        self.head = nn.Conv2d(b, 2, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([0.0, math.log(0.05**2)]))

    @staticmethod
    def _match(up: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != ref.shape[-2:]:
            up = F.interpolate(up, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = self.dec2(torch.cat([self._match(self.up2(b), e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1))
        return self.head(d1)


class _RunningNorm:
    """Running mean/variance normaliser (Chan et al. parallel update). Used for the odometry inputs."""

    def __init__(self, dim: int, device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_n = x.shape[0]
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10.0, 10.0)

    def state_dict(self) -> dict:
        return {"mean": self.mean.clone(), "var": self.var.clone(), "count": float(self.count)}

    def load_state_dict(self, sd: dict) -> None:
        self.mean = sd["mean"].to(self.mean.device)
        self.var = sd["var"].to(self.var.device)
        self.count = float(sd["count"])


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
        self._load_perception_params()

        self.tick_count_env: torch.Tensor = torch.zeros(N, device=self.device)
        self._step_count = 0
        self.period_hist_short_term = max(1, int(round(1 / self.cfg.freq_pos_short_term)))
        self.period_hist_long_term = max(1, int(round(1 / self.cfg.freq_pos_long_term)))
        self.all_env_ids: torch.Tensor = torch.arange(N, device=self.device)

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

        self._undesired_contact_body_ids_sensor = (
            self._thigh_ids_sensor + self._hip_ids_sensor + self._base_id_sensor
        )
        self._body_contact_info_teacher_sensor = (
            self._base_id_sensor + self._thigh_ids_sensor + self._calf_ids_sensor
        )

        MAZE_REGISTRY.set_dims(
            self.cfg.NUM_ROWS,
            self.cfg.NUM_COLS,
            self.cfg.MAX_MAZE_ROWS,
            self.cfg.MAX_MAZE_COLS,
            self.device,
        )
        self.maze_registery = MAZE_REGISTRY
        self._build_maze_path_tables()

        # ---------------- odometry state ----------------
        # network output: [roll, pitch, vx_b, vy_b, vz_b, wx_b, wy_b, wz_b]
        self.odom_out: torch.Tensor = torch.zeros(N, 8, device=self.device)
        self._prev_vel_b = torch.zeros(N, 3, device=self.device)
        self._prev_yaw_rate = torch.zeros(N, device=self.device)
        # estimated pose in the start frame: [x, y, z, yaw]
        self.pose_est: torch.Tensor = torch.zeros(N, 4, device=self.device)
        # dead-reckoning only (no scan-to-map correction) -> metrics
        self.pose_dr: torch.Tensor = torch.zeros(N, 4, device=self.device)
        self._drift_dist = torch.zeros(N, device=self.device)  # distance travelled since last accepted correction
        self._gyro_bias = torch.zeros(N, 3, device=self.device)
        self.odom_obs_proprio_hist = torch.zeros(N, int(self.cfg.length_odom_hist), 45, device=self.device)

        self.cmd_vel: torch.Tensor = torch.zeros(N, 3, device=self.device)
        self.prev_cmd_vel: torch.Tensor = torch.zeros_like(self.cmd_vel)
        self._num_joints = int(self._robot.data.default_joint_pos.shape[-1])
        self.loc_actions: torch.Tensor = torch.zeros(N, self._num_joints, device=self.device)
        self.prev_loc_actions: torch.Tensor = torch.zeros_like(self.loc_actions)

        self.pred_pos_hist_short_term = torch.zeros(N, int(self.cfg.length_short_term), 3, device=self.device)
        self.pred_pos_hist_long_term = torch.zeros(N, int(self.cfg.length_long_term), 3, device=self.device)

        # nominal base height above ground (tile-relative z of the base at spawn)
        self._base_z0 = self._robot.data.default_root_state[:, 2].clone()

        # ---------------- delay buffers ----------------
        self._proprio_lag = self._sample_lag(N)
        self._height_lag = self._sample_lag(N)  # shared by every height-sensor buffer and by the pose history
        if self.cfg.delay:
            self._proprio_delay_buffer = DelayBuffer(
                history_length=self.cfg.delay_length, batch_size=N, device=self.device
            )
            self._loc_heightmap_delay_buffer = DelayBuffer(
                history_length=self.cfg.delay_length, batch_size=N, device=self.device
            )
            self._nav_heightmap_delay_buffer = DelayBuffer(
                history_length=self.cfg.delay_length, batch_size=N, device=self.device
            )
            self._map_scan_delay_buffer = DelayBuffer(
                history_length=self.cfg.delay_length, batch_size=N, device=self.device
            )
            self._proprio_delay_buffer.set_time_lag(self._proprio_lag.clone())
            for buf in (
                self._loc_heightmap_delay_buffer,
                self._nav_heightmap_delay_buffer,
                self._map_scan_delay_buffer,
            ):
                buf.set_time_lag(self._height_lag.clone())

        # pose histories (index 0 = latest) so that a delayed scan is placed with the pose of its capture time
        self._hist_len = int(self.cfg.delay_length) + 1 if self.cfg.delay else 1
        self._pose_est_hist = torch.zeros(N, self._hist_len, 4, device=self.device)
        self._pose_true_hist = torch.zeros(N, self._hist_len, 4, device=self.device)

        self._init_goals_and_starts()
        self._finite_warn_counter = 0

        # ---------------- learned models ----------------
        self.locomotion_policy = self._load_locomotion_policy()
        self._train_perception = bool(self.pp.train_perception)
        self.odom_model = self._load_odom_model()
        self.odom_norm = _RunningNorm(45 * int(self.cfg.length_odom_hist), self.device)
        self.odom_optimizer = torch.optim.Adam(self.odom_model.parameters(), lr=float(self.pp.odom_lr))
        self.map_model = self._load_map_model()
        self.map_optimizer = torch.optim.Adam(self.map_model.parameters(), lr=float(self.pp.map_lr))
        self._maybe_load_perception(self.pp.perception_checkpoint)

        self.terrain_success_rate = torch.zeros(
            self.cfg.terrain.terrain_generator.num_rows,
            self.cfg.terrain.terrain_generator.num_cols,
            device=self.device,
        )
        self.terrain_episode_count = torch.zeros_like(self.terrain_success_rate)
        self._stall_counter = torch.zeros(N, device=self.device)

        # ---------------- persistent start-frame map ----------------
        self._cs = float(self.cfg.map_cell_size)
        self._Wc = int(round(float(self.cfg.map_width) / self._cs))
        self._Hc = int(round(float(self.cfg.map_height) / self._cs))
        self._Wm = self._Wc * self._cs
        self._Hm = self._Hc * self._cs
        self._WH = self._Wc * self._Hc
        self._crop_n = int(self.pp.map_crop_cells)
        self._map_meas_var = float(self.pp.map_meas_std) ** 2

        self.raw_map = torch.zeros(N, self._Wc, self._Hc, device=self.device)  # Kalman height, tile-relative z
        self._map_variance = torch.full_like(self.raw_map, 1.0e6)  # Kalman variance
        self.true_map = torch.zeros_like(self.raw_map)  # simulator heightfield in the start frame
        self.map_seen_gt = torch.zeros(N, self._Wc, self._Hc, dtype=torch.bool, device=self.device)
        self._fuse_sum = torch.zeros(N * self._WH, device=self.device)
        self._fuse_cnt = torch.zeros(N * self._WH, device=self.device)
        self.nav_heightmap = torch.zeros(N, 2, self.nav_x_cells, self.nav_y_cells, device=self.device)
        # planner map observation (ego crop): [height relative to robot, confidence]
        self._student_map_obs = torch.zeros(N, 2, self._crop_n, self._crop_n, device=self.device)

        local_x = (torch.arange(self._Wc, device=self.device) - self._Wc / 2 + 0.5) * self._cs
        local_y = (torch.arange(self._Hc, device=self.device) - self._Hc / 2 + 0.5) * self._cs
        lx, ly = torch.meshgrid(local_x, local_y, indexing="ij")
        self._map_local_pts = torch.stack([lx, ly], dim=-1).reshape(-1, 2)  # (cells, 2), env-independent

        # scan matching offsets (start-frame dx, dy, dyaw)
        sx, sy, sa = int(self.pp.match_xy_steps), int(self.pp.match_xy_steps), int(self.pp.match_yaw_steps)
        assert sx % 2 == 1 and sa % 2 == 1, "match_xy_steps and match_yaw_steps must be odd"
        ox = torch.linspace(-float(self.pp.match_xy_range), float(self.pp.match_xy_range), sx, device=self.device)
        oy = torch.linspace(-float(self.pp.match_xy_range), float(self.pp.match_xy_range), sy, device=self.device)
        yaw_r = math.radians(float(self.pp.match_yaw_range_deg))
        oa = torch.linspace(-yaw_r, yaw_r, sa, device=self.device)
        gx, gy, ga = torch.meshgrid(ox, oy, oa, indexing="ij")
        self._match_offs = torch.stack([gx, gy, ga], dim=-1).reshape(-1, 3)
        self._match_shape = (sx, sy, sa)
        self._match_zero_idx = (sx // 2) * sy * sa + (sy // 2) * sa + sa // 2
        self._match_idx = None
        self._match_idx_R = -1

        print(f"Global map initialized with shape: {tuple(self.raw_map.shape)}, ego crop: {self._crop_n}x{self._crop_n}")
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

    def _load_perception_params(self):
        """Optional config entries (all have defaults, so the cfg file does not need to change)."""
        defaults = dict(
            # models / training
            train_perception=True,  # False -> odom and map models are frozen (inference only)
            odom_lr=1e-3,
            map_lr=1e-3,
            perception_checkpoint=None,  # path written by save_perception_models()
            # odometry
            gyro_bias_max=0.01,  # rad/s, per-episode random gyro bias used by the yaw integrator (needs cfg.randomize)
            # map scan / Kalman map
            map_crop_cells=64,
            map_scan_noise=0.01,  # m, z noise added to the emulated scan (needs cfg.randomize)
            map_scan_max_range=6.0,  # m, ray hits farther than this (in xy) are ignored
            map_meas_std=0.05,  # m, per-hit height standard deviation
            map_min_variance=1e-4,  # variance floor, lets the map keep adapting to new measurements
            map_observed_var_thresh=1.0,  # cell counts as "observed" below this variance (prior is 1e6)
            map_drift_var_gain=0.02,  # m of height std added per metre travelled since the last correction
            map_conf_ref_std=0.1,  # m, std at which the planner confidence equals 0.5
            map_w_complete=0.5,  # loss weight for seen-but-unobserved cells
            map_w_unseen=0.05,  # loss weight for never-seen cells
            # scan-to-map matching
            scan_match=True,
            match_period=2,
            match_xy_range=0.2,
            match_xy_steps=5,
            match_yaw_range_deg=3.0,
            match_yaw_steps=5,
            match_max_points=256,
            match_chunk=128,
            match_var_thresh=0.05,
            match_min_points=40,
            match_err_clip=9.0,
            match_improve_thresh=0.3,
            match_axis_obs_thresh=0.5,
            match_gain=0.5,
            match_gain_z=0.5,
        )
        for key in list(defaults.keys()):
            defaults[key] = getattr(self.cfg, key, defaults[key])
        self.pp = SimpleNamespace(**defaults)

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
        self._offsets = torch.empty(self.num_envs, device=self.device)
        self._rots_yaw.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
        self._rots_roll.uniform_(-self.cfg.max_rot, self.cfg.max_rot)
        self._offsets.uniform_(-self.cfg.max_offset, self.cfg.max_offset)
        self._setup_lidar_values()
        self._create_dropout_distributions()
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

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=10000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._heightfield_ready = False

    def _setup_lidar_values(self):
        self.nav_inv_cell_size = 1.0 / float(self.cfg.nav_cell_size)
        self.nav_x_cells = max(
            1, int((float(self.cfg.nav_x_range[1]) - float(self.cfg.nav_x_range[0])) * self.nav_inv_cell_size)
        )
        self.nav_y_cells = max(
            1, int((float(self.cfg.nav_y_range[1]) - float(self.cfg.nav_y_range[0])) * self.nav_inv_cell_size)
        )
        self.nav_num_cells = self.nav_x_cells * self.nav_y_cells
        self._latest_lidar_obs = torch.zeros(self.num_envs, self.nav_num_cells, device=self.device)

        self.loc_inv_cell_size = 1.0 / float(self.cfg.loc_cell_size)
        self.loc_x_cells = max(
            1, int((float(self.cfg.loc_x_range[1]) - float(self.cfg.loc_x_range[0])) * self.loc_inv_cell_size)
        )
        self.loc_y_cells = max(
            1, int((float(self.cfg.loc_y_range[1]) - float(self.cfg.loc_y_range[0])) * self.loc_inv_cell_size)
        )
        self.loc_num_cells = self.loc_x_cells * self.loc_y_cells

    def _make_gaussian_prob(self, h: int, w: int) -> torch.Tensor:
        """Same weighting as before (kept on purpose), but computed per grid and without float overflow."""
        y = torch.arange(h, device=self.device, dtype=torch.float32)
        x = torch.arange(w, device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        expo = ((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.cfg.sigma**2)
        prob = torch.exp(expo - expo.max())
        return (prob / prob.sum()).flatten()

    def _create_dropout_distributions(self):
        """Cell-dropout distributions, one per grid (the nav grid has a different size than the loc grid)."""
        self._drop_prob_loc = self._make_gaussian_prob(self.loc_x_cells, self.loc_y_cells)
        self._drop_prob_nav = self._make_gaussian_prob(self.nav_x_cells, self.nav_y_cells)
        self._n_zeros_loc = int(self.cfg.n_zeros)
        self._n_zeros_nav = max(1, int(round(self.cfg.n_zeros * self.nav_num_cells / max(1, self.loc_num_cells))))
        self._resample_dropout_indices()
        self.same_zeros_count = 0
        self.reset_zeros_freq = int(torch.randint(1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device).item())

    def _resample_dropout_indices(self):
        self.sampled_indices_loc = torch.multinomial(self._drop_prob_loc, self._n_zeros_loc, replacement=True)
        self.sampled_indices_nav = torch.multinomial(self._drop_prob_nav, self._n_zeros_nav, replacement=True)

    #################################################################
    # PERCEPTION CHECKPOINTS
    #################################################################

    def save_perception_models(self, path: str) -> None:
        """Save odom model, odom input normaliser and map model (to pretrain once, then freeze)."""
        torch.save(
            {
                "odom": self.odom_model.state_dict(),
                "odom_norm": self.odom_norm.state_dict(),
                "map": self.map_model.state_dict(),
            },
            os.path.expanduser(path),
        )

    def _maybe_load_perception(self, path) -> None:
        if not path:
            return
        ckpt = torch.load(os.path.expanduser(path), map_location=self.device)
        self.odom_model.load_state_dict(ckpt["odom"])
        self.odom_norm.load_state_dict(ckpt["odom_norm"])
        self.map_model.load_state_dict(ckpt["map"])
        print(f"[Go2NavEnv] Loaded perception checkpoint from {path} (train_perception={self._train_perception})")

    #################################################################
    # POSE HISTORY / FRAMES
    #################################################################

    def _true_pose_s(self) -> torch.Tensor:
        """Ground-truth [x, y, z, yaw] in the start frame."""
        delta_w = self._robot.data.root_pos_w - self.start_pos_w
        c = torch.cos(self.start_yaw_w)
        s = torch.sin(self.start_yaw_w)
        x_s = c * delta_w[:, 0] + s * delta_w[:, 1]
        y_s = -s * delta_w[:, 0] + c * delta_w[:, 1]
        z_s = delta_w[:, 2]
        yaw = self._wrap_to_pi(self._quat_to_yaw(self._robot.data.root_quat_w) - self.start_yaw_w)
        return torch.stack([x_s, y_s, z_s, yaw], dim=-1)

    def _push_pose_hists(self) -> None:
        self._pose_est_hist = torch.roll(self._pose_est_hist, 1, dims=1)
        self._pose_est_hist[:, 0] = self.pose_est
        self._pose_true_hist = torch.roll(self._pose_true_hist, 1, dims=1)
        self._pose_true_hist[:, 0] = self._true_pose_s()

    def _gather_hist(self, hist: torch.Tensor) -> torch.Tensor:
        """Retrive the pos delayed frome time_lag(set at reset) from hist"""
        if not self.cfg.delay:
            return hist[:, 0]
        return hist[self.all_env_ids, self._height_lag]

    @staticmethod
    def _integrate_pose(pose, roll, pitch, v_b, yaw_rate, dt) -> torch.Tensor:
        """Dead-reckon [x, y, z, yaw] from body velocity, roll/pitch and yaw rate (R = Rz(yaw) Ry(pitch) Rx(roll))."""
        yaw = Go2NavEnv._wrap_to_pi(pose[:, 3] + yaw_rate * dt)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cr, sr = torch.cos(roll), torch.sin(roll)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        vx, vy, vz = v_b[:, 0], v_b[:, 1], v_b[:, 2]
        vwx = cy * cp * vx + (cy * sp * sr - sy * cr) * vy + (cy * sp * cr + sy * sr) * vz
        vwy = sy * cp * vx + (sy * sp * sr + cy * cr) * vy + (sy * sp * cr - cy * sr) * vz
        vwz = -sp * vx + cp * sr * vy + cp * cr * vz
        return torch.stack([pose[:, 0] + vwx * dt, pose[:, 1] + vwy * dt, pose[:, 2] + vwz * dt, yaw], dim=-1)

    def _dead_reckon(self) -> None:
        """Integrate the odometry network output (outside the network) into pose_est / pose_dr."""
        pred = self.odom_out
        roll, pitch = pred[:, 0], pred[:, 1]
        vel_b = pred[:, 2:5]
        gyro = self.proprio_delayed[:, 0:3] + self._gyro_bias  # delayed, noisy, biased gyro (body frame)
        cp = torch.cos(pitch).clamp(min=0.2)
        yaw_rate = (torch.sin(roll) * gyro[:, 1] + torch.cos(roll) * gyro[:, 2]) / cp
        # trapezoidal rule: the displacement over the last interval uses the mean of the previous and current estimate
        v_mid = 0.5 * (self._prev_vel_b + vel_b)
        w_mid = 0.5 * (self._prev_yaw_rate + yaw_rate)
        dt = self.step_dt
        prev_xy = self.pose_est[:, :2].clone()
        self.pose_est.copy_(self._integrate_pose(self.pose_est, roll, pitch, v_mid, w_mid, dt))
        self.pose_dr.copy_(self._integrate_pose(self.pose_dr, roll, pitch, v_mid, w_mid, dt))
        self._drift_dist += torch.linalg.norm(self.pose_est[:, :2] - prev_xy, dim=-1)
        self._prev_vel_b.copy_(vel_b)
        self._prev_yaw_rate.copy_(yaw_rate)

    #################################################################
    # MAP MODEL UTILS
    #################################################################

    def _load_map_model(self) -> torch.nn.Module:
        model = EgoMapDenoiser(in_channels=4, base_channels=32).to(self.device)
        model.train()
        return model

    def _build_nav_heightmap(self) -> torch.Tensor:
        """Policy-convention navigation grid (student_height_scan). Not used for map placement anymore."""
        height_data, valid = self._compute_height_data_from_cloud(
            randomize=self.cfg.randomize, apply_dropout=self.cfg.randomize, locomotion=False
        )
        height_data = height_data.view(self.num_envs, self.nav_x_cells, self.nav_y_cells).flip(dims=[1])
        valid = valid.view(self.num_envs, self.nav_x_cells, self.nav_y_cells).flip(dims=[1])
        height_data = self._sanitize_tensor(height_data, "height_data_nav", clamp_abs=10.0)
        nav_heightmap = torch.stack([height_data, valid], dim=1)
        if self.cfg.delay:
            nav_heightmap = self._nav_heightmap_delay_buffer.compute(nav_heightmap)
        self.nav_heightmap = nav_heightmap
        return nav_heightmap

    def _build_map_scan(self) -> torch.Tensor:
        """Emulated sensor output for the map: ray hits in the yaw-aligned body frame, [N, R, 4] = (x_b, y_b, z_rel, valid).

        The simulator ray hits are rotated/translated with the TRUE pose only to emulate what a real scanner reports
        in its own frame. Registration into the map is done later with the ESTIMATED pose.
        """
        hits = self._height_scanner.data.ray_hits_w
        root_pos = self._robot.data.root_pos_w
        yaw = self._quat_to_yaw(self._robot.data.root_quat_w)
        c = torch.cos(yaw)[:, None]
        s = torch.sin(yaw)[:, None]
        dx = hits[..., 0] - root_pos[:, None, 0]
        dy = hits[..., 1] - root_pos[:, None, 1]
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        bz = hits[..., 2] - root_pos[:, None, 2]
        valid = torch.isfinite(hits).all(dim=-1) & ((bx * bx + by * by) <= float(self.pp.map_scan_max_range) ** 2)
        if self.cfg.randomize:
            bz = bz + torch.randn_like(bz) * float(self.pp.map_scan_noise)
        scan = torch.stack([bx, by, bz, valid.float()], dim=-1)
        return torch.where(valid[..., None], scan, torch.zeros_like(scan))

    def _scan_to_cells(self, scan: torch.Tensor, pose: torch.Tensor):
        """
        Place body-frame points with ``pose`` ([N,4] start frame). Returns (env_idx, global_flat_cell_idx, height).
        From scan in robot's frame to heightmap in start frame.
        """
        N = scan.shape[0]
        valid = scan[..., 3] > 0.5
        c = torch.cos(pose[:, 3])[:, None]
        s = torch.sin(pose[:, 3])[:, None]
        x = pose[:, 0, None] + c * scan[..., 0] - s * scan[..., 1]
        y = pose[:, 1, None] + s * scan[..., 0] + c * scan[..., 1]
        ix = torch.floor((x + 0.5 * self._Wm) / self._cs).long()
        iy = torch.floor((y + 0.5 * self._Hm) / self._cs).long()
        ok = valid & (ix >= 0) & (ix < self._Wc) & (iy >= 0) & (iy < self._Hc)
        h = scan[..., 2] + (pose[:, 2] + self._base_z0)[:, None]  # tile-relative absolute height
        env = self.all_env_ids[:N, None].expand_as(ok)
        gflat = env * self._WH + ix * self._Hc + iy
        return env[ok], gflat[ok], h[ok]

    def _fuse(self, env: torch.Tensor, gflat: torch.Tensor, meas: torch.Tensor) -> None:
        """Vectorised per-cell Kalman update. Hits falling in the same cell are averaged first."""
        if gflat.numel() == 0:
            return
        self._fuse_sum.scatter_add_(0, gflat, meas)
        self._fuse_cnt.scatter_add_(0, gflat, torch.ones_like(meas))
        cnt = self._fuse_cnt[gflat]
        avg = self._fuse_sum[gflat] / cnt
        prior_var = self._map_variance.view(-1)[gflat]
        drift = self._drift_dist[env]
        meas_var = (self._map_meas_var + (float(self.pp.map_drift_var_gain) * drift) ** 2) / cnt
        gain = prior_var / (prior_var + meas_var)
        prior_mean = self.raw_map.view(-1)[gflat]
        # duplicated indices carry identical values, so the (non-deterministic) write order is irrelevant
        self.raw_map.view(-1)[gflat] = prior_mean + gain * (avg - prior_mean)
        self._map_variance.view(-1)[gflat] = ((1.0 - gain) * prior_var).clamp_min(float(self.pp.map_min_variance))
        self._fuse_sum[gflat] = 0.0
        self._fuse_cnt[gflat] = 0.0

    # ----- scan-to-map matching ---------------------------------------------------------------

    def _get_match_idx(self, num_rays: int) -> torch.Tensor:
        if self._match_idx is None or self._match_idx_R != num_rays:
            m = min(num_rays, int(self.pp.match_max_points))
            self._match_idx = torch.linspace(0, num_rays - 1, m, device=self.device).round().long()
            self._match_idx_R = num_rays
        return self._match_idx

    @torch.no_grad()
    def _scan_match_chunk(self, scan: torch.Tensor, pose: torch.Tensor, ids: torch.Tensor):
        """Score K candidate offsets of the scan against the already-known map cells (chi-square-like score).
        
        If dx, dy, dz, dyaw make a more consistent map, they are applied as a transformation

        scan [n, M, 4] body frame, pose [n, 4] (delayed est pose), ids [n] env ids.
        Returns correction [n, 4] = (dx, dy, dz, dyaw) in the start frame and a bool mask "xy/yaw accepted".
        """
        pp = self.pp
        n, M, _ = scan.shape
        offs = self._match_offs
        K = offs.shape[0]
        yaw = pose[:, 3, None] + offs[None, :, 2]  # [n,K]
        c = torch.cos(yaw)[..., None]
        s = torch.sin(yaw)[..., None]
        bx = scan[:, None, :, 0]
        by = scan[:, None, :, 1]
        x = pose[:, 0, None, None] + offs[None, :, 0, None] + c * bx - s * by  # [n,K,M]
        y = pose[:, 1, None, None] + offs[None, :, 1, None] + s * bx + c * by
        ix = torch.floor((x + 0.5 * self._Wm) / self._cs).long()
        iy = torch.floor((y + 0.5 * self._Hm) / self._cs).long()
        inb = (ix >= 0) & (ix < self._Wc) & (iy >= 0) & (iy < self._Hc)
        gidx = (ids[:, None, None] * self._WH + ix.clamp(0, self._Wc - 1) * self._Hc + iy.clamp(0, self._Hc - 1)).reshape(-1)
        m_h = self.raw_map.view(-1)[gidx].view(n, K, M)
        m_v = self._map_variance.view(-1)[gidx].view(n, K, M)
        z = scan[:, None, :, 2] + (pose[:, 2] + self._base_z0[ids])[:, None, None]
        valid = scan[:, None, :, 3] > 0.5
        known = inb & valid & (m_v < float(pp.match_var_thresh))

        w = known.float() / (m_v + self._map_meas_var)
        mu = (w * (z - m_h)).sum(-1) / w.sum(-1).clamp(min=1e-9)  # per-candidate mean height error (z drift)
        resid = z - m_h - mu[..., None]  # z is decoupled from the xy/yaw search
        clip = float(pp.match_err_clip)
        err = (resid**2 / (m_v + self._map_meas_var)).clamp(max=clip)
        nk = known.sum(-1)  # [n,K]
        score = (err * known).sum(-1) / nk.clamp(min=1)
        score = torch.where(nk >= int(pp.match_min_points), score, torch.full_like(score, clip))

        sx, sy, sa = self._match_shape
        S = score.reshape(n, sx, sy, sa)
        best_score, best = score.min(dim=1)
        zero_score = score[:, self._match_zero_idx]
        nk_best = nk.gather(1, best[:, None]).squeeze(1)

        # per-axis observability: the score profile along an axis (min over the others) must not be flat
        thr = float(pp.match_axis_obs_thresh)
        prof_x = S.amin(dim=(2, 3))
        prof_y = S.amin(dim=(1, 3))
        prof_a = S.amin(dim=(1, 2))
        obs_x = (prof_x.amax(-1) - prof_x.amin(-1)) > thr
        obs_y = (prof_y.amax(-1) - prof_y.amin(-1)) > thr
        obs_a = (prof_a.amax(-1) - prof_a.amin(-1)) > thr

        good = (
            (nk_best >= int(pp.match_min_points))
            & (best_score < clip * 0.999)
            & ((zero_score - best_score) > float(pp.match_improve_thresh))
        )
        off = offs[best]  # [n,3]
        gain = float(pp.match_gain)
        dx = gain * off[:, 0] * obs_x.float() * good.float()
        dy = gain * off[:, 1] * obs_y.float() * good.float()
        dyaw = gain * off[:, 2] * obs_a.float() * good.float()

        # z: take the mean height error at the chosen candidate if the xy match is trusted, else at zero offset
        mu_best = mu.gather(1, best[:, None]).squeeze(1)
        mu_zero = mu[:, self._match_zero_idx]
        nk_zero = nk[:, self._match_zero_idx]
        mu_sel = torch.where(good, mu_best, mu_zero)
        z_ok = torch.where(good, nk_best, nk_zero) >= int(pp.match_min_points)
        dz = -float(pp.match_gain_z) * mu_sel * z_ok.float()

        corr = torch.stack([dx, dy, dz, dyaw], dim=-1)
        accepted = good & (obs_x | obs_y | obs_a)
        return corr, accepted

    @torch.no_grad()
    def _scan_match_all(self, scan: torch.Tensor, pose: torch.Tensor):
        N = self.num_envs
        sub = scan[:, self._get_match_idx(scan.shape[1])]
        corr = torch.zeros(N, 4, device=self.device)
        accepted = torch.zeros(N, dtype=torch.bool, device=self.device)
        chunk = int(self.pp.match_chunk)
        for start in range(0, N, chunk):
            ids = self.all_env_ids[start : start + chunk]
            c, a = self._scan_match_chunk(sub[ids], pose[ids], ids)
            corr[ids] = c
            accepted[ids] = a
        return corr, accepted

    @torch.no_grad()
    def _apply_pose_correction(self, corr: torch.Tensor, accepted: torch.Tensor) -> None:
        """Apply the same rigid nudge to the estimate, its history buffer and the position histories."""
        self.pose_est += corr
        self.pose_est[:, 3] = self._wrap_to_pi(self.pose_est[:, 3])
        self._pose_est_hist += corr[:, None, :]
        self._pose_est_hist[..., 3] = self._wrap_to_pi(self._pose_est_hist[..., 3])
        self.pred_pos_hist_short_term += corr[:, None, 0:3]
        self.pred_pos_hist_long_term += corr[:, None, 0:3]
        self._drift_dist = torch.where(accepted, torch.zeros_like(self._drift_dist), self._drift_dist)

    # ----- ego crops --------------------------------------------------------------------------

    def _ego_grid(self, pose: torch.Tensor) -> torch.Tensor:
        """
        builds a normalized sampling grid for a square local map window centered at the robot, rotated according to its yaw
        """
        n = self._crop_n
        u = (torch.arange(n, device=self.device) - n / 2 + 0.5) * self._cs
        uu, vv = torch.meshgrid(u, u, indexing="ij")
        c = torch.cos(pose[:, 3])[:, None, None]
        s = torch.sin(pose[:, 3])[:, None, None]
        x = pose[:, 0, None, None] + c * uu - s * vv
        y = pose[:, 1, None, None] + s * uu + c * vv
        # maps are [B, C, Wc(x), Hc(y)] -> grid[..., 0] indexes the last dim (y), grid[..., 1] the x dim
        return torch.stack([y / (0.5 * self._Hm), x / (0.5 * self._Wm)], dim=-1)

    @staticmethod
    def _crop(maps: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(maps, grid, mode="nearest", padding_mode="zeros", align_corners=False)

    @torch.no_grad()
    def _compute_crops(self, ids: torch.Tensor | None = None) -> dict:
        """Denoiser input (estimated pose) and training target / masks (true pose) for the given envs.
        Returns {x, observed, seen, inb, target}
        x: input for the model
        observed: a bool mask of the crop being 1 if var < threshold and 0 otherwise
        seen: same mask but ground truth info (around the ground truth position)
        in: 
        """

        def sel(t):
            return t if ids is None else t[ids]

        est = sel(self.pose_est)
        tru = sel(self._true_pose_s())
        z0 = sel(self._base_z0)
        g_est = self._ego_grid(est)
        g_true = self._ego_grid(tru)

        h = self._crop(sel(self.raw_map).unsqueeze(1), g_est)[:, 0]
        var = self._crop(sel(self._map_variance).unsqueeze(1), g_est)[:, 0]
        var = torch.where(var > 0.0, var, torch.full_like(var, 1.0e6))  # zero padding -> unobserved
        observed = var < float(self.pp.map_observed_var_thresh)

        z_est = (est[:, 2] + z0)[:, None, None]
        rel = torch.where(observed, h, torch.zeros_like(h)) - z_est  # unobserved cells: flat-ground prior
        logv = (torch.log(var.clamp(1e-6, 1e6)).clamp(-8.0, 14.0) + 5.0) / 5.0
        conf_in = 1.0 / (1.0 + var)
        x = torch.stack([rel, logv, observed.float(), conf_in], dim=1)

        z_true = (tru[:, 2] + z0)[:, None, None]
        target = self._crop(sel(self.true_map).unsqueeze(1), g_true)[:, 0] - z_true
        seen = self._crop(sel(self.map_seen_gt).float().unsqueeze(1), g_true)[:, 0] > 0.5
        inb = (g_true.abs() <= 1.0).all(dim=-1)
        return {"x": x, "observed": observed, "seen": seen, "inb": inb, "target": target}

    def _map_losses(self, pred_h, logvar, crops):
        obs, seen, inb, target = crops["observed"], crops["seen"], crops["inb"], crops["target"]
        pp = self.pp
        w = obs.float() + (seen & ~obs).float() * float(pp.map_w_complete) + (~seen & ~obs).float() * float(pp.map_w_unseen)
        w = w * inb.float()
        err = pred_h - target
        l1 = F.smooth_l1_loss(pred_h, target, reduction="none", beta=0.05)
        loss_h = (w * l1).sum() / w.sum().clamp(min=1.0)

        m = inb & (obs | seen)
        mx = (m[:, 1:, :] & m[:, :-1, :]).float()
        my = (m[:, :, 1:] & m[:, :, :-1]).float()
        ex = ((pred_h[:, 1:, :] - pred_h[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :])).abs()
        ey = ((pred_h[:, :, 1:] - pred_h[:, :, :-1]) - (target[:, :, 1:] - target[:, :, :-1])).abs()
        loss_e = 0.5 * ((mx * ex).sum() / mx.sum().clamp(min=1.0) + (my * ey).sum() / my.sum().clamp(min=1.0))

        mm = m.float()
        nll = 0.5 * (logvar + err.detach() ** 2 * torch.exp(-logvar))
        loss_n = (mm * nll).sum() / mm.sum().clamp(min=1.0)
        loss = loss_h + 0.5 * loss_e + 0.1 * loss_n
        return loss, loss_h, loss_e, loss_n

    def _map_step(self, crops: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run (and optionally train) the denoiser. Returns (pred height relative to the robot, planner confidence)."""
        if not self._train_perception:
            with torch.no_grad():
                self.map_model.eval()
                out = self.map_model(crops["x"])
                pred_h = crops["x"][:, 0] + out[:, 0]
                logvar = out[:, 1].clamp(-10.0, 4.0)
        else:
            with torch.inference_mode(False), torch.enable_grad():
                self.map_model.train()
                c = {k: v.detach().clone() for k, v in crops.items()}
                out = self.map_model(c["x"])
                pred_h = c["x"][:, 0] + out[:, 0]
                logvar = out[:, 1].clamp(-10.0, 4.0)
                loss, loss_h, loss_e, loss_n = self._map_losses(pred_h, logvar, c)
                self.map_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.map_model.parameters(), max_norm=1.0)
                self.map_optimizer.step()
                self.map_model.eval()
                pred_h = pred_h.detach()
                logvar = logvar.detach()
            self.extras.setdefault("log", {})
            self.extras["log"].update(
                {
                    "map/loss_total": loss.detach(),
                    "map/loss_height": loss_h.detach(),
                    "map/loss_edge": loss_e.detach(),
                    "map/loss_nll": loss_n.detach(),
                }
            )
        # evaluation metrics (robot-relative heights, so they include z drift)
        with torch.no_grad():
            obs_m = (crops["observed"] & crops["inb"]).float()
            unobs_seen = (~crops["observed"] & crops["seen"] & crops["inb"]).float()
            target = crops["target"]
            raw_err = (crops["x"][:, 0] - target).abs()
            pred_err = (pred_h - target).abs()
            self.extras.setdefault("log", {})
            self.extras["log"].update(
                {
                    "map/raw_mae_observed": (raw_err * obs_m).sum() / obs_m.sum().clamp(min=1.0),
                    "map/pred_mae_observed": (pred_err * obs_m).sum() / obs_m.sum().clamp(min=1.0),
                    "map/pred_mae_seen_unobserved": (pred_err * unobs_seen).sum() / unobs_seen.sum().clamp(min=1.0),
                }
            )
        ref = float(self.pp.map_conf_ref_std)
        conf = 1.0 / (1.0 + torch.exp(logvar) / ref**2)
        return pred_h, conf

    def _update_map_perception(self) -> None:
        """Steps 3-6 of the perception pipeline (see module docstring)."""
        # student navigation grid (policy convention) -- unchanged semantics
        self._build_nav_heightmap()

        # emulated body-frame scan for the map, delayed with the same lag as the other height buffers
        scan = self._build_map_scan()
        if self.cfg.delay:
            scan = self._map_scan_delay_buffer.compute(scan)

        self._push_pose_hists()
        nav_pose = self._gather_hist(self._pose_est_hist)

        # scan-to-map correction (complementary-filter loop closure)
        if bool(self.pp.scan_match) and (self._step_count % max(1, int(self.pp.match_period)) == 0):
            corr, accepted = self._scan_match_all(scan, nav_pose)
            self._apply_pose_correction(corr, accepted)
            nav_pose = self._gather_hist(self._pose_est_hist)
            self.extras.setdefault("log", {})
            self.extras["log"].update(
                {
                    "match/accept_rate": accepted.float().mean(),
                    "match/mean_abs_dxy": corr[:, :2].abs().sum(-1).mean(),
                    "match/mean_abs_dz": corr[:, 2].abs().mean(),
                }
            )

        # Kalman fusion with the estimated capture-time pose
        env, gflat, h = self._scan_to_cells(scan, nav_pose)
        self._fuse(env, gflat, h)

        # ground-truth fog of war (true capture-time pose) -- supervision only
        _, gflat_t, _ = self._scan_to_cells(scan, self._gather_hist(self._pose_true_hist))
        self.map_seen_gt.view(-1)[gflat_t] = True

        # ego crop -> denoiser -> planner observation
        crops = self._compute_crops()
        pred_h, conf = self._map_step(crops)
        self._student_map_obs[:, 0] = pred_h
        self._student_map_obs[:, 1] = conf

    def _update_true_map(self, env_ids: torch.Tensor) -> None:
        """True tile-relative heightfield sampled in the start frame (training target / teacher input)."""
        n = env_ids.numel()
        gx, gy = self._global_heightfield.shape
        res = self._global_heightfield_res
        x_min = self._global_heightfield_x_min
        y_min = self._global_heightfield_y_min

        yaw = self.start_yaw_w[env_ids]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        wx = cos_y[:, None] * self._map_local_pts[None, :, 0] - sin_y[:, None] * self._map_local_pts[None, :, 1]
        wy = sin_y[:, None] * self._map_local_pts[None, :, 0] + cos_y[:, None] * self._map_local_pts[None, :, 1]
        world_x = wx + self.start_pos_w[env_ids, 0:1]
        world_y = wy + self.start_pos_w[env_ids, 1:2]

        gx_norm = (world_x - x_min) / (gx * res) * 2 - 1
        gy_norm = (world_y - y_min) / (gy * res) * 2 - 1
        sample_grid = torch.stack([gy_norm, gx_norm], dim=-1).reshape(n, self._Wc, self._Hc, 2)
        field = self._global_heightfield.unsqueeze(0).unsqueeze(0).expand(n, 1, gx, gy)

        sampled = F.grid_sample(field, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        # tile-relative heights, same convention as the fused map (pose z + nominal base height)
        self.true_map[env_ids] = sampled.squeeze(1) - self._terrain.env_origins[env_ids, 2][:, None, None]

    def _precompute_global_heightfield(self):
        if self._heightfield_ready:
            return
        from isaaclab.utils.warp import raycast_mesh

        prim_path = self._height_scanner.cfg.mesh_prim_paths[0]
        warp_mesh = self._height_scanner.meshes[prim_path]

        gen_cfg = self.cfg.terrain.terrain_generator
        tile_w, tile_l = gen_cfg.size

        # Bounds from ALL terrain tiles (not from the env origins at the first reset), so that curriculum
        # level changes never move an env outside the precomputed heightfield.
        origins = getattr(self._terrain, "terrain_origins", None)
        if origins is None:
            origins = self._terrain.env_origins
        origins = origins.reshape(-1, 3)
        margin = 0.5 * max(tile_w, tile_l) + 2.0
        x_min = origins[:, 0].min().item() - margin
        x_max = origins[:, 0].max().item() + margin
        y_min = origins[:, 1].min().item() - margin
        y_max = origins[:, 1].max().item() + margin

        res = float(self.cfg.map_cell_size)
        gx = max(2, int(math.ceil((x_max - x_min) / res)))
        gy = max(2, int(math.ceil((y_max - y_min) / res)))

        # cell centres, consistent with grid_sample(align_corners=False)
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

        self._global_heightfield = heights.reshape(gx, gy)
        self._global_heightfield_res = res
        self._global_heightfield_x_min = x_min
        self._global_heightfield_y_min = y_min
        self._heightfield_ready = True

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
        # direct Euclidean distance in METERS (before: divided by cell_size, which mixed units when cell_size != 1)
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
    # ODOM MODEL UTILS
    #################################################################

    def _load_odom_model(self):
        in_dim = 45 * int(self.cfg.length_odom_hist)
        obs = TensorDict({"proprio": torch.randn(self.num_envs, in_dim)})
        obs_groups = {"policy": ["proprio"]}
        model = RNNModel(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="policy",
            output_dim=8,  # [roll, pitch, vx_b, vy_b, vz_b, wx_b, wy_b, wz_b]
            hidden_dims=(128, 128),
            activation="elu",
            obs_normalization=False,  # inputs are normalised by self.odom_norm (running statistics)
            rnn_num_layers=1,
            rnn_hidden_dim=64,
            rnn_type="gru",
        )
        model = model.to(self.device)
        model.eval()
        model.reset()
        with torch.no_grad():
            model(TensorDict({"proprio": obs["proprio"].to(self.device)}, batch_size=[self.num_envs]))
        print("Odom model")
        print(model)
        return model

    def _update_odom_obs_history(self, odom_obs: torch.Tensor) -> None:
        """Most recent frame at the end of the buffer."""
        self.odom_obs_proprio_hist = torch.roll(self.odom_obs_proprio_hist, -1, 1)
        self.odom_obs_proprio_hist[:, -1, :] = odom_obs

    def _update_predicted_position_history(self, predicted_xyz: torch.Tensor) -> None:
        """Position histories (estimated start-frame positions); most recent at the end of the buffer."""
        should_update_mid = self.tick_count_env % self.period_hist_short_term == 0
        rolled = torch.roll(self.pred_pos_hist_short_term, shifts=-1, dims=1)
        rolled[:, -1, :] = predicted_xyz
        self.pred_pos_hist_short_term = torch.where(should_update_mid[:, None, None], rolled, self.pred_pos_hist_short_term)

        should_update_long = self.tick_count_env % self.period_hist_long_term == 0
        rolled = torch.roll(self.pred_pos_hist_long_term, shifts=-1, dims=1)
        rolled[:, -1, :] = predicted_xyz
        self.pred_pos_hist_long_term = torch.where(should_update_long[:, None, None], rolled, self.pred_pos_hist_long_term)

    def _get_true_odom_s(self) -> torch.Tensor:
        """Ground-truth [x, y, z, roll, pitch, yaw, v_b(3), w_b(3)] in the spawn frame."""
        root_quat_w = self._robot.data.root_quat_w
        root_lin_vel = self._robot.data.root_lin_vel_b
        root_ang_vel = self._robot.data.root_ang_vel_b
        pose = self._true_pose_s()

        w, x, y, z = root_quat_w.unbind(dim=-1)
        roll_w = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_w = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        rpy_s = torch.stack([roll_w, pitch_w, pose[:, 3]], dim=-1)
        return torch.cat([pose[:, :3], rpy_s, root_lin_vel, root_ang_vel], dim=-1).clone()

    def _odom_step(self, odom_obs: torch.Tensor) -> torch.Tensor:
        """Odometry inference (+ online training when train_perception). Returns [N, 8]."""
        if self._train_perception:
            self.odom_norm.update(odom_obs)
        x = self.odom_norm(odom_obs)

        if not self._train_perception:
            with torch.no_grad():
                self.odom_model.eval()
                pred = self.odom_model(TensorDict({"proprio": x}, batch_size=x.shape[:1]))
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            return pred.to(self.device).detach()

        t = self._get_true_odom_s()
        target = torch.cat([t[:, 3:5], t[:, 6:9], t[:, 9:12]], dim=-1)  # roll, pitch, v_b, w_b
        with torch.inference_mode(False), torch.enable_grad():
            x = x.detach().clone()
            target = target.detach().clone()
            self.odom_model.train()
            pred = self.odom_model(TensorDict({"proprio": x}, batch_size=x.shape[:1]))
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred.to(self.device)

            rp_loss = F.mse_loss(pred[:, 0:2], target[:, 0:2])
            lin_vel_loss = F.mse_loss(pred[:, 2:5], target[:, 2:5])
            ang_vel_loss = F.mse_loss(pred[:, 5:8], target[:, 5:8])
            loss = rp_loss + lin_vel_loss + 0.5 * ang_vel_loss

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
                "odom/loss_roll_pitch": rp_loss.detach(),
                "odom/loss_lin_vel": lin_vel_loss.detach(),
                "odom/loss_ang_vel": ang_vel_loss.detach(),
            }
        )
        return pred.detach()

    def _build_odom_observations(self) -> torch.Tensor:
        # per-frame obs shape = 30 (delayed proprio) + 3 (cmd) + 12 (loc action) = 45
        odom_obs = torch.cat([self.proprio_delayed.clone(), self.cmd_vel, self.loc_actions], dim=-1)
        self._update_odom_obs_history(odom_obs)
        return self.odom_obs_proprio_hist.reshape(self.num_envs, -1)

    def _log_pose_metrics(self) -> None:
        true = self._true_pose_s()
        err = torch.linalg.norm(self.pose_est[:, :2] - true[:, :2], dim=-1)
        err_dr = torch.linalg.norm(self.pose_dr[:, :2] - true[:, :2], dim=-1)
        yaw_err = self._wrap_to_pi(self.pose_est[:, 3] - true[:, 3]).abs()
        yaw_err_dr = self._wrap_to_pi(self.pose_dr[:, 3] - true[:, 3]).abs()
        self.extras.setdefault("log", {})
        self.extras["log"].update(
            {
                "odom/ate_xy_corrected": err.mean(),
                "odom/ate_xy_dead_reckoning": err_dr.mean(),
                "odom/yaw_err_corrected": yaw_err.mean(),
                "odom/yaw_err_dead_reckoning": yaw_err_dr.mean(),
                "odom/z_err_corrected": (self.pose_est[:, 2] - true[:, 2]).abs().mean(),
            }
        )

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

    def _query_locomotion_policy(self, locomotion_obs: torch.Tensor, height_data_loc: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            actions = self.locomotion_policy(locomotion_obs, [height_data_loc])
        if isinstance(actions, (tuple, list)):
            actions = actions[0]
        return actions.to(self.device)

    def _build_locomotion_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_ang_vel = self._robot.data.root_ang_vel_b
        projected_gravity = self._robot.data.projected_gravity_b
        joint_pos_rel = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_vel = self._robot.data.joint_vel

        height_data, valid = self._compute_height_data_from_cloud(
            randomize=self.cfg.randomize, apply_dropout=self.cfg.randomize, locomotion=True
        )
        height_data = height_data.view(self.num_envs, self.loc_x_cells, self.loc_y_cells).flip(dims=[1])
        height_data = self._sanitize_tensor(height_data, "height_data_loc", clamp_abs=10.0)
        valid = valid.view(self.num_envs, self.loc_x_cells, self.loc_y_cells).flip(dims=[1]).float()

        grid = torch.stack([height_data, valid], dim=1)
        if self.cfg.delay:
            grid = self._loc_heightmap_delay_buffer.compute(grid)

        proprio_delayed = torch.cat(
            [
                base_ang_vel + (2.0 * torch.rand_like(base_ang_vel) - 1.0) * float(0.1) * self.cfg.randomize,
                projected_gravity + (2.0 * torch.rand_like(projected_gravity) - 1.0) * float(0.05) * self.cfg.randomize,
                joint_pos_rel + (2.0 * torch.rand_like(joint_pos_rel) - 1.0) * float(0.01) * self.cfg.randomize,
                joint_vel + (2.0 * torch.rand_like(joint_vel) - 1.0) * float(0.1) * self.cfg.randomize,
            ],
            dim=-1,
        )
        proprio_delayed = self._sanitize_tensor(proprio_delayed, "proprio_delayed", clamp_abs=50.0)
        if self.cfg.delay:
            proprio_delayed = self._proprio_delay_buffer.compute(proprio_delayed)
        self.proprio_delayed = proprio_delayed
        proprio_loc = torch.cat([proprio_delayed, self.cmd_vel, self.loc_actions], dim=-1)

        return self._sanitize_tensor(proprio_loc, "proprio_loc", clamp_abs=50.0), grid

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions = [vxd, vyd, wzd]. prev_cmd_vel must be stored BEFORE overwriting cmd_vel (it was always equal before).
        self._ray_cache.clear()
        self.prev_cmd_vel.copy_(self.cmd_vel)
        self.cmd_vel.copy_(actions)

        # 1. Locomotion policy
        proprio_obs_loc, height_data_loc = self._build_locomotion_observations()
        self.prev_loc_actions.copy_(self.loc_actions)
        self.loc_actions.copy_(self._query_locomotion_policy(proprio_obs_loc, height_data_loc))

        # 2. Odometry: network predicts attitude + body velocities, the pose is integrated outside the network
        odom_obs = self._build_odom_observations()
        self.odom_out.copy_(self._odom_step(odom_obs))
        self._dead_reckon()

        # 3. Map: placement with the estimated pose, scan-to-map correction, Kalman fusion, ego-crop denoising
        self._update_map_perception()
        self._update_predicted_position_history(self.pose_est[:, :3])
        self._log_pose_metrics()

        self.tick_count_env += 1
        self._step_count += 1

    def _apply_action(self) -> None:
        low_level_joint_targets = self._robot.data.default_joint_pos + self.cfg.locomotion_action_scale * self.loc_actions
        self._robot.set_joint_position_target(low_level_joint_targets)

    def _rel_hist(self, hist: torch.Tensor) -> torch.Tensor:
        """Express a start-frame position history [N, L, 3] relative to the estimated robot pose (robot frame)."""
        d = hist[..., :2] - self.pose_est[:, None, :2]
        c = torch.cos(self.pose_est[:, 3])[:, None]
        s = torch.sin(self.pose_est[:, 3])[:, None]
        return torch.stack(
            [c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], hist[..., 2] - self.pose_est[:, None, 2]],
            dim=-1,
        )

    def _get_observations(self) -> dict:
        self._ray_cache.clear()
        height_data_teacher, valid_teacher = self._compute_height_data_from_cloud(
            randomize=False, apply_dropout=False, locomotion=False
        )
        height_data_teacher = height_data_teacher.view(self.num_envs, self.nav_x_cells, self.nav_y_cells).flip(dims=[1])
        height_data_teacher = self._sanitize_tensor(height_data_teacher, "height_data_teacher", clamp_abs=10.0)
        valid_teacher = valid_teacher.view(self.num_envs, self.nav_x_cells, self.nav_y_cells).flip(dims=[1])
        height_data_teacher = torch.stack([height_data_teacher, valid_teacher], dim=1)

        # ---- goal expressed in the ESTIMATED robot frame (goal is given in the start frame) ----
        pose = self.pose_est
        c = torch.cos(pose[:, 3])
        s = torch.sin(pose[:, 3])
        d = self._get_goal_pos_s() - pose[:, :2]
        goal_rel_xy = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)
        goal_rel_yaw = self._wrap_to_pi(self._get_goal_yaw_s().squeeze(-1) - pose[:, 3]).unsqueeze(-1)

        pred_pos_hist_short_term = self._rel_hist(self.pred_pos_hist_short_term).reshape(self.num_envs, -1)
        pred_pos_hist_long_term = self._rel_hist(self.pred_pos_hist_long_term).reshape(self.num_envs, -1)
        vel = self.odom_out[:, [2, 3, 7]]  # predicted [vx_b, vy_b, wz_b]

        student_proprio = torch.cat(
            [goal_rel_xy, goal_rel_yaw, pred_pos_hist_short_term, pred_pos_hist_long_term, vel, self.cmd_vel],
            dim=-1,
        )

        true_odom = self._get_true_odom_s()
        goal_delta = self.goal_pos_w - self._robot.data.root_pos_w
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w))
        goal_heading = torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)

        terrain_coords: torch.Tensor = env_ids_to_terrain_coords(self.all_env_ids, self._terrain)
        mazes: torch.Tensor = self.maze_registery.get_mazes_terrain_coords(terrain_coords).clone()
        terrain_type = mazes[:, :, :, 0].long()
        maze_encoded = torch.nn.functional.one_hot(terrain_type, self.maze_registery.num_terrain_types)
        maze_remaining = mazes[:, :, :, 1:]
        maze_encoded_flat = maze_encoded.reshape(self.num_envs, -1)
        maze_remaining_flat = maze_remaining.reshape(self.num_envs, -1)
        maze_features = torch.cat([maze_encoded_flat, maze_remaining_flat], dim=-1)

        teacher_proprio = torch.cat(
            [student_proprio.clone(), true_odom, goal_delta, goal_heading, maze_features], dim=-1
        )

        student_proprio = self._sanitize_tensor(student_proprio, "student_proprio", clamp_abs=100.0)
        teacher_proprio = self._sanitize_tensor(teacher_proprio, "teacher_proprio", clamp_abs=100.0)

        # ---- teacher map: true map, ego crop at the TRUE pose, heights relative to the true base z ----
        true_pose = self._true_pose_s()
        teacher_map = self._crop(self.true_map.unsqueeze(1), self._ego_grid(true_pose))
        teacher_map = teacher_map - (true_pose[:, 2] + self._base_z0)[:, None, None, None]

        return {
            "student_proprio": student_proprio,
            "student_height_scan": self.nav_heightmap.clone(),
            "student_map": self._student_map_obs.clone(),
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
        robot_yaw = self._quat_to_yaw(self._robot.data.root_quat_w)
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - robot_yaw)
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

        # Violent commands penalty (now really active: prev_cmd_vel is the previous command)
        pen_cmd_rate = torch.sum(torch.square(self.cmd_vel - self.prev_cmd_vel), dim=-1)

        # Commands bounds
        # 1. High commands (higher than what the locomotion policy has learned)
        cmd_bounds_high = torch.sum(
            torch.where(torch.abs(self.cmd_vel) >= 1.0, torch.abs(self.cmd_vel), 0.0), dim=-1
        )
        # 2. Low translational command while still far from the goal (was: signed min over [vx, vy, wz], which
        #    penalised every straight-line command because vy = wz = 0)
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
        print(self.max_episode_length)
        print(self.episode_length_buf)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        goal_distance = torch.linalg.norm(self.goal_pos_w[:, :2] - self._robot.data.root_pos_w[:, :2], dim=-1)
        yaw_error = self._wrap_to_pi(self.goal_yaw_w - self._quat_to_yaw(self._robot.data.root_quat_w)).abs()
        self.goal_reached = (goal_distance < self.cfg.goal_reached_distance) & (yaw_error < self.cfg.goal_reached_yaw)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died_base = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id_sensor], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )

        terminated = died_base | self.goal_reached
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        self._precompute_global_heightfield()
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids is not None and len(env_ids) != self.num_envs and self.cfg.plot:
            plot = False
            for env_id in env_ids:
                cond = env_id in self.vis_envs.tolist()
                print(cond)
                if cond:
                    plot = True
            if plot:
                self._plot_map_s(self.vis_envs, save_dir="/mnt/D/dev/robotics/nvidia/isaaclab/go2_nav/map_debug")
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

        odom_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        odom_reset_mask[env_ids_tensor] = True
        with torch.no_grad():
            self.odom_model.reset(odom_reset_mask)

        self.tick_count_env[env_ids_tensor] = 0

        if move_up is not None and move_down is not None:
            self._terrain.update_env_origins(env_ids_tensor, move_up, move_down)
            # Curriculum changes terrain_levels, so environments may now refer to
            # different maze layouts than when the lookup table was built.
            self._build_maze_path_tables(env_ids_tensor)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]

        default_root_state = self._robot.data.default_root_state[env_ids_tensor].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids_tensor]

        # ---- odometry / pose state ----
        self.odom_out[env_ids_tensor] = 0.0
        self._prev_vel_b[env_ids_tensor] = 0.0
        self._prev_yaw_rate[env_ids_tensor] = 0.0
        self.pose_est[env_ids_tensor] = 0.0
        self.pose_dr[env_ids_tensor] = 0.0
        self._pose_est_hist[env_ids_tensor] = 0.0
        self._pose_true_hist[env_ids_tensor] = 0.0
        self._drift_dist[env_ids_tensor] = 0.0
        self.odom_obs_proprio_hist[env_ids_tensor] = 0.0
        bias_max = float(self.pp.gyro_bias_max) * float(self.cfg.randomize)
        self._gyro_bias[env_ids_tensor] = (
            (2.0 * torch.rand(env_ids_tensor.numel(), 3, device=self.device) - 1.0) * bias_max
        )
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

        # ---- persistent map: the previous episode's map must not leak into the new one ----
        self.raw_map[env_ids_tensor] = 0.0
        self._map_variance[env_ids_tensor] = 1.0e6
        self.map_seen_gt[env_ids_tensor] = False
        self.nav_heightmap[env_ids_tensor] = 0.0
        self._student_map_obs[env_ids_tensor] = 0.0

        # randomisation of the scan misalignment, per env
        num_resets = env_ids_tensor.numel()
        self._rots_yaw[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.max_rot, self.cfg.max_rot
        )
        self._rots_roll[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.max_rot, self.cfg.max_rot
        )
        self._offsets[env_ids_tensor] = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.max_offset, self.cfg.max_offset
        )

        # ---- delay buffers: only the envs being reset get a new lag (others keep theirs) ----
        if self.cfg.delay:
            ids_list = env_ids_tensor.tolist()
            self._proprio_delay_buffer.reset(ids_list)
            self._nav_heightmap_delay_buffer.reset(ids_list)
            self._loc_heightmap_delay_buffer.reset(ids_list)
            self._map_scan_delay_buffer.reset(ids_list)
            self._proprio_lag[env_ids_tensor] = self._sample_lag(num_resets)
            # height-sensor buffers (nav, loc, map scan) and the pose history share one lag: same physical sensor
            self._height_lag[env_ids_tensor] = self._sample_lag(num_resets)
            self._proprio_delay_buffer.set_time_lag(self._proprio_lag.clone())
            for buf in (
                self._nav_heightmap_delay_buffer,
                self._loc_heightmap_delay_buffer,
                self._map_scan_delay_buffer,
            ):
                buf.set_time_lag(self._height_lag.clone())

        root_state = torch.cat([self.start_pos_w[env_ids], self.start_quat_w[env_ids]], dim=-1)

        self.prev_path_distance[env_ids_tensor] = self._compute_maze_path_distance(self.start_pos_w)[env_ids_tensor]

        self._robot.write_root_pose_to_sim(root_state, env_ids_tensor)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

        # ---- logging (keep the per-step odom/map/match metrics, they are not episodic) ----
        kept = {
            k: v for k, v in self.extras.get("log", {}).items() if k.startswith(("odom/", "map/", "match/"))
        }
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
    # HEIGHT SCAN (policy convention, unchanged for the frozen locomotion policy)
    #################################################################

    def _apply_offset(self, height_map):
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
        angles = torch.deg2rad(self._rots_roll).unsqueeze(-1)  # was self._rots (never defined)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        x_coord = points[..., 0]
        y_coord = points[..., 1]
        z_coord = points[..., 2]
        rotated_y = y_coord * cos_angles - z_coord * sin_angles
        rotated_z = y_coord * sin_angles + z_coord * cos_angles
        return torch.stack((x_coord, rotated_y, rotated_z), dim=-1)

    def _rays_in_lidar_frame(self, randomize: bool) -> torch.Tensor:
        """Ray hits in the lidar frame (policy convention), cached per step (loc, nav and teacher share them)."""
        key = bool(randomize)
        if key in self._ray_cache:
            return self._ray_cache[key]
        data = self._height_scanner.data
        ray_hits_w = data.ray_hits_w
        lidar_pos_w = data.pos_w
        lidar_quat_w = data.quat_w
        num_envs, num_rays, _ = ray_hits_w.shape
        rays_rel_w = -ray_hits_w + lidar_pos_w.unsqueeze(1)
        rays_lidar = quat_apply(
            quat_conjugate(lidar_quat_w).unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
            rays_rel_w.reshape(-1, 3),
        ).reshape(num_envs, num_rays, 3)
        if randomize:
            rays_lidar = self._apply_yaw_rotation(rays_lidar)
            rays_lidar = self._apply_roll_rotation(rays_lidar)
        self._ray_cache[key] = rays_lidar
        return rays_lidar

    def _compute_height_data_from_cloud(self, randomize: bool = False, apply_dropout: bool = True, locomotion: bool = True):
        """Compute flattened heightmap in lidar frame using cfg x/y bounds and cell size."""
        rays_lidar = self._rays_in_lidar_frame(randomize)
        num_envs, num_rays, _ = rays_lidar.shape

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
            return (
                torch.zeros((num_envs, num_cells), device=self.device),
                torch.zeros((num_envs, num_cells), dtype=torch.bool, device=self.device),
            )

        rays_valid = rays_flat[valid]
        env_ids = env_ids[valid]

        x_idx = torch.floor((rays_valid[:, 0] - x_min) * inv_cell_size).long()
        y_idx = torch.floor((rays_valid[:, 1] - y_min) * inv_cell_size).long()
        in_bounds = (x_idx >= 0) & (x_idx < x_cells) & (y_idx >= 0) & (y_idx < y_cells)
        if not torch.any(in_bounds):
            return (
                torch.zeros((num_envs, num_cells), device=self.device),
                torch.zeros((num_envs, num_cells), dtype=torch.bool, device=self.device),
            )

        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        env_ids = env_ids[in_bounds]
        z_vals = rays_valid[in_bounds, 2]

        # Flatten env and cell indexing into one reduce op.
        flat_idx = env_ids * num_cells + x_idx * y_cells + y_idx
        height_map = torch.full((num_envs * num_cells,), -torch.inf, device=self.device)
        height_map.scatter_reduce_(0, flat_idx, z_vals, reduce="amax", include_self=True)
        height_map = height_map.reshape(num_envs, num_cells) + self.cfg.desired_base_height
        hit_mask = torch.isfinite(height_map)  # capture BEFORE the where() destroys it
        height_map = torch.where(hit_mask, -height_map, torch.zeros_like(height_map))
        if randomize:
            height_map = self._apply_offset(height_map)
            height_map += (2.0 * torch.rand_like(height_map) - 1.0) * float(0.01)
        if apply_dropout:
            height_map, dropout_mask = self._zero_heightmap_cells(height_map, locomotion)
            valid = hit_mask & dropout_mask
        else:
            valid = hit_mask
        return height_map, valid

    def _zero_heightmap_cells(self, height_map, locomotion: bool = True):
        self.same_zeros_count += 1
        if self.same_zeros_count == self.reset_zeros_freq:
            self.reset_zeros_freq = int(
                torch.randint(1, self.cfg.max_reset_zeros_freq + 1, (1,), device=self.device).item()
            )
            self.same_zeros_count = 0
            self._resample_dropout_indices()
        indices = self.sampled_indices_loc if locomotion else self.sampled_indices_nav
        height_map_actor = height_map.clone()
        valid = torch.ones_like(height_map, dtype=torch.bool)
        height_map_actor[:, indices] = 0.0
        valid[:, indices] = False
        return height_map_actor, valid

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

    def _update_goals(self, env_ids: torch.Tensor, center: bool = False) -> None:
        """Update goal positions for the given envs.

        If ``center`` is True, the goal is placed at the center of each env's maze. Otherwise, the goal is sampled
        uniformly at random over the actual bounds of that env's maze.

        Per-env maze row count follows the curriculum: terrain levels 0 and 1 both produce a 1-row maze, and
        level k (k >= 2) produces a k-row maze, clamped to `maze_max_rows`. Column count is fixed (`maze_max_cols`).
        """
        terrain_generator = self.cfg.terrain.terrain_generator
        maze_cfg = terrain_generator.sub_terrains["maze"]
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
            goal_local_x = torch.zeros_like(x_extent)
            goal_local_y = torch.zeros_like(y_extent)
        else:
            x_min = -0.5 * x_extent
            x_min = torch.where(maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size)
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size)

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

        self.goal_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self.goal_quat_w[:, 0] = torch.cos(self.goal_yaw_w / 2)  # w
        self.goal_quat_w[:, 1] = 0.0  # x
        self.goal_quat_w[:, 2] = 0.0  # y
        self.goal_quat_w[:, 3] = torch.sin(self.goal_yaw_w / 2)  # z

        if self.cfg.vis:
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
        """Update start positions for the given envs (same sampling rules as `_update_goals`)."""
        terrain_generator = self.cfg.terrain.terrain_generator
        maze_cfg = terrain_generator.sub_terrains["maze"]
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
            start_local_x = torch.zeros_like(x_extent)
            start_local_y = torch.zeros_like(y_extent)
        else:
            x_min = -0.5 * x_extent
            x_min = torch.where(maze_rows_f % 2 == 1, -0.5 * x_extent, -0.5 * x_extent - cell_size)
            x_max = 0.5 * x_extent
            y_min = -0.5 * y_extent
            y_max = torch.where(maze_cols_f % 2 == 1, 0.5 * y_extent, 0.5 * y_extent + cell_size)

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
        self.start_pos_w[env_ids, 2] = env_origins[:, 2] + self._robot.data.default_root_state[env_ids, 2].clone()

        self.start_yaw_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self.start_quat_w[:, 0] = torch.cos(self.start_yaw_w / 2)  # w
        self.start_quat_w[:, 1] = 0.0  # x
        self.start_quat_w[:, 2] = 0.0  # y
        self.start_quat_w[:, 3] = torch.sin(self.start_yaw_w / 2)  # z

        if self.cfg.vis:
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

    #################################################################
    # DEBUG PLOT
    #################################################################

    def _plot_map_s(self, vis_envs: torch.Tensor, save_dir: str = None) -> None:
        """Save a debug image per env in vis_envs under save_dir/env{i}/plot{j}.png.

        Panels: true map (dimmed where never seen) with the true (blue x) and estimated (red +) pose, fused raw map
        (observed cells only), then the ego-centric crops: denoiser input height, denoiser output height, target
        height, planner confidence. Ego crops: vertical axis = forward, horizontal axis = left.
        """
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

            ids = torch.tensor([env_idx], device=self.device)
            crops = self._compute_crops(ids)
            true_pose = self._true_pose_s()[env_idx].detach().cpu().numpy()
            est_pose = self.pose_est[env_idx].detach().cpu().numpy()
            seen = self.map_seen_gt[env_idx].detach().cpu().numpy()
            var = self._map_variance[env_idx].detach().cpu().numpy()
            raw = self.raw_map[env_idx].detach().cpu().numpy()
            observed = var < float(self.pp.map_observed_var_thresh)

            panels = [
                dict(data=self.true_map[env_idx].detach().cpu().numpy(), title="true_map (dim = never seen)",
                     cmap="terrain", dim=~seen, full=True),
                dict(data=np.where(observed, raw, np.nan), title="fused raw_map (observed cells)", cmap="terrain", full=True),
                dict(data=crops["x"][0, 0].cpu().numpy(), title="ego input height (rel. to robot)", cmap="terrain"),
                dict(data=self._student_map_obs[env_idx, 0].detach().cpu().numpy(), title="ego predicted height", cmap="terrain"),
                dict(data=crops["target"][0].cpu().numpy(), title="ego target height", cmap="terrain"),
                dict(data=self._student_map_obs[env_idx, 1].detach().cpu().numpy(), title="planner confidence",
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
                    ax.plot(true_pose[0] / self._cs + self._Wc / 2 - 0.5, true_pose[1] / self._cs + self._Hc / 2 - 0.5,
                            "bx", markersize=10, markeredgewidth=2)
                    ax.plot(est_pose[0] / self._cs + self._Wc / 2 - 0.5, est_pose[1] / self._cs + self._Hc / 2 - 0.5,
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