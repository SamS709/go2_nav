# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg

from dataclasses import MISSING


from go2_nav.maze_terrain_cfg import make_maze_terrain_cfg  # isort: skip

@configclass
class EventCfg:
    """Configuration for randomization."""

     # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class Go2NavEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    sim_freq = 200
    policy_freq = sim_freq / 4
    sim_dt = 1 / sim_freq
    episode_length_s = 30.0
    # Planner output: [x, y, z, vx, vy, vz, wx, wy, wz, cmd_x, cmd_y, cmd_z]
    action_space = 12
    nav_cell_size = 0.2
    nav_x_range = (-0.4, 7.0) # 37
    nav_y_range = (-7.0, 7.0) # 70
    
    loc_cell_size = 0.1
    loc_x_range = [-0.5, 0.8]
    loc_y_range = [-0.5, 0.5]
    
    sigma = 4.00
    n_zeros = 20
    max_reset_zeros_freq = 8
    max_rot = 4.0
    max_offset = 0.05
    
    desired_base_height = 0.28
    
    # positions buffer
    length_mid_term: int = 50
    length_long_term: int = 50
    freq_pos_mid_term: float = 1 # (policy_tick^(⁻1)) 50 recordings every 1 policy tick
    freq_pos_long_term: float = 0.1 # (policy_tick^(⁻1)) 50 recordings every 10 policy_tick  
    interval_pos_mid_term: float = 1 / freq_pos_mid_term
    interval_pos_long_term: float = 1 / freq_pos_long_term
    
    lidar_num_cells = int((nav_x_range[1] - nav_x_range[0]) / nav_cell_size) * int(
        (nav_y_range[1] - nav_y_range[0]) / nav_cell_size
    )
    observation_space = lidar_num_cells + (length_mid_term + length_long_term) * 3
    # Teacher receives student obs + privileged terms.
    teacher_observation_space = observation_space + 17
    # Expose critic state-space as privileged teacher observations for asymmetric PPO.
    state_space = teacher_observation_space

    randomize = False
    # events: EventCfg = EventCfg()
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # terrain
    # terrain configuration to match the goal generator:
    # Ensure that NUM_ROWS == maze_max_rows. This implies at row 1 & 2 the mazes have 1 row, and after it is increasing by 1 each row.
    # For example, if NUM_ROWS == MAZE_WIDTH_SIZE == 10, at row 1 & 2, mazes will have 1 raw, at raw 3, 2 raws, ..., at row 10, 9 rows, thanks to difficulty (keep its range == [0,1])
    # Same for NUM_COLS == maze_max_cols
    
    # n_cols for the maze is constant fixed to maze_height
    NUM_ROWS = 4
    NUM_COLS = 1
    MAX_MAZE_ROWS = 8
    MAX_MAZE_COLS = 4
    cell_size = 2.0
    # y <=> cols & x <=> rows
    # debug: flat ground plane terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground_plane",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        max_init_terrain_level=1,
        terrain_generator=make_maze_terrain_cfg(
            cell_size=cell_size,
            maze_max_cols=MAX_MAZE_COLS,
            maze_max_rows=MAX_MAZE_ROWS,
            terrain_num_rows=NUM_ROWS,
            terrain_num_cols=NUM_COLS,
            p_wall_dest=0.3,
            curriculum=True,
            difficulty_range=(0.0, 1.0),
            wall_height_range=(0.75, 2.0),
            wall_thickness_range=(0.1, 0.5),
            floor_thickness=0.06,
            algorithm="dfs",
            seed=42,
            # stairs_prob_per_m2_range=(1.0, 1.0),
            # boxes_prob_per_m2_range=(0.0, 0.0),
            # rough_prob_per_m2_range=(0.0, 0.0),
            stairs_prob_per_m2_range=(0.02, 0.08),
            boxes_prob_per_m2_range=(0.04, 0.12),
            rough_prob_per_m2_range=(0.10, 0.24),
            stairs_step_height_range=(0.05, 0.08),
            stairs_step_depth_range=(0.18, 0.24),
            stairs_num_steps_range=(2, 15),
            stairs_start_down_prob=0.4,
            boxes_n_boxes_range=(10, 20),
            boxes_h_boxes_range=(0.10, 0.20),
            boxes_patch_size_ratio_range=(0.75, 0.90),
            rough_n_rough_range=(16, 36),
            rough_h_rough_xy_range=(0.4, 0.6),
            rough_h_rough_z_range=(0.01, 0.15),
            rough_patch_size_ratio_range=(0.75, 0.95),
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    # lidar (same mounting used in go2_lidar task)
    # lidar_offset = (0.28945, 0.0, -0.04682)
    # lidar_rotation = (0.13131596830945724, 0.0, 0.9913405653290647, 0.0)
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base/radar",
        update_period=1 / 60,
        offset=RayCasterCfg.OffsetCfg(
            # pos=lidar_offset,
            # rot=lidar_rotation,
        ),
        mesh_prim_paths=["/World"],
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=64, vertical_fov_range=[0.0, 90.0], horizontal_fov_range=[-180, 180], horizontal_res=2.0
        ),
        max_distance=13.0,
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # markers
    
    goal_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarkers",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    
    start_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/startMarkers",
        markers={
            "start": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )
    
    env_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/envMarkers",
        markers={
            "env": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    
    # planner -> locomotion policy interface
    locomotion_policy_path = "policies/policy_cnn_rnn_seq3.pt"
    require_locomotion_policy = True
    locomotion_observation_dim = 195
    locomotion_action_scale = 0.25
    locomotion_cmd_limits = (1.5, 1.0, 1.5)
    
      

    # goal and reward settings
    rew_scale_goal_distance = 3.0
    rew_scale_goal_progress = 7.0
    rew_scale_goal_orientation = 2.0
    rew_scale_time_penalty = -0.05
    rew_scale_goal_bonus = 200.0
    rew_scale_odom_prediction = 0.4
    rew_scale_cmd_bounds = -0.02
    rew_scale_cmd_rate = -0.001
    rew_scale_upright = 0.5
    rew_scale_terminated = -5.0
    rew_scale_undesired_contacts = -10.0

    goal_distance_sigma = 2.0
    goal_orientation_sigma = 0.5
    goal_reached_distance = cell_size / 3.0
    goal_reached_yaw = 0.5
    odom_prediction_scale = 5.0

    # reset and termination settings
    min_base_height = 0.18
    # projected_gravity_b[..., 2] is -1 when upright and increases as the robot tips.
    max_projected_gravity_z = -0.2
    reset_joint_pos_noise = 0.05
    reset_joint_vel_noise = 0.10
    