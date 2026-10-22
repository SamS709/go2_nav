# Go2 Navigation with Reinforcement Learning

This project trains a navigation planner for the Unitree Go2 in Isaac Lab. The planner outputs three velocity commands:

```text
[cmd_x, cmd_y, cmd_z] = [vx, vy, wz]
```

The navigation planner is separated from low-level gait control. A pretrained locomotion policy receives the current robot state, the planner command, and a local height map, then produces joint actions for the Go2.

## System Overview

```text
Go2 simulation and terrain
    |-- proprioception + cmd_vel + local height map
    |       --> locomotion_policy --> joint actions --> joint targets
    |
    |-- proprioception history + cmd_vel + previous odometry
    |       --> odom_model --> predicted odometry
    |                              |
    |                              +--> short/long position histories
    |
    |-- delayed navigation height map
    |       --> persistent start-frame map
    |             (cell placement + Kalman height/confidence fusion)
    |       --> MapDenoiser --> predicted height map + confidence map
    |
    +-- planner observations --> RL planner --> [vx, vy, wz]
```

The planner uses a student/teacher observation setup. The actor or student sees onboard-style observations and the predicted map. The critic or teacher additionally receives privileged simulator information, including true odometry, maze information, and the true map.

`go2_nav_env.py` is a vectorized `DirectRLEnv`. It owns the robot, terrain, sensors, online odometry and map learners, episode-frame transforms, maze path tables, reward terms, and student/teacher observation construction. The planner does not directly control joints: its three velocity commands are passed through the locomotion policy before being applied to the robot.

The map pipeline separates geometry from learning. The environment places navigation-grid measurements into a persistent map using robot pose and exact tensor operations. The learned map model only denoises and completes the resulting two-channel image; it does not receive odometry or raw coordinates.

## Installation

Install Isaac Lab first, then install this extension into the same Python environment:

```bash
python -m pip install -e source/go2_nav
```

The default locomotion checkpoint is `policies/policy_cnn_rnn_seq3.pt`. Use the Isaac Lab Python launcher when Isaac Lab is not the active interpreter.

List the registered task:

```bash
python scripts/list_envs.py
```

The registered task is `Template-Go2-Nav-Direct-v0`.

## Environment Execution

Each call to `step()` follows this sequence in `_pre_physics_step`:

1. The planner action `[vx, vy, wz]` is copied into `cmd_vel`.
2. The locomotion observation builder creates delayed proprioception, command/action features, and a local heightmap. The frozen locomotion policy returns joint actions.
3. The recurrent odometry model consumes delayed proprioception history, commands, locomotion actions, and its previous prediction. It is trained online against simulator state in the episode start frame.
4. The predicted odometry updates the short-term and long-term position histories.
5. The environment marks the area around the robot as revealed, builds the delayed navigation heightmap, places its valid cells into the persistent start-frame map, and trains the map model.
6. The environment returns student and teacher observations. Low-level joint targets are applied and the simulator advances.

The map update runs before `_get_observations`, so the same `self.nav_heightmap` produced for the current control step is used by both map fusion and the student observation.

### Episode reset

`_reset_idx` resets the robot, locomotion hidden state, odometry hidden state, command/action histories, predicted odometry, position histories, stall counters, goals, and start poses. It rebuilds the true start-frame map from the global terrain heightfield, clears the reveal mask, reveals the initial area around the start, and resets/resamples delay buffers for the environments being reset. Curriculum updates can move an environment to another terrain level and refresh its maze path table.

### Planner training modes

The repository provides:

- **Teacher pretraining:** PPO trains a recurrent CNN policy using privileged observations.
- **Distillation:** a student is trained from the privileged teacher.
- **Direct PPO:** the actor uses student observations while the critic uses privileged teacher observations.

Example commands:

```bash
python scripts/rsl_rl/train.py \
  --task Template-Go2-Nav-Direct-v0 \
  --agent rsl_rl_teacher_cfg_entry_point

python scripts/rsl_rl/train.py \
  --task Template-Go2-Nav-Direct-v0 \
  --agent rsl_rl_distillation_cfg_entry_point

python scripts/rsl_rl/train.py \
  --task Template-Go2-Nav-Direct-v0 \
  --agent rsl_rl_cfg_entry_point
```

The planner configurations use CNN-plus-GRU models. The CNN uses convolutional layers with 16 and 32 channels, stride 2, ReLU activations, and global average pooling. The recurrent hidden size is 128 with two GRU layers.

## Planner Observations

The planner observation groups are defined in `rsl_rl_ppo_cfg.py`.

### Student/actor observations

The student receives:

- `student_proprio`: concatenation of:
  - Goal XY position in the spawn frame: 2 values.
  - Goal yaw: 1 value.
  - Short-term predicted position history: `50 x 3` values.
  - Long-term predicted position history: `50 x 3` values.
  - Predicted roll, pitch, and yaw velocity/state slice from odometry: 3 values.
  - Current planner command `cmd_vel`: 3 values.
- `student_height_scan`: the delayed navigation height scan with two channels and shape `[2, 17, 30]` from the `0.2 m` navigation grid. Channel 0 is height and channel 1 is the valid-cell mask.
- `student_map`: two map channels:
  - Channel 0: predicted global height map.
  - Channel 1: predicted map confidence.

The predicted map is produced by the online map model. Confidence is therefore available to the planner instead of being discarded after map prediction.

### Teacher/critic observations

The teacher receives:

- `teacher_proprio`: student proprioception plus privileged terms:
  - True odometry in the spawn frame: position, roll/pitch/yaw, linear velocity, and angular velocity.
  - Goal displacement in world coordinates.
  - Goal heading encoded as sine and cosine.
  - Maze terrain features and wall information.
- `teacher_height_scan`: the non-randomized navigation height scan.
- `teacher_map`: the simulator-derived true global map, provided as one height channel.

The teacher observation is used by the critic or distillation teacher and is not intended to represent deployable sensor input.

## Navigation Heightmap and Persistent Map

The navigation scan is built by `_compute_height_data_from_cloud` and `_build_nav_heightmap`:

1. Ray hits are transformed from world coordinates into the height scanner frame.
2. Hits are rasterized into the configured navigation grid (`x = -0.4..3.0 m`, `y = -3.0..3.0 m`, cell size `0.2 m`).
3. Each grid cell stores its height and whether it contains a valid hit. The grid is flipped along the x axis to match the policy convention.
4. Optional dropout and delay are applied to the student navigation grid. The resulting tensor is stored in `self.nav_heightmap` and reused by both the map path and `student_height_scan`.

The environment maintains per-environment persistent map buffers at the `50 m x 50 m` map resolution:

- `raw_map`: Kalman-fused height estimate.
- `_map_variance`: per-cell uncertainty, initialized to a large prior variance.
- `raw_map_confidence`: bounded confidence derived from the variance as `1 / (1 + variance)`.
- `map_revealed`: fog-of-war mask used for supervision and plotting.
- `true_map`: simulator-derived heightfield sampled in the episode start frame.

For every valid navigation cell, the environment uses the cell center in the robot frame, rotates it by the robot yaw into world coordinates, then rotates it into the episode start frame. Measurements that land in the same global map cell are averaged before a vectorized Kalman update. This means deployment only needs the navigation heightmap and the robot pose/odometry needed to place it; the map network does not perform geometric placement.

## Delay Buffers

When `delay = True`, the environment uses three delay buffers:

1. **`_proprio_delay_buffer`** delays proprioceptive state groups. It affects the proprioceptive input used by the locomotion and odometry paths.
2. **`_loc_heightmap_delay_buffer`** delays the local height map sent to `locomotion_policy`.
3. **`_nav_heightmap_delay_buffer`** delays the two-channel navigation height map sent to the planner and map path.

The local and navigation height maps come from the same height sensor, so the environment samples one shared time-lag tensor and applies the same delay to both height-map buffers. This keeps the two views temporally consistent.

The delay buffers are reset for the environments that reset, and their time lags are resampled at reset.

## Reward Terms

The planner reward is strongly shaped by maze-aware shortest-path distance. The environment converts the robot and goal world positions into maze cells, then queries a wall-aware BFS distance table. The table stores cell-hop counts:

```text
same cell                 -> 0 hops
adjacent reachable cells  -> 1 hop
n cell transitions       -> n hops
```

The hop count is converted to meters using `cell_size`. The current path metric also uses progress along the next reachable corridor instead of forcing the robot to pass through the center of its current cell. If robot and goal are in the same cell, direct Euclidean distance is used.

The reward contains these terms:

- **`rew_goal_distance`**: exponential reward based on the maze-aware remaining path distance. This is the main navigation-shaping term.
- **`rew_goal_progress`**: rewards the reduction in maze path distance since the previous step, clipped by `goal_progress_clip`.
- **`pen_goal_orientation`**: penalizes yaw error only when the robot is very close to the goal, using the configured path-distance threshold.
- **`pen_goal_penalty`**: constant time penalty applied every step.
- **`rew_goal_bonus`**: large bonus when the goal position and yaw termination condition is reached.
- **`pen_cmd_rate`**: penalizes changes in `cmd_vel` to encourage smooth commands.
- **`pen_cmd_bounds`**: penalizes commands outside the locomotion command range and very small commands while the robot is still far from the goal.
- **`pen_undesired_contacts`**: penalizes contacts involving undesired robot bodies such as the base, hips, and thighs.
- **`pen_stagnation`**: penalizes the robot after it has failed to make sufficient maze-path progress for the configured stagnation window.

The reward is evaluated in every vectorized environment. Because the distance and progress terms use the maze shortest path, Euclidean distance alone does not encourage the robot to move toward walls or dead ends that look close but require a longer route.

Episodes terminate when the goal is reached or the robot has an invalid base contact, and truncate at the episode time limit.

## External Models

### `locomotion_policy`

**Role:** convert the planner command and robot state into low-level joint actions.

**Observations:**

- Base angular velocity.
- Projected gravity.
- Current `cmd_vel = [vx, vy, wz]`.
- Relative joint positions.
- Joint velocities.
- Previous locomotion action.
- Delayed local height map.

The planner command is now actually passed into this model through the proprioceptive observation. The checkpoint is frozen and evaluated without gradients.

**Output:** a 12-value joint action. The environment applies:

```text
joint_target = default_joint_position + 0.25 * locomotion_action
```

### `odom_model`

**Role:** estimate robot state in the episode spawn frame without exposing simulator ground truth to the planner.

**Structure:** observation normalization, a one-layer GRU with hidden size 64, and an MLP with hidden layers `(128, 128)`.

**Input:** ten frames of delayed proprioception history, current command and locomotion action information, plus the previous 12-value odometry prediction.

**Target:**

```text
[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
```

**Loss:**

```text
loss = MSE(position) + MSE(orientation)
     + 0.5 * MSE(linear_velocity)
     + 0.5 * MSE(angular_velocity)
```

### `map_model`

**Role:** denoise and complete the persistent, already-placed map. Geometry, memory, and coordinate transforms are handled by the environment rather than by the network.

**Structure:** `MapDenoiser`, a stateless 2D convolutional encoder-decoder with skip connections. It downsamples twice, uses a convolutional bottleneck, and restores the original map resolution with interpolation-aligned decoder features. It has no RNN and no odometry input.

**Input:** a two-channel map image with shape `[B, 2, map_x_cells, map_y_cells]`:

- Channel 0: `raw_map`, the Kalman-fused height estimate.
- Channel 1: `raw_map_confidence`, the bounded uncertainty/confidence channel.

**Output:** two map channels with the same spatial shape:

- Channel 0: refined height prediction.
- Channel 1: confidence prediction after a sigmoid.

The model is trained online on each control step. Height regression uses Smooth L1 loss only on revealed cells. Confidence uses binary cross-entropy against `map_revealed` over the full map:

```text
height_loss = masked SmoothL1(predicted_height, true_map)
confidence_loss = BCE(predicted_confidence, map_revealed)
loss = height_loss + 0.5 * confidence_loss
```

`true_map` is generated by sampling the simulator’s global heightfield in the episode start frame. It is a training target and is returned only in `teacher_map`; it is not an input to the deployable map path.

## Repository Layout

- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env.py`: simulation loop, observations, delay buffers, model wiring, rewards, resets, and map plotting.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env_cfg.py`: robot, terrain, sensor, action, observation, delay, and reward configuration.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/networks/`: custom CNN/RNN model implementations.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/agents/`: PPO, teacher, and distillation configurations.
- `scripts/rsl_rl/train.py`: Isaac Lab and RSL-RL training entry point.
- `policies/`: TorchScript locomotion checkpoints.
