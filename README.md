# Go2 Navigation with Reinforcement Learning

This project trains a navigation planner for the Unitree Go2 in Isaac Lab. The planner outputs three velocity commands:

```text
[cmd_x, cmd_y, cmd_z] = [vx, vy, wz]
```

The navigation planner is separated from low-level gait control. A pretrained locomotion policy receives the current robot state, the planner command, and a local height map, then produces joint actions for the Go2.

## System Overview

```text
Go2 simulation
    |-- proprioception + cmd_vel + local height map
    |       --> locomotion_policy --> joint actions --> joint targets
    |
    |-- proprioception history + cmd_vel + previous odometry
    |       --> odom_model --> predicted odometry
    |                              |
    |                              +--> short/long position histories
    |
    |-- navigation height map + predicted odometry
    |       --> map_model --> predicted height map + confidence map
    |
    +-- planner observations --> RL planner --> [vx, vy, wz]
```

The planner uses a student/teacher observation setup. The actor or student sees onboard-style observations and the predicted map. The critic or teacher additionally receives privileged simulator information, including true odometry, maze information, and the true map.

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

## Training Flow

Each policy step runs this sequence:

1. The planner receives its observation groups and outputs the three-dimensional action `[vx, vy, wz]`.
2. The environment copies this action directly into `cmd_vel`.
3. The locomotion policy receives delayed proprioception, the current `cmd_vel`, the previous locomotion action, and a delayed local height map. It outputs joint actions that are scaled by `0.25` and added to the default Go2 joint pose.
4. The odometry model receives a history of delayed proprioception, current `cmd_vel`, locomotion actions, and its previous odometry prediction. It is trained online against simulator odometry.
5. The predicted odometry updates the short-term and long-term position histories.
6. The map model receives the delayed navigation height map and predicted odometry. It predicts both a global height map and a confidence map.
7. The planner receives the updated observations. The simulator advances with the low-level joint targets and computes rewards and termination conditions.

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
- `student_height_scan`: the delayed navigation height scan, currently one channel with shape `[1, 17, 30]` from the `0.2 m` navigation grid.
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

## Delay Buffers

When `delay = True`, the environment uses three delay buffers:

1. **`_proprio_delay_buffer`** delays proprioceptive state groups. It affects the proprioceptive input used by the locomotion and odometry paths.
2. **`_loc_heightmap_delay_buffer`** delays the local height map sent to `locomotion_policy`.
3. **`_nav_heightmap_delay_buffer`** delays the navigation height map sent to the planner and map path.

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

**Role:** infer a global height map from partial navigation height observations and predicted odometry.

**Structure:** `CNNRNNSeqModel` with a two-layer CNN, odometry features, a one-layer GRU with hidden size 64, and an MLP head with hidden layers `(128, 64)`.

**Input:**

- Delayed navigation height map from the shared height sensor.
- Current predicted 12-value odometry.

**Output:** two flattened map predictions:

- Height prediction for every global map cell.
- Confidence logit for every global map cell.

The map loss is:

```text
height_loss = masked SmoothL1(height_prediction, true_map)
confidence_loss = BCEWithLogits(confidence_logits, map_revealed)
loss = height_loss + 0.5 * confidence_loss
```

Height regression is applied only to revealed cells. Confidence classification is applied everywhere because the simulator knows which cells have been revealed.

## Repository Layout

- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env.py`: simulation loop, observations, delay buffers, model wiring, rewards, resets, and map plotting.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env_cfg.py`: robot, terrain, sensor, action, observation, delay, and reward configuration.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/networks/`: custom CNN/RNN model implementations.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/agents/`: PPO, teacher, and distillation configurations.
- `scripts/rsl_rl/train.py`: Isaac Lab and RSL-RL training entry point.
- `policies/`: TorchScript locomotion checkpoints.
