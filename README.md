# Go2 Navigation with Reinforcement Learning

This project trains a navigation planner for the Unitree Go2 in Isaac Lab. The planner observes a local height scan, the goal, and histories of predicted positions, then produces three command values:

```text
[cmd_x, cmd_y, cmd_z]
```

These three values are used as the robot velocity command `[vx, vy, wz]`. The planner is separated from locomotion: a pretrained low-level policy converts proprioception and a local height scan into joint targets.

## System Overview

At every environment step, the stack is wired as follows:

```text
Go2 simulation
    |-- proprioception + local height scan --> locomotion_policy --> 12 joint actions
    |                                              |
    |                                              +--> joint position targets
    |
    |-- proprioception history + previous odometry --> odom_model --> predicted odometry
    |                                                                  |
    |                                                                  +--> position histories
    |
    |-- navigation height scan + predicted odometry --> map_model --> predicted global map
    |
    +-- goal + position histories + navigation scan --> RL planner --> [cmd_x, cmd_y, cmd_z]
```

The planner action space is `3`, configured in `Go2NavEnvCfg`, with the layout `[cmd_x, cmd_y, cmd_z]`. The environment copies all three action values into `cmd_vel`.

## Installation

Install Isaac Lab first, then install this extension into the same Python environment:

```bash
python -m pip install -e source/go2_nav
```

The default locomotion checkpoint is `policies/policy_cnn_rnn_seq3.pt`. Run commands with the Isaac Lab Python launcher when Isaac Lab is not the active interpreter.

List the registered task:

```bash
python scripts/list_envs.py
```

The registered task is `Template-Go2-Nav-Direct-v0`.

## Training Flow

The environment is a vectorized Isaac Lab `DirectRLEnv` with a default scene size of 4096 environments. Each policy step runs the following sequence:

1. The RL planner receives the student observations and samples a 3-dimensional action, `[cmd_x, cmd_y, cmd_z]`.
2. The three action values become `cmd_vel`, interpreted as `[vx, vy, wz]`.
3. The locomotion model receives current proprioception and a local height map. Its command field is currently a zero placeholder rather than the planner's `cmd_vel`. Its joint action is scaled by `0.25` and added to the Go2 default joint pose.
4. The odometry model receives a rolling proprioception history plus its previous 12-value prediction. It is trained against simulator ground truth and updates the predicted odometry.
5. The predicted position is inserted into short-term and long-term histories.
6. The map model receives the navigation height scan and predicted odometry. It is trained online against the simulator height map and updates the predicted map and confidence map.
7. The simulator advances with the resulting joint targets. Rewards and termination are then computed.

The reward combines goal distance, progress, goal orientation, time, odometry prediction quality, command smoothness and bounds, undesired contacts, a goal bonus, and stagnation penalties. Episodes terminate on goal completion or a base contact, and truncate at the episode time limit.

### Planner training modes

The repository provides three RSL-RL configurations:

- **Teacher pretraining:** PPO trains a recurrent CNN policy with privileged observations, including true odometry, goal-relative information, and maze features.
- **Distillation:** a student with only navigation observations is trained from the privileged teacher.
- **Direct PPO:** `PPORunnerCfg` trains the planner with asymmetric observations: the actor receives student observations while the critic receives privileged teacher observations.

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

The teacher and planner use a CNN plus GRU with two GRU layers of width 128. The CNN has convolutional layers with 16 and 32 channels, stride 2, ReLU activations, and global average pooling. PPO uses the settings in `source/go2_nav/go2_nav/tasks/direct/go2_nav/agents/rsl_rl_ppo_cfg.py`.

## External Models

Only the locomotion policy is loaded as a pretrained TorchScript artifact. The odometry and map models are instantiated and optimized online by the environment.

### `locomotion_policy`

**Role:** convert proprioception and a local height scan into low-level joint actions. The current environment leaves the command input at zero.

**Structure:** the environment calls the exported recurrent CNN/RNN interface with one 1-D tensor and one 2-D tensor. The configured artifact is `policy_cnn_rnn_seq3.pt`; its runtime hidden state has shape `[1, num_envs, 128]`. The model is evaluated without gradients and its hidden state is reset for environments that reset.

**Observations:**

- A 45-value proprioception vector: base angular velocity (3), projected gravity (3), a zero command placeholder (3), relative joint positions (12), joint velocities (12), and the previous locomotion action (12).
- A local height scan with shape `[1, 13, 10]`, generated from the `0.1 m` local grid covering `x = [-0.5, 0.8]` and `y = [-0.5, 0.5]`.

**Loss:** none in this repository. The checkpoint is frozen and queried under inference mode. Its original locomotion-training loss is outside this project.

**Output:** a 12-value joint action. The environment computes:

```text
joint_target = default_joint_position + 0.25 * locomotion_action
```

### `odom_model`

**Role:** estimate the robot state in the episode spawn frame without using simulator ground truth as a planner input.

**Structure:** a recurrent `RNNModel` with observation normalization, a one-layer GRU with hidden size 64, and an MLP with hidden layers `(128, 128)`. It is reset per environment episode and its recurrent state is detached after each online optimization step.

**Observations:** one flattened 462-value vector:

- Ten frames of 45-value proprioception history, where each frame contains base angular velocity, projected gravity, a zero command placeholder, relative joint position, joint velocity, and the previous locomotion action.
- The previous 12-value odometry prediction.

**Target and loss:** simulator ground truth in the spawn frame, ordered as:

```text
[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
```

The online loss is:

```text
loss = MSE(position) + MSE(orientation)
     + 0.5 * MSE(linear_velocity)
     + 0.5 * MSE(angular_velocity)
```

Gradients are clipped to norm 1.0 before the Adam update.

**Output:** a 12-value predicted odometry vector with the same ordering as the target. Its position component feeds the planner's short-term and long-term position histories, and the full vector is also used by the map model and odometry reward.

### `map_model`

**Role:** build a global height map from partial local height observations and predicted odometry.

**Structure:** `CNNRNNNewModel` with:

- A 2-D CNN over the navigation height scan: convolution channels `[16, 32]`, kernel size `3`, stride `2`, ReLU activations, and global average pooling.
- A 12-value odometry branch.
- Fusion of the CNN latent and odometry branch before a one-layer GRU with hidden size 64.
- An MLP head with hidden layers `(128, 64)`.

**Observations:**

- `height_data`: a one-channel navigation scan of shape `[1, 17, 30]`, using `0.2 m` cells over the configured navigation ranges.
- `odom_data`: the current 12-value predicted odometry vector.

**Target and loss:** the target is the simulator's global height map. `map_revealed` marks cells within the robot's lidar reveal radius. The output is split into a height prediction and confidence logits. The loss is:

```text
height_loss = masked SmoothL1(height_prediction, true_map)
confidence_loss = BCEWithLogits(confidence_logits, map_revealed)
loss = height_loss + 0.5 * confidence_loss
```

Height regression is applied only to revealed cells. Confidence classification is applied to every cell because the revealed/not-revealed label is known globally in simulation. Gradients are clipped to norm 1.0 before the Adam update.

**Output:** `2 * 250 * 250` values for the configured `50 m x 50 m` map at `0.2 m` resolution:

- `250 * 250` predicted heights.
- `250 * 250` confidence logits, converted to probabilities with a sigmoid.

## Repository Layout

- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env.py`: simulation loop, model wiring, online auxiliary training, rewards, and resets.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env_cfg.py`: robot, terrain, sensor, action, observation, and checkpoint configuration.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/networks/`: custom CNN/RNN model implementations.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/agents/`: PPO, teacher, and distillation configurations.
- `scripts/rsl_rl/train.py`: Isaac Lab and RSL-RL training entry point.
- `policies/`: TorchScript locomotion checkpoints used by the environment.
- `logs/`: training outputs and checkpoints.
