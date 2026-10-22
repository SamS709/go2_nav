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
    |-- delayed proprioception history + cmd_vel + loc actions
    |       --> odom_model --> [roll, pitch, v_body, w_body]
    |                              |
    |                              +--> dead-reckoning (outside the network) --> pose_est [x, y, z, yaw]
    |                                          ^                                      |
    |                                          | nudge (scan-to-map correction)       +--> position histories
    |                                          |                                      +--> map placement
    |-- ray hits (body frame, delayed, + pose history at capture time)
    |       --> scan-to-map matching ---------+
    |       --> persistent start-frame map (cell placement + Kalman height/variance fusion)
    |       --> ego crop (yaw-aligned, centred on pose_est)
    |       --> EgoMapDenoiser --> predicted height + confidence
    |
    +-- planner observations (all relative to pose_est) --> RL planner --> [vx, vy, wz]
```

The planner uses a student/teacher observation setup. The actor or student sees onboard-style observations and the predicted ego map. The critic or teacher additionally receives privileged simulator information, including true odometry, maze information, and the true map.

`go2_nav_env.py` is a vectorized `DirectRLEnv`. It owns the robot, terrain, sensors, online odometry and map learners, pose estimation, scan-to-map matching, maze path tables, reward terms, and student/teacher observation construction. The planner does not directly control joints: its three velocity commands are passed through the locomotion policy before being applied to the robot.

The map pipeline separates geometry from learning. The environment places measurements into a persistent map using the **estimated** pose and exact tensor operations. The learned map model only denoises and completes an ego-centric crop of the result; it receives neither odometry nor raw coordinates.

## Coordinate frames

| Frame | Definition | Used for |
|---|---|---|
| world | Isaac Sim frame | simulator state, rewards, maze tables |
| start frame | robot pose at the beginning of the episode (origin = spawn, x = spawn heading) | persistent map, goal, position histories, pose estimate |
| true pose | ground truth in the start frame (`_true_pose_s`) | supervision, teacher observations, sensor emulation. **Never** given to the student |
| estimated pose | `pose_est = [x, y, z, yaw]` in the start frame: dead-reckoned odometry + scan-matching nudges | map placement, student observations |
| body frame | yaw-aligned, attached to the robot (x forward, y left) | emulated scan, student goal and histories |
| ego crop | window centred on the robot, rotated by its yaw (dim 1 = forward, dim 2 = left) | denoiser and planner map |

Heights in the map are **tile-relative** (`z_world - env_origin_z`). The ego crops are expressed **relative to the robot base height**, so a flat floor reads about `-0.4`.

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

1. `prev_cmd_vel` is stored, then the planner action `[vx, vy, wz]` is copied into `cmd_vel`.
2. The locomotion observation builder creates delayed proprioception, command/action features, and a local heightmap. The frozen locomotion policy returns joint actions.
3. The recurrent odometry model consumes the delayed proprioception history, commands and locomotion actions. It predicts roll, pitch, body linear velocity and body angular velocity. It is trained online against simulator state.
4. The pose is **integrated outside the network** (`_dead_reckon`): yaw from the gyro (+ roll/pitch from the network), position from the rotated body velocity (trapezoidal rule). `pose_dr` is the same integration without corrections, kept for metrics.
5. The map update (`_update_map_perception`):
   1. the student navigation grid is built (policy convention, unchanged);
   2. the ray hits are expressed in the body frame (emulated sensor), pushed through the shared height-sensor delay buffer;
   3. the delayed scan is placed with the estimated pose **of its capture time** (pose history indexed with the same lag);
   4. optional scan-to-map matching nudges `pose_est`;
   5. the placed points are fused by a vectorised Kalman update;
   6. an ego crop is denoised by `EgoMapDenoiser`, which is trained online.
6. The position histories are updated with `pose_est`, then `_get_observations` builds student and teacher observations. Low-level joint targets are applied and the simulator advances.

The student map and student height scan are therefore computed from the state at the start of the control step. Envs that just reset get a blank map observation for their first step (the map memory is cleared at reset).

### Episode reset

`_reset_idx` resets the robot, locomotion hidden state, odometry hidden state and input history, command/action histories, odometry output, `pose_est`, `pose_dr`, pose history buffers, drift counter, gyro bias, position histories, stall counters, goals and start poses. It rebuilds the true start-frame map from the global terrain heightfield, **clears the persistent map** (`raw_map`, `_map_variance`, `map_seen_gt`, student map and nav grid) and resamples delay lags **only for the envs being reset**. Curriculum updates can move an environment to another terrain level and refresh its maze path table.

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

### Perception pretraining and freezing

The odometry and map models are trained online, which makes the planner input distribution non-stationary. Recommended workflow:

1. run a first training with `train_perception=True` (default);
2. call `env.save_perception_models("perception.pt")` (odom model, odom input normaliser, map model);
3. set `perception_checkpoint="perception.pt"` and `train_perception=False` for the PPO runs.

## Planner Observations

The planner observation groups are defined in `rsl_rl_ppo_cfg.py`. **Everything the student sees is expressed relative to the estimated pose.**

### Student/actor observations

- `student_proprio` (309 values, same size as before): concatenation of:
  - Goal XY in the estimated robot frame: 2 values.
  - Goal yaw relative to the estimated yaw: 1 value.
  - Short-term position history, relative to the estimated pose: `50 x 3`.
  - Long-term position history, relative to the estimated pose: `50 x 3`.
  - Predicted body velocities `[vx_b, vy_b, wz_b]`: 3 values.
  - Current planner command `cmd_vel`: 3 values.
- `student_height_scan`: the delayed navigation height scan, shape `[2, 17, 30]` (`0.2 m` grid). Channel 0 is height, channel 1 is the valid-cell mask.
- `student_map`: ego crop, shape `[2, map_crop_cells, map_crop_cells]` (**shape changed**, default `2 x 64 x 64`):
  - Channel 0: predicted height relative to the robot base.
  - Channel 1: predicted confidence, `1 / (1 + sigma^2 / ref^2)`, with `sigma^2` the predicted error variance.

### Teacher/critic observations

- `teacher_proprio`: student proprioception plus privileged terms (true odometry in the spawn frame, goal displacement in world coordinates, goal heading as sine/cosine, maze terrain features and wall information).
- `teacher_height_scan`: the non-randomized navigation height scan.
- `teacher_map`: the true map as an ego crop at the **true** pose, shape `[1, map_crop_cells, map_crop_cells]`, relative to the true base height.

## Navigation Heightmap and Persistent Map

Two different scan products are built from the same height sensor and share one time lag:

1. **Navigation grid** (`_build_nav_heightmap`): policy-convention grid used for `student_height_scan` (dropout, delay, calibration noise). It is **no longer used for map placement**, because its axis/sign convention depends on the sensor mounting.
2. **Map scan** (`_build_map_scan`): the raw ray hits expressed in the yaw-aligned body frame, `[N, R, 4] = (x_b, y_b, z_rel, valid)`. The simulator ground-truth pose is used here only to emulate what a real scanner reports in its own frame. Optional z noise (`map_scan_noise`) is added when `cfg.randomize` is set.

The environment maintains per-environment persistent map buffers at the `map_cell_size` resolution:

- `raw_map`: Kalman-fused height estimate (tile-relative z).
- `_map_variance`: per-cell variance, initialised to `1e6`, floored at `map_min_variance` so the map keeps adapting.
- `map_seen_gt`: ground-truth fog of war, built by placing the same delayed scan with the **true** capture-time pose. Used for supervision only.
- `true_map`: simulator heightfield sampled in the start frame (target / teacher input).

Fusion: each scan point is rotated by the estimated yaw and translated by the estimated position (from the pose history, at the scan's capture time), and its z is `z_rel + pose_z + base_height`. Points falling into the same cell are averaged, then a vectorised Kalman update is applied with measurement variance `(map_meas_std^2 + (map_drift_var_gain * distance_since_last_correction)^2) / n_hits`. The confidence is therefore lower when the pose has drifted.

### Scan-to-map matching (loop closure)

Every `match_period` steps, the delayed scan is matched against the cells the map already knows (variance below `match_var_thresh`). A grid of `5 x 5 x 5` candidate offsets (`dx, dy` in `±0.2 m`, `dyaw` in `±3°`) is scored with a variance-normalised squared height error. The mean height error is removed per candidate, so z drift does not disturb the xy search. A correction is accepted only when:

- enough known cells overlap (`match_min_points`);
- the best candidate improves the zero-offset score by `match_improve_thresh`;
- the score profile along the corrected axis is not flat (`match_axis_obs_thresh`). Along a straight corridor the match is unobservable in the corridor direction, so only the observable axes are corrected.

The correction is a gained nudge (`match_gain`, `match_gain_z`), applied to `pose_est`, to its history buffer and to the position histories. The drift counter is then reset. Because mazes contain revisits, this acts as loop closure.

### Ego crop and denoiser

The persistent map stays in the start frame. For the denoiser and the planner, `_compute_crops` samples a yaw-aligned `map_crop_cells x map_crop_cells` window centred on `pose_est` (nearest sampling, same resolution as the map). The target and masks are sampled at the **true** pose from `true_map` and `map_seen_gt`; near the robot both windows cover the same physical area even when the estimated frame has drifted.

## Delay Buffers

When `delay = True`, the environment uses four delay buffers plus a pose history:

1. **`_proprio_delay_buffer`** delays proprioceptive state groups (locomotion and odometry paths).
2. **`_loc_heightmap_delay_buffer`** delays the local height map sent to `locomotion_policy`.
3. **`_nav_heightmap_delay_buffer`** delays the navigation grid sent to the planner.
4. **`_map_scan_delay_buffer`** delays the body-frame map scan.
5. **Pose histories** (`_pose_est_hist`, `_pose_true_hist`) are indexed with the same lag, so that a delayed scan is placed with the pose valid at its capture time.

The three height buffers and the pose histories come from the same sensor and share one lag tensor (`_height_lag`). At reset, **only the resetting envs** get new lags; other envs keep theirs. A `delay_length` of 0 is handled.

## Reward Terms

The planner reward is strongly shaped by maze-aware shortest-path distance. The environment converts the robot and goal world positions into maze cells, then queries a wall-aware BFS distance table. The table stores cell-hop counts:

```text
same cell                 -> 0 hops
adjacent reachable cells  -> 1 hop
n cell transitions       -> n hops
```

The hop count is converted to meters using `cell_size`. The path metric also uses progress along the next reachable corridor. If robot and goal are in the same cell, the direct Euclidean distance **in meters** is used (it used to be divided by `cell_size`).

The reward contains these terms:

- **`rew_goal_distance`**: exponential reward on the maze-aware remaining path distance (main shaping term).
- **`rew_goal_progress`**: reduction in maze path distance since the previous step, clipped by `goal_progress_clip`.
- **`pen_goal_orientation`**: yaw error penalty only very close to the goal.
- **`pen_goal_penalty`**: constant time penalty applied every step.
- **`rew_goal_bonus`**: large bonus when the goal position and yaw termination condition is reached.
- **`pen_cmd_rate`**: penalizes changes in `cmd_vel`. It was always zero before (the previous command was overwritten too early) and is now active.
- **`pen_cmd_bounds`**: penalizes commands above the locomotion range and a too small **translational** command (`|[vx, vy]| <= 0.2`) while still far from the goal. The previous version used the signed minimum over `[vx, vy, wz]`, which penalized every straight-line command.
- **`pen_undesired_contacts`**: penalizes contacts involving the base, hips and thighs.
- **`pen_stagnation`**: penalizes the robot after it has failed to make sufficient maze-path progress for the stagnation window.

Episodes terminate when the goal is reached or the robot has an invalid base contact, and truncate at the episode time limit.

## External Models

### `locomotion_policy`

**Role:** convert the planner command and robot state into low-level joint actions.

**Observations:** base angular velocity, projected gravity, current `cmd_vel`, relative joint positions, joint velocities, previous locomotion action, delayed local height map. The checkpoint is frozen and evaluated without gradients.

**Output:** a 12-value joint action:

```text
joint_target = default_joint_position + 0.25 * locomotion_action
```

### `odom_model`

**Role:** estimate the robot's **motion** from proprioception (absolute position is not observable from proprioception; velocity is).

**Structure:** running-statistics input normalisation (`_RunningNorm`, in the environment), a one-layer GRU with hidden size 64, and an ELU MLP with hidden layers `(128, 128)`.

**Input:** ten frames of delayed proprioception (30 values), current command (3) and locomotion action (12), i.e. `10 x 45`. The previous prediction is no longer fed back.

**Output (8 values):**

```text
[roll, pitch, vx_b, vy_b, vz_b, wx_b, wy_b, wz_b]
```

**Loss:**

```text
loss = MSE(roll, pitch) + MSE(linear_velocity) + 0.5 * MSE(angular_velocity)
```

**Pose integration (not learned):** yaw rate from the delayed gyro (plus per-episode random bias when `cfg.randomize`) converted with the predicted roll/pitch; position from the body velocity rotated into the start frame. Metrics logged: `odom/ate_xy_corrected`, `odom/ate_xy_dead_reckoning`, `odom/yaw_err_*`, `odom/z_err_corrected`.

### `map_model` (`EgoMapDenoiser`)

**Role:** denoise and complete an ego-centric crop of the fused map. Geometry, memory and coordinate transforms are handled by the environment.

**Structure:** U-Net, two stride-2 downsamplings, dilated bottleneck (dilations 2 and 4), GroupNorm, ELU, skip connections, zero-initialised residual head (starts as the identity on the Kalman estimate). Stateless, no RNN, no odometry input. The receptive field covers roughly the whole 64x64 crop.

**Input** `[B, 4, n, n]`:

0. height relative to the robot, `0` (flat-ground prior) in unobserved cells;
1. normalised `log(variance)` of the Kalman cell;
2. observed mask (`variance < map_observed_var_thresh`);
3. `1 / (1 + variance)`.

**Output** `[B, 2, n, n]`: height residual and log-variance of the residual. `pred_height = input_height + residual`; the planner confidence is `1 / (1 + exp(logvar) / map_conf_ref_std^2)`.

**Loss** (online, every step):

```text
loss = weighted SmoothL1(pred_height, target)          # weights: observed 1, seen-unobserved 0.5, never seen 0.05
     + 0.5 * edge loss (|grad pred - grad target|)      # keeps wall edges sharp
     + 0.1 * Gaussian NLL(logvar, detached error)       # calibrated uncertainty
```

Logged: `map/loss_*`, `map/raw_mae_observed`, `map/pred_mae_observed`, `map/pred_mae_seen_unobserved`. If `pred_mae_observed` is not below `raw_mae_observed`, the denoiser is not earning its compute.

`true_map` is a training target and appears only in `teacher_map`; it is never an input of the deployable map path.

## Optional configuration

All entries below are read with `getattr(cfg, name, default)`: the environment config does not need to change.

| Name | Default | Meaning |
|---|---|---|
| `train_perception` | `True` | train odom/map models online; `False` freezes them |
| `perception_checkpoint` | `None` | file written by `save_perception_models` |
| `odom_lr`, `map_lr` | `1e-3` | learning rates |
| `gyro_bias_max` | `0.01` | rad/s, per-episode gyro bias (needs `cfg.randomize`) |
| `map_crop_cells` | `64` | ego crop size (cells) |
| `map_scan_noise` | `0.01` | m, z noise of the map scan (needs `cfg.randomize`) |
| `map_scan_max_range` | `6.0` | m, maximum xy range of used ray hits |
| `map_meas_std` | `0.05` | m, per-hit height std in the Kalman filter |
| `map_min_variance` | `1e-4` | variance floor |
| `map_observed_var_thresh` | `1.0` | "observed" threshold on the variance |
| `map_drift_var_gain` | `0.02` | measurement std added per metre since the last correction |
| `map_conf_ref_std` | `0.1` | std (m) giving confidence 0.5 |
| `map_w_complete`, `map_w_unseen` | `0.5`, `0.05` | loss weights for seen-unobserved / never-seen cells |
| `scan_match` | `True` | enable scan-to-map correction |
| `match_period` | `2` | run matching every k control steps |
| `match_xy_range`, `match_xy_steps` | `0.2`, `5` | translation search (steps must be odd) |
| `match_yaw_range_deg`, `match_yaw_steps` | `3.0`, `5` | rotation search (steps must be odd) |
| `match_max_points`, `match_chunk` | `256`, `128` | rays per match, envs per chunk (memory) |
| `match_var_thresh` | `0.05` | only cells below this variance are matched against |
| `match_min_points` | `40` | minimum overlapping known cells |
| `match_improve_thresh` | `0.3` | required score improvement over zero offset |
| `match_axis_obs_thresh` | `0.5` | minimum score-profile range for an axis to be corrected |
| `match_gain`, `match_gain_z` | `0.5`, `0.5` | fraction of the found offset that is applied |

Note: use `match_xy_range / (match_xy_steps - 1)` close to `map_cell_size`, otherwise candidates alias onto the same cells.

## Repository Layout

- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env.py`: simulation loop, pose estimation, scan matching, map fusion, observations, delay buffers, model wiring (including `EgoMapDenoiser`), rewards, resets, debug plotting.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/go2_nav_env_cfg.py`: robot, terrain, sensor, action, observation, delay, and reward configuration.
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/networks/`: custom CNN/RNN model implementations (the old `MapDenoiser` is no longer used by the environment).
- `source/go2_nav/go2_nav/tasks/direct/go2_nav/agents/`: PPO, teacher, and distillation configurations (`student_map` / `teacher_map` shapes must match `map_crop_cells`).
- `scripts/rsl_rl/train.py`: Isaac Lab and RSL-RL training entry point.
- `policies/`: TorchScript locomotion checkpoints.