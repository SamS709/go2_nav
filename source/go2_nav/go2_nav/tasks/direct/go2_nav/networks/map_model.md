# Map Model (`CNNConvGRUMapModel`)

This document explains the architecture of the online map-prediction model used in Go2
navigation: what it does, why it is built this way, and how observations flow through it.

## 1. Role in the system

The map model turns a **partial, robot-relative** height scan into a **global, persistent**
map of the environment, together with a confidence estimate of how much to trust each cell.

```text
delayed navigation height scan [1, 17, 30]   ---\
                                                   --> map_model --> global height map (H x W)
predicted odometry [x, y, z, r, p, sin, cos,       --> global confidence map (H x W)
                     vx, vy, vz, wx, wy, wz]   ---/
```

The planner then consumes this predicted map (height + confidence) as one of its
observation groups (`student_map`). Unlike the ground-truth map available to the critic,
the predicted map is something the robot could plausibly reconstruct online from its own
sensors, which is why the odometry fed in is the **predicted** odometry from `odom_model`,
never the simulator's true state.

## 2. Why not a flat CNN + MLP model?

A naive design (and the one the model started as) is a `CNNRNNSeqModel`: encode the scan
with a CNN, feed a GRU with hidden size 64, and decode the whole global map with a dense
MLP head.

This has two structural problems for a *mapping* task specifically:

1. **A vector hidden state is a poor map.** A 64-dim GRU state has to compress the entire
   history of everywhere the robot has seen on a 250x250 grid. It can encode coarse
   statistics ("about this much wall around me") but not "there's a wall at cell (130, 42)".
   As the map grows, this bottleneck gets worse, not better.
2. **A dense output head has no notion of space.** Predicting `H*W` values from a flat MLP
   means the network has to relearn spatial structure (neighboring cells are related, walls
   are contiguous) from scratch, with a huge parameter count and no inductive bias for it.

The map model instead uses a **spatially structured memory and a convolutional decoder**,
so the network's inductive biases match the problem: a map is a 2D grid, and updating it is
a local, spatial operation.

## 3. Architecture overview

```text
                     ┌─────────────────────────────────────────────┐
                     │                 per-step input               │
                     │  height_data [B,1,17,30]   odom_data [B,13]   │
                     └───────────────┬───────────────┬──────────────┘
                                      │               │
                              ┌───────▼──────┐        │
                              │   CNN (2D)   │        │ obs_normalizer
                              │  no pooling, │        │ (EmpiricalNormalization)
                              │   flattened  │        │
                              └───────┬──────┘        │
                                      │                ▼
                              latent_cnn (576)    latent_1d (13, normalized)
                                      │                │
                                      └───────┬────────┘
                                              ▼
                                   ┌──────────────────┐
                                   │   fusion MLP      │  -> conditioning vector f
                                   │  (128, 128), ELU   │
                                   └─────────┬──────────┘
                                              ▼
                                   ┌──────────────────┐
                                   │   FiLM (Linear)   │  -> gamma, beta (32 ch each)
                                   └─────────┬──────────┘
                                              │
      raw x, y from odom_data  ──►  robot-relative coordinate grid (dx, dy, r)
                                              │
                                   ┌──────────▼──────────┐
                                   │      coord_net       │  conv(3->32->32)
                                   └──────────┬──────────┘
                                              │  modulated by (gamma, beta)  [FiLM]
                                              ▼
                                   ┌──────────────────┐
                                   │     ConvGRU        │  spatial hidden state
                                   │  (32 -> 32 ch)     │  (L, B, 32, h/stride, w/stride)
                                   └─────────┬──────────┘
                                              ▼
                                   ┌──────────────────┐
                                   │   MapDecoder       │  conv, upsample, conv
                                   │  32->32->32->2 ch  │  bilinear to (H, W)
                                   └─────────┬──────────┘
                                              ▼
                             flattened [height(H*W), confidence_logit(H*W)]
```

### 3.1 Scan encoder (`CNN`)

A standard 2-layer conv stack (16, then 32 channels, stride 2, ReLU), **without** global
average pooling. Pooling was removed deliberately: an average-pooled scan latent tells the
model "how much height variation is visible" but throws away *where in the scan* it is,
which is exactly the information needed to place it correctly on the global map. Instead,
the 3x6 feature map is flattened (32 * 3 * 6 = 576), preserving the spatial layout of the
scan all the way into the fusion step.

### 3.2 Odometry input

`odom_data` is 13-dimensional: `[x, y, z, roll, pitch, sin(yaw), cos(yaw), vx, vy, vz, wx,
wy, wz]`. Yaw is passed as `sin`/`cos` rather than a raw angle so the network sees a
continuous representation of heading with no wraparound discontinuity at +/-pi. This
conversion is done by the **caller**, where `odom_data` is assembled from the odometry
model's output — the map model itself is a pure consumer and does not derive any new
features from its inputs. `x` and `y` are also used in raw (un-normalized) form separately,
for the geometric coordinate transform below (see 3.4); the normalized copy in `latent_1d`
still includes them and is used only for the fusion/FiLM conditioning.

### 3.3 Fusion + FiLM conditioning

The normalized odometry vector and the flattened scan latent are concatenated and passed
through a small MLP (`fusion`, 128-128, ELU) to produce a conditioning vector `f`. `f` is
then projected to `(gamma, beta)` pairs (one pair per channel of `coord_net`'s output) via a
FiLM layer (`film`). This lets the network modulate *how* the spatial write pattern below is
shaped, based on what the scan currently looks like and what the robot's orientation is —
without needing a separate learned spatial transform.

### 3.4 Geometric coordinate grid (`coord_net`)

For every cell of the (possibly downsampled) global map state, the model computes:

```text
dx = (cell_x - robot_x) / coord_scale
dy = (cell_y - robot_y) / coord_scale
r  = sqrt(dx^2 + dy^2)
```

using the **raw** predicted `x, y` (not normalized), and the map's known resolution and
origin. This `(dx, dy, r)` grid is a deterministic, geometrically correct encoding of "where
is this cell relative to the robot" — the network doesn't have to learn odometry-to-map
geometry from scratch, only how to use it. The 3-channel grid is passed through two conv
layers (`coord_net`) and then modulated with the FiLM `(gamma, beta)` from step 3.3, which is
how the scan content and heading actually influence what gets written into the map.

This is a middle ground between two extremes: a purely learned model that has to infer
robot-to-map geometry from data (slow to train, prone to drift), and a purely hand-coded
geometric warp of the scan into the map frame (fragile to odometry noise, no learned
denoising/completion). Here the geometry is exact, and only the *content* written at each
location is learned.

### 3.5 Spatial recurrent memory (`ConvGRU`)

The recurrent core is a `ConvGRUCell` stack: the same gating equations as a standard GRU,
but every gate is a convolution instead of a linear layer, and the hidden state is a spatial
tensor `(B, C, h, w)` rather than a vector. This is the central design choice of the model:

- **Memory has spatial identity.** A revealed wall at one location updates the hidden state
  at that location and persists there; it does not have to compete for capacity in a global
  vector against everything else the robot has seen.
- **Locality matches the physics of the problem.** Height-map information is inherently
  local — a scan mostly informs the map cells near the robot at that instant. A conv-based
  update naturally biases toward local, incremental writes, whereas a dense GRU has to learn
  this locality from data with no structural help.
- **The state can be run at reduced resolution (`state_stride`).** to keep memory and
  compute bounded on large maps, the ConvGRU operates at `map_shape / state_stride`
  resolution (e.g. state_stride=2 on a 250x250 map gives a 125x125 hidden state), and the
  decoder upsamples back to full resolution.

### 3.6 Decoder (`MapDecoder`)

A small fully-convolutional head: two conv+ELU blocks, then bilinear upsampling to the full
map resolution (if the state was run at reduced resolution), then two more conv layers down
to 2 output channels (height, confidence logit). Using convolutions instead of a dense layer
keeps the parameter count independent of map size and lets the same kernels denoise/complete
the map everywhere, rather than learning a separate weight per output cell. Bilinear
upsampling (not transposed convolution) is used to avoid checkerboard artifacts.

The output is flattened to `[height (H*W), confidence_logit (H*W)]` to match the loss and
the downstream consumer's expected flat layout — no change was needed elsewhere in the
pipeline for this.

## 4. Observation flow, step by step

For a single environment step:

1. **Inputs:** `height_data [B, 1, 17, 30]` (delayed navigation scan) and `odom_data [B, 13]`
   (predicted odometry, yaw already encoded as sin/cos by the caller).
2. `obs_normalizer` normalizes `odom_data` -> `latent_1d`. A raw (un-normalized) copy of
   `odom_data` is kept separately for the geometric transform (`raw_1d`).
3. `height_data` -> `CNN` -> `latent_cnn` (576-dim, spatially flattened, no pooling).
4. `latent_1d` and `latent_cnn` are concatenated -> `fusion` MLP -> `f`.
5. `f` -> `film` -> `(gamma, beta)`.
6. `raw_1d`'s `x, y` -> `(dx, dy, r)` grid over the (possibly downsampled) map -> `coord_net`
   -> FiLM-modulated by `(gamma, beta)` -> `out`.
7. `out` and the previous hidden state `hidden[i]` are passed through each `ConvGRUCell` in
   sequence -> new hidden state (stored on `self.rnn.hidden_state` in step/rollout mode, or
   threaded explicitly through time in batched training mode).
8. The final ConvGRU layer's output -> `MapDecoder` -> upsample to full map resolution ->
   `[height, confidence_logit]`, flattened.

Two execution modes are supported, mirroring the pattern in `CNNRNNSeqModel`:

- **Step mode** (`masks is None`): used at rollout/inference time. Reads and writes
  `self.rnn.hidden_state` directly, one step at a time.
- **Batch mode** (`masks` provided): used during PPO/training updates on padded trajectory
  batches. Unrolls `forward_step` explicitly over the time dimension from a given initial
  hidden state, then unpads with `unpad_trajectories`, matching how `RNN` is unrolled inside
  `CNNRNNSeqModel`.

## 5. Training

The loss is unchanged from the original flat-MLP design:

```text
height_loss     = masked SmoothL1(height_prediction, true_map)   # revealed cells only
confidence_loss = BCEWithLogits(confidence_logits, map_revealed) # all cells
loss            = height_loss + 0.5 * confidence_loss
```

Because training happens online during rollout (see `_train_map_step`), two details matter
for the recurrent state specifically:

- **Inference-mode hidden state.** Rollouts run under `torch.inference_mode()`, so the
  hidden state created there is an inference tensor. `get_latent` clones it before use if a
  gradient-carrying forward pass needs it, since autograd cannot save an inference tensor
  for backward.
- **Truncated BPTT.** The stored hidden state is detached after every step
  (`self.rnn.hidden_state = new_hs.detach()`), so each training step backpropagates through
  a 1-step recurrent horizon. This matches a per-step online update; a longer horizon would
  require accumulating loss over several steps before detaching.

## 6. Key hyperparameters

| Parameter | Meaning | Notes |
|---|---|---|
| `rnn_hidden_dim` | ConvGRU hidden **channels** (not a vector size) | Memory scales as `B x C x h x w`; lower this before scaling up env count. |
| `state_stride` | Downsampling factor between map resolution and ConvGRU resolution | Higher = cheaper, blurrier recovered detail. |
| `input_channels` | Channels in `coord_net` / ConvGRU input | Also the FiLM output width (`2 x input_channels`). |
| `decoder_channels` | Conv width in `MapDecoder` | Independent of map size. |
| `odom_indices` | Positions of `(x, y, sin_yaw, cos_yaw)` in `odom_data` | Must match how the caller assembles `odom_data`. |
| `coord_scale` | Normalizes `(dx, dy)` before `coord_net` | Defaults to half the map's largest dimension in meters. |
| `map_origin` | World/spawn-frame xy of map cell `[0, 0]` | Must match the frame `true_map` is expressed in. |

## 7. Known limitations / things to revisit

- **Map orientation convention.** The code assumes `map[i, j]` has `i` along x and `j` along
  y. If the true map buffer uses the opposite convention, swap `grid_x`/`grid_y` in
  `SpatialMemoryRNN`.
- **Roll/pitch are not sin/cos-encoded.** Only yaw is, since roll/pitch normally stay near
  zero during locomotion. If the robot operates on steep terrain and the map degrades under
  large pitch/roll, revisit this.
- **Scale at large env counts.** At high resolution and many parallel envs, the ConvGRU
  hidden state and decoder convs are the memory bottleneck. `state_stride`, `rnn_hidden_dim`,
  and `decoder_channels` are the levers to pull first.
- **The planner's map consumption is a separate concern.** The predicted map is only useful
  to the planner if the planner's own CNN preserves spatial structure when encoding it
  (i.e. it should not end in a global average pool either).