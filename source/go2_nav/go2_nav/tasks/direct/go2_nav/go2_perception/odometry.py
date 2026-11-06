"""Learned odometry: GRU predicts attitude + body velocities, the pose is dead-reckoned OUTSIDE the network.

Deployment usage (one call per control step)::

    odom = OdometryEstimator(1, device, params.odom, step_dt)
    odom.load_checkpoint_state(torch.load(path))
    ...
    odom.step(proprio_delayed, cmd_vel, loc.actions)      # proprio_delayed [N,30], first 3 = gyro
    odom.pose_est                                          # [N, 4] x, y, z, yaw in the start frame
    odom.pos_hist.push(odom.pose_est[:, :3])               # after any scan-to-map correction
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.rnn_model import RNNModel

from .config import OdomParams
from .math_utils import wrap_to_pi
from .training import odom_loss


class RunningNorm:
    """Running mean/variance normaliser (Chan et al. parallel update)."""

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


def build_odom_model(in_dim: int, num_envs: int, device) -> RNNModel:
    obs = TensorDict({"proprio": torch.randn(num_envs, in_dim)})
    model = RNNModel(
        obs=obs,
        obs_groups={"policy": ["proprio"]},
        obs_set="policy",
        output_dim=8,  # [roll, pitch, vx_b, vy_b, vz_b, wx_b, wy_b, wz_b]
        hidden_dims=(128, 128),
        activation="elu",
        obs_normalization=False,  # inputs are normalised by RunningNorm
        rnn_num_layers=1,
        rnn_hidden_dim=64,
        rnn_type="gru",
    ).to(device)
    model.eval()
    model.reset()
    with torch.no_grad():
        model(TensorDict({"proprio": obs["proprio"].to(device)}, batch_size=[num_envs]))
    return model


def integrate_pose(pose, roll, pitch, v_b, yaw_rate, dt) -> torch.Tensor:
    """Dead-reckon [x, y, z, yaw] from body velocity, roll/pitch and yaw rate (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    yaw = wrap_to_pi(pose[:, 3] + yaw_rate * dt)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    vx, vy, vz = v_b[:, 0], v_b[:, 1], v_b[:, 2]
    vwx = cy * cp * vx + (cy * sp * sr - sy * cr) * vy + (cy * sp * cr + sy * sr) * vz
    vwy = sy * cp * vx + (sy * sp * sr + cy * cr) * vy + (sy * sp * cr - cy * sr) * vz
    vwz = -sp * vx + cp * sr * vy + cp * cr * vz
    return torch.stack([pose[:, 0] + vwx * dt, pose[:, 1] + vwy * dt, pose[:, 2] + vwz * dt, yaw], dim=-1)


class PositionHistory:
    """Short / long-term histories of the estimated start-frame position (most recent at the end)."""

    def __init__(self, num_envs: int, device, p: OdomParams):
        self.short = torch.zeros(num_envs, p.length_short_term, 3, device=device)
        self.long = torch.zeros(num_envs, p.length_long_term, 3, device=device)
        self.period_short = max(1, int(round(1 / p.freq_pos_short_term)))
        self.period_long = max(1, int(round(1 / p.freq_pos_long_term)))
        self.tick = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids: torch.Tensor) -> None:
        self.short[ids] = 0.0
        self.long[ids] = 0.0
        self.tick[ids] = 0

    def push(self, xyz: torch.Tensor) -> None:
        for name, period in (("short", self.period_short), ("long", self.period_long)):
            hist = getattr(self, name)
            update = (self.tick % period == 0)[:, None, None]
            rolled = torch.roll(hist, shifts=-1, dims=1)
            rolled[:, -1, :] = xyz
            setattr(self, name, torch.where(update, rolled, hist))
        self.tick += 1

    def shift(self, delta_xyz: torch.Tensor) -> None:
        """Rigid nudge after a scan-to-map correction."""
        self.short += delta_xyz[:, None, :]
        self.long += delta_xyz[:, None, :]

    @staticmethod
    def _rel(hist: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        d = hist[..., :2] - pose[:, None, :2]
        c = torch.cos(pose[:, 3])[:, None]
        s = torch.sin(pose[:, 3])[:, None]
        return torch.stack(
            [c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], hist[..., 2] - pose[:, None, 2]], dim=-1
        )

    def relative_to(self, pose_est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both histories in the estimated robot frame, flattened to [N, L*3]."""
        n = pose_est.shape[0]
        return self._rel(self.short, pose_est).reshape(n, -1), self._rel(self.long, pose_est).reshape(n, -1)


class OdometryEstimator:
    def __init__(self, num_envs: int, device, params: OdomParams, step_dt: float, train: bool = False, lr: float = 1e-3):
        self.N, self.device, self.p, self.dt = num_envs, device, params, float(step_dt)
        self.train_enabled = bool(train)
        in_dim = params.frame_dim * params.length_odom_hist
        self.model = build_odom_model(in_dim, num_envs, device)
        self.norm = RunningNorm(in_dim, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr))

        self.obs_hist = torch.zeros(num_envs, params.length_odom_hist, params.frame_dim, device=device)
        self.out = torch.zeros(num_envs, 8, device=device)  # roll, pitch, v_b(3), w_b(3)
        self._prev_vel_b = torch.zeros(num_envs, 3, device=device)
        self._prev_yaw_rate = torch.zeros(num_envs, device=device)
        self.pose_est = torch.zeros(num_envs, 4, device=device)  # x, y, z, yaw (start frame), corrected
        self.pose_dr = torch.zeros(num_envs, 4, device=device)  # dead-reckoning only (metrics)
        self.drift_dist = torch.zeros(num_envs, device=device)  # distance since last accepted correction
        self.gyro_bias = torch.zeros(num_envs, 3, device=device)
        self.pose_est_hist = torch.zeros(num_envs, params.pose_hist_len, 4, device=device)  # 0 = latest
        self.pos_hist = PositionHistory(num_envs, device, params)
        self.last_log: dict = {}

    # ------------------------------------------------------------------ properties
    @property
    def roll_pitch(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.out[:, 0], self.out[:, 1]

    @property
    def student_vel(self) -> torch.Tensor:
        """[vx_b, vy_b, wz_b]."""
        return self.out[:, [2, 3, 7]]

    # ------------------------------------------------------------------ lifecycle
    def reset(self, env_ids: torch.Tensor, randomize: bool = False) -> None:
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        mask[env_ids] = True
        with torch.no_grad():
            self.model.reset(mask)
        for t in (self.out, self._prev_vel_b, self._prev_yaw_rate, self.pose_est, self.pose_dr,
                  self.pose_est_hist, self.drift_dist, self.obs_hist):
            t[env_ids] = 0.0
        bias_max = float(self.p.gyro_bias_max) * float(randomize)
        self.gyro_bias[env_ids] = (2.0 * torch.rand(env_ids.numel(), 3, device=self.device) - 1.0) * bias_max
        self.pos_hist.reset(env_ids)

    def checkpoint_state(self) -> dict:
        return {"odom": self.model.state_dict(), "odom_norm": self.norm.state_dict()}

    def load_checkpoint_state(self, ckpt: dict) -> None:
        self.model.load_state_dict(ckpt["odom"])
        self.norm.load_state_dict(ckpt["odom_norm"])

    # ------------------------------------------------------------------ per-step
    def build_obs(self, proprio_delayed: torch.Tensor, cmd_vel: torch.Tensor, loc_actions: torch.Tensor) -> torch.Tensor:
        """Per-frame obs = 30 (delayed proprio) + 3 (cmd) + 12 (loc action). Newest frame at the end."""
        frame = torch.cat([proprio_delayed, cmd_vel, loc_actions], dim=-1)
        self.obs_hist = torch.roll(self.obs_hist, -1, 1)
        self.obs_hist[:, -1, :] = frame
        return self.obs_hist.reshape(self.N, -1)

    def predict(self, obs: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        """Inference, or one online training step if ``train_enabled`` and a target [N, 8] is given."""
        if self.train_enabled:
            self.norm.update(obs)
        x = self.norm(obs)
        td = lambda t: TensorDict({"proprio": t}, batch_size=t.shape[:1])

        if not (self.train_enabled and target is not None):
            with torch.no_grad():
                self.model.eval()
                pred = self.model(td(x))
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            return pred.to(self.device).detach()

        with torch.inference_mode(False), torch.enable_grad():
            x, target = x.detach().clone(), target.detach().clone()
            self.model.train()
            pred = self.model(td(x))
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = pred.to(self.device)
            loss, parts = odom_loss(pred, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.detach_hidden_state()
            self.model.eval()
        self.last_log = {"odom/loss_total": loss.detach(), **{k: v.detach() for k, v in parts.items()}}
        return pred.detach()

    def dead_reckon(self, gyro: torch.Tensor) -> None:
        """Integrate the network output into pose_est / pose_dr. gyro [N,3]: delayed body-frame gyro (no bias)."""
        roll, pitch = self.out[:, 0], self.out[:, 1]
        vel_b = self.out[:, 2:5]
        gyro = gyro + self.gyro_bias
        cp = torch.cos(pitch).clamp(min=0.2)
        yaw_rate = (torch.sin(roll) * gyro[:, 1] + torch.cos(roll) * gyro[:, 2]) / cp
        # trapezoidal rule: displacement over the last interval uses the mean of previous and current estimate
        v_mid = 0.5 * (self._prev_vel_b + vel_b)
        w_mid = 0.5 * (self._prev_yaw_rate + yaw_rate)
        prev_xy = self.pose_est[:, :2].clone()
        self.pose_est.copy_(integrate_pose(self.pose_est, roll, pitch, v_mid, w_mid, self.dt))
        self.pose_dr.copy_(integrate_pose(self.pose_dr, roll, pitch, v_mid, w_mid, self.dt))
        self.drift_dist += torch.linalg.norm(self.pose_est[:, :2] - prev_xy, dim=-1)
        self._prev_vel_b.copy_(vel_b)
        self._prev_yaw_rate.copy_(yaw_rate)

    def step(self, proprio_delayed, cmd_vel, loc_actions, target: torch.Tensor | None = None) -> torch.Tensor:
        """obs -> network -> dead-reckoning -> pose history push. ``target`` only for online training (sim)."""
        obs = self.build_obs(proprio_delayed, cmd_vel, loc_actions)
        self.out.copy_(self.predict(obs, target))
        self.dead_reckon(proprio_delayed[:, 0:3])
        self.pose_est_hist = torch.roll(self.pose_est_hist, 1, dims=1)
        self.pose_est_hist[:, 0] = self.pose_est
        return self.out

    # ------------------------------------------------------------------ pose access / correction
    def pose_at_lag(self, lag: torch.Tensor | None) -> torch.Tensor:
        """Pose estimate valid when a (delayed) sensor sample was captured. lag [N] long (control steps) or None."""
        if lag is None:
            return self.pose_est_hist[:, 0]
        return self.pose_est_hist[torch.arange(self.N, device=self.device), lag]

    @torch.no_grad()
    def apply_correction(self, corr: torch.Tensor, accepted: torch.Tensor) -> None:
        """Apply the same rigid nudge (dx, dy, dz, dyaw) to the estimate, its history and the position histories."""
        self.pose_est += corr
        self.pose_est[:, 3] = wrap_to_pi(self.pose_est[:, 3])
        self.pose_est_hist += corr[:, None, :]
        self.pose_est_hist[..., 3] = wrap_to_pi(self.pose_est_hist[..., 3])
        self.pos_hist.shift(corr[:, 0:3])
        self.drift_dist = torch.where(accepted, torch.zeros_like(self.drift_dist), self.drift_dist)