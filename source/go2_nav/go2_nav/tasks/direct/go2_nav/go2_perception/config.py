"""Parameters of the perception stack.

``PerceptionParams.from_cfg`` reads them from the Isaac Lab env cfg (same names as before, optional entries keep their
defaults). On the robot just build the dataclasses by hand / from a yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class OdomParams:
    length_odom_hist: int
    length_short_term: int
    length_long_term: int
    freq_pos_short_term: float
    freq_pos_long_term: float
    pose_hist_len: int = 1  # delay_length + 1 when sensor delay is emulated, else 1
    frame_dim: int = 45  # 30 proprio + 3 cmd + 12 loc actions
    gyro_bias_max: float = 0.01  # rad/s, random per-episode gyro bias (sim randomisation)


@dataclass
class MapParams:
    cell_size: float
    width: float
    height: float
    crop_cells: int = 64
    scan_noise: float = 0.01  # m, z noise on the emulated scan (sim randomisation)
    scan_max_range: float = 6.0
    meas_std: float = 0.05
    min_variance: float = 1e-4
    observed_var_thresh: float = 1.0
    drift_var_gain: float = 0.02
    conf_ref_std: float = 0.1
    w_complete: float = 0.5  # loss weights (training only)
    w_unseen: float = 0.05


@dataclass
class MatchParams:
    enabled: bool = True
    period: int = 2
    xy_range: float = 0.2
    xy_steps: int = 5
    yaw_range_deg: float = 3.0
    yaw_steps: int = 5
    max_points: int = 256
    chunk: int = 128
    var_thresh: float = 0.05
    min_points: int = 40
    err_clip: float = 9.0
    improve_thresh: float = 0.3
    axis_obs_thresh: float = 0.5
    gain: float = 0.5
    gain_z: float = 0.5


@dataclass
class TrainParams:
    train_perception: bool = True  # False -> odom and map models frozen (inference only)
    odom_lr: float = 1e-3
    map_lr: float = 1e-3
    checkpoint: str | None = None


@dataclass
class PerceptionParams:
    odom: OdomParams
    map: MapParams
    match: MatchParams = field(default_factory=MatchParams)
    train: TrainParams = field(default_factory=TrainParams)

    @classmethod
    def from_cfg(cls, cfg) -> "PerceptionParams":
        def opt(dc, mapping: dict[str, str] | None = None, **required):
            """Instantiate dataclass ``dc``: optional fields come from cfg (under their old names), else default."""
            mapping = mapping or {}
            kw = dict(required)
            for f in fields(dc):
                if f.name in kw:
                    continue
                old = mapping.get(f.name, f.name)
                if hasattr(cfg, old):
                    kw[f.name] = getattr(cfg, old)
            return dc(**kw)

        delay_len = int(cfg.delay_length)
        odom = opt(
            OdomParams,
            length_odom_hist=int(cfg.length_odom_hist),
            length_short_term=int(cfg.length_short_term),
            length_long_term=int(cfg.length_long_term),
            freq_pos_short_term=float(cfg.freq_pos_short_term),
            freq_pos_long_term=float(cfg.freq_pos_long_term),
            pose_hist_len=delay_len + 1 if cfg.delay else 1,
        )
        map_names = {k: f"map_{k}" for k in (
            "crop_cells", "scan_noise", "scan_max_range", "meas_std", "min_variance", "observed_var_thresh",
            "drift_var_gain", "conf_ref_std", "w_complete", "w_unseen")}
        mp = opt(
            MapParams, map_names,
            cell_size=float(cfg.map_cell_size), width=float(cfg.map_width), height=float(cfg.map_height),
        )
        match_names = {
            "enabled": "scan_match",
            **{k: f"match_{k}" for k in (
                "period", "xy_range", "xy_steps", "yaw_range_deg", "yaw_steps", "max_points", "chunk",
                "var_thresh", "min_points", "err_clip", "improve_thresh", "axis_obs_thresh", "gain", "gain_z")},
        }
        match = opt(MatchParams, match_names)
        train = opt(TrainParams, {"checkpoint": "perception_checkpoint"})
        return cls(odom=odom, map=mp, match=match, train=train)