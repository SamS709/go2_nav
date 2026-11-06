"""Losses / metrics used to train the odometry and map networks online. Not needed for deployment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def odom_loss(pred: torch.Tensor, target: torch.Tensor):
    """pred / target [N, 8] = (roll, pitch, v_b(3), w_b(3))."""
    rp = F.mse_loss(pred[:, 0:2], target[:, 0:2])
    lin = F.mse_loss(pred[:, 2:5], target[:, 2:5])
    ang = F.mse_loss(pred[:, 5:8], target[:, 5:8])
    return rp + lin + 0.5 * ang, {"odom/loss_roll_pitch": rp, "odom/loss_lin_vel": lin, "odom/loss_ang_vel": ang}


def map_losses(pred_h, logvar, crops: dict, w_complete: float, w_unseen: float):
    obs, seen, inb, target = crops["observed"], crops["seen"], crops["inb"], crops["target"]
    w = obs.float() + (seen & ~obs).float() * w_complete + (~seen & ~obs).float() * w_unseen
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
    return loss_h + 0.5 * loss_e + 0.1 * loss_n, loss_h, loss_e, loss_n


@torch.no_grad()
def map_metrics(crops: dict, pred_h: torch.Tensor) -> dict:
    """Robot-relative height errors (they include z drift)."""
    obs_m = (crops["observed"] & crops["inb"]).float()
    unobs_seen = (~crops["observed"] & crops["seen"] & crops["inb"]).float()
    target = crops["target"]
    raw_err = (crops["x"][:, 0] - target).abs()
    pred_err = (pred_h - target).abs()
    return {
        "map/raw_mae_observed": (raw_err * obs_m).sum() / obs_m.sum().clamp(min=1.0),
        "map/pred_mae_observed": (pred_err * obs_m).sum() / obs_m.sum().clamp(min=1.0),
        "map/pred_mae_seen_unobserved": (pred_err * unobs_seen).sum() / unobs_seen.sum().clamp(min=1.0),
    }