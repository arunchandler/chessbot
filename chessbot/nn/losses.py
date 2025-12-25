"""Loss helpers for policy/value training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_loss(logits: torch.Tensor, target_pi: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy with soft targets (pi is a probability distribution).

    logits: [B, ACTION_DIM]
    target_pi: [B, ACTION_DIM], sums to 1 along dim=1
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_pi * log_probs).sum(dim=1)
    return loss.mean()


def value_loss(pred_value: torch.Tensor, target_z: torch.Tensor, *, huber_delta: float | None = None) -> torch.Tensor:
    """
    Regression loss for value head.

    pred_value: [B, 1]
    target_z: [B, 1] or [B], values in [-1, 1]
    """
    target_z = target_z.view_as(pred_value)
    if huber_delta is None:
        return F.mse_loss(pred_value, target_z)
    return F.huber_loss(pred_value, target_z, delta=huber_delta)


__all__ = ["policy_loss", "value_loss"]
