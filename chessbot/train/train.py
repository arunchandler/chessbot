"""Lightweight training helpers for policy/value network."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from chessbot.nn import losses


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Iterable[torch.Tensor],
    *,
    device: str = "cpu",
    huber_delta: Optional[float] = None,
) -> Dict[str, float]:
    """
    Run a single optimization step.

    batch: (x, pi, z)
      - x: [B, C, 8, 8]
      - pi: [B, ACTION_DIM]
      - z: [B, 1] or [B]
    """
    model.train()
    x, pi, z = batch
    x = x.to(device)
    pi = pi.to(device)
    z = z.to(device).view(-1, 1)

    optimizer.zero_grad()
    policy_logits, value_pred = model(x)

    p_loss = losses.policy_loss(policy_logits, pi)
    v_loss = losses.value_loss(value_pred, z, huber_delta=huber_delta)
    loss = p_loss + v_loss

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        # value accuracy: sign agreement (excluding zeros in target)
        sign_target = torch.sign(z)
        sign_pred = torch.sign(value_pred)
        mask = sign_target != 0
        if mask.any():
            value_acc = (sign_target[mask] == sign_pred[mask]).float().mean().item()
        else:
            value_acc = 0.0
        entropy = -(torch.softmax(policy_logits, dim=1) * torch.log_softmax(policy_logits, dim=1)).sum(
            dim=1
        ).mean().item()

    return {
        "loss": loss.item(),
        "policy_loss": p_loss.item(),
        "value_loss": v_loss.item(),
        "value_acc": value_acc,
        "policy_entropy": entropy,
    }


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: str = "cpu",
    huber_delta: Optional[float] = None,
) -> Dict[str, float]:
    """Train for one epoch over the dataloader and return averaged metrics."""
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "value_acc": 0.0, "policy_entropy": 0.0}
    count = 0
    for batch in dataloader:
        metrics = train_step(model, optimizer, batch, device=device, huber_delta=huber_delta)
        for k in totals:
            totals[k] += metrics[k]
        count += 1
    if count:
        for k in totals:
            totals[k] /= count
    return totals


__all__ = ["train_step", "train_epoch"]
