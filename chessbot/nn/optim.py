"""Optimizer and LR schedule helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.optim as optim


def make_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> optim.Optimizer:
    """Create an Adam optimizer with optional weight decay."""
    return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def make_scheduler(
    optimizer: optim.Optimizer,
    *,
    milestones: Optional[list[int]] = None,
    gamma: float = 0.5,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Optional step scheduler. If milestones is None/empty, returns None.
    """
    if not milestones:
        return None
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


__all__ = ["make_optimizer", "make_scheduler"]
