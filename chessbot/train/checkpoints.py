"""Checkpoint save/load helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
        "extra": extra,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | None = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if model is not None:
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        model.load_state_dict(state)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {
        "step": checkpoint.get("step", 0),
        "config": checkpoint.get("config"),
        "extra": checkpoint.get("extra"),
    }


__all__ = ["save_checkpoint", "load_checkpoint"]
