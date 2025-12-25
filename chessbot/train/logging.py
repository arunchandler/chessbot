"""Tiny logging helpers."""

from __future__ import annotations

from typing import Dict


def format_metrics(metrics: Dict[str, float]) -> str:
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return ", ".join(parts)


def log_metrics(metrics: Dict[str, float]) -> None:
    print(format_metrics(metrics))


__all__ = ["format_metrics", "log_metrics"]
