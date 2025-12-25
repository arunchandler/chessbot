"""Lightweight AlphaZero-style network with policy and value heads."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessbot.nn import encode


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    def __init__(
        self,
        channels: int = 128,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(
            encode.PLANE_COUNT, channels, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, encode.ACTION_DIM)

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """
        x: [B, C, 8, 8] float tensor
        Returns: (policy_logits [B, ACTION_DIM], value [B, 1] in [-1,1])
        """
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


__all__ = ["ChessNet"]
