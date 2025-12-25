"""Simple replay buffer for self-play examples."""

from __future__ import annotations

import pickle
from collections import deque
from typing import Deque, Iterable, List, Tuple

import torch

Example = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (x, pi, z)


class ReplayBuffer:
    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Example] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, examples: Iterable[Example]) -> None:
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, n: int) -> List[Example]:
        if n > len(self.buffer):
            raise ValueError("Not enough samples in buffer.")
        # deterministic slice from the end for simplicity; can switch to random.sample
        return list(list(self.buffer)[-n:])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "examples": list(self.buffer),
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "ReplayBuffer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        buf = cls(capacity=data["capacity"])
        buf.add(data["examples"])
        return buf
