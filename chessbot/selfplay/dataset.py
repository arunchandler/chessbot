"""PyTorch Dataset wrapper for self-play examples."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

Example = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (x, pi, z)


class SelfPlayDataset(Dataset[Example]):
    def __init__(self, examples: Iterable[Example]) -> None:
        self.examples: List[Example] = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_dataloader(
    examples: Iterable[Example],
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader[Example]:
    dataset = SelfPlayDataset(examples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
