"""Train the policy/value network on saved self-play examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chessbot.nn.model import ChessNet
from chessbot.nn.optim import make_optimizer, make_scheduler
from chessbot.selfplay.dataset import make_dataloader
from chessbot.selfplay.buffer import ReplayBuffer
from chessbot.train import train as train_mod
from chessbot.train import checkpoints
from chessbot.train import logging as log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train chessbot policy/value network.")
    p.add_argument("--data", type=str, default=None, help="Path to torch-saved examples (list of (x, pi, z)).")
    p.add_argument(
        "--buffer",
        type=str,
        default=None,
        help="Path to a replay buffer pickle produced by selfplay (--buffer-path).",
    )
    p.add_argument(
        "--buffer-samples",
        type=int,
        default=0,
        help="Number of examples to draw at random from the buffer (0 = use all).",
    )
    p.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--device", type=str, default="cpu", help="Training device (cpu or cuda).")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to save checkpoint.")
    p.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.buffer:
        buffer = ReplayBuffer.load(args.buffer)
        if len(buffer) == 0:
            raise ValueError(f"Replay buffer {args.buffer} is empty.")
        n = len(buffer) if args.buffer_samples <= 0 else min(len(buffer), args.buffer_samples)
        examples = buffer.sample(n)
        print(f"Loaded replay buffer {args.buffer} with {len(buffer)} examples; sampled {n} for training.")
        if args.data:
            print(f"[info] Ignoring --data {args.data} because --buffer was provided.")
    elif args.data:
        examples = torch.load(args.data, map_location="cpu")
        print(f"Loaded {len(examples)} examples from {args.data}.")
    else:
        raise ValueError("Provide either --data or --buffer.")

    dl = make_dataloader(examples, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device != "cpu"))

    model = ChessNet()
    model.to(args.device)
    opt = make_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(opt, milestones=None)

    start_step = 0
    if args.resume:
        meta = checkpoints.load_checkpoint(args.resume, model=model, optimizer=opt, map_location=args.device)
        start_step = meta.get("step", 0)
        print(f"Resumed from {args.resume} at step {start_step}")

    step = start_step
    for epoch in range(args.epochs):
        metrics = train_mod.train_epoch(model, opt, dl, device=args.device)
        step += len(dl)
        print(f"Epoch {epoch+1}/{args.epochs}: {log.format_metrics(metrics)}")
        if scheduler:
            scheduler.step()

    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        checkpoints.save_checkpoint(args.checkpoint, model=model, optimizer=opt, step=step)
        print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
