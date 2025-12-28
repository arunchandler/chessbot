"""Generate self-play episodes with MCTS + (optional) NN evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from chessbot.nn.model import ChessNet
from chessbot.selfplay.generate import generate_self_play_game
from chessbot.selfplay.buffer import ReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play games for training.")
    parser.add_argument("--games", type=int, default=1, help="Number of self-play games to generate.")
    parser.add_argument("--iterations", type=int, default=400, help="MCTS iterations per move.")
    parser.add_argument("--cpuct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Move sampling temperature for early moves.",
    )
    parser.add_argument(
        "--temperature-moves",
        type=int,
        default=20,
        help="Number of initial plies to sample; afterwards use argmax.",
    )
    parser.add_argument(
        "--temperature-final",
        type=float,
        default=0.0,
        help="Temperature after --temperature-moves plies (0.0 = argmax).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to a torch checkpoint for the neural evaluator.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the neural evaluator (cpu or cuda).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save a torch file of collected examples.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet concentration parameter for root exploration noise.",
    )
    parser.add_argument(
        "--dirichlet-frac",
        type=float,
        default=0.25,
        help="Fraction of root prior replaced by Dirichlet noise (aka epsilon).",
    )
    parser.add_argument(
        "--buffer-path",
        type=str,
        default=None,
        help="Optional path to maintain a replay buffer (appends new examples and saves).",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=200_000,
        help="Capacity for a new replay buffer if --buffer-path does not exist.",
    )
    args = parser.parse_args()

    model: Optional[ChessNet] = None
    if args.model_path:
        model = ChessNet()
        state = torch.load(args.model_path, map_location=args.device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    all_examples = []
    for i in range(args.games):
        examples = generate_self_play_game(
            model=model,
            device=args.device,
            iterations=args.iterations,
            c=args.cpuct,
            temperature=args.temperature,
            temperature_final=args.temperature_final,
            temperature_moves=args.temperature_moves,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_frac=args.dirichlet_frac,
        )
        all_examples.extend(examples)
        print(f"Game {i+1}/{args.games} complete: {len(examples)} positions.")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_examples, out_path)
        print(f"Saved {len(all_examples)} examples to {out_path}")
    else:
        print(f"Generated {len(all_examples)} examples (not saved).")

    if args.buffer_path:
        buf_path = Path(args.buffer_path)
        if buf_path.exists():
            buffer = ReplayBuffer.load(buf_path)
            print(f"Loaded replay buffer from {buf_path} (size={len(buffer)}, capacity={buffer.capacity})")
        else:
            buffer = ReplayBuffer(capacity=args.buffer_capacity)
            buf_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Created new replay buffer at {buf_path} (capacity={buffer.capacity})")

        buffer.add(all_examples)
        buffer.save(buf_path)
        print(f"Buffer now has {len(buffer)} examples (saved to {buf_path})")


if __name__ == "__main__":
    main()
