"""Automate self-play → buffer update → training → eval gating."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import chess
import torch

from chessbot.interfaces.agent import MaterialMCTSAgent, NeuralMCTSAgent, MCTSAgent
from chessbot.nn.model import ChessNet
from chessbot.nn.optim import make_optimizer
from chessbot.selfplay.buffer import ReplayBuffer
from chessbot.selfplay.generate import generate_self_play_game
from chessbot.selfplay.dataset import make_dataloader
from chessbot.train import train as train_mod
from chessbot.train import checkpoints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run iterative self-play/train/eval with gating.")
    p.add_argument("--iterations", type=int, default=400, help="MCTS iterations for self-play and eval.")
    p.add_argument("--cpuct", type=float, default=1.5, help="PUCT exploration constant.")
    p.add_argument("--games", type=int, default=10, help="Self-play games per iteration.")
    p.add_argument("--temperature", type=float, default=1.0, help="Opening temperature for self-play.")
    p.add_argument("--temperature-final", type=float, default=0.0, help="Late-move temperature for self-play.")
    p.add_argument("--temperature-moves", type=int, default=20, help="Plies to sample before argmax.")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for root noise.")
    p.add_argument("--dirichlet-frac", type=float, default=0.25, help="Dirichlet mix fraction (epsilon).")
    p.add_argument("--buffer-path", type=str, default="replay.pkl", help="Replay buffer path.")
    p.add_argument("--buffer-capacity", type=int, default=200_000, help="Replay buffer capacity if created.")
    p.add_argument("--buffer-samples", type=int, default=50_000, help="Samples drawn for training (0=all).")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs per iteration.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    p.add_argument("--train-device", type=str, default="cpu", help="Device for training (cpu|cuda).")
    p.add_argument("--model-device", type=str, default="cpu", help="Device for self-play/eval nets.")
    p.add_argument("--best-checkpoint", type=str, default="checkpoints/best.pt", help="Path to best checkpoint.")
    p.add_argument("--candidate-checkpoint", type=str, default="checkpoints/candidate.pt", help="Where to save candidate.")
    p.add_argument("--eval-games", type=int, default=20, help="Games for gating eval.")
    p.add_argument("--gating-threshold", type=float, default=0.55, help="Min win rate vs best to accept candidate.")
    p.add_argument("--iterations-count", type=int, default=1, help="How many loop iterations to run.")
    return p.parse_args()


def load_model(path: str, device: str) -> ChessNet:
    model = ChessNet()
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def make_agent(model_path: Optional[str], iterations: int, c: float, device: str) -> MCTSAgent:
    if model_path and Path(model_path).exists():
        return NeuralMCTSAgent(model_path=model_path, iterations=iterations, c=c, device=device)
    return MaterialMCTSAgent(iterations=iterations, c=c)


def play_match(agent_a: MCTSAgent, agent_b: MCTSAgent, games: int) -> dict[str, float]:
    """Return win/draw/loss rates for agent_a vs agent_b, alternating colors."""
    results = {"win": 0, "draw": 0, "loss": 0}
    for i in range(games):
        board = chess.Board()
        white = agent_a if i % 2 == 0 else agent_b
        black = agent_b if i % 2 == 0 else agent_a
        agents = {chess.WHITE: white, chess.BLACK: black}
        while not board.is_game_over(claim_draw=True):
            actor = agents[board.turn]
            move = actor.choose_move(board)
            board.push(move)
        res = board.result(claim_draw=True)
        if res == "1-0":
            results["win" if white is agent_a else "loss"] += 1
        elif res == "0-1":
            results["loss" if white is agent_a else "win"] += 1
        else:
            results["draw"] += 1
    total = float(games)
    return {k: v / total for k, v in results.items()}


def run_iteration(args: argparse.Namespace, buffer: ReplayBuffer, best_path: Path) -> None:
    # Prepare best agent for self-play.
    best_exists = best_path.exists()
    best_agent = make_agent(str(best_path) if best_exists else None, args.iterations, args.cpuct, args.model_device)
    best_model: Optional[ChessNet] = load_model(str(best_path), args.model_device) if best_exists else None

    # Self-play
    total_positions = 0
    for i in range(args.games):
        examples = generate_self_play_game(
            model=best_model,
            device=args.model_device,
            iterations=args.iterations,
            c=args.cpuct,
            temperature=args.temperature,
            temperature_final=args.temperature_final,
            temperature_moves=args.temperature_moves,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_frac=args.dirichlet_frac,
        )
        buffer.add(examples)
        total_positions += len(examples)
        print(f"[selfplay] Game {i+1}/{args.games} -> {len(examples)} positions (buffer size {len(buffer)})")
    buffer.save(args.buffer_path)
    print(f"[selfplay] Added {total_positions} positions; buffer saved to {args.buffer_path}")

    # Training data
    num_samples = len(buffer) if args.buffer_samples <= 0 else min(len(buffer), args.buffer_samples)
    train_examples = buffer.sample(num_samples)
    dl = make_dataloader(train_examples, batch_size=args.batch_size, shuffle=True, pin_memory=(args.train_device != "cpu"))

    # Candidate model starting from best (if any)
    candidate = ChessNet()
    if best_exists:
        candidate.load_state_dict(load_model(str(best_path), args.train_device).state_dict())
        print(f"[train] Initialized candidate from {best_path}")
    candidate.to(args.train_device)

    opt = make_optimizer(candidate.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        metrics = train_mod.train_epoch(candidate, opt, dl, device=args.train_device)
        print(f"[train] Epoch {epoch+1}/{args.epochs}: {metrics}")

    # Save candidate
    cand_path = Path(args.candidate_checkpoint)
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoints.save_checkpoint(str(cand_path), model=candidate, optimizer=opt, step=args.epochs * len(dl))
    print(f"[train] Saved candidate checkpoint to {cand_path}")

    # Gating eval vs current best
    candidate_agent = NeuralMCTSAgent(
        model_path=str(cand_path),
        iterations=args.iterations,
        c=args.cpuct,
        device=args.model_device,
    )
    if best_exists:
        ref_agent = best_agent
    else:
        # If no best yet, auto-accept after training.
        ref_agent = MaterialMCTSAgent(iterations=args.iterations, c=args.cpuct)

    eval_results = play_match(candidate_agent, ref_agent, args.eval_games)
    win_rate = eval_results["win"]
    print(f"[eval] Candidate vs best: W/D/L = {eval_results}, win_rate={win_rate:.2%}")

    if (not best_exists) or win_rate >= args.gating_threshold:
        best_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.save_checkpoint(str(best_path), model=candidate, optimizer=opt, step=args.epochs * len(dl))
        print(f"[gating] Candidate ACCEPTED (saved to {best_path})")
    else:
        print(f"[gating] Candidate REJECTED (threshold={args.gating_threshold:.2%})")


def main() -> None:
    args = parse_args()

    buf_path = Path(args.buffer_path)
    if buf_path.exists():
        buffer = ReplayBuffer.load(buf_path)
        print(f"[buffer] Loaded {len(buffer)} examples from {buf_path} (capacity={buffer.capacity})")
    else:
        buffer = ReplayBuffer(capacity=args.buffer_capacity)
        buf_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[buffer] Created new buffer at {buf_path} (capacity={buffer.capacity})")

    best_path = Path(args.best_checkpoint)

    for it in range(args.iterations_count):
        print(f"\n=== Iteration {it+1}/{args.iterations_count} ===")
        run_iteration(args, buffer, best_path)


if __name__ == "__main__":
    main()
