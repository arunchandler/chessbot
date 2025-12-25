"""UCI engine entrypoint."""

from __future__ import annotations

import argparse

from chessbot.interfaces.agent import MaterialMCTSAgent, NeuralMCTSAgent
from chessbot.interfaces.uci import run_uci_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chessbot as a UCI engine.")
    parser.add_argument("--iterations", type=int, default=400, help="MCTS iterations per move.")
    parser.add_argument("--cpuct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to a NN checkpoint; if absent uses material evaluator.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the neural evaluator (cpu/cuda).",
    )
    args = parser.parse_args()

    if args.model_path:
        bot = NeuralMCTSAgent(
            model_path=args.model_path,
            iterations=args.iterations,
            c=args.cpuct,
            device=args.device,
        )
    else:
        bot = MaterialMCTSAgent(iterations=args.iterations, c=args.cpuct)

    run_uci_loop(bot)


if __name__ == "__main__":
    main()
