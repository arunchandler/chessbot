"""Simple human vs. bot CLI loop (Random or MCTS)."""

from __future__ import annotations

import argparse
from typing import Callable, Optional

import chess
import random

from chessbot.chess import game
from chessbot.interfaces import agent as agent_mod


def play(
    white_agent: agent_mod.Agent | None = None,
    black_agent: agent_mod.Agent | None = None,
    *,
    board: Optional[chess.Board] = None,
    input_fn: Callable[[str], str] = input,
) -> chess.Board:
    """Run a full game between two agents (human or bot)."""
    board = board or game.new_game()

    white = white_agent or agent_mod.HumanAgent(input_fn=input_fn)
    black = black_agent or agent_mod.RandomAgent()
    agents = {chess.WHITE: white, chess.BLACK: black}
    color_label = {chess.WHITE: "White", chess.BLACK: "Black"}

    while not game.is_terminal(board):
        game.print_board(board)

        mover = board.turn
        actor = agents[mover]
        mover_label = color_label[mover]
        actor_name = agent_mod.agent_name(actor)
        print(f"{mover_label} to move ({actor_name})")

        try:
            move = actor.choose_move(board)
        except agent_mod.QuitGame:
            print("Game aborted.")
            return board

        try:
            game.push_move(board, move)
        except ValueError as exc:
            print(f"[{actor_name}] illegal move: {exc}")
            continue

        print(f"{mover_label} played {move.uci()}\n")

    game.print_board(board)
    print(game.describe_outcome(board))
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Play chess vs. a bot.")
    parser.add_argument(
        "--bot",
        choices=["random", "mcts"],
        default="random",
        help="Bot type (random legal move or MCTS).",
    )
    parser.add_argument(
        "--color",
        choices=["white", "black"],
        default=None,
        help="Your color. If omitted, a color is chosen at random.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="MCTS iterations (only if --bot mcts).",
    )
    parser.add_argument(
        "--cpuct",
        type=float,
        default=1.5,
        help="Exploration constant for MCTS.",
    )
    parser.add_argument(
        "--rollout-depth",
        type=int,
        default=20,
        help="Rollout depth for MCTS default policy.",
    )

    args = parser.parse_args()

    if args.color is None:
        human_color = random.choice([chess.WHITE, chess.BLACK])  # type: ignore[name-defined]
    else:
        human_color = chess.WHITE if args.color == "white" else chess.BLACK

    if args.bot == "mcts":
        bot = agent_mod.MCTSAgent(
            iterations=args.iterations, c=args.cpuct, rollout_depth=args.rollout_depth
        )
    else:
        bot = agent_mod.RandomAgent()

    if human_color == chess.WHITE:
        play(white_agent=agent_mod.HumanAgent(), black_agent=bot)
    else:
        play(white_agent=bot, black_agent=agent_mod.HumanAgent())


if __name__ == "__main__":
    main()
