"""Simple human vs. random-bot CLI loop."""

from __future__ import annotations

from typing import Callable, Optional

import chess

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
    play()


if __name__ == "__main__":
    main()
