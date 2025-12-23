"""Agent interface and simple implementations."""

from __future__ import annotations

import random
from typing import Callable, Protocol

import chess

from chessbot.chess import game


class QuitGame(Exception):
    """Signal that the user requested to end the game early."""


class Agent(Protocol):
    name: str

    def choose_move(self, board: chess.Board) -> chess.Move:
        ...


def agent_name(agent: Agent) -> str:
    """Return a human-friendly agent label."""
    return getattr(agent, "name", agent.__class__.__name__)


class RandomAgent:
    """Choose a random legal move."""

    name = "Random"

    def choose_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("No legal moves available.")
        return random.choice(moves)


class HumanAgent:
    """Prompt the user for a move via stdin."""

    name = "Human"

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] = input,
        prompt: str = "Your move (UCI or 'undo'/'fen'/'quit'): ",
    ) -> None:
        self.input_fn = input_fn
        self.prompt = prompt

    def choose_move(self, board: chess.Board) -> chess.Move:
        while True:
            raw = self.input_fn(self.prompt).strip()
            lowered = raw.lower()

            if lowered in {"quit", "exit"}:
                raise QuitGame()

            if lowered == "fen":
                print(board.fen())
                continue

            if lowered == "undo":
                try:
                    game.undo_last_move(board)
                    print("Undid last move.")
                except ValueError as exc:
                    print(f"[undo] {exc}")
                continue

            try:
                return game.parse_move(board, raw)
            except ValueError as exc:
                print(f"[invalid] {exc}")
