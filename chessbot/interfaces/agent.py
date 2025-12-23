"""Agent interface and simple implementations."""

from __future__ import annotations

import random
from typing import Callable, Optional, Protocol

import chess

from chessbot.chess import game
from chessbot.mcts import search as mcts_search

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
                continue


class MCTSAgent:
    """Run MCTS for a fixed number of iterations and choose the most visited move."""

    name = "MCTS"

    def __init__(
        self,
        *,
        iterations: int = 300,
        c: float = 1.5,
        rollout_depth: int = 20,
        rollout_fn: Optional[mcts_search.RolloutFn] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.iterations = iterations
        self.c = c
        self.rollout_depth = rollout_depth
        self.rollout_fn = rollout_fn
        self.rng = rng

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise ValueError("Cannot choose move: board is terminal.")

        root = mcts_search(
            board,
            iterations=self.iterations,
            c=self.c,
            rollout_depth=self.rollout_depth,
            rollout_fn=self.rollout_fn,
            rng=self.rng,
        )

        if not root.children:
            raise ValueError("No moves available after search.")

        best_move, _ = max(root.children.items(), key=lambda item: item[1].visits)
        return best_move
