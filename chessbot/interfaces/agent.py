"""Agent interface and simple implementations."""

from __future__ import annotations

import random
from typing import Callable, Optional, Protocol

import chess
import torch

from chessbot.chess import game
from chessbot.mcts import evaluator as mcts_evaluator
from chessbot.mcts import search as mcts_search
from chessbot.nn.model import ChessNet


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
        evaluator: Optional[mcts_evaluator.EvaluatorProtocol] = None,
    ) -> None:
        self.iterations = iterations
        self.c = c
        self.evaluator = evaluator

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise ValueError("Cannot choose move: board is terminal.")

        root = mcts_search.execute_search(
            board,
            iterations=self.iterations,
            c=self.c,
            evaluator=self.evaluator,
        )

        if not root.children:
            raise ValueError("No moves available after search.")

        best_move, _ = max(root.children.items(), key=lambda item: item[1].visits)
        return best_move


class MaterialMCTSAgent(MCTSAgent):
    """MCTS using material evaluator (no NN)."""

    name = "MaterialMCTS"

    def __init__(self, *, iterations: int = 300, c: float = 1.5) -> None:
        super().__init__(iterations=iterations, c=c, evaluator=mcts_evaluator.MaterialEvaluator())


class NeuralMCTSAgent(MCTSAgent):
    """MCTS using a neural network evaluator loaded from a checkpoint."""

    name = "NeuralMCTS"

    def __init__(
        self,
        *,
        model_path: str,
        iterations: int = 300,
        c: float = 1.5,
        device: str = "cpu",
    ) -> None:
        model = ChessNet()
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        evaluator = mcts_evaluator.NeuralNetEvaluator(model, device=device)
        super().__init__(iterations=iterations, c=c, evaluator=evaluator)
