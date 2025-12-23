"""Terminal detection helpers shared by MCTS/NN code."""

from __future__ import annotations

import chess
from typing import Dict, Iterable, List, Mapping


def is_terminal(board: chess.Board) -> bool:
    """Return True if the position is finished (checkmate, stalemate, draw, etc.)."""
    return board.is_game_over(claim_draw=True)


def terminal_value(board: chess.Board) -> float:
    """
    Return the game outcome from the side-to-move perspective.

    Convention:
    -1.0 if the side to move has lost (e.g., is checkmated)
     0.0 for any draw
    +1.0 if, unusually, the side to move has already won
    Raises ValueError if the game is not over.
    """
    if not is_terminal(board):
        raise ValueError("Board is not in a terminal state.")

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.0

    winner_color = chess.WHITE if outcome.winner else chess.BLACK
    return 1.0 if board.turn == winner_color else -1.0


# --- Policy target helpers ------------------------------------------------- #
PolicyTarget = Dict[chess.Move, int]


def legal_moves(board: chess.Board) -> List[chess.Move]:
    """
    Deterministically enumerate legal moves for the given board.

    Sorted by UCI string to guarantee stable ordering regardless of python-chess
    generator order, which keeps downstream encoding reproducible.
    """
    return sorted(board.legal_moves, key=lambda m: m.uci())


def empty_policy(board: chess.Board) -> PolicyTarget:
    """Create an empty policy target dict keyed by legal moves with zero counts."""
    return {move: 0 for move in legal_moves(board)}


def normalize_policy(counts: Mapping[chess.Move, int]) -> Dict[chess.Move, float]:
    """
    Convert visit counts to probabilities (sums to 1.0). Returns zeros if no counts.
    """
    total = float(sum(counts.values()))
    if total <= 0:
        return {move: 0.0 for move in counts}
    return {move: value / total for move, value in counts.items()}


def increment_policy(counts: PolicyTarget, move: chess.Move, increment: int = 1) -> None:
    """Bump the visit count for a move, adding the key if needed."""
    counts[move] = counts.get(move, 0) + increment
