"""Lightweight board evaluation helpers for MCTS rollouts."""

from __future__ import annotations

import chess

from chessbot.chess import utils

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

# Maximum material a side can have with the base piece values.
MAX_MATERIAL = 39.0  # 8P + 2N + 2B + 2R + 1Q


def material_evaluation(board: chess.Board, root_player: chess.Color) -> float:
    """
    Return a normalized material score in [-1, 1] from the root player's perspective.

    +1: root player is overwhelmingly ahead in material
    0:  material balance is equal
    -1: root player is far behind in material
    """
    if utils.is_terminal(board):
        return utils.terminal_value_from_root(board, root_player)

    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0.0)
        score += value if piece.color == root_player else -value

    normalized = score / MAX_MATERIAL
    # Promotions or captures can exceed nominal bounds; clamp to [-1, 1].
    return max(-1.0, min(1.0, normalized))
