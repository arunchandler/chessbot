"""Move encoding helpers for fixed-size policy vectors."""

from __future__ import annotations

import chess
import torch

# Order of promotion pieces in the action space.
PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

NON_PROMO_DIM = 64 * 64  # from-square x to-square without promotion
PROMO_DIM = 64 * 64 * len(PROMOTION_PIECES)
ACTION_DIM = NON_PROMO_DIM + PROMO_DIM


def move_to_index(move: chess.Move) -> int:
    """Map a chess.Move to a fixed action index."""
    idx = move.from_square * 64 + move.to_square
    if move.promotion is None:
        return idx

    try:
        promo_idx = PROMOTION_PIECES.index(move.promotion)
    except ValueError as exc:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}") from exc

    return NON_PROMO_DIM + (idx * len(PROMOTION_PIECES)) + promo_idx


def index_to_move(idx: int) -> chess.Move:
    """Inverse of move_to_index (does not check legality)."""
    if idx < 0 or idx >= ACTION_DIM:
        raise ValueError(f"Index {idx} out of range [0, {ACTION_DIM})")

    if idx < NON_PROMO_DIM:
        from_sq = idx // 64
        to_sq = idx % 64
        return chess.Move(from_sq, to_sq)

    promo_idx = (idx - NON_PROMO_DIM) % len(PROMOTION_PIECES)
    base = (idx - NON_PROMO_DIM) // len(PROMOTION_PIECES)
    from_sq = base // 64
    to_sq = base % 64
    promotion = PROMOTION_PIECES[promo_idx]
    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """Boolean mask over ACTION_DIM where legal moves are True."""
    mask = torch.zeros(ACTION_DIM, dtype=torch.bool)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = True
    return mask


__all__ = [
    "ACTION_DIM",
    "PROMOTION_PIECES",
    "move_to_index",
    "index_to_move",
    "legal_moves_mask",
]
