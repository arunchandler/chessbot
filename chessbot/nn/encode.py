"""Encoding helpers for policy/action space and board features."""

from __future__ import annotations

import chess
import torch

# Order of promotion pieces in the action space.
PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

NON_PROMO_DIM = 64 * 64  # from-square x to-square without promotion
PROMO_DIM = 64 * 64 * len(PROMOTION_PIECES)
ACTION_DIM = NON_PROMO_DIM + PROMO_DIM

# Plane layout (white perspective):
# 0-5:  white P, N, B, R, Q, K
# 6-11: black P, N, B, R, Q, K
# 12:   side to move (all ones if white to move else zeros)
# 13-16: castling rights planes (WK, WQ, BK, BQ), constant ones if available
# 17:   en passant target square (single 1 if available)
PLANE_COUNT = 18

PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


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


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode the board into planes (C, 8, 8), white perspective.

    Plane order:
    - 12 piece planes (white then black, P/N/B/R/Q/K)
    - 1 side-to-move plane (ones if white to move)
    - 4 castling planes (WK, WQ, BK, BQ) as constant ones if right is available
    - 1 en passant plane (single 1 at the target square, else zeros)
    """
    planes = torch.zeros((PLANE_COUNT, 8, 8), dtype=torch.float32)

    # Pieces
    for square, piece in board.piece_map().items():
        plane_idx = PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        row = 7 - rank  # rank 7 (8th rank) is row 0 to keep A8 at top
        col = file
        planes[plane_idx, row, col] = 1.0

    # Side to move
    if board.turn == chess.WHITE:
        planes[12].fill_(1.0)

    # Castling rights
    castling_planes = [
        (13, board.has_kingside_castling_rights(chess.WHITE)),
        (14, board.has_queenside_castling_rights(chess.WHITE)),
        (15, board.has_kingside_castling_rights(chess.BLACK)),
        (16, board.has_queenside_castling_rights(chess.BLACK)),
    ]
    for plane_idx, has_right in castling_planes:
        if has_right:
            planes[plane_idx].fill_(1.0)

    # En passant
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        row = 7 - rank
        col = file
        planes[17, row, col] = 1.0

    return planes


__all__ = [
    "ACTION_DIM",
    "PROMOTION_PIECES",
    "PLANE_COUNT",
    "move_to_index",
    "index_to_move",
    "legal_moves_mask",
    "encode_board",
]
