import pytest

chess = pytest.importorskip("chess")
torch = pytest.importorskip("torch")

from chessbot.nn import encode


def test_legal_moves_unique_indices() -> None:
    board = chess.Board()
    legal = list(board.legal_moves)
    indices = [encode.move_to_index(m) for m in legal]
    assert len(indices) == len(set(indices))
    mask = encode.legal_moves_mask(board)
    assert mask.sum().item() == len(legal)
    for idx in indices:
        assert mask[idx]


def test_promotion_mapping_roundtrip() -> None:
    # White pawn ready to promote.
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    legal = list(board.legal_moves)
    promo_moves = [m for m in legal if m.promotion is not None]
    assert promo_moves, "Expected promotion moves"

    for move in promo_moves:
        idx = encode.move_to_index(move)
        roundtrip = encode.index_to_move(idx)
        assert roundtrip.uci() == move.uci()
        assert idx < encode.ACTION_DIM


def test_mask_marks_exact_legal_moves() -> None:
    board = chess.Board()
    legal_set = {encode.move_to_index(m) for m in board.legal_moves}
    mask = encode.legal_moves_mask(board)
    active = set(torch.nonzero(mask, as_tuple=False).view(-1).tolist())
    assert active == legal_set
