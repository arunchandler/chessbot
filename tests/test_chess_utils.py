import random

import pytest

chess = pytest.importorskip("chess")

from chessbot.chess import utils


def test_fen_roundtrip() -> None:
    board = chess.Board()
    fen = board.fen()
    loaded = chess.Board(fen)
    assert len(utils.legal_moves(board)) == len(utils.legal_moves(loaded))


def test_random_play_does_not_crash() -> None:
    board = chess.Board()
    for _ in range(200):
        if utils.is_terminal(board):
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)

    assert len(board.move_stack) <= 200


def test_terminal_value_draw_stalemate() -> None:
    # Black to move; no legal moves but not in check.
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert utils.is_terminal(board)
    assert utils.terminal_value(board) == 0.0
