import random

import pytest

chess = pytest.importorskip("chess")

from chessbot.interfaces.agent import MCTSAgent


def test_agent_never_plays_illegal_move() -> None:
    board = chess.Board()
    agent = MCTSAgent(iterations=50, rng=random.Random(0))
    move = agent.choose_move(board)
    assert move in board.legal_moves


def test_agent_raises_on_terminal_board() -> None:
    # Fool's mate completed: Black is checkmated.
    board = chess.Board()
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(move)
    assert board.is_checkmate()

    agent = MCTSAgent(iterations=10)
    with pytest.raises(ValueError):
        agent.choose_move(board)


def test_mate_in_one_is_found() -> None:
    # White to move and mate in one: Qh7#.
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    agent = MCTSAgent(iterations=200, rng=random.Random(42))
    move = agent.choose_move(board)
    assert move == chess.Move.from_uci("d1h5")


def test_determinism_with_fixed_seed() -> None:
    board = chess.Board()
    rng = random.Random(123)
    agent = MCTSAgent(iterations=50, rng=rng)
    first_move = agent.choose_move(board)

    # Reset to the same seed and re-run.
    rng = random.Random(123)
    agent2 = MCTSAgent(iterations=50, rng=rng)
    second_move = agent2.choose_move(board)

    assert first_move == second_move
