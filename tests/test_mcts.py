import pytest

chess = pytest.importorskip("chess")

from chessbot.interfaces.agent import MCTSAgent
from chessbot.mcts import evaluator as mcts_evaluator
from chessbot.chess import utils


class DummyEvaluator:
    def __init__(self, target_move: chess.Move | None = None) -> None:
        self.target_move = target_move

    def evaluate(self, board: chess.Board, root_player: chess.Color):
        legal = list(board.legal_moves)
        priors = {}
        total = 0.0
        for move in legal:
            weight = 10.0 if self.target_move and move == self.target_move else 1.0
            priors[move] = weight
            total += weight
        if total > 0:
            priors = {m: w / total for m, w in priors.items()}
        value = 0.0
        if utils.is_terminal(board):
            value = utils.terminal_value_from_root(board, root_player)
        return priors, value


def test_agent_never_plays_illegal_move() -> None:
    board = chess.Board()
    agent = MCTSAgent(iterations=50, evaluator=DummyEvaluator())
    move = agent.choose_move(board)
    assert move in board.legal_moves


def test_agent_raises_on_terminal_board() -> None:
    board = chess.Board()
    for move in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        board.push_uci(move)
    assert board.is_checkmate()

    agent = MCTSAgent(iterations=10, evaluator=DummyEvaluator())
    with pytest.raises(ValueError):
        agent.choose_move(board)


def test_mate_in_one_is_found() -> None:
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    target = chess.Move.from_uci("d1h5")
    agent = MCTSAgent(iterations=200, evaluator=DummyEvaluator(target_move=target))
    move = agent.choose_move(board)
    assert move == target


def test_determinism_with_fixed_evaluator() -> None:
    board = chess.Board()
    agent1 = MCTSAgent(iterations=50, evaluator=DummyEvaluator())
    move1 = agent1.choose_move(board)

    agent2 = MCTSAgent(iterations=50, evaluator=DummyEvaluator())
    move2 = agent2.choose_move(board)

    assert move1 == move2
