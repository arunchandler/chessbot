"""Evaluation helpers (lightweight baseline)."""

from __future__ import annotations

from typing import Dict

import chess

from chessbot.interfaces.agent import MCTSAgent, RandomAgent
from chessbot.mcts.evaluator import MaterialEvaluator
from chessbot.chess import game


def eval_vs_material(
    agent: MCTSAgent,
    games: int = 2,
    iterations_baseline: int = 200,
) -> Dict[str, float]:
    """
    Play a few games versus a material-eval MCTS baseline and return W/D/L rates.
    """
    results = {"win": 0, "draw": 0, "loss": 0}
    baseline = MCTSAgent(iterations=iterations_baseline, evaluator=MaterialEvaluator())
    for i in range(games):
        board = game.new_game()
        white = agent if i % 2 == 0 else baseline
        black = baseline if i % 2 == 0 else agent
        while not game.is_terminal(board):
            actor = white if board.turn == chess.WHITE else black
            move = actor.choose_move(board)
            game.push_move(board, move)
        res = game.result(board)
        if res == "1-0":
            results["win" if white is agent else "loss"] += 1
        elif res == "0-1":
            results["loss" if white is agent else "win"] += 1
        else:
            results["draw"] += 1
    total = float(games)
    return {k: v / total for k, v in results.items()}


__all__ = ["eval_vs_material"]
