"""Basic MCTS loop (selection, expansion, evaluation, backpropagation)."""

from __future__ import annotations

import math
from typing import Optional

import chess

from chessbot.chess import utils
from chessbot.mcts import evaluator as mcts_eval
from chessbot.mcts.node import Node


def puct_score(parent: Node, child: Node, c: float) -> float:
    prior = parent.priors.get(child.move, 0.0) if child.move is not None else 0.0
    q = child.q_value
    u = c * prior * math.sqrt(parent.visits) / (1 + child.visits)
    return q + u


def select_child_puct(node: Node, c: float) -> Node:
    return max(node.children.values(), key=lambda child: puct_score(node, child, c))


def tree_policy(node: Node, c: float) -> Node:
    """
    Selection: descend using PUCT until a node with untried moves, zero visits, or terminal.
    """
    current = node
    while True:
        if current.is_terminal:
            return current
        if current.visits == 0 or current.untried_moves:
            return current
        if not current.children:
            return current
        current = select_child_puct(current, c)


def backpropagate(node: Node, value_from_root: float) -> None:
    current: Optional[Node] = node
    while current is not None:
        current.record(value_from_root)
        current = current.parent


def run_search(
    root_board: chess.Board,
    *,
    num_simulations: int = 200,
    c: float = 1.5,
    evaluator: Optional[mcts_eval.EvaluatorProtocol] = None,
) -> Node:
    """
    Run MCTS starting from root_board and return the root node with stats populated.

    Values are stored from the root player's perspective.
    """
    root = Node(board=root_board.copy(stack=True))
    eval_fn = evaluator or mcts_eval.MaterialEvaluator()

    for _ in range(num_simulations):
        leaf = tree_policy(root, c)

        if leaf.is_terminal:
            value = utils.terminal_value_from_root(leaf.board, root.root_player)  # type: ignore[arg-type]
            backpropagate(leaf, value)
            continue

        # Evaluate leaf if first visit to set priors.
        if leaf.visits == 0 or not leaf.priors:
            priors, value = eval_fn.evaluate(leaf.board, root.root_player)  # type: ignore[arg-type]
            leaf.priors = priors
            leaf.untried_moves = list(priors.keys())
            backpropagate(leaf, value)
            continue

        # Expansion
        move = max(leaf.untried_moves, key=lambda m: leaf.priors.get(m, 0.0))
        child = leaf.expand(move)

        if child.is_terminal:
            value = utils.terminal_value_from_root(child.board, root.root_player)  # type: ignore[arg-type]
            backpropagate(child, value)
            continue

        priors, value = eval_fn.evaluate(child.board, root.root_player)  # type: ignore[arg-type]
        child.priors = priors
        child.untried_moves = list(priors.keys())

        backpropagate(child, value)

    return root


def execute_search(
    board: chess.Board,
    *,
    iterations: int = 200,
    c: float = 1.5,
    evaluator: Optional[mcts_eval.EvaluatorProtocol] = None,
) -> Node:
    """Alias around run_search using 'iterations' terminology."""
    return run_search(
        board,
        num_simulations=iterations,
        c=c,
        evaluator=evaluator,
    )
