"""Basic MCTS loop (selection, expansion, evaluation, backpropagation)."""

from __future__ import annotations

import math
from typing import Optional

import chess
import torch

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


def tree_policy(node: Node, board: chess.Board, c: float) -> Node:
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
        if current.move is not None:
            board.push(current.move)


def backpropagate(node: Node, value_from_root: float) -> None:
    current: Optional[Node] = node
    while current is not None:
        current.record(value_from_root)
        current = current.parent


def _add_dirichlet_noise(
    priors: dict[chess.Move, float],
    alpha: float,
    frac: float,
) -> dict[chess.Move, float]:
    """
    Mix Dirichlet noise into root priors to encourage exploration.

    New priors = (1 - frac) * priors + frac * Dir(alpha)
    """
    if not priors:
        return priors

    frac = max(0.0, min(1.0, frac))
    moves = list(priors.keys())
    noise = torch.distributions.Dirichlet(torch.full((len(moves),), alpha)).sample()

    mixed: dict[chess.Move, float] = {}
    for move, eta in zip(moves, noise):
        mixed_prob = (1.0 - frac) * priors.get(move, 0.0) + frac * float(eta)
        mixed[move] = mixed_prob

    total = sum(mixed.values())
    if total > 0:
        mixed = {m: p / total for m, p in mixed.items()}
    return mixed


def run_search(
    root_board: chess.Board,
    *,
    num_simulations: int = 200,
    c: float = 1.5,
    evaluator: Optional[mcts_eval.EvaluatorProtocol] = None,
    root_dirichlet_alpha: float | None = None,
    root_dirichlet_frac: float = 0.25,
) -> Node:
    """
    Run MCTS starting from root_board and return the root node with stats populated.

    Values are stored from the root player's perspective.
    """
    root = Node.from_board(root_board, root_player=root_board.turn)
    eval_fn = evaluator or mcts_eval.MaterialEvaluator()
    root_noise_applied = False
    # Reuse a single board via push/pop to avoid per-node copies.
    playout_board = root_board.copy(stack=True)
    root_stack_len = len(playout_board.move_stack)

    for _ in range(num_simulations):
        # Reset board to root state (pop back to initial stack length).
        while len(playout_board.move_stack) > root_stack_len:
            playout_board.pop()

        leaf = tree_policy(root, playout_board, c)

        if leaf.is_terminal:
            value = utils.terminal_value_from_root(playout_board, root.root_player)  # type: ignore[arg-type]
            backpropagate(leaf, value)
            continue

        # Evaluate leaf if first visit to set priors.
        if leaf.visits == 0 or not leaf.priors:
            priors, value = eval_fn.evaluate(playout_board, root.root_player)  # type: ignore[arg-type]
            if (
                leaf is root
                and root_dirichlet_alpha is not None
                and not root_noise_applied
                and priors
            ):
                priors = _add_dirichlet_noise(priors, root_dirichlet_alpha, root_dirichlet_frac)
                root_noise_applied = True
            leaf.priors = priors
            leaf.untried_moves = list(priors.keys())
            backpropagate(leaf, value)
            continue

        # Expansion
        move = max(leaf.untried_moves, key=lambda m: leaf.priors.get(m, 0.0))
        playout_board.push(move)
        child = Node.from_board(playout_board, parent=leaf, move=move, root_player=leaf.root_player)
        leaf.children[move] = child
        if move in leaf.untried_moves:
            leaf.untried_moves.remove(move)

        if child.is_terminal:
            value = utils.terminal_value_from_root(playout_board, root.root_player)  # type: ignore[arg-type]
            backpropagate(child, value)
            continue

        priors, value = eval_fn.evaluate(playout_board, root.root_player)  # type: ignore[arg-type]
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
    root_dirichlet_alpha: float | None = None,
    root_dirichlet_frac: float = 0.25,
) -> Node:
    """Alias around run_search using 'iterations' terminology."""
    return run_search(
        board,
        num_simulations=iterations,
        c=c,
        evaluator=evaluator,
        root_dirichlet_alpha=root_dirichlet_alpha,
        root_dirichlet_frac=root_dirichlet_frac,
    )
