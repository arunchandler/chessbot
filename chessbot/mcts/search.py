"""Basic MCTS loop (selection, expansion, simulation, backpropagation)."""

from __future__ import annotations

import math
import random
from typing import Callable, Optional

import chess

from chessbot.chess import utils
from chessbot.mcts import evaluator as mcts_eval
from chessbot.mcts.node import Node

RolloutFn = Callable[[chess.Board, chess.Color, int, Optional[random.Random]], float]
RNG = random.Random


def uct_score(parent: Node, child: Node, c: float) -> float:
    if child.visits == 0:
        return float("inf")
    exploitation = child.q_value
    exploration = c * math.sqrt(math.log(parent.visits) / child.visits)
    return exploitation + exploration


def select_child_ucb(node: Node, c: float) -> Node:
    return max(node.children.values(), key=lambda child: uct_score(node, child, c))


def tree_policy(node: Node, c: float, *, rng: Optional[RNG] = None) -> Node:
    """Selection + expansion: descend until a node with untried moves or terminal."""
    current = node
    choice = rng.choice if rng is not None else random.choice
    while not current.is_terminal:
        if current.untried_moves:
            move = choice(current.untried_moves)
            return current.expand(move)
        current = select_child_ucb(current, c)
    return current


def default_rollout(
    board: chess.Board,
    root_player: chess.Color,
    depth_limit: int,
    rng: Optional[RNG] = None,
) -> float:
    """Play random moves until terminal or depth limit; return root-perspective value."""
    rollout_board = board.copy(stack=True)
    choice = rng.choice if rng is not None else random.choice
    for _ in range(depth_limit):
        if utils.is_terminal(rollout_board):
            break
        move = choice(list(rollout_board.legal_moves))
        rollout_board.push(move)

    if utils.is_terminal(rollout_board):
        return utils.terminal_value_from_root(rollout_board, root_player)

    # Non-terminal after depth limit: fall back to a lightweight evaluation.
    return mcts_eval.material_evaluation(rollout_board, root_player)


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
    rollout_depth: int = 200,
    rollout_fn: Optional[RolloutFn] = None,
    rng: Optional[RNG] = None,
) -> Node:
    """
    Run MCTS starting from root_board and return the root node with stats populated.

    Values are stored from the root player's perspective.
    """
    root = Node(board=root_board.copy(stack=True))
    rollout = rollout_fn or default_rollout
    choice_rng = rng

    for _ in range(num_simulations):
        leaf = tree_policy(root, c, rng=choice_rng)

        if leaf.is_terminal:
            value = utils.terminal_value_from_root(leaf.board, root.root_player)  # type: ignore[arg-type]
        else:
            value = rollout(leaf.board, root.root_player, rollout_depth, rng=choice_rng)  # type: ignore[arg-type]

        backpropagate(leaf, value)

    return root


def search(
    board: chess.Board,
    *,
    iterations: int = 200,
    c: float = 1.5,
    rollout_depth: int = 200,
    rollout_fn: Optional[RolloutFn] = None,
    rng: Optional[RNG] = None,
) -> Node:
    """Alias around run_search using 'iterations' terminology."""
    return run_search(
        board,
        num_simulations=iterations,
        c=c,
        rollout_depth=rollout_depth,
        rollout_fn=rollout_fn,
        rng=rng,
    )
