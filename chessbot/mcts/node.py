"""Lightweight MCTS node representation (no search logic yet).

We treat `chess.Board` as the game state. Nodes store a board copy for simplicity.
If we later optimize for speed, we can switch traversal to push/pop on a shared
board while keeping the node data the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import chess

from chessbot.chess import utils


@dataclass
class Node:
    parent: Optional["Node"] = None
    move: Optional[chess.Move] = None  # move used to reach this node from its parent
    root_player: Optional[chess.Color] = None
    children: Dict[chess.Move, "Node"] = field(default_factory=dict)
    untried_moves: list[chess.Move] = field(default_factory=list)
    priors: Dict[chess.Move, float] = field(default_factory=dict)  # P(a) from parent
    visits: int = 0
    value_sum: float = 0.0  # accumulated value from root player's perspective
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Mean value (root perspective)."""
        return self.value_sum / self.visits if self.visits else 0.0

    def record(self, value_from_root: float) -> None:
        """
        Backpropagate a rollout/terminal value (root player's perspective).

        Callers are responsible for converting terminal results to the root
        perspective, e.g., via utils.terminal_value_from_root(...).
        """
        self.visits += 1
        self.value_sum += value_from_root

    @classmethod
    def from_board(
        cls,
        board: chess.Board,
        *,
        parent: Optional["Node"] = None,
        move: Optional[chess.Move] = None,
        root_player: Optional[chess.Color] = None,
    ) -> "Node":
        node = cls(
            parent=parent,
            move=move,
            root_player=root_player if root_player is not None else board.turn,
        )
        node.untried_moves = utils.legal_moves(board)
        node.is_terminal = utils.is_terminal(board)
        return node
