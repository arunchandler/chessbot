"""Evaluation helpers for MCTS rollouts and neural priors."""

from __future__ import annotations

from typing import Dict, Tuple

import chess
import torch

from chessbot.chess import utils
from chessbot.nn import encode

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}

# Maximum material a side can have with the base piece values.
MAX_MATERIAL = 39.0  # 8P + 2N + 2B + 2R + 1Q


class EvaluatorProtocol:
    def evaluate(self, board: chess.Board, root_player: chess.Color) -> Tuple[Dict[chess.Move, float], float]:
        ...


class MaterialEvaluator:
    """Uniform priors over legal moves + material-based value."""

    def evaluate(self, board: chess.Board, root_player: chess.Color) -> Tuple[Dict[chess.Move, float], float]:
        legal = list(board.legal_moves)
        if not legal:
            priors: Dict[chess.Move, float] = {}
        else:
            p = 1.0 / len(legal)
            priors = {m: p for m in legal}
        value = material_evaluation(board, root_player)
        return priors, value


class NeuralNetEvaluator:
    """Use a policy/value network to produce priors and value."""

    def __init__(self, model: torch.nn.Module, device: torch.device | str | None = None) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, board: chess.Board, root_player: chess.Color) -> Tuple[Dict[chess.Move, float], float]:
        if utils.is_terminal(board):
            priors = {m: 0.0 for m in board.legal_moves}
            value = utils.terminal_value_from_root(board, root_player)
            return priors, value

        inp = encode.encode_board(board).unsqueeze(0).to(self.device)  # [1, C, 8, 8]
        logits, value = self.model(inp)
        logits = logits.squeeze(0)
        mask = encode.legal_moves_mask(board).to(self.device)
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        priors_tensor = torch.softmax(masked_logits, dim=0)

        priors: Dict[chess.Move, float] = {}
        for move in board.legal_moves:
            idx = encode.move_to_index(move)
            priors[move] = float(priors_tensor[idx].item())

        val = float(value.squeeze().item())
        # Convert to root perspective if model outputs from side-to-move perspective.
        if board.turn != root_player:
            val = -val

        return priors, val


def material_evaluation(board: chess.Board, root_player: chess.Color) -> float:
    """
    Return a normalized material score in [-1, 1] from the root player's perspective.

    +1: root player is overwhelmingly ahead in material
    0:  material balance is equal
    -1: root player is far behind in material
    """
    if utils.is_terminal(board):
        return utils.terminal_value_from_root(board, root_player)

    score = 0.0
    for piece in board.piece_map().values():
        value = PIECE_VALUES.get(piece.piece_type, 0.0)
        score += value if piece.color == root_player else -value

    normalized = score / MAX_MATERIAL
    # Promotions or captures can exceed nominal bounds; clamp to [-1, 1].
    return max(-1.0, min(1.0, normalized))


__all__ = ["EvaluatorProtocol", "MaterialEvaluator", "NeuralNetEvaluator", "material_evaluation"]
