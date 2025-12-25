"""Self-play episode generation using MCTS + (optional) neural evaluator."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import chess
import torch

from chessbot.mcts import search as mcts_search
from chessbot.mcts import evaluator as mcts_evaluator
from chessbot.nn import encode
from chessbot.nn.model import ChessNet

Example = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (x, pi, z)


def _policy_from_visits(
    visits_by_move: dict[chess.Move, int], temperature: float
) -> Tuple[torch.Tensor, Sequence[chess.Move], torch.Tensor]:
    moves = list(visits_by_move.keys())
    counts = torch.tensor([visits_by_move[m] for m in moves], dtype=torch.float32)
    if temperature <= 0:
        pi_moves = torch.zeros_like(counts)
        if counts.sum() == 0:
            pi_moves.fill_(1.0 / len(moves))
        else:
            best = torch.argmax(counts)
            pi_moves[best] = 1.0
    else:
        adjusted = torch.pow(counts, 1.0 / temperature)
        if adjusted.sum() == 0:
            pi_moves = torch.full_like(adjusted, 1.0 / len(moves))
        else:
            pi_moves = adjusted / adjusted.sum()
    full_pi = torch.zeros(encode.ACTION_DIM, dtype=torch.float32)
    for move, prob in zip(moves, pi_moves):
        full_pi[encode.move_to_index(move)] = prob
    return full_pi, moves, pi_moves


def generate_self_play_game(
    *,
    model: Optional[ChessNet] = None,
    device: str = "cpu",
    iterations: int = 400,
    c: float = 1.5,
    temperature: float = 1.0,
    temperature_moves: int = 20,
) -> List[Example]:
    """
    Play a self-play game using MCTS and return training examples.

    Returns a list of (x, pi, z) where:
    - x: encoded board planes [C,8,8]
    - pi: policy target over ACTION_DIM
    - z: outcome from the perspective of the player to move at that position
    """
    board = chess.Board()
    if model is not None:
        evaluator = mcts_evaluator.NeuralNetEvaluator(model, device=device)
    else:
        evaluator = mcts_evaluator.MaterialEvaluator()

    examples: List[Tuple[torch.Tensor, torch.Tensor, chess.Color]] = []

    move_idx = 0
    while not board.is_game_over(claim_draw=True):
        root = mcts_search.execute_search(board, iterations=iterations, c=c, evaluator=evaluator)

        visits = {move: child.visits for move, child in root.children.items()}
        if not visits:
            break

        pi, moves, move_probs = _policy_from_visits(
            visits, temperature if move_idx < temperature_moves else 0.0
        )

        x = encode.encode_board(board)
        to_move = board.turn
        examples.append((x, pi, to_move))

        # Choose move: sample early, argmax later.
        if move_idx < temperature_moves and temperature > 0:
            move_idx_sample = torch.multinomial(move_probs, 1).item()
            move = moves[move_idx_sample]
        else:
            move = max(moves, key=lambda m: visits[m])

        board.push(move)
        move_idx += 1

    # Final outcome
    outcome = board.outcome(claim_draw=True)
    winner = None if outcome is None else outcome.winner

    final_examples: List[Example] = []
    for x, pi, player in examples:
        if winner is None:
            z = torch.tensor([0.0], dtype=torch.float32)
        else:
            z = torch.tensor([1.0 if winner == player else -1.0], dtype=torch.float32)
        final_examples.append((x, pi, z))

    return final_examples
