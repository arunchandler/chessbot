"""Play a batch of games vs. a random bot and report win/draw/loss."""

from __future__ import annotations

import argparse

import chess

from chessbot.interfaces.agent import MaterialMCTSAgent, NeuralMCTSAgent, RandomAgent
from chessbot.chess import game


def play_match(bot, games: int) -> dict[str, int]:
    results = {"win": 0, "draw": 0, "loss": 0}
    for i in range(games):
        board = game.new_game()
        bot_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        rand_color = chess.BLACK if bot_color == chess.WHITE else chess.WHITE
        agents = {bot_color: bot, rand_color: RandomAgent()}
        while not game.is_terminal(board):
            actor = agents[board.turn]
            move = actor.choose_move(board)
            game.push_move(board, move)
        res = game.result(board)
        if (res == "1-0" and bot_color == chess.WHITE) or (res == "0-1" and bot_color == chess.BLACK):
            results["win"] += 1
        elif (res == "0-1" and bot_color == chess.WHITE) or (res == "1-0" and bot_color == chess.BLACK):
            results["loss"] += 1
        else:
            results["draw"] += 1
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bot vs random agent.")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play.")
    parser.add_argument("--iterations", type=int, default=300, help="MCTS iterations for the bot.")
    parser.add_argument("--cpuct", type=float, default=1.5, help="PUCT exploration constant.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional NN checkpoint for the bot.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for NN evaluator (cpu/cuda).")
    args = parser.parse_args()

    if args.model_path:
        bot = NeuralMCTSAgent(
            model_path=args.model_path,
            iterations=args.iterations,
            c=args.cpuct,
            device=args.device,
        )
    else:
        bot = MaterialMCTSAgent(iterations=args.iterations, c=args.cpuct)

    results = play_match(bot, args.games)
    total = float(args.games)
    print(
        f"W/D/L: {results['win']}/{results['draw']}/{results['loss']} "
        f"({results['win']/total:.2%} / {results['draw']/total:.2%} / {results['loss']/total:.2%})"
    )


if __name__ == "__main__":
    main()
