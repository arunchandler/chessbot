"""Utility functions and a lightweight play loop for the chess bot."""

from __future__ import annotations

import random
from typing import Callable, Optional, Protocol

import chess

from . import utils

Color = chess.Color
TurnCallback = Callable[[chess.Board, Color], None]
MoveCallback = Callable[[chess.Board, Color, chess.Move], None]


class AgentProtocol(Protocol):
    def choose_move(self, board: chess.Board) -> chess.Move:
        ...


def new_game() -> chess.Board:
    """Return a fresh chess board in the starting position."""
    return chess.Board()


def print_board(board: chess.Board) -> None:
    """Print an ASCII board plus a short status line."""
    print(board)  # Default string repr is ASCII-friendly.
    turn = "White" if board.turn == chess.WHITE else "Black"
    print(f"Move {board.fullmove_number} - {turn} to move (halfmove clock: {board.halfmove_clock})\n")


def parse_move(board: chess.Board, raw: str) -> chess.Move:
    """
    Parse a move string into a legal `chess.Move`.

    Accepts SAN (e.g., "Nf3") or UCI (e.g., "e2e4"). Raises ValueError if the
    move is not understood or not legal in the current position.
    """
    text = raw.strip()
    if not text:
        raise ValueError("Move cannot be empty.")

    move: Optional[chess.Move] = None

    # Try SAN first for nicer ergonomics, then fall back to raw UCI.
    try:
        move = board.parse_san(text)
    except ValueError:
        try:
            move = chess.Move.from_uci(text)
        except ValueError as exc:
            raise ValueError(f"Could not parse move: {text}") from exc

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move in this position: {text}")

    return move


def push_move(board: chess.Board, move: chess.Move) -> chess.Board:
    """Push a move onto the board after re-validating legality."""
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {move}")
    board.push(move)
    return board


def push(board: chess.Board, move: chess.Move) -> chess.Board:
    """Alias for push_move to mirror the python-chess API."""
    return push_move(board, move)


def undo_last_move(board: chess.Board) -> chess.Board:
    """Undo the last move; raises ValueError if there is nothing to undo."""
    if not board.move_stack:
        raise ValueError("No moves to undo.")
    board.pop()
    return board


def random_move(board: chess.Board) -> chess.Move:
    """Pick a random legal move (useful as a simple baseline bot)."""
    return random.choice(list(board.legal_moves))


def play_game(
    white_agent: AgentProtocol,
    black_agent: AgentProtocol,
    *,
    board: Optional[chess.Board] = None,
    on_turn: Optional[TurnCallback] = None,
    on_move: Optional[MoveCallback] = None,
) -> chess.Board:
    """
    Play a game between two agents until termination.

    Optional callbacks:
    - on_turn(board, color): called before an agent moves
    - on_move(board, color, move): called after a move is pushed
    """
    board = board or new_game()
    agents = {chess.WHITE: white_agent, chess.BLACK: black_agent}

    while not is_terminal(board):
        mover = board.turn
        if on_turn is not None:
            on_turn(board, mover)

        move = agents[mover].choose_move(board)
        push_move(board, move)

        if on_move is not None:
            on_move(board, mover, move)

    return board


def is_terminal(board: chess.Board) -> bool:
    """Return True if the game has ended by checkmate/stalemate/insufficient/etc."""
    return utils.is_terminal(board)


def result(board: chess.Board) -> str:
    """
    Return the PGN-style result string.

    "*": game still in progress, "1-0"/"0-1"/"1/2-1/2" otherwise.
    """
    return board.result() if board.is_game_over() else "*"


def describe_outcome(board: chess.Board) -> str:
    """Human-friendly description of the game outcome."""
    outcome = board.outcome()
    if outcome is None:
        return "Game in progress."

    if outcome.termination == chess.Termination.CHECKMATE:
        winner = "White" if outcome.winner else "Black"
        return f"Checkmate - {winner} wins ({board.result()})"

    if outcome.termination == chess.Termination.STALEMATE:
        return "Stalemate (draw)."

    if outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
        return "Draw by insufficient material."

    if outcome.termination == chess.Termination.FIVEFOLD_REPETITION:
        return "Draw by fivefold repetition."

    if outcome.termination == chess.Termination.SEVENTYFIVE_MOVES:
        return "Draw by seventy-five-move rule."

    if outcome.termination == chess.Termination.THREEFOLD_REPETITION:
        return "Draw by threefold repetition."

    if outcome.termination == chess.Termination.FIFTY_MOVES:
        return "Draw by fifty-move rule."

    return f"Game over ({board.result()})"


def play_human_vs_random(
    human_color: Color = chess.WHITE,
    *,
    input_fn: Callable[[str], str] = input,
    board: Optional[chess.Board] = None,
) -> chess.Board:
    """
    Run a simple human vs. random-move bot loop using the agent interface.

    - Moves can be entered as SAN or UCI.
    - Type 'quit' or 'exit' to stop early.
    """
    from chessbot.interfaces import agent as agent_mod

    board = board or new_game()

    bot_color = chess.BLACK if human_color == chess.WHITE else chess.WHITE
    agents = {
        human_color: agent_mod.HumanAgent(input_fn=input_fn),
        bot_color: agent_mod.RandomAgent(),
    }
    color_label = {chess.WHITE: "White", chess.BLACK: "Black"}

    print(f"You are {color_label[human_color]}. Bot plays {color_label[bot_color]}.")
    print("Enter moves as SAN (e.g., Nf3) or UCI (e.g., e2e4). Type 'quit' to exit.\n")

    def on_turn(current_board: chess.Board, color: Color) -> None:
        print_board(current_board)
        print(f"{color_label[color]} to move ({agent_mod.agent_name(agents[color])})")

    def on_move(current_board: chess.Board, color: Color, move: chess.Move) -> None:
        print(f"{color_label[color]} plays {move.uci()}\n")

    try:
        final_board = play_game(
            agents[chess.WHITE],
            agents[chess.BLACK],
            board=board,
            on_turn=on_turn,
            on_move=on_move,
        )
    except agent_mod.QuitGame:
        print("Game aborted.")
        return board

    print_board(final_board)
    print(describe_outcome(final_board))
    return final_board


def main() -> None:
    play_human_vs_random()


if __name__ == "__main__":
    main()
