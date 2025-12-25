"""Minimal UCI loop to expose the bot to GUIs."""

from __future__ import annotations

import sys
from typing import Optional

import chess

from chessbot.interfaces import agent as agent_mod
from chessbot.chess import game as game_mod


def _handle_position(cmd_parts: list[str], board: chess.Board) -> None:
    if len(cmd_parts) < 2:
        return
    if cmd_parts[1] == "startpos":
        board.reset()
        moves_idx = cmd_parts.index("moves") + 1 if "moves" in cmd_parts else len(cmd_parts)
        moves = cmd_parts[moves_idx:]
    elif cmd_parts[1] == "fen":
        # position fen <fen> [moves ...]
        try:
            fen_end = cmd_parts.index("moves")
            fen_parts = cmd_parts[2:fen_end]
            moves = cmd_parts[fen_end + 1 :]
        except ValueError:
            fen_parts = cmd_parts[2:]
            moves = []
        fen = " ".join(fen_parts)
        board.set_fen(fen)
    else:
        return
    for mv in moves:
        board.push_uci(mv)


def run_uci_loop(bot: agent_mod.Agent, *, stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    board = chess.Board()
    name = f"chessbot-{agent_mod.agent_name(bot)}"

    while True:
        line = stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]

        if cmd == "uci":
            print(f"id name {name}", file=stdout)
            print("id author you", file=stdout)
            print("uciok", file=stdout)
            stdout.flush()
        elif cmd == "isready":
            print("readyok", file=stdout)
            stdout.flush()
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            _handle_position(parts, board)
        elif cmd == "go":
            try:
                move = bot.choose_move(board)
                board.push(move)
                print(f"bestmove {move.uci()}", file=stdout)
            except Exception as exc:
                print(f"info string error {exc}", file=stdout)
                print("bestmove 0000", file=stdout)
            stdout.flush()
        elif cmd == "quit":
            break
        elif cmd == "stop":
            continue


__all__ = ["run_uci_loop"]
