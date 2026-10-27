# SPDX-License-Identifier: GPL-3.0-or-later
"""Original bounded mate-1..3 corpus construction with a pinned CPU legal oracle.

Stockfish supplies legal moves, check status and transitions through UCI ``d`` /
``go perft 1``; Mouse's independent finite AND/OR proof supplies mate distance.
No engine search scores, puzzle collection, or production Mouse search is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol, TypeVar

from mouse_tools.perft_reference import (
    ARCHIVE_URL,
    REFERENCE_NAME,
    REFERENCE_SHA256,
    RELEASE_URL,
    SOURCE_COMMIT,
    ReferenceError,
    parse_output,
    validate_fen,
)

SEED = 20261017
MOVE = re.compile(r"[a-h][1-8][a-h][1-8]")
T = TypeVar("T")


class MateError(ValueError):
    """Invalid input, incomplete proof, or untrusted oracle response."""


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _board(fen: str) -> dict[int, str]:
    board: dict[int, str] = {}
    for rank, row in enumerate(fen.split()[0].split("/")):
        file = 0
        for piece in row:
            if piece.isdecimal():
                file += int(piece)
            else:
                board[(7 - rank) * 8 + file] = piece
                file += 1
    return board


def _fen(board: dict[int, str], side: str, halfmove: int = 0, fullmove: int = 1) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            piece = board.get(rank * 8 + file)
            if piece is None:
                empty += 1
            else:
                row += (str(empty) if empty else "") + piece
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows) + f" {side} - - {halfmove} {fullmove}"


def _attacked(board: dict[int, str], square: int, by_white: bool) -> bool:
    x, y = square % 8, square // 8
    for source, piece in board.items():
        if piece.isupper() != by_white:
            continue
        dx, dy = x - source % 8, y - source // 8
        ax, ay = abs(dx), abs(dy)
        kind = piece.upper()
        if kind == "K" and max(ax, ay) == 1:
            return True
        if kind == "N" and {ax, ay} == {1, 2}:
            return True
        sliding = (kind in "RQ" and (dx == 0 or dy == 0)) or (kind in "BQ" and ax == ay)
        if not sliding or source == square:
            continue
        sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
        cx, cy = source % 8 + sx, source // 8 + sy
        blocked = False
        while (cx, cy) != (x, y):
            if cy * 8 + cx in board:
                blocked = True
                break
            cx, cy = cx + sx, cy + sy
        if not blocked:
            return True
    return False


def validate_position(fen: str) -> None:
    """The narrow, pawnless construction domain has no special-move ambiguity."""
    try:
        validate_fen(fen)
    except ReferenceError as error:
        raise MateError(str(error)) from error
    fields = fen.split()
    board = _board(fen)
    if fields[2:4] != ["-", "-"] or not 2 <= len(board) <= 6:
        raise MateError("Mate construction requires 2..6 pieces and no castling/en-passant")
    if any(piece not in "KQRBNkqrbn" for piece in board.values()):
        raise MateError("Mate construction supports pawnless king/major/minor pieces only")
    previous_king = "k" if fields[1] == "w" else "K"
    square = next(square for square, piece in board.items() if piece == previous_king)
    if _attacked(board, square, fields[1] == "w"):
        raise MateError("Non-side-to-move king is already in check")


def validate_root(fen: str) -> None:
    """Reserve rule-50 headroom at proof entry, not after reversible moves.

    The accepted construction domain keeps clock<=90 and proves at most five
    plies. Clocks are therefore irrelevant to all descendant proof identities;
    a reachable clock91 is valid, but a near-100 starting root is unsupported.
    """
    validate_position(fen)
    if int(fen.split()[4]) > 90:
        raise MateError("Mate proof root requires rule-50 clock <=90")


def _key(fen: str) -> str:
    # All proof horizons <=5 plies, initial clock <=90, no pawns/rights/EP.
    # Clocks cannot cause a rule draw in these proofs and do not affect legality.
    return " ".join(fen.split()[:4])


def _play(fen: str, move: str) -> str:
    if MOVE.fullmatch(move) is None:
        raise MateError("Expected a supported coordinate move")
    board = _board(fen)
    source = (int(move[1]) - 1) * 8 + ord(move[0]) - 97
    target = (int(move[3]) - 1) * 8 + ord(move[2]) - 97
    fields = fen.split()
    if source not in board or board[source].isupper() != (fields[1] == "w"):
        raise MateError("Oracle move has no mover of the active color")
    if target in board and (
        board[target].isupper() == board[source].isupper() or board[target].upper() == "K"
    ):
        raise MateError("Oracle move captures a friendly piece or king")
    capture = target in board
    board[target] = board.pop(source)
    return _fen(
        board,
        "b" if fields[1] == "w" else "w",
        0 if capture else int(fields[4]) + 1,
        int(fields[5]) + (fields[1] == "b"),
    )


@dataclass(frozen=True)
class State:
    fen: str
    side: str
    check: bool
    moves: tuple[str, ...]


def parse_state(lines: Sequence[str]) -> State:
    if any(
        re.match(r"^[a-z0-9]{4,5}:", line.strip())
        and re.fullmatch(r"[a-h][1-8][a-h][1-8]: 1", line.strip()) is None
        for line in lines
    ):
        raise MateError("Malformed depth-one divide line")
    try:
        result = parse_output(lines)
    except ReferenceError as error:
        raise MateError(str(error)) from error
    checks = [
        line.removeprefix("Checkers:").strip() for line in lines if line.startswith("Checkers:")
    ]
    if len(checks) != 1 or any(re.fullmatch(r"[a-h][1-8]", sq) is None for sq in checks[0].split()):
        raise MateError("Oracle must return exactly one valid Checkers field")
    if len(result.moves) > 218 or any(count != 1 for _, count in result.divide):
        raise MateError("Oracle depth-one divide must have unit counts and <=218 moves")
    if any(MOVE.fullmatch(move) is None for move in result.moves):
        raise MateError("Unsupported special move in pawnless mate oracle")
    return State(result.fen, result.fen.split()[1], bool(checks[0]), tuple(sorted(result.moves)))


class UciTransport:
    """One bounded reader thread; synchronous commands, fail-closed on protocol loss."""

    def __init__(self, command: Sequence[str], timeout: float = 10.0) -> None:
        if not command or not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise MateError("Oracle command and timeout (0,300] are required")
        self.command = tuple(command)
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self.reader: threading.Thread | None = None
        self.lines: queue.Queue[str | None] = queue.Queue(maxsize=2048)
        self.failed = threading.Event()
        self.closed = threading.Event()
        self.transcript: list[str] = []

    def _pump(self, stream: IO[str]) -> None:
        try:
            while not self.closed.is_set():
                line = stream.readline(4097)
                if not line:
                    self.lines.put_nowait(None)
                    return
                if len(line) > 4096 or not line.endswith("\n"):
                    self.failed.set()
                    return
                self.lines.put_nowait(line.rstrip("\r\n"))
        except (OSError, ValueError, UnicodeError, queue.Full):
            self.failed.set()

    def __enter__(self) -> UciTransport:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="ascii",
            errors="strict",
            bufsize=1,
        )
        if self.process.stdout is None:
            raise MateError("Oracle stdout was not connected")
        self.reader = threading.Thread(target=self._pump, args=(self.process.stdout,), daemon=True)
        self.reader.start()
        return self

    def exchange(self, command: str, marker: str) -> list[str]:
        if self.process is None or self.process.stdin is None:
            raise MateError("Oracle is not running")
        if "\r" in command or len(command) > 1024:
            raise MateError("Oracle command is invalid or too long")
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise MateError(f"Oracle write failed: {error}") from error
        deadline = time.monotonic() + self.timeout
        lines: list[str] = []
        while True:
            if self.failed.is_set():
                raise MateError("Oracle response exceeded bounds or was malformed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MateError(f"Oracle timeout awaiting {marker}")
            try:
                line = self.lines.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            if line is None:
                raise MateError(f"Oracle exited awaiting {marker}")
            lines.append(line)
            if len(lines) > 512:
                raise MateError("Oracle response exceeds 512 lines")
            if line == marker or line.startswith(marker + ":"):
                return lines

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.closed.set()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            if self.process.stdin:
                self.process.stdin.close()
            if self.reader:
                self.reader.join(timeout=5)
            if self.process.stdout:
                self.process.stdout.close()


class LegalOracle(Protocol):
    def state(self, fen: str, move: str | None = None) -> State: ...


def validate_greeting(lines: Sequence[str]) -> None:
    if lines.count("id name " + REFERENCE_NAME) != 1 or lines.count("uciok") != 1:
        raise MateError("Expected exact Stockfish 17.1 UCI identity and uciok")
    for option in ("Threads", "Hash", "UCI_Chess960"):
        if not any(line.startswith("option name " + option + " ") for line in lines):
            raise MateError(f"Oracle is missing required option {option}")


class MateOracle:
    """The production entry point always verifies the exact official binary."""

    def __init__(self, binary: Path, timeout: float = 10.0) -> None:
        self.binary = binary.resolve()
        if sha256(self.binary) != REFERENCE_SHA256:
            raise MateError("Stockfish SHA-256 mismatch")
        self.transport = UciTransport((str(self.binary),), timeout)
        self.cache: dict[str, State] = {}
        self.queries = 0
        self.records: list[dict[str, object]] = []
        self.greeting: list[str] = []

    def __enter__(self) -> MateOracle:
        if sha256(self.binary) != REFERENCE_SHA256:
            raise MateError("Stockfish SHA-256 changed before launch")
        self.transport.__enter__()
        try:
            self.greeting = self.transport.exchange("uci", "uciok")
            validate_greeting(self.greeting)
            self.transport.exchange(
                "setoption name Threads value 1\n"
                "setoption name Hash value 16\nsetoption name UCI_Chess960 value false\nisready",
                "readyok",
            )
        except BaseException:
            self.transport.__exit__(None, None, None)
            raise
        return self

    def state(self, fen: str, move: str | None = None) -> State:
        validate_position(fen)
        expected = _play(fen, move) if move else fen
        key = _key(expected)
        if key in self.cache:
            return self.cache[key]
        command = "position fen " + fen + (" moves " + move if move else "") + "\nd\ngo perft 1"
        lines = self.transport.exchange(command, "Nodes searched")
        result = parse_state(lines)
        if result.fen != expected:
            raise MateError(
                f"Oracle FEN transition mismatch: expected {expected}, got {result.fen}"
            )
        board = _board(result.fen)
        king = "K" if result.side == "w" else "k"
        square = next(square for square, piece in board.items() if piece == king)
        if result.check != _attacked(board, square, result.side == "b"):
            raise MateError("Oracle check status disagrees with independent geometry")
        self.cache[key] = result
        self.queries += 1
        self.records.append(
            {"query_fen": fen, "move": move, "state": asdict(result), "response": lines}
        )
        return result

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.transport.__exit__(exc_type, exc_value, traceback)
        if sha256(self.binary) != REFERENCE_SHA256:
            raise MateError("Stockfish SHA-256 changed during proof")


@dataclass(frozen=True)
class Proof:
    mate: int | None
    winning_moves: tuple[str, ...]
    pv: tuple[str, ...]
    refuted_horizons: tuple[int, ...]
    states: int


class Prover:
    """Finite minimax proof: OR attacker, AND defender, with memoized positions.

    Complete legal root move enumeration at minimal odd horizon. A terminal loss
    by the attacker, stalemate, or a live horizon leaf refutes that branch.
    No pruning heuristic or numerical evaluation participates in a proof.
    """

    def __init__(
        self, oracle: LegalOracle, maximum_states: int = 200_000, maximum_seconds: float = 300.0
    ) -> None:
        if type(maximum_states) is not int or not 1 <= maximum_states <= 2_000_000:
            raise MateError("State budget must be in 1..2000000")
        if not math.isfinite(maximum_seconds) or not 0 < maximum_seconds <= 3600:
            raise MateError("Proof wall-time budget must be in (0,3600] seconds")
        self.deadline = time.monotonic() + maximum_seconds
        self.oracle = oracle
        self.maximum_states = maximum_states
        self.states: dict[str, State] = {}
        self.transitions: dict[tuple[str, str | None], State] = {}
        self.memo: dict[tuple[str, int, str], tuple[str, ...] | None] = {}

    def _state(self, fen: str, move: str | None = None) -> State:
        if time.monotonic() >= self.deadline:
            raise MateError("Exact mate proof exceeded wall-time budget")
        transition = (_key(fen), move)
        if transition not in self.transitions:
            if len(self.states) >= self.maximum_states:
                raise MateError("Exact mate proof exceeded state budget")
            state = self.oracle.state(fen, move)
            key = _key(state.fen)
            if key not in self.states:
                self.states[key] = state
            self.transitions[transition] = self.states[key]
        return self.transitions[transition]

    def _solve(self, state: State, remaining: int, attacker: str) -> tuple[str, ...] | None:
        if time.monotonic() >= self.deadline:
            raise MateError("Exact mate proof exceeded wall-time budget")
        key = (_key(state.fen), remaining, attacker)
        if key in self.memo:
            return self.memo[key]
        result: tuple[str, ...] | None = None
        if not state.moves:
            result = () if state.check and state.side != attacker else None
        elif remaining > 0:
            attacker_turn = state.side == attacker
            lines: list[tuple[str, ...]] = []
            for move in state.moves:
                child = self._state(state.fen, move)
                tail = self._solve(child, remaining - 1, attacker)
                if tail is None:
                    if not attacker_turn:
                        lines = []
                        break
                else:
                    lines.append((move,) + tail)
                    if attacker_turn:
                        break
            if lines:
                result = min(lines, key=len) if attacker_turn else max(lines, key=len)
        self.memo[key] = result
        return result

    def prove(self, fen: str, maximum_mate: int = 3) -> Proof:
        if type(maximum_mate) is not int or not 1 <= maximum_mate <= 3:
            raise MateError("Exact proof supports maximum mate 1..3")
        validate_root(fen)
        root = self._state(fen)
        if not root.moves:
            raise MateError("Fixture root must have legal moves")
        refuted: list[int] = []
        for mate in range(1, maximum_mate + 1):
            wins = []
            lines: list[tuple[str, ...]] = []
            for move in root.moves:
                child = self._state(root.fen, move)
                line = self._solve(child, 2 * mate - 2, root.side)
                if line is not None:
                    wins.append(move)
                    lines.append((move,) + line)
            if wins:
                return Proof(mate, tuple(wins), lines[0], tuple(refuted), len(self.states))
            refuted.append(mate)
        return Proof(None, (), (), tuple(refuted), len(self.states))


def candidates(seed: int = SEED) -> tuple[str, ...]:
    """Original coordinate enumeration, shuffled with a fixed private RNG seed."""
    rng = random.Random(seed)
    positions = []
    for king in (42, 41, 34, 35, 43):  # c6,b6,c5,d5,d6; losing king a8.
        squares = list(range(64))
        rng.shuffle(squares)
        for material in ("Q", "R"):
            for square in squares:
                if square in (56, king):
                    continue
                fen = _fen({56: "k", king: "K", square: material}, "w")
                try:
                    validate_root(fen)
                except MateError:
                    continue
                positions.append(fen)
    # Block a queen's rank-eight ray to the losing king with our own knight.
    # Coordinates are constructed here, not taken from a composed puzzle.
    for king in (42, 41, 34, 35, 43):
        for queen in range(59, 64):
            for blocker in range(57, queen):
                fen = _fen({56: "k", king: "K", queen: "Q", blocker: "N"}, "w")
                try:
                    validate_root(fen)
                except MateError:
                    continue
                positions.append(fen)
    return tuple(positions)


def invert(fen: str) -> str:
    fields = fen.split()
    return _fen(
        {square ^ 56: piece.swapcase() for square, piece in _board(fen).items()},
        "b" if fields[1] == "w" else "w",
        int(fields[4]),
        int(fields[5]),
    )


def write_corpus(destination: Path, producer: Callable[[], T]) -> T:
    """Produce/validate fully before atomic replacement; failures preserve old data."""
    value = producer()
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix="." + destination.name,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return value


def validate_corpus(value: object) -> None:
    if not isinstance(value, dict) or (
        type(value.get("schema")) is not int or value.get("schema") != 1
    ):
        raise MateError("Expected mate corpus schema 1")
    if value.get("oracle_sha256") != REFERENCE_SHA256:
        raise MateError("Mate corpus oracle identity mismatch")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 18:
        raise MateError("Mate corpus requires exactly 18 cases")
    seen: set[str] = set()
    ids: set[str] = set()
    counts: dict[tuple[int, str], int] = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("fen"), str):
            raise MateError("Malformed mate case")
        fen = case["fen"]
        validate_root(fen)
        identifier = case.get("id")
        mate = case.get("mate")
        wins, pv = case.get("winning_moves"), case.get("pv")
        if (
            not isinstance(identifier, str)
            or re.fullmatch(r"mate[123]_[wb]_[123]", identifier) is None
        ):
            raise MateError("Malformed mate case id")
        if type(mate) is not int or mate not in (1, 2, 3):
            raise MateError("Mate distance must be 1..3")
        if (
            not isinstance(wins, list)
            or not wins
            or any(not isinstance(move, str) or MOVE.fullmatch(move) is None for move in wins)
        ):
            raise MateError("Missing or malformed complete winning move set")
        if wins != sorted(set(wins)):
            raise MateError("Winning moves must be unique and sorted")
        if (
            not isinstance(pv, list)
            or len(pv) != 2 * mate - 1
            or pv[0] not in wins
            or any(not isinstance(move, str) or MOVE.fullmatch(move) is None for move in pv)
        ):
            raise MateError("Missing or malformed exact mate PV")
        if not identifier.startswith(f"mate{mate}_{fen.split()[1]}_"):
            raise MateError("Mate case id disagrees with mate distance/active color")
        refuted = case.get("refuted_horizons")
        if (
            not isinstance(refuted, list)
            or any(type(item) is not int for item in refuted)
            or refuted != list(range(1, mate))
        ):
            raise MateError("Shallower mate horizons must all be refuted")
        if _key(fen) in seen or identifier in ids:
            raise MateError("Duplicate mate fixture")
        seen.add(_key(fen))
        ids.add(identifier)
        key = (mate, fen.split()[1])
        counts[key] = counts.get(key, 0) + 1
    if any(counts.get((mate, side)) != 3 for mate in (1, 2, 3) for side in ("w", "b")):
        raise MateError("Need three positions for each mate distance and active color")


class RecordedOracle:
    """Replay hash-bound real responses; this does not execute an engine."""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        if sha256(path) != expected_sha256:
            raise MateError("Oracle transcript SHA-256 mismatch")
        self.states: dict[str, State] = {}
        try:
            with gzip.open(path, "rb") as stream:
                payload = stream.read(64 * 1024 * 1024 + 1)
            if len(payload) > 64 * 1024 * 1024:
                raise MateError("Oracle transcript exceeds 64 MiB decompressed limit")
            for line in payload.splitlines():
                if len(line) > 8192 or len(self.states) >= 200_000:
                    raise MateError("Oracle transcript record/count exceeds bounds")
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) != {
                    "query_fen",
                    "move",
                    "state",
                    "response",
                }:
                    raise MateError("Malformed oracle transcript record")
                fen, move, value = record["query_fen"], record["move"], record["state"]
                if not isinstance(fen, str) or (move is not None and not isinstance(move, str)):
                    raise MateError("Malformed oracle transcript query")
                validate_position(fen)
                if not isinstance(value, dict) or set(value) != {"fen", "side", "check", "moves"}:
                    raise MateError("Malformed oracle transcript state")
                expected = _play(fen, move) if move else fen
                if value["fen"] != expected or value["side"] != expected.split()[1]:
                    raise MateError("Recorded oracle transition/side mismatch")
                moves = value["moves"]
                if (
                    type(value["check"]) is not bool
                    or not isinstance(moves, list)
                    or len(moves) > 218
                    or any(
                        not isinstance(item, str) or MOVE.fullmatch(item) is None for item in moves
                    )
                    or moves != sorted(set(moves))
                ):
                    raise MateError("Malformed recorded check status or legal move list")
                state = State(expected, value["side"], value["check"], tuple(moves))
                raw = record["response"]
                if (
                    not isinstance(raw, list)
                    or len(raw) > 512
                    or any(not isinstance(line, str) or len(line) > 4096 for line in raw)
                    or parse_state(raw) != state
                ):
                    raise MateError("Raw oracle response disagrees with recorded state")
                board = _board(expected)
                king = "K" if state.side == "w" else "k"
                square = next(square for square, piece in board.items() if piece == king)
                if state.check != _attacked(board, square, state.side == "b"):
                    raise MateError("Recorded check status disagrees with geometry")
                if _key(expected) in self.states:
                    raise MateError("Duplicate recorded oracle position")
                self.states[_key(expected)] = state
        except (OSError, EOFError, UnicodeError, ValueError, RecursionError) as error:
            raise MateError(f"Cannot read oracle transcript: {error}") from error
        if not self.states:
            raise MateError("Oracle transcript is empty")
        if sha256(path) != expected_sha256:
            raise MateError("Oracle transcript changed during loading")

    def state(self, fen: str, move: str | None = None) -> State:
        expected = _play(fen, move) if move else fen
        try:
            return self.states[_key(expected)]
        except KeyError as error:
            raise MateError(f"Oracle transcript is missing required state: {expected}") from error


def _document(path: Path) -> object:
    with path.open("rb") as stream:
        data = stream.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise MateError("Mate JSON exceeds 1 MiB limit")
    try:
        return json.loads(data)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise MateError(f"Invalid mate JSON: {error}") from error


def render_epd(value: dict[str, object]) -> str:
    cases = value.get("cases")
    if not isinstance(cases, list):
        raise MateError("Missing EPD cases")
    text = "# Original Mouse exact mate fixtures; see provenance.json and README.md.\n"
    for case in cases:
        moves = " ".join(case["winning_moves"])
        text += (
            f'{case["fen"]} ; id "{case["id"]}"; variant standard; dm {case["mate"]}; bm {moves};\n'
        )
    return text


def _suite(root: Path) -> tuple[dict[str, object], dict[str, str]]:
    names = (
        "mates.json",
        "mates.epd",
        "oracle-states.jsonl.gz",
        "uci-handshake.txt",
        "provenance.json",
    )
    frozen = {name: sha256(root / name) for name in names}
    value = _document(root / "mates.json")
    provenance = _document(root / "provenance.json")
    validate_corpus(value)
    if not isinstance(value, dict) or not isinstance(provenance, dict):
        raise MateError("Mate suite/provenance must be objects")
    files = provenance.get("files")
    if (
        provenance.get("binary_sha256") != REFERENCE_SHA256
        or not isinstance(files, dict)
        or set(files) != set(names[:-1])
        or any(files.get(name) != frozen[name] for name in names[:-1])
    ):
        raise MateError("Mate suite provenance/file SHA-256 mismatch")
    if (root / "mates.epd").read_text(encoding="utf-8") != render_epd(value):
        raise MateError("EPD facts disagree with the JSON proof corpus")
    return value, frozen


def _reprove(
    value: dict[str, object], oracle: LegalOracle, maximum_states: int = 200_000
) -> dict[str, object]:
    prover = Prover(oracle, maximum_states)
    cases = value["cases"]
    if not isinstance(cases, list):
        raise MateError("Missing mate cases")
    checked = []
    horizons = 0
    for case in cases:
        proof = prover.prove(case["fen"], case["mate"])
        if (
            proof.mate != case["mate"]
            or list(proof.winning_moves) != case["winning_moves"]
            or list(proof.refuted_horizons) != case["refuted_horizons"]
        ):
            raise MateError(f"Independent proof differs for {case['id']}")
        # Stored PV is replayed in full, rather than trusting a printed search score.
        state = prover._state(case["fen"])
        for move in case["pv"]:
            if move not in state.moves:
                raise MateError(f"Illegal stored PV for {case['id']}")
            state = prover._state(state.fen, move)
        if state.moves or not state.check or state.side == case["fen"].split()[1]:
            raise MateError(f"Stored PV does not terminate in opponent checkmate: {case['id']}")
        horizons += case["mate"]
        checked.append(
            {
                "id": case["id"],
                "mate": proof.mate,
                "winning_moves": list(proof.winning_moves),
                "pv_validated": True,
            }
        )
    return {
        "cases_verified": len(checked),
        "mate_horizons_verified": horizons,
        "states_reproved": len(prover.states),
        "checks": checked,
    }


def replay_corpus(root: Path) -> dict[str, object]:
    value, frozen = _suite(root)
    oracle = RecordedOracle(root / "oracle-states.jsonl.gz", frozen["oracle-states.jsonl.gz"])
    result = _reprove(value, oracle)
    if any(sha256(root / name) != digest for name, digest in frozen.items()):
        raise MateError("Mate suite changed during transcript replay")
    return {"schema": 1, "oracle_execution": False, "frozen_sha256": frozen, **result}


def verify_corpus(
    binary: Path, root: Path, *, maximum_states: int = 200_000, timeout: float = 10.0
) -> dict[str, object]:
    source = Path(__file__).resolve()
    source_sha256, binary_sha256 = sha256(source), sha256(binary)
    parser_source = source.with_name("perft_reference.py")
    parser_sha256 = sha256(parser_source)
    value, frozen = _suite(root)
    started = time.monotonic()
    with MateOracle(binary, timeout) as oracle:
        result = _reprove(value, oracle, maximum_states)
        queries = oracle.queries
    if (
        sha256(source) != source_sha256
        or sha256(parser_source) != parser_sha256
        or sha256(binary) != binary_sha256
        or any(sha256(root / name) != digest for name, digest in frozen.items())
    ):
        raise MateError("Mate source/oracle/suite changed during live verification")
    return {
        "schema": 1,
        "oracle_execution": True,
        "oracle_sha256": binary_sha256,
        "generator_sha256": source_sha256,
        "legal_parser_sha256": parser_sha256,
        "frozen_sha256": frozen,
        "cold_oracle_queries": queries,
        "wall_seconds": time.monotonic() - started,
        **result,
    }


def generate(
    binary: Path,
    output: Path,
    *,
    seed: int = SEED,
    maximum_states: int = 200_000,
    timeout: float = 10.0,
) -> dict[str, object]:
    if output.exists():
        raise MateError("Output directory must be new; regenerate separately for comparison")
    source = Path(__file__).resolve()
    parser_source = source.with_name("perft_reference.py")
    identities = {
        "generator": sha256(source),
        "oracle": sha256(binary),
        "legal_parser": sha256(parser_source),
    }
    started = time.monotonic()
    cases: list[dict[str, object]] = []
    counts = {1: 0, 2: 0, 3: 0}
    selected: set[tuple[int, str]] = set()
    examined = 0
    with MateOracle(binary, timeout) as oracle:
        prover = Prover(oracle, maximum_states)
        for candidate in candidates(seed):
            examined += 1
            proof = prover.prove(candidate)
            category = "blocked" if "N" in candidate else ("rook" if "R" in candidate else "queen")
            if proof.mate is None or (proof.mate, category) in selected:
                continue
            mate = proof.mate
            selected.add((mate, category))
            counts[mate] += 1
            for fen in (candidate, invert(candidate)):
                queries_before = oracle.queries
                result = prover.prove(fen)
                if result.mate != mate:
                    raise MateError("Color-inverted proof changed mate distance")
                case = {
                    "id": f"mate{mate}_{fen.split()[1]}_{counts[mate]}",
                    "fen": fen,
                    **asdict(result),
                    "new_oracle_queries": oracle.queries - queries_before,
                }
                cases.append(case)
                print(
                    f"{case['id']}: mate {mate}, {len(result.winning_moves)} winning roots, "
                    f"{len(prover.states)} cached states",
                    flush=True,
                )
            if all(count == 3 for count in counts.values()):
                break
        if not all(count == 3 for count in counts.values()):
            raise MateError("Candidate enumeration did not produce all 18 exact mate cases")
        cases.sort(key=lambda case: str(case["id"]))
        value: dict[str, object] = {
            "schema": 1,
            "oracle_name": REFERENCE_NAME,
            "oracle_sha256": REFERENCE_SHA256,
            "generator_sha256": identities["generator"],
            "seed": seed,
            "candidate_count_examined": examined,
            "algorithm": "bounded legal-oracle AND/OR, odd horizons 1,3,5; all winning root moves",
            "maximum_states": maximum_states,
            "oracle_queries": oracle.queries,
            "cached_positions": len(prover.states),
            "memo_entries": len(prover.memo),
            "mouse_node_budgets": None,
            "cases": cases,
        }
        # JSON normalizes tuples to arrays before schema validation.
        value = json.loads(json.dumps(value))
        validate_corpus(value)
        if (
            sha256(source) != identities["generator"]
            or sha256(binary) != identities["oracle"]
            or sha256(parser_source) != identities["legal_parser"]
        ):
            raise MateError("Generator/oracle artifact changed during proof")
        records = oracle.records
        greeting = oracle.greeting
    output.mkdir(parents=True)
    write_corpus(output / "mates.json", lambda: value)
    (output / "mates.epd").write_text(render_epd(value), encoding="utf-8", newline="\n")
    payload = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    with (output / "oracle-states.jsonl.gz").open("wb") as stream:
        stream.write(gzip.compress(payload.encode(), mtime=0))
    (output / "uci-handshake.txt").write_text(
        "\n".join(greeting) + "\n", encoding="utf-8", newline="\n"
    )
    provenance = {
        "schema": 1,
        "release_url": RELEASE_URL,
        "archive_url": ARCHIVE_URL,
        "source_commit": SOURCE_COMMIT,
        "binary_sha256": REFERENCE_SHA256,
        "binary_path": str(binary.resolve()),
        "source_sha256": identities["generator"],
        "legal_parser_sha256": identities["legal_parser"],
        "python": sys.version,
        "development_interpreter": True,
        "wall_seconds": time.monotonic() - started,
        "cold_oracle_queries": len(records),
        "persisted_cache_used": False,
        "commands": [
            "uci",
            "setoption name Threads value 1",
            "setoption name Hash value 16",
            "setoption name UCI_Chess960 value false",
            "isready",
            "position fen <query_fen> [moves <move>]",
            "d",
            "go perft 1",
        ],
        "files": {path.name: sha256(path) for path in output.iterdir() if path.is_file()},
    }
    write_corpus(output / "provenance.json", lambda: provenance)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", type=Path, help="Reprove an existing full suite directory")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--maximum-states", type=int, default=200_000)
    parser.add_argument("--timeout", type=float, default=10.0)
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        print("error: run downloaded-file readers with python -I", file=sys.stderr)
        return 2
    try:
        if arguments.verify is not None:
            report = verify_corpus(
                arguments.oracle,
                arguments.verify,
                maximum_states=arguments.maximum_states,
                timeout=arguments.timeout,
            )
            write_corpus(arguments.output, lambda: report)
            print(f"Verified {report['cases_verified']} exact mate fixtures against live oracle")
        else:
            result = generate(
                arguments.oracle,
                arguments.output,
                seed=arguments.seed,
                maximum_states=arguments.maximum_states,
                timeout=arguments.timeout,
            )
            result_cases = result["cases"]
            if not isinstance(result_cases, list):
                raise MateError("Generated cases must be an array")
            print(f"Verified {len(result_cases)} exact mate fixtures")
    except (MateError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
