# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate and verify Mouse's perft facts using the pinned Stockfish 17.1 oracle.

The UCI implementation here is original Mouse code. Reference facts are cited in
 tests/data/README.md; no upstream engine source is vendored by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Literal

Variant = Literal["standard", "chess960"]
REFERENCE_NAME = "Stockfish 17.1"
REFERENCE_SHA256 = "8f8c79837ddf2d0b7baccf555769778dd6a7e22bec823814fe19ea8bc37ff39f"
ARCHIVE_SHA256 = "8034b0a0725bcf19796c3695b8794698b15bc2168edacde06ab32fec23cc709d"
RELEASE_URL = "https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1"
ARCHIVE_URL = (
    "https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/"
    "stockfish-windows-x86-64-bmi2.zip"
)
SOURCE_COMMIT = "03e27488f3d21d8ff4dbf3065603afa21dbd0ef3"
SOURCE_TESTS = (
    "https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_17.1/tests/perft.sh"
)
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
DEFAULT_SEED = 20261017
UINT64_MAX = (1 << 64) - 1
# Keep input bounds aligned with engine/chess/fen.cpp and perft.h.
MAX_PERFT_DEPTH = 128


class ReferenceError(RuntimeError):
    """Invalid fixture data, oracle output, or executable identity."""


@dataclass(frozen=True)
class Case:
    identifier: str
    fen: str
    variant: Variant
    counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Candidate:
    identifier: str
    fen: str
    variant: Variant
    maximum_depth: int = 3
    known: tuple[tuple[int, int], ...] = ()
    must_include: tuple[str, ...] = ()
    must_exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerftResult:
    fen: str
    nodes: int
    divide: tuple[tuple[str, int], ...]

    @property
    def moves(self) -> tuple[str, ...]:
        return tuple(move for move, _ in self.divide)


def _bounded_decimal(token: str, context: str, minimum: int, maximum: int) -> int:
    """Check token length before int(), including excessive leading zeroes."""
    if not 1 <= len(token) <= len(str(maximum)) or not token.isascii() or not token.isdecimal():
        raise ReferenceError(
            f"{context} must be an ASCII decimal between {minimum} and {maximum} "
            f"with at most {len(str(maximum))} digits"
        )
    value = int(token)
    if not minimum <= value <= maximum:
        raise ReferenceError(f"{context} must be between {minimum} and {maximum}")
    return value


def validate_fen(fen: str) -> None:
    """Reject malformed input; full chess legality belongs to the oracle/core."""
    fields = fen.split()
    if len(fields) != 6 or any(character in fen for character in "\r\n;"):
        raise ReferenceError(f"Expected six clean FEN fields: {fen}")
    board, side, rights, ep, halfmove, fullmove = fields
    rows = board.split("/")
    if len(rows) != 8 or side not in ("w", "b"):
        raise ReferenceError(f"Invalid FEN board or active color: {fen}")
    kings: dict[str, tuple[int, int]] = {}
    for rank, row in enumerate(rows):
        file = 0
        for piece in row:
            if piece in "12345678":
                file += int(piece)
            elif piece in "PNBRQKpnbrqk":
                if piece in "Pp" and rank in (0, 7):
                    raise ReferenceError(f"FEN pawn on promotion rank: {fen}")
                if piece in "Kk":
                    if piece in kings:
                        raise ReferenceError(f"Duplicate king: {fen}")
                    kings[piece] = (file, rank)
                file += 1
            else:
                raise ReferenceError(f"Invalid FEN piece: {fen}")
        if file != 8:
            raise ReferenceError(f"FEN rank does not have eight squares: {fen}")
    if set(kings) != {"K", "k"}:
        raise ReferenceError(f"FEN must have one king per color: {fen}")
    if max(abs(a - b) for a, b in zip(kings["K"], kings["k"], strict=True)) <= 1:
        raise ReferenceError(f"Adjacent FEN kings: {fen}")
    if rights != "-" and (
        not rights
        or len(set(rights)) != len(rights)
        or any(right not in "KQkqABCDEFGHabcdefgh" for right in rights)
    ):
        raise ReferenceError(f"Invalid FEN castling rights: {fen}")
    expected_ep_rank = "6" if side == "w" else "3"
    if ep != "-" and re.fullmatch(r"[a-h]" + expected_ep_rank, ep) is None:
        raise ReferenceError(f"Invalid FEN en-passant square: {fen}")
    _bounded_decimal(halfmove, "FEN halfmove counter", 0, UINT64_MAX)
    _bounded_decimal(fullmove, "FEN fullmove counter", 1, UINT64_MAX)


def position_key(fen: str) -> str:
    fields = fen.split()[:4]
    fields[2] = "".join(sorted(fields[2]))
    return " ".join(fields)


def parse_epd(text: str) -> tuple[Case, ...]:
    cases: list[Case] = []
    identifiers: set[str] = set()
    positions: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [part.strip() for part in line.split(";")]
        fen = parts[0]
        validate_fen(fen)
        identifier: str | None = None
        variant: Variant | None = None
        counts: dict[int, int] = {}
        for operation in parts[1:]:
            if not operation:
                continue
            id_match = re.fullmatch(r'id "([a-zA-Z0-9_-]+)"', operation)
            variant_match = re.fullmatch(r"variant (standard|chess960)", operation)
            count_match = re.fullmatch(r"D([1-9][0-9]*) ([0-9]+)", operation)
            if id_match and identifier is None:
                identifier = id_match[1]
            elif variant_match and variant is None:
                variant = "standard" if variant_match[1] == "standard" else "chess960"
            elif count_match:
                depth = _bounded_decimal(
                    count_match[1], f"EPD depth on line {number}", 1, MAX_PERFT_DEPTH
                )
                if depth in counts:
                    raise ReferenceError(f"Duplicate EPD depth on line {number}: D{depth}")
                counts[depth] = _bounded_decimal(
                    count_match[2], f"EPD node count on line {number}", 0, UINT64_MAX
                )
            else:
                raise ReferenceError(
                    f"Malformed or duplicate EPD operation on line {number}: {operation}"
                )
        if identifier is None or variant is None or not counts:
            raise ReferenceError(f"Missing EPD id, variant or depth on line {number}")
        if identifier in identifiers:
            raise ReferenceError(f"Duplicate id: {identifier}")
        key = position_key(fen)
        if key in positions:
            raise ReferenceError(f"Duplicate position: {identifier}")
        identifiers.add(identifier)
        positions.add(key)
        cases.append(Case(identifier, fen, variant, tuple(sorted(counts.items()))))
    if not cases:
        raise ReferenceError("EPD suite is empty")
    return tuple(cases)


def parse_output(lines: Sequence[str]) -> PerftResult:
    totals = [
        _bounded_decimal(match[1], "Oracle total", 0, UINT64_MAX)
        for line in lines
        if (match := re.fullmatch(r"Nodes searched: ([0-9]+)", line.strip()))
    ]
    fens = [line.removeprefix("Fen: ").strip() for line in lines if line.startswith("Fen: ")]
    divide: list[tuple[str, int]] = []
    seen: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([a-h][1-8][a-h][1-8][nbrq]?): ([0-9]+)", line.strip())
        if match:
            if match[1] in seen:
                raise ReferenceError(f"Duplicate perft root move: {match[1]}")
            seen.add(match[1])
            divide.append(
                (match[1], _bounded_decimal(match[2], "Oracle divide count", 0, UINT64_MAX))
            )
    if len(totals) != 1 or len(fens) != 1:
        raise ReferenceError("Oracle must return one FEN and one perft total")
    validate_fen(fens[0])
    if sum(count for _, count in divide) != totals[0]:
        raise ReferenceError("Oracle perft divide does not match its total")
    return PerftResult(fens[0], totals[0], tuple(divide))


def verify_binary(binary: Path, sha256: str = REFERENCE_SHA256) -> None:
    with binary.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != sha256:
        raise ReferenceError(f"Stockfish SHA-256 mismatch: expected {sha256}, got {actual}")


class UciReference:
    """One CPU oracle. A reader thread feeds a queue; only the caller writes UCI."""

    def __init__(self, binary: Path, timeout: float = 120.0) -> None:
        verify_binary(binary)
        self.binary = binary.resolve()
        self.timeout = timeout
        self._process: subprocess.Popen[str] | None = None
        self._input: IO[str] | None = None
        self._output: IO[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()

    def _pump(self) -> None:
        if self._output is None:
            return
        try:
            for line in self._output:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _send(self, command: str) -> None:
        if self._input is None:
            raise ReferenceError("Reference process is not running")
        try:
            self._input.write(command + "\n")
            self._input.flush()
        except OSError as error:
            raise ReferenceError(f"Reference process input failed: {error}") from error

    def _until(self, marker: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReferenceError(f"Reference timed out awaiting {marker}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as error:
                raise ReferenceError(f"Reference timed out awaiting {marker}") from error
            if line is None:
                raise ReferenceError(f"Reference exited awaiting {marker}: {lines[-10:]}")
            lines.append(line)
            if len(lines) > 10_000:
                raise ReferenceError("Reference returned excessive output")
            if line == marker or line.startswith(marker + ":"):
                return lines

    def __enter__(self) -> UciReference:
        self._process = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._input = self._process.stdin
        self._output = self._process.stdout
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        try:
            self._send("uci")
            greeting = self._until("uciok")
            if "id name " + REFERENCE_NAME not in greeting:
                raise ReferenceError(f"Expected {REFERENCE_NAME}, got {greeting[:4]}")
            if not any(line.startswith("option name UCI_Chess960 ") for line in greeting):
                raise ReferenceError("Reference has no UCI_Chess960 option")
            self._send("setoption name Threads value 1")
            self._send("isready")
            self._until("readyok")
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._send("quit")
                    self._process.wait(timeout=5)
            except (ReferenceError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait(timeout=5)
            finally:
                if self._input is not None:
                    self._input.close()
                if self._output is not None:
                    self._output.close()
                if self._reader is not None:
                    self._reader.join(timeout=5)
                self._process = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def perft(
        self, fen: str, variant: Variant, depth: int, moves: tuple[str, ...] = ()
    ) -> PerftResult:
        validate_fen(fen)
        if depth < 1 or any(
            re.fullmatch(r"[a-h][1-8][a-h][1-8][nbrq]?", move) is None for move in moves
        ):
            raise ReferenceError("Invalid perft depth or UCI move")
        self._send("setoption name UCI_Chess960 value " + str(variant == "chess960").lower())
        command = "position fen " + fen
        if moves:
            command += " moves " + " ".join(moves)
        self._send(command)
        self._send("d")
        self._send(f"go perft {depth}")
        return parse_output(self._until("Nodes searched"))


def chess960_backrank(index: int) -> str:
    """Scharnagl numbering: bishops, queen, knight pair, then rook/king/rook."""
    if not 0 <= index < 960:
        raise ReferenceError(f"Chess960 index must be 0..959: {index}")
    rank = [""] * 8
    rank[(index % 4) * 2 + 1] = "B"
    index //= 4
    rank[(index % 4) * 2] = "B"
    index //= 4
    empty = [file for file, piece in enumerate(rank) if not piece]
    rank[empty[index % 6]] = "Q"
    index //= 6
    empty = [file for file, piece in enumerate(rank) if not piece]
    pair = tuple(itertools.combinations(range(5), 2))[index]
    for position in pair:
        rank[empty[position]] = "N"
    empty = [file for file, piece in enumerate(rank) if not piece]
    for file, piece in zip(empty, "RKR", strict=True):
        rank[file] = piece
    return "".join(rank)


def _fen_row(pieces: dict[int, str]) -> str:
    row = ""
    empty = 0
    for file in range(8):
        if file not in pieces:
            empty += 1
        else:
            if empty:
                row += str(empty)
                empty = 0
            row += pieces[file]
    return row + (str(empty) if empty else "")


def candidate_cases(seed: int = DEFAULT_SEED) -> tuple[Candidate, ...]:
    cases = [
        Candidate("spec_start", START_FEN, "standard", 6, ((5, 4865609), (6, 119060324))),
        Candidate(
            "spec_kiwipete",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "standard",
            5,
            ((4, 4085603), (5, 193690690)),
        ),
        Candidate(
            "spec_rook_endgame",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "standard",
            6,
            ((6, 11030083),),
        ),
        Candidate(
            "spec_promotions_castling",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "standard",
            5,
            ((5, 15833292),),
        ),
        Candidate(
            "spec_promotion_checks",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "standard",
            4,
            ((4, 2103487),),
        ),
        Candidate(
            "spec_middlegame",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            "standard",
            5,
            ((5, 164075551),),
        ),
    ]
    specials = (
        ("ep_white", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ("e5d6",), ()),
        ("ep_black", "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", ("e4d3",), ()),
        ("ep_pinned_white", "8/8/8/4KPpr/8/8/8/7k w - g6 0 1", (), ("f5g6",)),
        ("ep_pinned_black", "7K/8/8/8/4kpPR/8/8/8 b - g3 0 1", (), ("f4g3",)),
        (
            "promotion_white",
            "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
            ("a7a8n", "a7a8b", "a7a8r", "a7a8q"),
            (),
        ),
        (
            "promotion_black",
            "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
            ("h2h1n", "h2h1b", "h2h1r", "h2h1q"),
            (),
        ),
        ("capture_promotion_white", "1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1", ("a7b8n", "a7b8q"), ()),
        ("capture_promotion_black", "4k3/P7/8/8/8/8/7p/4K1R1 b - - 0 1", ("h2g1n", "h2g1q"), ()),
        ("castle_white", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ("e1g1", "e1c1"), ()),
        ("castle_black", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", ("e8g8", "e8c8"), ()),
        ("single_check", "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", (), ()),
        ("double_check", "4k3/8/8/8/1b6/8/4r3/4K3 w - - 0 1", (), ()),
        ("rook_pin", "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1", (), ()),
        ("knight_pin", "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1", (), ()),
        ("block_check", "4k3/8/8/8/8/8/4B3/4K2r w - - 0 1", (), ()),
        ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", (), ()),
        ("checkmate", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", (), ()),
        ("rule50_kings", "4k3/8/8/8/8/8/8/4K3 w - - 100 51", (), ()),
        ("ep_check_evasion_white", "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1", ("e5d6",), ()),
        ("ep_check_evasion_black", "8/8/8/4k3/3Pp3/8/8/4K3 b - d3 0 1", ("e4d3",), ()),
    )
    for name, fen, include, exclude in specials:
        cases.append(
            Candidate(
                "special_" + name, fen, "standard", must_include=include, must_exclude=exclude
            )
        )
    official = (
        ("rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2R2R b E - 1 20", 2, 1438),
        ("rr6/2kpp3/1ppn2p1/p2b1q1p/P4P1P/1PNN2P1/2PP4/1K2RR2 w E - 0 20", 3, 37340),
        ("rr6/2kpp3/1ppnb1p1/p2Q1q1p/P4P1P/1PNN2P1/2PP4/1K2RR2 b E - 2 19", 4, 2237725),
        ("rr6/2kpp3/1ppnb1p1/p4q1p/P4P1P/1PNN2P1/2PP2Q1/1K2RR2 w E - 1 19", 4, 2098209),
    )
    for number, (fen, depth, count) in enumerate(official, start=1):
        cases.append(
            Candidate(f"stockfish_frc_{number:02}", fen, "chess960", depth, ((depth, count),))
        )
    for index in random.Random(seed).sample([value for value in range(960) if value != 518], 24):
        backrank = chess960_backrank(index)
        rooks = [chr(ord("A") + file) for file, piece in enumerate(backrank) if piece == "R"]
        rights = "".join(rooks) + "".join(rooks).lower()
        fen = f"{backrank.lower()}/pppppppp/8/8/8/8/PPPPPPPP/{backrank} w {rights} - 0 1"
        cases.append(Candidate(f"frc_start_{index:03}", fen, "chess960"))
    # Stationary kings/rooks, origin/destination overlap, king-rook swaps, long
    # crossing routes and edge origins; each is mirrored for the other color.
    castling_pairs = (
        (6, 7),
        (2, 0),
        (4, 5),
        (4, 3),
        (5, 6),
        (3, 2),
        (4, 6),
        (4, 2),
        (5, 7),
        (3, 0),
        (1, 7),
        (1, 0),
        (0, 1),
        (7, 6),
        (6, 5),
        (2, 3),
    )
    for king, rook in castling_pairs:
        other_king = (rook + 2) % 8
        for side in ("w", "b"):
            row = _fen_row({king: "K", rook: "R"})
            opponent = _fen_row({other_king: "k"})
            right = chr(ord("A") + rook)
            if side == "w":
                board = opponent + "/8/8/8/8/8/8/" + row
                rank = "1"
            else:
                board = row.lower() + "/8/8/8/8/8/8/" + opponent.upper()
                right = right.lower()
                rank = "8"
            move = f"{chr(ord('a') + king)}{rank}{chr(ord('a') + rook)}{rank}"
            fen = f"{board} {side} {right} - 0 1"
            cases.append(
                Candidate(f"frc_castle_{side}_{king}{rook}", fen, "chess960", must_include=(move,))
            )
    for case in cases:
        validate_fen(case.fen)
    return tuple(cases)


def playout_cases(
    reference: UciReference, seed: int
) -> tuple[tuple[Candidate, ...], list[list[str]]]:
    generator = random.Random(seed)
    cases: list[Candidate] = []
    histories: list[list[str]] = [[]]
    fen = START_FEN
    signatures = {position_key(case.fen) for case in candidate_cases(seed)}
    for _ in range(2000):
        result = reference.perft(fen, "standard", 1)
        if not result.moves:
            fen = START_FEN
            histories.append([])
            continue
        move = generator.choice(sorted(result.moves))
        histories[-1].append(move)
        fen = reference.perft(fen, "standard", 1, (move,)).fen
        plies = len(histories[-1])
        key = position_key(fen)
        if plies >= 8 and (plies - 8) % 3 == 0 and key not in signatures:
            cases.append(Candidate(f"playout_{len(cases) + 1:03}", fen, "standard"))
            signatures.add(key)
            if len(cases) == 40:
                return tuple(cases), histories
    raise ReferenceError("Could not produce 40 distinct legal playout positions")


def _validate_generation_destinations(output: Path, provenance: Path) -> None:
    resolved_output = output.resolve()
    resolved_provenance = provenance.resolve()
    if resolved_output == resolved_provenance or (
        resolved_output.exists()
        and resolved_provenance.exists()
        and resolved_output.samefile(resolved_provenance)
    ):
        raise ReferenceError("EPD suite and provenance destinations must differ")


def generate(reference: UciReference, output: Path, provenance: Path, seed: int) -> int:
    _validate_generation_destinations(output, provenance)
    candidates = candidate_cases(seed)
    generated, histories = playout_cases(reference, seed)
    cases: list[Case] = []
    assertions: list[dict[str, str | int]] = []
    for candidate in (*candidates, *generated):
        counts: list[tuple[int, int]] = []
        for depth in range(1, candidate.maximum_depth + 1):
            result = reference.perft(candidate.fen, candidate.variant, depth)
            if depth == 1:
                if not set(candidate.must_include) <= set(result.moves):
                    raise ReferenceError(f"Required special move absent: {candidate.identifier}")
                if set(candidate.must_exclude) & set(result.moves):
                    raise ReferenceError(f"Forbidden special move present: {candidate.identifier}")
            known = dict(candidate.known).get(depth)
            if known is not None:
                if result.nodes != known:
                    raise ReferenceError(
                        f"Known count mismatch {candidate.identifier} D{depth}: "
                        f"{result.nodes} != {known}"
                    )
                assertions.append(
                    {"id": candidate.identifier, "depth": depth, "nodes": result.nodes}
                )
            counts.append((depth, result.nodes))
        cases.append(Case(candidate.identifier, candidate.fen, candidate.variant, tuple(counts)))
        print(
            f"verified {candidate.identifier} D{candidate.maximum_depth}={counts[-1][1]}",
            flush=True,
        )
    lines = [
        "# SPDX-License-Identifier: GPL-3.0-or-later",
        "# Generated by mouse_tools.perft_reference; see README.md and perft-provenance.json.",
    ]
    for case in cases:
        operations = "; ".join(f"D{depth} {count}" for depth, count in case.counts)
        lines.append(f'{case.fen} ; id "{case.identifier}"; variant {case.variant}; {operations};')
    text = "\n".join(lines) + "\n"
    parsed = parse_epd(text)
    if len(parsed) < 100:
        raise ReferenceError("Full suite must contain at least 100 distinct positions")
    metadata = {
        "schema": 1,
        "reference_name": REFERENCE_NAME,
        "source_commit": SOURCE_COMMIT,
        "release_url": RELEASE_URL,
        "archive_url": ARCHIVE_URL,
        "archive_sha256_local_observed": ARCHIVE_SHA256,
        "github_published_asset_digest": None,
        "executable_sha256_local_observed": REFERENCE_SHA256,
        "upstream_perft_facts_url": SOURCE_TESTS,
        "generator_seed": seed,
        "generator_count_cache": False,
        "oracle_process_reused": True,
        "oracle_threads": 1,
        "positions": len(parsed),
        "depth_checks": sum(len(case.counts) for case in parsed),
        "epd_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "known_contract_counts_verified": assertions,
        "legal_playout_games_uci": histories,
        "command": (
            "python -I tools/src/mouse_tools/perft_reference.py generate --engine "
            ".toolchain/reference/stockfish-17.1/stockfish/"
            "stockfish-windows-x86-64-bmi2.exe"
        ),
        "uci_query": [
            "setoption name Threads value 1",
            "setoption name UCI_Chess960 value <true|false>",
            "position fen <FEN>",
            "d",
            "go perft <depth>",
        ],
        "legality_checks": (
            "Legal playout moves come from oracle root divide. "
            "Every stripped castling case requires its king-to-rook "
            "castling move at D1. FEN shape is checked before UCI."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")
    provenance.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return len(parsed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify perft facts with Stockfish 17.1"
    )
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("tests/data/perft.epd"))
    parser.add_argument("--provenance", type=Path, default=Path("tests/data/perft-provenance.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=120.0)
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("reference files must be read with python -I")
    if not 0 < arguments.timeout < 3600:
        parser.error("timeout must be positive and less than one hour")
    try:
        if arguments.action == "generate":
            _validate_generation_destinations(arguments.suite, arguments.provenance)
        with UciReference(arguments.engine, arguments.timeout) as reference:
            if arguments.action == "generate":
                count = generate(reference, arguments.suite, arguments.provenance, arguments.seed)
                print(f"Generated {count} distinct positions; every stored depth verified")
            else:
                cases = parse_epd(arguments.suite.read_text(encoding="utf-8"))
                for case in cases:
                    for depth, expected in case.counts:
                        actual = reference.perft(case.fen, case.variant, depth).nodes
                        if actual != expected:
                            raise ReferenceError(
                                f"Mismatch {case.identifier} D{depth}: {actual} != {expected}"
                            )
                print(
                    f"Verified {len(cases)} positions, "
                    f"{sum(len(case.counts) for case in cases)} depth checks"
                )
    except (ReferenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
