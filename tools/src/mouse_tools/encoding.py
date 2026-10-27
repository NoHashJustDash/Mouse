# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent classical112 input and geometric Lc0 policy mirror.

Behavior reference: lc0 fd71a2d9 src/neural/encoder.cc. This is original code;
actual binary fixtures in tests/data/nn establish the history/policy conventions.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Literal

from mouse_tools.perft_reference import ReferenceError, validate_fen

FillHistory = Literal["no", "fen_only", "always"]
ALL_SQUARES = (1 << 64) - 1


class EncodingError(ValueError):
    """Unsupported or malformed input, history, or policy move."""


@dataclass(frozen=True)
class Plane:
    mask: int = 0
    value: float = 1.0

    def __post_init__(self) -> None:
        if type(self.mask) is not int or not 0 <= self.mask <= ALL_SQUARES:
            raise EncodingError("Plane mask must be an unsigned 64-bit integer")
        if not math.isfinite(self.value):
            raise EncodingError("Plane value must be finite")

    def dense(self) -> tuple[float, ...]:
        """A1-first row-major square values, matching [plane, rank, file]."""
        return tuple(self.value if (self.mask >> square) & 1 else 0.0 for square in range(64))


@dataclass(frozen=True)
class _Board:
    pieces: tuple[int, ...]
    black: bool
    rights: tuple[bool, bool, bool, bool]  # White Q/K, black Q/K.
    ep: str
    rule50: int
    start: bool


def _board(fen: str) -> _Board:
    try:
        validate_fen(fen)
    except ReferenceError as error:
        raise EncodingError(str(error)) from error
    placement, side, rights, ep, halfmove, _ = fen.split()
    pieces = [0] * 12
    for row, text in enumerate(placement.split("/")):
        file = 0
        for piece in text:
            if piece.isdecimal():
                file += int(piece)
            else:
                pieces["PNBRQKpnbrqk".index(piece)] |= 1 << ((7 - row) * 8 + file)
                file += 1
    flags = [False] * 4
    for right in rights.removeprefix("-"):
        black = right.islower()
        king = pieces[11 if black else 5].bit_length() - 1
        file = {"K": 7, "Q": 0}.get(right.upper(), ord(right.upper()) - ord("A"))
        rank = 7 if black else 0
        if king // 8 != rank or not pieces[9 if black else 3] & (1 << (rank * 8 + file)):
            raise EncodingError("Castling flag must name an existing home-rank king and rook")
        wing = int(file > king % 8)
        flag = int(black) * 2 + wing
        if file == king % 8 or flags[flag]:
            raise EncodingError("Ambiguous castling flags")
        flags[flag] = True
    if ep != "-":
        file = ord(ep[0]) - ord("a")
        destination = (3 if side == "b" else 4) * 8 + file
        pawn = 0 if side == "b" else 6
        origin = (1 if side == "b" else 6) * 8 + file
        occupied = 0
        for mask in pieces:
            occupied |= mask
        if not pieces[pawn] & (1 << destination) or occupied & (1 << origin):
            raise EncodingError("En-passant flag must describe a prior double pawn move")
    # ChessBoard equality compares parsed pieces, not the spelling of empty runs.
    initial_pieces = [
        0xFF00,
        0x42,
        0x24,
        0x81,
        0x8,
        0x10,
        0xFF000000000000,
        0x4200000000000000,
        0x2400000000000000,
        0x8100000000000000,
        0x0800000000000000,
        0x1000000000000000,
    ]
    start = pieces == initial_pieces and side == "w" and all(flags) and ep == "-"
    return _Board(
        tuple(pieces),
        side == "b",
        (flags[0], flags[1], flags[2], flags[3]),
        ep,
        int(halfmove),
        start,
    )


@dataclass(frozen=True)
class HistoryFrame:
    """Chronological frame; repetitions is Lc0's explicit input count, not FIDE key count."""

    fen: str
    repetitions: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.fen, str):
            raise EncodingError("Frame FEN must be a string")
        _board(self.fen)
        if type(self.repetitions) is not int or not 0 <= self.repetitions <= 2**32 - 1:
            raise EncodingError("Repetitions must be a nonnegative unsigned 32-bit count")


def _piece_masks(board: _Board, black: bool, infer_prior: bool) -> tuple[int, ...]:
    masks = list(board.pieces)
    if infer_prior and board.ep != "-":
        file = ord(board.ep[0]) - ord("a")
        pawn = 0 if board.black else 6
        destination = (3 if board.black else 4) * 8 + file
        origin = (1 if board.black else 6) * 8 + file
        masks[pawn] = (masks[pawn] & ~(1 << destination)) | (1 << origin)
    if black:
        masks = masks[6:] + masks[:6]
        masks = [int.from_bytes(mask.to_bytes(8, "little")[::-1], "little") for mask in masks]
    return tuple(masks)


def encode_classical(
    history: Sequence[HistoryFrame], *, input_format: int = 1, fill: FillHistory = "fen_only"
) -> tuple[Plane, ...]:
    if type(input_format) is not int or input_format != 1:
        raise EncodingError(f"Unsupported input format {input_format}; expected classical format 1")
    if not history or fill not in ("no", "fen_only", "always"):
        raise EncodingError("History must be nonempty and fill must be no, fen_only, or always")
    boards = [_board(frame.fen) for frame in history]
    if any(a.black == b.black for a, b in zip(boards, boards[1:], strict=False)):
        raise EncodingError("Chronological history frames must alternate side to move")
    current = boards[-1]
    planes = [Plane() for _ in range(112)]
    for offset in range(8):
        index = len(history) - 1 - offset
        board = boards[max(0, index)]
        if index < 0:
            if fill == "no" and (index < -1 or board.ep == "-"):
                break
            if fill == "fen_only" and board.start:
                break
        masks = _piece_masks(board, current.black, index < 0)
        planes[offset * 13 : offset * 13 + 12] = [Plane(mask) for mask in masks]
        if history[max(0, index)].repetitions >= 1:
            planes[offset * 13 + 12] = Plane(ALL_SQUARES)
    rights = current.rights[2:] + current.rights[:2] if current.black else current.rights
    for index, allowed in enumerate(rights, 104):
        planes[index] = Plane(ALL_SQUARES if allowed else 0)
    planes[108] = Plane(ALL_SQUARES if current.black else 0)
    # Match the C++ encoder's float32 scalar conversion, including large counters.
    value = struct.unpack("<f", struct.pack("<f", float(current.rule50)))[0]
    planes[109] = Plane(ALL_SQUARES, value)
    planes[111] = Plane(ALL_SQUARES)
    return tuple(planes)


def _square(index: int) -> str:
    return chr(ord("a") + index % 8) + str(index // 8 + 1)


@cache
def policy_moves() -> tuple[str, ...]:
    """Independent rank/file geometry; no copied Lc0 move table."""
    ordinary = []
    for source in range(64):
        for target in range(64):
            dx, dy = abs(source % 8 - target % 8), abs(source // 8 - target // 8)
            if source != target and (dx == 0 or dy == 0 or dx == dy or {dx, dy} == {1, 2}):
                ordinary.append(_square(source) + _square(target))
    promotions = [
        _square(source) + _square(target) + piece
        for source in range(48, 56)
        for target in range(56, 64)
        if abs(source % 8 - target % 8) <= 1
        for piece in "qrb"
    ]
    return tuple(ordinary + promotions)


@cache
def _policy_indices() -> Mapping[str, int]:
    return MappingProxyType({move: index for index, move in enumerate(policy_moves())})


def policy_index(move: str, *, black: bool = False) -> int:
    if not isinstance(move, str) or re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is None:
        raise EncodingError(f"Invalid coordinate move: {move!r}")
    canonical = move
    if black:
        canonical = move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3])) + move[4:]
    if len(canonical) == 5:
        if (
            canonical[1] != "7"
            or canonical[3] != "8"
            or abs(ord(canonical[0]) - ord(canonical[2])) > 1
        ):
            raise EncodingError("Promotion must move from rank 7 to rank 8 within one file")
        if canonical[4] == "n":
            canonical = canonical[:4]
    try:
        return _policy_indices()[canonical]
    except KeyError as error:
        raise EncodingError(f"Move has no policy geometry: {move}") from error


def move_from_policy(index: int, legal_moves: Sequence[str], *, black: bool = False) -> str:
    if type(index) is not int or not 0 <= index < 1858:
        raise EncodingError("Policy index must be 0..1857")
    matches = [move for move in legal_moves if policy_index(move, black=black) == index]
    if len(matches) != 1:
        raise EncodingError("Policy inverse requires exactly one matching legal board move")
    return matches[0]
