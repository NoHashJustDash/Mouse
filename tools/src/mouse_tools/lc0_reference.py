# SPDX-License-Identifier: GPL-3.0-or-later
"""Original strict harness for the pinned actual Lc0 CPU binary and v6 records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from mouse_tools.encoding import ALL_SQUARES, FillHistory, HistoryFrame, Plane
from mouse_tools.fetch_net import T1_SHA256
from mouse_tools.uci_gate import creation_flags

LC0_SHA256 = "e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b"
LC0_COMMIT = "fd71a2d921b689c5f479d3227c3806c8e272d9c5"
RECORD_BYTES = 8356
MAX_TRAINING_BYTES = 64 * 1024 * 1024


class Lc0ReferenceError(RuntimeError):
    """Incomplete, malformed, unsupported, or unpinned reference data."""


def parse_move_stats(
    lines: Sequence[str], *, expected_moves: Sequence[str] | None = None
) -> dict[str, int]:
    moves: dict[str, int] = {}
    counts = []
    bestmoves = []
    for line in lines:
        if line.startswith("info string node"):
            match = re.match(r"info string node\s+\(\s*([0-9]{1,4})\s*\)\s+N:", line)
            if match is None:
                raise Lc0ReferenceError("Malformed root move count")
            counts.append(int(match[1]))
        elif line.startswith("info string ") and re.match(r"[a-h][1-8]", line[12:]):
            match = re.match(
                r"info string ([a-h][1-8][a-h][1-8][qrbn]?)\s+\(\s*([0-9]{1,4})\s*\)\s+N:",
                line,
            )
            if match is None:
                raise Lc0ReferenceError(f"Malformed verbose move row: {line[:120]}")
            move, index = match[1], int(match[2])
            if index >= 1858 or move in moves or index in moves.values():
                raise Lc0ReferenceError("Duplicate move/index or out-of-range NN index")
            moves[move] = index
        elif line.startswith("bestmove "):
            match = re.fullmatch(
                r"bestmove ([a-h][1-8][a-h][1-8][qrbn]?|0000)"
                r"(?: ponder [a-h][1-8][a-h][1-8][qrbn]?)?",
                line,
            )
            if match is None:
                raise Lc0ReferenceError("Malformed bestmove")
            bestmoves.append(match[1])
    if len(bestmoves) != 1 or len(counts) != 1 or counts[0] != len(moves):
        raise Lc0ReferenceError("Incomplete search: require bestmove and every counted legal move")
    if bestmoves[0] != "0000" and bestmoves[0] not in moves:
        raise Lc0ReferenceError("Best move is absent from verbose rows")
    if expected_moves is not None and (
        len(set(expected_moves)) != len(expected_moves) or set(moves) != set(expected_moves)
    ):
        raise Lc0ReferenceError("Verbose rows do not match the complete expected legal move set")
    return moves


@dataclass(frozen=True)
class TrainingRecord:
    planes: tuple[Plane, ...]
    legal_indices: tuple[int, ...]
    played_index: int
    best_index: int
    visits: int


def _reverse_byte_bits(mask: int) -> int:
    result = 0
    for byte in range(8):
        value = (mask >> (8 * byte)) & 255
        reversed_value = sum(((value >> bit) & 1) << (7 - bit) for bit in range(8))
        result |= reversed_value << (8 * byte)
    return result


def parse_training_v6(data: bytes) -> tuple[TrainingRecord, ...]:
    """Decode explicit little-endian v6, never native-aligned structs or old versions.

    Layout facts: pinned lc0 src/trainingdata/trainingdata_v6.h and reader.cc.
    Original q/d/m may legitimately be NaN; policy, indices, flags, and planes may not.
    """
    if not data or len(data) > MAX_TRAINING_BYTES or len(data) % RECORD_BYTES:
        raise Lc0ReferenceError("Training data must contain complete bounded 8356-byte v6 records")
    results = []
    for offset in range(0, len(data), RECORD_BYTES):
        record = data[offset : offset + RECORD_BYTES]
        if struct.unpack_from("<II", record) != (6, 1):
            raise Lc0ReferenceError("Only training v6 with classical input format 1 is supported")
        probabilities = struct.unpack_from("<1858f", record, 8)
        if any(
            not math.isfinite(value) or (value != -1.0 and not 0.0 <= value <= 1.0)
            for value in probabilities
        ):
            raise Lc0ReferenceError("Nonfinite or invalid training policy probability")
        legal = tuple(index for index, value in enumerate(probabilities) if value >= 0.0)
        if not legal or not math.isclose(
            sum(probabilities[index] for index in legal), 1.0, abs_tol=1e-5
        ):
            raise Lc0ReferenceError(
                "Training probabilities must describe a complete normalized policy"
            )
        flags = record[8272:8277]
        if any(flag not in (0, 1) for flag in flags) or record[8278] & 0xC7 or record[8279]:
            raise Lc0ReferenceError("Invalid classical training flags")
        if struct.unpack_from("<I", record, 8352)[0]:
            raise Lc0ReferenceError("Nonzero reserved training field")
        visits, played, best = struct.unpack_from("<IHH", record, 8340)
        if visits == 0 or played not in legal or best not in legal:
            raise Lc0ReferenceError("Invalid training visit count or selected policy index")
        masks = struct.unpack_from("<104Q", record, 7440)
        planes = [Plane(_reverse_byte_bits(mask)) for mask in masks]
        planes.extend(Plane(ALL_SQUARES if flag else 0) for flag in flags)
        planes.extend((Plane(ALL_SQUARES, float(record[8277])), Plane(), Plane(ALL_SQUARES)))
        results.append(TrainingRecord(tuple(planes), legal, played, best, visits))
    return tuple(results)


def read_training_v6(path: Path) -> tuple[TrainingRecord, ...]:
    try:
        with gzip.open(path, "rb") as stream:
            data = stream.read(MAX_TRAINING_BYTES + 1)
    except (OSError, EOFError) as error:
        raise Lc0ReferenceError(f"Corrupt training gzip: {error}") from error
    return parse_training_v6(data)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify(engine: Path, weights: Path) -> None:
    if _hash(engine) != LC0_SHA256 or _hash(weights) != T1_SHA256:
        raise Lc0ReferenceError(
            "Lc0 executable or T1 weights SHA-256 does not match the pinned reference"
        )


def _arguments(engine: Path, weights: Path, fill: FillHistory, mode: str) -> list[str]:
    if fill not in ("no", "fen_only", "always"):
        raise Lc0ReferenceError("History fill must be explicit")
    command = [
        str(engine.resolve()),
        mode,
        "--backend=trivial",
        "--backend-opts=input_mode=1",
        "--weights=" + str(weights.resolve()),
        "--threads=1",
        "--minibatch-size=1",
        "--temperature=0",
        "--noise-epsilon=0",
        "--nncache=0",
        "--history-fill-new=" + fill,
        "--verbose-move-stats",
        "--syzygy-paths=",
    ]
    if mode == "classic":
        command.append("--config=")
    return command


class Lc0Reference:
    """One real CPU process; one caller writes UCI and a reader thread only drains output."""

    def __init__(
        self, engine: Path, weights: Path, *, fill: FillHistory = "fen_only", timeout: float = 30.0
    ) -> None:
        _verify(engine, weights)
        if not math.isfinite(timeout) or timeout <= 0:
            raise Lc0ReferenceError("Timeout must be finite and positive")
        self.command = _arguments(engine, weights, fill, "classic")
        self.timeout = timeout
        self.transcript: list[str] = []
        self.commands: list[str] = []
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()

    def _pump(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        try:
            for line in self._process.stdout:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _send(self, command: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise Lc0ReferenceError("Lc0 process is not running")
        self.commands.append(command)
        try:
            self._process.stdin.write(command + "\n")
            self._process.stdin.flush()
        except OSError as error:
            raise Lc0ReferenceError(f"Lc0 input failed: {error}") from error

    def _until(self, marker: str) -> list[str]:
        deadline = time.monotonic() + self.timeout
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Lc0ReferenceError(f"Lc0 timed out awaiting {marker}")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty as error:
                raise Lc0ReferenceError(f"Lc0 timed out awaiting {marker}") from error
            if line is None:
                raise Lc0ReferenceError(f"Lc0 exited awaiting {marker}: {lines[-5:]}")
            lines.append(line)
            self.transcript.append(line)
            if len(lines) > 100_000:
                raise Lc0ReferenceError("Excessive Lc0 output")
            if line == marker or line.startswith(marker + " "):
                return lines

    def __enter__(self) -> Lc0Reference:
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            creationflags=creation_flags(),
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        try:
            self._send("uci")
            greeting = self._until("uciok")
            if "id name Lc0 v0.32.1" not in greeting or not any(
                line.startswith("option name Backend type combo default trivial ")
                for line in greeting
            ):
                raise Lc0ReferenceError("Unexpected Lc0 identity or explicitly selected backend")
            self._send("setoption name UCI_Chess960 value true")
            self._send("isready")
            self._until("readyok")
        except BaseException:
            self.close()
            raise
        return self

    def policy(self, fen: str, moves: Sequence[str] = ()) -> tuple[dict[str, int], list[str]]:
        HistoryFrame(fen)
        if any(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is None for move in moves):
            raise Lc0ReferenceError("Malformed UCI history move")
        self._send("ucinewgame")
        self._send("position fen " + fen + (" moves " + " ".join(moves) if moves else ""))
        self._send("go nodes 32")
        lines = self._until("bestmove")
        return parse_move_stats(lines), lines

    def close(self) -> None:
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._send("quit")
                    self._process.wait(timeout=5)
            except (Lc0ReferenceError, subprocess.TimeoutExpired):
                self._process.kill()
                self._process.wait(timeout=5)
            finally:
                if self._process.stdin is not None:
                    self._process.stdin.close()
                if self._process.stdout is not None:
                    self._process.stdout.close()
                if self._reader is not None:
                    self._reader.join(timeout=5)
                self._process = None

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def capture_training(
    engine: Path,
    weights: Path,
    fen: str,
    opening_san: Sequence[str],
    output_dir: Path,
    *,
    fill: FillHistory = "fen_only",
    timeout: float = 120.0,
) -> dict[str, object]:
    """Run one complete CPU selfplay game and retain all original output/chunks."""
    _verify(engine, weights)
    HistoryFrame(fen)
    if any(re.fullmatch(r"[a-zA-Z0-9.+=#!?-]+", token) is None for token in opening_san):
        raise Lc0ReferenceError("Invalid PGN opening token")
    if not math.isfinite(timeout) or timeout <= 0:
        raise Lc0ReferenceError("Timeout must be finite and positive")
    output_dir.mkdir(parents=True, exist_ok=False)
    pgn = output_dir.resolve() / "opening.pgn"
    pgn.write_text(
        f'[FEN "{fen}"]\n[SetUp "1"]\n\n' + " ".join(opening_san) + " *\n", encoding="utf-8"
    )
    command = _arguments(engine, weights, fill, "selfplay") + [
        "--training",
        "--games=1",
        "--parallelism=1",
        "--visits=32",
        "--resign-percentage=0",
        "--opening-stop-prob=0",
        "--discarded-start-chance=0",
        "--openings-mode=sequential",
        "--openings-pgn=" + str(pgn),
        "--move-thinking",
        "--no-mirror-openings",
    ]
    environment = dict(os.environ)
    environment["XDG_CACHE_HOME"] = str(output_dir.resolve())
    started = time.monotonic()
    with subprocess.Popen(
        command,
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
        creationflags=creation_flags(),
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            process.kill()
            stdout, stderr = process.communicate()
            (output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
            (output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
            raise
    (output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    if process.returncode != 0:
        raise Lc0ReferenceError(f"Lc0 selfplay exit {process.returncode}: {stderr[-4000:]}")
    files = list(output_dir.rglob("game_000000.gz"))
    if len(files) != 1:
        raise Lc0ReferenceError("Completed selfplay must produce exactly one training chunk")
    records = read_training_v6(files[0])
    result: dict[str, object] = {
        "engine_sha256": LC0_SHA256,
        "source_commit": LC0_COMMIT,
        "weights_sha256": T1_SHA256,
        "backend": "trivial",
        "neural_weights_evaluated": False,
        "command": command,
        "cwd": str(output_dir.resolve()),
        "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "records": len(records),
        "chunk": str(files[0].resolve()),
        "chunk_sha256": _hash(files[0]),
        "pgn_sha256": _hash(pgn),
        "stdout_sha256": _hash(output_dir / "stdout.log"),
        "stderr_sha256": _hash(output_dir / "stderr.log"),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("policy", "training"))
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--fen", required=True)
    parser.add_argument("--moves", nargs="*", default=[])
    parser.add_argument("--fill", choices=("no", "fen_only", "always"), default="fen_only")
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("downloaded reference files must be read with python -I")
    try:
        if arguments.action == "training":
            result = capture_training(
                arguments.engine,
                arguments.weights,
                arguments.fen,
                arguments.moves,
                arguments.output_dir,
                fill=arguments.fill,
            )
            print(f"Captured {result['records']} genuine Lc0 training records")
        else:
            arguments.output_dir.mkdir(parents=True, exist_ok=False)
            with Lc0Reference(
                arguments.engine, arguments.weights, fill=arguments.fill
            ) as reference:
                mapping, lines = reference.policy(arguments.fen, arguments.moves)
                result = {
                    "engine_sha256": LC0_SHA256,
                    "source_commit": LC0_COMMIT,
                    "command": reference.command,
                    "uci_commands": reference.commands,
                    "moves": mapping,
                    "transcript": lines,
                }
            (arguments.output_dir / "policy.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Captured {len(mapping)} genuine Lc0 move/index rows")
    except (Lc0ReferenceError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
