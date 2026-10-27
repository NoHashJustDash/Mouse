# SPDX-License-Identifier: GPL-3.0-or-later
"""Run every original exact mate fixture against an explicit native Mouse backend.

No real engine is used by the unit suite. A passing report means all eighteen
root moves matched the independent minimal-horizon proof at the recorded node
budgets. MouseSearch1 does not expose an exhaustive forced-mate certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from mouse_tools import mate_fixtures as fixtures

MAX_STDOUT = 64 * 1024
MAX_STDERR = 1024 * 1024
UINT64_MAX = (1 << 64) - 1
APPROVED_CORPUS = {
    "mates.epd": "be84dd87800115c6272aff151a4e7bc9baa3385069b0bdf0d48cc5b89febcd97",
    "mates.json": "f1f97847789db7d1a901a93172f6f7aa3c4f2ddc1d53a398fbe4a098a315941f",
    "oracle-states.jsonl.gz": "fc547ba4453a7f68caf03f243b40e996a520aef20db3a0ff51b8aa93b727e3c2",
    "provenance.json": "583e78d16a0022dc2092180cdfa3fd05f05535888219230949a91d5c9927f66b",
    "uci-handshake.txt": "62171394a8c935c7c2a1efe26ef9a4dca6eb0ee81be0db3b31ab196c8f2d3123",
    "README.md": "6ead203bacf651a7d5347367ef17d555eb8925ff3f4204f57dccdb0bf5b199bf",
    "live-verification.json": "f58891e5b30eca5ddbe00af37f742e12b5777c8c569a705aae92c70fcb39773f",
}
PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}
REQUIRED_FIELDS = frozenset(
    (
        "schema",
        "version",
        "git",
        "provider",
        "precision",
        "model_sha256",
        "input_sha256",
        "fen",
        "deterministic",
        "threads",
        "batch",
        "nncache_bytes",
        "stop_reason",
        "nodes",
        "new_playouts",
        "inherited_visits",
        "depth",
        "seldepth",
        "q",
        "draw",
        "elapsed_ms",
        "nps",
        "charged_bytes",
        "budget_bytes",
        "tree_nodes",
        "tree_capacity",
        "bestmove",
        "pv",
    )
)


class SearchGateError(ValueError):
    """Invalid configuration/evidence never becomes a shortened successful gate."""


@dataclass(frozen=True)
class MateConfig:
    engine: Path
    weights: Path
    cases: Path
    backend: str
    precision: str
    mate1_nodes: int = 128
    mate2_nodes: int = 1024
    mate3_nodes: int = 8192
    batch: int = 8
    ram_limit_mb: int = 64
    timeout_seconds: float = 900.0
    gpu_dll_dirs: tuple[Path, ...] = ()
    cache_dir: Path | None = None

    def budget(self, mate: int) -> int:
        return (self.mate1_nodes, self.mate2_nodes, self.mate3_nodes)[mate - 1]

    def validate(self) -> None:
        if self.backend not in PROVIDERS or self.precision not in ("fp32", "fp16"):
            raise SearchGateError("Backend/precision must be explicit supported values")
        for name, value, maximum in (
            ("mate1 nodes", self.mate1_nodes, UINT64_MAX),
            ("mate2 nodes", self.mate2_nodes, UINT64_MAX),
            ("mate3 nodes", self.mate3_nodes, UINT64_MAX),
            ("batch", self.batch, 1024),
            ("RAM MiB", self.ram_limit_mb, 16384),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise SearchGateError(f"{name} must be an integer in 1..{maximum}")
        if not 0 < self.timeout_seconds <= 86400 or not math.isfinite(self.timeout_seconds):
            raise SearchGateError("Per-child timeout must be in (0,86400] seconds")
        if self.backend == "cpu":
            if self.precision != "fp32" or self.gpu_dll_dirs or self.cache_dir is not None:
                raise SearchGateError("CPU gate requires fp32 and no GPU directory/cache options")
        else:
            if not self.gpu_dll_dirs or self.cache_dir is None:
                raise SearchGateError(
                    "GPU gate requires explicit SDK directories and cache directory"
                )
            if not self.cache_dir.is_absolute():
                raise SearchGateError("GPU cache directory must be absolute")
            if any(
                not directory.is_absolute() or not directory.is_dir()
                for directory in self.gpu_dll_dirs
            ):
                raise SearchGateError("GPU SDK directories must exist and be absolute")
            if len({directory.resolve() for directory in self.gpu_dll_dirs}) != len(
                self.gpu_dll_dirs
            ):
                raise SearchGateError("GPU SDK directories must be distinct")


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    elapsed_seconds: float
    timed_out: bool = False
    output_limit: bool = False
    error: str | None = None
    interrupted: bool = False


def run_child(command: list[str], timeout: float) -> ProcessResult:
    """Bound both pipes and lifetime; cancellation kills/reaps the actual child."""
    started = time.monotonic()
    buffers = [bytearray(), bytearray()]
    exceeded, reader_error = threading.Event(), threading.Event()
    environment = os.environ.copy()
    environment["PATH"] = ""
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(command[0]).resolve().parent),
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as error:
        return ProcessResult(None, b"", b"", time.monotonic() - started, error=str(error))

    def drain(stream: IO[bytes], index: int, cap: int) -> None:
        try:
            while chunk := stream.read(4096):
                remaining = cap - len(buffers[index])
                buffers[index].extend(chunk[: max(remaining, 0)])
                if len(chunk) > remaining:
                    exceeded.set()
                    return
        except (OSError, ValueError):
            reader_error.set()

    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait(timeout=5)
        return ProcessResult(
            process.returncode,
            b"",
            b"",
            time.monotonic() - started,
            error="Child output pipes unavailable",
        )
    threads = [
        threading.Thread(target=drain, args=(stream, index, cap), daemon=True)
        for index, (stream, cap) in enumerate(
            ((process.stdout, MAX_STDOUT), (process.stderr, MAX_STDERR))
        )
    ]
    running_threads: list[threading.Thread] = []
    timed_out = interrupted = False
    issue: str | None = None
    try:
        for thread in threads:
            thread.start()
            running_threads.append(thread)
        while process.poll() is None:
            remaining = timeout - (time.monotonic() - started)
            if exceeded.is_set() or reader_error.is_set() or remaining <= 0:
                timed_out = remaining <= 0
                break
            try:
                process.wait(timeout=min(remaining, 0.02))
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        interrupted = True
        issue = "Search child interrupted"
    except (OSError, RuntimeError) as error:
        issue = f"Cannot wait for search child: {error}"
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except OSError as error:
                issue = f"Cannot terminate search child: {error}"
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as error:
            issue = f"Cannot reap search child: {error}"
        for thread in running_threads:
            thread.join(timeout=5)
        alive = any(thread.is_alive() for thread in running_threads)
        # Closing a BufferedReader while a descendant retains the pipe can block.
        # Never treat an incompletely drained/closed child as successful evidence.
        if not alive:
            process.stdout.close()
            process.stderr.close()
        if alive or reader_error.is_set():
            issue = "Cannot completely drain child output"
    return ProcessResult(
        process.returncode,
        bytes(buffers[0]),
        bytes(buffers[1]),
        time.monotonic() - started,
        timed_out,
        exceeded.is_set(),
        issue,
        interrupted,
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _library(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and (name.endswith((".dll", ".dylib", ".so")) or ".so." in name)


def _libraries(config: MateConfig) -> dict[str, Path]:
    result = {
        "runtime/" + path.name: path
        for path in config.engine.resolve().parent.iterdir()
        if _library(path)
    }
    if not any(
        path.name.lower() == "onnxruntime.dll" or path.name.lower().startswith("libonnxruntime.")
        for path in result.values()
    ):
        raise SearchGateError("Engine must have an app-local ONNX Runtime library")
    for index, directory in enumerate(config.gpu_dll_dirs):
        for path in directory.iterdir():
            if _library(path):
                result[f"sdk{index}/" + path.name] = path
    if len(result) > 4096:
        raise SearchGateError("Runtime library inventory exceeds 4096 files")
    return result


@dataclass(frozen=True)
class Snapshot:
    paths: dict[str, Path]
    hashes: dict[str, str]
    library_inventory: dict[str, str]
    corpus_inventory: tuple[str, ...]

    @classmethod
    def freeze(cls, config: MateConfig) -> Snapshot:
        libraries = _libraries(config)
        corpus = config.cases.resolve()
        names = tuple(sorted(path.name for path in corpus.iterdir()))
        if names != tuple(sorted(APPROVED_CORPUS)):
            raise SearchGateError("Require every original fixed corpus file and no extra members")
        source = Path(__file__).resolve()
        paths = {
            "engine": config.engine.resolve(),
            "model": config.weights.resolve(),
            **{"case/" + name: corpus / name for name in APPROVED_CORPUS},
            **libraries,
            **{
                "source/" + name: source.with_name(name)
                for name in ("search_gate.py", "mate_fixtures.py", "perft_reference.py")
            },
        }
        if (
            paths["engine"].stat().st_size > 256 * 1024 * 1024
            or paths["model"].stat().st_size > 512 * 1024 * 1024
        ):
            raise SearchGateError("Engine/model exceeds bounded artifact size")
        # Freeze all identities before reading JSON, model content, or starting a child.
        hashes = {name: _sha256(path) for name, path in paths.items()}
        for name, wanted in APPROVED_CORPUS.items():
            if hashes["case/" + name] != wanted:
                raise SearchGateError(f"Original fixed corpus SHA-256 mismatch: {name}")
        return cls(
            paths, hashes, {name: str(path.resolve()) for name, path in libraries.items()}, names
        )

    def verify(self, config: MateConfig) -> None:
        try:
            libraries = _libraries(config)
            inventory = {name: str(path.resolve()) for name, path in libraries.items()}
            names = tuple(sorted(path.name for path in config.cases.resolve().iterdir()))
            if inventory != self.library_inventory or names != self.corpus_inventory:
                raise SearchGateError("Runtime/corpus inventory changed during mate gate")
            for name, path in self.paths.items():
                if _sha256(path) != self.hashes[name]:
                    raise SearchGateError(f"{name} artifact changed during mate gate")
        except OSError as error:
            raise SearchGateError(f"Frozen artifact changed or disappeared: {error}") from error


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SearchGateError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise SearchGateError("Nonfinite JSON number")
    return value


def _nonfinite(token: str) -> None:
    raise SearchGateError(f"Nonfinite JSON constant: {token}")


def parse_json(data: bytes) -> object:
    if len(data) > MAX_STDOUT:
        raise SearchGateError("Search JSON exceeds output limit")
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_float=_finite_float,
            parse_constant=_nonfinite,
        )
        stack: list[tuple[object, int]] = [(value, 0)]
        count = 0
        while stack:
            item, depth = stack.pop()
            count += 1
            if depth > 16 or count > 8192:
                raise SearchGateError("Search JSON nesting depth/item count exceeds bounds")
            if isinstance(item, dict):
                stack.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                stack.extend((child, depth + 1) for child in item)
        return value
    except (UnicodeError, ValueError, RecursionError) as error:
        raise SearchGateError(f"Invalid complete search JSON: {error}") from error


@dataclass(frozen=True)
class MateCase:
    identifier: str
    fen: str
    mate: int
    winning_moves: tuple[str, ...]


def load_cases(root: Path) -> tuple[MateCase, ...]:
    data = parse_json((root / "mates.json").read_bytes())
    fixtures.validate_corpus(data)
    if not isinstance(data, dict):
        raise SearchGateError("Mate JSON must be an object")
    if (root / "mates.epd").read_text(encoding="utf-8") != fixtures.render_epd(data):
        raise SearchGateError("Exact EPD/JSON correspondence failed")
    values = data["cases"]
    if not isinstance(values, list):
        raise SearchGateError("Missing mate cases")
    return tuple(
        MateCase(value["id"], value["fen"], value["mate"], tuple(value["winning_moves"]))
        for value in values
    )


def _move(fen: str, move: str) -> str:
    """Independent pawnless move geometry plus own-king safety for PV auditing."""
    if fixtures.MOVE.fullmatch(move) is None:
        raise SearchGateError("PV requires supported coordinate moves")
    board = fixtures._board(fen)
    source = (int(move[1]) - 1) * 8 + ord(move[0]) - 97
    target = (int(move[3]) - 1) * 8 + ord(move[2]) - 97
    if source not in board:
        raise SearchGateError("PV mover is missing")
    dx, dy = target % 8 - source % 8, target // 8 - source // 8
    ax, ay = abs(dx), abs(dy)
    kind = board[source].upper()
    if kind == "K":
        valid = max(ax, ay) == 1
    elif kind == "N":
        valid = {ax, ay} == {1, 2}
    else:
        valid = source != target and (
            (kind in "RQ" and (dx == 0 or dy == 0)) or (kind in "BQ" and ax == ay)
        )
        if valid:
            sx, sy = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
            x, y = source % 8 + sx, source // 8 + sy
            while (x, y) != (target % 8, target // 8):
                if y * 8 + x in board:
                    valid = False
                    break
                x, y = x + sx, y + sy
    if not valid:
        raise SearchGateError("Illegal PV piece geometry or blocked ray")
    try:
        result = fixtures._play(fen, move)
    except fixtures.MateError as error:
        raise SearchGateError(str(error)) from error
    after = fixtures._board(result)
    white = fen.split()[1] == "w"
    king = next(square for square, piece in after.items() if piece == ("K" if white else "k"))
    if fixtures._attacked(after, king, not white):
        raise SearchGateError("PV leaves its own king in check")
    return result


def _legal_moves(fen: str) -> tuple[str, ...]:
    board = fixtures._board(fen)
    moves = []
    for source, piece in board.items():
        if piece.isupper() != (fen.split()[1] == "w"):
            continue
        for target in range(64):
            text = (
                chr(97 + source % 8)
                + str(source // 8 + 1)
                + chr(97 + target % 8)
                + str(target // 8 + 1)
            )
            try:
                _move(fen, text)
            except SearchGateError:
                continue
            moves.append(text)
    return tuple(sorted(moves))


def _integer(value: object, name: str, minimum: int, maximum: int = UINT64_MAX) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise SearchGateError(f"{name} requires an integer in {minimum}..{maximum}")
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SearchGateError(f"{name} requires a finite value in [{minimum},{maximum}]")
    # Compare as integers first: float(huge_json_integer) itself can overflow.
    if not minimum <= value <= maximum or not math.isfinite(value):
        raise SearchGateError(f"{name} requires a finite value in [{minimum},{maximum}]")
    return float(value)


def _wdl_pair(q_value: object, draw_value: object, name: str, visits: int) -> None:
    q = _number(q_value, f"{name} Q", -1, 1)
    draw = _number(draw_value, f"{name} draw", 0, 1)
    excess = abs(q) + draw - 1
    if excess <= 0:
        return
    # search.cpp normalizes float WDL in double, adds each sample to double
    # value_sum/draw_sum, then divides by visits. The absolute forward error
    # of these means is bounded by gamma_(visits+8), gamma_k=k*u/(1-k*u),
    # u=epsilon/2. Eight extra roundings cover normalization, mean division,
    # and the simplex sum. This is binary64 arithmetic, not NN/FP16 tolerance.
    # The pinned pawnless corpus reaches a rule terminal before the depth
    # ceiling, so it cannot replay float-rounded nonterminal Frontier values.
    scaled_roundoff = (visits + 8) * (sys.float_info.epsilon / 2)
    # A bound >=1 cannot distinguish an impossible pair; fail closed then.
    if scaled_roundoff >= 0.5 or excess > scaled_roundoff / (1 - scaled_roundoff):
        raise SearchGateError(f"{name} Q/draw violate the probability simplex")


def validate_result(
    value: object, case: MateCase, config: MateConfig, model_sha256: str
) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) - {"root_moves", "moves_left_utility"} != REQUIRED_FIELDS
    ):
        raise SearchGateError("Require the complete strict MouseSearch1 field set")
    wanted = {
        "schema": "MouseSearch1",
        "provider": PROVIDERS[config.backend],
        "precision": config.precision,
        "model_sha256": model_sha256,
        "input_sha256": hashlib.sha256(
            ("MouseSearchInput1\nchess960=0\n" + case.fen + "\n").encode()
        ).hexdigest(),
        "fen": case.fen,
        "stop_reason": "nodes",
    }
    if any(value.get(key) != expected for key, expected in wanted.items()):
        raise SearchGateError(
            "Search schema/provider/precision/model/input/FEN/stop identity mismatch"
        )
    if "moves_left_utility" in value and value["moves_left_utility"] is not True:
        raise SearchGateError("Moves-left utility differs from the requested default mode")
    if value["deterministic"] is not True:
        raise SearchGateError("Deterministic search was not reported")
    for name in ("version", "git"):
        item = value[name]
        if (
            not isinstance(item, str)
            or not 1 <= len(item) <= 128
            or not item.isascii()
            or any(ord(character) < 32 for character in item)
        ):
            raise SearchGateError("Missing/malformed build version or git identity")
    for name, expected in (
        ("threads", 1),
        ("batch", config.batch),
        ("nncache_bytes", 0),
        ("nodes", config.budget(case.mate)),
        ("new_playouts", config.budget(case.mate)),
        ("inherited_visits", 0),
        ("budget_bytes", config.ram_limit_mb * 1024 * 1024),
    ):
        if _integer(value[name], name, 0) != expected:
            raise SearchGateError(f"Search {name} differs from explicit fixed-node configuration")
    depth = _integer(value["depth"], "depth", 1, 512)
    if _integer(value["seldepth"], "seldepth", depth, 512) < depth:
        raise SearchGateError("Search seldepth is below PV depth")
    _integer(value["elapsed_ms"], "elapsed_ms", 0, (1 << 63) - 1)
    _wdl_pair(value["q"], value["draw"], "Search", config.budget(case.mate))
    _number(value["nps"], "nps", 0, sys.float_info.max)
    _integer(value["charged_bytes"], "charged_bytes", 1, config.ram_limit_mb * 1024 * 1024)
    _integer(value["tree_nodes"], "tree_nodes", 1)
    _integer(value["tree_capacity"], "tree_capacity", value["tree_nodes"])
    best, pv = value["bestmove"], value["pv"]
    if not isinstance(best, str) or fixtures.MOVE.fullmatch(best) is None:
        raise SearchGateError("Missing/malformed nonterminal best move")
    if (
        not isinstance(pv, list)
        or len(pv) != depth
        or not pv
        or pv[0] != best
        or any(not isinstance(move, str) for move in pv)
    ):
        raise SearchGateError("PV must be complete, bounded, and start with bestmove")
    fen = case.fen
    for move in pv:
        fen = _move(fen, move)
    if "root_moves" in value:
        rows = value["root_moves"]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 218:
            raise SearchGateError("Malformed optional root_moves array")
        legal = set(_legal_moves(case.fen))
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"move", "visits", "q", "draw", "prior"}:
                raise SearchGateError("Malformed root move diagnostic row")
            move = row["move"]
            if not isinstance(move, str) or move in seen or move not in legal:
                raise SearchGateError("Duplicate/illegal root move diagnostic")
            seen.add(move)
            visits = _integer(row["visits"], "root visits", 0, config.budget(case.mate))
            _wdl_pair(row["q"], row["draw"], "Root", visits)
            _number(row["prior"], "root prior", 0, 1)
        if seen != legal:
            raise SearchGateError("Optional root move array omitted legal moves")
    return value


def _command(config: MateConfig, case: MateCase) -> list[str]:
    buckets = [value for value in (1, 8, 32) if value < config.batch] + [config.batch]
    result = [
        str(config.engine.resolve()),
        "search",
        "--weights",
        str(config.weights.resolve()),
        "--backend",
        config.backend,
        "--precision",
        config.precision,
        "--fen",
        case.fen,
        "--nodes",
        str(config.budget(case.mate)),
        "--threads",
        "1",
        "--batch",
        str(config.batch),
        "--ram-limit-mb",
        str(config.ram_limit_mb),
        "--cpu-threads",
        "1",
        "--nncache-mb",
        "0",
        "--buckets",
        ",".join(map(str, buckets)),
        "--deterministic",
        "--no-smart-pruning",
        "--cpuct",
        "1.745",
        "--cpuct-base",
        "38739",
        "--cpuct-factor",
        "3.894",
        "--fpu-value",
        "0.330",
        "--policy-temperature",
        "1.359",
    ]
    for directory in config.gpu_dll_dirs:
        result.extend(("--gpu-dll-dir", str(directory.resolve())))
    if config.cache_dir is not None:
        result.extend(("--cache-dir", str(config.cache_dir.resolve())))
    return result


def _publish_new(destination: Path, report: dict[str, object]) -> None:
    """Atomic no-clobber publication; racing successful evidence stays intact."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _requested(value: int | float) -> int | float | str:
    """Preserve invalid nonfinite CLI requests in a strict JSON failure report."""
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def run_mates(config: MateConfig, output: Path) -> dict[str, object]:
    started = time.monotonic()
    checks: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema": 1,
        "gate": "v0.4-mates-v1",
        "status": "failed",
        "passed": False,
        "planned_cases": 18,
        "attempted_cases": 0,
        "passed_cases": 0,
        "checks": checks,
        "backend": config.backend,
        "precision": config.precision,
        "node_budgets": {
            "mate1": _requested(config.mate1_nodes),
            "mate2": _requested(config.mate2_nodes),
            "mate3": _requested(config.mate3_nodes),
        },
        "batch": _requested(config.batch),
        "ram_limit_mb": _requested(config.ram_limit_mb),
        "timeout_seconds": _requested(config.timeout_seconds),
        "search_proved_forced_mate": False,
        "hashes_unchanged": False,
        "fixture_proof": "original pinned independent Stockfish legal-oracle AND/OR corpus",
        "child_environment": {"PATH": ""},
        "artifact_sha256": {},
        "error": None,
        "report_path": None,
    }
    destination = output
    safe_destination = False
    snapshot: Snapshot | None = None
    try:
        destination = output.resolve()
        if (
            destination in (config.engine.resolve(), config.weights.resolve())
            or destination.is_relative_to(config.cases.resolve())
            or not destination.parent.is_dir()
            or destination.is_dir()
        ):
            raise SearchGateError(
                "Report destination overlaps an input or is not a valid file parent"
            )
        safe_destination = True
        config.validate()
        if destination.exists():
            raise SearchGateError(
                "Report output must be new; successful evidence is never overwritten"
            )
        snapshot = Snapshot.freeze(config)
        if destination in (path.resolve() for path in snapshot.paths.values()):
            safe_destination = False
            raise SearchGateError("Report destination overlaps a protected artifact")
        report["artifact_sha256"] = dict(snapshot.hashes)
        report["artifact_paths"] = {
            name: str(path.resolve()) for name, path in snapshot.paths.items()
        }
        cases = load_cases(config.cases.resolve())
        build_identity: tuple[object, object] | None = None
        for case in cases:
            command = _command(config, case)
            row: dict[str, object] = {
                "id": case.identifier,
                "fen": case.fen,
                "mate": case.mate,
                "node_budget": config.budget(case.mate),
                "winning_moves": list(case.winning_moves),
                "command": command,
                "passed": False,
                "winning_root_move_verified": False,
                "pv_terminal_checkmate": False,
                "error": None,
                "result": None,
                "stdout": "",
                "stderr": "",
            }
            checks.append(row)
            try:
                child = run_child(command, config.timeout_seconds)
                row.update(
                    {
                        "returncode": child.returncode,
                        "stdout": child.stdout.decode("utf-8", errors="replace"),
                        "stderr": child.stderr.decode("utf-8", errors="replace"),
                        "wall_seconds": child.elapsed_seconds,
                        "timed_out": child.timed_out,
                        "interrupted": child.interrupted,
                        "captured_stdout_sha256": hashlib.sha256(child.stdout).hexdigest(),
                        "captured_stderr_sha256": hashlib.sha256(child.stderr).hexdigest(),
                        "output_limit": child.output_limit,
                    }
                )
                if child.interrupted:
                    row["error"] = "Search child interrupted"
                    raise KeyboardInterrupt
                if (
                    child.error
                    or child.timed_out
                    or child.output_limit
                    or child.returncode != 0
                    or len(child.stdout) > MAX_STDOUT
                    or len(child.stderr) > MAX_STDERR
                ):
                    raise SearchGateError(
                        f"Child search failed: exit={child.returncode}, timeout={child.timed_out}, "
                        f"output_limit={child.output_limit}, error={child.error}"
                    )
                value = parse_json(child.stdout)
                row["result"] = value
                observed = validate_result(value, case, config, snapshot.hashes["model"])
                identity = (observed["version"], observed["git"])
                if build_identity is not None and identity != build_identity:
                    raise SearchGateError("Child build identity changed across cases")
                build_identity = identity
                best = observed["bestmove"]
                final_fen = case.fen
                pv = observed["pv"]
                if not isinstance(pv, list):
                    raise SearchGateError("Missing validated PV")
                for move in pv:
                    final_fen = _move(final_fen, move)
                board = fixtures._board(final_fen)
                white = final_fen.split()[1] == "w"
                king = next(
                    square for square, piece in board.items() if piece == ("K" if white else "k")
                )
                row["pv_terminal_checkmate"] = (
                    final_fen.split()[1] != case.fen.split()[1]
                    and fixtures._attacked(board, king, not white)
                    and not _legal_moves(final_fen)
                )
                row["winning_root_move_verified"] = best in case.winning_moves
                if not row["winning_root_move_verified"]:
                    raise SearchGateError(
                        "Best move is not in the exact minimal-horizon winning root set"
                    )
                row["passed"] = True
            except (SearchGateError, fixtures.MateError, OSError, ValueError) as error:
                row["error"] = str(error)
        snapshot.verify(config)
        report["hashes_unchanged"] = True
        report["attempted_cases"] = len(checks)
        report["passed_cases"] = sum(row["passed"] is True for row in checks)
        if len(checks) == 18 and report["passed_cases"] == 18:
            report["status"] = "passed"
            report["passed"] = True
        else:
            report["error"] = "Full mate gate failed; every attempted case is retained"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Mate gate interrupted; partial results cannot pass"
    except (SearchGateError, fixtures.MateError, OSError, ValueError) as error:
        report["error"] = str(error)
    if snapshot is not None and not report["hashes_unchanged"]:
        for row in checks:
            row["observed_passed"] = row["passed"]
            row["passed"] = False
            row["error"] = f"{row['error'] or ''}; frozen artifacts were not verified unchanged"
    report["attempted_cases"] = len(checks)
    report["passed_cases"] = sum(row["passed"] is True for row in checks)
    report["elapsed_seconds"] = time.monotonic() - started
    if safe_destination:
        report["report_path"] = str(destination)
        try:
            _publish_new(destination, report)
        except (OSError, ValueError) as error:
            report["status"], report["passed"] = "failed", False
            report["error"] = f"Cannot publish new report without replacing evidence: {error}"
            fallback = destination.with_name(
                destination.name + ".failed-" + uuid.uuid4().hex + ".json"
            )
            report["report_path"] = str(fallback)
            try:
                _publish_new(fallback, report)
            except (OSError, ValueError) as failure:
                report["report_path"] = None
                report["error"] = f"{report['error']}; failure report also unavailable: {failure}"
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    mates = sub.add_parser("mates", help="Run the complete fixed eighteen-position mate gate")
    for name in ("engine", "weights", "cases", "output"):
        mates.add_argument("--" + name, type=Path, required=True)
    mates.add_argument("--backend", choices=tuple(PROVIDERS), required=True)
    mates.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    for distance, default in ((1, 128), (2, 1024), (3, 8192)):
        mates.add_argument(f"--mate{distance}-nodes", type=int, default=default)
    mates.add_argument("--batch", type=int, default=8)
    mates.add_argument("--ram-limit-mb", type=int, default=64)
    mates.add_argument("--timeout", type=float, default=900.0)
    mates.add_argument("--gpu-dll-dir", type=Path, action="append", default=[])
    mates.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        print(
            "error: downloaded-file readers require python -I -m mouse_tools.search_gate",
            file=sys.stderr,
        )
        return 2
    config = MateConfig(
        args.engine,
        args.weights,
        args.cases,
        args.backend,
        args.precision,
        args.mate1_nodes,
        args.mate2_nodes,
        args.mate3_nodes,
        args.batch,
        args.ram_limit_mb,
        args.timeout,
        tuple(args.gpu_dll_dir),
        args.cache_dir,
    )
    report = run_mates(config, args.output)
    if report["passed"] is not True:
        print(f"error: {report['error']}; report={report['report_path']}", file=sys.stderr)
        return 130 if report["status"] == "interrupted" else 1
    print(f"Verified all 18 mate root moves; report={report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
