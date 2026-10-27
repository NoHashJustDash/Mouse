# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, hash-bound fastchess match plans and retained paired-game evidence.

Only explicit gate profiles can establish a roadmap match gate. Real execution is opt-in;
``plan`` performs file validation and emits the entire command without a child.
Fastchess behavior references pin f618e34540f94f4719ad3817950618dabe441318.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO, cast

from . import uci_gate

FASTCHESS_COMMIT = "f618e34540f94f4719ad3817950618dabe441318"
FASTCHESS_VERSION = "v1.8.2-alpha"
LC0_RELEASE_SHA256 = "e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b"
LC0_SOURCE_COMMIT = "fd71a2d921b689c5f479d3227c3806c8e272d9c5"
T1_SHA256 = "bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406"
FASTCHESS_SHA256 = "c6d7e4458a58025983c81bf38c524cb213e6f27559657dcfd72543392a431a71"
BOOKS = {
    "8moves_v3.pgn": (
        "5835239f88cc2c7511b177c32392a69f3ede21819cf0616f80a7f907cd21d17e",
        "pgn",
        34700,
    ),
    "UHO_Lichess_4852_v1.epd": (
        "7a7f6470615a69c6cf23d565417701d38732876f480af90d67b42abade35644a",
        "epd",
        2632036,
    ),
}
SHA = re.compile(r"[0-9a-f]{64}")
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
NORMAL_REASONS = {
    "White mates",
    "Black mates",
    "Draw by stalemate",
    "Draw by insufficient mating material",
    "Draw by 3-fold repetition",
    "Draw by fifty moves rule",
}
ADJUDICATED_RESULTS = {
    "White wins by adjudication": "1-0",
    "Black wins by adjudication": "0-1",
    "Draw by adjudication": "1/2-1/2",
}
RESULTS = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


class MatchError(ValueError):
    """Incomplete, invalid, changed or unbounded evidence cannot establish a gate."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise MatchError(message)


def _object(value: object) -> dict[str, object]:
    require(
        isinstance(value, dict) and all(isinstance(k, str) for k in value), "Expected JSON object"
    )
    return cast(dict[str, object], value)


def _strings(value: object) -> list[str]:
    require(
        isinstance(value, list) and all(isinstance(v, str) for v in value), "Expected string list"
    )
    result = cast(list[str], value)
    require(
        all(len(v) <= 16384 and not any(c in v for c in "\x00\r\n") for v in result),
        "Invalid argument text",
    )
    return result


def _pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise MatchError(f"Nonfinite JSON number: {value}")


def _json(path: Path, maximum: int = 4 * 1024 * 1024) -> dict[str, object]:
    require(path.is_file() and path.stat().st_size <= maximum, "Missing/oversized JSON evidence")
    return _object(
        json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
        )
    )


def _path(value: object) -> Path:
    require(
        isinstance(value, str) and value and not any(c in value for c in '\x00\r\n"'),
        "Invalid absolute artifact path",
    )
    path = Path(cast(str, value))
    require(path.is_absolute(), "Require absolute artifact paths")
    return path.resolve()


def _integer(value: object, low: int, high: int, label: str) -> int:
    require(type(value) is int and low <= value <= high, f"Invalid {label}")
    return cast(int, value)


def _positive(value: object, label: str, maximum: float) -> float:
    require(
        type(value) in (int, float)
        and 0 < cast(float, value) <= maximum
        and math.isfinite(cast(float, value)),
        f"Invalid finite {label}",
    )
    return float(cast(float, value))


def _numeric_option(value: str | None, expected: str) -> bool:
    try:
        return (
            value is not None and Decimal(value).is_finite() and Decimal(value) == Decimal(expected)
        )
    except InvalidOperation:
        return False


def _sha(path: Path) -> str:
    size = path.stat().st_size
    require(size <= 4 * 1024**3, "Input artifact exceeds4GiB hash bound")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            require(block, "Input artifact truncated while hashing")
            digest.update(block)
            remaining -= len(block)
        require(not stream.read(1), "Input artifact grew while hashing")
    return digest.hexdigest()


@dataclass(frozen=True)
class Profile:
    name: str
    games: int
    tc: str | None
    nodes: int | None = None
    sprt: tuple[int, int] | None = None
    book: str = "8moves_v3.pgn"
    concurrency: int = 1
    adjudication: bool = False


def profile(name: str, *, max_games: int | None = None, nodes: int | None = None) -> Profile:
    fixed = {
        "package-smoke": (10, "1+0.01"),
        "selfplay-fast": (1000, "1+0.01"),
        "uci-stress-fast": (2000, "1+0.01"),
        "uci-stress-stc": (500, "10+0.1"),
        "tm-safety-fast": (2000, "1+0.01"),
        "tm-safety-stc": (2000, "10+0.1"),
        "tm-safety-ltc": (2000, "60+0.6"),
        # Explicit 2026-10-18 user FAST TRACK; retain the original full gates.
        "v1-fast-track-fast": (1000, "1+0.01"),
        "v1-fast-track-stc": (200, "10+0.1"),
        "v1-fast-track-ltc": (40, "60+0.6"),
    }
    if name in fixed:
        require(
            max_games is None and nodes is None,
            "Fixed profiles cannot change game counts or limits",
        )
        return Profile(name, *fixed[name])
    require(name in ("parity-sprt", "cache-sprt", "tm-sprt"), "Unknown match profile")
    games = _integer(max_games, 2, 1000000, "predeclared SPRT game cap")
    require(games % 2 == 0, "SPRT cap must contain complete color pairs")
    if name == "parity-sprt":
        return Profile(
            name,
            games,
            None,
            _integer(nodes, 1, 2**63 - 1, "fixed node limit"),
            (-10, 0),
            concurrency=4,
            adjudication=True,
        )
    if name == "cache-sprt":
        require(type(nodes) is int and nodes == 800, "Cache non-regression requires800 fixed nodes")
        return Profile(name, games, None, nodes, (-5, 0), adjudication=True)
    require(nodes is None, "Timed TM SPRT cannot add node limits")
    return Profile(name, games, "10+0.1", None, (0, 5), "UHO_Lichess_4852_v1.epd")


@dataclass(frozen=True)
class Engine:
    name: str
    kind: str
    executable: Path
    model: Path
    source_net: Path
    args: tuple[str, ...]
    options: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Plan:
    config: Path
    selected: Profile
    fastchess: Path
    book: Path
    engines: tuple[Engine, Engine]
    runtime_directories: tuple[Path, ...]
    source_roots: tuple[Path, ...]
    hashes: tuple[tuple[Path, str], ...]
    idle_seconds: float
    total_seconds: float
    max_output_bytes: int

    def inventories(self) -> set[Path]:
        libraries = {
            p.resolve()
            for root in self.runtime_directories
            for p in root.iterdir()
            if p.is_file() and (p.suffix.lower() in (".dll", ".dylib") or ".so" in p.name)
        }
        sources = {
            p.resolve()
            for root in self.source_roots
            for p in root.rglob("*")
            if p.is_file() and p.suffix in (".cpp", ".h", ".cc", ".hpp")
        }
        require(
            len(libraries) <= 512 and 0 < len(sources) <= 4096,
            "Runtime/source inventory bound exceeded",
        )
        artifacts = {self.fastchess, self.book} | {
            p for e in self.engines for p in (e.executable, e.model, e.source_net)
        }
        return libraries | sources | artifacts

    def verify(self) -> None:
        declared = {p for p, _ in self.hashes}
        owned = {
            self.config,
            Path(__file__).resolve(),
            Path(uci_gate.__file__).resolve(),
            Path(sys.executable).resolve(),
        }
        require(
            self.inventories() | owned == declared,
            "Complete runtime/source/artifact inventory changed",
        )
        for path, digest in self.hashes:
            require(
                path.is_file() and _sha(path) == digest, f"Frozen artifact hash changed: {path}"
            )


def _engine(value: object, *, reference_auto_threads: bool = False) -> Engine:
    row = _object(value)
    require(
        set(row) == {"name", "kind", "executable", "model", "source_net", "args", "options"},
        "Unknown/missing engine fields",
    )
    name, kind = str(row["name"]), str(row["kind"])
    require(NAME.fullmatch(name) and kind in ("mouse", "lc0"), "Invalid engine name/kind")
    require(
        all(isinstance(v, str) for v in _object(row["options"]).values()),
        "UCI option values must be explicit strings",
    )
    options = {key: str(value) for key, value in _object(row["options"]).items()}
    require(
        all(
            NAME.fullmatch(key) and not any(c in value for c in "\x00\n\r") and len(value) <= 16384
            for key, value in options.items()
        ),
        "Invalid UCI option text",
    )
    require(
        all(
            options.get(k) == v
            for k, v in {"Ponder": "false", "MultiPV": "1", "UCI_Chess960": "false"}.items()
        ),
        "Explicit single-PV/no-ponder/standard-chess options required",
    )
    require(
        all(re.fullmatch(r"[0-9]{1,10}", options.get(k, "")) for k in ("Threads", "NNCacheSize")),
        "Explicit integer thread/cache settings required",
    )
    require(
        int(options["Threads"]) > 0 or reference_auto_threads and kind == "lc0",
        "Positive thread count or verified reference auto required",
    )
    model, source = _path(row["model"]), _path(row["source_net"])
    args = tuple(_strings(row["args"]))
    if kind == "mouse":
        require(
            not args and _path(options.get("WeightsFile")) == model,
            "Mouse model must be explicitly selected without extra argv",
        )
        backend, precision = options.get("Backend"), options.get("Precision")
        require(
            (backend == "cpu" and precision == "fp32")
            or (backend in ("cuda", "tensorrt") and precision == "fp16"),
            "Explicit supported Mouse backend/precision required",
        )
        require(
            options.get("MoveOverheadMs") == "0"
            and "MinibatchSize" in options
            and "RamLimitMb" in options,
            "Explicit Mouse batching/RAM/zero-overhead required",
        )
        if backend != "cpu":
            directories = options.get("GpuDllDirectories", "").split(";")
            require(
                directories
                and all(Path(v).is_absolute() and Path(v).is_dir() for v in directories),
                "Explicit GPU SDK directories required",
            )
            require(
                options.get("Device") == "0"
                and Path(options.get("CacheDirectory", "")).is_absolute(),
                "Explicit GPU device/cache directory required",
            )
    else:
        require(
            args == ("--config=", "--show-hidden") and _path(options.get("WeightsFile")) == model,
            "Lc0 requires exact --config= --show-hidden args and explicit weights option",
        )
        require(
            options.get("Backend") == "cuda-fp16",
            "Explicit Lc0 backend required; auto is forbidden",
        )
    return Engine(
        name, kind, _path(row["executable"]), model, source, args, tuple(sorted(options.items()))
    )


def load_plan(path: Path) -> Plan:
    path = path.resolve()
    config_hash = _sha(path)
    row = _json(path)
    required = {
        "schema",
        "profile",
        "seed",
        "fastchess",
        "book",
        "engines",
        "runtime_directories",
        "source_roots",
        "sha256",
        "idle_seconds",
        "total_seconds",
        "max_output_bytes",
    }
    require(
        required <= set(row) <= required | {"max_games", "nodes", "concurrency"},
        "Unknown/missing match plan fields",
    )
    require(
        row["schema"] == "MouseMatchPlan1" and type(row["seed"]) is int and row["seed"] == 20261017,
        "Wrong plan schema/seed",
    )
    selected = profile(
        str(row["profile"]),
        max_games=cast(int | None, row.get("max_games")),
        nodes=cast(int | None, row.get("nodes")),
    )
    if "concurrency" in row:
        # V1 permits four concurrent games only for the zero-time-loss stress
        # profiles. The TM gainer stays serial; counts and limits are unchanged.
        concurrency = row["concurrency"]
        parallel = selected.name in {
            "parity-sprt",
            "tm-safety-fast",
            "tm-safety-stc",
            "tm-safety-ltc",
            "uci-stress-fast",
            "v1-fast-track-fast",
            "v1-fast-track-stc",
            "v1-fast-track-ltc",
        }
        require(
            type(concurrency) is int and concurrency in ((1, 4) if parallel else (1,)),
            "Explicit concurrency must be1 (or4 for an explicitly allowed stress/parity profile)",
        )
        selected = replace(selected, concurrency=cast(int, concurrency))
    values = row["engines"]
    require(isinstance(values, list) and len(values) == 2, "Exactly two engines required")
    engines = tuple(
        _engine(value, reference_auto_threads=selected.name == "parity-sprt")
        for value in cast(list[object], values)
    )
    require(engines[0].name != engines[1].name, "Distinct engine labels required")
    require(engines[0].kind == "mouse", "Candidate must be Mouse")
    if selected.name in (
        "selfplay-fast",
        "package-smoke",
        "v1-fast-track-fast",
        "v1-fast-track-stc",
        "v1-fast-track-ltc",
    ):
        require(
            engines[0].executable == engines[1].executable
            and engines[0].options == engines[1].options
            and engines[0].model == engines[1].model,
            "Selfplay requires identical Mouse configurations",
        )
    if selected.name == "package-smoke":
        require(
            all(
                engine.kind == "mouse"
                and dict(engine.options).get("Backend") in ("cuda", "tensorrt")
                and dict(engine.options).get("Precision") == "fp16"
                and dict(engine.options).get("NNCacheSize") == "0"
                for engine in engines
            ),
            "Package smoke requires identical Mouse GPU fp16/cache0 configurations",
        )
    if selected.name == "parity-sprt":
        reference = dict(engines[1].options)
        require(
            engines[1].kind == "lc0"
            and all(
                _numeric_option(dict(engine.options).get("SmartPruningFactor"), "0")
                for engine in engines
            ),
            "Fixed-node parity requires explicit smart pruning off on both engines",
        )
        common = {
            "CPuct": "1.745",
            "CPuctBase": "38739",
            "CPuctFactor": "3.894",
            "FpuValue": "0.330",
            "PolicyTemperature": "1.359",
        }
        for engine in engines:
            options = dict(engine.options)
            require(
                all(_numeric_option(options.get(k), v) for k, v in common.items()),
                "Fixed-node search values must be explicitly matched",
            )
            require(
                options.get("Threads") == ("4" if engine.kind == "mouse" else "0")
                and options.get("MinibatchSize") == ("64" if engine.kind == "mouse" else "0")
                and options.get("NNCacheSize") == "0",
                "Parity requires Mouse threads4/batch64 and Lc0 threads0/batch0 auto; cache0",
            )
        require(
            all(
                dict(engine.options).get("FpuStrategyAtRoot") == "same"
                and _numeric_option(dict(engine.options).get("FpuValueAtRoot"), "1")
                for engine in engines
            ),
            "Fixed-node root FPU requires explicit same/1 on both engines",
        )
        require(
            dict(engines[0].options).get("Deterministic") == "false",
            "Mouse fixed-node parity uses its normal nondeterministic batching",
        )
        require(
            all(
                reference.get(k) == v
                for k, v in {
                    "FpuStrategy": "reduction",
                    "HistoryFill": "fen_only",
                }.items()
            ),
            "Lc0 root FPU/history strategy differs",
        )
        expected = {
            "MovesLeftMaxEffect": "0.0345",
            "MovesLeftThreshold": "0.8",
            "MovesLeftSlope": "0.0027",
            "MovesLeftConstantFactor": "0",
            "MovesLeftScaledFactor": "1.6521",
            "MovesLeftQuadraticFactor": "-0.6521",
        }
        require(
            all(_numeric_option(reference.get(k), v) for k, v in expected.items()),
            "Lc0 root FPU or enabled MLH parameters differ",
        )
    else:
        require(
            all(
                dict(e.options).get("Threads") == "4"
                and dict(e.options).get("MinibatchSize") == "64"
                for e in engines
            ),
            "Timed/selfplay profiles require explicit normal threads4/minibatch64",
        )
    if selected.name == "cache-sprt":
        # Capacities use different UCI units across the frozen implementations;
        # the native capacity measurement belongs in the gate report. Never
        # infer equal capacity from equal option strings.
        require(all(e.kind == "mouse" for e in engines), "Cache gate requires two Mouse builds")
        common = {
            "CPuct": "1.745",
            "CPuctBase": "38739",
            "CPuctFactor": "3.894",
            "FpuValue": "0.330",
            "FpuValueAtRoot": "1",
            "PolicyTemperature": "1.359",
        }
        for engine in engines:
            options = dict(engine.options)
            require(
                all(_numeric_option(options.get(k), v) for k, v in common.items())
                and options.get("FpuStrategyAtRoot") == "same"
                and options.get("Deterministic") == "false"
                and int(options["NNCacheSize"]) > 0
                and _numeric_option(options.get("SmartPruningFactor"), "0")
                and options.get("SmartPruningMinimumBatches", "0") == "0",
                "Cache gate requires enabled caches, matched search defaults and no smart pruning",
            )
        excluded = {"NNCacheSize", "SmartPruningFactor", "SmartPruningMinimumBatches"}
        require(
            engines[0].model == engines[1].model
            and {k: v for k, v in engines[0].options if k not in excluded}
            == {k: v for k, v in engines[1].options if k not in excluded},
            "Cache gate requires identical model and other runtime/search options",
        )
    book = _path(row["book"])
    require(book.name == selected.book and book.name in BOOKS, "Wrong opening book for profile")
    hashes: dict[Path, str] = {}
    for name, value in _object(row["sha256"]).items():
        target = _path(name)
        require(
            target not in hashes and isinstance(value, str) and SHA.fullmatch(value),
            "Malformed/duplicate artifact SHA256",
        )
        hashes[target] = cast(str, value)
    require(hashes.get(book) == BOOKS[book.name][0], "Book differs from pinned official content")
    require(
        all(p.is_file() and _sha(p) == h for p, h in hashes.items()),
        "Declared artifact hash mismatch",
    )
    require(
        hashes.get(_path(row["fastchess"])) == FASTCHESS_SHA256,
        "Require the publisher-verified pinned fastchess runner executable",
    )
    if selected.name == "parity-sprt":
        require(
            hashes.get(engines[1].executable) == LC0_RELEASE_SHA256,
            "Parity normal auto defaults require the pinned official Lc0 reference",
        )
    hashes[path] = config_hash
    for own in (
        Path(__file__).resolve(),
        Path(uci_gate.__file__).resolve(),
        Path(sys.executable).resolve(),
    ):
        hashes.setdefault(own, _sha(own))
    directories = tuple(_path(v) for v in _strings(row["runtime_directories"]))
    source_roots = tuple(_path(v) for v in _strings(row["source_roots"]))
    require(
        directories and source_roots and all(p.is_dir() for p in directories + source_roots),
        "Explicit existing source/runtime roots required",
    )
    require(
        _path(row["fastchess"]).parent in directories,
        "App-local fastchess runtime directory required",
    )
    require(
        all(e.executable.parent in directories for e in engines),
        "Both app-local runtime directories required",
    )
    for engine in engines:
        for directory in dict(engine.options).get("GpuDllDirectories", "").split(";"):
            if directory:
                require(
                    Path(directory).resolve() in directories, "GPU SDK runtime inventory missing"
                )
    idle = _positive(row["idle_seconds"], "idle watchdog", 86400)
    require(idle >= 300, "Watchdog must permit explicit cold-readiness deadline")
    total = _positive(row["total_seconds"], "total watchdog", 365 * 86400)
    require(total >= idle, "Total watchdog shorter than readiness")
    plan = Plan(
        path,
        selected,
        _path(row["fastchess"]),
        book,
        cast(tuple[Engine, Engine], engines),
        directories,
        source_roots,
        tuple(sorted(hashes.items(), key=lambda v: str(v[0]))),
        idle,
        total,
        _integer(row["max_output_bytes"], 1024, 2**40, "output byte bound"),
    )
    plan.verify()  # All identities frozen before loading ONNX metadata.
    import onnx

    for engine in engines:
        require(
            hashes[engine.source_net] == T1_SHA256, "Phase1 gate requires the pinned T1 source net"
        )
        require(
            hashes[engine.source_net] == hashes[engines[0].source_net],
            "Both engines must use the same source net",
        )
        if engine.kind == "mouse":
            require(engine.model.stat().st_size <= 512 * 1024 * 1024, "ONNX artifact exceeds bound")
            model = onnx.load_model(engine.model, load_external_data=False)
            metadata = _pairs([(entry.key, entry.value) for entry in model.metadata_props])
            require(
                metadata.get("mouse.source_sha256") == hashes[engine.source_net],
                "ONNX source-net hash differs",
            )
            require(
                metadata.get("mouse.precision", "fp32") == dict(engine.options)["Precision"],
                "ONNX precision differs",
            )
            require(
                not any(t.data_location or t.external_data for t in model.graph.initializer),
                "External ONNX tensor data is forbidden",
            )
        else:
            require(engine.model == engine.source_net, "Lc0 must load the exact source network")
    plan.verify()
    return plan


def build_command(plan: Plan, output: Path) -> list[str]:
    p = plan.selected
    fmt = BOOKS[plan.book.name][1]
    command = [
        str(plan.fastchess),
        "-concurrency",
        str(p.concurrency),
        "-repeat",
        "-variant",
        "standard",
        "-srand",
        "20261017",
        "-strict",
        "-report",
        "penta=true",
        "-output",
        "format=fastchess",
        "-openings",
        f"file={plan.book}",
        f"format={fmt}",
        "order=random",
        "start=1",
    ]
    if fmt == "pgn":
        command += ["plies=16"]
    command += [
        "-rounds",
        str(p.games // 2),
        "-games",
        "2",
        "-startup-ms",
        "300000",
        "-ping-ms",
        "300000",
        "-ucinewgame-ms",
        "300000",
        "-ratinginterval",
        "1",
        "-autosaveinterval",
        "1",
    ]
    for engine in plan.engines:
        executable = str(engine.executable)
        if sys.platform == "win32":
            executable = '"' + executable + '"'
        args = (
            subprocess.list2cmdline(engine.args)
            if sys.platform == "win32"
            else shlex.join(engine.args)
        )
        command += ["-engine", f"name={engine.name}", f"cmd={executable}"]
        if engine.args:
            command.append(f"args={args}")
        command += [f"option.{key}={value}" for key, value in engine.options]
    command += ["-each", "proto=uci", "timemargin=0", "restart=off"]
    command += [f"nodes={p.nodes}"] if p.nodes is not None else [f"tc={p.tc}"]
    if p.adjudication:
        # The parity contract (2026-10-17) explicitly adopts the pinned
        # native units. match.hpp:60/103 multiply these counts by two. Opening
        # insertion does not update MaxMovesTracker, so 16 + 2*192 = 400 plies.
        command += [
            "-draw",
            "movenumber=34",
            "movecount=8",
            "score=10",
            "-resign",
            "movecount=3",
            "score=1000",
            "twosided=true",
            "-maxmoves",
            "192",
        ]
    if p.sprt:
        command += [
            "-sprt",
            f"elo0={p.sprt[0]}",
            f"elo1={p.sprt[1]}",
            "alpha=0.05",
            "beta=0.05",
            "model=normalized",
        ]
    command += [
        "-pgnout",
        f"file={output / 'games.pgn'}",
        "append=false",
        "min=false",
        "notation=san",
        "nodes=true",
        "timeleft=true",
        "latency=true",
        "-log",
        f"file={output / 'protocol.log'}",
        "level=trace",
        "engine=true",
        "append=false",
        "compress=false",
        "realtime=true",
        "-config",
        f"outname={output / 'fastchess.json'}",
    ]
    return command


@dataclass
class Progress:
    """Only the stdout pump owns this bounded game-ID ledger.

    Fixed-time profiles preserve their strict sequential schedule. Concurrent
    profiles retain every start/finish separately instead of assuming output order.
    """

    planned_games: int
    names: tuple[str, str]
    concurrency: int = 1
    allow_adjudication: bool = False
    started: dict[int, tuple[int, str, str]] = field(default_factory=dict)
    finished: dict[int, tuple[str, str, str, str]] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)
    closed: bool = False
    peak_active: int = 0

    def __post_init__(self) -> None:
        _integer(self.planned_games, 2, 1000000, "planned game bound")
        _integer(self.concurrency, 1, 6, "concurrent game bound")

    def __call__(self, line: str) -> None:
        require(
            not re.search(
                r"(?i)(?:\b(?:warning|error)[:;]|\b(?:interrupted|aborted|crashed)\b)", line
            ),
            "Native runner reported a live failure",
        )
        if line.startswith("Started game "):
            match = re.fullmatch(r"Started game ([0-9]+) of ([0-9]+) \((\S+) vs (\S+)\)", line)
            require(match is not None, "Malformed live game-start record")
            token, cap, white, black = cast(re.Match[str], match).groups()
            index = int(token)
            expected = self.names if index % 2 else self.names[::-1]
            require(
                not self.closed
                and 1 <= index <= self.planned_games
                and int(cap) == self.planned_games
                and index not in self.started
                and len(self.active) < self.concurrency
                and (white, black) == expected,
                "Unexpected live count, concurrency or colors",
            )
            if self.concurrency == 1:
                require(index == len(self.started) + 1, "Nonsequential timed game start")
            self.started[index] = (int(cap), white, black)
            self.active.add(index)
            self.peak_active = max(self.peak_active, len(self.active))
        elif line.startswith("Finished game "):
            match = re.fullmatch(
                r"Finished game ([0-9]+) \((\S+) vs (\S+)\): (1-0|0-1|1/2-1/2) \{([^{}]+)\}", line
            )
            require(match is not None, "Malformed live finished-game record")
            token, white, black, result, reason = cast(re.Match[str], match).groups()
            index = int(token)
            require(
                not self.closed
                and index in self.active
                and self.started[index][1:] == (white, black),
                "Unexpected live finish record",
            )
            require(
                reason in NORMAL_REASONS
                or self.allow_adjudication
                and reason in ADJUDICATED_RESULTS,
                "Live time loss, illegal move, disconnect or unplanned adjudication",
            )
            self.finished[index] = (white, black, result, reason)
            self.active.remove(index)
        elif line == "Finished match":
            require(not self.closed, "Duplicate runner completion")
            self.closed = True


def capture(
    command: Sequence[str],
    output: Path,
    *,
    idle_seconds: float,
    total_seconds: float,
    max_bytes: int,
    on_stdout: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Stream bounded raw bytes to disk; all owned descendants die before return."""
    _positive(idle_seconds, "idle watchdog", 365 * 86400)
    _positive(total_seconds, "total watchdog", 365 * 86400)
    require(max_bytes > 0, "Positive output bound required")
    started = time.monotonic()
    stop = threading.Event()
    lock = threading.Lock()
    last = started
    failures: list[str] = []
    threads: list[threading.Thread] = []
    process: subprocess.Popen[bytes] | None = None
    job: uci_gate._WindowsJob | None = None
    terminated = False
    exit_code: int | None = None
    files = ["stdout.txt", "stderr.txt"]
    streams: list[BinaryIO] = []

    def pump(source: BinaryIO, target: BinaryIO, observe: Callable[[str], None] | None) -> None:
        nonlocal last
        size = 0
        pending = bytearray()
        try:
            while block := source.read(65536):
                target.write(block[: max(0, max_bytes - size)])
                target.flush()
                size += len(block)
                require(size <= max_bytes, "Native output stream exceeded declared byte bound")
                if observe is not None:
                    pending.extend(block)
                    while b"\n" in pending:
                        position = pending.index(10)
                        raw = bytes(pending[:position])
                        del pending[: position + 1]
                        require(len(raw) <= 1024 * 1024, "Match stdout line exceeds1MiB bound")
                        observe(raw.removesuffix(b"\r").decode("utf-8", errors="strict"))
                    require(len(pending) <= 1024 * 1024, "Match stdout line exceeds1MiB bound")
                with lock:
                    last = time.monotonic()
            require(observe is None or not pending, "Unterminated native stdout line")
        except (OSError, ValueError) as error:
            with lock:
                failures.append(str(error))
            stop.set()

    try:
        streams = [(output / name).open("xb") for name in files]
        job = uci_gate._WindowsJob() if sys.platform == "win32" else None
        process = subprocess.Popen(
            list(command),
            cwd=output,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PATH": ""},
            bufsize=0,
            start_new_session=sys.platform != "win32",
            creationflags=uci_gate.creation_flags(),
        )
        if job:
            job.assign(process.pid)
        require(process.stdout is not None and process.stderr is not None, "Missing child streams")
        for index, (source, target) in enumerate(
            zip((process.stdout, process.stderr), streams, strict=True)
        ):
            thread = threading.Thread(
                target=pump,
                args=(cast(BinaryIO, source), target, on_stdout if index == 0 else None),
            )
            thread.start()
            threads.append(thread)
        while process.poll() is None:
            with lock:
                idle = time.monotonic() - last
            now = time.monotonic()
            require(not stop.is_set(), "; ".join(failures))
            require(now - started < total_seconds, "Total match watchdog expired")
            require(idle < idle_seconds, "Idle match watchdog expired")
            require(
                sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= max_bytes,
                "Match disk evidence exceeded byte bound",
            )
            time.sleep(0.02)
        exit_code = process.returncode
    except KeyboardInterrupt:
        failures.append("Interrupted; partial evidence cannot pass")
    except (OSError, ValueError) as error:
        failures.append(str(error))
    finally:
        if process is not None:
            terminated = process.poll() is None
            if sys.platform != "win32":
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            elif job:
                job.close()
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=5)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                failures.append("Owned runner could not be reaped")
            for thread in threads:
                thread.join(timeout=5)
            if any(t.is_alive() for t in threads):
                failures.append("Owned output pump did not join")
            for source in (process.stdout, process.stderr):
                if source:
                    source.close()
        if job:
            job.close()
        for stream in streams:
            stream.close()
    return {
        "command": list(command),
        "exit_code": exit_code,
        "error": "; ".join(failures) or None,
        "terminated_by_runner": terminated,
        "pumps_joined": all(not t.is_alive() for t in threads),
        "wall_seconds": time.monotonic() - started,
        "pid": process.pid if process is not None else None,
    }


@dataclass(frozen=True)
class Game:
    tags: dict[str, str]
    result: str
    moves: tuple[str, ...]
    book_moves: tuple[str, ...]
    movetext: str
    nodes_by_ply: tuple[int | None, ...]


def _pgn_records(path: Path) -> Iterator[str]:
    record: list[str] = []
    size = 0
    body = False
    with path.open(encoding="utf-8-sig", errors="strict") as stream:
        for line in stream:
            require(len(line) <= 65536, "PGN line exceeds bound")
            if line.startswith("[Event ") and body:
                yield "".join(record)
                record, size, body = [], 0, False
            size += len(line)
            require(size <= 16 * 1024 * 1024, "PGN game exceeds bound")
            record.append(line)
            if line.strip() and not line.startswith("["):
                body = True
        if "".join(record).strip():
            yield "".join(record)


def _game(
    record: str, names: tuple[str, str], *, book: bool = False, allow_adjudication: bool = False
) -> Game:
    lines = record.splitlines()
    tags: dict[str, str] = {}
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].startswith("[")):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        match = re.fullmatch(r'\[([A-Za-z0-9_]+) "([^"\x00-\x1f]*)"\]', line)
        require(match is not None, "Malformed PGN header")
        key, value = cast(re.Match[str], match).groups()
        require(key not in tags, "Duplicate PGN header")
        tags[key] = value
    required = {"Event", "Round", "White", "Black", "Result"}
    require(required <= tags.keys(), "Missing required PGN headers")
    require(tags["Result"] in RESULTS, "Unfinished PGN result")
    if not book:
        require(
            tags.get("Termination")
            in (("normal", "adjudication") if allow_adjudication else ("normal",)),
            "Abnormal or unplanned adjudicated PGN termination",
        )
        require({tags["White"], tags["Black"]} == set(names), "Unexpected engine in PGN")
        require(tags.get("Round", "").isdigit() and int(tags["Round"]) > 0, "Invalid PGN round")
    body = " ".join(lines[index:])
    tokens = re.findall(r"\{[^{}]*\}|[^\s{}]+", body)
    require(" ".join(tokens).replace(" ", "") == body.replace(" ", ""), "Unparsed PGN text")
    require(tokens and tokens[-1] == tags["Result"], "PGN header/result mismatch or truncated game")
    moves: list[str] = []
    opening: list[str] = []
    book_ended = False
    annotated = 0
    nodes_by_ply: list[int | None] = []
    for token in tokens[:-1]:
        if token.startswith("{"):
            if not book:
                require(annotated == len(moves) - 1, "Duplicate/missing per-move annotation")
                annotated = len(moves)
            if token == "{book}":
                require(
                    moves and not book_ended and len(opening) + 1 == len(moves),
                    "Malformed book move annotations",
                )
                opening.append(moves[-1])
                if not book:
                    nodes_by_ply.append(None)
            else:
                book_ended = True
                require(
                    not re.search(r"\b(?:nan|inf|infinity)\b", token, re.I),
                    "Nonfinite PGN annotation",
                )
                if not book:
                    fields = re.findall(r"\bn=([^,\s}]+)", token)
                    require(
                        len(fields) == 1
                        and fields[0].isascii()
                        and fields[0].isdigit()
                        and len(fields[0]) <= 20,
                        "Missing, ambiguous or invalid per-move nodes",
                    )
                    actual_nodes = int(fields[0])
                    require(actual_nodes <= 2**64 - 1, "Per-move nodes exceed uint64")
                    nodes_by_ply.append(actual_nodes)
                    require(
                        re.search(r"\btl=-?[0-9]+(?:\.[0-9]+)?s?\b", token),
                        "Missing per-move remaining clock",
                    )
                    require(not re.search(r"\btl=-", token), "Negative remaining clock")
        elif re.fullmatch(r"[0-9]+\.(?:\.\.)?", token):
            continue
        else:
            if not book:
                require(annotated == len(moves), "Missing per-move annotation")
            require(
                re.fullmatch(
                    r"(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?", token
                ),
                "Malformed SAN move or extra result",
            )
            moves.append(token)
    if not book:
        require(annotated == len(moves), "Missing final per-move annotation")
    require(moves, "Empty PGN game")
    if not book:
        require(tags.get("PlyCount") == str(len(moves)), "PGN ply count mismatch")
    if not book and allow_adjudication:
        require(len(moves) <= 400, "Parity game exceeded 400 total plies")
        if tags["Termination"] == "adjudication":
            final_comment = tokens[-2]
            reasons = [
                reason
                for reason in ADJUDICATED_RESULTS
                if final_comment.endswith(", " + reason + "}")
            ]
            require(
                len(reasons) == 1 and ADJUDICATED_RESULTS[reasons[0]] == tags["Result"],
                "Adjudication reason/result differs from pinned native grammar",
            )
            reason = reasons[0]
            # These bounds are necessary native conditions, not an independent
            # reproduction of score tracking: match.hpp:60/103, match.cpp:558-570.
            # Source/actual argv/saved config bind the exact score-based policy.
            require(len(opening) == 16, "Adjudicated parity requires its full 16-ply opening")
            require(
                len(moves) >= (68 if reason == "Draw by adjudication" else 22),
                "Adjudication predates its native move/count threshold",
            )
            if reason != "Draw by adjudication":
                require(
                    (len(moves) % 2 == 0) == (reason == "White wins by adjudication"),
                    "Resignation winner differs from last moving losing side",
                )
    return Game(tags, tags["Result"], tuple(moves), tuple(opening), body, tuple(nodes_by_ply))


def read_games(
    path: Path, names: tuple[str, str], *, allow_adjudication: bool = False
) -> list[Game]:
    require(path.stat().st_size <= 512 * 1024 * 1024, "PGN evidence exceeds512MiB parse bound")
    games = [
        _game(record, names, allow_adjudication=allow_adjudication) for record in _pgn_records(path)
    ]
    require(0 < len(games) <= 1000000, "Missing/oversized game inventory")
    return games


def _game_ledger(
    plan: Plan,
    games: Sequence[Game],
    events: Progress,
    *,
    natural_sprt_stop: bool,
    verified_skipped_assignments: Sequence[int] = (),
) -> tuple[list[Game], dict[str, object]]:
    """Join every PGN to native game IDs; never hide an unscored native tail.

    The caller must first verify a real LLR-boundary decision for natural_sprt_stop.
    Pinned ScoreBoard only scores completed color pairs; the source stops active
    games after a boundary. A cap or fixed-count run still requires every pair.
    """
    names = (plan.engines[0].name, plan.engines[1].name)
    started, finished = set(events.started), set(events.finished)
    skipped = set(verified_skipped_assignments)
    require(
        len(skipped) == len(verified_skipped_assignments) and not skipped & started,
        "Duplicate or started skipped assignment",
    )
    require(
        not skipped or (natural_sprt_stop and plan.selected.concurrency > 1),
        "Skipped assignments require an authenticated concurrent natural stop",
    )
    observed = started | skipped
    require(events.closed and started, "Missing final runner closure or game starts")
    require(
        observed == set(range(1, max(observed) + 1))
        and max(observed) <= plan.selected.games
        and finished <= started
        and events.active == started - finished
        and len(events.active) <= plan.selected.concurrency,
        "Missing game-ID set or excessive pending concurrency tails",
    )
    by_id: dict[int, Game] = {}
    raw_ids: list[int] = []
    for game in games:
        round_text = game.tags["Round"]
        require(round_text.isascii() and round_text.isdigit(), "Invalid PGN pair round")
        pair = int(round_text)
        require(
            1 <= pair <= plan.selected.games // 2 and round_text == str(pair),
            "PGN pair round outside declared inventory",
        )
        colors = (game.tags["White"], game.tags["Black"])
        require(colors in (names, names[::-1]), "Unexpected PGN colors")
        index = 2 * pair - (1 if colors == names else 0)
        require(index not in by_id, "Duplicate PGN game ID/color pair")
        by_id[index] = game
        raw_ids.append(index)
    require(set(by_id) == finished, "Completed PGN lost or extra PGN without native finish")
    for index, game in by_id.items():
        white, black, result, reason = events.finished[index]
        expected = (
            "1-0"
            if reason == "White mates"
            else "0-1"
            if reason == "Black mates"
            else ADJUDICATED_RESULTS.get(reason, "1/2-1/2")
        )
        require(
            (white, black, result) == (game.tags["White"], game.tags["Black"], game.result)
            and result == expected
            and reason in game.movetext
            and game.tags["Termination"]
            == ("adjudication" if reason in ADJUDICATED_RESULTS else "normal")
            and (plan.selected.adjudication or reason not in ADJUDICATED_RESULTS),
            "PGN/finished-game identity, result or termination reason mismatch",
        )
    complete_rounds = sorted(
        pair
        for pair in {int(game.tags["Round"]) for game in games}
        if 2 * pair - 1 in by_id and 2 * pair in by_id
    )
    paired: list[Game] = []
    for pair in complete_rounds:
        left, right = by_id[2 * pair - 1], by_id[2 * pair]
        require(
            left.tags.get("FEN", "") == right.tags.get("FEN", "")
            and left.book_moves == right.book_moves,
            "Paired openings differ",
        )
        paired.extend((left, right))
    orphan = sorted(
        index for index in by_id if (index + 1 if index % 2 else index - 1) not in by_id
    )
    unstarted_mates = sorted(
        partner
        for index in orphan
        if (partner := index + 1 if index % 2 else index - 1) not in started
    )
    if natural_sprt_stop and plan.selected.concurrency > 1:
        require(plan.selected.sprt is not None, "Native tails require SPRT")
        require(
            all(
                (partner in skipped or partner == max(observed) + 1)
                and partner <= plan.selected.games
                for partner in unstarted_mates
            ),
            "Unpaired result has no native pending or next unscheduled mate",
        )
        incomplete_rounds = {
            (index + 1) // 2 for index in orphan + sorted(events.active) + sorted(skipped)
        }
        require(
            len(incomplete_rounds) <= plan.selected.concurrency,
            "Excessive incomplete color-pair tails",
        )
    else:
        require(
            not orphan and not events.active, "Incomplete color pair without natural SPRT decision"
        )
        if not natural_sprt_stop:
            require(
                started == set(range(1, plan.selected.games + 1)),
                "Fixed-count/cap game inventory incomplete",
            )
    require(paired, "No complete statistical color pairs")
    return paired, {
        "started_game_ids": sorted(started),
        "skipped_before_start_game_ids": sorted(skipped),
        "observed_assignment_prefix_ids": sorted(observed),
        "finished_game_ids": sorted(finished),
        "raw_pgn_game_ids": raw_ids,
        "complete_pair_round_ids": complete_rounds,
        "unpaired_completed_game_ids": orphan,
        "cancelled_inflight_game_ids": sorted(events.active),
        "unstarted_mate_game_ids": unstarted_mates,
        "raw_completed_games": len(games),
        "statistical_complete_pair_games": len(paired),
        "tail_basis": (
            "Only an independently verified natural SPRT boundary may retain unscored bounded tails"
        ),
    }


def pair_statistics(games: Sequence[Game], *, allow_round_gaps: bool = False) -> dict[str, object]:
    require(games and len(games) % 2 == 0, "Incomplete color pair")
    first = games[0].tags["White"]
    second = games[0].tags["Black"]
    counts = [0] * 5
    categories = {key: 0 for key in ("LL", "LD", "WL", "DD", "WD", "WW")}
    wdl = {"wins": 0, "draws": 0, "losses": 0}
    seen_rounds: set[str] = set()
    for index in range(0, len(games), 2):
        left, right = games[index : index + 2]
        require(
            (left.tags["White"], left.tags["Black"], right.tags["White"], right.tags["Black"])
            == (first, second, second, first),
            "Paired colors not reversed",
        )
        expected_round = left.tags["Round"] if allow_round_gaps else str(index // 2 + 1)
        require(
            left.tags["Round"] == right.tags["Round"] == expected_round
            and left.tags["Round"] not in seen_rounds,
            "Duplicate/missing pair round",
        )
        seen_rounds.add(left.tags["Round"])
        require(
            left.tags.get("FEN", "") == right.tags.get("FEN", "")
            and left.book_moves == right.book_moves,
            "Paired openings differ",
        )
        scores = [RESULTS[left.result], 1 - RESULTS[right.result]]
        for score in scores:
            wdl["wins" if score == 1 else "draws" if score == 0.5 else "losses"] += 1
        total = sum(scores)
        counts[int(total * 2)] += 1
        category = (
            "DD"
            if scores == [0.5, 0.5]
            else {0.0: "LL", 0.5: "LD", 1.0: "WL", 1.5: "WD", 2.0: "WW"}[total]
        )
        categories[category] += 1
    pairs = len(games) // 2
    mean = sum(i * 0.25 * n for i, n in enumerate(counts)) / pairs
    variance = sum(n * (i * 0.25 - mean) ** 2 for i, n in enumerate(counts)) / pairs
    result: dict[str, object] = dict(
        games=len(games),
        pairs=pairs,
        wdl=wdl,
        pentanomial=counts,
        pair_categories=categories,
        score=mean,
        score_variance_per_pair=variance,
        elo=None,
        elo_ci95=None,
        nelo=None,
        nelo_ci95=None,
        statistics_status="degenerate_pair_variance",
    )
    # Original formula implementation following pinned elo_pentanomial.cpp/elo.hpp.
    # The five probabilities describe per-game score averaged across each pair.
    if 0 < mean < 1:
        result["elo"] = 400 * math.log10(mean / (1 - mean))
    if variance > 0:
        se = math.sqrt(variance / pairs)
        lo, hi = mean - 1.959963984540054 * se, mean + 1.959963984540054 * se
        result["elo_ci95"] = [
            400 * math.log10(lo / (1 - lo)) if lo > 0 else None,
            400 * math.log10(hi / (1 - hi)) if hi < 1 else None,
        ]
        scale = (800 / math.log(10)) / math.sqrt(2 * variance)
        result["nelo"] = (mean - 0.5) * scale
        result["nelo_ci95"] = [(lo - 0.5) * scale, (hi - 0.5) * scale]
        result["statistics_status"] = "normal_approximation_paired_empirical_variance"
    return result


def _publish(path: Path, value: object) -> None:
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8-sig", errors="strict") as stream:
        for line in stream:
            require(len(line) <= 1024 * 1024, f"Oversized evidence line: {path.name}")
            yield line.rstrip("\r\n")


def _book_membership(plan: Plan, games: Sequence[Game]) -> dict[str, object]:
    format_name = BOOKS[plan.book.name][1]
    if format_name == "pgn":
        start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        require(
            all(g.tags.get("FEN", start_fen) == start_fen for g in games),
            "Balanced PGN initial position differs",
        )
        needed = {game.book_moves for game in games}
        require(
            all(len(opening) == 16 for opening in needed),
            "Each balanced opening must retain all16 prescribed book plies",
        )
        found: set[tuple[str, ...]] = set()
        records = 0
        for records, record in enumerate(_pgn_records(plan.book), 1):
            require(records <= BOOKS[plan.book.name][2], "Book record count exceeds pin")
            original = _game(record, ("", ""), book=True)
            require(
                len(original.moves) >= 16, "Invalid/short balanced opening; no skipping permitted"
            )
            opening = original.moves[:16]
            if opening in needed:
                found.add(opening)
        require(found == needed, "PGN game opening absent from frozen book")
    else:
        needed_fens = {game.tags.get("FEN", "") for game in games}
        require(
            all(not game.book_moves and game.tags.get("SetUp") == "1" for game in games),
            "EPD opening must retain complete starting FEN",
        )
        found_fens: set[str] = set()
        records = 0
        for records, line in enumerate(_lines(plan.book), 1):
            require(records <= BOOKS[plan.book.name][2], "Book record count exceeds pin")
            fields = line.split()
            require(
                len(fields) >= 4 and fields[1] in ("w", "b"),
                "Malformed EPD record; no skipping permitted",
            )
            fen = " ".join(fields[:4]) + " 0 1"
            if len(fields) >= 6 and fields[4].isdigit() and fields[5].isdigit():
                fen = " ".join(fields[:6])
            if fen in needed_fens:
                found_fens.add(fen)
        require(found_fens == needed_fens, "Game FEN absent from frozen EPD book")
    require(
        records == BOOKS[plan.book.name][2],
        "Opening book record count differs from frozen publisher inventory",
    )
    return {
        "records": records,
        "pairs": len({game.tags["Round"] for game in games}),
        "observed_pair_round_ids": sorted({int(game.tags["Round"]) for game in games}),
        "book_cycles_required": (max(int(game.tags["Round"]) for game in games) - 1) // records,
        "selection_evidence": (
            "Pinned book bytes, explicit seeded runner configuration, paired actual PGN "
            "openings and membership; no independent emulation of implementation-specific "
            "std::shuffle"
        ),
    }


def _saved_configuration(
    plan: Plan, raw: Path, statistics: Mapping[str, object]
) -> dict[str, object]:
    saved = _json(raw / "fastchess.json", 16 * 1024 * 1024)
    # The pinned Tournament JSON serializer omits the runtime strict member.
    # A serialized contradiction still fails; actual invocation is checked separately.
    require("strict" not in saved or saved["strict"] is True, "Native config differs: strict")
    p = plan.selected
    expected: dict[str, object] = {
        "games": 2,
        "rounds": p.games // 2,
        "concurrency": p.concurrency,
        "seed": 20261017,
        "recover": False,
        "noswap": False,
        "reverse": False,
        "report_penta": True,
    }
    for key, value in expected.items():
        require(
            type(saved.get(key)) is type(value) and saved[key] == value,
            f"Native config differs: {key}",
        )
    if p.adjudication:
        policies: dict[str, dict[str, object]] = {
            "draw": dict(enabled=True, move_number=34, move_count=8, score=10),
            "resign": dict(enabled=True, move_count=3, score=1000, twosided=True),
            "maxmoves": dict(enabled=True, move_count=192),
        }
        for key, expected_policy in policies.items():
            actual = _object(saved.get(key))
            require(
                set(actual) == set(expected_policy)
                and all(
                    type(actual[field]) is type(value) and actual[field] == value
                    for field, value in expected_policy.items()
                ),
                "Native parity adjudication differs from the exact parity contract",
            )
    else:
        for key in ("draw", "resign", "maxmoves"):
            require(
                _object(saved.get(key)).get("enabled") is False, "Unexpected early adjudication"
            )
    require(
        _object(saved.get("tb_adjudication")).get("enabled") is False,
        "Unexpected tablebase adjudication",
    )
    opening = _object(saved.get("opening"))
    require(
        _path(opening.get("file")) == plan.book
        and opening.get("format") == (1 if plan.book.suffix == ".pgn" else 0)
        and opening.get("order") == 0
        and opening.get("start") == 1
        and opening.get("plies") == (16 if plan.book.suffix == ".pgn" else -1),
        "Native opening config differs",
    )
    sprt = _object(saved.get("sprt"))
    require(sprt.get("enabled") is (p.sprt is not None), "Native SPRT configuration differs")
    if p.sprt:
        require(
            all(
                sprt.get(k) == v
                for k, v in {
                    "model": "normalized",
                    "alpha": 0.05,
                    "beta": 0.05,
                    "elo0": p.sprt[0],
                    "elo1": p.sprt[1],
                }.items()
            ),
            "Wrong normalized SPRT parameters",
        )
    engines = saved.get("engines")
    require(isinstance(engines, list) and len(engines) == 2, "Native engine inventory differs")
    for row, engine in zip(cast(list[object], engines), plan.engines, strict=True):
        native = _object(row)
        require(
            native.get("name") == engine.name
            and _path(str(native.get("cmd")).strip('"')) == engine.executable
            and native.get("restart") is False,
            "Native engine identity/restart differs",
        )
        require(
            native.get("args")
            == (
                subprocess.list2cmdline(engine.args)
                if sys.platform == "win32"
                else shlex.join(engine.args)
            ),
            "Native engine argv differs",
        )
        require(
            native.get("options") == [list(pair) for pair in engine.options],
            "Native UCI option configuration differs",
        )
        limits = _object(native.get("limit"))
        tc = _object(limits.get("tc"))
        require(
            limits.get("plies") == 0 and limits.get("nodes") == (p.nodes or 0),
            "Mixed or incorrect native search limits",
        )
        time_ms, increment_ms = (
            (0, 0) if p.tc is None else tuple(int(float(v) * 1000) for v in p.tc.split("+"))
        )
        require(
            tc
            == {
                "time": time_ms,
                "increment": increment_ms,
                "fixed_time": 0,
                "moves": 0,
                "timemargin": 0,
            },
            "Native time control/margin differs",
        )
    names = [engine.name for engine in plan.engines]
    raw_stats = _object(saved.get("stats"))
    require(set(raw_stats) == {names[0] + " vs " + names[1]}, "Unexpected native match statistics")
    values = _object(next(iter(raw_stats.values())))
    wdl = _object(statistics["wdl"])
    categories = _object(statistics["pair_categories"])
    expected_stats = {"wins": wdl["wins"], "losses": wdl["losses"], "draws": wdl["draws"]}
    expected_stats.update({"penta_" + key: value for key, value in categories.items()})
    require(
        values == expected_stats and all(type(v) is int and v >= 0 for v in values.values()),
        "Native raw WDL/pairs differ from every PGN result",
    )
    return saved


@dataclass
class _StartupCatalog:
    initialization_thread_marker: str
    received: list[str] = field(default_factory=list)


def _protocol(plan: Plan, path: Path) -> dict[str, object]:
    """Validate anonymous global-pool startups and exact global option refresh.

    Pinned base/tournament.cpp:95-114 uses a global cache without affinity;
    core/memory/cache.hpp:40-48 can lend an engine to another worker. Windows
    input logs retain the creator marker, while outgoing commands use the
    current caller. Markers are observation contexts, never process identity.
    UciEngine::start:310-337 initializes synchronously, so each outgoing uci
    opens an independent bounded catalog in that initialization context.
    """
    planned = {engine.name: dict(engine.options) for engine in plan.engines}
    totals = {name: dict.fromkeys(options, 0) for name, options in planned.items()}
    total_newgames = dict.fromkeys(planned, 0)
    outgoing: dict[str, list[str]] = {name: [] for name in planned}
    catalogs: dict[str, list[_StartupCatalog]] = {name: [] for name in planned}
    current: dict[tuple[str, str], _StartupCatalog] = {}
    refresh: dict[tuple[str, str], dict[str, int]] = {}
    newgames: dict[tuple[str, str], int] = {}
    names = "|".join(re.escape(engine.name) for engine in plan.engines)
    pattern = re.compile(r"(.*?) (" + names + r") (---\>|\<---) (.*)")
    for line in _lines(path):
        require(
            not re.match(r"\[(?:WARN|ERR|FATAL)\s*\]", line),
            "Native runner logged a warning/error/fatal event",
        )
        matched = pattern.fullmatch(line)
        if matched is None:
            continue
        prefix, name, direction, payload = matched.groups()
        require("<stderr>" not in prefix, "Engine wrote stderr in protocol trace")
        marker = re.search(r"<\s*([0-9]{1,24})\s*>\s*$", prefix)
        require(marker is not None, "Missing native worker observation marker")
        require(
            not payload.startswith("info string error "), "Engine reported an explicit UCI error"
        )
        context = (name, cast(re.Match[str], marker)[1])
        if payload == "uci":
            require(direction == "<---", "Unexpected engine-sent uci command")
            require(
                len(catalogs[name]) < plan.selected.concurrency,
                "More native startup catalogs than declared global pool capacity",
            )
            catalog = _StartupCatalog(context[1])
            catalogs[name].append(catalog)
            current[context] = catalog
            if payload not in outgoing[name]:
                outgoing[name].append(payload)
        elif payload == "uciok" or payload.startswith(("id name ", "id author ", "option name ")):
            require(
                direction == "--->" and context in current,
                "Catalog response has no own initialization command",
            )
            incoming = current[context].received
            require(len(incoming) < 4096, "Unexpected repeated handshake/options")
            incoming.append(payload)
        elif direction == "<---" and (payload.startswith("setoption") or payload == "ucinewgame"):
            if context not in refresh:
                refresh[context] = dict.fromkeys(planned[name], 0)
                newgames[context] = 0
            require(
                sum(key[0] == name for key in refresh) <= plan.selected.concurrency,
                "More refresh worker contexts than declared concurrency",
            )
            if payload == "ucinewgame":
                total_newgames[name] += 1
                newgames[context] += 1
                require(
                    total_newgames[name] <= plan.selected.games,
                    "Native ucinewgame count exceeded declared game cap",
                )
                continue
            setting = re.fullmatch(r"setoption name (\S+) value (.*)", payload)
            require(setting is not None, "Malformed native option setting")
            option, value = cast(re.Match[str], setting).groups()
            require(
                option in planned[name] and planned[name][option] == value,
                "Changed or unplanned native option setting",
            )
            # refreshUci:341-372 sends each configured setter once/game; the
            # extra Chess960=true is FRC-only. A stopped startup may receive no
            # refresh at all (Match::start:359), so don't attribute setters to
            # anonymous catalog/process instances using the worker marker.
            totals[name][option] += 1
            refresh[context][option] += 1
            require(
                totals[name][option] <= plan.selected.games,
                "Native option refresh count exceeded declared game cap",
            )
            if payload not in outgoing[name]:
                outgoing[name].append(payload)
    identities: dict[str, object] = {}
    for engine in plan.engines:
        require(catalogs[engine.name], "Missing actual engine identity/option handshake")
        evidence: list[dict[str, object]] = []
        for ordinal, catalog in enumerate(catalogs[engine.name], 1):
            incoming = catalog.received
            require(
                sum(line.startswith("id name ") for line in incoming) == 1
                and incoming.count("uciok") == 1
                and any(line.startswith("option name ") for line in incoming),
                "Missing or repeated actual engine identity/option handshake",
            )
            advertised = [
                re.fullmatch(
                    r"option name (.+) type (?:string|spin|combo|check|button)(?: .*)?", line
                )
                for line in incoming
                if line.startswith("option name ")
            ]
            require(
                all(match is not None for match in advertised), "Malformed advertised engine option"
            )
            option_names = [cast(re.Match[str], match).group(1) for match in advertised]
            require(
                len(set(option_names)) == len(option_names)
                and all(option in option_names for option, _ in engine.options),
                "Planned UCI option absent from an independent startup catalog",
            )
            evidence.append(
                dict(
                    catalog_ordinal=ordinal,
                    initialization_thread_marker=catalog.initialization_thread_marker,
                    received_handshake=incoming,
                    sent_initialization_command="uci",
                )
            )
        require(
            all(totals[engine.name][key] > 0 for key, _ in engine.options),
            "Not every planned option reached the global engine pool",
        )
        identities[engine.name] = {
            "startup_catalogs": evidence,
            "received_handshake": [
                line for catalog in catalogs[engine.name] for line in catalog.received
            ],
            "sent_options": outgoing[engine.name],
            "sent_option_counts": totals[engine.name],
            "ucinewgame_count": total_newgames[engine.name],
            "refresh_observation_contexts": [
                dict(
                    trace_thread_marker=context[1],
                    sent_option_counts=refresh[context],
                    ucinewgame_count=newgames[context],
                )
                for context in sorted(refresh)
                if context[0] == engine.name
            ],
            "identity_basis": (
                "Independent startup catalogs; shared global pool, no worker/process identity claim"
            ),
        }
    return identities


def _native_game_trace(
    plan: Plan, path: Path, events: Progress, *, natural_sprt_stop: bool
) -> dict[str, object]:
    """Keep worker game traces and prove each skipped internal assignment.

    RoundRobin::startNext allocates IDs before queued execution. Base playGame
    checks stop before the Game/start callback, but the wrapper still logs its
    finished-match receipt. FIFO ThreadPool queue clearing may erase unexecuted
    trailing IDs; it cannot explain an internal hole below a later executed ID.
    """
    pattern = re.compile(
        r".*? <\s*([0-9]{1,24})\s*> fastchess --- Game ([0-9]+) "
        r"between (\S+) and (\S+) (starting|finished)"
    )
    receipt_pattern = re.compile(
        r".*? <\s*([0-9]{1,24})\s*> fastchess --- Thread ([0-9]{1,24}) finished match ([0-9]+)"
    )
    begun: dict[int, tuple[str, str, str]] = {}
    ended: set[int] = set()
    occupied: dict[str, int] = {}
    pending_assignments: dict[str, int] = {}
    workers: set[str] = set()
    finish_lines: dict[int, int] = {}
    receipts: dict[int, tuple[str, int]] = {}
    stop_line: int | None = None
    for line_number, line in enumerate(_lines(path), 1):
        if stop_line is None and re.fullmatch(
            r"\[TRACE\s*\] .*? fastchess --- SPRT test finished, stopping tournament\.", line
        ):
            stop_line = line_number
        receipt = receipt_pattern.fullmatch(line)
        if receipt is not None:
            worker, payload_worker, token = receipt.groups()
            index = int(token)
            require(
                worker not in occupied,
                "Native assignment receipt worker is still executing a started game",
            )
            require(
                pending_assignments.get(worker) == index
                if index in events.started
                else worker not in pending_assignments,
                "Native receipt precedes completion of its worker's known assignment wrapper",
            )
            pending_assignments.pop(worker, None)
            require(
                worker == payload_worker
                and 1 <= index <= plan.selected.games
                and index not in receipts,
                "Invalid or duplicate native assignment receipt",
            )
            receipts[index] = (worker, line_number)
            workers.add(worker)
            require(len(workers) <= plan.selected.concurrency, "Excess native worker trace threads")
            continue
        require(
            "fastchess --- Thread " not in line or "finished match" not in line,
            "Malformed native assignment receipt",
        )
        matched = pattern.fullmatch(line)
        if matched is None:
            continue
        worker, token, white, black, phase = matched.groups()
        index = int(token)
        require(
            index in events.started and events.started[index][1:] == (white, black),
            "Native worker trace game ID or colors differ",
        )
        workers.add(worker)
        require(len(workers) <= plan.selected.concurrency, "Excess native worker trace threads")
        if phase == "starting":
            require(
                index not in begun and worker not in occupied and worker not in pending_assignments,
                "Duplicate or overlapping native worker game start or unfinished wrapper",
            )
            begun[index] = (worker, white, black)
            occupied[worker] = index
            pending_assignments[worker] = index
        else:
            require(
                index in begun
                and begun[index] == (worker, white, black)
                and index not in ended
                and occupied.get(worker) == index,
                "Missing, duplicate or wrong-worker native game finish",
            )
            ended.add(index)
            finish_lines[index] = line_number
            del occupied[worker]
    require(
        set(begun) == ended == set(events.started) and not occupied and not pending_assignments,
        "Native worker traces/receipts do not complete every started assignment",
    )
    skipped: list[int] = []
    for index, (worker, line_number) in receipts.items():
        if index in begun:
            require(
                worker == begun[index][0] and line_number > finish_lines[index],
                "Assignment receipt differs from its actual finished game trace",
            )
        else:
            require(
                natural_sprt_stop
                and plan.selected.concurrency > 1
                and stop_line is not None
                and line_number > stop_line,
                "Skipped assignment has no preceding verified native SPRT stop",
            )
            skipped.append(index)
    return dict(
        started_trace_game_ids=sorted(begun),
        completed_trace_game_ids=sorted(ended),
        trace_thread_markers=sorted(workers),
        game_trace_thread_markers={str(index): value[0] for index, value in begun.items()},
        cancelled_inflight_trace_ids=sorted(events.active),
        observed_assignment_receipt_ids=sorted(receipts),
        assignment_receipt_thread_markers={
            str(index): value[0] for index, value in receipts.items()
        },
        skipped_before_start_game_ids=sorted(skipped),
        first_native_sprt_stop_trace_line=stop_line,
        assignment_evidence_basis=(
            "Observed native wrapper receipts only; no unexecuted queued trailing IDs are invented"
        ),
    )


def _native_invocation(
    plan: Plan, raw: Path, captured_command: Sequence[str] | None
) -> dict[str, object]:
    require(captured_command is not None, "Missing actual captured native command")
    command = _strings(list(cast(Sequence[str], captured_command)))
    require(
        command.count("-strict") == 1 and command == build_command(plan, raw),
        "Actual captured native command differs from strict planned invocation",
    )
    require(
        command[0] == str(plan.fastchess)
        and dict(plan.hashes).get(plan.fastchess) == FASTCHESS_SHA256
        and _sha(plan.fastchess) == FASTCHESS_SHA256,
        "Actual invocation requires frozen pinned fastchess binary",
    )
    return {
        "command": command,
        "strict_occurrences": 1,
        "fastchess_sha256": FASTCHESS_SHA256,
        "fastchess_source_commit": FASTCHESS_COMMIT,
        "basis": "Actual captured full argv checked against pinned native CLI contract",
    }


def _native_boundary_snapshot(
    plan: Plan,
    games: Sequence[Game],
    events: Progress,
    summary: tuple[int, int, int, int] | None,
    penta: list[int] | None,
    llr: float | None,
    decision: str,
    line_number: int,
) -> dict[str, object]:
    """Replay the native complete-pair scoreboard in actual finish-event order."""
    names = (plan.engines[0].name, plan.engines[1].name)
    by_id: dict[int, Game] = {}
    for game in games:
        colors = (game.tags["White"], game.tags["Black"])
        require(colors in (names, names[::-1]), "Unexpected boundary PGN colors")
        index = 2 * int(game.tags["Round"]) - (1 if colors == names else 0)
        require(index not in by_id, "Duplicate boundary PGN game ID")
        by_id[index] = game
    seen: set[int] = set()
    paired: list[Game] = []
    rounds: list[int] = []
    for index in events.finished:
        require(index in by_id, "Boundary finish has no retained PGN")
        partner = index + 1 if index % 2 else index - 1
        if partner in seen:
            pair = (index + 1) // 2
            rounds.append(pair)
            paired.extend((by_id[2 * pair - 1], by_id[2 * pair]))
        seen.add(index)
    require(paired, "Native SPRT boundary has no complete color pair")
    statistics = pair_statistics(paired, allow_round_gaps=True)
    wdl = _object(statistics["wdl"])
    require(
        summary == (len(paired), wdl["wins"], wdl["losses"], wdl["draws"])
        and penta == statistics["pentanomial"],
        "Boundary WDL/penta differs from actual finish-order complete pairs",
    )
    require(llr is not None and math.isfinite(llr), "Missing finite boundary LLR")
    value = cast(float, llr)
    require(
        value + 0.005 >= math.log(19) if decision == "H1" else value - 0.005 <= -math.log(19),
        "Native decision inconsistent with its captured rounded normalized LLR",
    )
    return dict(
        decision=decision,
        stdout_line=line_number,
        started_game_ids_at_decision=list(events.started),
        active_game_ids_at_decision=sorted(events.active),
        completed_game_ids_in_order=list(events.finished),
        complete_pair_round_ids_in_order=rounds,
        normalized_llr_printed=value,
        captured_wdl_summary=list(cast(tuple[int, ...], summary)),
        captured_pentanomial=list(cast(list[int], penta)),
        statistics=statistics,
        normalized_model_basis=(
            "Pinned native normalized GLRT; prefix raw penta/WDL independently reconciled"
        ),
    )


def audit_results(
    plan: Plan,
    raw: Path,
    *,
    captured_command: Sequence[str] | None = None,
    process: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Reconcile captured invocation, every game, color pair and raw summary."""
    invocation = _native_invocation(plan, raw, captured_command)
    games = read_games(
        raw / "games.pgn",
        (plan.engines[0].name, plan.engines[1].name),
        allow_adjudication=plan.selected.adjudication,
    )
    require(len(games) <= plan.selected.games, "More games than declared cap/count")
    events = Progress(
        plan.selected.games,
        (plan.engines[0].name, plan.engines[1].name),
        concurrency=plan.selected.concurrency,
        allow_adjudication=plan.selected.adjudication,
    )
    summary: tuple[int, int, int, int] | None = None
    penta: list[int] | None = None
    decisions: list[str] = []
    decision_snapshots: list[dict[str, object]] = []
    llr: float | None = None
    for line_number, line in enumerate(_lines(raw / "stdout.txt"), 1):
        require(
            not re.search(
                r"(?i)(?:\b(?:warning|error);|\b(?:crashed|disconnects|interrupted|aborted|"
                r"illegal move|loses on time)\b)",
                line,
            ),
            "Native match failure in stdout",
        )
        if line.startswith(("Started game ", "Finished game ")):
            events(line)
        elif line.startswith("Games: "):
            match = re.fullmatch(
                r"Games: ([0-9]+), Wins: ([0-9]+), Losses: ([0-9]+), Draws: ([0-9]+), "
                r"Points: ([0-9]+\.[0-9]) \(([0-9]+\.[0-9]{2}) %\)",
                line,
            )
            require(match is not None, "Malformed WDL summary")
            values = cast(re.Match[str], match).groups()
            summary = cast(tuple[int, int, int, int], tuple(map(int, values[:4])))
            total, wins, losses, draws = summary
            require(total == wins + losses + draws and total > 0, "Inconsistent raw WDL total")
            twice_points = 2 * wins + draws
            printed_score = Decimal(values[5])
            require(
                Decimal(values[4]) * 2 == twice_points
                and (printed_score - Decimal(".005")) * total
                <= twice_points * 50
                <= (printed_score + Decimal(".005")) * total,
                "Printed points/score differ from WDL",
            )
        elif line.startswith("Ptnml(0-2): "):
            match = re.fullmatch(
                r"Ptnml\(0-2\): \[([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\], "
                r"WL/DD Ratio: (\S+)",
                line,
            )
            require(match is not None, "Malformed paired summary")
            penta = list(map(int, cast(re.Match[str], match).groups()[:5]))
        elif line.startswith("LLR: "):
            match = re.fullmatch(
                r"LLR: (-?[0-9]+\.[0-9]{2}) \(-?[0-9]+\.[0-9]%\) \(-2\.94, 2\.94\) "
                r"\[(-?[0-9]+\.[0-9]{2}), (-?[0-9]+\.[0-9]{2})\]",
                line,
            )
            require(
                match is not None and plan.selected.sprt is not None, "Malformed/wrong SPRT summary"
            )
            value, e0, e1 = cast(re.Match[str], match).groups()
            require((float(e0), float(e1)) == plan.selected.sprt, "SPRT bounds differ")
            llr = float(value)
        elif line.startswith("SPRT ("):
            match = re.fullmatch(
                r"SPRT \(\[(-?[0-9]+\.[0-9]{2}), (-?[0-9]+\.[0-9]{2})\]\) "
                r"completed - (H0|H1) was accepted",
                line,
            )
            require(
                match is not None and plan.selected.sprt is not None, "Wrong native SPRT decision"
            )
            e0, e1, accepted = cast(re.Match[str], match).groups()
            require((float(e0), float(e1)) == plan.selected.sprt, "Native SPRT hypothesis differs")
            require(not events.closed, "Native SPRT decision after runner closure")
            snapshot = _native_boundary_snapshot(
                plan, games, events, summary, penta, llr, accepted, line_number
            )
            if decisions:
                require(
                    plan.selected.concurrency > 1
                    and accepted == decisions[0]
                    and len(events.finished)
                    > len(cast(list[int], decision_snapshots[-1]["completed_game_ids_in_order"])),
                    "Duplicate native SPRT decision without new completed game or decision flipped",
                )
            decisions.append(accepted)
            decision_snapshots.append(snapshot)
        elif line == "Finished match":
            events(line)
    decision: str | None = None
    if plan.selected.sprt:
        require(llr is not None and math.isfinite(llr), "Missing finite final SPRT LLR")
        actual_llr = cast(float, llr)
        if decisions:
            decision = decisions[0]
        else:
            require(
                len(games) == plan.selected.games and abs(actual_llr) - 0.005 < math.log(19),
                "Stopped SPRT without valid decision or complete cap",
            )
            decision = "inconclusive"
    else:
        require(
            not decisions and len(games) == plan.selected.games,
            "Fixed-count gate incomplete or unexpected SPRT",
        )
    natural_stop = decision in ("H0", "H1")
    if natural_stop and plan.selected.concurrency > 1:
        require(process is not None, "Natural concurrent SPRT tails require actual process capture")
        native = cast(Mapping[str, object], process)
        require(
            type(native.get("exit_code")) is int
            and native["exit_code"] == 0
            and native.get("pumps_joined") is True
            and native.get("terminated_by_runner") is False
            and native.get("error") is None
            and native.get("command") == list(cast(Sequence[str], captured_command)),
            "Concurrent SPRT did not naturally exit cleanly with its actual invocation",
        )
    identities = _protocol(plan, raw / "protocol.log")
    for engine in plan.engines:
        catalogs = cast(list[object], _object(identities[engine.name])["startup_catalogs"])
        require(
            len(catalogs) >= events.peak_active,
            "Global pool startup count cannot support peak simultaneous native games",
        )
    trace = (
        _native_game_trace(plan, raw / "protocol.log", events, natural_sprt_stop=natural_stop)
        if plan.selected.concurrency > 1
        else {}
    )
    verified_skipped = cast(list[int], trace.get("skipped_before_start_game_ids", []))
    paired, inventory = _game_ledger(
        plan,
        games,
        events,
        natural_sprt_stop=natural_stop,
        verified_skipped_assignments=verified_skipped,
    )
    if plan.selected.concurrency == 1:
        require(
            inventory["raw_pgn_game_ids"] == sorted(events.finished),
            "Sequential profile PGN order differs from native game IDs",
        )
    statistics = pair_statistics(
        paired, allow_round_gaps=natural_stop and plan.selected.concurrency > 1
    )
    wdl = _object(statistics["wdl"])
    require(
        summary == (len(paired), wdl["wins"], wdl["losses"], wdl["draws"])
        and penta == statistics["pentanomial"],
        "Last native WDL/penta summary differs from actual complete PGN pairs",
    )
    for line in _lines(raw / "stderr.txt"):
        require(not line.strip(), "Unexpected runner stderr retained as failed evidence")
    saved = _saved_configuration(plan, raw, statistics)
    post_decision_tail_ids: list[int] = []
    if decision_snapshots and plan.selected.concurrency > 1:
        first_completed = set(cast(list[int], decision_snapshots[0]["completed_game_ids_in_order"]))
        started_tails = set(events.started) - first_completed
        post_decision_tail_ids = sorted(started_tails | set(verified_skipped))
        # Pinned base/tournament.cpp checks stop before starting a game; the
        # first decision sets stop under the output mutex. Started tails passed
        # that precheck and cannot reuse a worker. Queued skipped assignments
        # may later return serially on any worker, so don't claim their worker
        # ownership is unique. Count both kinds against the admitted-job bound.
        require(
            len(post_decision_tail_ids) <= plan.selected.concurrency,
            "More post-decision game tails than declared native concurrency",
        )
        markers = cast(dict[str, str], trace["game_trace_thread_markers"])
        require(
            len({markers[str(index)] for index in started_tails}) == len(started_tails),
            "Native worker restarted another game after the first SPRT decision",
        )
    book = _book_membership(plan, games)
    final_pair_rounds = cast(list[int], inventory["complete_pair_round_ids"])
    stop_pair_rounds = (
        cast(list[int], decision_snapshots[0]["complete_pair_round_ids_in_order"])
        if decision_snapshots
        else final_pair_rounds
    )
    return {
        "complete": True,
        "safety_pass": True,
        "native_invocation": invocation,
        "sprt_decision": decision,
        "statistics": decision_snapshots[0]["statistics"] if decision_snapshots else statistics,
        "native_final_statistics": statistics,
        "sprt_stop_point": decision_snapshots[0] if decision_snapshots else None,
        "native_sprt_decisions": decision_snapshots,
        "late_completion_game_ids": (
            list(events.finished)[
                len(cast(list[int], decision_snapshots[0]["completed_game_ids_in_order"])) :
            ]
            if decision_snapshots
            else []
        ),
        "game_inventory": inventory,
        "native_game_trace": trace,
        "peak_native_active_games": events.peak_active,
        "post_decision_tail_game_ids": post_decision_tail_ids,
        "nominal_fixed_nodes": plan.selected.nodes,
        **(
            {
                "fixed_node_settings": {
                    "schema": "MouseFixedNodeSettings1",
                    "concurrency": plan.selected.concurrency,
                    "nominal_nodes": plan.selected.nodes,
                    "reference_version": "v0.32.1",
                    "reference_source_commit": LC0_SOURCE_COMMIT,
                    "reference_sha256": LC0_RELEASE_SHA256,
                    "equal_work_claim": False,
                    "engine_options": {
                        engine.name: dict(engine.options) for engine in plan.engines
                    },
                    "adjudication": {
                        "draw": dict(movenumber=34, movecount=8, score=10),
                        "resign": dict(movecount=3, score=1000, twosided=True),
                        "maxmoves": 192,
                        "opening_plies": 16,
                        "maximum_total_plies": 400,
                        "basis": "parity contract 2026-10-17; pinned native units",
                    },
                }
            }
            if plan.selected.name == "parity-sprt" and plan.selected.adjudication
            else {}
        ),
        "node_accounting_basis": (
            "Actual per-ply PGN n values are retained separately from configured go nodes; "
            "no equal-visit claim or numerical overshoot tolerance is inferred. Pinned Lc0 "
            "classic/stoppers/stoppers.cc:59-68 checks completed total_nodes against the limit; "
            "classic/search.cc:275 reports total_playouts plus initial_visits."
        ),
        "book_membership_validated": True,
        "book_usage": book,
        "engine_handshakes": identities,
        "native_raw_statistics": saved["stats"],
        "native_normalized_llr_printed": llr,
        "normalized_llr_basis": (
            "Pinned fastchess normalized model; actual native decision and printed LLR/bounds "
            "reconciled, not an independently recomputed GLRT"
        ),
        "game_records": [
            {
                "id": game_id,
                "raw_pgn_record": i,
                "counted_in_native_final_pair_statistics": int(g.tags["Round"])
                in final_pair_rounds,
                "counted_at_statistical_stop_point": int(g.tags["Round"]) in stop_pair_rounds,
                "white": g.tags["White"],
                "black": g.tags["Black"],
                "result": g.result,
                "termination": g.tags["Termination"],
                "native_reason": events.finished[game_id][3],
                "round": g.tags["Round"],
                "plies": len(g.moves),
                "actual_nodes_by_ply": list(g.nodes_by_ply),
            }
            for i, (game_id, g) in enumerate(
                zip(cast(list[int], inventory["raw_pgn_game_ids"]), games, strict=True), 1
            )
        ],
    }


def run_match(config: Path, output: Path, *, execute: bool) -> dict[str, object]:
    plan = load_plan(config)
    output = output.resolve()
    require(not output.exists(), "Existing match evidence is immutable")
    require(
        not any(output == p or p.is_relative_to(output) for p, _ in plan.hashes),
        "Output directory overlaps immutable input",
    )
    require(
        not any(
            output.is_relative_to(p) or p.is_relative_to(output)
            for p in plan.source_roots + plan.runtime_directories
        ),
        "Output overlaps an immutable source/runtime directory",
    )
    output.mkdir(parents=True, exist_ok=False)
    command = build_command(plan, output)
    report: dict[str, object] = {
        "schema": "MouseMatchReport1",
        "status": "planned",
        "passed": False,
        "native_execution": False,
        "profile": asdict(plan.selected),
        "seed": 20261017,
        "fastchess_version": FASTCHESS_VERSION,
        "fastchess_source_commit": FASTCHESS_COMMIT,
        "command": command,
        "artifacts": {str(p): h for p, h in plan.hashes},
        "evidence_valid": False,
        "complete": False,
        "safety_pass": False,
        "sprt_decision": None,
        "errors": [],
        "full_release_gate": False,
        "search_contract": {
            "mouse_required_features": (
                "Parent-reviewed frozen sources enable MLH and terminal proof; "
                "no invented UCI switches"
            ),
            "node_accounting": (
                "Nominal per-move go nodes and actual per-ply PGN n are distinct; normal "
                "batched/auto reference work is retained without an invented overshoot "
                "or equal-visit claim"
            ),
            "source_feature_audit": (
                "Parent review binds implementation semantics; runner verifies complete "
                "source/artifact identities"
            ),
        },
    }
    if plan.selected.name == "package-smoke":
        report.update(acceptance_scope="package-smoke-only", roadmap_gate_claim=False)
    _publish(output / "plan.json", report)
    errors = cast(list[str], report["errors"])
    try:
        plan.verify()
        if execute:
            report["native_execution"] = True
            process = capture(
                command,
                output,
                idle_seconds=plan.idle_seconds,
                total_seconds=plan.total_seconds,
                max_bytes=plan.max_output_bytes,
                on_stdout=Progress(
                    plan.selected.games,
                    (plan.engines[0].name, plan.engines[1].name),
                    concurrency=plan.selected.concurrency,
                    allow_adjudication=plan.selected.adjudication,
                ),
            )
            report["process"] = process
            raw = {p.name: _sha(p) for p in output.iterdir() if p.is_file()}
            report["raw_sha256"] = raw
            require(
                process["exit_code"] == 0 and process["pumps_joined"] and not process["error"],
                f"Native match did not finish cleanly: {process}",
            )
            report.update(
                audit_results(
                    plan, output, captured_command=_strings(process["command"]), process=process
                )
            )
            require(
                raw == {p.name: _sha(p) for p in output.iterdir() if p.is_file()},
                "Raw match evidence changed during audit",
            )
            report["evidence_valid"] = True
            report["passed"] = report["sprt_decision"] in (None, "H1")
            report["status"] = (
                "passed"
                if report["passed"]
                else ("inconclusive" if report["sprt_decision"] == "inconclusive" else "failed")
            )
    except KeyboardInterrupt:
        errors.append("Interrupted; partial match is not a gate")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    try:
        plan.verify()
        report["hashes_unchanged"] = True
    except (OSError, ValueError) as error:
        errors.append(f"Artifact verification: {error}")
        report["hashes_unchanged"] = False
    if errors:
        report["observed_evidence_valid"] = report["evidence_valid"]
        report.update(status="failed", passed=False, evidence_valid=False, safety_pass=False)
    if execute:
        report["retained_file_sha256"] = {p.name: _sha(p) for p in output.iterdir() if p.is_file()}
    destination = output / "report.json"
    report["report_path"] = str(destination)
    try:
        _publish(destination, report)
    except FileExistsError:
        report.update(status="failed", passed=False, evidence_valid=False, safety_pass=False)
        errors.append("Final report publication raced; existing evidence preserved")
        destination = output / ("report.failed-" + uuid.uuid4().hex + ".json")
        report["report_path"] = str(destination)
        _publish(destination, report)
    return report


def _isolated() -> bool:
    return bool(sys.flags.isolated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(_isolated(), "Run with isolated Python: python -I -m mouse_tools.match")
        result = run_match(args.config, args.output, execute=args.action == "run")
    except (OSError, ValueError, TypeError) as error:
        print(f"Match runner error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "report": result["report_path"],
            }
        )
    )
    return 0 if result["status"] == "planned" or result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
