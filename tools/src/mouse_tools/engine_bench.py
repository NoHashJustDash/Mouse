# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen fixed16 common UCI go-to-bestmove benchmark; no native default discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from mouse_tools import uci_gate
from mouse_tools.uci_gate import UciGateError, UciProcess

CORPUS_SHA256 = "4b4302c2b2c164857d327eef55c82e7d6c702282bf611aae05b6dc22d7fdb9a5"
ORACLE_COMMIT = "03e27488f3d21d8ff4dbf3065603afa21dbd0ef3"
ORDER = ("mouse", "lc0", "lc0", "mouse", "mouse", "lc0")
MOVE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")
REQUIRED_OPTIONS = {"NNCacheSize": "0", "MultiPV": "1", "Ponder": "false", "UCI_Chess960": "false"}


class EngineBenchError(ValueError):
    """Invalid, changed or incomplete evidence cannot produce a passing ratio."""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _object(value: object, keys: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise EngineBenchError(f"Invalid or unknown {name} fields")
    return cast(dict[str, object], value)


def _text(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 65536
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise EngineBenchError("Expected bounded text without control characters")
    return value


def _digest(value: object) -> str:
    text = _text(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise EngineBenchError("Expected lowercase SHA256")
    return text


def _path(value: object, *, directory: bool = False) -> Path:
    path = Path(_text(value))
    if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
        raise EngineBenchError("Paths must be absolute existing regular files/directories")
    return path.resolve(strict=True)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EngineBenchError("Duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class FrozenFile:
    path: Path
    sha256: str
    role: str


def _file(value: object, role: str) -> FrozenFile:
    item = _object(value, {"path", "sha256"}, role)
    return FrozenFile(_path(item["path"]), _digest(item["sha256"]), role)


@dataclass(frozen=True)
class Child:
    argv: tuple[str, ...]
    cwd: Path
    executable: FrozenFile
    name_prefix: str
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Case:
    identifier: str
    fen: str
    moves: tuple[str, ...]

    def command(self) -> str:
        return "position fen " + self.fen + (" moves " + " ".join(self.moves) if self.moves else "")


@dataclass(frozen=True)
class Config:
    path: Path
    declaration: dict[str, object]
    engines: dict[str, Child]
    oracle: Child
    cases: tuple[Case, ...]
    files: tuple[FrozenFile, ...]
    startup: float
    search: float
    quit: float

    def verify(self) -> None:
        for item in self.files:
            if not item.path.is_file() or _sha(item.path) != item.sha256:
                raise EngineBenchError(f"Immutable hash changed: {item.path}")


def _argv(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 128:
        raise EngineBenchError("Expected bounded argv array")
    result = tuple(_text(x) for x in value)
    _path(result[0])
    return result


def _flag(argv: tuple[str, ...], flag: str) -> str:
    found = []
    for index, arg in enumerate(argv):
        if arg == flag:
            if index + 1 == len(argv):
                raise EngineBenchError("Missing explicit engine flag value")
            found.append(argv[index + 1])
        elif arg.startswith(flag + "="):
            found.append(arg[len(flag) + 1 :])
    if len(found) != 1:
        raise EngineBenchError(f"Require one explicit {flag}")
    return found[0]


def _seconds(value: object) -> float:
    if (
        type(value) not in (int, float)
        or not 0 < cast(float, value) <= 300
        or not math.isfinite(cast(float, value))
    ):
        raise EngineBenchError("Deadlines must be finite in (0,300] seconds")
    return float(cast(float, value))


def load_config(path: Path) -> Config:
    path = _path(str(path))
    if path.stat().st_size > 1024 * 1024:
        raise EngineBenchError("Config exceeds bounded JSON size")
    config_bytes = path.read_bytes()
    raw: object = json.loads(config_bytes.decode("utf-8"), object_pairs_hook=_unique)
    data = _object(
        raw,
        {
            "schema",
            "nodes",
            "repetitions",
            "order",
            "corpus",
            "corpus_sha256",
            "source_net",
            "timeouts",
            "oracle",
            "engines",
        },
        "config",
    )
    if (
        data["schema"] != "MouseEngineBench1"
        or type(data["nodes"]) is not int
        or data["nodes"] != 800
        or type(data["repetitions"]) is not int
        or data["repetitions"] != 3
        or data["order"] != list(ORDER)
    ):
        raise EngineBenchError("Fixed16/800/three-repetition method cannot be changed")
    corpus = _path(data["corpus"])
    if data["corpus_sha256"] != CORPUS_SHA256 or _sha(corpus) != CORPUS_SHA256:
        raise EngineBenchError("Fixed corpus hash mismatch")
    cases = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line:
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise EngineBenchError("Malformed fixed corpus")
        cases.append(Case(fields[0], fields[1], tuple(fields[2].split())))
    if len(cases) != 16 or len({x.identifier for x in cases}) != 16:
        raise EngineBenchError("Require complete fixed16 corpus")
    source = _file(data["source_net"], "source_net")
    files = [
        source,
        FrozenFile(path, hashlib.sha256(config_bytes).hexdigest(), "config"),
        FrozenFile(corpus, CORPUS_SHA256, "corpus"),
    ]
    # Freeze the exact producer and reused bounded transport too.
    files += [
        FrozenFile(p.resolve(), _sha(p), "tool_source")
        for p in (Path(__file__), Path(uci_gate.__file__))
    ]
    engines = {}
    definitions = _object(data["engines"], {"mouse", "lc0"}, "engines")
    for role in ("mouse", "lc0"):
        item = _object(
            definitions[role],
            {
                "argv",
                "cwd",
                "executable_sha256",
                "name_prefix",
                "backend",
                "precision",
                "model",
                "source_sha256",
                "artifacts",
                "options",
            },
            role,
        )
        argv = _argv(item["argv"])
        executable = FrozenFile(_path(argv[0]), _digest(item["executable_sha256"]), "executable")
        model = _file(item["model"], "model")
        if item["source_sha256"] != source.sha256:
            raise EngineBenchError("Engines must declare the same frozen source-net hash")
        if role == "lc0" and model != FrozenFile(source.path, source.sha256, "model"):
            raise EngineBenchError("Reference must load the exact original source net")
        backend = _text(item["backend"])
        if backend not in (
            ("cpu", "cuda", "tensorrt") if role == "mouse" else ("cuda-fp16", "cuda", "onnx-cpu")
        ) or item["precision"] not in ("fp16", "fp32"):
            raise EngineBenchError("Explicit supported backend and precision required")
        if _path(_flag(argv, "--weights")) != model.path or _flag(argv, "--backend") != backend:
            raise EngineBenchError("Engine argv must bind declared model and backend")
        if role == "lc0" and argv.count("--config=") != 1:
            raise EngineBenchError(
                "Reference UCI requires exactly --config= to disable config discovery"
            )
        if role == "mouse" and _flag(argv, "--precision") != item["precision"]:
            raise EngineBenchError("Mouse argv must bind declared precision")
        if role == "lc0" and (
            (backend == "cuda-fp16" and item["precision"] != "fp16")
            or (backend == "cuda" and item["precision"] != "fp32")
        ):
            raise EngineBenchError("Reference backend does not match declared precision")
        options = item["options"]
        if not isinstance(options, dict) or not 4 <= len(options) <= 128:
            raise EngineBenchError("Explicit bounded UCI options required")
        transactions = tuple((_text(k), _text(v)) for k, v in options.items())
        if any(
            cast(dict[str, object], options).get(k) != v for k, v in REQUIRED_OPTIONS.items()
        ) or len({k.casefold() for k, _ in transactions}) != len(transactions):
            raise EngineBenchError("Cache0, MultiPV1, standard chess and no pondering are required")
        if any(
            k.casefold() in ("weightsfile", "backend", "configfile", "precision")
            for k, _ in transactions
        ):
            raise EngineBenchError(
                "Model/backend/precision cannot be overridden by UCI transactions"
            )
        artifacts = item["artifacts"]
        if not isinstance(artifacts, list) or not 2 <= len(artifacts) <= 4096:
            raise EngineBenchError("Frozen source/runtime inventories required")
        roles = set()
        for artifact in artifacts:
            entry = _object(artifact, {"path", "sha256", "role"}, "artifact")
            artifact_role = _text(entry["role"])
            if artifact_role not in ("source", "runtime"):
                raise EngineBenchError("Artifact role must be source or runtime")
            roles.add(artifact_role)
            files.append(FrozenFile(_path(entry["path"]), _digest(entry["sha256"]), artifact_role))
        if roles != {"source", "runtime"}:
            raise EngineBenchError("Both source and runtime inventories are mandatory")
        files += [executable, model]
        prefix = _text(item["name_prefix"])
        if prefix != ("Mouse " if role == "mouse" else "Lc0 "):
            raise EngineBenchError("Require expected native engine identity prefix")
        engines[role] = Child(
            argv, _path(item["cwd"], directory=True), executable, prefix, transactions
        )
    oracle = _object(
        data["oracle"], {"argv", "cwd", "executable_sha256", "source_commit"}, "oracle"
    )
    if oracle["source_commit"] != ORACLE_COMMIT:
        raise EngineBenchError("Require pinned Stockfish17.1 oracle source")
    argv = _argv(oracle["argv"])
    executable = FrozenFile(_path(argv[0]), _digest(oracle["executable_sha256"]), "oracle")
    files.append(executable)
    by_path: dict[Path, FrozenFile] = {}
    for item_file in files:
        if item_file.path in by_path and by_path[item_file.path].sha256 != item_file.sha256:
            raise EngineBenchError("Conflicting hash declarations for one artifact")
        by_path[item_file.path] = item_file
    deadlines = _object(
        data["timeouts"], {"startup_seconds", "search_seconds", "quit_seconds"}, "timeouts"
    )
    return Config(
        path,
        data,
        engines,
        Child(argv, _path(oracle["cwd"], directory=True), executable, "Stockfish 17.1"),
        tuple(cases),
        tuple(by_path.values()),
        _seconds(deadlines["startup_seconds"]),
        _seconds(deadlines["search_seconds"]),
        _seconds(deadlines["quit_seconds"]),
    )


def _explicit_error(text: str) -> bool:
    # Stockfish17.1 nnue/network.cpp emits uppercase ERROR: through info string.
    return re.match(r"(?:info string )?error\b", text, re.IGNORECASE) is not None


def _line(process: UciProcess, deadline: float) -> tuple[str, int]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise EngineBenchError("Protocol phase deadline exceeded")
    text = process.line(timeout=remaining)
    received = time.perf_counter_ns()  # Before parsing: driver's complete-line receipt boundary.
    if any(ord(c) < 32 or ord(c) == 127 for c in text) or _explicit_error(text):
        raise EngineBenchError("Engine emitted a malformed line or explicit error")
    return text, received


def _ready(process: UciProcess, timeout: float) -> None:
    process.timeout = timeout
    process.send("isready")
    deadline = time.monotonic() + timeout
    while True:
        line, _ = _line(process, deadline)
        if line == "readyok":
            return
        if line and not line.startswith("info string "):
            raise EngineBenchError("Unexpected response outside a search/readiness barrier")


def _handshake(process: UciProcess, child: Child, timeout: float) -> list[str]:
    process.send("uci")
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    options: set[str] = set()
    names: list[str] = []
    stockfish_separator_seen = False
    while True:
        line, _ = _line(process, deadline)
        lines.append(line)
        if len(lines) > 1024:
            raise EngineBenchError("Handshake exceeds bound")
        if line == "uciok":
            break
        if (
            child.name_prefix == "Stockfish 17.1"
            and len(lines) == 1
            and line == "Stockfish 17.1 by the Stockfish developers (see AUTHORS file)"
        ):
            continue
        if (
            child.name_prefix == "Stockfish 17.1"
            and line == ""
            and not stockfish_separator_seen
            and names == ["Stockfish 17.1"]
            and not options
            and lines[-2:] == ["id author the Stockfish developers (see AUTHORS file)", ""]
        ):
            # The pinned official17.1 binary emits this one author/options
            # separator. A duplicate or blank from another child still rejects.
            stockfish_separator_seen = True
            continue
        if line.startswith("id name "):
            names.append(line.removeprefix("id name "))
        elif line.startswith("id author "):
            continue
        elif line.startswith("option name "):
            match = re.fullmatch(
                r"option name (.+) type (?:string|spin|check|combo|button)(?: .*)?", line
            )
            if match is None or match[1] in options:
                raise EngineBenchError("Malformed/duplicate UCI option")
            options.add(match[1])
        else:
            raise EngineBenchError("Unexpected UCI handshake output")
    if len(names) != 1 or not names[0].startswith(child.name_prefix):
        raise EngineBenchError("Unexpected engine identity")
    if any(name not in options for name, _ in child.options):
        raise EngineBenchError("Requested UCI option not advertised")
    for name, value in child.options:
        process.send(f"setoption name {name} value {value}")
    _ready(process, timeout)
    return lines


def _integer(text: str) -> int:
    if re.fullmatch(r"[0-9]{1,20}", text) is None or int(text) > (1 << 64) - 1:
        raise EngineBenchError("Malformed/overflowing native node count")
    return int(text)


def _perft_diagnostic(line: str) -> str | None:
    # Pinned Stockfish17.1 uci.cpp prints NUMA/thread information before go;
    # engine.cpp verifies two networks before perft via nnue/network.cpp.
    cpus = r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*"
    if re.fullmatch(r"info string Available processors: " + cpus + r"(?::" + cpus + r")*", line):
        return "processors"
    if re.fullmatch(
        r"info string Using 1 thread(?: with NUMA node thread binding: "
        r"[0-9]+/[0-9]+(?::[0-9]+/[0-9]+)*)?",
        line,
    ):
        return "threads"
    network = re.fullmatch(
        r"info string NNUE evaluation using ([A-Za-z0-9_.-]+\.nnue) "
        r"\([0-9]+MiB, \([0-9]+, [0-9]+, [0-9]+, [0-9]+, 1\)\)",
        line,
    )
    return "network:" + network[1] if network else None


def _oracle(config: Config, report: dict[str, object]) -> dict[str, tuple[str, ...]]:
    legal: dict[str, tuple[str, ...]] = {}
    record: dict[str, object] = {"cases": []}
    report["oracle"] = record
    process = UciProcess(config.oracle.argv, timeout=config.startup, cwd=config.oracle.cwd)
    try:
        with process:
            record["handshake"] = _handshake(process, config.oracle, config.startup)
            process.send("setoption name Threads value 1")
            process.send("setoption name UCI_Chess960 value false")
            _ready(process, config.startup)
            for case in config.cases:
                process.send(case.command())
                process.send("go perft 1")
                deadline = time.monotonic() + config.search
                moves: list[str] = []
                info: list[str] = []
                diagnostics: set[str] = set()
                while True:
                    line, _ = _line(process, deadline)
                    if line.startswith("Nodes searched: "):
                        total = _integer(line.removeprefix("Nodes searched: "))
                        if total == 0 or total != len(moves) or len(set(moves)) != len(moves):
                            raise EngineBenchError(
                                "Incomplete/nonterminal legal-root oracle evidence"
                            )
                        break
                    if line.startswith("info string "):
                        kind = _perft_diagnostic(line)
                        if (
                            moves
                            or kind is None
                            or kind in diagnostics
                            or len(info) >= 4
                            or (
                                kind.startswith("network:")
                                and sum(key.startswith("network:") for key in diagnostics) >= 2
                            )
                        ):
                            raise EngineBenchError("Unexpected/duplicate perft oracle diagnostic")
                        diagnostics.add(kind)
                        info.append(line)
                        continue
                    match = re.fullmatch(r"([a-h][1-8][a-h][1-8][qrbn]?): ([0-9]+)", line)
                    if match:
                        if _integer(match[2]) != 1:
                            raise EngineBenchError("Perft1 edge must have one leaf")
                        moves.append(match[1])
                    elif line:
                        raise EngineBenchError("Unexpected perft oracle output")
                legal[case.identifier] = tuple(moves)
                cast(list[object], record["cases"]).append(
                    {
                        "id": case.identifier,
                        "position": case.command(),
                        "legal_moves": moves,
                        "info": info,
                    }
                )
                _ready(process, config.startup)
            _quit(process, config.quit)
    finally:
        if hasattr(process, "process"):
            record["process"] = process.evidence()
    _check_exit(cast(dict[str, object], record["process"]))
    _audit_capture(cast(dict[str, object], record["process"]), 0)
    return legal


def _check_exit(evidence: dict[str, object]) -> None:
    if (
        evidence["pumps_joined"] is not True
        or evidence["capture_complete"] is not True
        or evidence["terminated_by_runner"] is not False
    ):
        raise EngineBenchError("Native process did not quit/drain/join naturally")


def _audit_capture(evidence: dict[str, object], expected_bestmoves: int) -> None:
    lines = cast(str, evidence["stdout_text"]).splitlines()
    if sum(line.startswith("bestmove ") for line in lines) != expected_bestmoves or any(
        _explicit_error(line) for line in lines
    ):
        raise EngineBenchError("Complete native capture has duplicate/late bestmove or error")


def _quit(process: UciProcess, timeout: float) -> None:
    process.timeout = timeout
    process.send("quit")
    process.close_stdin()
    if process.wait_exit() != 0:
        raise EngineBenchError("Native process exited nonzero after quit")


def _search(
    process: UciProcess, config: Config, case: Case, legal: tuple[str, ...], role: str
) -> dict[str, object]:
    process.send("ucinewgame")
    _ready(process, config.startup)
    process.send(case.command())
    _ready(process, config.startup)
    process.timeout = config.search
    deadline = time.monotonic() + config.search
    start = time.perf_counter_ns()
    process.send("go nodes 800")
    nodes: int | None = None
    info: list[str] = []
    while True:
        line, received = _line(process, deadline)
        if line.startswith("bestmove "):
            end = received
            fields = line.split(" ")
            if (
                len(fields) not in (2, 4)
                or fields[1] not in legal
                or (
                    len(fields) == 4
                    and (fields[2] != "ponder" or MOVE.fullmatch(fields[3]) is None)
                )
            ):
                raise EngineBenchError("Nonterminal bestmove is malformed or illegal")
            break
        if line.startswith("info string "):
            info.append(line)
            continue
        if not line.startswith("info "):
            raise EngineBenchError("Unexpected response during measured search")
        tokens = line.split()
        indices = [i for i, value in enumerate(tokens) if value == "nodes"]
        if len(indices) != 1 or indices[0] + 1 == len(tokens):
            raise EngineBenchError("Every structured info must report exactly one nodes field")
        current = _integer(tokens[indices[0] + 1])
        if nodes is not None and current < nodes:
            raise EngineBenchError("Native cumulative node count decreased")
        if "multipv" in tokens and (
            tokens.count("multipv") != 1
            or tokens.index("multipv") + 1 == len(tokens)
            or tokens[tokens.index("multipv") + 1] != "1"
        ):
            raise EngineBenchError("Only the declared single PV may report node totals")
        nodes = current
        info.append(line)
    if nodes is None or (nodes != 800 if role == "mouse" else nodes < 800):
        raise EngineBenchError("Final actual node count violates declared fixed-node method")
    if end <= start:
        raise EngineBenchError("Common monotonic interval must be positive")
    _ready(process, config.startup)
    return {
        "id": case.identifier,
        "position": case.command(),
        "requested_nodes": 800,
        "actual_nodes": nodes,
        "overshoot_nodes": nodes - 800,
        "start_ns": start,
        "bestmove_received_ns": end,
        "elapsed_ns": end - start,
        "bestmove": fields[1],
        "bestmove_line": line,
        "info": info,
    }


def _run(
    config: Config,
    role: str,
    repetition: int,
    legal: dict[str, tuple[str, ...]],
    report: dict[str, object],
) -> None:
    child = config.engines[role]
    record: dict[str, object] = {
        "engine": role,
        "repetition": repetition,
        "warmup": [],
        "measured": [],
    }
    cast(list[object], report["runs"]).append(record)
    process = UciProcess(
        child.argv, timeout=config.startup, cwd=child.cwd, stdout_limit=8 * 1024 * 1024
    )
    try:
        with process:
            record["handshake"] = _handshake(process, child, config.startup)
            for phase in ("warmup", "measured"):
                for case in config.cases:
                    result = _search(process, config, case, legal[case.identifier], role)
                    cast(list[object], record[phase]).append(result)
            _quit(process, config.quit)
    finally:
        if hasattr(process, "process"):
            record["process"] = process.evidence()
    evidence = cast(dict[str, object], record["process"])
    _check_exit(evidence)
    _audit_capture(evidence, 32)
    measured = cast(list[dict[str, object]], record["measured"])
    total_nodes = sum(cast(int, x["actual_nodes"]) for x in measured)
    total_ns = sum(cast(int, x["elapsed_ns"]) for x in measured)
    record.update(
        {
            "actual_nodes": total_nodes,
            "elapsed_ns": total_ns,
            "nps": total_nodes * 1_000_000_000 / total_ns,
        }
    )


def _publish(output: Path, report: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # Atomic no-clobber, including a concurrent creator.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _audit(config: Config, report: dict[str, object], phase: str) -> None:
    entries: list[dict[str, object]] = []
    for item in config.files:
        try:
            actual = _sha(item.path)
            entries.append(
                {
                    "path": str(item.path),
                    "expected_sha256": item.sha256,
                    "actual_sha256": actual,
                    "matched": actual == item.sha256,
                }
            )
        except OSError as error:
            entries.append(
                {
                    "path": str(item.path),
                    "expected_sha256": item.sha256,
                    "actual_sha256": None,
                    "matched": False,
                    "error": str(error),
                }
            )
    cast(list[object], report["hash_audits"]).append({"phase": phase, "files": entries})
    if any(row["matched"] is not True for row in entries):
        raise EngineBenchError("Immutable hash changed during " + phase)


def run_benchmark(config_path: Path, output: Path) -> dict[str, object]:
    if not output.is_absolute() or output.exists():
        raise EngineBenchError("Report must be a new absolute pathname")
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "schema": "MouseEngineBenchReport1",
        "evidence_valid": False,
        "planned_searches": 192,
        "planned_measured_searches": 96,
        "runs": [],
        "method": (
            "driver monotonic perf_counter_ns before go send through complete bestmove receipt; "
            "full fixed16; three median rates; no retries"
        ),
        "backend_gate_claim": False,
    }
    config: Config | None = None
    report["hash_audits"] = []
    try:
        clock = time.get_clock_info("perf_counter")
        if (
            not clock.monotonic
            or clock.adjustable
            or not math.isfinite(clock.resolution)
            or clock.resolution <= 0
        ):
            raise EngineBenchError("Measurement clock must be monotonic and nonadjustable")
        report["clock"] = {
            "api": "time.perf_counter_ns",
            "implementation": clock.implementation,
            "resolution_seconds": clock.resolution,
            "monotonic": clock.monotonic,
            "adjustable": clock.adjustable,
        }
        config = load_config(config_path)
        if output.resolve() in {item.path for item in config.files}:
            raise EngineBenchError("Report collides with immutable input")
        report["config"] = config.declaration
        report["immutable_files"] = [
            {"path": str(x.path), "sha256": x.sha256, "role": x.role} for x in config.files
        ]
        _audit(config, report, "before_oracle")
        legal = _oracle(config, report)
        _audit(config, report, "after_oracle")
        for index, role in enumerate(ORDER):
            _audit(config, report, f"before_process_{index + 1}")
            _run(config, role, index // 2 + 1, legal, report)
            _audit(config, report, f"after_process_{index + 1}")
        runs = cast(list[dict[str, object]], report["runs"])
        completed = sum(len(cast(list[object], r[p])) for r in runs for p in ("warmup", "measured"))
        measured = sum(len(cast(list[object], r["measured"])) for r in runs)
        if len(runs) != 6 or completed != 192 or measured != 96:
            raise EngineBenchError("Incomplete fixed benchmark schedule")
        medians = {
            role: statistics.median(cast(float, r["nps"]) for r in runs if r["engine"] == role)
            for role in ("mouse", "lc0")
        }
        ratio = medians["mouse"] / medians["lc0"]
        report.update(
            {
                "median_nps": medians,
                "ratio": ratio,
                "engine_90_percent_met": ratio >= 0.9,
                "completed_searches": completed,
                "measured_searches": measured,
                "evidence_valid": True,
            }
        )
        _audit(config, report, "before_publication")
    except BaseException as error:
        report["evidence_valid"] = False
        for key in ("ratio", "engine_90_percent_met", "median_nps"):
            report.pop(key, None)
        report["error"] = str(error)
        if config is not None:
            try:
                _audit(config, report, "after_failure")
            except EngineBenchError as audit_error:
                report["post_failure_audit_error"] = str(audit_error)
        partial = cast(list[dict[str, object]], report["runs"])
        report["completed_searches"] = sum(
            len(cast(list[object], r[p])) for r in partial for p in ("warmup", "measured")
        )
        report["measured_searches"] = sum(len(cast(list[object], r["measured"])) for r in partial)
        _publish(output, report)
        if isinstance(error, KeyboardInterrupt | SystemExit):
            raise
        raise EngineBenchError(str(error)) from error
    _publish(output, report)
    return report


def _isolated() -> bool:
    return bool(sys.flags.isolated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if not _isolated():
            raise EngineBenchError(
                "Run with isolated Python: python -I -m mouse_tools.engine_bench"
            )
        report = run_benchmark(args.config, args.output)
        print(
            json.dumps(
                {
                    "evidence_valid": True,
                    "ratio": report["ratio"],
                    "engine_90_percent_met": report["engine_90_percent_met"],
                },
                allow_nan=False,
            )
        )
        return 0
    except (EngineBenchError, UciGateError, OSError, ValueError, RecursionError) as error:
        print(f"Engine benchmark failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
