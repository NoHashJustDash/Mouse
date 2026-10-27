# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, CPU-only native legacy-TM elapsed-time capture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from mouse_tools.uci_gate import UciProcess

Case = tuple[int, int, int, int]
COLUMNS = (
    "clock_ms",
    "increment_ms",
    "movestogo",
    "lc0_internal_ply",
    "elapsed_ms",
    "uci_info_ms",
    "bestmove",
    "binary_sha256",
    "model_sha256",
)


def grid() -> tuple[Case, ...]:
    return tuple(
        (clock, inc, mgt, ply)
        for clock in (1000, 10000, 60000, 300000)
        for inc in (0, clock // 100, clock // 10)
        for mgt in (0, 40)
        for ply in (1, 20, 60, 120)
    )


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _until(process: UciProcess, marker: str) -> None:
    deadline = time.monotonic() + 300
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise ValueError("Readiness/handshake deadline exceeded")
        line = process.line(timeout=min(300, left))
        if line.startswith("info string error"):
            raise ValueError(line)
        if line == marker:
            return


def measure(binary: Path, model: Path, kind: str, case: Case) -> tuple[float, str, str]:
    clock, inc, mgt, ply = case
    command = [str(binary)]
    if kind == "lc0":
        command += [
            "--config=",
            "--backend=random",
            "--backend-opts=uniform=true,delay=1",
            "--threads=1",
            "--minibatch-size=1",
            "--nncache=0",
            "--move-overhead=0",
            "--smart-pruning-factor=0",
            "--weights=" + str(model),
        ]
    elif kind != "mouse":
        raise ValueError("Capture kind must be lc0 or mouse")
    with UciProcess(
        command, timeout=300, stdout_limit=8 * 1024 * 1024, stderr_limit=2 * 1024 * 1024
    ) as process:
        process.send("uci")
        _until(process, "uciok")
        if kind == "mouse":
            for name, value in (
                ("Backend", "cpu"),
                ("Precision", "fp32"),
                ("Threads", "1"),
                ("CpuThreads", "1"),
                ("MinibatchSize", "1"),
                ("NNCacheSize", "0"),
                ("RamLimitMb", "512"),
                ("MoveOverheadMs", "0"),
                ("SmartPruningFactor", "0"),
                ("WeightsFile", str(model)),
            ):
                process.send(f"setoption name {name} value {value}")
        process.send("ucinewgame")
        # Actual v0.32.1 FromFen uses fullmove directly as its internal game ply.
        process.send(f"position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 {ply}")
        process.send("isready")
        _until(process, "readyok")
        go = f"go wtime {clock} btime {clock} winc {inc} binc {inc}"
        if mgt:
            go += f" movestogo {mgt}"
        started = time.perf_counter()
        deadline = time.monotonic() + clock / 1000 + 300
        process.send(go)
        last_info = ""
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ValueError(f"Total move watchdog exceeded at {case}")
            line = process.line(timeout=min(300, left))
            if line.startswith("info string error"):
                raise ValueError(line)
            if line.startswith("info ") and not line.startswith("info string "):
                match = re.search(r"(?:^| )time ([0-9]+)(?: |$)", line)
                if match:
                    last_info = match[1]
            if line.startswith("bestmove "):
                elapsed = (time.perf_counter() - started) * 1000
                best = line.split()[1]
                if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", best):
                    raise ValueError("Missing legal-form bestmove in timing sample")
                break
        process.send("quit")
        if process.wait_exit() != 0:
            raise ValueError("Native timing child failed")
    evidence = process.evidence()
    if (
        not evidence["capture_complete"]
        or not evidence["pumps_joined"]
        or evidence["terminated_by_runner"]
    ):
        raise ValueError("Native timing transport did not complete and join")
    return elapsed, last_info, best


def capture(binary: Path, model: Path, kind: str, expected_sha: str, output: Path) -> None:
    binary, model = binary.resolve(strict=True), model.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or digest(binary) != expected_sha:
        raise ValueError("Native timing binary SHA256 mismatch")
    model_sha = digest(model)
    pending = output.with_suffix(output.suffix + ".pending")
    if output.exists():
        raise ValueError("Refusing to overwrite a timing table")
    with pending.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for index, case in enumerate(grid(), 1):
            elapsed, info, best = measure(binary, model, kind, case)
            writer.writerow((*case, f"{elapsed:.6f}", info, best, expected_sha, model_sha))
            stream.flush()
            print(f"{kind} {index}/96 {case} elapsed_ms={elapsed:.3f} native0 joined", flush=True)
    if digest(binary) != expected_sha or digest(model) != model_sha:
        raise ValueError("Timing binary/model changed during collection")
    # Atomic exclusive publication; an existing completed table is never replaced.
    os.link(pending, output)
    pending.unlink()


def _table(path: Path) -> dict[Case, float]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unexpected timing table schema")
        rows = list(reader)
    if len(rows) != 96:
        raise ValueError("Complete96-case elapsed table required")
    values: dict[Case, float] = {}
    identities = {(row["binary_sha256"], row["model_sha256"]) for row in rows}
    if len(identities) != 1 or any(
        not re.fullmatch(r"[0-9a-f]{64}", x) for x in next(iter(identities))
    ):
        raise ValueError("Changed or malformed timing identities")
    for row in rows:
        case = tuple(int(row[key]) for key in COLUMNS[:4])
        if len(case) != 4:
            raise ValueError("Malformed timing case")
        key = (case[0], case[1], case[2], case[3])
        value = float(row["elapsed_ms"])
        if key in values or not math.isfinite(value) or value <= 0:
            raise ValueError("Duplicate or invalid elapsed sample")
        values[key] = value
    if set(values) != set(grid()):
        raise ValueError("Timing grid does not match the planned96 cases")
    return values


def compare(reference: Path, mouse: Path, output: Path) -> int:
    expected, actual = _table(reference), _table(mouse)
    failures = 0
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (*COLUMNS[:4], "lc0_elapsed_ms", "mouse_elapsed_ms", "tolerance_ms", "passed")
        )
        for case in grid():
            tolerance = max(15.0, expected[case] * 0.1)
            passed = abs(actual[case] - expected[case]) <= tolerance
            failures += not passed
            writer.writerow((*case, expected[case], actual[case], tolerance, int(passed)))
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    if not sys.flags.isolated:
        raise ValueError("Run timing capture with python -I")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("--binary", type=Path, required=True)
    collect.add_argument("--model", type=Path, required=True)
    collect.add_argument("--kind", choices=("lc0", "mouse"), required=True)
    collect.add_argument("--sha256", required=True)
    collect.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("compare")
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--mouse", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "capture":
        capture(args.binary, args.model, args.kind, args.sha256, args.output)
        return 0
    failures = compare(args.reference, args.mouse, args.output)
    print(f"Legacy elapsed parity: {96 - failures}/96 PASS; tolerance=max15ms,10percent")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
