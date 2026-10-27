# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the complete v0.1 perft and million-game process-sharded fuzz gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mouse_tools.perft_reference import (
    Case,
    ReferenceError,
    candidate_cases,
    parse_epd,
    position_key,
)
from mouse_tools.uci_gate import creation_flags

# Keep injected polling interrupts local; subprocess cleanup shares time.sleep.
_sleep = time.sleep

FUZZ_GAMES = 1_000_000
MAX_WORKERS = 16

# Approved fixture contract: tests/data/perft.epd, SHA-256 910feba4...dfa62c6.
# Candidate positions/depths come from perft_reference; these pin its 40 playouts.
APPROVED_FACTS_SHA256 = "a88ad185c25e9abca7026a4e35fd088d17fc4694be4baf8e6861412021380d5f"
APPROVED_PLAYOUT_DEPTHS: tuple[tuple[str, int], ...] = (
    ("5c468c494296a98d0a837ea0ed2e049af67c58a90423cf58fb8771adcde342a1", 3),
    ("802cd34841234c3d40b3ebbe8177000f26a0784187cb78fc227f7f17640df716", 3),
    ("4a778e2b055177413e2aafc0e40c7134280727fac82bdfd330f04fcb1d29b830", 3),
    ("2095aeb2b38e1cec02bebbea9cd1ce82f7674044f9542264472dbb49183a469c", 3),
    ("b9779babdd3b3640a3dd67825742c62474593aae989342d5fd66cc0cabf11e4a", 3),
    ("e481405c4964e9dc3308a8fc3c3e675e65b5175f44baebcec124119949855a38", 3),
    ("7ab55c2cb818c6798b3677fdffe92b65f213fce688f42c2f6d5a9c5c2b60ac11", 3),
    ("f78c36359fb8d7917db9b51500f73d426f3f58dc234f2f810c94c760143c9aad", 3),
    ("70f21363c1618b7f0c6e707129584fd9ab82d24ff893fc074bf85f6ee9cccc62", 3),
    ("1906fc6d13e0844e7e63c6b5030792497f93af8f5ca365758444afb8be4f4483", 3),
    ("cb76fd1c9aad6b8a8b9ab9c2eac76dfb7cca9c42ff1f910142e33ba4c3e50c4a", 3),
    ("7ddc9afda57309d81d5dd062a4d5c502c87f591e98608bc6c12284b6f97f9b07", 3),
    ("c7ffc52c652a6b603fb22f52b6c3d68fb4956ee68e870b63a347386721838705", 3),
    ("63b841acced87b13f1d5877e76093b1b4c71cf339d50014b92315ccba68ec474", 3),
    ("6745ab83672940ace98f3c5f16728cf20b51d9e648d5a4b9ac54aa2eeee6d3e4", 3),
    ("233c3059365f47b3aea4b42b07dca6b75bbba10ac644cd8521b55ec9deb7fb21", 3),
    ("76aa91ff0168001d197ce40a21d66c738c9e4f4d8c9170b73dbbb9811e4cda49", 3),
    ("bcb6a6583dd0c9b4b503633ec39f2df851857454cffbf46683038a185701205e", 3),
    ("e4ea647f269205bc463ecec8608e5da6af2cf0cc4657ecd9ffd0c202de40a39e", 3),
    ("b4ebe7d79e62d726e02f71852a33f68a13976622c9177baccb4a59a697c4fbd0", 3),
    ("862147c51e290545e8e0310019eade49fb1fdc5633ce6aa8eb7ac513fa02d69c", 3),
    ("0d7a74a9e39679da23404d47ae215c58ad644fd120ea2f6f7de3bb9444e87952", 3),
    ("a1827de1e3f2d8e24bff59349bd9fb02576070e2152c07c7db2f0b883af3c886", 3),
    ("8131526be604985c69a33831aa5b68fd42a0a1ddf2f6c0d2dffc3ee568ee1a53", 3),
    ("7c31e3ec0b9a3c11100253629d7d2f03bb9d614686bf49ae25ea2c1f229ca958", 3),
    ("7c640147473162b734814f43cddfa1ba012cf05b9d517cfd31b222f80b1607dc", 3),
    ("15a4099a6ac23cc8159bf46ef934d580499e5bc80988a34b65ff72c84b9bf282", 3),
    ("972f93fbf877bb7ff78dd062216be54af190d55b4e9ad7a34cebae4361fa5487", 3),
    ("4254ee8427820e62ed9d0a569670ab3bd3546e4eaafde7ad073516190eed3471", 3),
    ("8042f10ffdfcc46a543b255c0783d9336aebd75388f8a47661ef2d46f3329cd8", 3),
    ("561a628945d2ace03982f5c3d9212d9c98403ba2d912287de36549e9d99f8877", 3),
    ("ac4e5ba37752222f97a919be1e6b953766ab08513a6ad6c8faebb0815135fae5", 3),
    ("656ab32d69c9c95de02ce9a31c177bdcaa2ee78599b36c4a66b58e02e8e53f8c", 3),
    ("c5c342908d3edd1dc572f43bed7cf5cc0de9fd4a7242f23cb3c5af02dd025384", 3),
    ("86b32724b7e1e240c0630493f6b157e6b9666437498277727a76f577734d06ff", 3),
    ("5f9b076960b58b9c66220977b7dbb54ce1dfe62ac77d39f1233039730dd7fb19", 3),
    ("b7852af95aaa1aaf37b7e573ab02d4f52854840c3ed65e67588004fe8ebb7be8", 3),
    ("6ac2cb37162b3d2cef1817e74b025ee050f3894bded18ee2d5191dbfa3831e2b", 3),
    ("a3ee2976dedb39716eac9f77d5eb48f66133740ac05fa3620e8fcf034d1f9776", 3),
    ("1d7a660d81cfd8809be24095adfd9b79a8d125022fb5b9420a51bb1c65043908", 3),
)


class GateError(RuntimeError):
    """A full gate cannot be claimed from these inputs or results."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    stdout: str
    stderr: str
    exit_code: int | None
    elapsed_seconds: float
    timed_out: bool = False
    interrupted: bool = False


@dataclass
class PerftCheck:
    identifier: str
    variant: str
    depth: int
    expected: int
    result: CommandResult
    actual: int | None = None
    status: str = "failed"


@dataclass
class PerftReport:
    engine: str
    suite: str
    schema: int = 1
    gate: str = "v0.1-perft"
    suite_contract: str = "mouse-v0.1-approved-392-v1"
    baseline_facts_sha256: str = APPROVED_FACTS_SHA256
    status: str = "running"
    engine_sha256: str | None = None
    suite_sha256: str | None = None
    engine_version: CommandResult | None = None
    positions: int = 0
    planned_checks: int = 0
    completed_checks: int = 0
    checks: list[PerftCheck] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None


def _creation_flags() -> int:
    return creation_flags()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _valid_timeout(timeout_seconds: float) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise GateError("timeout must be finite and positive")


def run_command(command: Sequence[str], timeout_seconds: float) -> CommandResult:
    """Own, terminate, and reap one command; retain both diagnostic streams."""
    _valid_timeout(timeout_seconds)
    started = time.monotonic()
    timed_out = False
    interrupted = False
    with subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            interrupted = isinstance(error, KeyboardInterrupt)
            process.kill()
            stdout, stderr = process.communicate()
        return CommandResult(
            list(command),
            stdout,
            stderr,
            process.returncode,
            time.monotonic() - started,
            timed_out,
            interrupted,
        )


def _check_command(result: CommandResult, context: str) -> None:
    if result.interrupted:
        raise KeyboardInterrupt
    if result.timed_out:
        raise GateError(f"{context} timed out after {result.elapsed_seconds:.3f}s")
    if result.exit_code != 0:
        raise GateError(f"{context} exit code {result.exit_code}; stderr: {result.stderr.strip()}")


def _position_digest(fen: str) -> str:
    return hashlib.sha256(position_key(fen).encode("ascii")).hexdigest()


def _validate_approved_baseline(cases: Sequence[Case]) -> None:
    """Protect existing facts; extra positions and depths still run in full."""
    requirements = {
        _position_digest(case.fen): (case.variant, case.maximum_depth) for case in candidate_cases()
    }
    requirements.update((key, ("standard", depth)) for key, depth in APPROVED_PLAYOUT_DEPTHS)
    positions = {_position_digest(case.fen): case for case in cases}
    facts: list[tuple[str, str, int, int]] = []
    for key, (variant, maximum_depth) in requirements.items():
        case = positions.get(key)
        if case is None or case.variant != variant:
            raise GateError(f"Missing or changed approved baseline position: {key}")
        counts = dict(case.counts)
        for depth in range(1, maximum_depth + 1):
            if depth not in counts:
                raise GateError(f"Missing approved baseline depth {case.identifier} D{depth}")
            facts.append((position_key(case.fen), case.variant, depth, counts[depth]))
    digest = hashlib.sha256(
        json.dumps(sorted(facts), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if digest != APPROVED_FACTS_SHA256:
        raise GateError(
            f"Approved baseline facts changed: SHA-256 {digest} != {APPROVED_FACTS_SHA256}"
        )


def validate_suite(cases: Sequence[Case]) -> None:
    """No depth cap or subset switch can turn a shortened suite into a gate."""
    positions = {position_key(case.fen): case for case in cases}
    if len(positions) != len(cases) or len(positions) < 100:
        raise GateError("Full perft gate requires at least 100 distinct positions")
    for mandatory in candidate_cases():
        if not mandatory.identifier.startswith("spec_"):
            continue
        actual = positions.get(position_key(mandatory.fen))
        if actual is None or actual.variant != "standard":
            raise GateError(f"Missing mandatory spec position: {mandatory.identifier}")
        for depth, nodes in mandatory.known:
            if dict(actual.counts).get(depth) != nodes:
                raise GateError(
                    f"Missing or altered mandatory count {mandatory.identifier} D{depth}={nodes}"
                )
    if not any(case.variant == "chess960" for case in cases):
        raise GateError("Full perft gate requires Chess960 positions")
    _validate_approved_baseline(cases)


def _parse_nodes(stdout: str) -> int:
    lines = [line for line in stdout.splitlines() if line.startswith("nodes")]
    if len(lines) != 1:
        raise GateError("Expected exactly one nodes result line")
    match = re.fullmatch(
        r"nodes ([0-9]{1,20}) time_ms ([0-9]+(?:\.[0-9]+)?) nps ([0-9]{1,20})", lines[0]
    )
    if match is None:
        raise GateError(f"Malformed nodes result: {lines[0]}")
    return int(match[1])


def run_perft(engine: Path, suite: Path, output: Path, timeout_seconds: float = 120.0) -> int:
    engine = engine.resolve()
    suite = suite.resolve()
    output = output.resolve()
    report = PerftReport(str(engine), str(suite))
    started = time.monotonic()
    try:
        _valid_timeout(timeout_seconds)
        if output in (engine, suite):
            raise GateError("Evidence output must differ from the engine and suite")
        report.engine_sha256 = _sha256(engine)
        data = suite.read_bytes()
        report.suite_sha256 = hashlib.sha256(data).hexdigest()
        cases = parse_epd(data.decode("utf-8"))
        report.positions = len(cases)
        report.planned_checks = sum(len(case.counts) for case in cases)
        validate_suite(cases)
        report.engine_version = run_command([str(engine), "--version"], timeout_seconds)
        _check_command(report.engine_version, "engine --version")
        if not report.engine_version.stdout.strip().startswith("Mouse "):
            raise GateError("engine --version did not identify Mouse")
        _write_json(output, asdict(report))
        for case in cases:
            for depth, expected in case.counts:
                command = [str(engine), "perft", str(depth), "--fen", case.fen]
                if case.variant == "chess960":
                    command.append("--chess960")
                result = run_command(command, timeout_seconds)
                check = PerftCheck(case.identifier, case.variant, depth, expected, result)
                report.checks.append(check)
                _check_command(result, f"{case.identifier} D{depth}")
                check.actual = _parse_nodes(result.stdout)
                if check.actual != expected:
                    raise GateError(
                        f"Perft mismatch {case.identifier} D{depth}: {check.actual} != {expected}"
                    )
                check.status = "passed"
                report.completed_checks += 1
                report.elapsed_seconds = time.monotonic() - started
                _write_json(output, asdict(report))
        if _sha256(engine) != report.engine_sha256 or _sha256(suite) != report.suite_sha256:
            raise GateError("Engine or suite changed during the gate")
        report.status = "passed"
    except KeyboardInterrupt:
        report.status = "interrupted"
        report.error = "Perft gate interrupted; partial results are not a pass"
    except (GateError, ReferenceError, OSError, UnicodeError) as error:
        report.status = "failed"
        report.error = str(error)
    report.elapsed_seconds = time.monotonic() - started
    # An invalid destination must never overwrite an input while recording its error.
    if output not in (engine, suite):
        _write_json(output, asdict(report))
    if report.status != "passed":
        print(f"error: {report.error}", file=sys.stderr)
        return 130 if report.status == "interrupted" else 1
    print(f"Verified {report.positions} positions, {report.completed_checks} depth checks")
    return 0


@dataclass
class FuzzShard:
    index: int
    workers: int
    begin: int
    end: int
    command: list[str]
    stdout_log: str
    stderr_log: str
    catch_log: str
    status: str = "running"
    exit_code: int | None = None
    elapsed_seconds: float = 0.0
    completed: int = 0
    reported: dict[str, int] | None = None
    error: str | None = None


@dataclass
class FuzzReport:
    tests: str
    workers: int
    schema: int = 1
    gate: str = "v0.1-fuzz"
    status: str = "running"
    required_games: int = FUZZ_GAMES
    completed_games: int = 0
    tests_sha256: str | None = None
    shards: list[FuzzShard] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class OwnedShard:
    process: subprocess.Popen[bytes]
    record: FuzzShard
    started: float


def start_fuzz_shard(
    command: Sequence[str], environment: dict[str, str], stdout: Path, stderr: Path
) -> subprocess.Popen[bytes]:
    """Start only our child; file streams avoid pipe backpressure in long runs."""
    with stdout.open("wb") as output, stderr.open("wb") as diagnostic:
        return subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=diagnostic,
            env=environment,
            creationflags=_creation_flags(),
        )


def _record_marker(shard: FuzzShard) -> None:
    markers: list[str] = []
    with Path(shard.stdout_log).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("MOUSE_FUZZ_RESULT"):
                markers.append(line.rstrip("\r\n"))
                if len(markers) > 1:
                    raise GateError(f"Shard {shard.index} returned duplicate result markers")
    if not markers:
        raise GateError(f"Shard {shard.index} is missing MOUSE_FUZZ_RESULT")
    match = re.fullmatch(
        r"MOUSE_FUZZ_RESULT total=([0-9]{1,20}) shard=([0-9]{1,20}) shards=([0-9]{1,20}) "
        r"begin=([0-9]{1,20}) end=([0-9]{1,20}) completed=([0-9]{1,20}) plies=([0-9]{1,20})",
        markers[0],
    )
    if match is None:
        raise GateError(f"Shard {shard.index} returned a malformed result marker")
    shard.reported = dict(
        zip(
            ("total", "shard", "shards", "begin", "end", "completed", "plies"),
            (int(value) for value in match.groups()),
            strict=True,
        )
    )


def _validate_shard(shard: FuzzShard) -> None:
    marker_error: GateError | None = None
    try:
        _record_marker(shard)
    except GateError as error:
        marker_error = error
    if shard.exit_code != 0:
        diagnostic = Path(shard.stderr_log).read_text(encoding="utf-8", errors="replace")[-4096:]
        raise GateError(f"Shard {shard.index} exit code {shard.exit_code}; stderr: {diagnostic}")
    if marker_error is not None:
        raise marker_error
    result = shard.reported
    if result is None:
        raise GateError(f"Shard {shard.index} has no parsed result")
    expected = {
        "total": FUZZ_GAMES,
        "shard": shard.index,
        "shards": shard.workers,
        "begin": shard.begin,
        "end": shard.end,
        "completed": shard.end - shard.begin,
    }
    for name, value in expected.items():
        if result[name] != value:
            raise GateError(
                f"Shard {shard.index} {name} mismatch: {result[name]} != {value}; "
                "partial results are not a gate pass"
            )
    if result["plies"] < result["completed"]:
        raise GateError(f"Shard {shard.index} has fewer plies than complete games")
    shard.completed = result["completed"]
    shard.status = "passed"


def _stop_owned(owned: Sequence[OwnedShard]) -> None:
    """Terminate/reap every owned child before returning from failure or Ctrl+C."""
    for child in owned:
        if child.process.poll() is None:
            child.process.terminate()
        if child.record.status == "running":
            child.record.status = "cancelled"
    for child in owned:
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.process.kill()
            child.process.wait(timeout=5)
        child.record.exit_code = child.process.returncode
        if child.record.status != "passed":
            child.record.elapsed_seconds = time.monotonic() - child.started
            # Original gate failure stays primary; raw logs are retained.
            with suppress(GateError, OSError):
                _record_marker(child.record)


def run_fuzz(tests: Path, workers: int, output_dir: Path, timeout_seconds: float = 86400.0) -> int:
    tests = tests.resolve()
    output_dir = output_dir.resolve()
    report = FuzzReport(str(tests), workers)
    owned: list[OwnedShard] = []
    started = time.monotonic()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"error: create a new evidence directory: {error}", file=sys.stderr)
        return 1
    output = output_dir / "result.json"
    try:
        _valid_timeout(timeout_seconds)
        if not 1 <= workers <= MAX_WORKERS:
            raise GateError(f"workers must be between 1 and {MAX_WORKERS}")
        report.tests_sha256 = _sha256(tests)
        _write_json(output, asdict(report))
        for index in range(workers):
            stdout = output_dir / f"shard-{index:02}.stdout.log"
            stderr = output_dir / f"shard-{index:02}.stderr.log"
            catch = output_dir / f"shard-{index:02}.catch.log"
            command = [str(tests), "[property-long]", "--reporter", "compact", "--out", str(catch)]
            shard = FuzzShard(
                index,
                workers,
                FUZZ_GAMES * index // workers,
                FUZZ_GAMES * (index + 1) // workers,
                command,
                str(stdout),
                str(stderr),
                str(catch),
            )
            environment = dict(os.environ)
            environment["MOUSE_FUZZ_SHARD_INDEX"] = str(index)
            environment["MOUSE_FUZZ_SHARD_COUNT"] = str(workers)
            report.shards.append(shard)
            child_started = time.monotonic()
            try:
                process = start_fuzz_shard(command, environment, stdout, stderr)
            except OSError as error:
                shard.status = "failed"
                shard.error = str(error)
                raise
            owned.append(OwnedShard(process, shard, child_started))
            _write_json(output, asdict(report))
        while any(child.record.status == "running" for child in owned):
            for child in owned:
                if child.record.status != "running":
                    continue
                shard = child.record
                shard.elapsed_seconds = time.monotonic() - child.started
                shard.exit_code = child.process.poll()
                if shard.exit_code is None:
                    if shard.elapsed_seconds >= timeout_seconds:
                        shard.status = "failed"
                        shard.error = f"Shard {shard.index} timed out after {timeout_seconds}s"
                        raise GateError(shard.error)
                    continue
                try:
                    _validate_shard(shard)
                except GateError as error:
                    shard.status = "failed"
                    shard.error = str(error)
                    raise
                report.completed_games += shard.completed
                _write_json(output, asdict(report))
            if any(child.record.status == "running" for child in owned):
                _sleep(0.1)
        if report.completed_games != FUZZ_GAMES:
            raise GateError(f"Full fuzz gate requires {FUZZ_GAMES} complete games")
        if _sha256(tests) != report.tests_sha256:
            raise GateError("Test executable changed during the gate")
        report.status = "passed"
    except KeyboardInterrupt:
        report.status = "interrupted"
        report.error = "Fuzz gate interrupted; partial results are not a pass"
    except (GateError, OSError) as error:
        report.status = "failed"
        report.error = str(error)
    finally:
        _stop_owned(owned)
        report.elapsed_seconds = time.monotonic() - started
        _write_json(output, asdict(report))
    if report.status != "passed":
        print(f"error: {report.error}", file=sys.stderr)
        return 130 if report.status == "interrupted" else 1
    print(f"Verified {report.completed_games} complete games in {workers} process shards")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="gate", required=True)
    perft = subcommands.add_parser("perft", help="verify every stored perft depth")
    perft.add_argument("--engine", type=Path, required=True)
    perft.add_argument("--suite", type=Path, required=True)
    perft.add_argument("--output", type=Path, required=True)
    perft.add_argument("--timeout", type=float, default=120.0, help="seconds per engine command")
    fuzz = subcommands.add_parser("fuzz", help="verify the fixed million complete game gate")
    fuzz.add_argument("--tests", type=Path, required=True)
    fuzz.add_argument("--workers", type=int, required=True)
    fuzz.add_argument("--output-dir", type=Path, required=True, help="new evidence directory")
    fuzz.add_argument("--timeout", type=float, default=86400.0, help="seconds per shard")
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("gate files must be read with python -I")
    try:
        if arguments.gate == "perft":
            return run_perft(arguments.engine, arguments.suite, arguments.output, arguments.timeout)
        return run_fuzz(arguments.tests, arguments.workers, arguments.output_dir, arguments.timeout)
    except OSError as error:
        print(f"error: could not preserve gate evidence: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
