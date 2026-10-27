# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-gate orchestration and failure evidence contracts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from mouse_tools.perft_reference import Case, candidate_cases, parse_epd, position_key

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "tests/data/perft.epd"


def gate_module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.chess_gate") is not None, (
        "The full v0.1 gate runner must be implemented"
    )
    return importlib.import_module("mouse_tools.chess_gate")


def fixture_engine(tmp_path: Path) -> Path:
    binary = tmp_path / "fixture-engine"
    binary.write_bytes(b"test-only engine identity")
    return binary


def successful_perft_runner(commands: list[list[str]], cases: Sequence[Case]) -> object:
    facts = {(case.fen, depth): count for case in cases for depth, count in case.counts}
    gate = gate_module()

    def execute(command: Sequence[str], timeout_seconds: float) -> object:
        assert timeout_seconds > 0
        commands.append(list(command))
        if command[-1] == "--version":
            output = "Mouse test-fixture; git fixture; Release; TestCompiler; ORT fixture\n"
        else:
            count = facts[(command[4], int(command[2]))]
            output = f"nodes {count} time_ms 1.25 nps 100\n"
        return gate.CommandResult(list(command), output, "", 0, 0.01)

    return execute


def test_perft_runs_every_stored_depth_and_records_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    cases = parse_epd(SUITE.read_text(encoding="utf-8"))
    engine = fixture_engine(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(gate, "run_command", successful_perft_runner(commands, cases))
    output = tmp_path / "perft.json"
    assert gate.run_perft(engine, SUITE, output) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["positions"] == 126
    assert report["planned_checks"] == 392
    assert report["completed_checks"] == 392
    assert len(commands) == 393  # version plus every stored depth; no cap.
    assert len(report["engine_sha256"]) == 64
    assert len(report["suite_sha256"]) == 64
    assert report["engine_version"]["stdout"].startswith("Mouse test-fixture;")
    checks = report["checks"]
    assert all(item["status"] == "passed" for item in checks)
    assert any(item["depth"] == 6 for item in checks)
    frc_commands = [command for command in commands if "--chess960" in command]
    assert len(frc_commands) == sum(
        len(case.counts) for case in cases if case.variant == "chess960"
    )


@pytest.mark.parametrize("mutation", ["subset", "missing", "altered", "no-chess960"])
def test_perft_rejects_reduced_or_altered_gate_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    gate = gate_module()
    lines = SUITE.read_text(encoding="utf-8").splitlines()
    if mutation == "subset":
        lines = lines[:15]
    elif mutation == "missing":
        lines = [line for line in lines if 'id "spec_start"' not in line]
    elif mutation == "altered":
        lines = [line.replace("D6 119060324", "D6 119060323") for line in lines]
    else:
        lines = [line.replace("variant chess960", "variant standard") for line in lines]

    def forbidden_execution(command: Sequence[str], timeout_seconds: float) -> object:
        raise AssertionError("A reduced gate must fail before executing the engine")

    monkeypatch.setattr(gate, "run_command", forbidden_execution)
    suite = tmp_path / "altered.epd"
    suite.write_text("\n".join(lines), encoding="utf-8")
    output = tmp_path / "rejected.json"
    assert gate.run_perft(fixture_engine(tmp_path), suite, output) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["completed_checks"] == 0
    assert report["error"]


@pytest.mark.parametrize(
    ("stdout", "exit_code", "timed_out", "reason"),
    [
        ("", 0, False, "nodes"),
        ("nodes 1 time_ms 1 nps 1\n", 0, False, "mismatch"),
        ("nodes 20 time_ms 1 nps 1\nnodes 20 time_ms 1 nps 1\n", 0, False, "nodes"),
        ("", 7, False, "exit"),
        ("", -1, True, "timed out"),
    ],
)
def test_perft_failure_keeps_command_output_and_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    exit_code: int,
    timed_out: bool,
    reason: str,
) -> None:
    gate = gate_module()

    def execute(command: Sequence[str], timeout_seconds: float) -> object:
        if command[-1] == "--version":
            return gate.CommandResult(list(command), "Mouse fixture", "", 0, 0.01)
        return gate.CommandResult(
            list(command), stdout, "fixture diagnostic", exit_code, 0.02, timed_out
        )

    monkeypatch.setattr(gate, "run_command", execute)
    output = tmp_path / "failed.json"
    assert gate.run_perft(fixture_engine(tmp_path), SUITE, output) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert reason in report["error"]
    assert report["checks"][0]["result"]["stderr"] == "fixture diagnostic"
    assert report["checks"][0]["result"]["exit_code"] == exit_code


def test_missing_engine_produces_failed_evidence(tmp_path: Path) -> None:
    gate = gate_module()
    output = tmp_path / "missing.json"
    assert gate.run_perft(tmp_path / "missing.exe", SUITE, output) == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_command_timeout_kills_child_and_preserves_output(tmp_path: Path) -> None:
    gate = gate_module()
    result = gate.run_command(
        [sys.executable, "-I", "-c", "import time; print('before', flush=True); time.sleep(30)"],
        0.5,
    )
    assert result.timed_out
    assert result.exit_code != 0
    assert "before" in result.stdout
    assert result.elapsed_seconds < 10


def test_main_requires_isolation_before_reading_fixture(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mouse_tools.chess_gate",
            "perft",
            "--engine",
            "absent",
            "--suite",
            "absent",
            "--output",
            str(tmp_path / "absent.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "python -I" in result.stderr
    assert not (tmp_path / "absent.json").exists()


def fuzz_fixture(tmp_path: Path, mode: str) -> Path:
    """A protocol child only: no product engine or reduced production gate."""
    script = tmp_path / "fuzz_protocol_fixture.py"
    script.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "i = int(os.environ['MOUSE_FUZZ_SHARD_INDEX'])\n"
        "n = int(os.environ['MOUSE_FUZZ_SHARD_COUNT'])\n"
        "total = 1000000\n"
        "begin, end = total * i // n, total * (i + 1) // n\n"
        f"mode = {mode!r}\n"
        "print('fixture stdout', flush=True)\n"
        "print('fixture stderr', file=sys.stderr, flush=True)\n"
        "Path(sys.argv[sys.argv.index('--out') + 1]).write_text('fixture Catch log')\n"
        "if mode == 'sleep' or (mode == 'fail-kills' and i > 0): time.sleep(30)\n"
        "if mode in ('exit', 'fail-kills') and i == 0: sys.exit(7)\n"
        "completed = end - begin - (1 if mode == 'partial' else 0)\n"
        "if mode == 'total': total = 999999\n"
        "if mode == 'range': begin += 1\n"
        "if mode == 'shard': i = 0\n"
        "marker = (f'MOUSE_FUZZ_RESULT total={total} shard={i} shards={n} '"
        "           f'begin={begin} end={end} completed={completed} plies={completed * 2}')\n"
        "if mode == 'malformed': marker = marker.replace('completed=', 'completed=bad')\n"
        "if mode == 'oversized': marker = marker.replace('total=1000000', 'total=' + '9' * 5000)\n"
        "if mode != 'missing': print(marker, flush=True)\n"
        "if mode == 'duplicate': print(marker, flush=True)\n",
        encoding="utf-8",
    )
    return script


def use_fuzz_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> list[subprocess.Popen[bytes]]:
    gate = gate_module()
    assert hasattr(gate, "run_fuzz"), "The process-sharded fuzz gate must be implemented"
    script = fuzz_fixture(tmp_path, mode)
    launch = gate.start_fuzz_shard
    children: list[subprocess.Popen[bytes]] = []

    def start(
        command: Sequence[str], environment: dict[str, str], stdout: Path, stderr: Path
    ) -> subprocess.Popen[bytes]:
        child: subprocess.Popen[bytes] = launch(
            [sys.executable, "-I", str(script), *command[1:]], environment, stdout, stderr
        )
        children.append(child)
        return child

    monkeypatch.setattr(gate, "start_fuzz_shard", start)
    return children


def test_fuzz_requires_exact_million_across_disjoint_process_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    children = use_fuzz_fixture(tmp_path, monkeypatch, "correct")
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 3, output) == 0
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["required_games"] == 1_000_000
    assert report["completed_games"] == 1_000_000
    assert [(item["begin"], item["end"]) for item in report["shards"]] == [
        (0, 333333),
        (333333, 666666),
        (666666, 1000000),
    ]
    assert all(item["status"] == "passed" for item in report["shards"])
    assert all(child.poll() == 0 for child in children)
    assert all("[property-long]" in item["command"] for item in report["shards"])
    assert all(
        Path(item["stderr_log"]).read_text().strip() == "fixture stderr"
        for item in report["shards"]
    )


@pytest.mark.parametrize(
    "mode",
    [
        "partial",
        "exit",
        "malformed",
        "range",
        "total",
        "shard",
        "duplicate",
        "missing",
        "oversized",
    ],
)
def test_fuzz_rejects_incomplete_faked_or_failed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    gate = gate_module()
    children = use_fuzz_fixture(tmp_path, monkeypatch, mode)
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 3, output) == 1
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"]
    assert report["completed_games"] < 1_000_000
    assert all(child.poll() is not None for child in children)
    assert all(Path(item["stdout_log"]).exists() for item in report["shards"])
    assert all(Path(item["stderr_log"]).exists() for item in report["shards"])


def test_fuzz_failure_terminates_and_reaps_other_owned_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    children = use_fuzz_fixture(tmp_path, monkeypatch, "fail-kills")
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 3, output) == 1
    assert len(children) == 3
    assert all(child.poll() is not None for child in children)
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert all(item["exit_code"] is not None for item in report["shards"])
    assert any(item["status"] == "cancelled" for item in report["shards"])


def test_fuzz_timeout_reaps_children_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    children = use_fuzz_fixture(tmp_path, monkeypatch, "sleep")
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 2, output, timeout_seconds=0.3) == 1
    assert all(child.poll() is not None for child in children)
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "timed out" in report["error"]
    assert report["completed_games"] == 0


def test_fuzz_interruption_reaps_children_and_retains_incomplete_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    children = use_fuzz_fixture(tmp_path, monkeypatch, "sleep")

    shared_sleep = time.sleep
    calls = 0

    def interrupt(seconds: float) -> None:
        nonlocal calls
        calls += 1
        assert seconds == 0.1
        assert time.sleep is shared_sleep
        # Exercise another real caller while the interrupt hook is installed.
        # POSIX subprocess timed waits also use the shared time module.
        time.sleep(0)
        probe = subprocess.run(
            [sys.executable, "-I", "-c", "import time;time.sleep(0.02);print('probe')"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert probe.returncode == 0 and probe.stdout == "probe\n"
        raise KeyboardInterrupt

    monkeypatch.setattr(gate, "_sleep", interrupt, raising=False)
    assert time.sleep is shared_sleep
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 2, output, timeout_seconds=0.3) == 130
    assert calls == 1
    assert time.sleep is shared_sleep
    time.sleep(0)
    assert len(children) == 2
    assert all(child.poll() is not None for child in children)
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["completed_games"] == 0
    assert report["required_games"] == 1_000_000
    assert all(item["exit_code"] is not None for item in report["shards"])
    assert all(item["status"] == "cancelled" for item in report["shards"])
    assert all(item["completed"] == 0 for item in report["shards"])


@pytest.mark.parametrize("workers", [0, 17])
def test_fuzz_rejects_worker_count_outside_fixed_partition_contract(
    tmp_path: Path, workers: int
) -> None:
    gate = gate_module()
    assert hasattr(gate, "run_fuzz"), "The process-sharded fuzz gate must be implemented"
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), workers, output) == 1
    assert json.loads((output / "result.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_perft_cannot_overwrite_its_input_when_recording_failure(tmp_path: Path) -> None:
    gate = gate_module()
    engine = fixture_engine(tmp_path)
    original = engine.read_bytes()
    assert gate.run_perft(engine, SUITE, engine) == 1
    assert engine.read_bytes() == original


def test_perft_interruption_preserves_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()

    def execute(command: Sequence[str], timeout_seconds: float) -> object:
        if command[-1] == "--version":
            return gate.CommandResult(list(command), "Mouse fixture", "", 0, 0.01)
        return gate.CommandResult(
            list(command), "partial stdout", "partial stderr", -1, 0.02, False, True
        )

    monkeypatch.setattr(gate, "run_command", execute)
    output = tmp_path / "interrupted.json"
    assert gate.run_perft(fixture_engine(tmp_path), SUITE, output) == 130
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["completed_checks"] == 0
    assert report["checks"][0]["result"]["stdout"] == "partial stdout"


def test_fuzz_will_not_overwrite_an_existing_evidence_directory(tmp_path: Path) -> None:
    gate = gate_module()
    output = tmp_path / "run"
    output.mkdir()
    evidence = output / "result.json"
    evidence.write_text("prior evidence", encoding="utf-8")
    assert gate.run_fuzz(fixture_engine(tmp_path), 1, output) == 1
    assert evidence.read_text(encoding="utf-8") == "prior evidence"


def test_fuzz_spawn_failure_reaps_already_launched_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    children = use_fuzz_fixture(tmp_path, monkeypatch, "sleep")
    fixture_launch = gate.start_fuzz_shard

    def fail_second(
        command: Sequence[str], environment: dict[str, str], stdout: Path, stderr: Path
    ) -> subprocess.Popen[bytes]:
        if environment["MOUSE_FUZZ_SHARD_INDEX"] == "1":
            raise OSError("fixture spawn failure")
        child: subprocess.Popen[bytes] = fixture_launch(command, environment, stdout, stderr)
        return child

    monkeypatch.setattr(gate, "start_fuzz_shard", fail_second)
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 3, output) == 1
    assert len(children) == 1
    assert children[0].poll() is not None
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "fixture spawn failure" in report["error"]
    assert report["shards"][1]["status"] == "failed"
    assert report["shards"][1]["index"] == 1
    assert "[property-long]" in report["shards"][1]["command"]


def test_fuzz_every_finished_child_has_a_final_status_after_another_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = gate_module()
    use_fuzz_fixture(tmp_path, monkeypatch, "exit")
    fixture_launch = gate.start_fuzz_shard

    def wait_before_return(
        command: Sequence[str], environment: dict[str, str], stdout: Path, stderr: Path
    ) -> subprocess.Popen[bytes]:
        child: subprocess.Popen[bytes] = fixture_launch(command, environment, stdout, stderr)
        child.wait(timeout=5)
        return child

    monkeypatch.setattr(gate, "start_fuzz_shard", wait_before_return)
    output = tmp_path / "run"
    assert gate.run_fuzz(fixture_engine(tmp_path), 3, output) == 1
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert all(item["status"] != "running" for item in report["shards"])


def test_atomic_evidence_write_does_not_clobber_an_existing_tmp_input(tmp_path: Path) -> None:
    gate = gate_module()
    output = tmp_path / "result.json"
    sentinel = tmp_path / "result.json.tmp"
    sentinel.write_bytes(b"existing unrelated input")
    gate._write_json(output, {"status": "running"})
    assert sentinel.read_bytes() == b"existing unrelated input"
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "running"


@pytest.mark.parametrize("field", ["depth", "count", "halfmove", "fullmove"])
def test_invalid_numeric_suite_preserves_failed_gate_evidence(tmp_path: Path, field: str) -> None:
    gate = gate_module()
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    token = "9" * 5000
    if field in ("halfmove", "fullmove"):
        fields = fen.split()
        fields[4 if field == "halfmove" else 5] = token
        fen = " ".join(fields)
    depth = token if field == "depth" else "1"
    count = token if field == "count" else "5"
    suite = tmp_path / "numeric.epd"
    suite.write_text(f'{fen} ; id "large"; variant standard; D{depth} {count};', encoding="utf-8")
    output = tmp_path / "failed-input.json"
    assert gate.run_perft(fixture_engine(tmp_path), suite, output) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["completed_checks"] == 0
    assert report["error"]
    assert report["suite_sha256"]


def test_approved_baseline_rejects_392_checks_shortened_to_151() -> None:
    gate = gate_module()
    original = parse_epd(SUITE.read_text(encoding="utf-8"))
    shortened = tuple(
        case
        if case.identifier.startswith("spec_")
        else Case(case.identifier, case.fen, case.variant, (case.counts[0],))
        for case in original
    )
    assert sum(len(case.counts) for case in shortened) == 151
    with pytest.raises(gate.GateError, match="baseline"):
        gate.validate_suite(shortened)


def test_approved_baseline_rejects_removal_of_a_generated_playout() -> None:
    gate = gate_module()
    original = parse_epd(SUITE.read_text(encoding="utf-8"))
    removed = tuple(case for case in original if case.identifier != "playout_001")
    assert len(removed) == 125
    with pytest.raises(gate.GateError, match="baseline"):
        gate.validate_suite(removed)


def test_approved_baseline_rejects_altering_a_nonmandatory_node_count() -> None:
    gate = gate_module()
    original = parse_epd(SUITE.read_text(encoding="utf-8"))
    case = original[0]
    altered = Case(
        case.identifier, case.fen, case.variant, ((1, case.counts[0][1] + 1), *case.counts[1:])
    )
    with pytest.raises(gate.GateError, match="baseline"):
        gate.validate_suite((altered, *original[1:]))


def test_package_manifest_reproduces_the_approved_392_fact_contract() -> None:
    gate = gate_module()
    assert hasattr(gate, "APPROVED_PLAYOUT_DEPTHS"), "Package must pin all 40 playout contracts"
    cases = parse_epd(SUITE.read_text(encoding="utf-8"))
    playouts = {
        hashlib.sha256(position_key(case.fen).encode("ascii")).hexdigest(): max(
            depth for depth, _ in case.counts
        )
        for case in cases
        if case.identifier.startswith("playout_")
    }
    assert len(gate.APPROVED_PLAYOUT_DEPTHS) == 40
    assert len(dict(gate.APPROVED_PLAYOUT_DEPTHS)) == 40
    assert dict(gate.APPROVED_PLAYOUT_DEPTHS) == playouts
    candidates = candidate_cases()
    assert len(candidates) == 86
    assert sum(case.maximum_depth for case in candidates) + sum(playouts.values()) == 392
    facts = sorted(
        (position_key(case.fen), case.variant, depth, nodes)
        for case in cases
        for depth, nodes in case.counts
    )
    digest = hashlib.sha256(json.dumps(facts, separators=(",", ":")).encode("ascii")).hexdigest()
    assert digest == gate.APPROVED_FACTS_SHA256
    assert digest == "a88ad185c25e9abca7026a4e35fd088d17fc4694be4baf8e6861412021380d5f"
    gate.validate_suite(cases)


def test_approved_baseline_allows_extensions_renamed_ids_and_reordered_clocks() -> None:
    gate = gate_module()
    original = parse_epd(SUITE.read_text(encoding="utf-8"))
    extended = []
    for index, case in enumerate(reversed(original)):
        fields = case.fen.split()
        fields[2] = "".join(reversed(fields[2]))
        fields[4:] = ["0", "1"]
        counts = case.counts + ((max(depth for depth, _ in case.counts) + 1, 123),)
        extended.append(Case(f"renamed_{index}", " ".join(fields), case.variant, counts))
    extra = Case("additional", "7k/8/8/8/8/8/P7/K7 w - - 0 1", "standard", ((1, 4),))
    assert position_key(extra.fen) not in {position_key(case.fen) for case in original}
    gate.validate_suite((*extended, extra))
