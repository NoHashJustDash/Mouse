# SPDX-License-Identifier: GPL-3.0-or-later
"""Real bounded child I/O plus strict protocol failures, without running Mouse."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest


def gate() -> Any:
    try:
        return importlib.import_module("mouse_tools.uci_gate")
    except ModuleNotFoundError:
        pytest.fail("UCI conformance runner missing")


def child(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


def test_real_child_preserves_exact_commands_stdout_stderr_and_exit() -> None:
    api = gate()
    code = (
        "import sys; data=sys.stdin.buffer.readline(); "
        "sys.stdout.buffer.write(data);sys.stdout.buffer.flush(); "
        "sys.stderr.buffer.write(b'diagnostic\\n');sys.stderr.buffer.flush()"
    )
    with api.UciProcess(child(code), timeout=2) as process:
        process.send("hello\x00world")
        assert process.line() == "hello\x00world"
        assert process.wait_exit() == 0
    evidence = process.evidence()
    assert evidence["stdin_text"] == "hello\x00world\n"
    assert evidence["stdout_text"] == "hello\x00world\n"
    assert evidence["stderr_text"] == "diagnostic\n"
    assert evidence["stdout_sha256"] == hashlib.sha256(b"hello\x00world\n").hexdigest()
    assert not any(thread.is_alive() for thread in process.threads)


def test_idle_child_deadline_kills_reaps_and_joins_all_pumps() -> None:
    api = gate()
    started = time.monotonic()
    with (
        pytest.raises(api.UciGateError, match="deadline"),
        api.UciProcess(child("import time;time.sleep(60)"), timeout=0.15) as process,
    ):
        process.line()
    assert time.monotonic() - started < 3
    assert process.process.poll() is not None
    assert not any(thread.is_alive() for thread in process.threads)


def test_stdin_blocked_child_is_still_cancelled_and_reaped() -> None:
    api = gate()
    with (
        pytest.raises(api.UciGateError, match="deadline"),
        api.UciProcess(child("import time;time.sleep(60)"), timeout=0.15) as process,
    ):
        process.send("x" * (1024 * 1024 + 1))
    assert process.process.poll() is not None
    assert not any(thread.is_alive() for thread in process.threads)


@pytest.mark.parametrize(
    ("code", "limits", "message"),
    [
        (
            "import sys;sys.stdout.buffer.write(b'x'*5000);sys.stdout.buffer.flush()",
            {"line_limit": 1024},
            "line",
        ),
        (
            "import sys;sys.stderr.buffer.write(b'x'*8192);sys.stderr.buffer.flush()",
            {"stderr_limit": 4096},
            "stderr",
        ),
        ("print('x\\n'*3000,flush=True)", {"queue_limit": 8}, "queue"),
        (
            "import sys;sys.stdout.buffer.write(b'bad\\xff\\n');sys.stdout.buffer.flush()",
            {},
            "UTF-8",
        ),
        ("print('unterminated',end='',flush=True)", {}, "unterminated"),
    ],
)
def test_malformed_or_unbounded_child_output_fails_without_thread_leaks(
    code: str, limits: dict[str, int], message: str
) -> None:
    api = gate()
    with (
        pytest.raises(api.UciGateError, match=message),
        api.UciProcess(child(code), timeout=2, **limits) as process,
    ):
        time.sleep(0.1)
        while True:
            process.line()
    assert process.process.poll() is not None
    assert not any(thread.is_alive() for thread in process.threads)


def test_keyboard_interrupt_reaps_owned_child() -> None:
    api = gate()
    with (
        pytest.raises(KeyboardInterrupt),
        api.UciProcess(child("import time;time.sleep(60)"), timeout=2) as process,
    ):
        raise KeyboardInterrupt
    assert process.process.poll() is not None
    assert not any(thread.is_alive() for thread in process.threads)


def test_process_tree_cleanup_prevents_inherited_pipe_orphans(tmp_path: Path) -> None:
    api = gate()
    marker = tmp_path / "orphan.txt"
    grandchild = (
        "import time;from pathlib import Path;time.sleep(0.7);"
        f"Path({str(marker)!r}).write_text('leaked');time.sleep(60)"
    )
    code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-I','-c',{grandchild!r}]);"
        "print('ready',flush=True);time.sleep(60)"
    )
    with api.UciProcess(child(code), timeout=2) as process:
        assert process.line() == "ready"
    assert not any(thread.is_alive() for thread in process.threads)
    time.sleep(0.8)
    assert not marker.exists()


INFO = (
    "info depth 2 seldepth 2 nodes 8 nps 100 time 80 hashfull 1 tbhits 0 "
    "multipv 1 score cp 20 wdl 510 200 290 pv e2e4 e7e5"
)


def test_strict_info_parser_preserves_volatile_and_deterministic_fields() -> None:
    parsed = gate().parse_info(INFO)
    assert parsed["nodes"] == 8
    assert parsed["score_cp"] == 20
    assert parsed["wdl"] == [510, 200, 290]
    assert parsed["pv"] == ["e2e4", "e7e5"]
    assert parsed["time"] == 80


@pytest.mark.parametrize(
    "line",
    [
        INFO + " nodes 9",
        INFO.replace("nodes 8", "nodes -1"),
        INFO.replace("nps 100", "nps nan"),
        INFO.replace("time 80", "time 1e309"),
        INFO.replace("tbhits 0", "tbhits 1"),
        INFO.replace("hashfull 1", "hashfull 1001"),
        INFO.replace("multipv 1", "multipv 0"),
        INFO.replace("510 200 290", "510 200 291"),
        INFO.replace("score cp 20", "score mate 1"),
        INFO.replace("e7e5", "e7e9"),
        INFO.replace("seldepth 2 ", ""),
        INFO.replace("depth 2", "depth 3", 1),
        INFO + " mystery 1",
    ],
)
def test_info_parser_refuses_missing_duplicate_nonfinite_and_wrong_fields(line: str) -> None:
    api = gate()
    with pytest.raises(api.UciGateError):
        api.parse_info(line)


@pytest.mark.parametrize(
    "line", ["bestmove", "bestmove a9a8", "bestmove e2e4 extra", "bestmove 0000 ponder e7e5"]
)
def test_bestmove_parser_refuses_malformed_output(line: str) -> None:
    api = gate()
    with pytest.raises(api.UciGateError):
        api.parse_bestmove(line)


def test_bestmove_parser_accepts_only_explicit_complete_transport() -> None:
    assert gate().parse_bestmove("bestmove e2e4") == "e2e4"
    assert gate().parse_bestmove("bestmove 0000") == "0000"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_process_deadlines_are_rejected_before_launch(timeout: float) -> None:
    api = gate()
    with pytest.raises(api.UciGateError):
        api.UciProcess(child("raise SystemExit(99)"), timeout=timeout)


def test_import_has_no_child_process_or_thread_side_effects() -> None:
    before = {thread.ident for thread in threading.enumerate()}
    gate()
    assert {thread.ident for thread in threading.enumerate()} == before


SPEC = Path(__file__).parents[2] / "tests/data/uci/protocol-v1"
BACKEND = Path(__file__).parents[2] / "tests/data/backend"


def config(tmp_path: Path, *, failure: bool = True) -> Any:
    api = gate()
    binary = tmp_path / "mouse.exe"
    binary.write_bytes(b"controlled unit engine identity")
    (tmp_path / "onnxruntime.dll").write_bytes(b"controlled unit runtime identity")
    weights = tmp_path / "valid.onnx"
    weights.write_bytes((BACKEND / "valid.onnx").read_bytes())
    broken = tmp_path / "search-nonfinite.onnx"
    broken.write_bytes((BACKEND / "search-nonfinite.onnx").read_bytes())
    return api.GateConfig(binary, weights, broken if failure else None, spec_dir=SPEC)


def test_approved_spec_keeps_all_21_cases_and_eight_stop_activation_attempts() -> None:
    api = gate()
    spec = api.load_spec(SPEC)
    assert len(spec.cases) == 21
    assert len({case.identifier for case in spec.cases}) == 21
    race = next(case for case in spec.cases if case.identifier == "stop_activation_race")
    assert sum(step.get("send") == "stop" for step in race.steps) == 8
    assert spec.cases[-1].identifier == "fatal_inference_idle"


def test_spec_edit_or_subset_is_not_an_accepted_golden(tmp_path: Path) -> None:
    api = gate()
    for source in SPEC.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    path = tmp_path / "expectations.json"
    path.write_text(path.read_text().replace('"id": "handshake_options"', '"id": "changed"'))
    with pytest.raises(api.UciGateError, match="golden"):
        api.load_spec(tmp_path)


@pytest.mark.parametrize("kind", ["engine", "weights", "runtime", "added_runtime"])
def test_artifact_snapshot_refuses_replacement_or_new_libraries(tmp_path: Path, kind: str) -> None:
    api = gate()
    cfg = config(tmp_path)
    snapshot = api.Artifacts.freeze(cfg)
    target = {
        "engine": cfg.engine,
        "weights": cfg.weights,
        "runtime": tmp_path / "onnxruntime.dll",
        "added_runtime": tmp_path / "new.dll",
    }[kind]
    target.write_bytes(b"mutated after first identity")
    with pytest.raises(api.UciGateError, match="changed"):
        snapshot.verify()


def test_large_or_unpinned_model_is_refused_before_any_child(tmp_path: Path) -> None:
    api = gate()
    cfg = config(tmp_path)
    cfg.weights.write_bytes(b"not a tiny approved CPU fixture")
    with pytest.raises(api.UciGateError, match="fixture"):
        api.Artifacts.freeze(cfg)


def test_full_runner_requires_all_rows_and_retains_frozen_not_replacement_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    before = hashlib.sha256(cfg.engine.read_bytes()).hexdigest()
    seen: list[str] = []

    def fake_case(configuration: Any, case: Any) -> dict[str, object]:
        seen.append(case.identifier)
        if len(seen) == 1:
            configuration.engine.write_bytes(b"replaced during inference")
        return {"id": case.identifier, "passed": True, "process": {"exit_code": 0}}

    monkeypatch.setattr(api, "run_case", fake_case)
    report = api.run_gate(cfg, tmp_path / "mutated.json")
    assert len(seen) == 21
    assert not report["passed"]
    assert not report["hashes_unchanged"]
    assert report["artifacts"][str(cfg.engine.resolve())] == before
    assert not any(row["passed"] for row in report["cases"])


def test_full_runner_does_not_stop_after_ordinary_case_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    seen: list[str] = []

    def fake_case(configuration: Any, case: Any) -> dict[str, object]:
        del configuration
        seen.append(case.identifier)
        return {
            "id": case.identifier,
            "passed": len(seen) != 1,
            "error": "test mismatch" if len(seen) == 1 else None,
        }

    monkeypatch.setattr(api, "run_case", fake_case)
    report = api.run_gate(cfg, tmp_path / "failed.json")
    assert len(seen) == 21
    assert not report["passed"]
    assert report["completed_cases"] == 21


def test_missing_explicit_failure_model_never_claims_complete_protocol_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path, failure=False)
    seen: list[str] = []

    def fake_case(configuration: Any, case: Any) -> dict[str, object]:
        del configuration
        seen.append(case.identifier)
        return {"id": case.identifier, "passed": True}

    monkeypatch.setattr(api, "run_case", fake_case)
    report = api.run_gate(cfg, tmp_path / "partial.json")
    assert len(seen) == 20
    assert not report["passed"]
    assert not report["coverage_complete"]
    assert report["planned_cases"] == 21
    assert report["attempted_cases"] == report["completed_cases"] == len(seen) == 20
    omitted = report["cases"][-1]
    assert omitted["id"] == "fatal_inference_idle"
    assert omitted["not_run"] is True
    assert not omitted["completed"]
    assert not omitted["passed"]
    assert not any(row.get("not_run") for row in report["cases"][:-1])


def test_atomic_publication_preserves_existing_success_and_launches_no_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    destination = tmp_path / "success.json"
    destination.write_text("original immutable success")
    seen: list[str] = []
    monkeypatch.setattr(api, "run_case", lambda *_: seen.append("must not run"))
    report = api.run_gate(cfg, destination)
    assert not seen
    assert not report["passed"]
    assert destination.read_text() == "original immutable success"
    assert list(tmp_path.glob("success.failed-*.json"))


def test_report_destination_cannot_replace_a_model(tmp_path: Path) -> None:
    api = gate()
    cfg = config(tmp_path)
    original = cfg.weights.read_bytes()
    with pytest.raises(api.UciGateError, match="input"):
        api.run_gate(cfg, cfg.weights)
    assert cfg.weights.read_bytes() == original


def test_interruption_is_incomplete_failed_evidence_with_no_false_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)

    def interrupted(*_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(api, "run_case", interrupted)
    report = api.run_gate(cfg, tmp_path / "interrupted.json")
    assert report["status"] == "interrupted"
    assert not report["passed"]
    assert report["completed_cases"] == 0


HANDSHAKE = [
    "id name Mouse 0.5.0-dev",
    "id author Zulfeqar Malik",
    "option name WeightsFile type string default <empty>",
    "option name Backend type combo default cpu var cpu var cuda var tensorrt",
    "option name Precision type combo default fp32 var fp32 var fp16",
    "option name Threads type spin default 1 min 1 max 128",
    "option name MinibatchSize type spin default 0 min 0 max 1024",
    "option name NNCacheSize type spin default 0 min 0 max 16000000",
    "option name CacheHistoryLength type spin default 0 min 0 max 7",
    "option name RamLimitMb type spin default 64 min 1 max 16384",
    "option name CPuct type string default 1.745",
    "option name CPuctBase type string default 38739",
    "option name CPuctFactor type string default 3.894",
    "option name FpuValue type string default 0.33",
    "option name FpuStrategyAtRoot type combo default same var same var reduction var absolute",
    "option name FpuValueAtRoot type string default 1",
    "option name PolicyTemperature type string default 1.359",
    "option name MultiPV type spin default 1 min 1 max 256",
    "option name UCI_ShowWDL type check default false",
    "option name UCI_Chess960 type check default false",
    "option name Ponder type check default false",
    "option name LogSearchStats type check default false",
    "option name MoveOverheadMs type spin default 0 min 0 max 60000",
    "option name SmartPruningFactor type string default 1.33",
    "option name SmartPruningMinimumBatches type spin default 0 min 0 max 10000",
    "option name GpuDllDirectories type string default <empty>",
    "option name CacheDirectory type string default <empty>",
    "option name Device type spin default 0 min 0 max 1023",
    "option name CpuThreads type spin default 1 min 1 max 128",
    "option name WorkspaceMb type spin default 16 min 1 max 16384",
    "option name Deterministic type check default true",
    "uciok",
]


def test_handshake_requires_exact_option_contract_and_original_identity() -> None:
    parsed = gate().validate_handshake(HANDSHAKE, {"Threads": "1", "CPuct": "1.745"})
    assert parsed["Backend"]["default"] == "cpu"
    assert parsed["CacheHistoryLength"] == {"type": "spin", "default": "0"}
    assert parsed["LogSearchStats"] == {"type": "check", "default": "false"}
    assert len(parsed) == 29
    assert parsed["FpuStrategyAtRoot"] == {"type": "combo", "default": "same"}
    assert parsed["FpuValueAtRoot"] == {"type": "string", "default": "1"}


@pytest.mark.parametrize(
    "change",
    [
        lambda rows: rows[:-2] + rows[-1:],
        lambda rows: rows[:3] + rows[2:3] + rows[3:],
        lambda rows: [value.replace("min 1 max 128", "min 0 max 128") for value in rows],
        lambda rows: [
            value.replace("type check default false", "type check default maybe") for value in rows
        ],
        lambda rows: rows + ["unexpected output"],
        lambda rows: [
            value.replace("CPuct type string default 1.745", "CPuct type string default nan")
            for value in rows
        ],
    ],
)
def test_handshake_refuses_missing_duplicate_or_malformed_options(change: Any) -> None:
    api = gate()
    with pytest.raises(api.UciGateError):
        api.validate_handshake(change(HANDSHAKE), {})


def run_controlled_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: str, *, expected_wdl: bool = False
) -> dict[str, Any]:
    api = gate()
    cfg = config(tmp_path)
    code = (
        "import sys\nfor line in sys.stdin:\n"
        " if line.startswith('go '):\n" + "  " + replies.replace("\n", "\n  ") + "\n"
        " if line.strip()=='quit': break\n"
    )
    monkeypatch.setattr(api, "_command", lambda *_: child(code))
    case = api.Case(
        "controlled",
        "configured",
        (
            {"send": "go nodes 8 searchmoves e2e4"},
            {
                "best": {
                    "allowed": ["e2e4"],
                    "nodes_max": 8,
                    "multipv": 1,
                    "wdl": expected_wdl,
                    "terminal": False,
                }
            },
            {"send": "quit"},
            {"exit": 0},
        ),
    )
    return cast(dict[str, Any], api.run_case(cfg, case))


BASIC_INFO = (
    INFO.replace("depth 2", "depth 1", 1)
    .replace(" pv e2e4 e7e5", " pv e2e4")
    .replace(" wdl 510 200 290", "")
)


def test_programmatic_driver_accepts_a_complete_controlled_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reply = f"print({BASIC_INFO!r},flush=True);print('bestmove e2e4',flush=True)"
    report = run_controlled_case(tmp_path, monkeypatch, reply)
    assert report["passed"]
    assert report["process"]["exit_code"] == 0
    assert report["process"]["pumps_joined"]
    assert report["searches"][0]["bestmove"] == "e2e4"


@pytest.mark.parametrize(
    "replies",
    [
        "print('bestmove e2e4',flush=True)",
        f"print({BASIC_INFO!r},flush=True);print('bestmove e2e5',flush=True)",
        f"print({BASIC_INFO!r},flush=True);"
        "print('bestmove e2e4',flush=True);print('bestmove e2e4',flush=True)",
        f"print({BASIC_INFO!r},flush=True);print('info string error unexpected',flush=True)",
        f"print({BASIC_INFO.replace('nodes 8', 'nodes 9')!r},flush=True);"
        "print('bestmove e2e4',flush=True)",
        f"print({BASIC_INFO.replace('e2e4', 'd2d4')!r},flush=True);"
        "print('bestmove e2e4',flush=True)",
        f"print({BASIC_INFO!r},flush=True);"
        "print('bestmove e2e4',flush=True);"
        "print('unexpected native stderr',file=sys.stderr,flush=True)",
    ],
)
def test_programmatic_driver_rejects_each_unexpected_error_or_short_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replies: str
) -> None:
    report = run_controlled_case(tmp_path, monkeypatch, replies)
    assert not report["passed"]
    assert report["error"]
    assert report["process"]["pumps_joined"]


def test_cleanup_error_cannot_leave_a_successful_case_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    original = api.UciProcess.__exit__

    def fail_after_cleanup(process: Any, *args: Any) -> None:
        original(process, *args)
        raise api.UciGateError("controlled cleanup failure")

    monkeypatch.setattr(api.UciProcess, "__exit__", fail_after_cleanup)
    report = run_controlled_case(
        tmp_path,
        monkeypatch,
        f"print({BASIC_INFO!r},flush=True);print('bestmove e2e4',flush=True)",
    )
    assert not report["passed"]
    assert "cleanup" in report["error"]


def test_required_fatal_exit_is_waited_with_stdin_still_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    cfg = api.GateConfig(cfg.engine, cfg.weights, cfg.failure_weights, SPEC, 0.2, 2)
    code = (
        "import sys\nfor line in sys.stdin:\n"
        " if line.startswith('go '): print('info string error finite policy required',flush=True)\n"
    )
    monkeypatch.setattr(api, "_command", lambda *_: child(code))
    case = api.Case(
        "fatal", "failure", ({"send": "go nodes 8"}, {"error": "finite"}, {"exit": "nonzero"})
    )
    report = api.run_case(cfg, case)
    assert not report["passed"]
    assert "deadline" in report["error"]
    assert report["process"]["stdin_text"] == "go nodes 8\n"
    assert report["process"]["pumps_joined"]


def test_atomic_racing_report_is_preserved_and_not_a_passing_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    output = tmp_path / "racing.json"
    original = api.os.link
    calls = 0

    def race(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            output.write_text("previous success won the publication race")
        original(source, destination)

    monkeypatch.setattr(api.os, "link", race)
    monkeypatch.setattr(api, "run_case", lambda _, case: {"id": case.identifier, "passed": True})
    report = api.run_gate(cfg, output)
    assert not report["passed"]
    assert output.read_text() == "previous success won the publication race"
    assert len(list(tmp_path.glob("racing.failed-*.json"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_unchanged_full_mock_matrix_is_counted_completely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    monkeypatch.setattr(api, "run_case", lambda _, case: {"id": case.identifier, "passed": True})
    report = api.run_gate(cfg, tmp_path / "complete-unit-simulation.json")
    assert report["passed"]
    assert report["attempted_cases"] == report["completed_cases"] == report["planned_cases"] == 21
    assert report["coverage_complete"]
    assert not report["full_release_gate"]


def test_golden_covers_true_unbounded_infinite_and_queued_oversize_recovery() -> None:
    spec = gate().load_spec(SPEC)
    infinite = next(case for case in spec.cases if case.identifier == "infinite_stop_ready")
    assert any(step.get("send") == "go infinite searchmoves e2e4" for step in infinite.steps)
    framing = next(case for case in spec.cases if case.identifier == "oversized_framing")
    assert framing.steps[0] == {"oversize": True}
    assert framing.steps[1] == {"send": "isready"}
    assert framing.steps[2] == {"error": "maximum"}


def test_held_interval_does_not_pass_a_negative_timeout_after_clock_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    process = api.UciProcess(child("raise SystemExit(99)"))
    driver = api.Driver(process, cfg)
    clock = iter((0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(api.time, "monotonic", lambda: next(clock))
    driver._quiet(0.1)
    monkeypatch.undo()


def test_expired_step_does_not_consume_an_already_queued_response(
    tmp_path: Path,
) -> None:
    api = gate()
    cfg = config(tmp_path)
    process = api.UciProcess(child("raise SystemExit(99)"))
    process.lines.put_nowait("readyok")
    driver = api.Driver(process, cfg)
    with pytest.raises(api.UciGateError, match="deadline"):
        driver._line(time.monotonic() - 1)
    assert process.lines.get_nowait() == "readyok"


ROOT_ONLY_INFO = (
    "info depth 0 seldepth 0 nodes 1 nps 8 time 110 hashfull 0 tbhits 0 multipv 1 score cp 12800"
)


def test_documented_root_only_neural_result_has_no_fabricated_child_pv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = run_controlled_case(
        tmp_path,
        monkeypatch,
        f"print({ROOT_ONLY_INFO!r},flush=True);print('bestmove e2e4',flush=True)",
    )
    assert report["passed"]
    assert report["searches"][0]["final_info"][0]["pv"] == []
    assert report["searches"][0]["final_info"][0]["depth"] == 0


@pytest.mark.parametrize(
    "info",
    [
        ROOT_ONLY_INFO.replace("nodes 1", "nodes 2"),
        ROOT_ONLY_INFO.replace("depth 0", "depth 1", 1),
        ROOT_ONLY_INFO.replace("seldepth 0", "seldepth 1"),
    ],
)
def test_empty_primary_pv_requires_exactly_the_documented_root_only_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, info: str
) -> None:
    report = run_controlled_case(
        tmp_path,
        monkeypatch,
        f"print({info!r},flush=True);print('bestmove e2e4',flush=True)",
    )
    assert not report["passed"]


def test_failed_script_records_actual_partial_step_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = run_controlled_case(
        tmp_path,
        monkeypatch,
        f"print({BASIC_INFO!r},flush=True);print('bestmove e2e5',flush=True)",
    )
    assert not report["passed"]
    assert not report["completed"]
    assert report["steps_completed"] < report["steps_planned"]
    assert report["process"]["terminated_by_runner"]


def test_model_pin_validation_cannot_refreeze_a_mid_read_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = gate()
    cfg = config(tmp_path)
    original = api._sha256
    replaced = False

    def replace_after_first_identity(path: Path) -> str:
        nonlocal replaced
        digest = str(original(path))
        if path.resolve() == cfg.weights.resolve() and not replaced:
            cfg.weights.write_bytes(b"replacement after the approved first model read")
            replaced = True
        return digest

    monkeypatch.setattr(api, "_sha256", replace_after_first_identity)
    with pytest.raises(api.UciGateError, match="changed|fixture"):
        api.Artifacts.freeze(cfg)
    assert replaced


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "default same var same var reduction var absolute",
            "default random var same var reduction var absolute",
        ),
        ("default same var same var reduction var absolute", "default same var same var absolute"),
        ("FpuValueAtRoot type string default 1", "FpuValueAtRoot type string default nan"),
        ("FpuValueAtRoot type string default 1", "FpuValueAtRoot type string default inf"),
    ],
)
def test_root_fpu_option_contract_refuses_wrong_strategy_and_nonfinite_defaults(
    before: str, after: str
) -> None:
    module = gate()
    assert len(module.validate_handshake(HANDSHAKE, {})) == 29
    lines = [line.replace(before, after) for line in HANDSHAKE]
    with pytest.raises(module.UciGateError):
        module.validate_handshake(lines, {})


def test_root_fpu_value_transactions_use_exact_decimal_semantics() -> None:
    module = gate()
    assert len(module.validate_handshake(HANDSHAKE, {"FpuValueAtRoot": "1.0"})) == 29
    with pytest.raises(module.UciGateError, match="retain"):
        module.validate_handshake(HANDSHAKE, {"FpuValueAtRoot": "0.5"})


def test_root_golden_contract_sets_both_options_without_losing_a_protocol_case() -> None:
    root = Path(__file__).resolve().parents[2] / "tests/data/uci/protocol-v1"
    document = json.loads((root / "expectations.json").read_text(encoding="utf-8"))
    assert document["options"].get("FpuStrategyAtRoot") == "same"
    assert document["options"].get("FpuValueAtRoot") == "1"
    assert len(document["cases"]) == 21
    case = document["cases"][0]
    assert case["id"] == "handshake_options"
    commands = [step["send"] for step in case["steps"] if "send" in step]
    assert "setoption name FpuStrategyAtRoot value same" in commands
    assert "setoption name FpuValueAtRoot value 1" in commands
    values = [step["handshake"] for step in case["steps"] if step.get("handshake")]
    assert len(values) == 1
    assert values[0]["FpuStrategyAtRoot"] == "same"
    assert values[0]["FpuValueAtRoot"] == "1"
    assert len(gate().load_spec(root).cases) == 21


def test_failed_cli_exposes_case_reason_and_progress_to_ci_logs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = dict(
        status="failed",
        passed=False,
        completed_cases=20,
        planned_cases=21,
        report_path="retained.json",
        errors=["artifact mismatch"],
        cases=[
            dict(
                id="ponder_hit",
                passed=False,
                error="response deadline",
                steps_completed=8,
                steps_planned=14,
            )
        ],
    )
    monkeypatch.setattr(gate(), "run_gate", lambda *_: report)
    assert (
        gate().main(
            ["--engine", "unused.exe", "--weights", "unused.onnx", "--output", "retained.json"]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["completed_cases"] == 20
    assert "ponder_hit" in captured.err and "response deadline" in captured.err
    assert "steps_completed" in captured.err and "artifact mismatch" in captured.err


def test_timed_golden_cases_explicitly_warm_the_model_and_preserve_usable_time() -> None:
    root = Path(__file__).resolve().parents[2] / "tests/data/uci/protocol-v1"
    document = json.loads((root / "expectations.json").read_text(encoding="utf-8"))
    cases = {row["id"]: row["steps"] for row in document["cases"]}
    limits = cases["finite_limits"]
    assert {"send": "setoption name MoveOverheadMs value 0"} in limits
    assert {"send": "go movetime 25 searchmoves e2e4"} in limits
    ponder = cases["ponder_hit"]
    go = next(i for i, step in enumerate(ponder) if str(step.get("send", "")).startswith("go "))
    assert {"send": "isready"} in ponder[:go]
    assert {"expect": "readyok"} in ponder[:go]
    assert any("wtime 1000 btime 1000 winc 1 binc 1" in str(step) for step in ponder)
    assert len(cases) == 21


@pytest.mark.parametrize("name", ["CacheHistoryLength", "LogSearchStats"])
def test_handshake_requires_new_search_options_without_duplicates(name: str) -> None:
    api = gate()
    declaration = next(row for row in HANDSHAKE if row.startswith(f"option name {name} "))
    with pytest.raises(api.UciGateError):
        api.validate_handshake([row for row in HANDSHAKE if row != declaration], {})
    with pytest.raises(api.UciGateError):
        api.validate_handshake(HANDSHAKE[:-1] + [declaration, "uciok"], {})
