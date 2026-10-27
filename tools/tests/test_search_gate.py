# SPDX-License-Identifier: GPL-3.0-or-later
"""Fake children test the gate; no native engine or GPU is executed here."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import struct
import sys
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mouse_tools.search_gate import MateConfig

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/data/search/mates-v1"


def module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.search_gate") is not None, "Mate gate missing"
    return importlib.import_module("mouse_tools.search_gate")


def config(tmp_path: Path) -> MateConfig:
    from mouse_tools.search_gate import MateConfig

    module()
    engine, weights = tmp_path / "mouse.exe", tmp_path / "model.onnx"
    engine.write_bytes(b"unit fake executable; never launched")
    weights.write_bytes(b"unit fake model; never loaded")
    (tmp_path / "onnxruntime.dll").write_bytes(b"unit fake app-local runtime; never loaded")
    return MateConfig(engine, weights, CASES, "cpu", "fp32")


def response(command: list[str]) -> dict[str, object]:
    fen = command[command.index("--fen") + 1]
    case = next(
        row for row in json.loads((CASES / "mates.json").read_text())["cases"] if row["fen"] == fen
    )
    nodes = int(command[command.index("--nodes") + 1])
    weights = Path(command[command.index("--weights") + 1])
    return {
        "schema": "MouseSearch1",
        "version": "0.4.0-dev",
        "git": "80d2fb5-dirty",
        "provider": "CPUExecutionProvider",
        "precision": "fp32",
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(
            ("MouseSearchInput1\nchess960=0\n" + fen + "\n").encode()
        ).hexdigest(),
        "fen": fen,
        "deterministic": True,
        "threads": 1,
        "batch": int(command[command.index("--batch") + 1]),
        "nncache_bytes": 0,
        "stop_reason": "nodes",
        "nodes": nodes,
        "new_playouts": nodes,
        "inherited_visits": 0,
        "depth": len(case["pv"]),
        "seldepth": len(case["pv"]),
        "q": 0.5,
        "draw": 0.1,
        "elapsed_ms": 10,
        "nps": nodes * 100.0,
        "charged_bytes": 10_000,
        "budget_bytes": int(command[command.index("--ram-limit-mb") + 1]) * 1024 * 1024,
        "tree_nodes": 7,
        "tree_capacity": 100,
        "bestmove": case["winning_moves"][0],
        "pv": case["pv"],
    }


def fake_child(command: list[str], timeout: float) -> object:
    m = module()
    return m.ProcessResult(
        0, json.dumps(response(command)).encode(), b"fake child diagnostics", 0.01
    )


def test_full_fixed18_gate_records_every_explicit_budget_and_actual_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    c = config(tmp_path)
    monkeypatch.setattr(m, "run_child", fake_child)
    path = tmp_path / "pass.json"
    report = m.run_mates(c, path)
    assert report["status"] == "passed" and report["passed"] is True
    assert report["planned_cases"] == report["attempted_cases"] == report["passed_cases"] == 18
    assert len(report["checks"]) == 18
    for row in report["checks"]:
        expected = {1: 128, 2: 1024, 3: 8192}[row["mate"]]
        assert row["node_budget"] == row["result"]["nodes"] == expected
        assert row["winning_root_move_verified"] and row["pv_terminal_checkmate"]
        command = row["command"]
        for flag, value in [
            ("--nodes", str(expected)),
            ("--threads", "1"),
            ("--batch", "8"),
            ("--cpu-threads", "1"),
            ("--nncache-mb", "0"),
            ("--ram-limit-mb", "64"),
        ]:
            assert command[command.index(flag) + 1] == value
        assert "--deterministic" in command and "--no-smart-pruning" in command
        assert row["stdout"] and row["stderr"] == "fake child diagnostics"
    assert json.loads(path.read_text())["passed_cases"] == 18
    assert report["search_proved_forced_mate"] is False
    assert report["hashes_unchanged"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("nodes", 127),
        ("nodes", 129),
        ("stop_reason", "memory"),
        ("stop_reason", "time"),
        ("inherited_visits", 1),
        ("new_playouts", 0),
        ("provider", "CUDAExecutionProvider"),
        ("precision", "fp16"),
        ("model_sha256", "0" * 64),
        ("input_sha256", "0" * 64),
        ("deterministic", False),
        ("threads", True),
        ("batch", 1),
        ("nncache_bytes", 1),
        ("bestmove", "a1a8"),
        ("q", 2.0),
        ("draw", -0.1),
        ("elapsed_ms", -1),
        ("pv", ["b2b7", "a1a8"]),
        ("missing", None),
        ("unexpected", 1),
    ],
)
def test_mismatches_never_pass_and_all18_attempts_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    m = module()
    c = config(tmp_path)

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        if field == "missing":
            del record["git"]
        else:
            record[field] = value
        return m.ProcessResult(0, json.dumps(record).encode(), b"bad diagnostics", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(c, tmp_path / "failure.json")
    assert report["status"] == "failed" and report["passed"] is False
    assert report["attempted_cases"] == 18
    assert any(not row["passed"] and row["error"] for row in report["checks"])
    assert all(row["stdout"] and row["stderr"] for row in report["checks"])


@pytest.mark.parametrize(
    "body",
    [
        b'{"nodes":1,"nodes":2}',
        b'{"q":NaN}',
        b'{"q":1e309}',
        b'{"q":Infinity}',
        b"[]",
        b"{} trailing",
    ],
)
def test_invalid_json_does_not_become_a_partial_search_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    m = module()
    monkeypatch.setattr(m, "run_child", lambda *_: m.ProcessResult(0, body, b"", 0.01))
    report = m.run_mates(config(tmp_path), tmp_path / "bad.json")
    assert not report["passed"] and report["attempted_cases"] == 18
    assert report["passed_cases"] == 0


@pytest.mark.parametrize("artifact", ["engine", "model", "runtime", "corpus", "new_runtime"])
def test_changed_artifact_cannot_relabel_a_pass_with_new_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    m = module()
    c = config(tmp_path)
    copied = tmp_path / "cases"
    shutil.copytree(CASES, copied)
    c = replace(c, cases=copied)
    target = {
        "engine": tmp_path / "mouse.exe",
        "model": tmp_path / "model.onnx",
        "runtime": tmp_path / "onnxruntime.dll",
        "corpus": copied / "mates.json",
        "new_runtime": tmp_path / "unexpected.dll",
    }[artifact]
    before = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    changed = False

    def child(command: list[str], timeout: float) -> object:
        nonlocal changed
        result = fake_child(command, timeout)
        if not changed:
            target.write_bytes(b"changed identity B")
            changed = True
        return result

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(c, tmp_path / "mutated.json")
    assert not report["passed"] and not report["hashes_unchanged"]
    assert "changed" in report["error"].lower()
    if artifact == "model":
        assert report["artifact_sha256"]["model"] == before


@pytest.mark.parametrize("damage", ["subset", "bm", "epd", "extra"])
def test_only_the_original_full_hash_pinned_corpus_is_accepted(tmp_path: Path, damage: str) -> None:
    m = module()
    c = config(tmp_path)
    copied = tmp_path / "cases"
    shutil.copytree(CASES, copied)
    if damage == "epd":
        path = copied / "mates.epd"
        path.write_text(path.read_text().replace("dm 1;", "dm 3;", 1))
    elif damage == "extra":
        (copied / "extra.txt").write_text("unexpected corpus member")
    else:
        path = copied / "mates.json"
        value = json.loads(path.read_text())
        if damage == "subset":
            value["cases"].pop()
        else:
            value["cases"][0]["winning_moves"] = ["a1a2"]
        path.write_text(json.dumps(value))
    report = m.run_mates(replace(c, cases=copied), tmp_path / "rejected.json")
    assert not report["passed"] and report["attempted_cases"] == 0


def test_existing_success_and_racing_success_are_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    c = config(tmp_path)
    output = tmp_path / "report.json"
    original = b'{"status":"passed","old":true}'
    output.write_bytes(original)
    report = m.run_mates(c, output)
    assert not report["passed"] and report["attempted_cases"] == 0
    assert output.read_bytes() == original
    output.unlink()

    def child(command: list[str], timeout: float) -> object:
        if not output.exists():
            output.write_bytes(original)
        return fake_child(command, timeout)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(c, output)
    assert not report["passed"]
    assert output.read_bytes() == original
    assert json.loads(Path(report["report_path"]).read_text())["status"] == "failed"


def test_bounded_real_child_capture_and_timeout_do_not_pass_partial_output(tmp_path: Path) -> None:
    m = module()
    script = tmp_path / "fake.py"
    script.write_text("import time; print('partial',flush=True); time.sleep(3)")
    result = m.run_child([sys.executable, "-I", str(script)], 0.15)
    assert result.timed_out and result.returncode != 0
    assert result.stdout == b"partial\n" or result.stdout == b"partial\r\n"
    script.write_text("print('x'*200000,flush=True)")
    result = m.run_child([sys.executable, "-I", str(script)], 3)
    assert result.output_limit and len(result.stdout) <= m.MAX_STDOUT


def test_extreme_integer_statistics_fail_cleanly_instead_of_overflowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        record["q"] = 10**400
        return m.ProcessResult(0, json.dumps(record).encode(), b"overflow diagnostics", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(config(tmp_path), tmp_path / "overflow.json")
    assert not report["passed"] and report["attempted_cases"] == 18
    assert report["passed_cases"] == 0


def test_json_nesting_is_bounded_before_retaining_a_result() -> None:
    m = module()
    with pytest.raises(m.SearchGateError, match="nest|depth"):
        m.parse_json(b"[" * 40 + b"0" + b"]" * 40)


def test_partial_legal_pv_is_reported_without_claiming_an_engine_mate_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        pv = record["pv"]
        assert isinstance(pv, list)
        record["pv"] = pv[:1]
        record["depth"] = 1
        return m.ProcessResult(0, json.dumps(record).encode(), b"", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(config(tmp_path), tmp_path / "partial.json")
    assert report["passed"] and not report["search_proved_forced_mate"]
    assert sum(row["pv_terminal_checkmate"] for row in report["checks"]) == 6


def test_optional_root_diagnostics_are_complete_legal_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        record["root_moves"] = [
            {"move": move, "visits": 0, "q": 0.0, "draw": 0.0, "prior": 0.0}
            for move in m._legal_moves(record["fen"])
        ]
        return m.ProcessResult(0, json.dumps(record).encode(), b"", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    assert m.run_mates(config(tmp_path), tmp_path / "roots.json")["passed"]

    def incomplete(command: list[str], timeout: float) -> object:
        original = child(command, timeout)
        assert hasattr(original, "stdout")
        record = json.loads(original.stdout)
        record["root_moves"].pop()
        return m.ProcessResult(0, json.dumps(record).encode(), b"", 0.01)

    monkeypatch.setattr(m, "run_child", incomplete)
    assert not m.run_mates(config(tmp_path), tmp_path / "incomplete-roots.json")["passed"]


def test_pv_legality_rejects_blocked_rays_and_exposed_king() -> None:
    m = module()
    with pytest.raises(m.SearchGateError):
        m._move("kN1Q4/8/2K5/8/8/8/8/8 w - - 0 1", "d8a8")
    with pytest.raises(m.SearchGateError):
        m._move("4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1", "e2d2")


def test_local_pawnless_legal_sets_match_real_pinned_recorded_oracle() -> None:
    from mouse_tools.mate_fixtures import RecordedOracle

    m = module()
    oracle = RecordedOracle(
        CASES / "oracle-states.jsonl.gz",
        "fc547ba4453a7f68caf03f243b40e996a520aef20db3a0ff51b8aa93b727e3c2",
    )
    for case in m.load_cases(CASES):
        assert m._legal_moves(case.fen) == oracle.state(case.fen).moves


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch", 0),
        ("batch", 1025),
        ("ram_limit_mb", 0),
        ("mate3_nodes", 0),
        ("mate1_nodes", 1 << 64),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", 0),
        ("timeout_seconds", 10**1000),
        ("timeout_seconds", -(10**1000)),
        ("precision", "fp16"),
    ],
)
def test_invalid_configuration_never_launches_or_loses_failure_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    m = module()
    parameters = asdict(config(tmp_path))
    parameters[field] = value
    c = m.MateConfig(**parameters)
    output = tmp_path / "config-failure.json"
    report = m.run_mates(c, output)
    assert not report["passed"] and report["attempted_cases"] == 0
    assert report["report_path"] is not None
    assert json.loads(Path(report["report_path"]).read_text())["status"] == "failed"


def test_child_failure_and_timeout_keep_all18_failure_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    monkeypatch.setattr(
        m,
        "run_child",
        lambda *_: m.ProcessResult(
            1, b"partial result", b"timeout error details", 0.2, timed_out=True
        ),
    )
    report = m.run_mates(config(tmp_path), tmp_path / "timeout.json")
    assert not report["passed"] and report["attempted_cases"] == 18
    for row in report["checks"]:
        assert row["timed_out"] and row["stdout"] == "partial result"
        assert row["stderr"] == "timeout error details" and row["error"]


def test_report_can_never_replace_an_input(tmp_path: Path) -> None:
    m = module()
    c = config(tmp_path)
    original = c.weights.read_bytes()
    report = m.run_mates(c, c.weights)
    assert not report["passed"] and report["attempted_cases"] == 0
    assert c.weights.read_bytes() == original


def test_mutated_runtime_invalidates_individual_pass_labels_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    c = config(tmp_path)
    changed = False

    def child(command: list[str], timeout: float) -> object:
        nonlocal changed
        result = fake_child(command, timeout)
        if not changed:
            (tmp_path / "onnxruntime.dll").write_bytes(b"replaced runtime B")
            changed = True
        return result

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(c, tmp_path / "runtime-invalid.json")
    assert report["passed_cases"] == 0
    assert all(not row["passed"] for row in report["checks"])
    assert all(row["observed_passed"] for row in report["checks"])


def test_interrupt_reaps_a_real_fake_child_and_remains_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    m = module()
    real_launch = m.subprocess.Popen
    processes: list[subprocess.Popen[bytes]] = []

    def launch(*args: object, **kwargs: object) -> object:
        process = real_launch(*args, **kwargs)
        processes.append(process)
        real_wait = process.wait
        first = True

        def wait(*args: object, **kwargs: object) -> object:
            nonlocal first
            if first:
                first = False
                raise KeyboardInterrupt
            return real_wait(*args, **kwargs)

        monkeypatch.setattr(process, "wait", wait)
        return process

    monkeypatch.setattr(m.subprocess, "Popen", launch)
    script = tmp_path / "interrupt.py"
    script.write_text("import time; time.sleep(10)")
    result = None
    try:
        with suppress(KeyboardInterrupt):
            result = m.run_child([sys.executable, "-I", str(script)], 3)
        assert result is not None and result.interrupted
        assert processes[0].poll() is not None
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_run_interruption_is_retained_but_never_a_shortened_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    monkeypatch.setattr(
        m,
        "run_child",
        lambda *_: m.ProcessResult(
            1, b"interrupted partial output", b"cancelled", 0.2, interrupted=True
        ),
    )
    output = tmp_path / "interrupted.json"
    report = m.run_mates(config(tmp_path), output)
    assert not report["passed"] and report["status"] == "interrupted"
    assert report["attempted_cases"] == 1 and report["planned_cases"] == 18
    assert report["checks"][0]["stdout"] == "interrupted partial output"
    assert json.loads(output.read_text())["status"] == "interrupted"


@pytest.mark.parametrize("backend", ["cuda", "tensorrt"])
def test_gpu_commands_use_only_explicit_directories_and_freeze_the_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    m = module()
    c = config(tmp_path)
    sdk = tmp_path / "explicit-sdk"
    sdk.mkdir()
    library = sdk / "gpu-test-only.dll"
    library.write_bytes(b"fake SDK identity A; never loaded")
    c = replace(
        c, backend=backend, precision="fp16", gpu_dll_dirs=(sdk,), cache_dir=tmp_path / "cache"
    )

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        record["provider"] = {
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
        }[backend]
        record["precision"] = "fp16"
        return m.ProcessResult(0, json.dumps(record).encode(), b"fake provider, no GPU work", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(c, tmp_path / "fake-gpu.json")
    assert report["passed"] and report["child_environment"]["PATH"] == ""
    assert (
        report["artifact_sha256"]["sdk0/gpu-test-only.dll"]
        == hashlib.sha256(library.read_bytes()).hexdigest()
    )
    for row in report["checks"]:
        args = row["command"]
        assert args[args.index("--gpu-dll-dir") + 1] == str(sdk)
        assert args[args.index("--cache-dir") + 1] == str(tmp_path / "cache")


def test_actual_child_has_empty_path_without_changing_parent_environment(tmp_path: Path) -> None:
    import os

    m = module()
    original = os.environ.get("PATH")
    script = tmp_path / "environment.py"
    script.write_text("import os; print(repr(os.environ.get('PATH')),flush=True)")
    result = m.run_child([sys.executable, "-I", str(script)], 3)
    assert result.returncode == 0 and result.stdout.strip() == b"''"
    assert os.environ.get("PATH") == original


def test_isolated_cli_failure_is_concise_and_persists_a_failed_report(tmp_path: Path) -> None:
    import subprocess

    config(tmp_path)
    source = str(ROOT / "tools/src")
    code = (
        "import sys;sys.path.insert(0," + repr(source) + ");"
        "from mouse_tools.search_gate import main;raise SystemExit(main())"
    )
    output = tmp_path / "cli-failure.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            "mates",
            "--engine",
            str(tmp_path / "mouse.exe"),
            "--weights",
            str(tmp_path / "model.onnx"),
            "--cases",
            str(CASES),
            "--backend",
            "cpu",
            "--precision",
            "fp32",
            "--batch",
            "0",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1 and not result.stdout
    assert "batch" in result.stderr and "Traceback" not in result.stderr
    assert json.loads(output.read_text())["status"] == "failed"


def test_slower_mating_line_is_retained_but_fails_the_original_minimal_mate_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    fen = "8/8/8/8/8/2k5/1r6/K7 b - - 0 1"

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        if record["fen"] == fen:
            record["bestmove"] = "b2b7"
            record["pv"] = ["b2b7", "a1a2", "b7h7", "a2b1", "h7a7", "b1c1", "a7a1"]
            record["depth"] = record["seldepth"] = 7
        return m.ProcessResult(0, json.dumps(record).encode(), b"slower mate diagnostic", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(config(tmp_path), tmp_path / "slower.json")
    assert not report["passed"] and report["passed_cases"] == 17
    row = next(row for row in report["checks"] if row["fen"] == fen)
    assert row["node_budget"] == row["result"]["nodes"] == 8192
    assert row["pv_terminal_checkmate"] and not row["winning_root_move_verified"]
    assert row["result"]["bestmove"] == "b2b7" and len(row["result"]["pv"]) == 7


@pytest.mark.parametrize("enabled", [True, False, 1, None, "true"])
def test_optional_moves_left_field_matches_the_explicit_default_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: object
) -> None:
    m = module()

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        record["moves_left_utility"] = enabled
        return m.ProcessResult(0, json.dumps(record).encode(), b"", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(config(tmp_path), tmp_path / "mlh-field.json")
    assert report["passed"] is (enabled is True)
    assert report["attempted_cases"] == 18


def test_clean_git_lf_documentation_keeps_the_same_original_corpus_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    c = config(tmp_path)
    copied = tmp_path / "lf-cases"
    shutil.copytree(CASES, copied)
    readme = copied / "README.md"
    readme.write_bytes(readme.read_bytes().replace(b"\r\n", b"\n"))
    monkeypatch.setattr(m, "run_child", fake_child)
    report = m.run_mates(replace(c, cases=copied), tmp_path / "lf-checkout.json")
    assert report["passed"] and report["passed_cases"] == 18


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("q,draw", [(1.0, 1.0), (-1.0, 1.0), (0.75, 0.5), (0.8, 0.20000001)])
def test_impossible_wdl_simplex_fails_all18_without_shortening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic: bool, q: float, draw: float
) -> None:
    m = module()

    def child(command: list[str], timeout: float) -> object:
        record = response(command)
        if diagnostic:
            record["root_moves"] = [
                {"move": move, "visits": 1, "q": q, "draw": draw, "prior": 0.0}
                for move in m._legal_moves(record["fen"])
            ]
        else:
            record.update(q=q, draw=draw)
        return m.ProcessResult(0, json.dumps(record).encode(), b"invalid WDL pair", 0.01)

    monkeypatch.setattr(m, "run_child", child)
    report = m.run_mates(config(tmp_path), tmp_path / "impossible-wdl.json")
    assert not report["passed"] and report["attempted_cases"] == 18
    assert report["passed_cases"] == 0
    assert all("probability simplex" in row["error"] for row in report["checks"])


@pytest.mark.parametrize("diagnostic", [False, True])
@pytest.mark.parametrize("visits", [8192, 1_000_000])
@pytest.mark.parametrize("sign", [-1, 1])
def test_native_double_normalization_and_accumulation_rounding_is_accepted(
    tmp_path: Path, diagnostic: bool, visits: int, sign: int
) -> None:
    m = module()
    # Replay native search.cpp's float WDL -> double normalization -> plain
    # double backup -> mean. Repeated .8/.2 exceeds 8*epsilon at both counts.
    w, d, loss = (struct.unpack("f", struct.pack("f", x))[0] for x in (0.8, 0.2, 0.0))
    total = w + d + loss
    sample_q, sample_draw = (w - loss) / total, d / total
    q_sum = draw_sum = 0.0
    for _ in range(visits):
        q_sum += sign * sample_q
        draw_sum += sample_draw
    q, draw = q_sum / visits, draw_sum / visits
    assert abs(q) + draw - 1 > 8 * sys.float_info.epsilon
    c = replace(config(tmp_path), mate1_nodes=visits)
    case = m.load_cases(CASES)[0]
    record = response(m._command(c, case))
    if diagnostic:
        record["root_moves"] = [
            {"move": move, "visits": visits, "q": q, "draw": draw, "prior": 0.0}
            for move in m._legal_moves(case.fen)
        ]
    else:
        record.update(q=q, draw=draw)
    assert m.validate_result(record, case, c, record["model_sha256"]) == record


@pytest.mark.parametrize("q,draw", [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.75, 0.25), (1.0, 1.0)])
def test_extreme_visit_count_never_turns_roundoff_bound_into_a_vacuous_check(
    tmp_path: Path, q: float, draw: float
) -> None:
    m = module()
    c = replace(config(tmp_path), mate1_nodes=(1 << 64) - 1)
    case = m.load_cases(CASES)[0]
    record = response(m._command(c, case))
    record.update(q=q, draw=draw)
    if abs(q) + draw <= 1:
        assert m.validate_result(record, case, c, record["model_sha256"]) == record
    else:
        with pytest.raises(m.SearchGateError, match="probability simplex"):
            m.validate_result(record, case, c, record["model_sha256"])
