# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic complete sweeps; no native engine, network or GPU is launched."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest


def module() -> ModuleType:
    return importlib.import_module("mouse_tools.backend_throughput")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    build, reference, sources = tmp_path / "build", tmp_path / "reference", tmp_path / "engine"
    for directory in (build, reference, sources):
        directory.mkdir()
    engine, lc0 = build / "mouse.exe", reference / "lc0.exe"
    model, net = tmp_path / "model.onnx", tmp_path / "net.pb.gz"
    source = sources / "main.cpp"
    for path in (engine, lc0, model, net, source):
        path.write_bytes(path.name.encode())
    sdks = [tmp_path / name for name in ("cuda", "cudnn", "trt")]
    runtime = {}
    for directory in (build, reference, *sdks):
        directory.mkdir(exist_ok=True)
        library = directory / "runtime.dll"
        library.write_bytes(str(directory).encode())
        runtime[str(library)] = sha(library)
    cache = raw / "cache"
    cache.mkdir()
    mouse = [
        str(engine),
        "backendbench",
        "--weights",
        str(model),
        "--backend",
        "tensorrt",
        "--precision",
        "fp16",
    ]
    for directory in sdks:
        mouse.extend(("--gpu-dll-dir", str(directory)))
    mouse.extend(
        (
            "--cache-dir",
            str(cache),
            "--start-batch-size",
            "1",
            "--max-batch-size",
            "1024",
            "--batch-step",
            "1",
            "--batches",
            "100",
            "--cpu-threads",
            "1",
        )
    )
    reference_command = [
        str(lc0),
        "backendbench",
        f"--weights={net}",
        "--backend=cuda-fp16",
        "--backend-opts=max_batch=1024,min_batch=1",
        "--nncache=0",
        "--threads=1",
        "--start-batch-size=1",
        "--max-batch-size=1024",
        "--batch-step=1",
        "--batches=100",
        "--policy-softmax-temp=1.359",
        "--history-fill-new=fen_only",
    ]
    header = (
        'benchmark_config version "0.3.0-dev" git "80d2fb5f626d-dirty" '
        'build "Release" compiler "MSVC 19.44.35229.0" '
        "ort_version 1.22.0 model_sha256 "
        + sha(model)
        + " start_batch 1 max_batch 1024 step 1 batches 100 "
        "device_index 0 cpu_threads 1 workspace_bytes 1073741824 cache disabled"
    )
    lines = [
        header,
        "artifact_path " + json.dumps(str(model)) + " cache_directory " + json.dumps(str(cache)),
    ]
    lines.extend("gpu_sdk_directory " + json.dumps(str(directory)) for directory in sdks)
    lines.append(
        'device_identity name "GPU" uuid "GPU-123" driver "617.42" '
        'cuda "12.8" tensorrt "10.9.0" cache_namespace "' + "a" * 64 + '"'
    )
    mouse_rows, reference_rows, ref_lines = [], [], []
    for batch in range(1, 1025):
        mouse_rate = {1: 606.708, 235: 30958.773}.get(batch, 1000.0 + batch)
        time_ms = float(f"{batch * 100 * 1000 / mouse_rate:.3f}")
        lines.append(
            f"batch {batch} physical_batch {batch} batches 100 nps {mouse_rate:.3f} "
            f"time_ms {time_ms:.3f} initialization_ms 100.000 warm_ms 1.000 "
            f"provider TensorrtExecutionProvider precision fp16 model_sha256 {sha(model)}"
        )
        mouse_rows.append(
            dict(batch=batch, physical=batch, batches=100, nps=mouse_rate, time_ms=time_ms)
        )
        reference_rate = {1: 572.169, 288: 34052.0}.get(batch, 2000.0 + batch)
        average = 1.74773 if batch == 1 else float(f"{batch * 1000 / reference_rate:.6g}")
        ref_lines.append(
            f"Benchmark batch size {batch} with inference average time {average:.6g}ms "
            f"- throughput {reference_rate:.6g} nps."
        )
        reference_rows.append(dict(batch=batch, average_ms=average, nps=reference_rate))
    lines.append("best_batch 235 best_nps 30958.773 cache disabled")
    state = dict(
        schema=1,
        status="complete",
        phase="reference-full",
        source_commit="d" * 40,
        executable_build_git="80d2fb5f626d-dirty",
        frozen_sha256={
            k: sha(v)
            for k, v in dict(
                engine=engine, reference_engine=lc0, source_net=net, onnx=model
            ).items()
        },
        runtime_sha256=runtime,
        source_sha256={str(source): sha(source)},
        mouse_command=mouse,
        reference_command=reference_command,
        counts=dict(start=1, max=1024, step=1, batches=100),
        parallel_gpu_jobs=1,
        mouse_exit=0,
        reference_exit=0,
        mouse_process_seconds=100000.0,
        reference_process_seconds=100000.0,
        mouse_rows=mouse_rows,
        reference_rows=reference_rows,
        best_mouse=mouse_rows[0],
        best_reference=reference_rows[0],
        ratio=1.06036503200977,
        gate_80_percent=True,
        rc_backend_90_percent=True,
    )
    (raw / "state.json").write_text(json.dumps(state))
    (raw / "mouse-full.stdout.txt").write_text("\n".join(lines) + "\n")
    (raw / "reference-full.stdout.txt").write_text("\n".join(ref_lines) + "\n")
    (raw / "mouse-full.stderr.txt").write_text("retained runtime warning\n")
    (raw / "reference-full.stderr.txt").write_text("Lc0 v0.32.1\nGPU: GPU\n")
    (raw / "mouse-model.txt").write_text("source_sha256: " + sha(net) + "\n")
    (raw / "mouse-version.txt").write_text(
        "Mouse 0.3.0-dev\ngit: 80d2fb5f626d-dirty\nbuild: Release\n"
        "compiler: MSVC 19.44.35229.0\nONNX Runtime: 1.22.0\n"
    )
    gpu = (
        "name, uuid, driver_version, memory.total [MiB], utilization.gpu [%], temperature.gpu\n"
        "GPU, GPU-123, 617.42, 12227 MiB, 0 %, 28\n"
    )
    (raw / "gpu-before.txt").write_text(gpu)
    (raw / "gpu-after.txt").write_text(gpu)
    (raw / "run-full-sweeps.ps1").write_text("# retained original controller; never executed\n")
    return raw


def test_original_ordered_dictionary_best_bug_is_rejected_and_raw_maxima_recomputed(
    tmp_path: Path,
) -> None:
    m = module()
    raw = bundle(tmp_path)
    before = {p: sha(p) for p in raw.iterdir() if p.is_file()}
    report = m.audit_sweep(raw, tmp_path / "audit.json")
    assert report["raw_evidence_valid"] and report["status"] == "audited"
    assert report["controller_summary_valid"] is False
    assert report["original_controller_summary"]["best_mouse"]["batch"] == 1
    assert report["original_controller_summary"]["best_reference"]["batch"] == 1
    assert report["best_mouse"]["batch"] == 235 and report["best_mouse"]["nps"] == 30958.773
    assert report["best_reference"]["batch"] == 288 and report["best_reference"]["nps"] == 34052.0
    assert report["ratio"] == pytest.approx(30958.773 / 34052.0)
    assert report["gate_80_percent"] and report["rc_backend_90_percent"]
    assert len(report["mouse_rows"]) == len(report["reference_rows"]) == 1024
    assert {p: sha(p) for p in before} == before
    assert report["raw_file_sha256"]["state.json"] == before[raw / "state.json"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_row",
        "duplicate_row",
        "reordered",
        "padding",
        "batches",
        "zero",
        "nan",
        "inconsistent_time",
        "provider",
        "precision",
        "summary",
        "extra_output",
        "model",
    ],
)
def test_bad_mouse_table_never_passes(tmp_path: Path, change: str) -> None:
    m = module()
    raw = bundle(tmp_path)
    p = raw / "mouse-full.stdout.txt"
    s = p.read_text()
    lines = s.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("batch 1 "))
    if change == "missing_row":
        del lines[first]
    elif change == "duplicate_row":
        lines.insert(first, lines[first])
    elif change == "reordered":
        lines[first], lines[first + 1] = lines[first + 1], lines[first]
    else:
        old, new = {
            "padding": ("physical_batch 1 ", "physical_batch 8 "),
            "batches": ("batches 100 nps", "batches 99 nps"),
            "zero": ("nps 606.708", "nps 0.000"),
            "nan": ("nps 606.708", "nps NaN"),
            "inconsistent_time": ("time_ms 164.824", "time_ms 1.000"),
            "provider": ("TensorrtExecutionProvider", "CPUExecutionProvider"),
            "precision": ("precision fp16", "precision fp32"),
            "summary": ("best_batch 235", "best_batch 1"),
            "extra_output": ("cache disabled\n", "cache disabled\nunknown output\n"),
            "model": ("model_sha256 " + sha(tmp_path / "model.onnx"), "model_sha256 " + "0" * 64),
        }[change]
        s = s.replace(old, new, 1)
        lines = s.splitlines()
    p.write_text("\n".join(lines) + "\n")
    report = m.audit_sweep(raw, tmp_path / "bad.json")
    assert not report["raw_evidence_valid"] and report["status"] == "failed"
    assert report["error"] and "ratio" not in report


@pytest.mark.parametrize(
    "change", ["missing_row", "nonfinite", "inconsistent_time", "extra_output", "false_precision"]
)
def test_bad_reference_table_never_passes(tmp_path: Path, change: str) -> None:
    m = module()
    raw = bundle(tmp_path)
    p = raw / "reference-full.stdout.txt"
    s = p.read_text()
    if change == "missing_row":
        s = s[s.index("\n") + 1 :]
    elif change == "nonfinite":
        s = s.replace("572.169 nps", "Infinity nps", 1)
    elif change == "inconsistent_time":
        s = s.replace("1.74773ms", "0.00001ms", 1)
    elif change == "false_precision":
        s = s.replace("572.169 nps", "572.16901 nps", 1)
    else:
        s += "garbage\n"
    p.write_text(s)
    assert not m.audit_sweep(raw, tmp_path / "bad.json")["raw_evidence_valid"]


@pytest.mark.parametrize(
    "change",
    [
        "counts",
        "native_exit",
        "parallel",
        "state_row",
        "hash",
        "runtime",
        "new_runtime",
        "source",
        "new_source",
        "command",
        "device",
        "source_net",
        "duplicate_json",
    ],
)
def test_incomplete_or_inconsistent_identity_never_passes(tmp_path: Path, change: str) -> None:
    m = module()
    raw = bundle(tmp_path)
    p = raw / "state.json"
    state = json.loads(p.read_text())
    if change == "counts":
        state["counts"]["step"] = 2
    elif change == "native_exit":
        state["reference_exit"] = 1
    elif change == "parallel":
        state["parallel_gpu_jobs"] = 2
    elif change == "state_row":
        state["mouse_rows"][0]["nps"] = 999.0
    elif change == "hash":
        state["frozen_sha256"]["engine"] = "0" * 64
    elif change == "runtime":
        (tmp_path / "cuda/runtime.dll").write_bytes(b"changed")
    elif change == "new_runtime":
        (tmp_path / "cuda/new.dll").write_bytes(b"added")
    elif change == "source":
        (tmp_path / "engine/main.cpp").write_bytes(b"changed")
    elif change == "new_source":
        (tmp_path / "engine/new.h").write_bytes(b"added")
    elif change == "command":
        state["reference_command"] = state["reference_command"][:-1]
    elif change == "device":
        (raw / "gpu-after.txt").write_text(
            (raw / "gpu-after.txt").read_text().replace("GPU-123", "GPU-other")
        )
    elif change == "source_net":
        (raw / "mouse-model.txt").write_text("source_sha256: " + "0" * 64 + "\n")
    p.write_text(json.dumps(state))
    if change == "duplicate_json":
        p.write_text(p.read_text()[:-1] + ',"schema":1}')
    report = m.audit_sweep(raw, tmp_path / "bad.json")
    assert not report["raw_evidence_valid"] and report["error"]


def test_atomic_new_output_cannot_replace_evidence_or_an_existing_report(tmp_path: Path) -> None:
    m = module()
    raw = bundle(tmp_path)
    state = raw / "state.json"
    before = state.read_bytes()
    with pytest.raises(m.ThroughputError):
        m.audit_sweep(raw, state)
    assert state.read_bytes() == before
    output = tmp_path / "audit.json"
    output.write_bytes(b"retained prior report")
    with pytest.raises(m.ThroughputError):
        m.audit_sweep(raw, output)
    assert output.read_bytes() == b"retained prior report"


def test_changed_artifact_during_audit_cannot_publish_valid_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = module()
    raw = bundle(tmp_path)
    original = m.parse_reference

    def mutation(text: str) -> object:
        result = original(text)
        (tmp_path / "model.onnx").write_bytes(b"mutated while parsing")
        return result

    monkeypatch.setattr(m, "parse_reference", mutation)
    report = m.audit_sweep(raw, tmp_path / "changed.json")
    assert not report["raw_evidence_valid"]


def test_cli_reports_corrected_raw_success_but_never_approves_original_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    m = module()
    raw = bundle(tmp_path)
    path = tmp_path / "cli.json"
    assert m.main(["--raw-dir", str(raw), "--output", str(path)]) == 0
    captured = capsys.readouterr()
    assert "controller_summary_valid=False" in captured.out and not captured.err
    assert json.loads(path.read_text())["controller_summary_valid"] is False
    (raw / "reference-full.stdout.txt").write_text("truncated")
    assert m.main(["--raw-dir", str(raw), "--output", str(tmp_path / "failed.json")]) == 1
    assert "failed" in capsys.readouterr().err
