# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import hashlib
import importlib
import struct
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from mouse_tools.encoding import Plane


def module() -> ModuleType:
    return importlib.import_module("mouse_tools.backend_parity")


def test_exact_little_endian_input_transport_roundtrip() -> None:
    m = module()
    rows = ((Plane(1, 1.25),) + (Plane(),) * 111, (Plane(1 << 63, -2.5),) * 112)
    raw = m.encode_inputs(rows)
    assert raw[:16] == b"MOUSEIN1" + struct.pack("<II", 2, 112)
    assert raw[16:28] == struct.pack("<Qf", 1, 1.25)
    assert len(raw) == 16 + 2 * 112 * 12
    assert m.decode_inputs(raw) == rows


@pytest.mark.parametrize(
    "damage", ["magic", "empty", "count0", "count1025", "planes", "short", "extra", "nan"]
)
def test_input_reader_rejects_malformed_transport(damage: str) -> None:
    m = module()
    raw = bytearray(m.encode_inputs(((Plane(),) * 112,)))
    if damage == "magic":
        raw[0] = 0
    elif damage == "empty":
        raw.clear()
    elif damage == "count0":
        struct.pack_into("<I", raw, 8, 0)
    elif damage == "count1025":
        struct.pack_into("<I", raw, 8, 1025)
    elif damage == "planes":
        struct.pack_into("<I", raw, 12, 111)
    elif damage == "short":
        raw.pop()
    elif damage == "extra":
        raw.extend(b"\0")
    else:
        struct.pack_into("<f", raw, 24, float("nan"))
    with pytest.raises(m.BackendParityError):
        m.decode_inputs(bytes(raw))


def test_input_writer_refuses_invalid_batch_and_half_missing_planes() -> None:
    m = module()
    for rows in ((), ((Plane(),) * 111,), ((Plane(1, 1e100),) * 112,)):
        with pytest.raises(m.BackendParityError):
            m.encode_inputs(rows)


def output_bytes(rows: NDArray[np.float32], model: str = "a" * 64, inputs: str = "b" * 64) -> bytes:
    return (
        b"MOUSEOU1"
        + struct.pack("<II", len(rows), 1862)
        + model.encode()
        + inputs.encode()
        + rows.astype("<f4").tobytes()
    )


def heads() -> NDArray[np.float32]:
    rows = np.zeros((2, 1862), dtype=np.float32)
    rows[:, 1858:1861] = [0.2, 0.3, 0.5]
    rows[:, 1861] = 12
    return rows


def test_output_reader_binds_both_hashes_and_exact_heads() -> None:
    m = module()
    actual = m.parse_output(
        output_bytes(heads()), positions=2, model_sha256="a" * 64, inputs_sha256="b" * 64
    )
    np.testing.assert_array_equal(actual, heads())
    assert actual.dtype == np.float32


@pytest.mark.parametrize(
    "damage",
    [
        "magic",
        "count",
        "width",
        "model",
        "input",
        "short",
        "extra",
        "nan",
        "wdl_negative",
        "wdl_sum",
        "mlh",
    ],
)
def test_output_reader_rejects_incomplete_unbound_or_invalid_heads(damage: str) -> None:
    m = module()
    values = heads()
    if damage == "nan":
        values[0, 1] = float("nan")
    elif damage == "wdl_negative":
        values[0, 1858:1861] = [-0.2, 0.7, 0.5]
    elif damage == "wdl_sum":
        values[0, 1858:1861] = [0.1, 0.1, 0.1]
    elif damage == "mlh":
        values[0, 1861] = -1
    raw = bytearray(output_bytes(values))
    if damage == "magic":
        raw[0] = 0
    elif damage == "count":
        struct.pack_into("<I", raw, 8, 1)
    elif damage == "width":
        struct.pack_into("<I", raw, 12, 1861)
    elif damage == "model":
        raw[16] = ord("c")
    elif damage == "input":
        raw[80] = ord("c")
    elif damage == "short":
        raw.pop()
    elif damage == "extra":
        raw.extend(b"\0")
    with pytest.raises(m.BackendParityError):
        m.parse_output(bytes(raw), positions=2, model_sha256="a" * 64, inputs_sha256="b" * 64)


def test_legal_softmax_uses_only_complete_legal_indices_and_temperature_one() -> None:
    m = module()
    logits = np.full(1858, 1000.0, dtype=np.float32)
    logits[[3, 8, 19]] = [1.0, 2.0, 3.0]
    expected = np.exp(np.array([-2.0, -1.0, 0.0]))
    expected /= expected.sum()
    np.testing.assert_allclose(
        m.legal_probabilities(logits, (3, 8, 19)), expected, rtol=0, atol=1e-15
    )
    for indices in ((), (1, 1), (-1,), (1858,)):
        with pytest.raises(m.BackendParityError):
            m.legal_probabilities(logits, indices)


def test_frozen_fixture_retains_all_edges_exact50_distinct_and_binary() -> None:
    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    cases = m.load_cases(root / "cases.json")
    assert len(cases) == 50
    assert len({c.tensor_sha256 for c in cases}) == 50
    assert len([c for c in cases if c.identifier in m.REQUIRED_EDGE_CASES]) == 27
    assert m.encode_inputs(tuple(c.planes for c in cases)) == (root / "planes.bin").read_bytes()
    assert hashlib.sha256((root / "cases.json").read_bytes()).hexdigest() == m.FIXTURE_SHA256
    provenance = root / "provenance.json"
    assert provenance.is_file()


def test_run_gate_uses_native_command_and_actual_ort_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from test_convert_parity import constant_model

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    engine, model, reference = (
        tmp_path / "mouse.exe",
        tmp_path / "model.onnx",
        tmp_path / "oracle.onnx",
    )
    engine.write_bytes(b"unit transport process identity")
    model_hash = constant_model(model, reference=False)
    oracle_hash = constant_model(reference, reference=True)
    monkeypatch.setattr(m, "OFFICIAL_REFERENCE_SHA256", oracle_hash)
    values = np.zeros((50, 1862), dtype=np.float32)
    values[:, 1858:1861] = 1 / 3
    values[:, 1861] = 7
    commands = []

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        assert kwargs["capture_output"] is True and kwargs["check"] is False
        return subprocess.CompletedProcess(
            command,
            0,
            output_bytes(
                values, model_hash, hashlib.sha256((root / "planes.bin").read_bytes()).hexdigest()
            ),
            b"CPU diagnostic\n",
        )

    monkeypatch.setattr(subprocess, "run", execute)
    report = m.run_gate(
        engine,
        model,
        reference,
        root / "cases.json",
        root / "planes.bin",
        backend="cpu",
        precision="fp32",
        buckets=(64,),
    )
    assert report["positions"] == 50 and report["passed"]
    assert not report["gate_satisfied"]
    assert report["max_abs"]["policy_probability"] == 0
    assert report["max_abs"]["wdl"] == 0
    assert len(report["results"]) == 50
    assert commands == [
        [
            str(engine.resolve()),
            "backend-eval",
            "--weights",
            str(model.resolve()),
            "--backend",
            "cpu",
            "--precision",
            "fp32",
            "--inputs",
            str((root / "planes.bin").resolve()),
            "--buckets",
            "64",
        ]
    ]


@pytest.mark.parametrize("changed", ["engine", "model", "reference", "cases", "inputs"])
def test_run_gate_rejects_artifact_changes_during_native_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    import shutil
    import subprocess

    from test_convert_parity import constant_model

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    engine, model, reference = (
        tmp_path / "mouse.exe",
        tmp_path / "model.onnx",
        tmp_path / "oracle.onnx",
    )
    engine.write_bytes(b"unit transport process identity")
    model_hash = constant_model(model, reference=False)
    monkeypatch.setattr(m, "OFFICIAL_REFERENCE_SHA256", constant_model(reference, reference=True))
    cases, inputs = tmp_path / "cases.json", tmp_path / "planes.bin"
    shutil.copyfile(root / "cases.json", cases)
    shutil.copyfile(root / "planes.bin", inputs)
    input_hash = hashlib.sha256(inputs.read_bytes()).hexdigest()
    paths = {
        "engine": engine,
        "model": model,
        "reference": reference,
        "cases": cases,
        "inputs": inputs,
    }
    values = np.zeros((50, 1862), dtype=np.float32)
    values[:, 1858:1861] = 1 / 3

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        target = paths[changed]
        raw = target.read_bytes()
        target.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        return subprocess.CompletedProcess(
            command, 0, output_bytes(values, model_hash, input_hash), b""
        )

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(m.BackendParityError, match=changed + ".*changed"):
        m.run_gate(
            engine, model, reference, cases, inputs, backend="cpu", precision="fp32", buckets=(64,)
        )


def test_run_gate_cannot_accept_large_policy_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from test_convert_parity import constant_model

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    engine, model, reference = (
        tmp_path / "mouse.exe",
        tmp_path / "model.onnx",
        tmp_path / "oracle.onnx",
    )
    engine.write_bytes(b"unit process")
    model_hash = constant_model(model, reference=False)
    monkeypatch.setattr(m, "OFFICIAL_REFERENCE_SHA256", constant_model(reference, reference=True))
    values = np.zeros((50, 1862), dtype=np.float32)
    values[:, 1858:1861] = 1 / 3
    values[0, 34] = 1

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            0,
            output_bytes(
                values, model_hash, hashlib.sha256((root / "planes.bin").read_bytes()).hexdigest()
            ),
            b"",
        )

    monkeypatch.setattr(subprocess, "run", execute)
    report = m.run_gate(
        engine,
        model,
        reference,
        root / "cases.json",
        root / "planes.bin",
        backend="cpu",
        precision="fp32",
        buckets=(64,),
    )
    assert not report["passed"] and not report["gate_satisfied"]
    assert report["tolerance"] == 0.005 and len(report["results"]) == 50
    assert report["max_abs"]["policy_probability"] > 0.005


def test_run_gate_refuses_nonzero_child_and_concise_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys

    from test_convert_parity import constant_model

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    engine, model, reference = (
        tmp_path / "mouse.exe",
        tmp_path / "model.onnx",
        tmp_path / "oracle.onnx",
    )
    engine.write_bytes(b"unit process")
    constant_model(model, reference=False)
    monkeypatch.setattr(m, "OFFICIAL_REFERENCE_SHA256", constant_model(reference, reference=True))

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 9, b"", b"backend unavailable")

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(m.BackendParityError, match="exited 9.*backend unavailable"):
        m.run_gate(
            engine,
            model,
            reference,
            root / "cases.json",
            root / "planes.bin",
            backend="cpu",
            precision="fp32",
            buckets=(64,),
        )
    monkeypatch.undo()
    command = [
        sys.executable,
        "-I",
        "-m",
        "mouse_tools.backend_parity",
        "--engine",
        str(engine),
        "--weights",
        str(model),
        "--reference",
        str(reference),
        "--cases",
        str(root / "cases.json"),
        "--inputs",
        str(root / "planes.bin"),
        "--backend",
        "cpu",
        "--precision",
        "fp32",
        "--buckets",
        "64",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    assert completed.returncode == 1 and not completed.stdout
    assert (
        "reference ONNX SHA256 mismatch" in completed.stderr and "Traceback" not in completed.stderr
    )
    completed = subprocess.run(
        [sys.executable, "-m", "mouse_tools.backend_parity"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2 and "python -I" in completed.stderr


@pytest.mark.parametrize("phase", ["cases", "oracle"])
def test_hashes_are_frozen_before_any_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    import subprocess

    from test_convert_parity import constant_model

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    engine, model, reference = (
        tmp_path / "mouse.exe",
        tmp_path / "model.onnx",
        tmp_path / "oracle.onnx",
    )
    engine.write_bytes(b"unit process")
    model_hash = constant_model(model, reference=False)
    monkeypatch.setattr(m, "OFFICIAL_REFERENCE_SHA256", constant_model(reference, reference=True))
    original = getattr(m, "load_cases" if phase == "cases" else "_oracle")

    def load(*args: object) -> object:
        engine.write_bytes(b"changed-before-native-load")
        return original(*args)

    monkeypatch.setattr(m, "load_cases" if phase == "cases" else "_oracle", load)
    values = np.zeros((50, 1862), dtype=np.float32)
    values[:, 1858:1861] = 1 / 3

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            command,
            0,
            output_bytes(
                values, model_hash, hashlib.sha256((root / "planes.bin").read_bytes()).hexdigest()
            ),
            b"",
        )

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(m.BackendParityError, match="engine.*changed"):
        m.run_gate(
            engine,
            model,
            reference,
            root / "cases.json",
            root / "planes.bin",
            backend="cpu",
            precision="fp32",
            buckets=(64,),
        )


def test_fixture_planes_and_legal_indices_are_from_actual_binary_edges() -> None:
    import json

    from mouse_tools.lc0_reference import read_training_v6

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn"
    cases = m.load_cases(root / "backend-parity-v1/cases.json")
    manifest = json.loads((root / "lc0-classical-v1/reference.json").read_text())
    provenance = json.loads((root / "backend-parity-v1/provenance.json").read_text())
    assert provenance["positions"] == provenance["distinct_tensors"] == 50
    assert provenance["named_edges"] == 27 and provenance["additional_accepted"] == 23
    assert provenance["cases_sha256"] == m.FIXTURE_SHA256
    assert (
        provenance["inputs_sha256"]
        == hashlib.sha256((root / "backend-parity-v1/planes.bin").read_bytes()).hexdigest()
    )
    for case, original in zip(cases[:27], manifest["cases"], strict=True):
        assert case.identifier == original["id"]
        path = root / "lc0-classical-v1" / original["training"]["fen_only"]["first_record_file"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == provenance["training_chunks_sha256"][path.name]
        )
        (record,) = read_training_v6(path)
        assert case.planes == record.planes
        assert set(index for _, index in case.moves) == set(record.legal_indices)
        assert dict(case.moves) == original["policy"]["moves"]


def test_changed_or_resampled_case_fixture_is_rejected(tmp_path: Path) -> None:
    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/nn/backend-parity-v1"
    target = tmp_path / "cases.json"
    target.write_bytes((root / "cases.json").read_bytes() + b" ")
    with pytest.raises(m.BackendParityError, match="SHA256"):
        m.load_cases(target)
