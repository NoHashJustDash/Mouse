# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]  # ORT 1.22 has no type marker.
import pytest
from test_convert_fixture import synthetic_net
from test_convert_model import numpy_forward

from mouse_tools.convert.export import convert_network, validate_model
from mouse_tools.convert.schema import parse_net
from mouse_tools.convert.wire import ConvertError


def test_export_metadata_interface_and_dynamic_cpu_execution(tmp_path: Path) -> None:
    source = tmp_path / "net.pb.gz"
    source.write_bytes(gzip.compress(synthetic_net(), mtime=0))
    target = tmp_path / "mouse.onnx"
    report = convert_network(source, target)
    assert report.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report.output_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    model = onnx.load(target)
    metadata = validate_model(model)
    assert metadata["mouse.schema_version"] == "1"
    assert metadata["mouse.converter_version"] == "0.2.0-dev"
    assert metadata["mouse.raw_network_format"] == "4"
    assert metadata["mouse.source_sha256"] == report.source_sha256
    assert metadata["mouse.policy_mapping"] == "lc0-1858-v1"
    assert [value.name for value in model.graph.input] == ["planes"]
    assert [value.name for value in model.graph.output] == ["policy", "wdl", "mlh"]
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    session.disable_fallback()
    assert session.get_providers() == ["CPUExecutionProvider"]
    net = parse_net(synthetic_net(), source_sha256=report.source_sha256)
    for batch in (1, 3):
        inputs = np.random.default_rng(42).normal(size=(batch, 112, 8, 8)).astype(np.float32)
        expected = numpy_forward(net, inputs)
        observed = session.run(None, {"planes": inputs})
        for wanted, actual in zip(expected, observed, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=1e-4)


def valid_model(tmp_path: Path) -> onnx.ModelProto:
    source = tmp_path / "small.pb.gz"
    target = tmp_path / "small.onnx"
    source.write_bytes(gzip.compress(synthetic_net(), mtime=0))
    convert_network(source, target)
    return onnx.load(target)


@pytest.mark.parametrize(
    "damage",
    [
        "missing_metadata",
        "duplicate_metadata",
        "bad_hash",
        "wrong_input",
        "wrong_type",
        "wrong_output",
        "wrong_shape",
        "static_batch",
        "wrong_opset",
    ],
)
def test_rejects_incompatible_or_ambiguous_export_contract(tmp_path: Path, damage: str) -> None:
    model = valid_model(tmp_path)
    if damage == "missing_metadata":
        del model.metadata_props[:]
    elif damage == "duplicate_metadata":
        model.metadata_props.add(key="mouse.schema_version", value="1")
    elif damage == "bad_hash":
        next(
            item for item in model.metadata_props if item.key == "mouse.source_sha256"
        ).value = "bad"
    elif damage == "wrong_input":
        model.graph.input[0].name = "wrong"
    elif damage == "wrong_type":
        model.graph.input[0].type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
    elif damage == "wrong_output":
        model.graph.output[0].name = "wrong"
    elif damage == "wrong_shape":
        model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1857
    elif damage == "static_batch":
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
    elif damage == "wrong_opset":
        model.opset_import[0].version = 18
    with pytest.raises(ConvertError):
        validate_model(model)


def test_failure_and_default_no_overwrite_preserve_existing_output(tmp_path: Path) -> None:
    good = tmp_path / "good.onnx"
    source = tmp_path / "bad.pb.gz"
    good.write_bytes(b"previous-good-artifact")
    source.write_bytes(gzip.compress(b"bad"))
    with pytest.raises(ConvertError):
        convert_network(source, good, overwrite=True)
    assert good.read_bytes() == b"previous-good-artifact"
    source.write_bytes(gzip.compress(synthetic_net()))
    with pytest.raises(ConvertError, match="exists"):
        convert_network(source, good)
    assert good.read_bytes() == b"previous-good-artifact"
    convert_network(source, good, overwrite=True)
    validate_model(onnx.load(good))
    assert not list(tmp_path.glob("*.tmp.onnx"))


def test_cli_requires_isolation_and_returns_machine_readable_provenance(tmp_path: Path) -> None:
    source, target = tmp_path / "net.pb.gz", tmp_path / "net.onnx"
    source.write_bytes(gzip.compress(synthetic_net(), mtime=0))
    refused = subprocess.run(
        [sys.executable, "-m", "mouse_tools.convert", str(source), str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2 and "python -I" in refused.stderr
    completed = subprocess.run(
        [sys.executable, "-I", "-m", "mouse_tools.convert", str(source), str(target)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["parameter_count"] > 0
    assert report["raw_format"]["3"] == 4 and report["upgrades"]


def test_cli_reports_every_effective_format_field(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pb.gz"
    source.write_bytes(gzip.compress(synthetic_net(), mtime=0))
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "mouse_tools.convert",
            str(source),
            str(tmp_path / "net.onnx"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["effective_format"] == {
        "input": 1,
        "output": 2,
        "network": 6,
        "policy": 3,
        "value": 2,
        "moves_left": 1,
        "default_activation": 1,
        "smolgen_activation": 7,
        "ffn_activation": 8,
        "input_embedding": 1,
    }
