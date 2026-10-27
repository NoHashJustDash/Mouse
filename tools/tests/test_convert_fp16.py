# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]  # ORT 1.22 has no type marker.
import pytest
import torch
from test_convert_fixture import synthetic_net
from test_convert_model import numpy_forward

from mouse_tools.convert.export import convert_network, validate_model
from mouse_tools.convert.model import T1Model
from mouse_tools.convert.schema import parse_net
from mouse_tools.convert.wire import ConvertError


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "small.pb.gz"
    source.write_bytes(gzip.compress(synthetic_net(), mtime=0))
    return source


def test_half_model_uses_half_parameters_and_float_interface() -> None:
    torch.set_num_threads(1)
    net = parse_net(synthetic_net(), source_sha256="a" * 64)
    model = T1Model(net, precision="fp16").eval()
    assert {item.dtype for item in model.parameters()} == {torch.float16}
    assert model.position_map.dtype == torch.float16
    assert model.policy.mapping.dtype == torch.int64
    assert sum(item.numel() for item in model.parameters()) == net.parameter_count
    for batch in (1, 3):
        planes = np.random.default_rng(2718).normal(size=(batch, 112, 8, 8)).astype(np.float32)
        with torch.inference_mode():
            actual = model(torch.from_numpy(planes))
        expected = numpy_forward(net, planes)
        for observed, wanted in zip(actual, expected, strict=True):
            assert observed.dtype == torch.float32 and bool(torch.isfinite(observed).all())
            # A separate small-model half-precision diagnostic; the existing FP32
            # oracle and full conversion gate retain their unchanged 1e-4 bound.
            np.testing.assert_allclose(observed.numpy(), wanted, rtol=0, atol=0.005)
        assert [tuple(item.shape) for item in actual] == [(batch, 1858), (batch, 3), (batch, 1)]
        np.testing.assert_allclose(actual[1].sum(1).numpy(), np.ones(batch), rtol=0, atol=0.0005)
        assert bool((actual[2] >= 0).all())


def test_half_export_has_half_matmul_float_io_and_dynamic_execution(tmp_path: Path) -> None:
    source, target = _source(tmp_path), tmp_path / "half.onnx"
    report = convert_network(source, target, precision="fp16")
    assert report.precision == "fp16"
    graph = onnx.load(target)
    assert validate_model(graph)["mouse.precision"] == "fp16"
    floating = [item for item in graph.graph.initializer if item.data_type in (1, 10, 11, 16)]
    assert floating and {item.data_type for item in floating} == {onnx.TensorProto.FLOAT16}
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    types = {
        item.name: item.type.tensor_type.elem_type
        for item in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]
    }
    types.update({item.name: item.data_type for item in inferred.graph.initializer})
    matmuls = [item for item in inferred.graph.node if item.op_type == "MatMul"]
    assert matmuls and all(
        types[name] == onnx.TensorProto.FLOAT16 for n in matmuls for name in n.input
    )
    assert all(
        item.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for item in [*graph.graph.input, *graph.graph.output]
    )
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    session.disable_fallback()
    assert session.get_providers() == ["CPUExecutionProvider"]
    net = parse_net(synthetic_net(), source_sha256=report.source_sha256)
    for batch in (1, 3):
        planes = np.random.default_rng(618).normal(size=(batch, 112, 8, 8)).astype(np.float32)
        observed = session.run(None, {"planes": planes})
        for actual, expected in zip(observed, numpy_forward(net, planes), strict=True):
            assert actual.dtype == np.float32 and np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0.005)
        assert [item.shape for item in observed] == [(batch, 1858), (batch, 3), (batch, 1)]


def test_fp32_default_explicit_and_legacy_precision_contract(tmp_path: Path) -> None:
    source = _source(tmp_path)
    default, explicit = tmp_path / "default.onnx", tmp_path / "explicit.onnx"
    assert convert_network(source, default).precision == "fp32"
    assert convert_network(source, explicit, precision="fp32").precision == "fp32"
    assert default.read_bytes() == explicit.read_bytes()
    graph = onnx.load(default)
    assert validate_model(graph)["mouse.precision"] == "fp32"
    legacy = [item for item in graph.metadata_props if item.key != "mouse.precision"]
    del graph.metadata_props[:]
    graph.metadata_props.extend(legacy)
    assert validate_model(graph)["mouse.precision"] == "fp32"


@pytest.mark.parametrize("precision", ["fp16", "bf16", "FP32", ""])
def test_rejects_false_or_unknown_precision_metadata(tmp_path: Path, precision: str) -> None:
    target = tmp_path / "float.onnx"
    convert_network(_source(tmp_path), target)
    graph = onnx.load(target)
    next(item for item in graph.metadata_props if item.key == "mouse.precision").value = precision
    with pytest.raises(ConvertError, match="precision"):
        validate_model(graph)


def test_half_without_precision_metadata_is_not_accepted_as_legacy_float(tmp_path: Path) -> None:
    target = tmp_path / "half.onnx"
    convert_network(_source(tmp_path), target, precision="fp16")
    graph = onnx.load(target)
    retained = [item for item in graph.metadata_props if item.key != "mouse.precision"]
    del graph.metadata_props[:]
    graph.metadata_props.extend(retained)
    with pytest.raises(ConvertError, match="precision"):
        validate_model(graph)


def test_unsupported_precision_preserves_output(tmp_path: Path) -> None:
    target = tmp_path / "existing.onnx"
    target.write_bytes(b"previous-good-artifact")
    with pytest.raises(ConvertError, match="precision"):
        convert_network(_source(tmp_path), target, precision="bf16", overwrite=True)
    assert target.read_bytes() == b"previous-good-artifact"
    assert not list(tmp_path.glob("*.tmp.onnx"))


def test_half_rejects_unrepresentable_finite_weights_and_input() -> None:
    net = parse_net(synthetic_net(), source_sha256="a" * 64)
    model = T1Model(net, precision="fp16")
    with pytest.raises(ConvertError, match="input"):
        model(torch.full((1, 112, 8, 8), 1e10, dtype=torch.float32))
    damaged = dict(net.tensors)
    damaged["ip_emb_w"] = np.full_like(damaged["ip_emb_w"], 1e10)
    from dataclasses import replace

    with pytest.raises(ConvertError, match="finite.*fp16|fp16.*finite"):
        T1Model(replace(net, tensors=damaged), precision="fp16")


def test_isolated_cli_selects_precision_and_reports_it(tmp_path: Path) -> None:
    source, target = _source(tmp_path), tmp_path / "cli-half.onnx"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "mouse_tools.convert",
            str(source),
            str(target),
            "--precision",
            "fp16",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["precision"] == "fp16"
    assert validate_model(onnx.load(target))["mouse.precision"] == "fp16"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "mouse_tools.convert",
            str(source),
            str(target),
            "--precision",
            "bf16",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2 and "invalid choice" in result.stderr
    assert "Traceback" not in result.stderr
