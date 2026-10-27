# SPDX-License-Identifier: GPL-3.0-or-later
"""Only the tiny analogue executes; the 64GiB graph is inspected, never run."""

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort  # type: ignore[import-untyped]
import pytest
from onnx import TensorProto, numpy_helper

from mouse_tools.make_oom_fixture import DEFAULT_ELEMENTS, MAX_ELEMENTS, generate, main


def test_large_fixture_is_bounded_reproducible_and_preserves_batch_symbol(tmp_path: Path) -> None:
    assert DEFAULT_ELEMENTS == MAX_ELEMENTS == 1 << 34
    output = tmp_path / "oom.onnx"
    generate(output)
    data = output.read_bytes()
    assert len(data) < 16 * 1024
    assert data == (Path(__file__).parents[2] / "tests/data/backend/oom.onnx").read_bytes()
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model, full_check=True)
    assert model.ir_version == 8
    assert [(entry.domain, entry.version) for entry in model.opset_import] == [("", 17)]
    assert all(math.prod(tensor.dims) <= 1858 for tensor in model.graph.initializer)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    shape = numpy_helper.to_array(tensors["stress_shape"])
    assert shape.dtype == np.int64
    np.testing.assert_array_equal(shape, [1, 1 << 34])
    planned_bytes = math.prod(int(value) for value in shape) * np.dtype(np.float32).itemsize
    assert planned_bytes == 68_719_476_736 == 64 * 1024**3
    assert len(tensors["stress_shape"].raw_data) == 16
    assert all(not tensor.external_data for tensor in model.graph.initializer)
    assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    for output_info, width in zip(model.graph.output, (1858, 3, 1), strict=True):
        assert output_info.type.tensor_type.elem_type == TensorProto.FLOAT
        assert output_info.type.tensor_type.shape.dim[0].dim_param == "batch"
        assert output_info.type.tensor_type.shape.dim[1].dim_value == width
    nodes = {node.name: node for node in model.graph.node}
    assert nodes["stress_expand"].op_type == "Expand"
    assert list(nodes["stress_expand"].input) == ["source_row", "stress_shape"]
    assert list(nodes["stress_sin"].input) == list(nodes["stress_expand"].output)
    assert nodes["stress_sin"].op_type == "Sin"
    assert list(nodes["stress_reduce"].input)[0] == nodes["stress_sin"].output[0]
    assert nodes["stress_reduce"].op_type == "ReduceSum"


def test_small_analogue_executes_input_dependent_cpu_heads_without_eliminating_stress(
    tmp_path: Path,
) -> None:
    path = tmp_path / "small.onnx"
    generate(path, elements=16)
    # Refuse CPU execution before creating any ORT session if generation ever
    # regresses to the dangerous default instead of honoring the small size.
    small_model = onnx.load(path)
    small_shape = next(
        tensor for tensor in small_model.graph.initializer if tensor.name == "stress_shape"
    )
    np.testing.assert_array_equal(numpy_helper.to_array(small_shape), [1, 16])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.optimized_model_filepath = str(tmp_path / "optimized.onnx")
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert [entry.name for entry in session.get_outputs()] == ["policy", "wdl", "mlh"]
    metadata = session.get_modelmeta().custom_metadata_map
    assert metadata["mouse.precision"] == "fp32"
    assert metadata["mouse.schema_version"] == "1"
    assert metadata["mouse.input_format"] == "1"
    assert metadata["mouse.policy_format"] == "3"
    assert metadata["mouse.value_format"] == "2"
    assert metadata["mouse.moves_left_format"] == "1"
    assert metadata["mouse.policy_mapping"] == "lc0-1858-v1"
    assert metadata["mouse.learned_parameter_count"] == "1862"
    assert (
        metadata["mouse.source_sha256"]
        == hashlib.sha256(b"Mouse original OOM fixture v1 elements=16").hexdigest()
    )
    optimized = onnx.load(options.optimized_model_filepath)
    assert {"Expand", "Sin", "ReduceSum"}.issubset({node.op_type for node in optimized.graph.node})
    previous: Any = None
    for scalar in (0.0, 0.25, -0.5, 1.0):
        values = np.zeros((1, 112, 8, 8), dtype=np.float32)
        values[0, 0, 0, 0] = scalar
        policy, wdl, mlh = session.run(["policy", "wdl", "mlh"], {"planes": values})
        assert policy.shape == (1, 1858)
        assert wdl.shape == (1, 3)
        assert mlh.shape == (1, 1)
        assert all(
            head.dtype == np.float32 and np.isfinite(head).all() for head in (policy, wdl, mlh)
        )
        expected_scalar = math.sin(scalar)
        weights = (np.arange(1858, dtype=np.float64) % 17 - 8) / 16
        np.testing.assert_allclose(policy[0], expected_scalar * weights, rtol=0, atol=1e-7)
        logits = np.asarray([expected_scalar, 0.0, -expected_scalar])
        probabilities = np.exp(logits) / np.exp(logits).sum()
        np.testing.assert_allclose(wdl[0], probabilities, rtol=0, atol=1e-7)
        np.testing.assert_allclose(mlh, [[max(expected_scalar + 2.0, 0.0)]], rtol=0, atol=1e-6)
        if previous is not None:
            assert not np.array_equal(previous, policy)
        previous = policy.copy()


@pytest.mark.parametrize("elements", [0, -1, (1 << 34) + 1, True, 16.0, "16", None])
def test_generator_rejects_invalid_element_counts_before_creating_output(
    tmp_path: Path, elements: Any
) -> None:
    path = tmp_path / "not-created" / "invalid.onnx"
    with pytest.raises(ValueError, match="integer in 1..17179869184"):
        generate(path, elements=elements)
    assert not path.parent.exists()


def test_cli_reports_actual_file_size_and_planned_allocation(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "cli.onnx"
    assert main(["--output", str(path), "--elements", "16"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "elements=16" in captured.out
    assert "planned_float32_bytes=64" in captured.out
    assert f"model_bytes={path.stat().st_size}" in captured.out
    assert "sha256=" + hashlib.sha256(path.read_bytes()).hexdigest() in captured.out
    invalid = tmp_path / "invalid.onnx"
    assert main(["--output", str(invalid), "--elements", "0"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "integer in 1..17179869184" in captured.err
    assert not invalid.exists()
