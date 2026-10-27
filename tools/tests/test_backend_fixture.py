# SPDX-License-Identifier: GPL-3.0-or-later
"""The native backend fixture must detect stale inputs and preserve all heads."""

import struct
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from mouse_tools.make_backend_fixtures import generate


def session(path: Path) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    result = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    result.disable_fallback()
    assert result.get_providers() == ["CPUExecutionProvider"]
    return result


def check_outputs(model: Any, values: list[float]) -> None:
    scalars = np.asarray(values, dtype=np.float32)
    inputs = np.zeros((len(values), 112, 8, 8), dtype=np.float32)
    inputs[:, 0, 0, 0] = scalars
    policy, wdl, mlh = model.run(["policy", "wdl", "mlh"], {"planes": inputs})
    weights = (np.arange(1858, dtype=np.float32) % 17 - 8) / 16
    expected_policy = scalars[:, None] * weights[None, :]
    logits = np.stack([scalars, np.zeros_like(scalars), -scalars], axis=1)
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected_wdl = exponentials / exponentials.sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(policy, expected_policy)
    np.testing.assert_allclose(wdl, expected_wdl, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(mlh, np.maximum(scalars[:, None] + 2, 0))


def test_tracked_fixture_and_reused_input_dependence() -> None:
    model = session(Path(__file__).parents[2] / "tests/data/backend/valid.onnx")
    check_outputs(model, [0.0, 0.25, -0.5, 1.0])
    check_outputs(model, [1.0, -0.25, 0.5, -4.0])
    check_outputs(model, [0.0])


def test_generated_fixture_is_reproducible(tmp_path: Path) -> None:
    generate(tmp_path)
    committed = Path(__file__).parents[2] / "tests/data/backend"
    for name in (
        "valid.onnx",
        "nonfinite.onnx",
        "search-nonfinite.onnx",
        "wrong-precision.onnx",
        "warmup-nonfinite.onnx",
        "padding-nonfinite.onnx",
        "failure-after-valid.bin",
    ):
        assert (tmp_path / name).read_bytes() == (committed / name).read_bytes()
    check_outputs(session(tmp_path / "valid.onnx"), [0.0, 0.25, -0.5])


def test_nonfinite_fixture_is_finite_during_zero_warmup(tmp_path: Path) -> None:
    generate(tmp_path)
    model = session(tmp_path / "nonfinite.onnx")
    inputs = np.zeros((1, 112, 8, 8), dtype=np.float32)
    warm = model.run(None, {"planes": inputs})
    assert all(np.isfinite(head).all() for head in warm)
    inputs[0, 0, 0, 0] = 1
    actual = model.run(["policy", "wdl", "mlh"], {"planes": inputs})
    assert np.isinf(actual[0]).all()
    assert np.isfinite(actual[1]).all()
    assert np.isfinite(actual[2]).all()


def test_failure_after_valid_transport_fixture(tmp_path: Path) -> None:
    generate(tmp_path)
    data = (tmp_path / "failure-after-valid.bin").read_bytes()
    assert len(data) == 2704
    assert data[:16] == b"MOUSEIN1" + struct.pack("<II", 2, 112)
    rows = [
        [struct.unpack_from("<Qf", data, 16 + (row * 112 + plane) * 12) for plane in range(112)]
        for row in range(2)
    ]
    assert rows[0][0] == (1, 2.0)
    assert rows[1][0] == (1, 1.0)
    assert all(plane == (0, 0.0) for row in rows for plane in row[1:])
    committed = Path(__file__).parents[2] / "tests/data/backend/failure-after-valid.bin"
    assert data == committed.read_bytes()
    model = session(tmp_path / "nonfinite.onnx")
    inputs = np.zeros((1, 112, 8, 8), dtype=np.float32)
    inputs[0, 0, 0, 0] = 2.0
    assert all(np.isfinite(head).all() for head in model.run(None, {"planes": inputs}))
    inputs[0, 0, 0, 0] = 1.0
    assert np.isinf(model.run(["policy"], {"planes": inputs})[0]).all()


def test_search_failure_fixture_accepts_warmup_and_rejects_nonzero_rows(tmp_path: Path) -> None:
    generate(tmp_path)
    model = session(tmp_path / "search-nonfinite.onnx")
    inputs = np.zeros((2, 112, 8, 8), dtype=np.float32)
    assert all(np.isfinite(head).all() for head in model.run(None, {"planes": inputs}))
    inputs[0, 0, 0, 0] = 100.0
    inputs[1, 0, 0, 0] = -100.0
    policy, wdl, mlh = model.run(["policy", "wdl", "mlh"], {"planes": inputs})
    assert np.isinf(policy).all()
    assert np.isfinite(wdl).all()
    assert np.isfinite(mlh).all()


def test_warmup_failure_fixture_is_always_invalid(tmp_path: Path) -> None:
    generate(tmp_path)
    model = session(tmp_path / "warmup-nonfinite.onnx")
    for values in ([0.0], [0.0, 0.0, 0.0], [0.25, -0.5]):
        inputs = np.zeros((len(values), 112, 8, 8), dtype=np.float32)
        inputs[:, 0, 0, 0] = values
        policy, wdl, mlh = model.run(None, {"planes": inputs})
        assert np.isinf(policy).all()
        assert np.isfinite(wdl).all()
        assert np.isfinite(mlh).all()


def test_padding_failure_fixture_only_rejects_zero_tail_of_nonzero_batch(tmp_path: Path) -> None:
    generate(tmp_path)
    model = session(tmp_path / "padding-nonfinite.onnx")
    for batch in (1, 4, 8):
        zeros = np.zeros((batch, 112, 8, 8), dtype=np.float32)
        assert all(np.isfinite(head).all() for head in model.run(None, {"planes": zeros}))
    logical = np.zeros((3, 112, 8, 8), dtype=np.float32)
    logical[:, 0, 0, 0] = [0.25, 0.5, 0.75]
    valid = model.run(None, {"planes": logical})
    assert all(np.isfinite(head).all() for head in valid)
    physical = np.concatenate([logical, np.zeros((1, 112, 8, 8), dtype=np.float32)])
    policy, wdl, mlh = model.run(None, {"planes": physical})
    np.testing.assert_array_equal(policy[:3], valid[0])
    assert np.isinf(policy[3]).all()
    assert np.isfinite(wdl).all()
    assert np.isfinite(mlh).all()
