# SPDX-License-Identifier: GPL-3.0-or-later
"""Original missing-head net proves the native three-head rejection boundary."""

from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from mouse_tools.make_metadata_fixtures import generate


def test_missing_mlh_fixture_is_valid_two_head_onnx_and_reproducible(tmp_path: Path) -> None:
    generate(tmp_path)
    generated = tmp_path / "missing-mlh.onnx"
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    model = ort.InferenceSession(
        str(generated), sess_options=options, providers=["CPUExecutionProvider"]
    )
    model.disable_fallback()
    assert [output.name for output in model.get_outputs()] == ["policy", "wdl"]
    outputs = model.run(None, {"planes": np.zeros((2, 112, 8, 8), dtype=np.float32)})
    assert [output.shape for output in outputs] == [(2, 1858), (2, 3)]
    assert all(np.isfinite(output).all() for output in outputs)
    fixtures = Path(__file__).resolve().parents[2] / "tests/data/model-metadata"
    for fixture in fixtures.glob("*.onnx"):
        assert (tmp_path / fixture.name).read_bytes() == fixture.read_bytes()
    assert generated.read_bytes() == (fixtures / generated.name).read_bytes()
