# SPDX-License-Identifier: GPL-3.0-or-later
"""The generated SDK fixture must retain its exact numerical contract."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]  # ORT 1.22 publishes no type marker.
from numpy.typing import NDArray

from mouse_tools.make_toolchain_fixture import generate_fixture

# Independent reference captured by the accepted toolchain fixture, using integer
# multiples of 1/256 so each result is represented exactly by float32.
EXPECTED = (
    np.array(
        [
            [0, 68, 223, 344, 431, 484, 503, 488, 439, 356, 239, 88, 0, 0, 0, 0],
            [295, 420, 511, 568, 591, 580, 535, 456, 343, 196, 15, 0, 0, 0, 0, 0],
            [711, 772, 799, 792, 751, 676, 567, 424, 247, 36, 0, 0, 0, 0, 0, 0],
            [1127, 1124, 1087, 1016, 911, 772, 599, 392, 151, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    / 256
)


def evaluate(path: Path) -> NDArray[np.float32]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    session.disable_fallback()
    assert session.get_providers() == ["CPUExecutionProvider"]
    inputs = np.arange(128, dtype=np.float32).reshape(4, 32) / 16
    return cast(NDArray[np.float32], session.run(["output"], {"input": inputs})[0])


def test_committed_fixture_matches_reference() -> None:
    path = Path(__file__).parent / "data" / "toolchain.onnx"
    np.testing.assert_array_equal(evaluate(path), EXPECTED)


def test_generated_fixture_matches_reference(tmp_path: Path) -> None:
    path = tmp_path / "generated.onnx"
    generate_fixture(path)
    np.testing.assert_array_equal(evaluate(path), EXPECTED)
