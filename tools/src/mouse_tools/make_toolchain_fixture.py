# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate Mouse's small SDK acceptance fixture, independently of any net."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def generate_fixture(destination: Path) -> None:
    weights = (np.arange(512, dtype=np.float32).reshape(32, 16) % 17 - 8) / 16
    bias = np.arange(16, dtype=np.float32) / 16
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weights"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["biased"]),
            helper.make_node("Relu", ["biased"], ["output"]),
        ],
        "mouse_toolchain_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 32])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [4, 16])],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Mouse's MatMul/Add/Relu ONNX fixture")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    generate_fixture(arguments.output)
    print(f"Generated {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
