# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate original input-dependent three-head fixtures for native Backend tests."""

from __future__ import annotations

import argparse
import hashlib
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def generate(destination: Path) -> None:
    """Emit dynamic per-row heads, plus deliberate failure fixtures."""
    weights = ((np.arange(1858, dtype=np.float32) % 17 - 8) / 16).reshape(1, 1858)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, nonfinite, precision in (
        ("valid.onnx", False, "fp32"),
        ("nonfinite.onnx", True, "fp32"),
        ("search-nonfinite.onnx", True, "fp32"),
        ("wrong-precision.onnx", False, "unsupported"),
        ("warmup-nonfinite.onnx", True, "fp32"),
        ("padding-nonfinite.onnx", True, "fp32"),
    ):
        policy_name = "policy_base" if nonfinite else "policy"
        nodes = [
            helper.make_node("ReduceSum", ["planes", "sum_axes"], ["sum"], keepdims=0),
            helper.make_node("Unsqueeze", ["sum", "axis_one"], ["scalar"]),
            helper.make_node("MatMul", ["scalar", "policy_weights"], [policy_name]),
            helper.make_node("MatMul", ["scalar", "wdl_weights"], ["wdl_logits"]),
            helper.make_node("Softmax", ["wdl_logits"], ["wdl"], axis=1),
            helper.make_node("Add", ["scalar", "mlh_bias"], ["mlh_unclamped"]),
            helper.make_node("Relu", ["mlh_unclamped"], ["mlh"]),
        ]
        initializers = [
            numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64), "sum_axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_one"),
            numpy_helper.from_array(weights, "policy_weights"),
            numpy_helper.from_array(np.array([[1, 0, -1]], dtype=np.float32), "wdl_weights"),
            numpy_helper.from_array(np.array([[2]], dtype=np.float32), "mlh_bias"),
        ]
        if nonfinite:
            if filename not in ("search-nonfinite.onnx", "warmup-nonfinite.onnx"):
                initializers.append(
                    numpy_helper.from_array(np.array([[1]], dtype=np.float32), "one")
                )
            if filename == "warmup-nonfinite.onnx":
                nodes.append(helper.make_node("Sub", ["scalar", "scalar"], ["denominator"]))
            elif filename == "padding-nonfinite.onnx":
                initializers.extend(
                    [
                        numpy_helper.from_array(np.array([[0]], dtype=np.float32), "zero"),
                        numpy_helper.from_array(np.array([0], dtype=np.int64), "axis_zero"),
                    ]
                )
                nodes.extend(
                    [
                        helper.make_node("Abs", ["scalar"], ["absolute"]),
                        helper.make_node(
                            "ReduceSum", ["absolute", "axis_zero"], ["batch_sum"], keepdims=1
                        ),
                        helper.make_node("Greater", ["batch_sum", "zero"], ["batch_nonzero"]),
                        helper.make_node("Equal", ["scalar", "zero"], ["row_zero"]),
                        helper.make_node("And", ["batch_nonzero", "row_zero"], ["invalid"]),
                        helper.make_node(
                            "Cast", ["invalid"], ["invalid_float"], to=TensorProto.FLOAT
                        ),
                        helper.make_node("Sub", ["one", "invalid_float"], ["denominator"]),
                    ]
                )
            elif filename == "search-nonfinite.onnx":
                initializers.append(
                    numpy_helper.from_array(np.array([[0]], dtype=np.float32), "zero")
                )
                nodes.extend(
                    [
                        helper.make_node("Equal", ["scalar", "zero"], ["is_zero"]),
                        helper.make_node(
                            "Cast", ["is_zero"], ["denominator"], to=TensorProto.FLOAT
                        ),
                    ]
                )
            else:
                nodes.append(helper.make_node("Sub", ["one", "scalar"], ["denominator"]))
            nodes.extend(
                [
                    helper.make_node("Reciprocal", ["denominator"], ["offset"]),
                    helper.make_node("Add", ["policy_base", "offset"], ["policy"]),
                ]
            )
        graph = helper.make_graph(
            nodes,
            "mouse_backend_fixture_" + filename,
            [helper.make_tensor_value_info("planes", TensorProto.FLOAT, ["batch", 112, 8, 8])],
            [
                helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width])
                for name, width in (("policy", 1858), ("wdl", 3), ("mlh", 1))
            ],
            initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            ir_version=8,
            producer_name="Mouse original backend fixture v1",
        )
        helper.set_model_props(
            model,
            {
                "mouse.schema_version": "1",
                "mouse.converter_version": "backend-fixture-v1",
                "mouse.source_sha256": hashlib.sha256(
                    b"Mouse original backend fixture v1"
                ).hexdigest(),
                "mouse.input_format": "1",
                "mouse.network_format": "123",
                "mouse.raw_network_format": "123",
                "mouse.policy_format": "3",
                "mouse.value_format": "2",
                "mouse.moves_left_format": "1",
                "mouse.policy_mapping": "lc0-1858-v1",
                "mouse.learned_parameter_count": "1862",
                "mouse.precision": precision,
            },
        )
        onnx.checker.check_model(model)
        onnx.save(model, destination / filename)

    # First chunk is valid; the second produces Inf in nonfinite.onnx. This
    # exercises CLI atomic publication across separately evaluated batches.
    rows = [[(1, value)] + [(0, 0.0)] * 111 for value in (2.0, 1.0)]
    transport = b"MOUSEIN1" + struct.pack("<II", 2, 112)
    transport += b"".join(struct.pack("<Qf", mask, value) for row in rows for mask, value in row)
    (destination / "failure-after-valid.bin").write_bytes(transport)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    generate(arguments.output_dir)
    print(f"Generated original backend fixtures in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
