# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate original tiny ONNX interface fixtures; these are not chess networks."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def generate(destination: Path) -> None:
    """All outputs depend on the dynamic input batch; no external net is read."""
    metadata = {
        "mouse.schema_version": "1",
        "mouse.converter_version": "0.2.0-dev",
        "mouse.source_sha256": hashlib.sha256(b"Mouse scalar metadata fixture v1").hexdigest(),
        "mouse.input_format": "1",
        "mouse.network_format": "6",
        "mouse.raw_network_format": "4",
        "mouse.policy_format": "3",
        "mouse.value_format": "2",
        "mouse.moves_left_format": "1",
        "mouse.policy_mapping": "lc0-1858-v1",
        "mouse.learned_parameter_count": "1",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for filename, include_metadata, policy_width, input_type in (
        ("valid.onnx", True, 1858, TensorProto.FLOAT),
        ("missing-mlh.onnx", True, 1858, TensorProto.FLOAT),
        ("missing-metadata.onnx", False, 1858, TensorProto.FLOAT),
        ("wrong-policy-shape.onnx", True, 17, TensorProto.FLOAT),
        ("wrong-input-type.onnx", True, 1858, TensorProto.DOUBLE),
    ):
        nodes = [
            helper.make_node("Shape", ["planes"], ["input_shape"]),
            helper.make_node("Gather", ["input_shape", "zero_index"], ["batch"], axis=0),
            helper.make_node("Unsqueeze", ["batch", "axis_zero"], ["batch_vector"]),
        ]
        initializers = [
            helper.make_tensor("zero_index", TensorProto.INT64, [], [0]),
            helper.make_tensor("axis_zero", TensorProto.INT64, [1], [0]),
            helper.make_tensor("scalar_weight", TensorProto.FLOAT, [1, 1], [0.0]),
            helper.make_tensor("wdl_row", TensorProto.FLOAT, [1, 3], [1.0, 0.0, 0.0]),
        ]
        outputs = []
        for name, width, source in (
            ("policy", policy_width, "scalar_weight"),
            ("wdl", 3, "wdl_row"),
            ("mlh", 1, "scalar_weight"),
        ):
            if filename == "missing-mlh.onnx" and name == "mlh":
                continue
            initializers.append(
                helper.make_tensor(name + "_width", TensorProto.INT64, [1], [width])
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Concat", ["batch_vector", name + "_width"], [name + "_shape"], axis=0
                    ),
                    helper.make_node("Expand", [source, name + "_shape"], [name]),
                ]
            )
            outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width]))
        graph = helper.make_graph(
            nodes,
            "mouse_metadata_fixture",
            [helper.make_tensor_value_info("planes", input_type, ["batch", 112, 8, 8])],
            outputs,
            initializers,
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        if include_metadata:
            helper.set_model_props(model, metadata)
        onnx.checker.check_model(model)
        onnx.save(model, destination / filename)
    (destination / "not-onnx.bin").write_bytes(b"Mouse invalid ONNX fixture\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    generate(arguments.output_dir)
    print(f"Generated metadata fixtures in {arguments.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
