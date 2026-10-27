# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate a tiny ONNX file requesting a real, input-dependent CUDA allocation.

The default requests one 64GiB float32 intermediate. Never execute it on CPU.
Only the explicitly small analogue belongs in CPU numerical tests. Generation
itself constructs fixed-size constants and never allocates the stress tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MAX_ELEMENTS = 1 << 34
DEFAULT_ELEMENTS = MAX_ELEMENTS


def generate(output: Path, elements: int = DEFAULT_ELEMENTS) -> None:
    """Write a deterministic graph; its large intermediate exists only at Run."""
    if (
        isinstance(elements, bool)
        or not isinstance(elements, int)
        or not 1 <= elements <= MAX_ELEMENTS
    ):
        raise ValueError(f"elements must be an integer in 1..{MAX_ELEMENTS}")
    # Every head depends on this chain, preventing dead-code removal. Sin makes
    # Expand->ReduceSum algebraic elimination invalid. All operators have CUDA
    # kernels in the pinned ORT1.22 registry; the final native probe must still
    # observe an actual allocation error rather than assume one from the shape.
    # https://github.com/microsoft/onnxruntime/blob/v1.22.0/onnxruntime/core/providers/cuda/cuda_execution_provider.cc
    nodes = [
        helper.make_node(
            "ReduceSum",
            ["planes", "source_axes"],
            ["source_sum"],
            name="source_summary",
            keepdims=0,
        ),
        helper.make_node(
            "Unsqueeze", ["source_sum", "axis_one"], ["source_row"], name="source_row"
        ),
        helper.make_node(
            "Expand", ["source_row", "stress_shape"], ["stress_expanded"], name="stress_expand"
        ),
        helper.make_node("Sin", ["stress_expanded"], ["stress_sine"], name="stress_sin"),
        helper.make_node(
            "ReduceSum",
            ["stress_sine", "axis_one"],
            ["stress_total"],
            name="stress_reduce",
            keepdims=1,
        ),
        helper.make_node("Mul", ["stress_total", "mean_scale"], ["mean_sine"], name="stress_mean"),
        helper.make_node("MatMul", ["mean_sine", "policy_weights"], ["policy"], name="policy_head"),
        helper.make_node("MatMul", ["mean_sine", "wdl_weights"], ["wdl_logits"], name="wdl_logits"),
        helper.make_node("Softmax", ["wdl_logits"], ["wdl"], axis=1, name="wdl_head"),
        helper.make_node("Add", ["mean_sine", "mlh_bias"], ["mlh_unclamped"], name="mlh_bias"),
        helper.make_node("Relu", ["mlh_unclamped"], ["mlh"], name="mlh_head"),
    ]
    weights = ((np.arange(1858, dtype=np.float32) % 17 - 8) / 16).reshape(1, 1858)
    constants = [
        numpy_helper.from_array(np.asarray([1, 2, 3], dtype=np.int64), "source_axes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis_one"),
        numpy_helper.from_array(np.asarray([1, elements], dtype=np.int64), "stress_shape"),
        numpy_helper.from_array(np.asarray([[1.0 / elements]], dtype=np.float32), "mean_scale"),
        numpy_helper.from_array(weights, "policy_weights"),
        numpy_helper.from_array(np.asarray([[1, 0, -1]], dtype=np.float32), "wdl_weights"),
        numpy_helper.from_array(np.asarray([[2]], dtype=np.float32), "mlh_bias"),
    ]
    graph = helper.make_graph(
        nodes,
        "mouse_original_allocation_stress_v1",
        [helper.make_tensor_value_info("planes", TensorProto.FLOAT, ["batch", 112, 8, 8])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", width])
            for name, width in (("policy", 1858), ("wdl", 3), ("mlh", 1))
        ],
        constants,
        value_info=[
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, elements])
            for name in ("stress_expanded", "stress_sine")
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
        producer_name="Mouse original OOM fixture v1",
    )
    provenance = f"Mouse original OOM fixture v1 elements={elements}".encode("ascii")
    helper.set_model_props(
        model,
        {
            "mouse.schema_version": "1",
            "mouse.converter_version": "oom-fixture-v1",
            "mouse.source_sha256": hashlib.sha256(provenance).hexdigest(),
            "mouse.input_format": "1",
            "mouse.network_format": "123",
            "mouse.raw_network_format": "123",
            "mouse.policy_format": "3",
            "mouse.value_format": "2",
            "mouse.moves_left_format": "1",
            "mouse.policy_mapping": "lc0-1858-v1",
            "mouse.learned_parameter_count": "1862",
            "mouse.precision": "fp32",
        },
    )
    onnx.checker.check_model(model, full_check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=DEFAULT_ELEMENTS)
    arguments = parser.parse_args(argv)
    try:
        generate(arguments.output, arguments.elements)
        data = arguments.output.read_bytes()
    except (OSError, ValueError) as error:
        print(f"OOM fixture generation failed: {error}", file=sys.stderr)
        return 2
    print(
        f"OOM_FIXTURE elements={arguments.elements} "
        f"planned_float32_bytes={arguments.elements * 4} model_bytes={len(data)} "
        f"sha256={hashlib.sha256(data).hexdigest()} output={arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
