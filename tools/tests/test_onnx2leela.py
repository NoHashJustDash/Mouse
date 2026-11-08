# SPDX-License-Identifier: GPL-3.0-or-later
"""onnx2leela writes what `lc0 onnx2leela` writes, and refuses what it refuses."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from mouse_tools.convert.wire import byte_field, int_field, message
from mouse_tools.onnx2leela import Names, Onnx2LeelaError, main, weights_bytes

NAMES = Names(input_planes="planes", policy="policy", wdl="wdl", mlh="mlh")


def _model(elem: int = TensorProto.FLOAT) -> bytes:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["planes"], ["policy"]),
            helper.make_node("Identity", ["planes"], ["wdl"]),
            helper.make_node("Identity", ["planes"], ["mlh"]),
        ],
        "net",
        [helper.make_tensor_value_info("planes", elem, [None, 112, 8, 8])],
        [
            helper.make_tensor_value_info("policy", elem, [None, 112, 8, 8]),
            helper.make_tensor_value_info("wdl", elem, [None, 112, 8, 8]),
            helper.make_tensor_value_info("mlh", elem, [None, 112, 8, 8]),
        ],
    )
    return helper.make_model(graph).SerializeToString()


def test_message_layout() -> None:
    raw = _model()
    root = message(weights_bytes(raw, NAMES), {1: 5, 3: 2, 4: 2, 11: 2})
    assert struct.unpack("<I", byte_field(root, 1))[0] == 0x1C0
    version = message(byte_field(root, 3), {1: 0, 2: 0})
    assert int_field(version, 1) == 0
    assert int_field(version, 2) == 28
    network = message(
        byte_field(message(byte_field(root, 4), {2: 2}), 2), {i: 0 for i in range(1, 7)}
    )
    assert [int_field(network, i) for i in range(1, 7)] == [1, 2, 5, 1, 2, 1]
    model = message(byte_field(root, 11), {1: 2, 2: 0, 3: 2, 5: 2, 6: 2, 7: 2})
    assert bytes(byte_field(model, 1)) == raw
    assert int_field(model, 2) == 1
    assert bytes(byte_field(model, 3)) == b"planes"
    assert bytes(byte_field(model, 5)) == b"wdl"
    assert bytes(byte_field(model, 6)) == b"policy"
    assert bytes(byte_field(model, 7)) == b"mlh"


def test_fp16_input_sets_the_data_type() -> None:
    root = message(weights_bytes(_model(TensorProto.FLOAT16), NAMES), {1: 5, 3: 2, 4: 2, 11: 2})
    model = message(byte_field(root, 11), {1: 2, 2: 0, 3: 2, 5: 2, 6: 2, 7: 2})
    assert int_field(model, 2) == 10


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (Names(policy="policy", wdl="wdl"), "--onnx-input"),
        (Names(input_planes="planes", wdl="wdl"), "--onnx-output-policy"),
        (Names(input_planes="planes", policy="policy"), "--onnx-output-value"),
        (Names(input_planes="missing", policy="policy", wdl="wdl"), "does not exist"),
        (Names(input_planes="planes", policy="missing", wdl="wdl"), "does not exist"),
        (Names(input_planes="planes", policy="policy", wdl="wdl", value="mlh"), "only one"),
    ],
)
def test_validation(names: Names, expected: str) -> None:
    with pytest.raises(Onnx2LeelaError, match=expected):
        weights_bytes(_model(), names)


def test_fixes_rewrite_the_model() -> None:
    raw = _model()
    fixed = weights_bytes(raw, NAMES, fix_rule50=True, fix_wdl_softmax=True)
    root = message(fixed, {1: 5, 3: 2, 4: 2, 11: 2})
    model = onnx.load_from_string(
        bytes(byte_field(message(byte_field(root, 11), {1: 2, 2: 0, 3: 2, 5: 2, 6: 2, 7: 2}), 1))
    )
    kinds = {node.op_type for node in model.graph.node}
    assert {"Mul", "Softmax"} <= kinds
    assert any(item.name == "rule50fix_weights" for item in model.graph.initializer)


def test_command_line_writes_gzip(tmp_path: Path) -> None:
    source = tmp_path / "net.onnx"
    source.write_bytes(_model())
    output = tmp_path / "net.pb.gz"
    arguments = ["--input", str(source), "--output", str(output), "--onnx-input", "planes"]
    assert main([*arguments, "--onnx-output-policy", "policy", "--onnx-output-wdl", "wdl"]) == 0
    assert gzip.decompress(output.read_bytes())[:5] == b"\x0d" + struct.pack("<I", 0x1C0)
    assert main(arguments) == 2
