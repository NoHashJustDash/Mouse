# SPDX-License-Identifier: GPL-3.0-or-later
"""Wrap an ONNX network in a Leela weights file, as `lc0 onnx2leela` does.

Behaviour follows lc0 fd71a2d src/tools/onnx2leela.cc. Field numbers are the ones lc0's own output
uses; tools/tests/test_onnx2leela.py checks the file against that output.
"""

from __future__ import annotations

import argparse
import gzip
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper

MAGIC = 0x1C0
# lc0's generated proto enums, from pblczero::NetworkFormat.
INPUT_FORMATS = {
    "INPUT_UNKNOWN": 0,
    "INPUT_CLASSICAL_112_PLANE": 1,
    "INPUT_112_WITH_CASTLING_PLANE": 2,
    "INPUT_112_WITH_CANONICALIZATION": 3,
    "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES": 4,
    "INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON": 132,
    "INPUT_112_WITH_CANONICALIZATION_V2": 5,
    "INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON": 133,
}
POLICY_FORMATS = {
    "POLICY_UNKNOWN": 0,
    "POLICY_CLASSICAL": 1,
    "POLICY_CONVOLUTION": 2,
    "POLICY_ATTENTION": 3,
}
VALUE_FORMATS = {"VALUE_UNKNOWN": 0, "VALUE_CLASSICAL": 1, "VALUE_WDL": 2, "VALUE_PARAM": 3}
MOVES_LEFT_FORMATS = {"MOVES_LEFT_NONE": 0, "MOVES_LEFT_V1": 1}
NETWORK_ONNX = 5
OUTPUT_CLASSICAL = 1
OUTPUT_WDL = 2
DATA_FLOAT = 1
DATA_FLOAT16 = 10


class Onnx2LeelaError(ValueError):
    """The network or the options cannot make a valid weights file."""


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _int_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _text_field(number: int, value: str) -> bytes:
    return _bytes_field(number, value.encode("utf-8"))


@dataclass(frozen=True)
class Names:
    """ONNX node names, each optional as on lc0's command line."""

    input_planes: str | None = None
    policy: str | None = None
    value: str | None = None
    wdl: str | None = None
    mlh: str | None = None


@dataclass(frozen=True)
class Formats:
    """Lc0's --input-format, --policy-format, --value-format and --moves-left-format."""

    input: str = "INPUT_CLASSICAL_112_PLANE"
    policy: str = "POLICY_CLASSICAL"
    value: str = "VALUE_WDL"
    moves_left: str = "MOVES_LEFT_V1"


DEFAULT_FORMATS = Formats()


def _data_type(model: onnx.ModelProto, name: str | None) -> int:
    if name is not None:
        for item in model.graph.input:
            if item.name == name and item.type.tensor_type.elem_type:
                elem = item.type.tensor_type.elem_type
                if elem == onnx.TensorProto.FLOAT:
                    return DATA_FLOAT
                if elem == onnx.TensorProto.FLOAT16:
                    return DATA_FLOAT16
                raise Onnx2LeelaError(f"unsupported input data type: {elem}")
    return DATA_FLOAT


def _ensure_output_type(model: onnx.ModelProto, name: str, data_type: int) -> bool:
    """lc0 EnsureOutDataType: fix the declared type, or Cast an output that has none."""
    for output in model.graph.output:
        if output.name == name:
            tensor = output.type.tensor_type
            if tensor.elem_type == data_type:
                return False
            tensor.elem_type = data_type
            break
    for node in model.graph.node:
        for index, produced in enumerate(node.output):
            if produced == name:
                node.output[index] = name + "/cast"
                break
    model.graph.node.append(
        onnx.helper.make_node("Cast", [name + "/cast"], [name], name=name + "/cast", to=data_type)
    )
    return True


def _fix_rule50(model: onnx.ModelProto, name: str, fp16: bool) -> None:
    """lc0 FixRule50: scale the rule-50 plane of a TensorFlow export down by 99."""
    fix = "rule50fix"
    for node in model.graph.node:
        for index, consumed in enumerate(node.input):
            if consumed == name:
                node.input[index] = fix
    weights = [1.0] * 112
    weights[109] = 1.0 / 99
    data = onnx.numpy_helper.from_array(
        np.array(weights, dtype=np.float16 if fp16 else np.float32).reshape(112, 1, 1),
        name=fix + "_weights",
    )
    model.graph.initializer.append(data)
    model.graph.node.append(onnx.helper.make_node("Mul", [name, fix + "_weights"], [fix], name=fix))


def _fix_wdl_softmax(model: onnx.ModelProto, name: str) -> None:
    """lc0 FixWdlSoftmax: add the softmax a TensorFlow export leaves off its WDL output."""
    fix = "softmax_fix"
    for node in model.graph.node:
        for index, produced in enumerate(node.output):
            if produced == name:
                node.output[index] = fix
                break
    model.graph.node.append(onnx.helper.make_node("Softmax", [fix], [name], name=fix))


def _validate(model: onnx.ModelProto, names: Names) -> None:
    """lc0 ValidateNetwork."""
    if not model.HasField("ir_version"):
        raise Onnx2LeelaError("ONNX file has no version; likely not an ONNX file")
    inputs = {item.name for item in model.graph.input}
    outputs = {item.name for item in model.graph.output}
    if names.input_planes is not None and names.input_planes not in inputs:
        raise Onnx2LeelaError(f"node {names.input_planes!r} does not exist in the ONNX file")
    for name in (names.value, names.wdl, names.policy, names.mlh):
        if name is not None and name not in outputs:
            raise Onnx2LeelaError(f"node {name!r} does not exist in the ONNX file")
    if names.input_planes is None:
        raise Onnx2LeelaError("--onnx-input must be defined")
    if names.policy is None:
        raise Onnx2LeelaError("--onnx-output-policy must be defined")
    if names.value is None and names.wdl is None:
        raise Onnx2LeelaError("either --onnx-output-value or --onnx-output-wdl must be defined")
    if names.value is not None and names.wdl is not None:
        raise Onnx2LeelaError("only one of --onnx-output-value and --onnx-output-wdl can be set")


def _lookup(table: dict[str, int], name: str) -> int:
    if name not in table:
        raise Onnx2LeelaError(f"enum value {name} is unknown")
    return table[name]


def weights_bytes(
    raw: bytes,
    names: Names,
    formats: Formats = DEFAULT_FORMATS,
    *,
    validate: bool = True,
    fix_rule50: bool = False,
    fix_wdl_softmax: bool = False,
) -> bytes:
    """Return the uncompressed Net message for an ONNX file's bytes."""
    model = onnx.load_from_string(raw)
    data_type = _data_type(model, names.input_planes)
    updated = False
    if names.input_planes is not None and fix_rule50:
        _fix_rule50(model, names.input_planes, data_type == DATA_FLOAT16)
        updated = True
    if names.policy is not None:
        updated |= _ensure_output_type(model, names.policy, data_type)
    if names.value is not None:
        updated |= _ensure_output_type(model, names.value, data_type)
    if names.wdl is not None:
        if fix_wdl_softmax:
            _fix_wdl_softmax(model, names.wdl)
            updated = True
        updated |= _ensure_output_type(model, names.wdl, data_type)
    if names.mlh is not None:
        updated |= _ensure_output_type(model, names.mlh, data_type)
    if validate:
        _validate(model, names)

    network = _int_field(1, _lookup(INPUT_FORMATS, formats.input))
    if names.value is not None:
        network += _int_field(2, OUTPUT_CLASSICAL)
    if names.wdl is not None:
        network += _int_field(2, OUTPUT_WDL)
    network += _int_field(3, NETWORK_ONNX)
    network += _int_field(4, _lookup(POLICY_FORMATS, formats.policy))
    network += _int_field(5, _lookup(VALUE_FORMATS, formats.value))
    if names.mlh is not None:
        network += _int_field(6, _lookup(MOVES_LEFT_FORMATS, formats.moves_left))

    onnx_message = _bytes_field(1, model.SerializeToString() if updated else raw)
    onnx_message += _int_field(2, data_type)
    if names.input_planes is not None:
        onnx_message += _text_field(3, names.input_planes)
    if names.value is not None:
        onnx_message += _text_field(4, names.value)
    if names.wdl is not None:
        onnx_message += _text_field(5, names.wdl)
    if names.policy is not None:
        onnx_message += _text_field(6, names.policy)
    if names.mlh is not None:
        onnx_message += _text_field(7, names.mlh)

    return (
        _varint((1 << 3) | 5)
        + struct.pack("<I", MAGIC)
        + _bytes_field(3, _int_field(1, 0) + _int_field(2, 28))
        + _bytes_field(4, _bytes_field(2, network))
        + _bytes_field(11, onnx_message)
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-format", default=Formats.input, choices=INPUT_FORMATS)
    parser.add_argument("--policy-format", default=Formats.policy, choices=POLICY_FORMATS)
    parser.add_argument("--value-format", default=Formats.value, choices=VALUE_FORMATS)
    parser.add_argument(
        "--moves-left-format", default=Formats.moves_left, choices=MOVES_LEFT_FORMATS
    )
    parser.add_argument("--onnx-input")
    parser.add_argument("--onnx-output-policy")
    parser.add_argument("--onnx-output-value")
    parser.add_argument("--onnx-output-wdl")
    parser.add_argument("--onnx-output-mlh")
    parser.add_argument("--no-validate-weights", action="store_true")
    parser.add_argument("--fix-rule50", action="store_true")
    parser.add_argument("--fix-wdl-softmax", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = weights_bytes(
            args.input.read_bytes(),
            Names(
                args.onnx_input,
                args.onnx_output_policy,
                args.onnx_output_value,
                args.onnx_output_wdl,
                args.onnx_output_mlh,
            ),
            Formats(
                args.input_format, args.policy_format, args.value_format, args.moves_left_format
            ),
            validate=not args.no_validate_weights,
            fix_rule50=args.fix_rule50,
            fix_wdl_softmax=args.fix_wdl_softmax,
        )
    except (Onnx2LeelaError, OSError) as error:
        print(f"onnx2leela error: {error}", file=sys.stderr)
        return 2
    args.output.write_bytes(gzip.compress(data))
    print(f"Wrote {args.output} ({len(data)} bytes before compression).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
