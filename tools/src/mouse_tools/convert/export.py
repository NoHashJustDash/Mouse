# SPDX-License-Identifier: GPL-3.0-or-later
"""Validated, atomic FP32/FP16 opset-17 export with an explicit consumer contract."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnx import TensorProto

from .model import T1Model
from .schema import Network, parse_net
from .wire import ConvertError, read_gzip


@dataclass(frozen=True)
class ExportReport:
    source_sha256: str
    output_sha256: str
    output: str
    parameter_count: int
    precision: str
    raw_format: dict[int, int]
    effective_format: dict[str, int]
    upgrades: tuple[str, ...]
    warnings: tuple[str, ...]


def _metadata(net: Network, precision: str) -> dict[str, str]:
    return {
        "mouse.schema_version": "1",
        "mouse.converter_version": "0.2.0-dev",
        "mouse.source_sha256": net.source_sha256,
        "mouse.input_format": "1",
        "mouse.network_format": "6",
        "mouse.raw_network_format": str(net.format.raw[3]),
        "mouse.policy_format": "3",
        "mouse.value_format": "2",
        "mouse.moves_left_format": "1",
        "mouse.policy_mapping": "lc0-1858-v1",
        "mouse.learned_parameter_count": str(net.parameter_count),
        "mouse.precision": precision,
    }


def validate_model(model: onnx.ModelProto) -> dict[str, str]:
    """Require unambiguous metadata AND the real float32 tensor interface."""
    metadata: dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in metadata:
            raise ConvertError(f"duplicate ONNX metadata key {item.key}")
        metadata[item.key] = item.value
    expected = {
        "mouse.schema_version": "1",
        "mouse.converter_version": "0.2.0-dev",
        "mouse.input_format": "1",
        "mouse.network_format": "6",
        "mouse.policy_format": "3",
        "mouse.value_format": "2",
        "mouse.moves_left_format": "1",
        "mouse.policy_mapping": "lc0-1858-v1",
    }
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise ConvertError(f"unsupported or missing ONNX metadata {key}")
    if metadata.get("mouse.raw_network_format") not in ("4", "6"):
        raise ConvertError("unsupported or missing raw network format metadata")
    if not re.fullmatch("[0-9a-f]{64}", metadata.get("mouse.source_sha256", "")):
        raise ConvertError("invalid or missing ONNX source SHA256 metadata")
    if not re.fullmatch("[1-9][0-9]*", metadata.get("mouse.learned_parameter_count", "")):
        raise ConvertError("invalid or missing ONNX learned parameter count")
    precision = metadata.setdefault("mouse.precision", "fp32")
    if precision not in ("fp32", "fp16"):
        raise ConvertError(f"unsupported ONNX precision metadata {precision!r}")
    if [(item.domain, item.version) for item in model.opset_import] != [("", 17)]:
        raise ConvertError("ONNX model must use exactly standard opset 17")
    if model.ir_version != 8:
        raise ConvertError("ONNX model must use IR version 8")
    if len(model.graph.input) != 1 or len(model.graph.output) != 3:
        raise ConvertError("ONNX model requires one input and three outputs")
    interface: list[tuple[onnx.ValueInfoProto, str, tuple[int, ...]]] = [
        (model.graph.input[0], "planes", (112, 8, 8))
    ]
    interface.extend(
        (value, name, shape)
        for value, name, shape in zip(
            model.graph.output, ("policy", "wdl", "mlh"), ((1858,), (3,), (1,)), strict=True
        )
    )
    for value, name, shape in interface:
        tensor = value.type.tensor_type
        dimensions = tensor.shape.dim
        if (
            value.name != name
            or tensor.elem_type != TensorProto.FLOAT
            or len(dimensions) != len(shape) + 1
            or dimensions[0].WhichOneof("value") != "dim_param"
            or dimensions[0].dim_param != "batch"
            or any(
                item.WhichOneof("value") != "dim_value" or item.dim_value != wanted
                for item, wanted in zip(dimensions[1:], shape, strict=True)
            )
        ):
            raise ConvertError(
                f"invalid ONNX {name} name/type/shape; expected [batch,{shape}] float32"
            )
    if any(item.data_location == TensorProto.EXTERNAL for item in model.graph.initializer):
        raise ConvertError("external ONNX tensor data is unsupported")
    try:
        onnx.checker.check_model(model, full_check=True)
    except onnx.checker.ValidationError as error:
        raise ConvertError(f"ONNX validation failed: {error}") from error
    _validate_precision(model, precision)
    return metadata


def _validate_precision(model: onnx.ModelProto, precision: str) -> None:
    """Check real weight and matrix-compute types, not only a precision label."""
    wanted = TensorProto.FLOAT16 if precision == "fp16" else TensorProto.FLOAT
    floating_types = {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    }
    weights = [item for item in model.graph.initializer if item.data_type in floating_types]
    if not weights or any(item.data_type != wanted for item in weights):
        raise ConvertError(f"ONNX floating weights disagree with {precision} precision")
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except (onnx.shape_inference.InferenceError, onnx.checker.ValidationError) as error:
        raise ConvertError(
            f"cannot validate ONNX {precision} compute precision: {error}"
        ) from error
    types = {
        item.name: item.type.tensor_type.elem_type
        for item in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]
    }
    types.update({item.name: item.data_type for item in inferred.graph.initializer})
    matmuls = [item for item in inferred.graph.node if item.op_type == "MatMul"]
    if (precision == "fp16" and not matmuls) or any(
        types.get(name) != wanted for node in matmuls for name in node.input
    ):
        raise ConvertError(f"ONNX matrix compute disagrees with {precision} precision")


def export_network(
    net: Network, destination: Path, *, overwrite: bool = False, precision: str = "fp32"
) -> ExportReport:
    """A failed export never replaces an existing valid target artifact."""
    temporary: Path | None = None
    try:
        if destination.exists() and not overwrite:
            raise ConvertError(f"output already exists: {destination}; use --overwrite explicitly")
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(
            suffix=".tmp.onnx", prefix=".mouse-", dir=destination.parent
        )
        os.close(descriptor)
        temporary = Path(filename)
        model = T1Model(net, precision=precision).cpu().eval()
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != net.parameter_count:
            raise ConvertError(
                f"learned parameter count mismatch: parsed {net.parameter_count}, model {count}"
            )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.inference_mode():
                torch.onnx.export(
                    model,
                    (torch.zeros((1, 112, 8, 8), dtype=torch.float32),),
                    str(temporary),
                    export_params=True,
                    input_names=["planes"],
                    output_names=["policy", "wdl", "mlh"],
                    dynamic_axes={
                        name: {0: "batch"} for name in ("planes", "policy", "wdl", "mlh")
                    },
                    opset_version=17,
                    dynamo=False,
                    external_data=False,
                )
        exported = onnx.load(temporary, load_external_data=False)
        onnx.helper.set_model_props(exported, _metadata(net, precision))
        validate_model(exported)
        onnx.save_model(exported, temporary)
        output_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Hard-link creation is atomic and refuses an existing destination on
            # both Windows and POSIX; a check followed by replace would race.
            os.link(temporary, destination)
        return ExportReport(
            net.source_sha256,
            output_hash,
            str(destination),
            count,
            precision,
            dict(net.format.raw),
            net.format.effective,
            net.format.upgrades,
            tuple(str(item.message) for item in caught),
        )
    except (OSError, RuntimeError) as error:
        raise ConvertError(f"cannot export ONNX to {destination}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def convert_network(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = False,
    precision: str = "fp32",
) -> ExportReport:
    raw, digest = read_gzip(source, expected_sha256=expected_sha256)
    return export_network(
        parse_net(raw, source_sha256=digest), destination, overwrite=overwrite, precision=precision
    )
