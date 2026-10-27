# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent-export parity on CPU; the reference artifact is never generated here."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnx
from numpy.typing import NDArray

from .export import validate_model
from .wire import ConvertError


@dataclass(frozen=True)
class ParityReport:
    positions: int
    distinct_positions: int
    tolerance: float
    diagnostic: bool
    passed: bool
    gate_satisfied: bool
    max_abs: dict[str, float]
    worst_position: dict[str, int]
    providers: tuple[str, ...]
    optimized: bool
    source_sha256: str
    converted_sha256: str
    reference_sha256: str
    inputs_sha256: str


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(path: Path, diagnostic: bool) -> NDArray[np.float32]:
    try:
        inputs = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(inputs, np.ndarray):
            inputs.close()
            raise ConvertError("input corpus must be a single NPY array")
        if (
            inputs.dtype != np.float32
            or inputs.ndim != 4
            or inputs.shape[0] < 1
            or inputs.shape[0] > 100000
            or tuple(inputs.shape[1:]) != (112, 8, 8)
        ):
            raise ConvertError("input corpus must be float32 [1..100000,112,8,8]")
        if not diagnostic and inputs.shape[0] < 1000:
            raise ConvertError("full parity gate requires at least 1000 encoded positions")
        for start in range(0, inputs.shape[0], 64):
            if not np.isfinite(inputs[start : start + 64]).all():
                raise ConvertError("input corpus contains nonfinite planes")
        return cast(NDArray[np.float32], inputs)
    except (OSError, ValueError, EOFError) as error:
        raise ConvertError(f"invalid input corpus {path}: {error}") from error


def _reference_interface(model: onnx.ModelProto) -> None:
    if len(model.graph.input) != 1 or len(model.graph.output) != 3:
        raise ConvertError("reference requires one input and three output heads")
    values = [(model.graph.input[0], "/input/planes", (112, 8, 8))]
    interface: list[tuple[onnx.ValueInfoProto, str, tuple[int, ...]]] = list(values)
    interface.extend(
        (value, name, (width,))
        for value, name, width in zip(
            model.graph.output,
            ("/output/policy", "/output/wdl", "/output/mlh"),
            (1858, 3, 1),
            strict=True,
        )
    )
    for value, name, shape in interface:
        tensor = value.type.tensor_type
        dims = tensor.shape.dim
        if (
            value.name != name
            or tensor.elem_type != onnx.TensorProto.FLOAT
            or len(dims) != len(shape) + 1
            or not dims[0].dim_param
            or any(dim.dim_value != wanted for dim, wanted in zip(dims[1:], shape, strict=True))
        ):
            raise ConvertError(f"invalid reference tensor interface {name}")
    onnx.checker.check_model(model, full_check=True)


def _output(value: object, batch: int, width: int, name: str) -> NDArray[np.float32]:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (batch, width)
        or not np.isfinite(value).all()
    ):
        raise ConvertError(f"invalid/nonfinite runtime output {name}")
    return cast(NDArray[np.float32], value)


def run_parity(
    converted: Path,
    reference: Path,
    inputs_path: Path,
    *,
    reference_sha256: str,
    diagnostic: bool = False,
    batch_size: int = 8,
    optimized: bool = True,
) -> ParityReport:
    """Compare every corpus row; the full gate never caps depth/count or relaxes 1e-4."""
    if not 1 <= batch_size <= 256:
        raise ConvertError("parity batch size must be in 1..256")
    try:
        artifacts = {"converted": converted, "reference": reference, "inputs": inputs_path}
        for path in (converted, reference):
            if path.stat().st_size > 512 * 1024 * 1024:
                raise ConvertError("ONNX parity artifact exceeds 512 MiB limit")
        # Freeze every artifact identity before loading any graph or corpus.
        # Inference sessions and NPY mappings may retain data after a pathname
        # changes, so a hash taken only after inference cannot identify its input.
        frozen = {name: _digest(path) for name, path in artifacts.items()}
        if (
            not re.fullmatch("[0-9a-f]{64}", reference_sha256)
            or frozen["reference"] != reference_sha256
        ):
            raise ConvertError("reference SHA256 does not match the explicitly pinned oracle")
        inputs = _inputs(inputs_path, diagnostic)
        distinct = len({hashlib.sha256(row.tobytes(order="C")).digest() for row in inputs})
        if not diagnostic and distinct < 1000:
            raise ConvertError("full parity gate requires at least 1000 distinct complete tensors")
        own_model = onnx.load(converted, load_external_data=False)
        metadata = validate_model(own_model)
        _reference_interface(onnx.load(reference, load_external_data=False))
        # Load the optional runtime only after artifacts and inputs are validated.
        # Its platform diagnostics must not replace a precise corpus error.
        import onnxruntime as ort  # type: ignore[import-untyped]

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if optimized
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        own = ort.InferenceSession(
            str(converted), sess_options=options, providers=["CPUExecutionProvider"]
        )
        oracle = ort.InferenceSession(
            str(reference), sess_options=options, providers=["CPUExecutionProvider"]
        )
        for session in (own, oracle):
            session.disable_fallback()
            if session.get_providers() != ["CPUExecutionProvider"]:
                raise ConvertError("parity requires exactly CPUExecutionProvider")
        maxima = dict.fromkeys(("policy", "wdl", "mlh"), 0.0)
        worst = dict.fromkeys(maxima, 0)
        for start in range(0, inputs.shape[0], batch_size):
            batch = np.ascontiguousarray(inputs[start : start + batch_size])
            actual = own.run(["policy", "wdl", "mlh"], {"planes": batch})
            expected = oracle.run(
                ["/output/policy", "/output/wdl", "/output/mlh"], {"/input/planes": batch}
            )
            for name, width, left, right in zip(
                ("policy", "wdl", "mlh"), (1858, 3, 1), actual, expected, strict=True
            ):
                a = _output(left, len(batch), width, name)
                b = _output(right, len(batch), width, name)
                difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
                maximum = float(difference.max())
                if maximum > maxima[name]:
                    maxima[name] = maximum
                    worst[name] = start + int(difference.argmax()) // width
        # Never publish a gate result bound to replacement or modified bytes.
        for name, path in artifacts.items():
            if _digest(path) != frozen[name]:
                raise ConvertError(f"{name} artifact changed during parity: {path}")
        passed = all(value <= 1e-4 for value in maxima.values())
        return ParityReport(
            len(inputs),
            distinct,
            1e-4,
            diagnostic,
            passed,
            passed and not diagnostic and len(inputs) >= 1000,
            maxima,
            worst,
            ("CPUExecutionProvider",),
            optimized,
            metadata["mouse.source_sha256"],
            frozen["converted"],
            frozen["reference"],
            frozen["inputs"],
        )
    except ConvertError:
        raise
    except Exception as error:
        # ORT/protobuf expose several extension exception classes. This process
        # boundary keeps all invalid artifacts loud with the original cause.
        raise ConvertError(f"cannot compare ONNX artifacts: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    if not sys.flags.isolated:
        print(
            "parity file readers require python -I -m mouse_tools.convert.parity", file=sys.stderr
        )
        return 2
    parser = argparse.ArgumentParser(
        description="Fixed 1e-4 CPU parity against a pinned Lc0 export"
    )
    parser.add_argument("converted", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--diagnostic", action="store_true", help="debug only; never reports gate success"
    )
    parser.add_argument("--disable-optimizations", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = run_parity(
            arguments.converted,
            arguments.reference,
            arguments.inputs,
            reference_sha256=arguments.reference_sha256,
            diagnostic=arguments.diagnostic,
            batch_size=arguments.batch_size,
            optimized=not arguments.disable_optimizations,
        )
    except ConvertError as error:
        print(f"parity failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
