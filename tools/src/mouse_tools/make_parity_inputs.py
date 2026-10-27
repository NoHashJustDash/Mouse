# SPDX-License-Identifier: GPL-3.0-or-later
"""Publish exactly 1000 distinct FP32 inputs from a passed 10000-position gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mouse_tools.encoding import ALL_SQUARES
from mouse_tools.encoding_gate import (
    POSITIONS,
    EncodingGateError,
    _reject_constant,
    _unique_object,
    parse_probe_line,
)
from mouse_tools.perft_reference import REFERENCE_SHA256

INPUTS = 1000


class ParityInputError(ValueError):
    """Incomplete, mismatched, or duplicate-padded parity inputs are rejected."""


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _destinations(probe: Path, report: Path, output: Path, provenance: Path) -> None:
    paths = [path.resolve() for path in (probe, report, output, provenance)]
    if len(set(paths)) != 4:
        raise ParityInputError("Source and output paths must all differ")
    for index, first in enumerate(paths):
        for second in paths[index + 1 :]:
            if first.exists() and second.exists() and first.samefile(second):
                raise ParityInputError("Source/output hard-link collision")
    if output.exists() or provenance.exists():
        raise ParityInputError("Input artifacts must use new output paths")


def _gate(report: Path, probe_sha256: str) -> set[int]:
    if report.stat().st_size > 64 * 1024 * 1024:
        raise ParityInputError("Gate report exceeds 64 MiB")
    value = json.loads(
        report.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_unique_object,
    )
    expected: dict[str, object] = {
        "schema": 1,
        "gate": "v0.2-input-policy-10000",
        "status": "passed",
        "probe_sha256": probe_sha256,
        "required_positions": POSITIONS,
        "positions_verified": POSITIONS,
        "planes_verified": POSITIONS * 112,
        "oracle_queries": POSITIONS,
        "oracle_sha256": REFERENCE_SHA256,
        "oracle_count_cache": False,
    }
    if not isinstance(value, dict) or any(
        type(value.get(key)) is not type(item) or value.get(key) != item
        for key, item in expected.items()
    ):
        raise ParityInputError(
            "Require a passed complete gate report with matching probe SHA/counts"
        )
    white, black = value.get("white"), value.get("black")
    if (
        type(white) is not int
        or type(black) is not int
        or white <= 0
        or black <= 0
        or white + black != POSITIONS
    ):
        raise ParityInputError(
            "Gate report must account for both colors across all 10000 positions"
        )
    results = value.get("results")
    if not isinstance(results, list) or len(results) != POSITIONS:
        raise ParityInputError("Gate report must retain all 10000 per-position results")
    identifiers: set[int] = set()
    for row in results:
        if (
            not isinstance(row, dict)
            or row.get("status") != "passed"
            or type(row.get("id")) is not int
            or row["id"] in identifiers
        ):
            raise ParityInputError("Gate report contains incomplete or duplicate position results")
        identifiers.add(row["id"])
    return identifiers


def _expand(values: object) -> NDArray[np.float32]:
    if not isinstance(values, list) or len(values) != 112:
        raise ParityInputError("Each input must contain every one of 112 planes")
    tensor: NDArray[np.float32] = np.zeros((112, 64), dtype=np.float32)
    squares = np.arange(64, dtype=np.uint64)
    for index, pair in enumerate(values):
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or type(pair[0]) is not int
            or not 0 <= pair[0] <= ALL_SQUARES
            or type(pair[1]) not in (int, float)
            or not math.isfinite(pair[1])
            or abs(pair[1]) > float(np.finfo(np.float32).max)
        ):
            raise ParityInputError(f"Malformed FP32 plane {index}")
        # Independent expansion from packed probe data; do not call the encoder's dense().
        tensor[index] = ((np.uint64(pair[0]) >> squares) & np.uint64(1)).astype(
            np.float32
        ) * np.float32(pair[1])
    return tensor.reshape((112, 8, 8))


def build_inputs(probe: Path, report: Path, output: Path, provenance: Path) -> None:
    _destinations(probe, report, output, provenance)
    if probe.stat().st_size > 512 * 1024 * 1024:
        raise ParityInputError("Probe exceeds 512 MiB")
    probe_hash = _hash(probe)
    passed_ids = _gate(report, probe_hash)
    report_hash = _hash(report)
    with probe.open(encoding="utf-8") as stream:
        if sum(1 for _ in stream) != POSITIONS:
            raise ParityInputError("Validated source must retain exactly 10000 records")
    array: NDArray[np.float32] = np.empty((INPUTS, 112, 8, 8), dtype=np.float32)
    seen: set[bytes] = set()
    selected: list[dict[str, object]] = []
    with probe.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = parse_probe_line(line)
            except EncodingGateError as error:
                raise ParityInputError(str(error)) from error
            if record["id"] not in passed_ids:
                raise ParityInputError("Selected position is absent from the passed report")
            tensor = _expand(record["planes"])
            key = tensor.astype("<f4", copy=False).tobytes(order="C")
            if key in seen:
                continue
            seen.add(key)
            array[len(selected)] = tensor
            selected.append(
                {
                    "id": record["id"],
                    "source_line": line_number,
                    "tensor_sha256_le_f32": hashlib.sha256(key).hexdigest(),
                }
            )
            if len(selected) == INPUTS:
                break
    if len(selected) != INPUTS:
        raise ParityInputError(
            f"Need exactly {INPUTS} distinct full input tensors; found {len(selected)}"
        )
    if _hash(probe) != probe_hash or _hash(report) != report_hash:
        raise ParityInputError("Source probe or passed gate report changed while selecting inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    published: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=output.name + ".", delete=False
        ) as stream:
            staged.append(Path(stream.name))
            np.save(stream, array, allow_pickle=False)
        metadata: dict[str, object] = {
            "schema": 1,
            "positions": INPUTS,
            "source_positions": POSITIONS,
            "shape": [INPUTS, 112, 8, 8],
            "dtype": "float32",
            "order": "C",
            "deduplication": "exact little-endian float32 tensor bytes; no padding",
            "probe": str(probe.resolve()),
            "probe_sha256": probe_hash,
            "gate_report": str(report.resolve()),
            "gate_report_sha256": report_hash,
            "output": str(output.resolve()),
            "npy_sha256": _hash(staged[0]),
            "tensor_data_sha256_le_f32": hashlib.sha256(
                array.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "selected": selected,
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=provenance.parent,
            prefix=provenance.name + ".",
            delete=False,
        ) as stream:
            staged.append(Path(stream.name))
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        # Exclusive publication preserves prior evidence. Provenance is the final commit marker.
        for temporary, target in zip(staged, (output, provenance), strict=True):
            os.link(temporary, target)
            published.append(target)
    except BaseException:
        for path in published:
            with suppress(FileNotFoundError):
                path.unlink()
        raise
    finally:
        for path in staged:
            with suppress(FileNotFoundError):
                path.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("validation and probe files must be read with python -I")
    try:
        build_inputs(arguments.probe, arguments.gate_report, arguments.output, arguments.provenance)
    except (ParityInputError, EncodingGateError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Published exactly {INPUTS} distinct full 112-plane FP32 inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
