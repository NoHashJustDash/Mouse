# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict fixed 50 backend parity and the independently specified binary transport."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from mouse_tools.encoding import EncodingError, HistoryFrame, Plane, policy_index

FIXTURE_SHA256 = "e581f042ab3918d1d87f82106e58b0e553841b6787c11ab90344a13310b61be5"
OFFICIAL_REFERENCE_SHA256 = "0e2699c19b617781d4fbe3d00286756b125836a30c0658afc620e197bec35d04"
POSITIONS = 50
TOLERANCE = 0.005
REQUIRED_EDGE_CASES = frozenset(
    (
        "start_white",
        "start_clock",
        "start_black",
        "black_asymmetric",
        "rule50_99",
        "castle_white",
        "castle_black",
        "promotion_white",
        "promotion_black",
        "capture_promotion_white",
        "capture_promotion_black",
        "ep_white",
        "ep_black",
        "ep_pinned_white",
        "ep_pinned_black",
        "frc_stationary_king_white",
        "frc_stationary_king_black",
        "frc_swap_white",
        "frc_swap_black",
        "frc_stationary_rook_white",
        "frc_stationary_rook_black",
        "frc_queenside_white",
        "frc_queenside_black",
        "single_check",
        "pawnless_rook",
        "repetition_cycle",
        "pawn_opening_history",
    )
)


class BackendParityError(ValueError):
    """Incomplete, changed, malformed, or numerically incompatible evidence."""


@dataclass(frozen=True)
class Case:
    identifier: str
    history: tuple[HistoryFrame, ...]
    planes: tuple[Plane, ...]
    moves: tuple[tuple[str, int], ...]
    tensor_sha256: str


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encode_inputs(rows: Sequence[Sequence[Plane]]) -> bytes:
    if not 1 <= len(rows) <= 1024 or any(len(row) != 112 for row in rows):
        raise BackendParityError("Input requires 1..1024 complete 112-plane rows")
    try:
        data = b"".join(struct.pack("<Qf", p.mask, p.value) for row in rows for p in row)
    except (OverflowError, struct.error) as error:
        raise BackendParityError("Input plane cannot be represented as finite float32") from error
    result = b"MOUSEIN1" + struct.pack("<II", len(rows), 112) + data
    decode_inputs(result)
    return result


def decode_inputs(data: bytes) -> tuple[tuple[Plane, ...], ...]:
    if len(data) < 16 or data[:8] != b"MOUSEIN1":
        raise BackendParityError("Invalid or truncated MOUSEIN1 header")
    count, planes = struct.unpack_from("<II", data, 8)
    if not 1 <= count <= 1024 or planes != 112 or len(data) != 16 + count * 112 * 12:
        raise BackendParityError("Invalid input row/plane count or exact binary length")
    values = tuple(struct.iter_unpack("<Qf", data[16:]))
    if any(not math.isfinite(value) for _, value in values):
        raise BackendParityError("Input transport contains nonfinite float32 plane")
    return tuple(
        tuple(Plane(mask, value) for mask, value in values[i : i + 112])
        for i in range(0, len(values), 112)
    )


def _heads(values: NDArray[np.float32], positions: int) -> NDArray[np.float32]:
    if (
        values.dtype != np.float32
        or values.shape != (positions, 1862)
        or not np.isfinite(values).all()
    ):
        raise BackendParityError("Output heads require finite float32[positions,1862]")
    wdl = values[:, 1858:1861]
    if np.any(wdl < 0) or np.any(wdl > 1) or np.any(np.abs(wdl.sum(1) - 1) > 0.001):
        raise BackendParityError("WDL must be normalized probabilities in W,D,L order")
    if np.any(values[:, 1861] < 0):
        raise BackendParityError("Moves-left output must be nonnegative")
    return values


def parse_output(
    data: bytes, *, positions: int, model_sha256: str, inputs_sha256: str
) -> NDArray[np.float32]:
    if not 1 <= positions <= 1024 or len(data) != 144 + positions * 7448 or data[:8] != b"MOUSEOU1":
        raise BackendParityError("Invalid/truncated MOUSEOU1 exact length/header")
    if struct.unpack_from("<II", data, 8) != (positions, 1862):
        raise BackendParityError("Output count or floats-per-row differs")
    try:
        model, inputs = data[16:80].decode("ascii"), data[80:144].decode("ascii")
    except UnicodeError as error:
        raise BackendParityError("Output hashes must be ASCII SHA256") from error
    if any(re.fullmatch("[0-9a-f]{64}", h) is None for h in (model, inputs)):
        raise BackendParityError("Output hashes must be lowercase SHA256")
    if model != model_sha256 or inputs != inputs_sha256:
        raise BackendParityError("Output model/input SHA256 does not match frozen artifacts")
    values = np.frombuffer(data, dtype="<f4", offset=144).reshape(positions, 1862).copy()
    return _heads(values, positions)


def legal_probabilities(logits: NDArray[np.float32], indices: Sequence[int]) -> NDArray[np.float64]:
    if (
        logits.shape != (1858,)
        or not np.isfinite(logits).all()
        or not indices
        or any(type(i) is not int or not 0 <= i < 1858 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise BackendParityError("Require finite logits and every distinct legal policy index")
    legal = logits[list(indices)].astype(np.float64)
    probabilities = np.exp(legal - legal.max())
    return cast(NDArray[np.float64], probabilities / probabilities.sum())


def expand_inputs(rows: Sequence[Sequence[Plane]]) -> NDArray[np.float32]:
    return np.array(
        [
            [[p.value if (p.mask >> square) & 1 else 0.0 for square in range(64)] for p in row]
            for row in rows
        ],
        dtype=np.float32,
    ).reshape(-1, 112, 8, 8)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise BackendParityError(f"Duplicate JSON key {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise BackendParityError(f"Nonfinite JSON constant {value}")


def load_cases(path: Path) -> tuple[Case, ...]:
    try:
        if path.stat().st_size > 4 * 1024 * 1024 or _hash(path) != FIXTURE_SHA256:
            raise BackendParityError("Fixed50 case artifact SHA256/size mismatch; no resampling")
        data = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite
        )
        if (
            not isinstance(data, dict)
            or set(data) != {"schema", "positions", "history_fill", "cases"}
            or data["schema"] != 1
            or data["positions"] != 50
            or data["history_fill"] != "fen_only"
            or not isinstance(data["cases"], list)
            or len(data["cases"]) != 50
        ):
            raise BackendParityError("Require the complete fixed 50 FEN_ONLY fixture")
        cases = []
        for row in data["cases"]:
            if (
                not isinstance(row, dict)
                or set(row) != {"id", "history", "planes", "moves", "tensor_sha256", "source"}
                or not isinstance(row["id"], str)
                or re.fullmatch("[a-z0-9_]+", row["id"]) is None
            ):
                raise BackendParityError("Malformed case identifier/fields")
            frames = row["history"]
            if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
                raise BackendParityError("Require 1..8 explicit history frames")
            history = []
            for frame in frames:
                if (
                    not isinstance(frame, dict)
                    or set(frame) != {"fen", "repetitions"}
                    or not isinstance(frame["fen"], str)
                    or type(frame["repetitions"]) is not int
                ):
                    raise BackendParityError("Malformed history frame")
                history.append(HistoryFrame(frame["fen"], frame["repetitions"]))
            pairs = row["planes"]
            if not isinstance(pairs, list) or len(pairs) != 112:
                raise BackendParityError("Each case must retain 112 explicit planes")
            planes = []
            for pair in pairs:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or type(pair[0]) is not int
                    or type(pair[1]) not in (int, float)
                    or not math.isfinite(pair[1])
                    or abs(pair[1]) > float(np.finfo(np.float32).max)
                ):
                    raise BackendParityError("Invalid packed plane")
                planes.append(Plane(pair[0], float(pair[1])))
            moves = row["moves"]
            if not isinstance(moves, list) or not moves:
                raise BackendParityError("Complete legal move/index rows are required")
            parsed = []
            black = history[-1].fen.split()[1] == "b"
            for move in moves:
                if (
                    not isinstance(move, list)
                    or len(move) != 2
                    or not isinstance(move[0], str)
                    or type(move[1]) is not int
                    or policy_index(move[0], black=black) != move[1]
                ):
                    raise BackendParityError("Malformed/mismatched legal policy row")
                parsed.append((move[0], move[1]))
            if len({m for m, _ in parsed}) != len(parsed) or len({i for _, i in parsed}) != len(
                parsed
            ):
                raise BackendParityError("Duplicate legal moves or indices")
            key = hashlib.sha256(
                expand_inputs((planes,)).astype("<f4", copy=False).tobytes()
            ).hexdigest()
            if key != row["tensor_sha256"]:
                raise BackendParityError("Case tensor hash differs from packed planes")
            cases.append(Case(row["id"], tuple(history), tuple(planes), tuple(parsed), key))
        if (
            len({c.identifier for c in cases}) != 50
            or len({c.tensor_sha256 for c in cases}) != 50
            or not {c.identifier for c in cases} >= REQUIRED_EDGE_CASES
        ):
            raise BackendParityError("Retain all 27 named edges and 50 distinct inputs")
        return tuple(cases)
    except (OSError, ValueError, EncodingError) as error:
        raise BackendParityError(f"Invalid fixed 50 fixture: {error}") from error


def _oracle(reference: Path, planes: NDArray[np.float32]) -> NDArray[np.float32]:
    import onnxruntime as ort  # type: ignore[import-untyped]  # ORT has no type marker.

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(reference), sess_options=options, providers=["CPUExecutionProvider"]
    )
    session.disable_fallback()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise BackendParityError("Official ONNX oracle must run explicitly on CPU only")
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "/input/planes"
        or inputs[0].type != "tensor(float)"
        or inputs[0].shape[1:] != [112, 8, 8]
        or len(outputs) != 3
    ):
        raise BackendParityError("Unsupported official ONNX oracle input interface")
    names = ["/output/policy", "/output/wdl", "/output/mlh"]
    for value, name, width in zip(outputs, names, (1858, 3, 1), strict=True):
        if value.name != name or value.type != "tensor(float)" or value.shape[1:] != [width]:
            raise BackendParityError("Unsupported official ONNX oracle output interface")
    results = []
    for first in range(0, len(planes), 8):
        batch = planes[first : first + 8]
        heads = session.run(names, {"/input/planes": batch})
        if any(
            not isinstance(h, np.ndarray) or h.dtype != np.float32 or h.shape != (len(batch), w)
            for h, w in zip(heads, (1858, 3, 1), strict=True)
        ):
            raise BackendParityError("Official oracle returned invalid output heads")
        results.append(_heads(np.concatenate(heads, axis=1), len(batch)))
    return _heads(np.concatenate(results, axis=0), len(planes))


def run_gate(
    engine: Path,
    model: Path,
    reference: Path,
    cases_path: Path,
    inputs_path: Path,
    *,
    backend: str,
    precision: str,
    buckets: tuple[int, ...],
    gpu_dll_dirs: Sequence[Path] = (),
    cache_dir: Path | None = None,
    timeout_seconds: float = 900,
) -> dict[str, object]:
    """Evaluate all fixed 50 rows; CPU evidence never satisfies the FP16 GPU gate."""
    import subprocess

    import onnx

    from mouse_tools.convert.export import validate_model

    if backend not in ("cpu", "cuda", "tensorrt") or precision not in ("fp32", "fp16"):
        raise BackendParityError("Backend/precision must be explicit supported values")
    if backend == "cpu" and precision != "fp32":
        raise BackendParityError("Native CPU configuration supports fp32 only")
    if (
        not buckets
        or tuple(sorted(set(buckets))) != buckets
        or any(type(b) is not int or not 1 <= b <= 1024 for b in buckets)
    ):
        raise BackendParityError("Require increasing distinct batch buckets 1..1024")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise BackendParityError("Timeout must be positive and finite")
    if any(not p.is_absolute() or not p.is_dir() for p in gpu_dll_dirs):
        raise BackendParityError("GPU DLL directories must be explicit existing absolute paths")
    if cache_dir is not None and not cache_dir.is_absolute():
        raise BackendParityError("Cache directory must be an absolute path")
    paths = {
        "engine": engine.resolve(),
        "model": model.resolve(),
        "reference": reference.resolve(),
        "cases": cases_path.resolve(),
        "inputs": inputs_path.resolve(),
    }
    if len(set(paths.values())) != len(paths):
        raise BackendParityError("All evidence artifacts must have distinct paths")
    try:
        limits = {
            "engine": 512 * 1024 * 1024,
            "model": 512 * 1024 * 1024,
            "reference": 512 * 1024 * 1024,
            "cases": 4 * 1024 * 1024,
            "inputs": 16 + 1024 * 112 * 12,
        }
        if any(path.stat().st_size > limits[name] for name, path in paths.items()):
            raise BackendParityError("Evidence artifact exceeds its size limit")
        # Freeze every identity before loading any model, corpus, session or executable.
        frozen = {name: _hash(path) for name, path in paths.items()}
        if frozen["reference"] != OFFICIAL_REFERENCE_SHA256:
            raise BackendParityError("Official Lc0 reference ONNX SHA256 mismatch")
        cases = load_cases(paths["cases"])
        binary = paths["inputs"].read_bytes()
        rows = decode_inputs(binary)
        if len(rows) != POSITIONS or binary != encode_inputs(tuple(c.planes for c in cases)):
            raise BackendParityError(
                "Input transport differs from the frozen complete fixed 50 cases"
            )
        metadata = validate_model(onnx.load(paths["model"], load_external_data=False))
        actual_precision = metadata["mouse.precision"]
        if precision == "fp32" and actual_precision != "fp32":
            raise BackendParityError("FP32 execution requires an FP32 model")
        if backend == "cuda" and precision == "fp16" and actual_precision != "fp16":
            raise BackendParityError("CUDA fp16 execution requires a genuine fp16 graph")
        expected = _oracle(paths["reference"], expand_inputs(rows))
        command = [
            str(paths["engine"]),
            "backend-eval",
            "--weights",
            str(paths["model"]),
            "--backend",
            backend,
            "--precision",
            precision,
            "--inputs",
            str(paths["inputs"]),
            "--buckets",
            ",".join(map(str, buckets)),
        ]
        for directory in gpu_dll_dirs:
            command.extend(("--gpu-dll-dir", str(directory)))
        if cache_dir is not None:
            command.extend(("--cache-dir", str(cache_dir)))
        completed = subprocess.run(
            command, capture_output=True, check=False, timeout=timeout_seconds
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise BackendParityError(
                f"Native backend-eval exited {completed.returncode}: {detail[:4000]}"
            )
        observed = parse_output(
            completed.stdout,
            positions=POSITIONS,
            model_sha256=frozen["model"],
            inputs_sha256=frozen["inputs"],
        )
        results = []
        maxima = {"policy_probability": 0.0, "wdl": 0.0, "mlh": 0.0, "raw_policy": 0.0}
        for index, case in enumerate(cases):
            indices = tuple(i for _, i in case.moves)
            difference = np.abs(
                legal_probabilities(observed[index, :1858], indices)
                - legal_probabilities(expected[index, :1858], indices)
            )
            policy = float(difference.max())
            wdl = float(
                np.abs(
                    observed[index, 1858:1861].astype(np.float64) - expected[index, 1858:1861]
                ).max()
            )
            mlh = abs(float(observed[index, 1861]) - float(expected[index, 1861]))
            raw = float(
                np.abs(observed[index, :1858].astype(np.float64) - expected[index, :1858]).max()
            )
            maxima = {
                key: max(value, maxima[key])
                for key, value in [
                    ("policy_probability", policy),
                    ("wdl", wdl),
                    ("mlh", mlh),
                    ("raw_policy", raw),
                ]
            }
            results.append(
                {
                    "id": case.identifier,
                    "legal_moves": len(indices),
                    "policy_probability_max_abs": policy,
                    "worst_policy_move": case.moves[int(difference.argmax())][0],
                    "wdl_max_abs": wdl,
                    "mlh_abs": mlh,
                    "raw_policy_max_abs": raw,
                    "passed": policy <= TOLERANCE and wdl <= TOLERANCE,
                }
            )
        # Sessions and subprocesses may retain old data after pathname replacement.
        # The report always identifies the frozen pre-load artifacts, never replacements.
        for name, path in paths.items():
            if _hash(path) != frozen[name]:
                raise BackendParityError(f"{name} artifact changed during parity evaluation")
        passed = all(row["passed"] for row in results)
        return {
            "schema": 1,
            "gate": "v0.3-fixed50-backend-parity",
            "positions": POSITIONS,
            "distinct_positions": POSITIONS,
            "backend": backend,
            "precision": precision,
            "model_precision": actual_precision,
            "tolerance": TOLERANCE,
            "policy_temperature": 1,
            "passed": passed,
            "gate_satisfied": passed and backend in ("cuda", "tensorrt") and precision == "fp16",
            "cpu_diagnostic": backend == "cpu",
            "reference_providers": ["CPUExecutionProvider"],
            "artifact_sha256": frozen,
            "source_sha256": metadata["mouse.source_sha256"],
            "max_abs": maxima,
            "results": results,
            "command": command,
            "native_stderr": completed.stderr.decode("utf-8", errors="replace"),
        }
    except BackendParityError:
        raise
    except Exception as error:
        # ORT/protobuf expose extension exception classes. Keep process-boundary
        # failures concise and nonzero while retaining their original cause.
        raise BackendParityError(f"Cannot evaluate fixed 50 backend parity: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    if not sys.flags.isolated:
        print(
            "Backend parity readers require python -I -m mouse_tools.backend_parity",
            file=sys.stderr,
        )
        return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "tensorrt"), required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--gpu-dll-dir", type=Path, action="append", default=[])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=900)
    a = parser.parse_args(argv)
    try:
        buckets = tuple(int(value) for value in a.buckets.split(","))
        report = run_gate(
            a.engine,
            a.weights,
            a.reference,
            a.cases,
            a.inputs,
            backend=a.backend,
            precision=a.precision,
            buckets=buckets,
            gpu_dll_dirs=a.gpu_dll_dir,
            cache_dir=a.cache_dir,
            timeout_seconds=a.timeout,
        )
    except (BackendParityError, ValueError) as error:
        print(f"backend parity failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
