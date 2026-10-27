# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit complete retained Windows GPU backendbench sweeps without executing engines.

The schema-1 controller's summaries are untrusted: derive maxima from every raw
positive finite binary64 throughput. Only summaries are correctable; truncated
measurements, inconsistent row tables or changed identities invalidate evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import cast

COUNTS = {"start": 1, "max": 1024, "step": 1, "batches": 100}
RAW_FILES = (
    "state.json",
    "run-full-sweeps.ps1",
    "mouse-full.stdout.txt",
    "mouse-full.stderr.txt",
    "reference-full.stdout.txt",
    "reference-full.stderr.txt",
    "mouse-model.txt",
    "mouse-version.txt",
    "gpu-before.txt",
    "gpu-after.txt",
)
SUMMARY_FIELDS = (
    "best_mouse",
    "best_reference",
    "ratio",
    "gate_80_percent",
    "rc_backend_90_percent",
)
SHA256 = re.compile(r"[0-9a-f]{64}")
FLOAT = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
TOKEN = re.compile(r'"(?:[^"\\\x00-\x1f]|\\.)*"|[^"\\\s]+')
Row = dict[str, int | float]


class ThroughputError(ValueError):
    """Incomplete or inconsistent sweep evidence cannot establish throughput."""


@dataclass(frozen=True)
class MouseSweep:
    config: dict[str, str]
    artifact: dict[str, str]
    sdk_directories: tuple[str, ...]
    device: dict[str, str]
    rows: list[Row]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ThroughputError(message)


def _dict(value: object) -> dict[str, object]:
    _require(
        isinstance(value, dict) and all(isinstance(k, str) for k in value),
        "Require object with string keys",
    )
    return cast(dict[str, object], value)


def _positive(value: object) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, int | float),
        "Require positive finite binary64 number",
    )
    number = cast(int | float, value)
    _require(
        0 < number <= sys.float_info.max and math.isfinite(number),
        "Require positive finite binary64 number",
    )
    return float(number)


def _int(value: object, minimum: int, maximum: int) -> int:
    _require(type(value) is int and minimum <= value <= maximum, "Invalid integer/count")
    return cast(int, value)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ThroughputError(f"Nonfinite JSON constant: {value}")


def _fields(line: str, expected: set[str]) -> dict[str, str]:
    tokens: list[str] = []
    end = 0
    for match in TOKEN.finditer(line):
        _require(not line[end : match.start()].strip(), "Malformed quoted benchmark token")
        token = match.group()
        tokens.append(json.loads(token) if token.startswith('"') else token)
        end = match.end()
    _require(not line[end:].strip() and len(tokens) % 2 == 0, "Malformed benchmark fields")
    result: dict[str, str] = {}
    for index in range(0, len(tokens), 2):
        key, value = tokens[index : index + 2]
        _require(key not in result, "Duplicate benchmark field")
        result[key] = value
    _require(set(result) == expected, "Unknown/missing benchmark field")
    return result


def _decimal(text: str, fixed: bool) -> tuple[float, Decimal, Decimal]:
    _require(FLOAT.fullmatch(text) is not None, "Malformed finite numeric benchmark token")
    value = _positive(float(text))
    number = Decimal(text)
    # Mouse prints fixed three decimals. Pinned Lc0 v0.32.1 backendbench prints
    # default iostream precision (six significant digits), omitting zero tails.
    if fixed:
        _require(
            re.fullmatch(r"[0-9]+\.[0-9]{3}", text) is not None,
            "Require Mouse fixed-three-decimal output",
        )
        half = Decimal("0.0005")
    else:
        _require(
            len(number.normalize().as_tuple().digits) <= 6,
            "Reference token exceeds six significant digits",
        )
        half = Decimal(5).scaleb(number.adjusted() - 6)
    return value, number - half, number + half


def _timing(nps: str, milliseconds: str, positions: int, *, fixed: bool) -> tuple[float, float]:
    with localcontext() as context:
        context.prec = 64
        rate, rate_low, rate_high = _decimal(nps, fixed)
        duration, time_low, time_high = _decimal(milliseconds, fixed)
        product = Decimal(positions) * 1000
        _require(
            rate_low * time_low <= product <= rate_high * time_high,
            "Throughput/time inconsistent beyond printed rounding intervals",
        )
    return rate, duration


def _sequence(rows: list[Row]) -> None:
    _require(
        len(rows) == 1024 and all(row["batch"] == index for index, row in enumerate(rows, 1)),
        "Require all ordered batches 1..1024 exactly once",
    )


def _best(rows: list[Row]) -> Row:
    # Numeric maxima; preserve native first-encountered ordering on exact ties.
    return max(rows, key=lambda row: row["nps"])


def parse_mouse(text: str) -> MouseSweep:
    lines = text.splitlines()
    _require(
        len(lines) >= 1030 and lines[0].startswith("benchmark_config "),
        "Missing Mouse benchmark configuration",
    )
    config = _fields(
        lines[0][len("benchmark_config ") :],
        set(
            [
                "version",
                "git",
                "build",
                "compiler",
                "ort_version",
                "model_sha256",
                "start_batch",
                "max_batch",
                "step",
                "batches",
                "device_index",
                "cpu_threads",
                "workspace_bytes",
                "cache",
            ]
        ),
    )
    _require(
        (
            config["start_batch"],
            config["max_batch"],
            config["step"],
            config["batches"],
            config["device_index"],
            config["cpu_threads"],
            config["cache"],
        )
        == ("1", "1024", "1", "100", "0", "1", "disabled"),
        "Mouse full sweep configuration differs",
    )
    _require(
        SHA256.fullmatch(config["model_sha256"]) is not None
        and config["workspace_bytes"].isdigit()
        and int(config["workspace_bytes"]) > 0,
        "Missing model/workspace identity",
    )
    artifact = _fields(lines[1], {"artifact_path", "cache_directory"})
    index = 2
    sdks = []
    while index < len(lines) and lines[index].startswith("gpu_sdk_directory "):
        sdks.append(_fields(lines[index], {"gpu_sdk_directory"})["gpu_sdk_directory"])
        index += 1
    _require(
        index < len(lines) and lines[index].startswith("device_identity "),
        "Missing device identity",
    )
    device = _fields(
        lines[index][len("device_identity ") :],
        set(["name", "uuid", "driver", "cuda", "tensorrt", "cache_namespace"]),
    )
    _require(
        all(device.values()) and SHA256.fullmatch(device["cache_namespace"]) is not None,
        "Incomplete device identity",
    )
    index += 1
    rows: list[Row] = []
    provider_precision: tuple[str, str] | None = None
    for line in lines[index:-1]:
        row = _fields(
            line,
            set(
                [
                    "batch",
                    "physical_batch",
                    "batches",
                    "nps",
                    "time_ms",
                    "initialization_ms",
                    "warm_ms",
                    "provider",
                    "precision",
                    "model_sha256",
                ]
            ),
        )
        _require(
            all(row[key].isdigit() for key in ("batch", "physical_batch", "batches")),
            "Invalid Mouse batch count",
        )
        batch = int(row["batch"])
        _require(
            int(row["physical_batch"]) == batch and row["batches"] == "100",
            "Padding or non-100-call Mouse point",
        )
        identity = (row["provider"], row["precision"])
        _require(
            identity in (("TensorrtExecutionProvider", "fp16"), ("CUDAExecutionProvider", "fp16")),
            "Unexpected provider/precision",
        )
        _require(provider_precision in (None, identity), "Provider/precision changed across sweep")
        provider_precision = identity
        _require(row["model_sha256"] == config["model_sha256"], "Model changed across sweep")
        for key in ("initialization_ms", "warm_ms"):
            _require(
                re.fullmatch(r"[0-9]+\.[0-9]{3}", row[key]) is not None
                and math.isfinite(float(row[key])),
                "Invalid warm/init duration",
            )
        nps, duration = _timing(row["nps"], row["time_ms"], batch * 100, fixed=True)
        rows.append(dict(batch=batch, physical=batch, batches=100, nps=nps, time_ms=duration))
    _sequence(rows)
    summary = _fields(lines[-1], {"best_batch", "best_nps", "cache"})
    best = _best(rows)
    _require(
        summary
        == {
            "best_batch": str(best["batch"]),
            "best_nps": f"{best['nps']:.3f}",
            "cache": "disabled",
        },
        "Native Mouse best summary disagrees with numeric rows",
    )
    config.update(provider=cast(tuple[str, str], provider_precision)[0], precision="fp16")
    return MouseSweep(config, artifact, tuple(sdks), device, rows)


def parse_reference(text: str) -> list[Row]:
    # Primary format: pinned Lc0 src/tools/backendbench.cc, six significant
    # digits under the default iostream precision. No upstream code is copied.
    pattern = re.compile(
        r"Benchmark batch size ([0-9]+) with inference average time (\S+)ms "
        r"- throughput (\S+) nps\."
    )
    rows: list[Row] = []
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        _require(match is not None, "Malformed/unexpected reference benchmark output")
        assert match is not None
        batch = int(match[1])
        nps, duration = _timing(match[3], match[2], batch, fixed=False)
        rows.append(dict(batch=batch, average_ms=duration, nps=nps))
    _sequence(rows)
    return rows


def _path(value: object) -> Path:
    _require(
        isinstance(value, str) and bool(value) and not any(ord(c) < 32 for c in value),
        "Invalid artifact path",
    )
    path = Path(cast(str, value))
    _require(path.is_absolute(), "Artifact paths must be absolute")
    return path.resolve()


def _strings(value: object) -> list[str]:
    _require(
        isinstance(value, list) and all(isinstance(v, str) for v in value),
        "Expected command argument vector",
    )
    return cast(list[str], value)


def _commands(
    state: dict[str, object], mouse: MouseSweep
) -> tuple[dict[str, Path], set[Path], Path]:
    command = _strings(state.get("mouse_command"))
    _require(
        len(command) > 2 and command[1] == "backendbench" and len(command) % 2 == 0,
        "Invalid Mouse command",
    )
    options: dict[str, str] = {}
    sdks: list[Path] = []
    for i in range(2, len(command), 2):
        key, value = command[i : i + 2]
        if key == "--gpu-dll-dir":
            sdks.append(_path(value))
        else:
            _require(key not in options, "Duplicate Mouse option")
            options[key] = value
    expected = {
        "--backend": "tensorrt"
        if mouse.config["provider"] == "TensorrtExecutionProvider"
        else "cuda",
        "--precision": "fp16",
        "--start-batch-size": "1",
        "--max-batch-size": "1024",
        "--batch-step": "1",
        "--batches": "100",
        "--cpu-threads": "1",
    }
    _require(
        set(options) == set(expected) | {"--weights", "--cache-dir"}
        and all(options.get(k) == v for k, v in expected.items()),
        "Mouse command configuration differs",
    )
    _require(
        len(sdks) == 3
        and len(set(sdks)) == 3
        and set(sdks) == {_path(v) for v in mouse.sdk_directories},
        "SDK directory declarations differ",
    )
    model = _path(options["--weights"])
    _require(
        model == _path(mouse.artifact["artifact_path"])
        and _path(options["--cache-dir"]) == _path(mouse.artifact["cache_directory"]),
        "Mouse artifact/cache path identity differs",
    )
    reference = _strings(state.get("reference_command"))
    _require(len(reference) > 2 and reference[1] == "backendbench", "Invalid reference command")
    ref_options: dict[str, str] = {}
    for option in reference[2:]:
        key, separator, value = option.partition("=")
        _require(bool(separator) and key not in ref_options, "Duplicate/malformed reference option")
        ref_options[key] = value
    ref_expected = {
        "--backend": "cuda-fp16",
        "--backend-opts": "max_batch=1024,min_batch=1",
        "--nncache": "0",
        "--threads": "1",
        "--start-batch-size": "1",
        "--max-batch-size": "1024",
        "--batch-step": "1",
        "--batches": "100",
        "--policy-softmax-temp": "1.359",
        "--history-fill-new": "fen_only",
    }
    _require(
        set(ref_options) == set(ref_expected) | {"--weights"}
        and all(ref_options.get(k) == v for k, v in ref_expected.items()),
        "Reference full sweep command differs",
    )
    artifacts = {
        "engine": _path(command[0]),
        "reference_engine": _path(reference[0]),
        "source_net": _path(ref_options["--weights"]),
        "onnx": model,
    }
    source_roots = [p.parent / "engine" for p in artifacts["engine"].parents if p.name == "build"]
    _require(len(source_roots) == 1, "Cannot identify declared build's engine source root")
    return (
        artifacts,
        set(sdks) | {artifacts["engine"].parent, artifacts["reference_engine"].parent},
        source_roots[0],
    )


def _hash_map(value: object) -> dict[Path, str]:
    result: dict[Path, str] = {}
    for name, digest in _dict(value).items():
        _require(
            isinstance(digest, str) and SHA256.fullmatch(digest) is not None,
            "Malformed declared SHA256",
        )
        path = _path(name)
        _require(path not in result, "Duplicate normalized artifact path")
        result[path] = cast(str, digest)
    _require(bool(result), "Empty declared hash inventory")
    return result


def _inventory(directories: set[Path], source_root: Path) -> tuple[set[Path], set[Path]]:
    _require(
        all(p.is_dir() for p in directories) and source_root.is_dir(),
        "Missing runtime/source directory",
    )
    libraries = {
        p.resolve()
        for directory in directories
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".dll"
    }
    sources = {
        p.resolve() for p in source_root.rglob("*") if p.is_file() and p.suffix in (".h", ".cpp")
    }
    return libraries, sources


def _verify(hashes: dict[Path, str]) -> None:
    for path, digest in hashes.items():
        _require(_hash(path) == digest, f"Frozen artifact changed: {path}")


def _stored_rows(value: object, raw: list[Row]) -> None:
    _require(isinstance(value, list) and len(value) == 1024, "Missing original row table")
    for stored, expected in zip(cast(list[object], value), raw, strict=True):
        row = _dict(stored)
        _require(set(row) == set(expected), "Original row table fields differ")
        for key, number in row.items():
            if key in ("batch", "physical", "batches"):
                _int(number, 1, 1024)
            else:
                _positive(number)
        _require(row == expected, "Original row table differs from raw stdout")


def _device(raw: dict[str, str], mouse: MouseSweep) -> None:
    observations = []
    for name in ("gpu-before.txt", "gpu-after.txt"):
        rows = list(csv.reader(io.StringIO(raw[name]), skipinitialspace=True))
        _require(
            len(rows) == 2 and len(rows[0]) == len(rows[1]) == 6,
            "Require single-GPU before/after observations",
        )
        row = [v.strip() for v in rows[1]]
        _require(
            row[:3] == [mouse.device["name"], mouse.device["uuid"], mouse.device["driver"]]
            and row[4] == "0 %",
            "GPU identity or idle observation differs",
        )
        observations.append(row[:4])
    _require(observations[0] == observations[1], "GPU hardware changed across sweeps")
    _require(
        "v0.32.1" in raw["reference-full.stderr.txt"]
        and f"GPU: {mouse.device['name']}\n" in raw["reference-full.stderr.txt"],
        "Missing pinned Lc0/GPU diagnostic identity",
    )


def _publish_new(path: Path, report: dict[str, object]) -> None:
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication and no-clobber on Windows/Linux.
    finally:
        Path(temporary).unlink(missing_ok=True)


def audit_sweep(raw_dir: Path, output: Path) -> dict[str, object]:
    raw_dir, output = raw_dir.resolve(), output.resolve()
    if output.is_relative_to(raw_dir) or output.exists() or not output.parent.is_dir():
        raise ThroughputError("Require a new report outside the immutable raw directory")
    report: dict[str, object] = {
        "schema": "MouseBackendThroughputAudit1",
        "status": "failed",
        "raw_evidence_valid": False,
        "controller_summary_valid": False,
        "raw_directory": str(raw_dir),
    }
    try:
        data: dict[str, bytes] = {}
        for name in RAW_FILES:
            path = raw_dir / name
            _require(path.stat().st_size <= 4 * 1024 * 1024, "Raw evidence file exceeds size limit")
            data[name] = path.read_bytes()
        raw_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}
        report["raw_file_sha256"] = raw_hashes
        raw = {
            name: value.decode("utf-8-sig").replace("\r\n", "\n") for name, value in data.items()
        }
        state = _dict(
            json.loads(raw["state.json"], object_pairs_hook=_pairs, parse_constant=_constant)
        )
        _require(
            type(state.get("schema")) is int
            and state["schema"] == 1
            and state.get("status") == "complete"
            and state.get("phase") == "reference-full",
            "Controller did not complete both sweeps",
        )
        counts = _dict(state.get("counts"))
        _require(
            set(counts) == set(COUNTS)
            and all(type(counts[k]) is int and counts[k] == v for k, v in COUNTS.items()),
            "Incomplete/shortened full sweep counts",
        )
        for name in ("mouse_exit", "reference_exit"):
            _int(state.get(name), 0, 0)
        _int(state.get("parallel_gpu_jobs"), 1, 1)
        for name in ("mouse_process_seconds", "reference_process_seconds"):
            _positive(state.get(name))
        mouse = parse_mouse(raw["mouse-full.stdout.txt"])
        artifacts, directories, source_root = _commands(state, mouse)
        frozen = _dict(state.get("frozen_sha256"))
        _require(
            set(frozen) == set(artifacts) and frozen.get("onnx") == mouse.config["model_sha256"],
            "Missing/different frozen artifact identities",
        )
        hashes: dict[Path, str] = {}
        for name, path in artifacts.items():
            digest = frozen[name]
            _require(
                isinstance(digest, str) and SHA256.fullmatch(digest) is not None,
                "Invalid frozen artifact hash",
            )
            hashes[path] = cast(str, digest)
        runtimes, sources = (
            _hash_map(state.get("runtime_sha256")),
            _hash_map(state.get("source_sha256")),
        )
        _require(
            _inventory(directories, source_root) == (set(runtimes), set(sources)),
            "Source/runtime file inventory changed or incomplete",
        )
        hashes.update(runtimes)
        hashes.update(sources)
        tool = Path(__file__).resolve()
        hashes[tool] = _hash(tool)
        _verify(hashes)
        reference = parse_reference(raw["reference-full.stdout.txt"])
        _stored_rows(state.get("mouse_rows"), mouse.rows)
        _stored_rows(state.get("reference_rows"), reference)
        _require(
            state.get("executable_build_git") == mouse.config["git"]
            and isinstance(state.get("source_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", cast(str, state["source_commit"])) is not None,
            "Source/build git identity differs",
        )
        expected_version = (
            f"Mouse {mouse.config['version']}\ngit: {mouse.config['git']}\n"
            f"build: {mouse.config['build']}\ncompiler: {mouse.config['compiler']}\n"
            f"ONNX Runtime: {mouse.config['ort_version']}\n"
        )
        _require(
            raw["mouse-version.txt"].replace("\r\n", "\n") == expected_version,
            "Version preflight/header mismatch",
        )
        source_lines = [
            line
            for line in raw["mouse-model.txt"].splitlines()
            if line.startswith("source_sha256:")
        ]
        _require(
            source_lines == ["source_sha256: " + cast(str, frozen["source_net"])],
            "ONNX/source-net identity differs",
        )
        _device(raw, mouse)
        best_mouse, best_reference = _best(mouse.rows), _best(reference)
        ratio = cast(float, best_mouse["nps"]) / cast(float, best_reference["nps"])
        corrected: dict[str, object] = dict(
            best_mouse=best_mouse,
            best_reference=best_reference,
            ratio=ratio,
            gate_80_percent=ratio >= 0.8,
            rc_backend_90_percent=ratio >= 0.9,
        )
        original = {key: state.get(key) for key in SUMMARY_FIELDS}
        mismatches = []
        for key in SUMMARY_FIELDS:
            if key == "ratio":
                value = original[key]
                same = (
                    not isinstance(value, bool)
                    and isinstance(value, int | float)
                    and 0 < value <= sys.float_info.max
                    and math.isclose(value, ratio, rel_tol=8 * sys.float_info.epsilon)
                )
            elif key.endswith("percent"):
                same = type(original[key]) is bool and original[key] == corrected[key]
            else:
                same = original[key] == corrected[key]
            if not same:
                mismatches.append(key)
        _require(
            _inventory(directories, source_root) == (set(runtimes), set(sources)),
            "Source/runtime inventory changed during audit",
        )
        _verify(hashes)
        _require(
            all(_hash(raw_dir / name) == digest for name, digest in raw_hashes.items()),
            "Raw evidence changed during audit",
        )
        report.update(corrected)
        report.update(
            status="audited",
            raw_evidence_valid=True,
            controller_summary_valid=not mismatches,
            controller_summary_mismatches=mismatches,
            original_controller_summary=original,
            mouse_rows=mouse.rows,
            reference_rows=reference,
            counts=COUNTS,
            native_exits={"mouse": 0, "reference": 0},
            mouse_configuration=mouse.config,
            mouse_device=mouse.device,
            mouse_command=state["mouse_command"],
            reference_command=state["reference_command"],
            source_commit=state["source_commit"],
            executable_build_git=state["executable_build_git"],
            frozen_sha256=frozen,
            runtime_sha256=state["runtime_sha256"],
            source_sha256=state["source_sha256"],
            auditor_sha256=hashes[tool],
            identity_rechecked_after_parsing=True,
            reference_physical_batch_evidence=(
                "cuda-fp16 max_batch=1024,min_batch=1; "
                "raw points1..1024 (not separately instrumented)"
            ),
            rounding_contract=(
                "Mouse fixed3; pinned Lc0 default6 significant digits; "
                "exact product interval intersection"
            ),
        )
    except (ThroughputError, OSError, ValueError, KeyError, IndexError, RecursionError) as error:
        report["error"] = str(error)
    try:
        _publish_new(output, report)
    except (OSError, ValueError) as error:
        raise ThroughputError(f"Cannot publish new audit report: {error}") from error
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = audit_sweep(args.raw_dir, args.output)
    except (ThroughputError, OSError) as error:
        print(f"Throughput audit error: {error}", file=sys.stderr)
        return 1
    if not report["raw_evidence_valid"]:
        print(f"Throughput audit failed: {report.get('error')}", file=sys.stderr)
        return 1
    print(
        f"Audited complete sweeps: ratio={report['ratio']}; "
        f"controller_summary_valid={report['controller_summary_valid']}; report={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
