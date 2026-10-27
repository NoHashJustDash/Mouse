# SPDX-License-Identifier: GPL-3.0-or-later
"""Freeze the approved 27 actual-Lc0 edges plus 23 distinct accepted inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mouse_tools.lc0_reference import read_training_v6

EDGE_SHA256 = "5dee45734358db88d6818a59ae334e41936d27d88f52d49beb889606205fe865"
PROBE_SHA256 = "5df515c842a956b477d94e066d07695fde83b0ffe724f6631fc477d236b604c8"
NPY_SHA256 = "e09f29c803a5cb896c150e3055b1770490a9696561c223b226e42a22cb856632"
META_SHA256 = "3d7c69b4f410cf921c9d5fb4295297f4379fdc81a2992175fc3ee271ca179107"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(edges: Path, probe: Path, inputs: Path, metadata: Path, output: Path) -> None:
    sources = {
        "edge_manifest": edges / "reference.json",
        "probe": probe,
        "accepted_inputs": inputs,
        "accepted_metadata": metadata,
    }
    frozen = {name: _hash(path) for name, path in sources.items()}
    for name, expected in [
        ("edge_manifest", EDGE_SHA256),
        ("probe", PROBE_SHA256),
        ("accepted_inputs", NPY_SHA256),
        ("accepted_metadata", META_SHA256),
    ]:
        if frozen[name] != expected:
            raise ValueError(f"{name} does not match the approved source SHA256")
    if output.exists():
        raise ValueError("Freeze output must be a new directory")
    edge_data = json.loads(sources["edge_manifest"].read_text(encoding="utf-8"))
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    array = np.load(inputs, allow_pickle=False)
    if (
        array.shape != (1000, 112, 8, 8)
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise ValueError("Accepted array shape/type/finiteness mismatch")
    records = {
        row["id"]: row for row in map(json.loads, probe.read_text(encoding="utf-8").splitlines())
    }
    cases = []
    seen = set()
    chunks = {}
    for case in edge_data["cases"]:
        capture = case["training"]["fen_only"]
        path = edges / capture["first_record_file"]
        chunks[path.name] = _hash(path)
        import gzip

        raw = gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest() != capture["first_record_sha256"]:
            raise ValueError("Actual Lc0 training record hash mismatch")
        (record,) = read_training_v6(path)
        planes = [[p.mask, p.value] for p in record.planes]
        dense = np.array([p.dense() for p in record.planes], dtype="<f4").tobytes()
        key = hashlib.sha256(dense).hexdigest()
        if key in seen:
            raise ValueError(f"Duplicate named edge tensor: {case['id']}; retain and escalate")
        seen.add(key)
        cases.append(
            {
                "id": case["id"],
                "history": case["history"],
                "planes": planes,
                "moves": sorted(case["policy"]["moves"].items()),
                "tensor_sha256": key,
                "source": {"kind": "lc0-reference", "id": case["id"]},
            }
        )
    if len(cases) != 27:
        raise ValueError("Retain exactly all 27 named actual Lc0 cases")
    for selected, tensor in zip(meta["selected"], array, strict=True):
        raw = tensor.astype("<f4", copy=False).tobytes()
        key = hashlib.sha256(raw).hexdigest()
        if key != selected["tensor_sha256_le_f32"]:
            raise ValueError("Accepted tensor selection hash mismatch")
        if key in seen:
            continue
        source_record = records[selected["id"]]
        dense = np.array(
            [
                [v if (mask >> i) & 1 else 0 for i in range(64)]
                for mask, v in source_record["planes"]
            ],
            dtype="<f4",
        ).tobytes()
        if dense != raw:
            raise ValueError("Accepted array and source probe planes disagree")
        seen.add(key)
        cases.append(
            {
                "id": f"corpus_{source_record['id']:06d}",
                "history": source_record["frames"],
                "planes": source_record["planes"],
                "moves": sorted(source_record["moves"]),
                "tensor_sha256": key,
                "source": {"kind": "accepted-1000", "id": source_record["id"]},
            }
        )
        if len(cases) == 50:
            break
    if len(cases) != 50 or len(seen) != 50:
        raise ValueError("Need exactly 50 distinct complete tensors without padding")
    if {name: _hash(path) for name, path in sources.items()} != frozen:
        raise ValueError("Source artifact changed during freeze")
    if any(_hash(edges / name) != value for name, value in chunks.items()):
        raise ValueError("Actual reference chunk changed during freeze")
    payload = {"schema": 1, "positions": 50, "history_fill": "fen_only", "cases": cases}
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    binary = (
        b"MOUSEIN1"
        + struct.pack("<II", 50, 112)
        + b"".join(
            struct.pack("<Qf", mask, value) for case in cases for mask, value in case["planes"]
        )
    )
    provenance = {
        "schema": 1,
        "positions": 50,
        "named_edges": 27,
        "additional_accepted": 23,
        "distinct_tensors": 50,
        "deduplication": "exact complete little-endian float32 tensors",
        "source_sha256": frozen,
        "training_chunks_sha256": chunks,
        "cases_sha256": hashlib.sha256(encoded).hexdigest(),
        "inputs_sha256": hashlib.sha256(binary).hexdigest(),
        "selected_source_ids": [c["source"]["id"] for c in cases[27:]],
    }
    output.mkdir(parents=True)
    (output / "cases.json").write_bytes(encoded)
    (output / "planes.bin").write_bytes(binary)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(provenance, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    if not sys.flags.isolated:
        print("Fixture freeze requires python -I", file=sys.stderr)
        return 2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    try:
        freeze(a.edges, a.probe, a.inputs, a.metadata, a.output)
    except (OSError, ValueError) as error:
        print(f"fixture freeze failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
