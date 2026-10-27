# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate actual-Lc0 classical edge fixtures, sequentially on the CPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mouse_tools.encoding import FillHistory, HistoryFrame, encode_classical, policy_index
from mouse_tools.lc0_reference import (
    LC0_COMMIT,
    LC0_SHA256,
    MAX_TRAINING_BYTES,
    RECORD_BYTES,
    Lc0Reference,
    Lc0ReferenceError,
    capture_training,
    parse_training_v6,
)

FILLS: tuple[FillHistory, ...] = ("no", "fen_only", "always")


@dataclass(frozen=True)
class ReferenceCase:
    identifier: str
    fen: str
    moves_uci: tuple[str, ...]
    moves_san: tuple[str, ...]
    history: tuple[HistoryFrame, ...]


def parse_cases(value: object) -> tuple[ReferenceCase, ...]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "cases"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
    ):
        raise Lc0ReferenceError("Reference requests require schema 1 and cases")
    rows = value["cases"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 1000:
        raise Lc0ReferenceError("Reference requests require 1..1000 explicit cases")
    results = []
    names = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "id",
            "fen",
            "moves_uci",
            "moves_san",
            "history",
        }:
            raise Lc0ReferenceError("Malformed reference case")
        name = row["id"]
        if not isinstance(name, str) or re.fullmatch(r"[a-z0-9_]+", name) is None or name in names:
            raise Lc0ReferenceError("Case names must be distinct safe identifiers")
        names.add(name)
        if not isinstance(row["fen"], str):
            raise Lc0ReferenceError("Case FEN must be a string")
        HistoryFrame(row["fen"])
        for key in ("moves_uci", "moves_san"):
            if not isinstance(row[key], list) or any(
                not isinstance(move, str) for move in row[key]
            ):
                raise Lc0ReferenceError("Opening moves must be explicit string lists")
        if len(row["moves_uci"]) != len(row["moves_san"]):
            raise Lc0ReferenceError("UCI and SAN opening lengths differ")
        if any(
            re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is None for move in row["moves_uci"]
        ):
            raise Lc0ReferenceError("Malformed UCI opening")
        values = row["history"]
        if not isinstance(values, list) or len(values) != len(row["moves_uci"]) + 1:
            raise Lc0ReferenceError("Each opening ply requires an explicit history frame")
        frames = []
        for item in values:
            if (
                not isinstance(item, dict)
                or set(item) != {"fen", "repetitions"}
                or not isinstance(item["fen"], str)
                or type(item["repetitions"]) is not int
            ):
                raise Lc0ReferenceError("Malformed reference frame")
            frames.append(HistoryFrame(item["fen"], item["repetitions"]))
        if frames[0].fen != row["fen"]:
            raise Lc0ReferenceError("History must begin at the opening FEN")
        encode_classical(frames)
        results.append(
            ReferenceCase(
                name, row["fen"], tuple(row["moves_uci"]), tuple(row["moves_san"]), tuple(frames)
            )
        )
    return tuple(results)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode("ascii")).hexdigest()


def generate_corpus(engine: Path, weights: Path, requests: Path, output: Path, raw: Path) -> None:
    if requests.stat().st_size > 1024 * 1024:
        raise Lc0ReferenceError("Reference request file exceeds 1 MiB")
    cases = parse_cases(json.loads(requests.read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    policies: dict[str, dict[str, object]] = {}
    mappings: dict[str, dict[str, int]] = {}
    with Lc0Reference(engine, weights) as oracle:
        for case in cases:
            mapping, lines = oracle.policy(case.fen, case.moves_uci)
            black = case.history[-1].fen.split()[1] == "b"
            if any(policy_index(move, black=black) != index for move, index in mapping.items()):
                raise Lc0ReferenceError(f"Actual Lc0 policy mismatch: {case.identifier}")
            transcript = "\n".join(lines) + "\n"
            filename = case.identifier + ".policy.txt"
            (output / filename).write_text(transcript, encoding="utf-8", newline="\n")
            mappings[case.identifier] = mapping
            policies[case.identifier] = {
                "command": oracle.command,
                "fen": case.fen,
                "moves_uci": list(case.moves_uci),
                "uci_commands": [
                    "ucinewgame",
                    "position fen "
                    + case.fen
                    + (" moves " + " ".join(case.moves_uci) if case.moves_uci else ""),
                    "go nodes 32",
                ],
                "moves": mapping,
                "transcript_file": filename,
                "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            }
    entries = []
    normalized = []
    for case in cases:
        captures: dict[str, object] = {}
        for fill in FILLS:
            evidence = capture_training(
                engine,
                weights,
                case.fen,
                case.moves_san,
                raw / (case.identifier + "-" + fill),
                fill=fill,
            )
            with gzip.open(Path(str(evidence["chunk"])), "rb") as stream:
                data = stream.read(MAX_TRAINING_BYTES + 1)
            first = parse_training_v6(data)[0]
            if set(first.legal_indices) != set(mappings[case.identifier].values()):
                raise Lc0ReferenceError(f"Training/UCI legal index mismatch: {case.identifier}")
            expected = encode_classical(case.history, fill=fill)
            if first.planes != expected:
                differences = [
                    i for i, (a, b) in enumerate(zip(first.planes, expected, strict=True)) if a != b
                ]
                raise Lc0ReferenceError(
                    f"Actual input mismatch {case.identifier}/{fill}: planes {differences}"
                )
            filename = case.identifier + "." + fill + ".v6.gz"
            (output / filename).write_bytes(gzip.compress(data[:RECORD_BYTES], mtime=0))
            planes = [[plane.mask, plane.value] for plane in first.planes]
            captures[fill] = {
                **evidence,
                "first_record_file": filename,
                "first_record_sha256": hashlib.sha256(data[:RECORD_BYTES]).hexdigest(),
                "planes_sha256": _digest(planes),
            }
            normalized.append(
                [case.identifier, fill, planes, sorted(mappings[case.identifier].items())]
            )
        entries.append(
            {
                "id": case.identifier,
                "fen": case.fen,
                "history": [
                    {"fen": frame.fen, "repetitions": frame.repetitions} for frame in case.history
                ],
                "policy": policies[case.identifier],
                "training": captures,
            }
        )
        print(f"Verified actual Lc0 {case.identifier}: all 3 history fills", flush=True)
    result = {
        "schema": 1,
        "engine_sha256": LC0_SHA256,
        "source_commit": LC0_COMMIT,
        "backend": "trivial",
        "neural_weights_evaluated": False,
        "requests_sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
        "positions": len(cases),
        "training_records_selected": len(cases) * 3,
        "normalized_input_policy_sha256": _digest(normalized),
        "cases": entries,
    }
    (output / "reference.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Captured {len(cases)} cases; normalized SHA-256 {_digest(normalized)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("reference files must be read with python -I")
    try:
        generate_corpus(
            arguments.engine,
            arguments.weights,
            arguments.cases,
            arguments.output_dir,
            arguments.raw_dir,
        )
    except (Lc0ReferenceError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
