# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate the fixed 10000-position C++ encoder probe against Python and Stockfish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from mouse_tools.encoding import (
    EncodingError,
    HistoryFrame,
    encode_classical,
    move_from_policy,
    policy_index,
)
from mouse_tools.perft_reference import (
    REFERENCE_NAME,
    REFERENCE_SHA256,
    ReferenceError,
    UciReference,
)

POSITIONS = 10_000


class EncodingGateError(ValueError):
    """A malformed or incomplete cross-language probe is never a gate pass."""


def _frames(record: object) -> tuple[HistoryFrame, ...]:
    if not isinstance(record, dict) or set(record) != {"id", "frames", "planes", "moves"}:
        raise EncodingGateError("Probe record must contain id, frames, planes, and moves")
    if type(record["id"]) is not int or not 0 <= record["id"] < 2**32:
        raise EncodingGateError("Probe ID must be an unsigned 32-bit integer")
    values = record["frames"]
    if not isinstance(values, list) or not 1 <= len(values) <= 8:
        raise EncodingGateError("Probe requires 1..8 chronological history frames")
    frames = []
    for value in values:
        if (
            not isinstance(value, dict)
            or set(value) != {"fen", "repetitions"}
            or not isinstance(value["fen"], str)
            or type(value["repetitions"]) is not int
        ):
            raise EncodingGateError("Malformed probe history frame")
        try:
            frames.append(HistoryFrame(value["fen"], value["repetitions"]))
        except EncodingError as error:
            raise EncodingGateError(str(error)) from error
    return tuple(frames)


def validate_probe_record(record: object, legal_moves: Sequence[str]) -> int:
    frames = _frames(record)
    if not isinstance(record, dict):
        raise EncodingGateError("Probe record must be an object")
    try:
        expected_planes = encode_classical(frames)
        values = record["planes"]
        if not isinstance(values, list) or len(values) != 112:
            raise EncodingGateError("Probe must provide every one of 112 planes")
        for index, (value, expected) in enumerate(zip(values, expected_planes, strict=True)):
            if (
                not isinstance(value, list)
                or len(value) != 2
                or type(value[0]) is not int
                or type(value[1]) not in (int, float)
                or not math.isfinite(value[1])
            ):
                raise EncodingGateError(f"Malformed plane {index}")
            if value[0] != expected.mask or value[1] != expected.value:
                raise EncodingGateError(
                    f"Input plane {index} differs from independent Python encoder"
                )
        moves = record["moves"]
        if not isinstance(moves, list):
            raise EncodingGateError("Probe moves must be complete rows")
        parsed: dict[str, int] = {}
        black = frames[-1].fen.split()[1] == "b"
        for row in moves:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not isinstance(row[0], str)
                or type(row[1]) is not int
                or row[0] in parsed
                or row[1] in parsed.values()
            ):
                raise EncodingGateError("Malformed or duplicate move/index")
            if policy_index(row[0], black=black) != row[1]:
                raise EncodingGateError(f"Policy index differs for {row[0]}")
            parsed[row[0]] = row[1]
        if len(legal_moves) != len(set(legal_moves)) or set(parsed) != set(legal_moves):
            raise EncodingGateError("Probe rows differ from the oracle's complete legal move set")
        for move, index in parsed.items():
            if move_from_policy(index, legal_moves, black=black) != move:
                raise EncodingGateError(f"Policy inverse differs for {move}")
        return len(parsed)
    except EncodingError as error:
        raise EncodingGateError(str(error)) from error


def _reject_constant(value: str) -> None:
    raise EncodingGateError(f"Nonfinite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EncodingGateError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_probe_line(line: str) -> dict[str, object]:
    if len(line) > 128 * 1024:
        raise EncodingGateError("Oversized probe record")
    value = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    _frames(value)
    if not isinstance(value, dict):
        raise EncodingGateError("Probe record must be an object")
    return value


def _write_report(output: Path, report: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=output.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2)
            stream.write("\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            with suppress(FileNotFoundError):
                temporary.unlink()


def validate_probe(probe: Path, oracle: Path, output: Path) -> int:
    results: list[dict[str, object]] = []
    report: dict[str, object] = {
        "schema": 1,
        "gate": "v0.2-input-policy-10000",
        "status": "running",
        "probe": str(probe.resolve()),
        "oracle": str(oracle.resolve()),
        "oracle_name": REFERENCE_NAME,
        "oracle_sha256": REFERENCE_SHA256,
        "required_positions": POSITIONS,
        "positions_verified": 0,
        "planes_verified": 0,
        "moves_verified": 0,
        "oracle_queries": 0,
        "oracle_count_cache": False,
        "oracle_threads": 1,
        "uci_chess960": True,
        "results": results,
        "query": [
            "setoption name UCI_Chess960 value true",
            "position fen <current FEN>",
            "d",
            "go perft 1",
        ],
    }
    verified = queries = moves_verified = white = black = 0
    try:
        if output.resolve() in (probe.resolve(), oracle.resolve()):
            raise EncodingGateError("Report output must differ from the probe and oracle")
        if probe.stat().st_size > 512 * 1024 * 1024:
            raise EncodingGateError("Probe exceeds the bounded 512 MiB input size")
        with probe.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report["probe_sha256"] = digest
        with probe.open(encoding="utf-8") as stream:
            count = sum(1 for _ in stream)
        if count != POSITIONS:
            raise EncodingGateError(
                f"Complete gate requires exactly {POSITIONS} records; got {count}"
            )
        seen: set[int] = set()
        with UciReference(oracle) as reference, probe.open(encoding="utf-8") as stream:
            for line in stream:
                if len(line) > 128 * 1024:
                    raise EncodingGateError("Oversized probe record")
                record = parse_probe_line(line)
                frames = _frames(record)
                identifier = record["id"]
                if type(identifier) is not int:
                    raise EncodingGateError("Probe ID must be an integer")
                if identifier in seen:
                    raise EncodingGateError(f"Duplicate probe ID {record['id']}")
                seen.add(identifier)
                fen = frames[-1].fen
                independent = reference.perft(fen, "chess960", 1)
                queries += 1
                moves = validate_probe_record(record, independent.moves)
                verified += 1
                moves_verified += moves
                black += int(fen.split()[1] == "b")
                white += int(fen.split()[1] == "w")
                results.append(
                    {
                        "id": record["id"],
                        "status": "passed",
                        "fen": fen,
                        "oracle_legal_moves": list(independent.moves),
                        "moves_verified": moves,
                    }
                )
        if not black or not white:
            raise EncodingGateError("Complete corpus must contain both sides to move")
        with probe.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise EncodingGateError("Probe changed during verification")
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted corpus is not a gate pass"
    except (EncodingGateError, ReferenceError, ValueError, OSError) as error:
        report["status"] = "failed"
        report["error"] = str(error)
    report.update(
        positions_verified=verified,
        planes_verified=verified * 112,
        moves_verified=moves_verified,
        oracle_queries=queries,
        white=white,
        black=black,
    )
    if output.resolve() not in (probe.resolve(), oracle.resolve()):
        _write_report(output, report)
    if report["status"] != "passed":
        print(f"error: {report.get('error')}", file=sys.stderr)
        return 130 if report["status"] == "interrupted" else 1
    print(
        f"Verified {verified} histories, {verified * 112} planes, {moves_verified} legal moves; "
        f"{queries} uncached oracle queries"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("probe and oracle files must be read with python -I")
    try:
        return validate_probe(arguments.probe, arguments.oracle, arguments.output)
    except OSError as error:
        print(f"error: cannot preserve validation evidence: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
