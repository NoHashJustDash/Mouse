# SPDX-License-Identifier: GPL-3.0-or-later
"""Complete cross-language gate rejects missing data and incomplete legal sets."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest


def gate_module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.encoding_gate") is not None, "Gate must exist"
    return importlib.import_module("mouse_tools.encoding_gate")


def start_record() -> dict[str, object]:
    from mouse_tools.encoding import HistoryFrame, encode_classical, policy_index
    from mouse_tools.perft_reference import START_FEN

    moves = [file + "2" + file + rank for file in "abcdefgh" for rank in "34"]
    moves += ["b1a3", "b1c3", "g1f3", "g1h3"]
    return {
        "id": 0,
        "frames": [{"fen": START_FEN, "repetitions": 0}],
        "planes": [
            [plane.mask, plane.value] for plane in encode_classical([HistoryFrame(START_FEN)])
        ],
        "moves": [[move, policy_index(move)] for move in moves],
    }


def test_record_checks_every_plane_and_independent_complete_legal_set() -> None:
    module = gate_module()
    record = start_record()
    legal = [str(pair[0]) for pair in cast(list[list[str | int]], record["moves"])]
    assert module.validate_probe_record(record, legal) == 20
    for field in ("planes", "moves"):
        altered = dict(record)
        altered[field] = []
        with pytest.raises(module.EncodingGateError):
            module.validate_probe_record(altered, legal)
    mutated = json.loads(json.dumps(record))
    mutated["planes"][57][0] = 1
    with pytest.raises(module.EncodingGateError, match="plane"):
        module.validate_probe_record(mutated, legal)
    mutated = json.loads(json.dumps(record))
    mutated["moves"].pop()
    with pytest.raises(module.EncodingGateError, match="legal"):
        module.validate_probe_record(mutated, legal)


def test_partial_probe_corpus_fails_before_opening_the_oracle(tmp_path: Path) -> None:
    module = gate_module()
    path = tmp_path / "partial.jsonl"
    path.write_text(json.dumps(start_record()) + "\n", encoding="utf-8")
    output = tmp_path / "result.json"
    assert module.validate_probe(path, tmp_path / "absent-stockfish.exe", output) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "10000" in report["error"]
    assert report["oracle_queries"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "nan", "boolean", "oversized", "unknown"])
def test_probe_json_rejects_ambiguous_or_unbounded_records(mutation: str) -> None:
    module = gate_module()
    assert hasattr(module, "parse_probe_line"), "Probe JSON must have one strict entry point"
    line = json.dumps(start_record())
    if mutation == "duplicate":
        line = line.replace('"id": 0', '"id": 1, "id": 0')
    elif mutation == "nan":
        line = line.replace('"id": 0', '"id": NaN')
    elif mutation == "boolean":
        line = line.replace('"id": 0', '"id": true')
    elif mutation == "oversized":
        line += " " * (128 * 1024)
    else:
        line = line.replace('"id": 0', '"id": 0, "ignored": 1')
    with pytest.raises(module.EncodingGateError):
        module.parse_probe_line(line)
