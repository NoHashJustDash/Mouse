# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict corpus requests and actual-binary fixture reproduction."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def corpus_module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.reference_corpus") is not None, (
        "Generator must exist"
    )
    return importlib.import_module("mouse_tools.reference_corpus")


def request() -> dict[str, object]:
    fen = "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"
    return {
        "schema": 1,
        "cases": [
            {
                "id": "promotion",
                "fen": fen,
                "moves_uci": [],
                "moves_san": [],
                "history": [{"fen": fen, "repetitions": 0}],
            }
        ],
    }


def test_corpus_requests_require_exact_safe_history_and_distinct_ids() -> None:
    module = corpus_module()
    parsed = module.parse_cases(request())
    assert len(parsed) == 1
    assert parsed[0].identifier == "promotion"
    for mode in ("duplicate", "path", "history", "moves", "extra"):
        value = json.loads(json.dumps(request()))
        if mode == "duplicate":
            value["cases"].append(value["cases"][0])
        elif mode == "path":
            value["cases"][0]["id"] = "../escape"
        elif mode == "history":
            value["cases"][0]["history"] = []
        elif mode == "moves":
            value["cases"][0]["moves_uci"] = ["a7a8q"]
        else:
            value["cases"][0]["ignored"] = True
        with pytest.raises(module.Lc0ReferenceError):
            module.parse_cases(value)


def test_actual_reference_edge_corpus_exists_and_parses_every_selected_record() -> None:
    corpus_module()
    from mouse_tools.encoding import HistoryFrame, encode_classical, policy_index
    from mouse_tools.lc0_reference import LC0_SHA256, parse_move_stats, read_training_v6

    root = Path(__file__).resolve().parents[2] / "tests/data/nn/lc0-classical-v1"
    assert (root / "reference.json").is_file(), "Generate actual reference edge corpus"
    data = json.loads((root / "reference.json").read_text(encoding="utf-8"))
    assert data["engine_sha256"] == LC0_SHA256
    assert data["positions"] >= 20
    assert data["training_records_selected"] == data["positions"] * 3
    assert (
        data["requests_sha256"]
        == hashlib.sha256((root.parent / "cases.json").read_bytes()).hexdigest()
    )
    normalized = []
    for case in data["cases"]:
        frames = [HistoryFrame(frame["fen"], frame["repetitions"]) for frame in case["history"]]
        transcript = (root / (case["id"] + ".policy.txt")).read_text(encoding="utf-8")
        assert (
            hashlib.sha256(transcript.encode("utf-8")).hexdigest()
            == case["policy"]["transcript_sha256"]
        )
        mapping = parse_move_stats(
            transcript.splitlines(), expected_moves=list(case["policy"]["moves"])
        )
        black = frames[-1].fen.split()[1] == "b"
        assert mapping == {move: policy_index(move, black=black) for move in mapping}
        for fill, capture in case["training"].items():
            raw = gzip.decompress((root / capture["first_record_file"]).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == capture["first_record_sha256"]
            records = read_training_v6(root / capture["first_record_file"])
            assert len(records) == 1
            assert records[0].planes == encode_classical(frames, fill=fill)
            assert set(records[0].legal_indices) == set(mapping.values())
            planes = [[plane.mask, plane.value] for plane in records[0].planes]
            encoded = json.dumps(planes, separators=(",", ":")).encode("ascii")
            assert hashlib.sha256(encoded).hexdigest() == capture["planes_sha256"]
            normalized.append([case["id"], fill, planes, sorted(mapping.items())])
    encoded = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    assert hashlib.sha256(encoded).hexdigest() == data["normalized_input_policy_sha256"]


def test_equivalent_fen_start_padding_matches_actual_binary_record() -> None:
    from mouse_tools.encoding import HistoryFrame, encode_classical
    from mouse_tools.lc0_reference import LC0_SHA256, read_training_v6

    root = Path(__file__).resolve().parents[2] / "tests/data/nn"
    evidence = json.loads((root / "equivalent-start.json").read_text(encoding="utf-8"))
    assert evidence["engine_sha256"] == LC0_SHA256
    packed = (root / evidence["first_record_file"]).read_bytes()
    assert hashlib.sha256(gzip.decompress(packed)).hexdigest() == evidence["first_record_sha256"]
    records = read_training_v6(root / evidence["first_record_file"])
    assert len(records) == 1
    assert records[0].planes == encode_classical([HistoryFrame(evidence["fen"])], fill="fen_only")
