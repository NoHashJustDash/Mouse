# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed parity input selection, exact byte deduplication, and safe artifacts."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def parity_module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.make_parity_inputs") is not None, (
        "Helper must exist"
    )
    return importlib.import_module("mouse_tools.make_parity_inputs")


def contract_fixture(tmp_path: Path, *, distinct: bool = True) -> tuple[Path, Path]:
    """Writer protocol data only; it does not claim an encoder or engine gate."""
    import hashlib

    from mouse_tools.perft_reference import REFERENCE_SHA256, START_FEN

    probe = tmp_path / "probe.jsonl"
    results = []
    with probe.open("w", encoding="utf-8", newline="\n") as stream:
        for identifier in range(10000):
            # First100 records repeat exactly; require later rows to reach1000 unique.
            mask = max(0, identifier - 99) if distinct else 0
            record = {
                "id": identifier,
                "frames": [{"fen": START_FEN, "repetitions": 0}],
                "planes": [[mask, 2.0], *([[0, 1.0]] * 111)],
                "moves": [],
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            results.append(
                {
                    "id": identifier,
                    "status": "passed",
                    "fen": START_FEN,
                    "oracle_legal_moves": [],
                    "moves_verified": 0,
                }
            )
    digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    report = tmp_path / "gate.json"
    report.write_text(
        json.dumps(
            {
                "schema": 1,
                "gate": "v0.2-input-policy-10000",
                "status": "passed",
                "probe_sha256": digest,
                "required_positions": 10000,
                "positions_verified": 10000,
                "planes_verified": 1120000,
                "oracle_queries": 10000,
                "oracle_sha256": REFERENCE_SHA256,
                "oracle_count_cache": False,
                "white": 5000,
                "black": 5000,
                "results": results,
            }
        ),
        encoding="utf-8",
    )
    return probe, report


def test_parity_inputs_select_1000_distinct_complete_float32_tensors(tmp_path: Path) -> None:
    module = parity_module()
    probe, report = contract_fixture(tmp_path)
    output, metadata = tmp_path / "inputs.npy", tmp_path / "inputs.json"
    module.build_inputs(probe, report, output, metadata)
    array = np.load(output, allow_pickle=False)
    assert array.shape == (1000, 112, 8, 8)
    assert array.dtype == np.dtype("float32")
    assert np.count_nonzero(array[0]) == 0
    assert array[1, 0, 0, 0] == 2.0
    assert len({record.tobytes() for record in array}) == 1000
    provenance = json.loads(metadata.read_text(encoding="utf-8"))
    assert len(provenance["selected"]) == 1000
    assert provenance["selected"][0]["id"] == 0
    assert provenance["selected"][-1]["id"] == 1098
    assert provenance["source_positions"] == 10000


@pytest.mark.parametrize(
    "mutation", ["failed", "count", "sha", "duplicate-output", "source-output"]
)
def test_invalid_gate_or_path_collision_cannot_publish_inputs(
    tmp_path: Path, mutation: str
) -> None:
    module = parity_module()
    probe, report = contract_fixture(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    output, metadata = tmp_path / "inputs.npy", tmp_path / "inputs.json"
    if mutation == "failed":
        data["status"] = "failed"
    elif mutation == "count":
        data["positions_verified"] = 9999
    elif mutation == "sha":
        data["probe_sha256"] = "0" * 64
    elif mutation == "duplicate-output":
        metadata = output
    else:
        output = probe
    report.write_text(json.dumps(data), encoding="utf-8")
    original = probe.read_bytes()
    with pytest.raises(module.ParityInputError):
        module.build_inputs(probe, report, output, metadata)
    assert probe.read_bytes() == original
    assert not metadata.exists()
    if output != probe:
        assert not output.exists()


def test_insufficient_distinct_tensors_fail_without_duplicate_padding(tmp_path: Path) -> None:
    module = parity_module()
    probe, report = contract_fixture(tmp_path, distinct=False)
    output, metadata = tmp_path / "inputs.npy", tmp_path / "inputs.json"
    with pytest.raises(module.ParityInputError, match="1000 distinct"):
        module.build_inputs(probe, report, output, metadata)
    assert not output.exists()
    assert not metadata.exists()


@pytest.mark.parametrize("white,black", [(-1, 10001), (10001, -1)])
def test_gate_requires_positive_counts_for_both_colors(
    tmp_path: Path, white: int, black: int
) -> None:
    module = parity_module()
    probe, report = contract_fixture(tmp_path)
    data = json.loads(report.read_text(encoding="utf-8"))
    data.update(white=white, black=black)
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(module.ParityInputError, match="both colors"):
        module.build_inputs(probe, report, tmp_path / "out.npy", tmp_path / "out.json")


def test_failed_second_publication_removes_own_first_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = parity_module()
    probe, report = contract_fixture(tmp_path)
    output, metadata = tmp_path / "inputs.npy", tmp_path / "inputs.json"
    original = module.os.link

    def fail_metadata(source: Path, destination: Path) -> None:
        if destination == metadata:
            raise OSError("simulated provenance publication failure")
        original(source, destination)

    monkeypatch.setattr(module.os, "link", fail_metadata)
    with pytest.raises(OSError, match="publication failure"):
        module.build_inputs(probe, report, output, metadata)
    assert not output.exists()
    assert not metadata.exists()
    assert set(tmp_path.iterdir()) == {probe, report}


def test_hard_link_to_source_is_rejected_without_modifying_it(tmp_path: Path) -> None:
    module = parity_module()
    probe, report = contract_fixture(tmp_path)
    output, metadata = tmp_path / "inputs.npy", tmp_path / "inputs.json"
    output.hardlink_to(probe)
    original = probe.read_bytes()
    with pytest.raises(module.ParityInputError, match="hard-link"):
        module.build_inputs(probe, report, output, metadata)
    assert probe.read_bytes() == output.read_bytes() == original
    assert not metadata.exists()
