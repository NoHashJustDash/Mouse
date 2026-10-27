# SPDX-License-Identifier: GPL-3.0-or-later
"""Legacy-TM table integrity and the exact planned elapsed-time gate."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mouse_tools import time_parity as api


def table(path: Path, elapsed: float, *, count: int = 96) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(api.COLUMNS)
        for case in api.grid()[:count]:
            writer.writerow((*case, elapsed, "", "d2d4", "a" * 64, "b" * 64))


@pytest.mark.parametrize(
    "reference,mouse,passed",
    [(50, 65, True), (50, 65.001, False), (200, 220, True), (200, 220.001, False)],
)
def test_elapsed_gate_uses_the_unmodified_larger_tolerance(
    tmp_path: Path,
    reference: float,
    mouse: float,
    passed: bool,
) -> None:
    table(tmp_path / "ref.csv", reference)
    table(tmp_path / "mouse.csv", mouse)
    failures = api.compare(tmp_path / "ref.csv", tmp_path / "mouse.csv", tmp_path / "result.csv")
    assert failures == (0 if passed else 96)


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "wrong-grid", "nan", "identity"])
def test_complete_timing_evidence_rejects_missing_or_inconsistent_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "table.csv"
    table(path, 100, count=95 if mutation == "partial" else 96)
    rows = list(csv.reader(path.open(newline="", encoding="utf-8")))
    if mutation == "duplicate":
        rows[-1] = rows[1]
    if mutation == "wrong-grid":
        rows[-1][0] = "999999"
    if mutation == "nan":
        rows[-1][4] = "nan"
    if mutation == "identity":
        rows[-1][-1] = "c" * 64
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        api._table(path)


def test_failed_native_measurement_never_publishes_a_completed_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary, model, result = (tmp_path / name for name in ("binary", "model", "result.csv"))
    binary.write_bytes(b"controlled timing binary identity")
    model.write_bytes(b"controlled timing model identity")

    def failed(*args: object) -> tuple[float, str, str]:
        raise ValueError("native measurement failed")

    monkeypatch.setattr(api, "measure", failed)
    with pytest.raises(ValueError, match="native measurement failed"):
        api.capture(binary, model, "mouse", api.digest(binary), result)
    assert not result.exists()
    assert result.with_suffix(".csv.pending").exists()


def test_timing_capture_checks_binary_identity_before_running_or_writing(tmp_path: Path) -> None:
    binary, model, result = (tmp_path / name for name in ("binary", "model", "result.csv"))
    binary.write_bytes(b"native identity")
    model.write_bytes(b"model identity")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        api.capture(binary, model, "lc0", "0" * 64, result)
    assert not result.exists()
    assert not result.with_suffix(".csv.pending").exists()
