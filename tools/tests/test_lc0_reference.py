# SPDX-License-Identifier: GPL-3.0-or-later
"""Strict actual-Lc0 verbose and training-v6 reference contracts."""

from __future__ import annotations

import importlib
import importlib.util
import struct
from pathlib import Path
from types import ModuleType

import pytest


def reference_module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.lc0_reference") is not None, "Harness must exist"
    return importlib.import_module("mouse_tools.lc0_reference")


def test_verbose_move_rows_use_binary_indices_and_require_complete_output() -> None:
    module = reference_module()
    lines = [
        "info string a7a8n (1401) N: 1",
        "info string a7a8q (1792) N: 5",
        "info string node (2) N: 32",
        "bestmove a7a8q",
    ]
    assert module.parse_move_stats(lines, expected_moves=["a7a8n", "a7a8q"]) == {
        "a7a8n": 1401,
        "a7a8q": 1792,
    }
    for bad in [
        lines[:-1],
        lines + [lines[0]],
        lines[:1] + lines[2:],
        [line.replace("1401", "1858") for line in lines],
    ]:
        with pytest.raises(module.Lc0ReferenceError):
            module.parse_move_stats(bad, expected_moves=["a7a8n", "a7a8q"])


def training_record() -> bytes:
    record = bytearray(8356)
    struct.pack_into("<II", record, 0, 6, 1)
    struct.pack_into("<1858f", record, 8, *([-1.0] * 1858))
    struct.pack_into("<f", record, 8 + 1401 * 4, 1.0)
    struct.pack_into("<Q", record, 7440, 0x80)  # Stored byte bits reverse to A1.
    record[8272:8278] = bytes([1, 0, 0, 1, 1, 87])
    struct.pack_into("<IHH", record, 8340, 32, 1401, 1401)
    return bytes(record)


def test_training_v6_decodes_every_plane_and_preserves_raw_rule50() -> None:
    module = reference_module()
    records = module.parse_training_v6(training_record())
    assert len(records) == 1
    record = records[0]
    assert record.planes[0].mask == 1
    assert [plane.mask != 0 for plane in record.planes[104:109]] == [True, False, False, True, True]
    assert record.planes[109].value == 87.0
    assert record.legal_indices == (1401,)
    assert record.played_index == record.best_index == 1401
    assert record.visits == 32


@pytest.mark.parametrize("mutation", ["truncated", "version", "format", "flags", "nan", "index"])
def test_training_v6_rejects_incomplete_or_inconsistent_records(mutation: str) -> None:
    module = reference_module()
    record = bytearray(training_record())
    if mutation == "truncated":
        record.pop()
    elif mutation == "version":
        record[0] = 5
    elif mutation == "format":
        record[4] = 2
    elif mutation == "flags":
        record[8272] = 2
    elif mutation == "nan":
        struct.pack_into("<f", record, 8, float("nan"))
    else:
        struct.pack_into("<H", record, 8344, 1858)
    with pytest.raises(module.Lc0ReferenceError):
        module.parse_training_v6(bytes(record))


def test_selfplay_command_only_uses_options_supported_by_that_mode() -> None:
    from pathlib import Path

    module = reference_module()
    command = module._arguments(Path("lc0.exe"), Path("weights.pb.gz"), "fen_only", "selfplay")
    assert not any(option.startswith("--config") for option in command)
    assert "--backend=trivial" in command


def test_reference_identity_is_checked_before_any_process_launch(tmp_path: Path) -> None:
    module = reference_module()
    binary = tmp_path / "wrong.exe"
    weights = tmp_path / "wrong.pb.gz"
    binary.write_bytes(b"untrusted executable fixture")
    weights.write_bytes(b"untrusted weights fixture")
    with pytest.raises(module.Lc0ReferenceError, match="SHA-256"):
        module.Lc0Reference(binary, weights)
