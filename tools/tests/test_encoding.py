# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent classical plane and geometric policy contracts."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

import pytest

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ALL = (1 << 64) - 1


def encoding_module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.encoding") is not None, "Encoder must exist"
    return importlib.import_module("mouse_tools.encoding")


def test_classical_start_planes_and_explicit_fill() -> None:
    module = encoding_module()
    frame = module.HistoryFrame(START)
    planes = module.encode_classical([frame])
    assert len(planes) == 112
    assert planes[0].mask == 0xFF00
    assert planes[1].mask == 0x42
    assert planes[5].mask == 0x10
    assert planes[6].mask == 0x00FF000000000000
    assert all(plane.mask == 0 for plane in planes[13:104])
    assert [plane.mask for plane in planes[104:]] == [ALL] * 4 + [0, ALL, 0, ALL]
    assert planes[109].value == 0
    filled = module.encode_classical([frame], fill="always")
    assert filled[:13] == filled[13:26] == filled[91:104]
    assert len(filled[111].dense()) == 64
    assert filled[111].dense() == (1.0,) * 64


def test_black_current_perspective_repetition_and_raw_rule50() -> None:
    module = encoding_module()
    frame = module.HistoryFrame("4k3/8/3n4/8/8/2B5/8/4K3 b - - 87 60", 1)
    planes = module.encode_classical([frame], fill="no")
    assert planes[1].mask == 1 << 19  # d6 becomes d3.
    assert planes[8].mask == 1 << 42  # c3 becomes c6.
    assert planes[12].mask == ALL
    assert planes[108].mask == ALL
    assert planes[109].value == 87.0
    assert planes[109].dense() == (87.0,) * 64


def test_ep_fen_reconstructs_one_missing_pawn_history_even_with_no_fill() -> None:
    module = encoding_module()
    frame = module.HistoryFrame("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    planes = module.encode_classical([frame], fill="no")
    assert planes[6].mask == 1 << 35
    assert planes[19].mask == 1 << 51
    assert planes[13].mask == 1 << 36
    assert all(plane.mask == 0 for plane in planes[26:104])
    repeated = module.encode_classical([frame], fill="fen_only")
    assert repeated[19].mask == repeated[32].mask == repeated[97].mask == 1 << 51


@pytest.mark.parametrize("format_id", [0, 2, 3, 132])
def test_only_actual_classical_input_format_is_accepted(format_id: int) -> None:
    module = encoding_module()
    with pytest.raises(module.EncodingError, match="format"):
        module.encode_classical([module.HistoryFrame(START)], input_format=format_id)


@pytest.mark.parametrize("fen", [START + " extra", START.replace("0 1", "9" * 5000 + " 1")])
def test_frame_rejects_malformed_fen(fen: str) -> None:
    module = encoding_module()
    with pytest.raises(module.EncodingError):
        module.HistoryFrame(fen)


def test_history_and_fill_inputs_are_explicit_and_validated() -> None:
    module = encoding_module()
    with pytest.raises(module.EncodingError):
        module.encode_classical([])
    with pytest.raises(module.EncodingError):
        module.encode_classical([module.HistoryFrame(START)], fill="automatic")
    with pytest.raises(module.EncodingError):
        module.HistoryFrame(START, -1)


def test_policy_geometry_has_unique_1858_entries_and_reference_conventions() -> None:
    module = encoding_module()
    moves = module.policy_moves()
    assert len(moves) == len(set(moves)) == 1858
    assert len([move for move in moves if len(move) == 4]) == 1792
    assert moves[1792:1795] == ("a7a8q", "a7a8r", "a7a8b")
    assert module.policy_index("a7a8n") == 1401
    assert module.policy_index("a2a1n", black=True) == 1401
    assert module.policy_index("e1h1") == 103
    assert module.policy_index("e8a8", black=True) == 97
    for index, move in enumerate(moves):
        assert module.policy_index(move) == index
    assert module.move_from_policy(1401, ["a7a8n", "e1d1"]) == "a7a8n"
    assert module.move_from_policy(103, ["e1h1", "e1g1"]) == "e1h1"


@pytest.mark.parametrize("move", ["a1a1", "a1b4", "a7a8k", "a2a3q", "a7h8q", "x1a2"])
def test_policy_rejects_invalid_coordinates_and_promotion_geometry(move: str) -> None:
    module = encoding_module()
    with pytest.raises(module.EncodingError):
        module.policy_index(move)


def test_explicit_repetition_count_uses_the_shared_unsigned_frame_contract() -> None:
    module = encoding_module()
    frame = module.HistoryFrame(START, (1 << 32) - 1)
    assert module.encode_classical([frame])[12].mask == ALL
    with pytest.raises(module.EncodingError):
        module.HistoryFrame(START, 1 << 32)


def test_start_detection_uses_board_identity_instead_of_fen_spelling() -> None:
    module = encoding_module()
    canonical = module.encode_classical([module.HistoryFrame(START)])
    equivalent = START.replace("8/8/8/8", "44/44/44/44")
    assert module.encode_classical([module.HistoryFrame(equivalent)]) == canonical
