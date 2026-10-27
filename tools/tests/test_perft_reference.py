# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference protocol, EPD validation, and independent Chess960 setup contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from mouse_tools.perft_reference import (
    ReferenceError,
    candidate_cases,
    chess960_backrank,
    parse_epd,
    parse_output,
    validate_fen,
    verify_binary,
)

FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
TRANSCRIPT = f"Fen: {FEN}\ne1d1: 1\ne1f1: 1\ne1d2: 1\ne1e2: 1\ne1f2: 1\nNodes searched: 5\n"


def test_reference_output_requires_matching_divide_and_total() -> None:
    result = parse_output(TRANSCRIPT.splitlines())
    assert result.nodes == 5
    assert result.fen == FEN
    assert set(result.moves) == {"e1d1", "e1f1", "e1d2", "e1e2", "e1f2"}


@pytest.mark.parametrize(
    "text",
    [
        TRANSCRIPT.replace("Nodes searched: 5", "Nodes searched: 6"),
        TRANSCRIPT.replace("Nodes searched: 5", ""),
        TRANSCRIPT + "Nodes searched: 5\n",
        TRANSCRIPT.replace("e1f1: 1", "e1d1: 1"),
        TRANSCRIPT.replace(f"Fen: {FEN}\n", ""),
    ],
)
def test_incomplete_or_inconsistent_reference_output_is_rejected(text: str) -> None:
    with pytest.raises(ReferenceError):
        parse_output(text.splitlines())


def test_epd_preserves_deep_gate_counts() -> None:
    rows = parse_epd(f'{FEN} ; id "example"; variant standard; D1 5; D6 53896;\n')
    assert rows[0].identifier == "example"
    assert rows[0].counts == ((1, 5), (6, 53896))


@pytest.mark.parametrize(
    "operations",
    [
        'id "x"; variant unknown; D1 5;',
        'id "x"; variant standard; D1 -1;',
        'id "x"; variant standard; D1 5; D1 5;',
        'id "x"; variant standard;',
        "variant standard; D1 5;",
        'id "x"; variant standard; D1 5; extra 3;',
    ],
)
def test_malformed_epd_operations_are_rejected(operations: str) -> None:
    with pytest.raises(ReferenceError):
        parse_epd(f"{FEN} ; {operations}\n")


def test_epd_rejects_same_position_with_different_clocks() -> None:
    line = f'{FEN} ; id "one"; variant standard; D1 5;\n'
    duplicate = line.replace('id "one"', 'id "two"').replace(" w - - 0 1", " w - - 50 26")
    with pytest.raises(ReferenceError, match="Duplicate position"):
        parse_epd(line + duplicate)


def test_epd_rejects_duplicate_identifier() -> None:
    first = f'{FEN} ; id "one"; variant standard; D1 5;\n'
    second = first.replace(" w - - ", " b - - ")
    with pytest.raises(ReferenceError, match="Duplicate id"):
        parse_epd(first + second)


@pytest.mark.parametrize(
    "fen",
    [
        "4k3/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/8 w - - 0 1",
        "4k3/8/8/8/8/8/4K3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/4K3/P7 w - - 0 1",
        "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
        FEN.replace(" 0 1", " -1 1"),
        FEN.replace(" w ", " x "),
        FEN.replace(" w - - ", " w - e4 "),
    ],
)
def test_bad_fen_shape_is_rejected_before_reference_execution(fen: str) -> None:
    with pytest.raises(ReferenceError):
        validate_fen(fen)


def test_chess960_numbering_covers_all_legal_backranks() -> None:
    ranks = [chess960_backrank(index) for index in range(960)]
    assert len(set(ranks)) == 960
    assert ranks[518] == "RNBQKBNR"
    for rank in ranks:
        bishops = [file for file, piece in enumerate(rank) if piece == "B"]
        rooks = [file for file, piece in enumerate(rank) if piece == "R"]
        assert bishops[0] % 2 != bishops[1] % 2
        assert rooks[0] < rank.index("K") < rooks[1]


@pytest.mark.parametrize("index", [-1, 960])
def test_chess960_out_of_range_is_rejected(index: int) -> None:
    with pytest.raises(ReferenceError):
        chess960_backrank(index)


def test_candidates_are_distinct_and_include_special_castling_contracts() -> None:
    cases = candidate_cases(20261017)
    signatures = {" ".join(case.fen.split()[:4]) for case in cases}
    assert len(signatures) == len(cases)
    stripped = [case for case in cases if case.identifier.startswith("frc_castle_")]
    assert len(stripped) == 32
    assert all(case.variant == "chess960" and len(case.must_include) == 1 for case in stripped)


def test_binary_hash_mismatch_fails_before_execution(tmp_path: Path) -> None:
    binary = tmp_path / "reference.exe"
    binary.write_bytes(b"not an executable")
    with pytest.raises(ReferenceError, match="SHA-256"):
        verify_binary(binary, "0" * 64)


def test_committed_suite_retains_all_six_spec_gates() -> None:
    path = Path(__file__).resolve().parents[2] / "tests" / "data" / "perft.epd"
    cases = parse_epd(path.read_text(encoding="utf-8"))
    assert len(cases) >= 100
    by_fen = {case.fen: dict(case.counts) for case in cases}
    expected = (
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6, 11030083),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551),
    )
    for fen, depth, count in expected:
        assert by_fen[fen][depth] == count
    assert len([case for case in cases if case.variant == "chess960"]) >= 32


@pytest.mark.parametrize("field", ["depth", "count"])
@pytest.mark.parametrize("token", ["9" * 5000, str(1 << 64)], ids=["oversized", "overflow"])
def test_epd_numeric_tokens_raise_reference_errors(field: str, token: str) -> None:
    depth = token if field == "depth" else "1"
    count = token if field == "count" else "5"
    with pytest.raises(ReferenceError, match="EPD"):
        parse_epd(f'{FEN} ; id "large"; variant standard; D{depth} {count};')


@pytest.mark.parametrize("field", ["halfmove", "fullmove"])
@pytest.mark.parametrize(
    "token",
    ["9" * 5000, str(1 << 64), "0" * 5000],
    ids=["oversized", "overflow", "oversized-zero"],
)
def test_fen_counters_are_bounded_before_integer_conversion(field: str, token: str) -> None:
    fields = FEN.split()
    fields[4 if field == "halfmove" else 5] = token
    with pytest.raises(ReferenceError, match="FEN"):
        validate_fen(" ".join(fields))


def test_numeric_boundaries_match_the_engine_contract() -> None:
    maximum = str((1 << 64) - 1)
    validate_fen(FEN.replace(" 0 1", f" {maximum} {maximum}"))
    parsed = parse_epd(f'{FEN} ; id "bounds"; variant standard; D128 {maximum};')
    assert parsed[0].counts == ((128, (1 << 64) - 1),)
    with pytest.raises(ReferenceError, match="depth"):
        parse_epd(f'{FEN} ; id "deep"; variant standard; D129 5;')


@pytest.mark.parametrize("field", ["total", "divide"])
def test_oracle_numeric_output_is_bounded(field: str) -> None:
    text = TRANSCRIPT
    if field == "total":
        text = text.replace("Nodes searched: 5", "Nodes searched: " + "9" * 5000)
    else:
        text = text.replace("e1d1: 1", "e1d1: " + "9" * 5000)
    with pytest.raises(ReferenceError, match="Oracle"):
        parse_output(text.splitlines())


def test_generate_api_rejects_destination_collision_before_querying_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mouse_tools.perft_reference as reference_module

    def forbidden_queries(reference: object, seed: int) -> object:
        raise AssertionError("Output collision must be rejected before any oracle query")

    monkeypatch.setattr(reference_module, "playout_cases", forbidden_queries)
    destination = tmp_path / "same-output"
    destination.write_bytes(b"preserve existing data")
    with pytest.raises(ReferenceError, match="destinations must differ"):
        reference_module.generate(
            reference_module.UciReference.__new__(reference_module.UciReference),
            destination,
            destination,
            20261017,
        )
    assert destination.read_bytes() == b"preserve existing data"
