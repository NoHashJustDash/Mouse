# SPDX-License-Identifier: GPL-3.0-or-later
"""Binding native parity settings; authored fixtures, no match or GPU."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_match import api, full_plan, parity_document, retained_evidence


def normal_parity_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    path, document = parity_document(tmp_path, monkeypatch)
    document["max_games"] = 20000
    document["engines"][0]["options"].update(Threads="4", MinibatchSize="64", Deterministic="false")
    document["engines"][1]["options"].update(Threads="0", MinibatchSize="0")
    path.write_text(json.dumps(document))
    return path, document


def test_binding_parity_factory_is_normal_batched_four_way_with_unchanged_bounds() -> None:
    selected = api().profile("parity-sprt", max_games=20000, nodes=800)
    assert (
        selected.games,
        selected.nodes,
        selected.sprt,
        selected.concurrency,
        selected.adjudication,
    ) == (20000, 800, (-10, 0), 4, True)
    assert selected.tc is None and selected.book == "8moves_v3.pgn"
    for name in ("selfplay-fast", "uci-stress-fast", "uci-stress-stc", "tm-safety-stc"):
        selected = api().profile(name)
        assert selected.concurrency == 1 and selected.adjudication is False


def test_normal_parity_plan_emits_exact_native_units_and_auto_reference_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    path, _ = normal_parity_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    command = module.build_command(plan, tmp_path / "output")
    assert command[command.index("-concurrency") + 1] == "4"
    assert command[command.index("-draw") + 1 : command.index("-draw") + 4] == [
        "movenumber=34",
        "movecount=8",
        "score=10",
    ]
    assert command[command.index("-resign") + 1 : command.index("-resign") + 4] == [
        "movecount=3",
        "score=1000",
        "twosided=true",
    ]
    assert command[command.index("-maxmoves") + 1] == "192"
    assert command.count("option.Threads=4") == command.count("option.Threads=0") == 1
    assert command.count("option.MinibatchSize=64") == command.count("option.MinibatchSize=0") == 1
    assert "option.Deterministic=false" in command and command.count("option.NNCacheSize=0") == 2
    assert "nodes=800" in command and "args=--config= --show-hidden" in command
    assert not any(part.startswith(("tc=", "st=", "depth=")) for part in command)
    assert "elo0=-10" in command and "elo1=0" in command
    assert command.count("300000") == 3 and "-affinity" not in command


@pytest.mark.parametrize(
    ("engine", "key", "value"),
    [
        (0, "Threads", "1"),
        (0, "MinibatchSize", "1"),
        (0, "Deterministic", "true"),
        (1, "Threads", "1"),
        (1, "MinibatchSize", "1"),
        (1, "Threads", "2"),
        (0, "NNCacheSize", "1"),
        (1, "NNCacheSize", "1"),
        (1, "SmartPruningFactor", "1.33"),
    ],
)
def test_old_serial_or_tuned_parity_settings_cannot_alias_the_revised_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: int,
    key: str,
    value: str,
) -> None:
    path, document = normal_parity_document(tmp_path, monkeypatch)
    document["engines"][engine]["options"][key] = value
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError):
        api().load_plan(path)


def test_normal_timed_profiles_do_not_acquire_any_adjudication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, plan = full_plan(tmp_path, monkeypatch)
    command = module.build_command(plan, tmp_path / "timed")
    assert not any(option in command for option in ("-draw", "-resign", "-maxmoves"))
    assert command[command.index("-concurrency") + 1] == "1"


def adjudicated_record(
    reason: str, result: str, plies: int = 68, opening: tuple[str, ...] = ("Nf3",) * 16
) -> str:
    moves = []
    for ply in range(plies):
        comment = "book" if ply < 16 else "0.00/1 0.001s, tl=0.000s, n=1800"
        if ply == plies - 1:
            comment += ", " + reason
        move = opening[ply] if ply < 16 else "Nf3"
        moves.append(f"{ply // 2 + 1}. {move} {{{comment}}}")
    return (
        '[Event "Authored grammar"]\n[Round "1"]\n[White "A"]\n[Black "B"]\n'
        f'[Result "{result}"]\n[PlyCount "{plies}"]\n[Termination "adjudication"]\n'
        '[TimeControl "-"]\n\n' + " ".join(moves) + f" {result}\n\n"
    )


@pytest.mark.parametrize(
    ("reason", "result", "plies"),
    [
        ("Draw by adjudication", "1/2-1/2", 68),
        ("White wins by adjudication", "1-0", 68),
        ("Black wins by adjudication", "0-1", 69),
    ],
)
def test_only_enabled_exact_native_adjudication_reasons_are_accepted(
    tmp_path: Path,
    reason: str,
    result: str,
    plies: int,
) -> None:
    module = api()
    path = tmp_path / "games.pgn"
    path.write_text(adjudicated_record(reason, result, plies))
    games = module.read_games(path, ("A", "B"), allow_adjudication=True)
    assert games[0].tags["Termination"] == "adjudication"
    assert games[0].nodes_by_ply[-1] == 1800
    with pytest.raises(module.MatchError):
        module.read_games(path, ("A", "B"))


@pytest.mark.parametrize(
    ("reason", "result", "plies"),
    [
        ("Draw by adjudication: SyzygyTB", "1/2-1/2", 68),
        ("White wins by adjudication", "0-1", 68),
        ("Draw by adjudication", "1/2-1/2", 66),
        ("White wins by adjudication", "1-0", 20),
        ("Draw by adjudication", "1/2-1/2", 402),
    ],
)
def test_adjudication_mode_keeps_result_policy_length_and_reason_refusals(
    tmp_path: Path,
    reason: str,
    result: str,
    plies: int,
) -> None:
    path = tmp_path / "games.pgn"
    path.write_text(adjudicated_record(reason, result, plies))
    with pytest.raises(api().MatchError):
        api().read_games(path, ("A", "B"), allow_adjudication=True)


def test_saved_configuration_requires_exact_native_adjudication_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, adjudication=True))
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw)
    path = raw / "fastchess.json"
    saved = json.loads(path.read_text())
    saved.update(
        draw=dict(enabled=True, move_number=34, move_count=8, score=10),
        resign=dict(enabled=True, move_count=3, score=1000, twosided=True),
        maxmoves=dict(enabled=True, move_count=192),
    )
    path.write_text(json.dumps(saved))
    statistics = module.pair_statistics(module.read_games(raw / "games.pgn", ("A", "B")))
    module._saved_configuration(plan, raw, statistics)
    saved["draw"]["move_count"] = 4
    path.write_text(json.dumps(saved))
    with pytest.raises(module.MatchError):
        module._saved_configuration(plan, raw, statistics)


def test_full_concurrent_gate_reconciles_enabled_adjudicated_pgn_reason_tag_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_match_concurrent import concurrent_evidence

    module = api()
    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    plan = replace(plan, selected=replace(plan.selected, name="parity-sprt", adjudication=True))
    path = raw / "games.pgn"
    records = list(module._pgn_records(path))
    game = module._game(records[2], ("A", "B"))
    replacement = adjudicated_record("Draw by adjudication", "1/2-1/2", opening=game.book_moves)
    records[2] = replacement
    path.write_text("".join(records))
    stdout = raw / "stdout.txt"
    stdout.write_text(
        stdout.read_text().replace(
            "Finished game 1 (A vs B): 1/2-1/2 {Draw by 3-fold repetition}",
            "Finished game 1 (A vs B): 1/2-1/2 {Draw by adjudication}",
        )
    )
    saved_path = raw / "fastchess.json"
    saved = json.loads(saved_path.read_text())
    saved.update(
        draw=dict(enabled=True, move_number=34, move_count=8, score=10),
        resign=dict(enabled=True, move_count=3, score=1000, twosided=True),
        maxmoves=dict(enabled=True, move_count=192),
    )
    saved_path.write_text(json.dumps(saved))
    process["command"] = module.build_command(plan, raw)
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["complete"] and result["safety_pass"]
    adjudicated = next(row for row in result["game_records"] if row["id"] == 1)
    assert adjudicated["termination"] == "adjudication"
    assert adjudicated["native_reason"] == "Draw by adjudication"
    assert adjudicated["actual_nodes_by_ply"][-1] == 1800
    assert result["nominal_fixed_nodes"] == 800
    assert result["fixed_node_settings"]["equal_work_claim"] is False


def test_an_arbitrary_rehashed_reference_binary_cannot_borrow_pinned_auto_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    path, document = normal_parity_document(tmp_path, monkeypatch)
    reference = Path(document["engines"][1]["executable"])
    reference.write_bytes(b"not the independently pinned Lc0 build")
    document["sha256"][str(reference)] = hashlib.sha256(reference.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError, match="reference|Lc0"):
        api().load_plan(path)


def test_explicit_single_game_parity_plan_keeps_limits_and_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = normal_parity_document(tmp_path, monkeypatch)
    document["concurrency"] = 1
    path.write_text(json.dumps(document))
    module = api()
    plan = module.load_plan(path)
    command = module.build_command(plan, tmp_path / "output")
    assert plan.selected.concurrency == 1
    assert command[command.index("-concurrency") + 1] == "1"
    assert plan.selected.sprt == (-10, 0)
    assert plan.selected.games == 20000 and plan.selected.nodes == 800
    assert plan.selected.adjudication
    assert command.count("option.MinibatchSize=64") == 1
    assert command.count("option.MinibatchSize=0") == 1


@pytest.mark.parametrize("concurrency", [0, 2, 3, 5, 6, True, "1", 1.0, None])
def test_unapproved_parity_concurrency_cannot_change_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrency: object
) -> None:
    path, document = normal_parity_document(tmp_path, monkeypatch)
    document["concurrency"] = concurrency
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError, match="concurrency"):
        api().load_plan(path)


@pytest.mark.parametrize("value", ["1.33", None])
def test_current_fixed_node_candidate_requires_pruning_explicitly_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    path, document = normal_parity_document(tmp_path, monkeypatch)
    options = document["engines"][0]["options"]
    if value is None:
        options.pop("SmartPruningFactor", None)
    else:
        options["SmartPruningFactor"] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(api().MatchError):
        api().load_plan(path)
