# SPDX-License-Identifier: GPL-3.0-or-later
"""Concurrent fixed-node audit regressions; no native engines or GPU execution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_match import api, full_plan, retained_evidence


def start(index: int, cap: int = 20) -> str:
    white, black = ("A", "B") if index % 2 else ("B", "A")
    return f"Started game {index} of {cap} ({white} vs {black})"


def finish(index: int, reason: str = "Draw by 3-fold repetition") -> str:
    white, black = ("A", "B") if index % 2 else ("B", "A")
    return f"Finished game {index} ({white} vs {black}): 1/2-1/2 {{{reason}}}"


def test_four_concurrent_games_may_start_and_finish_out_of_order() -> None:
    guard = api().Progress(20, ("A", "B"), concurrency=4)
    for index in (2, 1, 4, 3):
        guard(start(index))
    for index in (3, 1, 4, 2):
        guard(finish(index))
    for index in (5, 6):
        guard(start(index))
    guard(finish(6))
    guard(finish(5))
    assert set(guard.started) == set(guard.finished) == set(range(1, 7))
    assert not guard.active


@pytest.mark.parametrize(
    "mutation",
    [
        "fifth-active",
        "duplicate-start",
        "duplicate-finish",
        "finish-before-start",
        "wrong-color",
        "wrong-cap",
        "out-of-range",
        "late-start",
    ],
)
def test_concurrent_progress_retains_bounded_identity_and_protocol_refusals(mutation: str) -> None:
    module = api()
    guard = module.Progress(20, ("A", "B"), concurrency=4)
    for index in (1, 2, 3, 4):
        guard(start(index))
    if mutation == "duplicate-finish":
        guard(finish(1))
    if mutation == "late-start":
        guard("Finished match")
    bad = {
        "fifth-active": start(5),
        "duplicate-start": start(1),
        "duplicate-finish": finish(1),
        "finish-before-start": finish(9),
        "wrong-color": finish(2).replace("(B vs A)", "(A vs B)"),
        "wrong-cap": start(5, 18),
        "out-of-range": start(21),
        "late-start": start(5),
    }[mutation]
    with pytest.raises(module.MatchError):
        guard(bad)


def test_concurrent_protocol_validates_each_worker_catalog_independently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    single = path.read_text()
    path.write_text("".join(single.replace("<1>", f"<{worker}>") for worker in (1, 2, 3, 4)))
    result = module._protocol(plan, path)
    assert set(result) == {"A", "B"}
    assert all(len(value["startup_catalogs"]) == 4 for value in result.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-handshake",
        "duplicate-option",
        "unsent-option",
        "worker-overflow",
        "stderr",
        "malformed-marker",
    ],
)
def test_one_bad_concurrent_worker_cannot_borrow_another_workers_valid_handshake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    single = path.read_text()
    fourth = single.replace("<1>", "<4>")
    if mutation == "missing-handshake":
        fourth = fourth.replace("A ---> uciok\n", "A ---> info depth 1\n")
    elif mutation == "duplicate-option":
        option = next(line for line in fourth.splitlines() if "A ---> option name " in line)
        fourth += option + "\n"
    elif mutation == "unsent-option":
        fourth = (
            "\n".join(
                line for line in fourth.splitlines() if "A <--- setoption name Threads " not in line
            )
            + "\n"
        )
    elif mutation == "worker-overflow":
        fourth += single.replace("<1>", "<5>")
    elif mutation == "stderr":
        fourth += "[Engine] [time] <4> <stderr> A ---> fatal diagnostic\n"
    elif mutation == "malformed-marker":
        fourth = fourth.replace("<4>", "<>")
    combined = "".join(single.replace("<1>", f"<{worker}>") for worker in (1, 2, 3)) + fourth
    if mutation == "unsent-option":
        combined = (
            "\n".join(
                line
                for line in combined.splitlines()
                if "A <--- setoption name Threads " not in line
            )
            + "\n"
        )
    path.write_text(combined)
    with pytest.raises(module.MatchError):
        module._protocol(plan, path)


def test_long_reused_engine_traffic_does_not_inflate_startup_catalog_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, games=20000))
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    with path.open("a") as stream:
        stream.write("[Engine] [time] <1> A <--- ucinewgame\n" * 5000)
    assert module._protocol(plan, path)["A"]["startup_catalogs"]


def ledger_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cap: int = 20,
) -> tuple[Any, Any, list[Any]]:
    from test_match import full_game

    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(
        original,
        selected=replace(
            original.selected, games=cap, concurrency=4, sprt=(-10, 0), nodes=800, tc=None
        ),
    )
    events = module.Progress(cap, ("A", "B"), concurrency=4)
    for index in (1, 2, 3, 4):
        events(start(index, cap))
    for completed, new in ((1, 5), (2, 6), (3, 7), (6, 8)):
        events(finish(completed))
        events(start(new, cap))
    events(finish(7))
    events(finish(4))
    events("Finished match")
    games = [
        module._game(
            full_game((index + 1) // 2, *("A", "B")[:: 1 if index % 2 else -1]), ("A", "B")
        )
        for index in (7, 4, 1, 6, 3, 2)
    ]
    return plan, events, games


def test_natural_sprt_ledger_retains_every_orphan_and_inflight_id_without_counting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, events, games = ledger_fixture(tmp_path, monkeypatch)
    paired, result = module._game_ledger(plan, games, events, natural_sprt_stop=True)
    assert len(paired) == result["statistical_complete_pair_games"] == 4
    assert result["raw_completed_games"] == 6
    assert result["raw_pgn_game_ids"] == [7, 4, 1, 6, 3, 2]
    assert result["complete_pair_round_ids"] == [1, 2]
    assert result["unpaired_completed_game_ids"] == [6, 7]
    assert result["cancelled_inflight_game_ids"] == [5, 8]
    assert result["started_game_ids"] == list(range(1, 9))


@pytest.mark.parametrize(
    "mutation",
    [
        "no-decision",
        "lost-pgn",
        "duplicate-pgn",
        "missing-start",
        "wrong-result",
        "wrong-book",
        "wrong-color",
        "nonterminal-closure",
        "excess-tails",
    ],
)
def test_natural_sprt_ledger_never_conceals_missing_or_forged_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = api()
    plan, events, games = ledger_fixture(tmp_path, monkeypatch)
    natural = mutation != "no-decision"
    if mutation == "lost-pgn":
        games.pop()
    elif mutation == "duplicate-pgn":
        games.append(games[0])
    elif mutation == "missing-start":
        del events.started[1]
    elif mutation == "wrong-result":
        games[1] = replace(games[1], result="1-0")
    elif mutation == "wrong-book":
        games[1] = replace(games[1], book_moves=("d4",) + games[1].book_moves[1:])
    elif mutation == "wrong-color":
        games[1] = replace(games[1], tags={**games[1].tags, "White": "A", "Black": "B"})
    elif mutation == "nonterminal-closure":
        events.closed = False
    elif mutation == "excess-tails":
        for index in (9, 10, 11, 12, 13):
            white, black = ("A", "B") if index % 2 else ("B", "A")
            events.started[index] = (20, white, black)
            events.active.add(index)
    with pytest.raises(module.MatchError):
        module._game_ledger(plan, games, events, natural_sprt_stop=natural)


def concurrent_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, dict[str, Any]]:
    import json

    module = api()
    plan, events, games = ledger_fixture(tmp_path, monkeypatch)
    raw = tmp_path / "concurrent-evidence"
    raw.mkdir()
    retained_evidence(plan, raw, count=8)
    records = list(module._pgn_records(raw / "games.pgn"))
    (raw / "games.pgn").write_text("".join(records[index - 1] for index in (7, 4, 1, 6, 3, 2)))
    lines = [start(index) for index in (1, 2, 3, 4)]
    for completed, new in ((1, 5), (2, 6), (3, 7), (6, 8)):
        lines += [finish(completed), start(new)]
    lines += [finish(7), finish(4)]
    lines += [
        "Games: 4, Wins: 0, Losses: 0, Draws: 4, Points: 2.0 (50.00 %)",
        "Ptnml(0-2): [0, 0, 2, 0, 0], WL/DD Ratio: 0.00",
        "LLR: 3.00 (100.0%) (-2.94, 2.94) [-10.00, 0.00]",
        "SPRT ([-10.00, 0.00]) completed - H1 was accepted",
        "Finished match",
    ]
    (raw / "stdout.txt").write_text("\n".join(lines) + "\n")
    saved_path = raw / "fastchess.json"
    saved = json.loads(saved_path.read_text())
    saved["concurrency"] = 4
    saved["sprt"] = dict(enabled=True, model="normalized", alpha=0.05, beta=0.05, elo0=-10, elo1=0)
    for native in saved["engines"]:
        native["limit"]["nodes"] = 800
        native["limit"]["tc"].update(time=0, increment=0)
    saved["stats"]["A vs B"].update(draws=4, penta_DD=2)
    saved_path.write_text(json.dumps(saved))
    protocol = raw / "protocol.log"
    single = protocol.read_text()
    protocol.write_text("".join(single.replace("<1>", f"<{worker}>") for worker in (1, 2, 3, 4)))
    with protocol.open("a") as stream:
        for index in range(1, 9):
            white, black = ("A", "B") if index % 2 else ("B", "A")
            worker = (index - 1) % 4 + 1
            stream.write(
                f"[TRACE] [time] <{worker}> fastchess --- Game {index} "
                f"between {white} and {black} starting\n"
            )
            stream.write(
                f"[TRACE] [time] <{worker}> fastchess --- Game {index} "
                f"between {white} and {black} finished\n"
            )
            stream.write(
                f"[TRACE] [time] <{worker}> fastchess --- Thread {worker} finished match {index}\n"
            )
    process = dict(
        command=module.build_command(plan, raw),
        exit_code=0,
        pumps_joined=True,
        terminated_by_runner=False,
        error=None,
    )
    return plan, raw, process


def test_full_concurrent_sprt_audit_reports_raw_and_statistical_counts_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["complete"] and result["safety_pass"]
    assert result["sprt_decision"] == "H1"
    assert result["statistics"]["games"] == 4
    assert result["statistics"]["pairs"] == 2
    assert result["game_inventory"]["raw_completed_games"] == 6
    assert result["game_inventory"]["unpaired_completed_game_ids"] == [6, 7]
    assert result["game_inventory"]["cancelled_inflight_game_ids"] == [5, 8]
    assert result["native_game_trace"]["completed_trace_game_ids"] == list(range(1, 9))


@pytest.mark.parametrize(
    "mutation",
    [
        "no-process",
        "nonzero-exit",
        "not-joined",
        "runner-killed",
        "process-error",
        "wrong-process-command",
        "no-decision",
        "false-boundary",
        "forged-penta",
        "missing-pgn",
        "missing-trace-start",
        "missing-trace-finish",
        "wrong-trace-worker",
        "native-error",
        "missing-roundset",
    ],
)
def test_full_concurrent_sprt_tails_require_authentic_complete_native_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = api()
    plan, raw, captured = concurrent_evidence(tmp_path, monkeypatch)
    process: dict[str, Any] | None = captured
    if mutation == "no-process":
        process = None
    elif mutation == "nonzero-exit":
        captured["exit_code"] = 3
    elif mutation == "not-joined":
        captured["pumps_joined"] = False
    elif mutation == "runner-killed":
        captured["terminated_by_runner"] = True
    elif mutation == "process-error":
        captured["error"] = "watchdog"
    elif mutation == "wrong-process-command":
        captured["command"] = ["invented runner"]
    elif mutation in ("no-decision", "false-boundary", "forged-penta", "missing-roundset"):
        path = raw / "stdout.txt"
        text = path.read_text()
        if mutation == "no-decision":
            text = (
                "\n".join(line for line in text.splitlines() if not line.startswith("SPRT ("))
                + "\n"
            )
        elif mutation == "false-boundary":
            text = text.replace("LLR: 3.00", "LLR: 0.00")
        elif mutation == "forged-penta":
            text = text.replace("[0, 0, 2, 0, 0]", "[0, 0, 3, 0, 0]")
        else:
            text = (
                "\n".join(
                    line for line in text.splitlines() if not line.startswith("Started game 2 ")
                )
                + "\n"
            )
        path.write_text(text)
    elif mutation == "missing-pgn":
        path = raw / "games.pgn"
        records = list(module._pgn_records(path))
        path.write_text("".join(records[1:]))
    else:
        path = raw / "protocol.log"
        text = path.read_text()
        if mutation == "missing-trace-start":
            text = (
                "\n".join(
                    line
                    for line in text.splitlines()
                    if "Game 5 between" not in line or "starting" not in line
                )
                + "\n"
            )
        elif mutation == "missing-trace-finish":
            text = (
                "\n".join(
                    line
                    for line in text.splitlines()
                    if "Game 5 between" not in line or "finished" not in line
                )
                + "\n"
            )
        elif mutation == "wrong-trace-worker":
            text = text.replace("<1> fastchess --- Game 5", "<9> fastchess --- Game 5")
        else:
            text += "[ERR] [time] <1> fastchess --- real native failure\n"
        path.write_text(text)
    with pytest.raises(module.MatchError):
        module.audit_results(
            plan, raw, captured_command=module.build_command(plan, raw), process=process
        )


def late_pair_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_llr: str = "0.00",
    repeat: str | None = None,
) -> tuple[Any, Path, dict[str, Any]]:
    import json

    from test_match import full_game

    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    with (raw / "games.pgn").open("a") as stream:
        stream.write(full_game(3, "A", "B"))
    extra = [
        finish(5),
        "Games: 6, Wins: 0, Losses: 0, Draws: 6, Points: 3.0 (50.00 %)",
        "Ptnml(0-2): [0, 0, 3, 0, 0], WL/DD Ratio: 0.00",
        f"LLR: {final_llr} (100.0%) (-2.94, 2.94) [-10.00, 0.00]",
    ]
    if repeat is not None:
        extra.append(f"SPRT ([-10.00, 0.00]) completed - {repeat} was accepted")
    path = raw / "stdout.txt"
    path.write_text(
        path.read_text().replace("Finished match\n", "\n".join(extra) + "\nFinished match\n")
    )
    saved = raw / "fastchess.json"
    native = json.loads(saved.read_text())
    native["stats"]["A vs B"].update(draws=6, penta_DD=3)
    saved.write_text(json.dumps(native))
    return plan, raw, process


def test_first_authentic_boundary_is_frozen_even_if_late_pair_returns_llr_inside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = late_pair_evidence(tmp_path, monkeypatch)
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["sprt_decision"] == "H1"
    assert result["statistics"]["games"] == 4
    assert result["native_final_statistics"]["games"] == 6
    assert result["sprt_stop_point"]["completed_game_ids_in_order"] == [1, 2, 3, 6, 7, 4]
    assert result["sprt_stop_point"]["complete_pair_round_ids_in_order"] == [1, 2]
    assert result["late_completion_game_ids"] == [5]
    assert result["native_normalized_llr_printed"] == 0.0
    assert result["sprt_stop_point"]["normalized_llr_printed"] == 3.0


def test_repeated_same_native_boundary_needs_new_completed_game_and_exact_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = late_pair_evidence(tmp_path, monkeypatch, final_llr="3.10", repeat="H1")
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert len(result["native_sprt_decisions"]) == 2
    assert result["native_sprt_decisions"][0]["statistics"]["games"] == 4
    assert result["native_sprt_decisions"][1]["statistics"]["games"] == 6
    assert result["statistics"]["games"] == 4


@pytest.mark.parametrize(
    "mutation",
    [
        "no-progress-duplicate",
        "flip",
        "forged-first-penta",
        "missing-first-summary",
        "receded-repeat",
    ],
)
def test_late_native_messages_cannot_forge_or_retroactively_replace_the_first_stop_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = api()
    plan, raw, process = late_pair_evidence(tmp_path, monkeypatch)
    path = raw / "stdout.txt"
    text = path.read_text()
    if mutation == "no-progress-duplicate":
        text = text.replace(
            "completed - H1 was accepted\n",
            "completed - H1 was accepted\nSPRT ([-10.00, 0.00]) completed - H1 was accepted\n",
            1,
        )
    elif mutation == "flip":
        text = text.replace("LLR: 0.00", "LLR: -3.10")
        text = text.replace(
            "Finished match", "SPRT ([-10.00, 0.00]) completed - H0 was accepted\nFinished match"
        )
    elif mutation == "forged-first-penta":
        text = text.replace("[0, 0, 2, 0, 0]", "[0, 0, 3, 0, 0]", 1)
    elif mutation == "missing-first-summary":
        text = text.replace(
            "Games: 4, Wins: 0, Losses: 0, Draws: 4, Points: 2.0 (50.00 %)\n", "", 1
        )
    else:
        text = text.replace(
            "Finished match", "SPRT ([-10.00, 0.00]) completed - H1 was accepted\nFinished match"
        )
    path.write_text(text)
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=process["command"], process=process)


def test_native_refresh_resends_identical_settings_without_growing_startup_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, games=20000))
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    with path.open("a") as stream:
        stream.write("[Engine] [time] <1> A <--- setoption name Threads value 4\n" * 5000)
    result = module._protocol(plan, path)["A"]
    assert result["sent_option_counts"]["Threads"] == 5001
    assert len(result["sent_options"]) <= len(plan.engines[0].options) + 1


@pytest.mark.parametrize(
    "line",
    [
        "setoption name Threads value 128",
        "setoption name Unplanned value true",
        "setoption name Threads",
    ],
)
def test_repeated_refresh_can_never_override_or_add_unplanned_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line: str,
) -> None:
    module = api()
    _, plan = full_plan(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    with path.open("a") as stream:
        stream.write("[Engine] [time] <1> A <--- " + line + "\n")
    with pytest.raises(module.MatchError):
        module._protocol(plan, path)


@pytest.mark.parametrize("migrated", ["A", "B", "both"])
def test_global_pool_startup_context_does_not_own_game_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    migrated: str,
) -> None:
    module = api()
    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    path = raw / "protocol.log"
    path.write_text(
        "\n".join(
            line.replace("<4>", "<1>")
            if line.startswith("[Engine]") and (migrated == "both" or f" {migrated} " in line)
            else line
            for line in path.read_text().splitlines()
        )
        + "\n"
    )
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["complete"]
    assert result["native_game_trace"]["trace_thread_markers"] == ["1", "2", "3", "4"]


def test_command_uses_declared_concurrency_without_changing_sequential_timed_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, sequential = full_plan(tmp_path, monkeypatch)
    concurrent = replace(sequential, selected=replace(sequential.selected, concurrency=4))
    for plan, expected in ((sequential, "1"), (concurrent, "4")):
        command = module.build_command(plan, tmp_path / "raw")
        assert command.count("-concurrency") == 1
        assert command[command.index("-concurrency") + 1] == expected


def complete_concurrent_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    import json

    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(
        original,
        selected=replace(
            original.selected,
            games=20,
            concurrency=4,
            sprt=(-10, 0),
            nodes=800,
            tc=None,
        ),
    )
    raw = tmp_path / "full-cap"
    raw.mkdir()
    retained_evidence(plan, raw, count=20)
    records = list(module._pgn_records(raw / "games.pgn"))
    # Physical PGN completion order need not be sorted by native game ID.
    order = [offset + item for offset in range(0, 20, 4) for item in (2, 4, 3, 1)]
    (raw / "games.pgn").write_text("".join(records[index - 1] for index in order))
    stdout = []
    for offset in range(0, 20, 4):
        stdout += [start(offset + item) for item in (4, 2, 1, 3)]
        stdout += [finish(offset + item) for item in (2, 4, 3, 1)]
    stdout += [
        "Games: 20, Wins: 0, Losses: 0, Draws: 20, Points: 10.0 (50.00 %)",
        "Ptnml(0-2): [0, 0, 10, 0, 0], WL/DD Ratio: 0.00",
        "LLR: 0.00 (0.0%) (-2.94, 2.94) [-10.00, 0.00]",
        "Finished match",
    ]
    (raw / "stdout.txt").write_text("\n".join(stdout) + "\n")
    saved_path = raw / "fastchess.json"
    saved = json.loads(saved_path.read_text())
    saved["concurrency"] = 4
    saved["sprt"] = dict(enabled=True, model="normalized", alpha=0.05, beta=0.05, elo0=-10, elo1=0)
    for native in saved["engines"]:
        native["limit"]["nodes"] = 800
        native["limit"]["tc"].update(time=0, increment=0)
    saved_path.write_text(json.dumps(saved))
    protocol = raw / "protocol.log"
    catalog = protocol.read_text()
    protocol.write_text("".join(catalog.replace("<1>", f"<{worker}>") for worker in range(1, 5)))
    with protocol.open("a") as stream:
        for offset in range(0, 20, 4):
            for phase in ("starting", "finished"):
                for item in (2, 4, 3, 1):
                    index = offset + item
                    white, black = ("A", "B") if index % 2 else ("B", "A")
                    stream.write(
                        f"[TRACE] [time] <{item}> fastchess --- Game {index} "
                        f"between {white} and {black} {phase}\n"
                    )
                    if phase == "finished":
                        stream.write(
                            f"[TRACE] [time] <{item}> fastchess --- "
                            f"Thread {item} finished match {index}\n"
                        )
    return plan, raw


def test_full_cap_reconciles_all_out_of_order_games_and_stays_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw = complete_concurrent_cap(tmp_path, monkeypatch)
    result = module.audit_results(plan, raw, captured_command=module.build_command(plan, raw))
    assert result["sprt_decision"] == "inconclusive"
    assert result["statistics"]["games"] == 20
    assert result["nominal_fixed_nodes"] == 800
    assert all(
        record["actual_nodes_by_ply"] == [None] * 16 + [8, 8] for record in result["game_records"]
    )
    assert result["game_inventory"]["started_game_ids"] == list(range(1, 21))
    assert result["game_inventory"]["raw_pgn_game_ids"][:4] == [2, 4, 3, 1]
    assert not result["game_inventory"]["unpaired_completed_game_ids"]
    assert not result["game_inventory"]["cancelled_inflight_game_ids"]


@pytest.mark.parametrize(
    "mutation", ["lost-final-pgn", "missing-round", "orphan", "false-h1-at-cap"]
)
def test_concurrent_cap_cannot_hide_missing_pairs_or_invent_h1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = api()
    plan, raw = complete_concurrent_cap(tmp_path, monkeypatch)
    if mutation in ("lost-final-pgn", "orphan", "missing-round"):
        records = list(module._pgn_records(raw / "games.pgn"))
        if mutation == "lost-final-pgn":
            records.pop()
        elif mutation == "orphan":
            records = [
                record
                for record in records
                if '[Round "10"]' not in record or '[White "B"]' not in record
            ]
        else:
            records = [record for record in records if '[Round "5"]' not in record]
        (raw / "games.pgn").write_text("".join(records))
    else:
        stdout = raw / "stdout.txt"
        stdout.write_text(
            stdout.read_text().replace(
                "Finished match",
                "SPRT ([-10.00, 0.00]) completed - H1 was accepted\nFinished match",
            )
        )
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=module.build_command(plan, raw))


def post_decision_extra_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: int,
) -> tuple[Any, Path, dict[str, Any]]:
    import json

    from test_match import full_game

    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    added_ids = [5, 8] + list(range(9, 9 + extra))
    stdout = [finish(5), finish(8)]
    with (raw / "games.pgn").open("a") as stream:
        for index in added_ids:
            colors = ("A", "B") if index % 2 else ("B", "A")
            stream.write(full_game((index + 1) // 2, *colors))
    for index in range(9, 9 + extra):
        stdout += [start(index), finish(index)]
    total = 8 + extra
    stdout += [
        f"Games: {total}, Wins: 0, Losses: 0, Draws: {total}, Points: {total / 2:.1f} (50.00 %)",
        f"Ptnml(0-2): [0, 0, {total // 2}, 0, 0], WL/DD Ratio: 0.00",
        "LLR: 3.10 (100.0%) (-2.94, 2.94) [-10.00, 0.00]",
    ]
    path = raw / "stdout.txt"
    path.write_text(
        path.read_text().replace("Finished match\n", "\n".join(stdout) + "\nFinished match\n")
    )
    with (raw / "protocol.log").open("a") as stream:
        for index in range(9, 9 + extra):
            colors = ("A", "B") if index % 2 else ("B", "A")
            for phase in ("starting", "finished"):
                worker = (2, 3, 1, 4)[index - 9]
                stream.write(
                    f"[TRACE] [time] <{worker}> fastchess --- Game {index} between "
                    f"{colors[0]} and {colors[1]} {phase}\n"
                )
            stream.write(
                f"[TRACE] [time] <{worker}> fastchess --- Thread {worker} finished match {index}\n"
            )
    path = raw / "fastchess.json"
    saved = json.loads(path.read_text())
    saved["stats"]["A vs B"].update(draws=total, penta_DD=total // 2)
    path.write_text(json.dumps(saved))
    return plan, raw, process


def test_natural_decision_cannot_adopt_unbounded_late_complete_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = post_decision_extra_games(tmp_path, monkeypatch, extra=4)
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=process["command"], process=process)


def test_pending_tail_at_first_decision_cannot_restart_the_same_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = post_decision_extra_games(tmp_path, monkeypatch, extra=2)
    path = raw / "protocol.log"
    text = path.read_text()
    for index, worker in ((9, 2), (10, 3)):
        text = text.replace(
            f"<{worker}> fastchess --- Game {index} ", f"<1> fastchess --- Game {index} "
        )
        text = text.replace(
            f"<{worker}> fastchess --- Thread {worker} finished match {index}",
            f"<1> fastchess --- Thread 1 finished match {index}",
        )
    path.write_text(text)
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=process["command"], process=process)


def test_bounded_unique_worker_tails_keep_all_late_pairs_without_replacing_first_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = post_decision_extra_games(tmp_path, monkeypatch, extra=2)
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["statistics"]["games"] == 4
    assert result["native_final_statistics"]["games"] == 10
    assert result["late_completion_game_ids"] == [5, 8, 9, 10]


def test_real_child_capture_pump_retains_out_of_order_progress_and_joins(tmp_path: Path) -> None:
    import threading

    from test_match import child

    module = api()
    guard = module.Progress(4, ("A", "B"), concurrency=4)
    lines = [start(index, 4) for index in (2, 4, 1, 3)]
    lines += [finish(index) for index in (4, 2, 3, 1)] + ["Finished match"]
    command = child(
        f"import sys;sys.stdout.write({chr(10).join(lines) + chr(10)!r});sys.stdout.flush()"
    )
    before = {thread.ident for thread in threading.enumerate()}
    result = module.capture(
        command, tmp_path, idle_seconds=2, total_seconds=3, max_bytes=65536, on_stdout=guard
    )
    assert result["exit_code"] == 0 and result["error"] is None
    assert result["pumps_joined"] and not result["terminated_by_runner"]
    assert (tmp_path / "stdout.txt").read_text().splitlines() == lines
    assert list(guard.finished) == [4, 2, 3, 1]
    assert guard.closed and not guard.active
    assert {thread.ident for thread in threading.enumerate()} == before


def test_pgn_actual_per_move_node_work_is_preserved_separately_from_nominal_limit() -> None:
    from test_match import full_game

    module = api()
    game = module._game(full_game(1, "A", "B").replace("n=8", "n=849"), ("A", "B"))
    assert game.nodes_by_ply == (None,) * 16 + (849, 849)


@pytest.mark.parametrize(
    "value",
    ["8, n=9", "8.5", "18446744073709551616", "9" * 5000],
    ids=["duplicate", "fraction", "uint64-overflow", "huge-integer"],
)
def test_ambiguous_or_unrepresentable_pgn_node_work_is_not_silently_chosen(value: str) -> None:
    from test_match import full_game

    module = api()
    record = full_game(1, "A", "B").replace("n=8", f"n={value}", 1)
    with pytest.raises(module.MatchError):
        module._game(record, ("A", "B"))


def test_statistical_stop_prefix_uses_actual_pair_completion_order_not_sorted_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, _ = concurrent_evidence(tmp_path, monkeypatch)
    events = module.Progress(20, ("A", "B"), concurrency=4)
    for index in (2, 1, 4, 3):
        events(start(index))
    for index in (4, 3, 2, 1):
        events(finish(index))
    snapshot = module._native_boundary_snapshot(
        plan,
        module.read_games(raw / "games.pgn", ("A", "B")),
        events,
        (4, 0, 0, 4),
        [0, 0, 2, 0, 0],
        3.0,
        "H1",
        123,
    )
    assert snapshot["completed_game_ids_in_order"] == [4, 3, 2, 1]
    assert snapshot["complete_pair_round_ids_in_order"] == [2, 1]
    assert snapshot["started_game_ids_at_decision"] == [2, 1, 4, 3]


def test_authentic_h0_cannot_be_replaced_by_late_interior_llr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = late_pair_evidence(tmp_path, monkeypatch)
    path = raw / "stdout.txt"
    path.write_text(
        path.read_text()
        .replace("LLR: 3.00", "LLR: -3.00", 1)
        .replace("H1 was accepted", "H0 was accepted", 1)
    )
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["sprt_decision"] == "H0"
    assert result["sprt_stop_point"]["decision"] == "H0"
    assert result["statistics"]["games"] == 4
    assert result["native_final_statistics"]["games"] == 6
    assert result["native_normalized_llr_printed"] == 0.0


def test_global_pool_engine_refresh_can_migrate_without_worker_local_handshake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "global-migration"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    with path.open("a") as stream:
        for engine in plan.engines:
            for key, value in engine.options:
                stream.write(
                    f"[Engine] [time] <2> {engine.name} <--- setoption name {key} value {value}\n"
                )
            stream.write(f"[Engine] [time] <2> {engine.name} <--- ucinewgame\n")
    result = module._protocol(plan, path)
    for engine in plan.engines:
        assert len(result[engine.name]["startup_catalogs"]) == 1
        assert result[engine.name]["sent_option_counts"]["Threads"] == 2
        assert result[engine.name]["ucinewgame_count"] == 1


def test_global_pool_can_initialize_two_complete_catalogs_on_same_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "same-worker-initialization"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    original_text = path.read_text()
    path.write_text(original_text + original_text)
    result = module._protocol(plan, path)
    for engine in plan.engines:
        catalogs = result[engine.name]["startup_catalogs"]
        assert len(catalogs) == 2
        assert [catalog["initialization_thread_marker"] for catalog in catalogs] == ["1", "1"]


def test_complete_unused_global_catalog_does_not_need_a_per_process_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "unused-catalog"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    text = path.read_text()
    unused = "\n".join(line for line in text.splitlines() if "<--- setoption" not in line)
    path.write_text(text + unused.replace("<1>", "<2>") + "\n")
    result = module._protocol(plan, path)
    assert all(len(value["startup_catalogs"]) == 2 for value in result.values())


@pytest.mark.parametrize("damage", ["missing-option", "missing-uciok", "duplicate-option"])
def test_invalid_second_catalog_on_same_worker_cannot_borrow_a_good_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "bad-global-catalog"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    text = path.read_text()
    second = text
    if damage == "missing-option":
        second = (
            "\n".join(
                line for line in second.splitlines() if "A ---> option name Threads " not in line
            )
            + "\n"
        )
    elif damage == "missing-uciok":
        second = second.replace("A ---> uciok\n", "A ---> readyok\n")
    else:
        line = next(line for line in text.splitlines() if "A ---> option name Threads " in line)
        second += line + "\n"
    path.write_text(text + second)
    with pytest.raises(module.MatchError):
        module._protocol(plan, path)


def skipped_assignment_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Path, dict[str, Any]]:
    import json

    from test_match import full_game

    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    (raw / "games.pgn").write_text(
        "".join(
            full_game((index + 1) // 2, *(("A", "B") if index % 2 else ("B", "A")))
            for index in (5, 1, 2)
        )
    )
    lines = [start(index) for index in (1, 2, 3)]
    lines += [
        finish(1),
        start(5),
        finish(5),
        finish(2),
        "Games: 2, Wins: 0, Losses: 0, Draws: 2, Points: 1.0 (50.00 %)",
        "Ptnml(0-2): [0, 0, 1, 0, 0], WL/DD Ratio: 0.00",
        "LLR: 3.00 (100.0%) (-2.94, 2.94) [-10.00, 0.00]",
        "SPRT ([-10.00, 0.00]) completed - H1 was accepted",
        "Finished match",
    ]
    (raw / "stdout.txt").write_text("\n".join(lines) + "\n")
    saved_path = raw / "fastchess.json"
    saved = json.loads(saved_path.read_text())
    saved["stats"]["A vs B"].update(draws=2, penta_DD=1)
    saved_path.write_text(json.dumps(saved))
    protocol = raw / "protocol.log"
    catalogs = [line for line in protocol.read_text().splitlines() if line.startswith("[Engine]")]
    trace = []

    def game(index: int, worker: int, phase: str) -> str:
        white, black = ("A", "B") if index % 2 else ("B", "A")
        return (
            f"[TRACE] [time] <{worker}> fastchess --- Game {index} "
            f"between {white} and {black} {phase}"
        )

    def receipt(index: int, worker: int) -> str:
        return f"[TRACE] [time] <{worker}> fastchess --- Thread {worker} finished match {index}"

    trace += [game(index, index, "starting") for index in (1, 2, 3)]
    trace += [
        game(1, 1, "finished"),
        receipt(1, 1),
        game(5, 1, "starting"),
        game(5, 1, "finished"),
        receipt(5, 1),
        game(2, 2, "finished"),
        "[TRACE] [time] <> fastchess --- SPRT test finished, stopping tournament.",
        receipt(2, 2),
        game(3, 3, "finished"),
        receipt(3, 3),
        receipt(4, 4),
    ]
    protocol.write_text("\n".join(catalogs + trace) + "\n")
    return plan, raw, process


def test_natural_stop_can_prove_an_internal_skipped_assignment_without_inventing_a_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = skipped_assignment_evidence(tmp_path, monkeypatch)
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["statistics"]["games"] == 2
    assert result["game_inventory"]["started_game_ids"] == [1, 2, 3, 5]
    assert result["game_inventory"]["skipped_before_start_game_ids"] == [4]
    assert result["native_game_trace"]["observed_assignment_receipt_ids"] == [1, 2, 3, 4, 5]
    assert result["post_decision_tail_game_ids"] == [3, 4]
    assert result["game_inventory"]["unpaired_completed_game_ids"] == [5]
    assert result["game_inventory"]["unstarted_mate_game_ids"] == [6]


@pytest.mark.parametrize(
    "damage",
    [
        "missing-receipt",
        "missing-stop-trace",
        "receipt-before-stop",
        "duplicate-receipt",
        "wrong-receipt-worker",
        "missing-internal-id",
        "no-authentic-boundary",
        "extra-tail-receipts",
        "unexpected-skipped-pgn",
        "native-error",
        "occupied-worker-receipt",
        "pending-wrapper-receipt",
    ],
)
def test_internal_gap_needs_exact_clean_bounded_native_skip_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    from test_match import full_game

    module = api()
    plan, raw, process = skipped_assignment_evidence(tmp_path, monkeypatch)
    path = raw / "protocol.log"
    text = path.read_text()
    receipt = "[TRACE] [time] <4> fastchess --- Thread 4 finished match 4\n"
    stop = "[TRACE] [time] <> fastchess --- SPRT test finished, stopping tournament.\n"
    if damage == "missing-receipt":
        text = text.replace(receipt, "")
    elif damage == "missing-stop-trace":
        text = text.replace(stop, "")
    elif damage == "receipt-before-stop":
        text = receipt + text.replace(receipt, "")
    elif damage == "occupied-worker-receipt":
        bad = receipt.replace("<4>", "<3>").replace("Thread 4", "Thread 3")
        text = text.replace(receipt, "").replace(stop, stop + bad)
    elif damage == "pending-wrapper-receipt":
        bad = receipt.replace("<4>", "<3>").replace("Thread 4", "Thread 3")
        preceding = "[TRACE] [time] <3> fastchess --- Thread 3 finished match 3\n"
        text = text.replace(receipt, "").replace(preceding, bad + preceding)
    elif damage == "duplicate-receipt":
        text += receipt
    elif damage == "wrong-receipt-worker":
        text = text.replace(receipt, receipt.replace("Thread 4", "Thread 3"))
    elif damage == "missing-internal-id":
        text = text.replace(receipt, receipt.replace("match 4", "match 7"))
    elif damage == "extra-tail-receipts":
        text += "".join(
            f"[TRACE] [time] <4> fastchess --- Thread 4 finished match {index}\n"
            for index in (6, 7, 8)
        )
    elif damage == "native-error":
        text += "[ERR] [time] <4> fastchess --- real backend error\n"
    elif damage == "no-authentic-boundary":
        stdout = raw / "stdout.txt"
        stdout.write_text(
            "\n".join(
                line for line in stdout.read_text().splitlines() if not line.startswith("SPRT (")
            )
            + "\n"
        )
    else:
        with (raw / "games.pgn").open("a") as stream:
            stream.write(full_game(2, "B", "A"))
    path.write_text(text)
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=process["command"], process=process)


def test_proven_skipped_queued_assignments_may_return_serially_on_same_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = skipped_assignment_evidence(tmp_path, monkeypatch)
    with (raw / "protocol.log").open("a") as stream:
        stream.write("[TRACE] [time] <4> fastchess --- Thread 4 finished match 6\n")
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["game_inventory"]["skipped_before_start_game_ids"] == [4, 6]
    assert result["post_decision_tail_game_ids"] == [3, 4, 6]
    assert result["game_inventory"]["raw_completed_games"] == 3
    assert result["statistics"]["games"] == 2


def test_catalog_capacity_tracks_pool_size_without_worker_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    path = raw / "protocol.log"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if not (line.startswith("[Engine]") and "<4>" in line)
        )
        + "\n"
    )
    with pytest.raises(module.MatchError, match="startup count"):
        module.audit_results(plan, raw, captured_command=process["command"], process=process)


def test_many_same_context_catalogs_still_cannot_exceed_declared_pool_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    _, original = full_plan(tmp_path, monkeypatch)
    plan = replace(original, selected=replace(original.selected, concurrency=4))
    raw = tmp_path / "pool-overflow"
    raw.mkdir()
    retained_evidence(plan, raw, count=4)
    path = raw / "protocol.log"
    text = path.read_text()
    path.write_text(text * 5)
    with pytest.raises(module.MatchError, match="pool capacity"):
        module._protocol(plan, path)


def test_known_wrapper_then_skipped_receipt_can_share_a_now_idle_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = skipped_assignment_evidence(tmp_path, monkeypatch)
    path = raw / "protocol.log"
    path.write_text(
        path.read_text().replace(
            "<4> fastchess --- Thread 4 finished match 4",
            "<3> fastchess --- Thread 3 finished match 4",
        )
    )
    result = module.audit_results(plan, raw, captured_command=process["command"], process=process)
    assert result["game_inventory"]["skipped_before_start_game_ids"] == [4]


def test_every_started_assignment_keeps_its_known_wrapper_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw, process = concurrent_evidence(tmp_path, monkeypatch)
    path = raw / "protocol.log"
    path.write_text(
        "\n".join(
            line
            for line in path.read_text().splitlines()
            if "Thread 4 finished match 8" not in line
        )
        + "\n"
    )
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=process["command"], process=process)
