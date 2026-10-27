# SPDX-License-Identifier: GPL-3.0-or-later
"""Pinned fastchess per-game option refresh; authored records, no engine runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_match import api, full_plan, retained_evidence


def refresh_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repetitions: int = 1000,
) -> tuple[Any, Path]:
    _, plan = full_plan(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw)
    path = raw / "protocol.log"
    catalog = [line for line in path.read_text().splitlines() if "<--- setoption " not in line]
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(catalog) + "\n")
        for _ in range(repetitions):
            for engine in plan.engines:
                # Pinned refreshUci sends Threads first, then all configured options.
                options = sorted(engine.options, key=lambda item: item[0] != "Threads")
                for key, value in options:
                    stream.write(
                        f"[Engine] [time] <1> {engine.name} <--- "
                        f"setoption name {key} value {value}\n"
                    )
                stream.write(f"[Engine] [time] <1> {engine.name} <--- ucinewgame\n")
    return plan, raw


def test_all1000_timed_games_accept_exact_per_game_refresh_with_one_bounded_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    plan, raw = refresh_evidence(tmp_path, monkeypatch)
    before = (raw / "protocol.log").read_bytes()
    result = module.audit_results(plan, raw, captured_command=module.build_command(plan, raw))
    assert result["complete"] and result["safety_pass"]
    assert result["statistics"]["games"] == 1000
    for engine in plan.engines:
        evidence = result["engine_handshakes"][engine.name]
        assert evidence["sent_option_counts"] == {key: 1000 for key, _ in engine.options}
        assert evidence["ucinewgame_count"] == 1000
        assert len(evidence["received_handshake"]) <= 4096
        assert len(evidence["sent_options"]) == len(engine.options) + 1
    assert (raw / "protocol.log").read_bytes() == before


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("Threads", "128"),
        ("Ponder", "true"),
        ("UCI_Chess960", "true"),
        ("Unplanned", "false"),
    ],
)
def test_any_changed_or_unplanned_refresh_fails_even_if_later_corrected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad: str,
) -> None:
    module = api()
    plan, raw = refresh_evidence(tmp_path, monkeypatch, repetitions=1)
    correct = dict(plan.engines[0].options).get(key, "false")
    with (raw / "protocol.log").open("a") as stream:
        stream.write(f"[Engine] [time] <1> A <--- setoption name {key} value {bad}\n")
        stream.write(f"[Engine] [time] <1> A <--- setoption name {key} value {correct}\n")
    with pytest.raises(module.MatchError, match="option|setting"):
        module._protocol(plan, raw / "protocol.log")


@pytest.mark.parametrize(
    "payload",
    [
        "setoption name Threads",
        "setoption name Threads value 4 extra",
        "setoption Threads value 4",
        "setoption name  value 4",
    ],
)
def test_malformed_refresh_never_hides_behind_a_prior_valid_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    module = api()
    plan, raw = refresh_evidence(tmp_path, monkeypatch, repetitions=1)
    with (raw / "protocol.log").open("a") as stream:
        stream.write("[Engine] [time] <1> A <--- " + payload + "\n")
    with pytest.raises(module.MatchError, match="option|setting"):
        module._protocol(plan, raw / "protocol.log")


@pytest.mark.parametrize(
    "payload",
    ["uci", "id name Duplicate", "uciok", "option name Threads type spin default 4 min 1 max 128"],
)
def test_refresh_does_not_relax_unique_fixed_startup_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    module = api()
    plan, raw = refresh_evidence(tmp_path, monkeypatch, repetitions=1)
    direction = "<---" if payload == "uci" else "--->"
    with (raw / "protocol.log").open("a") as stream:
        stream.write(f"[Engine] [time] <1> A {direction} {payload}\n")
    with pytest.raises(module.MatchError):
        module._protocol(plan, raw / "protocol.log")


@pytest.mark.parametrize("kind", ["setoption", "ucinewgame"])
def test_per_game_command_occurrences_cannot_exceed_source_derived_game_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module = api()
    plan, raw = refresh_evidence(tmp_path, monkeypatch, repetitions=1000)
    payload = "setoption name Ponder value false" if kind == "setoption" else "ucinewgame"
    with (raw / "protocol.log").open("a") as stream:
        stream.write(f"[Engine] [time] <1> A <--- {payload}\n")
    with pytest.raises(module.MatchError, match="bound|count|cap"):
        module._protocol(plan, raw / "protocol.log")


def test_exact_planned_string_value_can_itself_contain_the_value_separator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    module = api()
    _, plan = full_plan(tmp_path, monkeypatch)
    value = "C:/Users/a value segment/Mouse/cache"
    engines = tuple(
        replace(engine, options=(*engine.options, ("CacheDirectory", value)))
        for engine in plan.engines
    )
    plan = replace(plan, engines=engines)
    raw = tmp_path / "raw"
    raw.mkdir()
    retained_evidence(plan, raw, count=2)
    result = module._protocol(plan, raw / "protocol.log")
    assert result["A"]["sent_option_counts"]["CacheDirectory"] == 1
