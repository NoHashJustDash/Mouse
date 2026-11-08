# SPDX-License-Identifier: GPL-3.0-or-later
"""SPSA driver: schedule, plan checks, updates, failure handling and resume."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mouse_tools import spsa

REPORT = """Results of plus vs minus (5+0.05, NULL, NULL, UHO):
Elo: 0.00 +/- 0.00, nElo: 0.00 +/- 0.00
Games: {games}, Wins: {wins}, Losses: {losses}, Draws: {draws}, Points: 1.0 (50.00 %)
"""


def plan(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "fastchess": "C:/tools/fastchess.exe",
        "engine": "C:/engines/mouse.exe",
        "options": {"WeightsFile": "C:/nets/net.onnx", "Threads": 2},
        "parameters": [
            {"name": "CPuct", "start": 1.745, "min": 1.0, "max": 3.0, "c_end": 0.1, "r_end": 0.002},
            {
                "name": "FpuValue",
                "start": 0.33,
                "min": 0.0,
                "max": 1.0,
                "c_end": 0.03,
                "r_end": 0.002,
            },
        ],
        "pairs": 4,
        "pairs_per_step": 2,
        "tc": "5+0.05",
        "book": "C:/books/UHO.epd",
        "seed": 7,
        "extra_arguments": ["-draw", "movenumber=40"],
    }
    raw.update(overrides)
    return raw


class FakeMatch:
    """Stands in for fastchess: records each command and replays scripted outcomes."""

    def __init__(self, outcomes: list[tuple[int, int, int] | str | int]) -> None:
        self.outcomes = outcomes
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(argv)
        outcome = self.outcomes[len(self.commands) - 1]
        for flag in ("-log", "-pgnout"):
            Path(argv[argv.index(flag) + 1].removeprefix("file=")).write_text(flag)
        if isinstance(outcome, int):
            return subprocess.CompletedProcess(argv, outcome, "", "")
        if isinstance(outcome, str):
            return subprocess.CompletedProcess(argv, 0, outcome, "")
        wins, losses, draws = outcome
        text = REPORT.format(games=wins + losses + draws, wins=wins, losses=losses, draws=draws)
        return subprocess.CompletedProcess(argv, 0, text, "")


def _option_values(argv: list[str], name: str) -> list[str]:
    prefix = f"option.{name}="
    return [item.removeprefix(prefix) for item in argv if item.startswith(prefix)]


def test_integer_parameter_is_sent_whole_and_keeps_its_fraction_in_theta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = plan(
        parameters=[
            {
                "name": "MaxCollisionVisitsScalingEnd",
                "start": 1450000.4,
                "min": 1000,
                "max": 9000000,
                "c_end": 1000.3,
                "r_end": 0.002,
                "integer": True,
            }
        ]
    )
    match = FakeMatch([(2, 0, 2), (1, 1, 2)])
    monkeypatch.setattr(spsa.subprocess, "run", match)
    final = spsa.run(json.dumps(raw), tmp_path / "run")
    for argv in match.commands:
        values = _option_values(argv, "MaxCollisionVisitsScalingEnd")
        assert len(values) == 2
        assert all(value.isdigit() for value in values)
    assert final["MaxCollisionVisitsScalingEnd"] != round(final["MaxCollisionVisitsScalingEnd"])


def test_composed_option_carries_its_parameters_in_one_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = plan(
        parameters=[
            {
                "name": "slowmover",
                "start": 1.0,
                "min": 0.5,
                "max": 2.0,
                "c_end": 0.1,
                "r_end": 0.002,
            },
            {"name": "CPuct", "start": 1.5, "min": 1.0, "max": 3.0, "c_end": 0.1, "r_end": 0.002},
        ],
        composed={"TimeManager": "legacy(slowmover={slowmover})"},
    )
    match = FakeMatch([(2, 0, 2), (1, 1, 2)])
    monkeypatch.setattr(spsa.subprocess, "run", match)
    spsa.run(json.dumps(raw), tmp_path / "run")
    argv = match.commands[0]
    assert _option_values(argv, "slowmover") == []
    assert len(_option_values(argv, "CPuct")) == 2
    managers = _option_values(argv, "TimeManager")
    assert len(managers) == 2
    assert managers[0] != managers[1]
    for manager in managers:
        assert manager.startswith("legacy(slowmover=") and manager.endswith(")")
        float(manager.removeprefix("legacy(slowmover=").removesuffix(")"))


@pytest.mark.parametrize(
    ("composed", "message"),
    [
        ({"TimeManager": "legacy(steepness={steepness})"}, "unknown parameter"),
        ({"TimeManager": "legacy(CPuct={CPuct})", "Other": "x={CPuct}"}, "more than one"),
        ({"WeightsFile": "x={CPuct}"}, "fixed option"),
        ({"CPuct": "x={FpuValue}"}, "tuned parameter"),
    ],
)
def test_bad_composed_option_is_rejected(composed: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        spsa.load_plan(json.dumps(plan(composed=composed)))


def test_gains_end_at_the_configured_final_values() -> None:
    parameter = spsa.Parameter("CPuct", 1.745, 1.0, 3.0, c_end=0.1, r_end=0.002)
    a_last, c_last = spsa.gains(parameter, 6000, 5999)
    assert c_last == pytest.approx(0.1)
    assert a_last == pytest.approx(0.002 * 0.1**2)
    a_first, c_first = spsa.gains(parameter, 6000, 0)
    assert c_first == pytest.approx(0.1 * 6000**spsa.GAMMA)
    assert a_first > a_last
    assert c_first > c_last


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"pairs": 5}, "multiple of pairs_per_step"),
        ({"pairs_per_step": 0}, "pairs_per_step must be positive"),
        ({"engine": "relative/mouse.exe"}, "absolute"),
        ({"options": {"CPuct": "2"}}, "fixed option"),
        (
            {
                "parameters": [
                    {"name": "CPuct", "start": 5, "min": 1, "max": 3, "c_end": 1, "r_end": 1}
                ]
            },
            "within min..max",
        ),
        (
            {
                "parameters": [
                    {"name": "CPuct", "start": 2, "min": 1, "max": 3, "c_end": 0, "r_end": 1}
                ]
            },
            "gains must be > 0",
        ),
    ],
)
def test_invalid_plans_are_rejected(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        spsa.load_plan(json.dumps(plan(**change)))


def test_a_step_plays_both_sides_and_moves_theta_toward_the_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMatch([(3, 0, 1), (0, 0, 4)])
    monkeypatch.setattr(subprocess, "run", fake)
    text = json.dumps(plan())
    final = spsa.run(text, tmp_path / "run")
    loaded = spsa.load_plan(text)
    first = fake.commands[0]
    assert first[:3] == [str(Path("C:/tools/fastchess.exe")), "-engine", "name=plus"]
    assert "option.WeightsFile=C:/nets/net.onnx" in first
    assert first[first.index("-rounds") + 1] == "2"
    log = first.index("-log")
    games = tmp_path / "run" / "games"
    assert first[log : log + 4] == [
        "-log",
        f"file={games / 'step-00000.engines.log'}",
        "level=info",
        "engine=true",
    ]
    # Engine traffic is only kept for failed attempts.
    assert not list(games.glob("*.engines.log"))
    assert not (tmp_path / "run" / "failures.jsonl").exists()
    flip = spsa.flips(loaded, 0)
    history = [
        json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in history] == [0, 1]
    for index, parameter in enumerate(loaded.parameters):
        a_k, c_k = spsa.gains(parameter, 4, 0)
        plus = history[0]["plus"][parameter.name]
        minus = history[0]["minus"][parameter.name]
        assert plus - minus == pytest.approx(2 * c_k * flip[index])
        expected = parameter.start + a_k / c_k * 3 * flip[index]
        assert history[0]["theta"][parameter.name] == pytest.approx(expected)
        assert f"option.{parameter.name}={spsa.option_text(plus)}" in first
    # An even second step leaves theta where the first put it.
    assert final == history[0]["theta"]


def test_values_stay_inside_the_parameter_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edge = plan(
        parameters=[
            {"name": "FpuValue", "start": 0.0, "min": 0.0, "max": 1.0, "c_end": 0.5, "r_end": 50}
        ],
        pairs=2,
    )
    fake = FakeMatch([(4, 0, 0)])
    monkeypatch.setattr(subprocess, "run", fake)
    final = spsa.run(json.dumps(edge), tmp_path / "run")
    row = json.loads((tmp_path / "run" / "history.jsonl").read_text())
    assert 0.0 <= row["plus"]["FpuValue"] <= 1.0
    assert 0.0 <= row["minus"]["FpuValue"] <= 1.0
    assert 0.0 <= final["FpuValue"] <= 1.0


@pytest.mark.parametrize(
    "outcome",
    [
        "Engine minus disconnects\nGames: 1, Wins: 1, Losses: 0, Draws: 0, Points",
        "Game 3 stalled / disconnected and no recover option set",
        "nothing useful",
        3,
    ],
)
def test_failed_steps_stop_the_run_without_counting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str | int
) -> None:
    monkeypatch.setattr(subprocess, "run", FakeMatch([(1, 0, 3), outcome]))
    with pytest.raises((RuntimeError, ValueError)):
        spsa.run(json.dumps(plan()), tmp_path / "run")
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["played"] == 2
    assert len((tmp_path / "run" / "history.jsonl").read_text().splitlines()) == 1
    failures = (tmp_path / "run" / "failures.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in failures] == [1]
    assert (tmp_path / "run" / "games" / "step-00001.engines.log").exists()


def test_a_time_loss_counts_as_a_game_but_a_crash_count_stops_the_step() -> None:
    # fastchess 1.8.2 output of step 1147, whose one time loss adds a per-player block.
    report = """Results of plus vs minus (5+0.05, 2t, NULL, UHO_Lichess_4852_v1.epd):
Elo: -0.00 +/- 125.57, nElo: 0.00 +/- 240.76
LOS: 50.00 %, DrawRatio: 50.00 %, PairsRatio: 1.00
Games: 8, Wins: 1, Losses: 1, Draws: 6, Points: 4.0 (50.00 %)
Ptnml(0-2): [0, 1, 2, 1, 0], WL/DD Ratio: 0.00
--------------------------------------------------

Player: minus
  Timeouts: 1
  Crashed: {crashed}

Finished match
"""
    assert spsa.match_result(report.format(crashed=0)) == (1, 1, 6)
    with pytest.raises(RuntimeError, match="Crashed: 1"):
        spsa.match_result(report.format(crashed=1))


def test_a_failed_step_is_replayed_unchanged_and_its_files_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stalled = "Game 1 stalled / disconnected and no recover option set"
    fake = FakeMatch([(1, 0, 3), stalled, (2, 0, 0)])
    monkeypatch.setattr(subprocess, "run", fake)
    text = json.dumps(plan())
    final = spsa.run(text, tmp_path / "run", attempts=2)
    assert fake.commands[1] == fake.commands[2]
    games = tmp_path / "run" / "games"
    assert (games / "step-00001.failed-1.log").read_text(encoding="utf-8") == stalled
    assert (games / "step-00001.failed-1.engines.log").exists()
    assert (games / "step-00001.failed-1.pgn").exists()
    assert not (games / "step-00001.engines.log").exists()
    failures = [
        json.loads(line) for line in (tmp_path / "run" / "failures.jsonl").read_text().splitlines()
    ]
    assert [(row["step"], row["attempt"]) for row in failures] == [(1, 1)]
    assert "stalled" in failures[0]["error"]
    history = [
        json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text().splitlines()
    ]
    assert [(row["step"], row["wins"]) for row in history] == [(0, 1), (1, 2)]
    monkeypatch.setattr(subprocess, "run", FakeMatch([(1, 0, 3), (2, 0, 0)]))
    assert final == pytest.approx(spsa.run(text, tmp_path / "reference"))


def test_a_step_that_fails_every_attempt_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subprocess, "run", FakeMatch([(1, 0, 3), 3, 3]))
    with pytest.raises(ValueError, match="fastchess exited 3 at 1"):
        spsa.run(json.dumps(plan()), tmp_path / "run", attempts=2)
    games = tmp_path / "run" / "games"
    assert (games / "step-00001.failed-1.engines.log").exists()
    assert (games / "step-00001.engines.log").exists()
    failures = (tmp_path / "run" / "failures.jsonl").read_text().splitlines()
    assert [json.loads(line)["attempt"] for line in failures] == [1, 2]
    assert json.loads((tmp_path / "run" / "state.json").read_text())["played"] == 2
    # Resuming sets the last attempt aside as well before playing the step again.
    monkeypatch.setattr(subprocess, "run", FakeMatch([(0, 0, 4)]))
    spsa.run(json.dumps(plan()), tmp_path / "run")
    assert (games / "step-00001.failed-2.engines.log").exists()
    assert not (games / "step-00001.engines.log").exists()
    with pytest.raises(ValueError, match="attempts must be at least 1"):
        spsa.run(json.dumps(plan()), tmp_path / "other", attempts=0)


def test_a_stopped_run_resumes_with_the_same_flips_and_rejects_another_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = json.dumps(plan())
    monkeypatch.setattr(subprocess, "run", FakeMatch([(2, 1, 1), 1]))
    with pytest.raises(ValueError):
        spsa.run(text, tmp_path / "run")
    uninterrupted = FakeMatch([(2, 1, 1), (1, 2, 1)])
    monkeypatch.setattr(subprocess, "run", uninterrupted)
    reference = spsa.run(text, tmp_path / "reference")
    resumed = FakeMatch([(1, 2, 1)])
    monkeypatch.setattr(subprocess, "run", resumed)
    assert spsa.run(text, tmp_path / "run") == pytest.approx(reference)
    moved = [
        argument.replace(str(tmp_path / "run"), str(tmp_path / "reference"))
        for argument in resumed.commands[0]
    ]
    assert moved == uninterrupted.commands[1]
    with pytest.raises(ValueError, match="another plan"):
        spsa.run(json.dumps(plan(seed=8)), tmp_path / "run")
