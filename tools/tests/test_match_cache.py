# SPDX-License-Identifier: GPL-3.0-or-later
"""The V1 3.3 cache speed gate uses fixed nodes and a separate [-5,0] SPRT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_match import api, controlled_config


def cache_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    path, document = controlled_config(tmp_path, monkeypatch, "cache-sprt")
    document.update(max_games=20000, nodes=800, concurrency=1)
    common = dict(
        CPuct="1.745",
        CPuctBase="38739",
        CPuctFactor="3.894",
        FpuValue="0.330",
        FpuStrategyAtRoot="same",
        FpuValueAtRoot="1",
        PolicyTemperature="1.359",
        Deterministic="false",
    )
    # Native Windows probe: the unchanged 256-MiB raw cache has 28,876 entries
    # at the actual 64-byte namespace. The new cache is configured in entries.
    for engine, size in zip(document["engines"], ("28876", "256"), strict=True):
        engine["options"] = {**engine["options"], **common, "NNCacheSize": size}
    for engine in document["engines"]:
        engine["options"].update(SmartPruningFactor="0", SmartPruningMinimumBatches="0")
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_cache_speed_profile_is_serial_fixed_node_nonregression() -> None:
    selected = api().profile("cache-sprt", max_games=20000, nodes=800)
    assert (selected.games, selected.nodes, selected.tc, selected.sprt) == (
        20000,
        800,
        None,
        (-5, 0),
    )
    assert selected.concurrency == 1 and selected.adjudication
    assert selected.book == "8moves_v3.pgn"


def test_cache_speed_plan_has_matched_normal_search_without_clock_or_tuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, _ = cache_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    command = module.build_command(plan, tmp_path / "out")
    assert command[command.index("-concurrency") + 1] == "1"
    assert "elo0=-5" in command and "elo1=0" in command and "nodes=800" in command
    assert "alpha=0.05" in command and "beta=0.05" in command and "model=normalized" in command
    assert "-draw" in command and "-resign" in command and "-maxmoves" in command
    assert not any(token.startswith(("tc=", "st=", "depth=")) for token in command)
    assert command.count("option.Threads=4") == command.count("option.MinibatchSize=64") == 2
    assert "option.NNCacheSize=28876" in command and "option.NNCacheSize=256" in command


@pytest.mark.parametrize(
    ("engine", "key", "value"),
    [
        (0, "NNCacheSize", "0"),
        (1, "NNCacheSize", "0"),
        (0, "CPuct", "2"),
        (1, "FpuStrategyAtRoot", "absolute"),
        (0, "FpuValueAtRoot", "0"),
        (1, "PolicyTemperature", "1"),
        (1, "Deterministic", "true"),
        (0, "SmartPruningFactor", "1.33"),
        (0, "SmartPruningMinimumBatches", "1"),
        (1, "SmartPruningFactor", "1"),
        (0, "Threads", "1"),
        (1, "MinibatchSize", "1"),
    ],
)
def test_cache_gate_rejects_disabled_cache_or_mismatched_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: int, key: str, value: str
) -> None:
    path, document = cache_document(tmp_path, monkeypatch)
    document["engines"][engine]["options"][key] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(api().MatchError):
        api().load_plan(path)


@pytest.mark.parametrize("concurrency", [4, 2, 0, True])
def test_cache_sprt_never_runs_parallel_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrency: object
) -> None:
    path, document = cache_document(tmp_path, monkeypatch)
    document["concurrency"] = concurrency
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(api().MatchError):
        api().load_plan(path)


@pytest.mark.parametrize(
    "limits",
    [
        {},
        {"max_games": 3, "nodes": 800},
        {"max_games": 20000},
        {"max_games": 20000, "nodes": 1},
        {"max_games": 20000, "nodes": True},
    ],
)
def test_cache_sprt_requires_complete_pairs_and_the_fixed_budget(limits: dict[str, Any]) -> None:
    with pytest.raises(api().MatchError):
        api().profile("cache-sprt", **limits)


@pytest.mark.parametrize(("key", "value"), [("RamLimitMb", "128"), ("UCI_ShowWDL", "true")])
def test_cache_gate_rejects_unrelated_runtime_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    path, document = cache_document(tmp_path, monkeypatch)
    document["engines"][0]["options"][key] = value
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(api().MatchError):
        api().load_plan(path)


def test_cache_gate_requires_explicit_pruning_off_on_the_pre_cache_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = cache_document(tmp_path, monkeypatch)
    del document["engines"][1]["options"]["SmartPruningFactor"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(api().MatchError):
        api().load_plan(path)
