# SPDX-License-Identifier: GPL-3.0-or-later
"""Auditable match evidence using authored child processes, never real engines."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest


def api() -> Any:
    try:
        return importlib.import_module("mouse_tools.match")
    except ModuleNotFoundError:
        pytest.fail("Production match runner is missing")


def child(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


def test_all_fixed_profiles_keep_exact_gate_counts_and_limits() -> None:
    module = api()
    expected = {
        "selfplay-fast": (1000, "1+0.01"),
        "uci-stress-fast": (2000, "1+0.01"),
        "uci-stress-stc": (500, "10+0.1"),
        "tm-safety-fast": (2000, "1+0.01"),
        "tm-safety-stc": (2000, "10+0.1"),
        "tm-safety-ltc": (2000, "60+0.6"),
    }
    for name, (games, tc) in expected.items():
        profile = module.profile(name)
        assert (profile.games, profile.tc, profile.sprt) == (games, tc, None)
    with pytest.raises(module.MatchError):
        module.profile("diagnostic-subset")


def test_sprt_parameters_are_normalized_and_caps_are_not_fixed_passes() -> None:
    module = api()
    parity = module.profile("parity-sprt", max_games=2000, nodes=800)
    assert (parity.games, parity.nodes, parity.tc, parity.sprt) == (2000, 800, None, (-10, 0))
    tm = module.profile("tm-sprt", max_games=4000)
    assert (tm.games, tm.tc, tm.sprt) == (4000, "10+0.1", (0, 5))
    for kwargs in ({}, {"max_games": 3}, {"max_games": 2000, "nodes": 0}):
        with pytest.raises(module.MatchError):
            module.profile("parity-sprt", **kwargs)
    with pytest.raises(module.MatchError):
        module.profile("selfplay-fast", max_games=2)


def pgn_game(
    round_id: int,
    white: str,
    black: str,
    result: str = "1/2-1/2",
    termination: str = "normal",
    opening: str = "e4",
) -> str:
    return (
        f'[Event "Controlled evidence"]\n[Round "{round_id}"]\n'
        f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n'
        f'[PlyCount "2"]\n[Termination "{termination}"]\n'
        '[TimeControl "1+0.01"]\n\n'
        f"1. {opening} {{book}} e5 {{0.00/1 0.001s, tl=1.000s, n=8, Draw by 3-fold repetition}} "
        f"{result}\n\n"
    )


def test_pgn_parser_checks_complete_result_and_pairing(tmp_path: Path) -> None:
    module = api()
    path = tmp_path / "games.pgn"
    path.write_text(pgn_game(1, "A", "B") + pgn_game(1, "B", "A"))
    games = module.read_games(path, ("A", "B"))
    assert len(games) == 2
    assert games[0].book_moves == ("e4",)
    assert games[0].result == "1/2-1/2"
    assert module.pair_statistics(games)["pentanomial"] == [0, 0, 1, 0, 0]


@pytest.mark.parametrize(
    "mutation",
    ["result", "duplicate_tag", "unfinished", "termination", "colors", "opening", "round"],
)
def test_bad_pgn_or_incomplete_pair_never_establishes_evidence(
    tmp_path: Path, mutation: str
) -> None:
    module = api()
    first = pgn_game(1, "A", "B")
    second = pgn_game(1, "B", "A")
    if mutation == "result":
        first = first.replace(" repetition} 1/2-1/2", " repetition} 1-0")
    elif mutation == "duplicate_tag":
        first = first.replace("[Result", '[Round "1"]\n[Result')
    elif mutation == "unfinished":
        second = second[:-10]
    elif mutation == "termination":
        first = first.replace("normal", "time forfeit")
    elif mutation == "colors":
        second = pgn_game(1, "A", "B")
    elif mutation == "opening":
        second = pgn_game(1, "B", "A", opening="d4")
    else:
        second = pgn_game(2, "B", "A")
    path = tmp_path / "bad.pgn"
    path.write_text(first + second)
    with pytest.raises(module.MatchError):
        module.pair_statistics(module.read_games(path, ("A", "B")))


def test_pair_statistics_use_actual_candidate_colors_and_paired_variance(tmp_path: Path) -> None:
    module = api()
    path = tmp_path / "scores.pgn"
    path.write_text(
        "".join(
            pgn_game(pair + 1, white, black, result)
            for pair, (left, right) in enumerate(
                [("0-1", "1-0"), ("1-0", "1-0"), ("1/2-1/2", "1/2-1/2"), ("1-0", "0-1")]
            )
            for white, black, result in [("A", "B", left), ("B", "A", right)]
        )
    )
    stats = module.pair_statistics(module.read_games(path, ("A", "B")))
    assert stats["wdl"] == {"wins": 3, "draws": 2, "losses": 3}
    assert stats["pentanomial"] == [1, 0, 2, 0, 1]
    assert stats["pair_categories"]["WL"] == stats["pair_categories"]["DD"] == 1
    assert stats["elo"] == stats["nelo"] == 0
    assert stats["score_variance_per_pair"] == pytest.approx(0.125)
    expected = 1.959963984540054 * (800 / math.log(10)) / math.sqrt(8)
    assert stats["nelo_ci95"] == pytest.approx([-expected, expected])


def test_degenerate_statistics_are_explicitly_unavailable_not_fake_finite_values(
    tmp_path: Path,
) -> None:
    module = api()
    path = tmp_path / "draws.pgn"
    path.write_text(pgn_game(1, "A", "B") + pgn_game(1, "B", "A"))
    stats = module.pair_statistics(module.read_games(path, ("A", "B")))
    assert stats["nelo"] is None
    assert stats["nelo_ci95"] is None
    assert stats["statistics_status"] == "degenerate_pair_variance"
    json.dumps(stats, allow_nan=False)


def test_disk_capture_preserves_exact_streams_and_reaps_child(tmp_path: Path) -> None:
    module = api()
    command = child("import sys;print('retained');print('diagnostic',file=sys.stderr)")
    result = module.capture(
        command, tmp_path, idle_seconds=2, total_seconds=3, max_bytes=1024 * 1024
    )
    assert result["exit_code"] == 0
    assert result["pumps_joined"]
    assert (tmp_path / "stdout.txt").read_bytes() == (
        b"retained\r\n" if sys.platform == "win32" else b"retained\n"
    )
    assert b"diagnostic" in (tmp_path / "stderr.txt").read_bytes()
    assert result["command"] == command


@pytest.mark.parametrize(
    "code",
    [
        "import time;time.sleep(60)",
        "import sys;sys.stdout.write('x'*5000);sys.stdout.flush();import time;time.sleep(60)",
    ],
)
def test_watchdog_fails_without_invented_game_or_detached_pump(tmp_path: Path, code: str) -> None:
    module = api()
    before = {thread.ident for thread in threading.enumerate()}
    result = module.capture(
        child(code), tmp_path, idle_seconds=0.15, total_seconds=0.3, max_bytes=1024
    )
    assert result["error"]
    assert result["terminated_by_runner"]
    assert result["pumps_joined"]
    assert {thread.ident for thread in threading.enumerate()} == before


def test_process_tree_cleanup_prevents_orphaned_descendant(tmp_path: Path) -> None:
    module = api()
    marker = tmp_path / "orphan"
    descendant = (
        "import time;from pathlib import Path;time.sleep(.6);"
        f"Path({str(marker)!r}).write_text('leaked');time.sleep(60)"
    )
    code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-I','-c',{descendant!r}]);"
        "print('ready',flush=True);time.sleep(60)"
    )
    result = module.capture(
        child(code), tmp_path, idle_seconds=0.15, total_seconds=0.3, max_bytes=1024 * 1024
    )
    assert result["error"]
    assert result["pumps_joined"]
    time.sleep(0.7)
    assert not marker.exists()


def controlled_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_name: str = "selfplay-fast"
) -> tuple[Path, dict[str, Any]]:
    module = api()
    root = tmp_path / "inputs with spaces"
    root.mkdir()
    engine = root / "mouse engine.exe"
    engine.write_bytes(b"controlled binary identity")
    model = root / "model.onnx"
    import onnx
    from onnx import helper

    source = root / "source.pb.gz"
    source.write_bytes(b"controlled network source")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "T1_SHA256", digest, raising=False)
    net = helper.make_model(helper.make_graph([], "metadata-only validation", [], []))
    net.metadata_props.add(key="mouse.source_sha256", value=digest)
    net.metadata_props.add(key="mouse.precision", value="fp32")
    onnx.save_model(net, model)
    library = root / "onnxruntime.dll"
    library.write_bytes(b"controlled runtime")
    source_root = root / "engine"
    source_root.mkdir()
    (source_root / "owned.cpp").write_text("// original test source\n")
    book = root / "8moves_v3.pgn"
    book.write_text(pgn_game(1, "?", "?"))
    runner = root / "fastchess.exe"
    runner.write_bytes(b"controlled fastchess identity")
    monkeypatch.setitem(
        module.BOOKS, "8moves_v3.pgn", (hashlib.sha256(book.read_bytes()).hexdigest(), "pgn", 1)
    )
    monkeypatch.setattr(
        module, "FASTCHESS_SHA256", hashlib.sha256(runner.read_bytes()).hexdigest(), raising=False
    )
    options = {
        "WeightsFile": str(model),
        "Backend": "cpu",
        "Precision": "fp32",
        "Threads": "4",
        "MinibatchSize": "64",
        "NNCacheSize": "0",
        "RamLimitMb": "64",
        "Ponder": "false",
        "MultiPV": "1",
        "UCI_Chess960": "false",
        "MoveOverheadMs": "0",
    }
    config: dict[str, Any] = {
        "schema": "MouseMatchPlan1",
        "profile": profile_name,
        "seed": 20261017,
        "fastchess": str(runner),
        "book": str(book),
        "engines": [
            {
                "name": name,
                "kind": "mouse",
                "executable": str(engine),
                "model": str(model),
                "source_net": str(source),
                "args": [],
                "options": options,
            }
            for name in ("A", "B")
        ],
        "runtime_directories": [str(root)],
        "source_roots": [str(source_root)],
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (runner, engine, model, source, library, source_root / "owned.cpp", book)
        },
        "idle_seconds": 900,
        "total_seconds": 86400,
        "max_output_bytes": 16 * 1024 * 1024,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(config))
    return path, config


def test_plan_produces_exact_reviewable_bounded_command_and_no_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    command = module.build_command(plan, tmp_path / "run")
    assert command[0] == document["fastchess"]
    assert command[command.index("-rounds") + 1] == "500"
    assert command[command.index("-concurrency") + 1] == "1"
    for token in (
        "-repeat",
        "-strict",
        "tc=1+0.01",
        "timemargin=0",
        "restart=off",
        "option.Ponder=false",
        "nodes=true",
        "timeleft=true",
    ):
        assert token in command
    for option in ("-startup-ms", "-ping-ms", "-ucinewgame-ms"):
        assert command[command.index(option) + 1] == "300000"
    assert not any(
        token in command
        for token in ("-quick", "-recover", "-draw", "-resign", "-maxmoves", "-noswap")
    )
    assert any("inputs with spaces" in token for token in command)
    report = module.run_match(path, tmp_path / "review", execute=False)
    assert report["status"] == "planned"
    assert not report["passed"]
    assert not report["native_execution"]
    assert not (tmp_path / "review/stdout.txt").exists()


@pytest.mark.parametrize(
    "mutation", ["count", "seed", "unknown", "model_hash", "inventory", "no_ponder"]
)
def test_unsafe_or_changed_plan_is_rejected_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch)
    if mutation == "count":
        document["max_games"] = 2
    elif mutation == "seed":
        document["seed"] = 42
    elif mutation == "unknown":
        document["recover"] = True
    elif mutation == "model_hash":
        Path(document["engines"][0]["model"]).write_bytes(b"replacement")
    elif mutation == "inventory":
        (Path(document["runtime_directories"][0]) / "late.dll").write_bytes(b"added runtime")
    else:
        document["engines"][0]["options"]["Ponder"] = "true"
    path.write_text(json.dumps(document))
    with pytest.raises(module.MatchError):
        module.load_plan(path)


def test_plan_snapshot_refuses_changes_after_model_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    Path(document["engines"][0]["executable"]).write_bytes(b"changed after freeze")
    with pytest.raises(module.MatchError, match="changed|hash"):
        plan.verify()


def test_existing_evidence_directory_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, _ = controlled_config(tmp_path, monkeypatch)
    output = tmp_path / "retained"
    output.mkdir()
    (output / "report.json").write_text("original successful evidence")
    with pytest.raises(module.MatchError):
        module.run_match(path, output, execute=False)
    assert (output / "report.json").read_text() == "original successful evidence"


OPENING = (
    "e4",
    "e5",
    "Nf3",
    "Nc6",
    "Bb5",
    "a6",
    "Ba4",
    "Nf6",
    "O-O",
    "Be7",
    "Re1",
    "b5",
    "Bb3",
    "d6",
    "c3",
    "O-O",
)


def full_game(round_id: int, white: str, black: str) -> str:
    tags = (
        pgn_game(round_id, white, black)
        .split("\n\n", 1)[0]
        .replace('[PlyCount "2"]', '[PlyCount "18"]')
    )
    body = " ".join(
        f"{index // 2 + 1}. {move} {{book}}" if index % 2 == 0 else f"{move} {{book}}"
        for index, move in enumerate(OPENING)
    )
    body += (
        " 9. h3 {0.00/1 0.001s, tl=1.000s, n=8} "
        "Nb8 {0.00/1 0.001s, tl=1.000s, n=8, Draw by 3-fold repetition} 1/2-1/2\n\n"
    )
    return tags + "\n\n" + body


def retained_evidence(plan: Any, output: Path, count: int | None = None) -> None:
    count = plan.selected.games if count is None else count
    names = [engine.name for engine in plan.engines]
    pgn = "".join(
        full_game(index // 2 + 1, *names[:: 1 if index % 2 == 0 else -1]) for index in range(count)
    )
    (output / "games.pgn").write_text(pgn)
    lines = []
    for index in range(1, count + 1):
        white, black = names if index % 2 else names[::-1]
        lines += [
            f"Started game {index} of {plan.selected.games} ({white} vs {black})",
            f"Finished game {index} ({white} vs {black}): 1/2-1/2 {{Draw by 3-fold repetition}}",
        ]
    lines += [
        f"Games: {count}, Wins: 0, Losses: 0, Draws: {count}, Points: {count / 2:.1f} (50.00 %)",
        f"Ptnml(0-2): [0, 0, {count // 2}, 0, 0], WL/DD Ratio: 0.00",
        "Finished match",
        "Total Time: 00:01:00 (hours:minutes:seconds)",
    ]
    (output / "stdout.txt").write_text("\n".join(lines) + "\n")
    (output / "stderr.txt").write_text("")
    protocols = ["[INFO] [time] <1> fastchess --- fastchess alpha 1.8.2"]
    for engine in plan.engines:
        prefix = f"[Engine] [time] <1> {engine.name}"
        protocols += [
            f"{prefix} <--- uci",
            f"{prefix} ---> id name Controlled original engine",
            f"{prefix} ---> uciok",
        ]
        protocols += [
            f"{prefix} ---> option name {key} type string default {value}"
            for key, value in engine.options
        ]
        protocols += [
            f"{prefix} <--- setoption name {key} value {value}" for key, value in engine.options
        ]
    (output / "protocol.log").write_text("\n".join(protocols) + "\n")
    stats = {
        "wins": 0,
        "losses": 0,
        "draws": count,
        "penta_WW": 0,
        "penta_WD": 0,
        "penta_WL": 0,
        "penta_DD": count // 2,
        "penta_LD": 0,
        "penta_LL": 0,
    }
    tc = {"time": 1000, "increment": 10, "fixed_time": 0, "moves": 0, "timemargin": 0}
    saved = {
        "games": 2,
        "rounds": plan.selected.games // 2,
        "concurrency": 1,
        "seed": 20261017,
        "recover": False,
        "noswap": False,
        "reverse": False,
        "report_penta": True,
        "strict": True,
        "opening": {"file": str(plan.book), "format": 1, "order": 0, "plies": 16, "start": 1},
        "sprt": {
            "enabled": False,
            "model": "normalized",
            "alpha": 0,
            "beta": 0,
            "elo0": 0,
            "elo1": 0,
        },
        "engines": [
            {
                "name": e.name,
                "cmd": str(e.executable),
                "args": "",
                "options": [list(p) for p in e.options],
                "restart": False,
                "limit": {"nodes": 0, "plies": 0, "tc": tc},
            }
            for e in plan.engines
        ],
        "stats": {names[0] + " vs " + names[1]: stats},
    }
    for key in ("draw", "resign", "maxmoves", "tb_adjudication"):
        saved[key] = {"enabled": False}
    (output / "fastchess.json").write_text(json.dumps(saved))


def full_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch)
    book = Path(document["book"])
    book.write_text(full_game(1, "?", "?"))
    digest = hashlib.sha256(book.read_bytes()).hexdigest()
    document["sha256"][str(book)] = digest
    monkeypatch.setitem(module.BOOKS, book.name, (digest, "pgn", 1))
    path.write_text(json.dumps(document))
    return path, module.load_plan(path)


def test_complete_all1000_game_evidence_reconciles_pgn_wdl_pairs_book_and_native_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    _, plan = full_plan(tmp_path, monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    retained_evidence(plan, output)
    result = module.audit_results(plan, output, captured_command=module.build_command(plan, output))
    assert result["complete"] and result["safety_pass"]
    assert result["statistics"]["games"] == 1000
    assert result["statistics"]["pairs"] == 500
    assert result["statistics"]["wdl"]["draws"] == 1000
    assert result["book_membership_validated"]


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "duplicate_game",
        "wrong_book",
        "forfeit",
        "counts",
        "options",
        "adjudication",
        "unknown_error",
        "late_stdout",
        "no_handshake",
        "fake_success",
    ],
)
def test_reassuring_summary_cannot_hide_incomplete_or_bad_match_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    module = api()
    _, plan = full_plan(tmp_path, monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    retained_evidence(plan, output)
    pgn = output / "games.pgn"
    stdout = output / "stdout.txt"
    saved = output / "fastchess.json"
    if mutation == "partial":
        pgn.write_text(pgn.read_text().rsplit("[Event", 1)[0])
    elif mutation == "duplicate_game":
        pgn.write_text(pgn.read_text() + full_game(500, "B", "A"))
    elif mutation == "wrong_book":
        pgn.write_text(pgn.read_text().replace("e4 {book}", "d4 {book}"))
    elif mutation == "forfeit":
        pgn.write_text(pgn.read_text().replace("normal", "time forfeit", 1))
    elif mutation == "counts":
        stdout.write_text(stdout.read_text().replace("Draws: 1000", "Draws: 998"))
    elif mutation == "options":
        saved.write_text(saved.read_text().replace('"restart": false', '"restart": true'))
    elif mutation == "adjudication":
        saved.write_text(saved.read_text().replace('"enabled": false', '"enabled": true', 1))
    elif mutation == "unknown_error":
        (output / "protocol.log").write_text(
            (output / "protocol.log").read_text()
            + "[ERR] [time] <1> A ---> info string error real failure\n"
        )
    elif mutation == "late_stdout":
        stdout.write_text(stdout.read_text() + "Warning; engine crashed after final result\n")
    elif mutation == "no_handshake":
        (output / "protocol.log").write_text("missing real option handshake\n")
    else:
        pgn.unlink()
    with pytest.raises((module.MatchError, FileNotFoundError)):
        module.audit_results(plan, output, captured_command=module.build_command(plan, output))


def test_actual_controlled_process_full_run_preserves1000_games_and_frozen_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, plan = full_plan(tmp_path, monkeypatch)
    source = tmp_path / "generated"
    source.mkdir()
    retained_evidence(plan, source)
    install_simulated_capture(monkeypatch, source)
    report = module.run_match(path, tmp_path / "controlled-run", execute=True)
    assert report["passed"]
    assert report["native_execution"]
    assert report["complete"] and report["hashes_unchanged"]
    assert report["statistics"]["games"] == 1000
    assert report["process"]["pumps_joined"]
    assert (tmp_path / "controlled-run/report.json").is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_mate",
        "missing_move_comment",
        "balanced_fen",
        "interleaved_games",
        "wrong_points",
        "plain_warning",
        "missing_advertised_option",
    ],
)
def test_strict_native_evidence_rejects_more_coherent_but_invalid_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    module = api()
    _, plan = full_plan(tmp_path, monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    retained_evidence(plan, output)
    pgn = output / "games.pgn"
    stdout = output / "stdout.txt"
    if mutation == "wrong_mate":
        pgn.write_text(pgn.read_text().replace("Draw by 3-fold repetition", "Black mates", 1))
        stdout.write_text(stdout.read_text().replace("Draw by 3-fold repetition", "Black mates", 1))
    elif mutation == "missing_move_comment":
        pgn.write_text(pgn.read_text().replace(" {0.00/1 0.001s, tl=1.000s, n=8}", "", 1))
    elif mutation == "balanced_fen":
        pgn.write_text(
            pgn.read_text().replace(
                "[TimeControl", '[SetUp "1"]\n[FEN "8/8/8/8/8/8/4K3/7k w - - 0 1"]\n[TimeControl'
            )
        )
    elif mutation == "interleaved_games":
        lines = stdout.read_text().splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        stdout.write_text("\n".join(lines) + "\n")
    elif mutation == "wrong_points":
        stdout.write_text(stdout.read_text().replace("Points: 500.0", "Points: 1.0"))
    elif mutation == "plain_warning":
        stdout.write_text(stdout.read_text() + "Warning; Unknown option ignored\n")
    else:
        path = output / "protocol.log"
        path.write_text(
            "\n".join(
                line
                for line in path.read_text().splitlines()
                if "option name WeightsFile type " not in line
            )
            + "\n"
        )
    with pytest.raises(module.MatchError):
        module.audit_results(plan, output, captured_command=module.build_command(plan, output))


def test_unknown_runner_binary_cannot_impersonate_pinned_fastchess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch)
    binary = Path(document["fastchess"])
    binary.write_bytes(b"a different runner not verified by the publisher")
    document["sha256"][str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(module.MatchError, match="fastchess|runner"):
        module.load_plan(path)


def test_output_limit_retains_the_actual_captured_prefix(tmp_path: Path) -> None:
    result = api().capture(
        child("import sys;sys.stdout.buffer.write(b'x'*4096);sys.stdout.flush()"),
        tmp_path,
        idle_seconds=1,
        total_seconds=2,
        max_bytes=1024,
    )
    assert result["error"]
    assert (tmp_path / "stdout.txt").read_bytes() == b"x" * 1024


def controlled_command(source: Path, extra: str = "") -> list[str]:
    code = (
        "from pathlib import Path;import sys\n"
        f"root=Path({str(source)!r})\n"
        "for p in root.iterdir():\n"
        " if p.name not in ('stdout.txt','stderr.txt'): Path(p.name).write_bytes(p.read_bytes())\n"
        "sys.stdout.write((root/'stdout.txt').read_text());sys.stdout.flush()\n" + extra
    )
    return child(code)


def install_simulated_capture(
    monkeypatch: pytest.MonkeyPatch, source: Path, extra: str = ""
) -> None:
    """Keep authored-child transport separate from simulated native argv metadata."""
    module = api()
    transport = module.capture

    def capture(command: list[str], output: Path, **kwargs: Any) -> dict[str, Any]:
        result = dict(transport(controlled_command(source, extra), output, **kwargs))
        result["fixture_transport_command"] = result["command"]
        result["command"] = list(command)  # Explicit native-invocation unit fixture.
        return result

    monkeypatch.setattr(module, "capture", capture)


@pytest.mark.parametrize(
    "mutation", ["late_nonzero", "source_changed", "raw_changed", "publication_race"]
)
def test_full_runner_retains_failed_evidence_without_overwriting_good_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    module = api()
    path, plan = full_plan(tmp_path, monkeypatch)
    source = tmp_path / "generated"
    source.mkdir()
    retained_evidence(plan, source)
    out = tmp_path / "failed-evidence"
    extra = ""
    if mutation == "late_nonzero":
        extra = "raise SystemExit(7)\n"
    elif mutation == "source_changed":
        extra = f'Path({str(plan.engines[0].executable)!r}).write_bytes(b"changed during match")\n'
    elif mutation == "raw_changed":
        original_audit = module.audit_results

        def replace_after_audit(p: Any, directory: Path, **kwargs: Any) -> dict[str, Any]:
            result = dict(original_audit(p, directory, **kwargs))
            (directory / "stdout.txt").write_text("replacement after validation")
            return result

        monkeypatch.setattr(module, "audit_results", replace_after_audit)
    else:
        original_link = module.os.link

        def race(source: str | Path, target: str | Path) -> None:
            if Path(target).name == "report.json":
                Path(target).write_text("original success won publication")
            original_link(source, target)

        monkeypatch.setattr(module.os, "link", race)
    install_simulated_capture(monkeypatch, source, extra)
    report = module.run_match(path, out, execute=True)
    assert not report["passed"]
    assert report["errors"]
    assert (out / "games.pgn").is_file()
    if mutation == "publication_race":
        assert (out / "report.json").read_text() == "original success won publication"
        assert len(list(out.glob("report.failed-*.json"))) == 1
    else:
        assert json.loads((out / "report.json").read_text())["passed"] is False
    assert not list(out.glob("*.tmp"))


FEN_EP = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 1"


def sprt_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch, "tm-sprt")
    old_book = document["book"]
    book = Path(old_book).with_name("UHO_Lichess_4852_v1.epd")
    book.write_text(" ".join(FEN_EP.split()[:4]) + "\n")
    digest = hashlib.sha256(book.read_bytes()).hexdigest()
    document["book"] = str(book)
    document["sha256"].pop(old_book)
    document["sha256"][str(book)] = digest
    document["max_games"] = 1000
    monkeypatch.setitem(module.BOOKS, book.name, (digest, "epd", 1))
    path.write_text(json.dumps(document))
    return path, module.load_plan(path)


def sprt_evidence(plan: Any, output: Path, decision: str | None, llr: float) -> None:
    retained_evidence(plan, output)
    games = []
    for index in range(plan.selected.games):
        white, black = ("A", "B") if index % 2 == 0 else ("B", "A")
        text = pgn_game(index // 2 + 1, white, black)
        text = text.replace(
            '[TimeControl "1+0.01"]', f'[SetUp "1"]\n[FEN "{FEN_EP}"]\n[TimeControl "10+0.1"]'
        )
        text = text.replace("e4 {book}", "Nf3 {0.00/1 0.001s, tl=10.000s, n=8}").replace(
            "e5 {", "Nc6 {"
        )
        games.append(text)
    (output / "games.pgn").write_text("".join(games))
    path = output / "fastchess.json"
    saved = json.loads(path.read_text())
    saved["opening"].update(format=0, plies=-1)
    saved["sprt"].update(enabled=True, elo0=0, elo1=5, alpha=0.05, beta=0.05)
    for engine in saved["engines"]:
        engine["limit"]["tc"].update(time=10000, increment=100)
    path.write_text(json.dumps(saved))
    stdout = output / "stdout.txt"
    addition = f"LLR: {llr:.2f} (0.0%) (-2.94, 2.94) [0.00, 5.00]\n"
    if decision:
        addition += f"SPRT ([0.00, 5.00]) completed - {decision} was accepted\n"
    stdout.write_text(stdout.read_text().replace("Finished match\n", addition + "Finished match\n"))


@pytest.mark.parametrize(
    ("decision", "llr", "status", "passed"),
    [
        (None, 0.0, "inconclusive", False),
        ("H0", -3.0, "failed", False),
        ("H1", 3.0, "passed", True),
        ("H1", 0.0, "failed", False),
    ],
)
def test_full_declared_sprt_cap_requires_actual_consistent_native_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str | None,
    llr: float,
    status: str,
    passed: bool,
) -> None:
    module = api()
    path, plan = sprt_plan(tmp_path, monkeypatch)
    source = tmp_path / "generated"
    source.mkdir()
    sprt_evidence(plan, source, decision, llr)
    install_simulated_capture(monkeypatch, source)
    report = module.run_match(path, tmp_path / "sprt-evidence", execute=True)
    assert report["status"] == status, report["errors"]
    assert report["passed"] is passed
    assert report["profile"]["games"] == 1000
    if decision == "H1" and llr == 0:
        assert report["errors"]
    else:
        assert report["complete"] and report["evidence_valid"]
        assert report["statistics"]["games"] == 1000
        assert report["sprt_decision"] == (decision or "inconclusive")


def test_fixed_node_parity_command_never_accidentally_adds_time_or_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, _ = parity_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    command = module.build_command(plan, tmp_path / "nodes-only")
    assert "nodes=800" in command
    assert not any(
        token.startswith(("tc=", "st=", "depth=", "plies="))
        for token in command[command.index("-each") :]
    )
    assert "alpha=0.05" in command and "beta=0.05" in command and "model=normalized" in command
    assert "elo0=-10" in command and "elo1=0" in command
    assert "args=--config= --show-hidden" in command
    assert command.count("option.FpuStrategyAtRoot=same") == 2
    assert command.count("option.FpuValueAtRoot=1") == 2


def parity_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    path, document = controlled_config(tmp_path, monkeypatch, "parity-sprt")
    document.update(max_games=2000, nodes=800)
    common = {
        "CPuct": "1.745",
        "CPuctBase": "38739",
        "CPuctFactor": "3.894",
        "FpuValue": "0.33",
        "FpuStrategyAtRoot": "same",
        "FpuValueAtRoot": "1",
        "PolicyTemperature": "1.359",
        "Threads": "4",
        "MinibatchSize": "64",
        "NNCacheSize": "0",
        "SmartPruningFactor": "0",
    }
    for engine in document["engines"]:
        engine["options"] = {**engine["options"], **common}
    document["engines"][0]["options"]["Deterministic"] = "false"
    reference = document["engines"][1]
    reference.update(kind="lc0", model=reference["source_net"], args=["--config=", "--show-hidden"])
    reference["options"].update(
        Backend="cuda-fp16",
        Threads="0",
        MinibatchSize="0",
        WeightsFile=reference["source_net"],
        SmartPruningFactor="0",
        FpuStrategy="reduction",
        FpuStrategyAtRoot="same",
        FpuValueAtRoot="1",
        HistoryFill="fen_only",
        MovesLeftMaxEffect="0.0345",
        MovesLeftThreshold="0.8",
        MovesLeftSlope="0.0027",
        MovesLeftConstantFactor="0",
        MovesLeftScaledFactor="1.6521",
        MovesLeftQuadraticFactor="-0.6521",
    )
    monkeypatch.setattr(
        api(),
        "LC0_RELEASE_SHA256",
        hashlib.sha256(Path(reference["executable"]).read_bytes()).hexdigest(),
        raising=False,
    )
    path.write_text(json.dumps(document))
    return path, document


def test_parity_plan_accepts_semantically_exact_numeric_root_fpu_and_mlh_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = parity_document(tmp_path, monkeypatch)
    for engine in document["engines"]:
        engine["options"]["FpuValue"] = "0.330"
        engine["options"]["FpuValueAtRoot"] = "1.0"
    path.write_text(json.dumps(document))
    assert api().load_plan(path).selected.nodes == 800


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("FpuStrategyAtRoot", "absolute"),
        ("FpuValueAtRoot", "0"),
        ("HistoryFill", "always"),
        ("MovesLeftMaxEffect", "0"),
        ("MovesLeftQuadraticFactor", "0"),
    ],
)
def test_parity_plan_refuses_silent_root_fpu_history_or_mlh_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    path, document = parity_document(tmp_path, monkeypatch)
    document["engines"][1]["options"][key] = value
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError):
        api().load_plan(path)


def test_cli_requires_isolated_python_before_reading_any_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = api()
    monkeypatch.setattr(module, "_isolated", lambda: False, raising=False)
    assert (
        module.main(
            ["plan", "--config", str(tmp_path / "absent.json"), "--output", str(tmp_path / "out")]
        )
        == 1
    )
    assert "python -I" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_recognized_live_protocol_failure_terminates_process_before_its_watchdog(
    tmp_path: Path,
) -> None:
    module = api()
    guard = module.Progress(1000, ("A", "B"))
    result = module.capture(
        child("import time;print('Finished game malformed',flush=True);time.sleep(60)"),
        tmp_path,
        idle_seconds=2,
        total_seconds=3,
        max_bytes=1024 * 1024,
        on_stdout=guard,
    )
    assert "Malformed" in result["error"]
    assert result["wall_seconds"] < 1.5
    assert result["pumps_joined"] and result["terminated_by_runner"]


def test_equivalent_absolute_model_path_spelling_is_validated_as_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = controlled_config(tmp_path, monkeypatch)
    for engine in document["engines"]:
        engine["options"]["WeightsFile"] = Path(engine["model"]).as_posix()
    path.write_text(json.dumps(document))
    assert api().load_plan(path).selected.games == 1000


def test_gate_cannot_silently_use_a_different_same_net_instead_of_pinned_t1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, _ = controlled_config(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "T1_SHA256", "0" * 64, raising=False)
    with pytest.raises(module.MatchError, match="T1|source"):
        module.load_plan(path)


@pytest.mark.parametrize("engine_index", [0, 1])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("FpuStrategyAtRoot", "absolute"),
        ("FpuStrategyAtRoot", None),
        ("FpuValueAtRoot", "0"),
        ("FpuValueAtRoot", None),
    ],
)
def test_parity_requires_explicit_same_root_strategy_on_each_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine_index: int,
    key: str,
    value: str | None,
) -> None:
    path, document = parity_document(tmp_path, monkeypatch)
    module = api()
    assert module.load_plan(path).selected.nodes == 800
    options = document["engines"][engine_index]["options"]
    if value is None:
        options.pop(key)
    else:
        options[key] = value
    path.write_text(json.dumps(document))
    with pytest.raises(module.MatchError, match="root"):
        module.load_plan(path)


def test_old_absolute_root_parity_plan_is_not_a_current_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = parity_document(tmp_path, monkeypatch)
    for engine in document["engines"]:
        engine["options"]["FpuStrategyAtRoot"] = "absolute"
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError, match="root"):
        api().load_plan(path)


def test_lc0_parity_plan_requires_explicit_hidden_option_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path, document = parity_document(tmp_path, monkeypatch)
    document["engines"][1]["args"] = ["--config=", "--show-hidden"]
    path.write_text(json.dumps(document))
    plan = module.load_plan(path)
    assert plan.engines[1].args == ("--config=", "--show-hidden")
    command = module.build_command(plan, tmp_path / "explicit visibility")
    assert "args=--config= --show-hidden" in command
    assert command.count("option.FpuStrategyAtRoot=same") == 2
    assert command.count("option.FpuValueAtRoot=1") == 2


@pytest.mark.parametrize(
    "args",
    [
        ["--config="],
        ["--show-hidden"],
        ["--show-hidden", "--config="],
        ["--config=", "--show-hidden", "--show-hidden"],
        ["--config=", "--show-hidden", "--backend=auto"],
    ],
)
def test_lc0_plan_refuses_missing_reordered_duplicate_or_extra_native_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    path, document = parity_document(tmp_path, monkeypatch)
    document["engines"][1]["args"] = args
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError):
        api().load_plan(path)


def package_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, Any]]:
    path, document = controlled_config(tmp_path, monkeypatch, "package-smoke")
    import onnx

    model = Path(document["engines"][0]["model"])
    network = onnx.load_model(model)
    for entry in network.metadata_props:
        if entry.key == "mouse.precision":
            entry.value = "fp16"
    onnx.save_model(network, model)
    document["sha256"][str(model)] = hashlib.sha256(model.read_bytes()).hexdigest()
    book = Path(document["book"])
    book.write_text(full_game(1, "?", "?"))
    digest = hashlib.sha256(book.read_bytes()).hexdigest()
    document["sha256"][str(book)] = digest
    monkeypatch.setitem(api().BOOKS, book.name, (digest, "pgn", 1))
    for engine in document["engines"]:
        engine["options"] = {
            **engine["options"],
            "Backend": "cuda",
            "Precision": "fp16",
            "NNCacheSize": "0",
            "Device": "0",
            "GpuDllDirectories": str(model.parent),
            "CacheDirectory": str(tmp_path / "cache"),
            "RamLimitMb": "512",
        }
    path.write_text(json.dumps(document))
    return path, document


def test_package_smoke_is_exactly_ten_paired_fast_games_with_diagnostic_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    selected = module.profile("package-smoke")
    assert (selected.games, selected.tc, selected.sprt, selected.book) == (
        10,
        "1+0.01",
        None,
        "8moves_v3.pgn",
    )
    path, _ = package_document(tmp_path, monkeypatch)
    report = module.run_match(path, tmp_path / "package-plan", execute=False)
    assert report["profile"]["name"] == "package-smoke"
    assert report["profile"]["games"] == 10
    assert report["acceptance_scope"] == "package-smoke-only"
    assert report["roadmap_gate_claim"] is False
    assert report["full_release_gate"] is False
    assert report["native_execution"] is False
    command = report["command"]
    assert command[command.index("-rounds") + 1] == "5"
    assert "tc=1+0.01" in command and "-repeat" in command and "-strict" in command
    assert command.count("option.Threads=4") == 2
    assert command.count("option.MinibatchSize=64") == 2
    assert command.count("option.NNCacheSize=0") == 2


@pytest.mark.parametrize("kwargs", [{"max_games": 10}, {"max_games": 2}, {"nodes": 800}])
def test_package_smoke_cannot_override_its_fixed_count_or_time_control(
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(api().MatchError):
        api().profile("package-smoke", **kwargs)


@pytest.mark.parametrize(
    "key,value", [("Threads", "1"), ("MinibatchSize", "1"), ("NNCacheSize", "1")]
)
def test_package_smoke_requires_identical_normal_gpu_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    path, document = package_document(tmp_path, monkeypatch)
    for engine in document["engines"]:
        engine["options"][key] = value
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError):
        api().load_plan(path)


def test_package_smoke_requires_the_same_executable_and_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = package_document(tmp_path, monkeypatch)
    document["engines"][1]["options"]["RamLimitMb"] = "1024"
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError):
        api().load_plan(path)


def test_package_smoke_audits_all_ten_games_and_cannot_satisfy_the_selfplay_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    module = api()
    path, _ = package_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    output = tmp_path / "ten-retained-games"
    output.mkdir()
    retained_evidence(plan, output)
    result = module.audit_results(plan, output, captured_command=module.build_command(plan, output))
    assert result["complete"] and result["safety_pass"]
    assert result["statistics"]["games"] == 10
    assert result["statistics"]["pairs"] == 5
    with pytest.raises(module.MatchError):
        module.audit_results(
            replace(plan, selected=module.profile("selfplay-fast")),
            output,
            captured_command=module.build_command(plan, output),
        )


@pytest.mark.parametrize("failure", ["missing_game", "time_loss", "illegal", "disconnect"])
def test_package_smoke_reuses_strict_complete_pgn_and_failure_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    module = api()
    path, _ = package_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    output = tmp_path / "failed-smoke"
    output.mkdir()
    retained_evidence(plan, output, count=8 if failure == "missing_game" else 10)
    pgn = output / "games.pgn"
    if failure == "time_loss":
        pgn.write_text(pgn.read_text().replace("normal", "time forfeit", 1))
    elif failure in ("illegal", "disconnect"):
        log = output / "protocol.log"
        log.write_text(log.read_text() + f"[ERR] [time] <1> A ---> error {failure}\n")
    with pytest.raises(module.MatchError):
        module.audit_results(plan, output, captured_command=module.build_command(plan, output))


def test_package_smoke_refuses_an_otherwise_valid_cpu_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = controlled_config(tmp_path, monkeypatch, "package-smoke")
    with pytest.raises(api().MatchError, match="GPU"):
        api().load_plan(path)


def test_package_smoke_refuses_different_executable_paths_even_when_bytes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, document = package_document(tmp_path, monkeypatch)
    original = Path(document["engines"][0]["executable"])
    other = original.with_name("other mouse.exe")
    other.write_bytes(original.read_bytes())
    document["engines"][1]["executable"] = str(other)
    document["sha256"][str(other)] = hashlib.sha256(other.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(api().MatchError, match="identical"):
        api().load_plan(path)


@pytest.mark.parametrize("name", ["selfplay-fast", "package-smoke", "parity-sprt"])
def test_native_engine_command_omits_empty_args_and_keeps_explicit_lc0_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = api()
    if name == "package-smoke":
        path, _ = package_document(tmp_path, monkeypatch)
    elif name == "parity-sprt":
        path, _ = parity_document(tmp_path, monkeypatch)
    else:
        path, _ = controlled_config(tmp_path, monkeypatch)
    command = module.build_command(module.load_plan(path), tmp_path / "native command")
    assert "args=" not in command
    assert [token for token in command if token.startswith("args=")] == (
        ["args=--config= --show-hidden"] if name == "parity-sprt" else []
    )


@pytest.mark.parametrize("serialized", [True, None])
def test_strict_runtime_flag_uses_captured_command_even_when_serializer_omits_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serialized: bool | None
) -> None:
    module = api()
    path, _ = package_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    raw = tmp_path / "serialized-runtime-flags"
    raw.mkdir()
    retained_evidence(plan, raw)
    saved = json.loads((raw / "fastchess.json").read_text())
    if serialized is None:
        del saved["strict"]
        (raw / "fastchess.json").write_text(json.dumps(saved))
    command = module.build_command(plan, raw)
    result = module.audit_results(plan, raw, captured_command=command)
    assert result["complete"] and result["statistics"]["games"] == 10
    assert result["native_invocation"]["command"] == command
    assert result["native_invocation"]["strict_occurrences"] == 1
    assert result["native_invocation"]["fastchess_sha256"] == module.FASTCHESS_SHA256


@pytest.mark.parametrize("serialized", [True, None])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_capture",
        "missing_strict",
        "duplicate_strict",
        "startup",
        "executable",
        "binary_hash",
    ],
)
def test_actual_invocation_is_required_and_cannot_be_replaced_by_saved_strict_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, serialized: bool | None, mutation: str
) -> None:
    module = api()
    path, _ = package_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    raw = tmp_path / "bad-native-invocation"
    raw.mkdir()
    retained_evidence(plan, raw)
    saved = json.loads((raw / "fastchess.json").read_text())
    if serialized is None:
        del saved["strict"]
        (raw / "fastchess.json").write_text(json.dumps(saved))
    command = module.build_command(plan, raw)
    if mutation == "missing_strict":
        command.remove("-strict")
    elif mutation == "duplicate_strict":
        command.append("-strict")
    elif mutation == "startup":
        command[command.index("-startup-ms") + 1] = "1000"
    elif mutation == "executable":
        command[0] = str(plan.engines[0].executable)
    elif mutation == "binary_hash":
        plan.fastchess.write_bytes(b"replacement after frozen plan")
    with pytest.raises(module.MatchError):
        module.audit_results(
            plan, raw, captured_command=None if mutation == "missing_capture" else command
        )


@pytest.mark.parametrize("strict", [False, 1, "true"])
def test_serialized_strict_false_or_wrong_type_still_refuses_a_correct_captured_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strict: object
) -> None:
    module = api()
    path, _ = package_document(tmp_path, monkeypatch)
    plan = module.load_plan(path)
    raw = tmp_path / "contradictory-native-json"
    raw.mkdir()
    retained_evidence(plan, raw)
    saved = json.loads((raw / "fastchess.json").read_text())
    saved["strict"] = strict
    (raw / "fastchess.json").write_text(json.dumps(saved))
    with pytest.raises(module.MatchError):
        module.audit_results(plan, raw, captured_command=module.build_command(plan, raw))


@pytest.mark.parametrize(
    "name", ["tm-safety-fast", "tm-safety-stc", "tm-safety-ltc", "uci-stress-fast"]
)
def test_v1_stress_concurrency_preserves_the_complete_planned_game_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch, name)
    document["concurrency"] = 4
    path.write_text(json.dumps(document))
    plan = module.load_plan(path)
    assert plan.selected.concurrency == 4
    assert plan.selected.games == 2000
    assert plan.selected.sprt is None


@pytest.mark.parametrize("value", [0, 2, 4, True])
def test_v1_tm_gainer_stays_at_one_game_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object
) -> None:
    module = api()
    path, document = controlled_config(tmp_path, monkeypatch, "tm-sprt")
    document["max_games"] = 20000
    document["concurrency"] = value
    path.write_text(json.dumps(document))
    with pytest.raises(module.MatchError):
        module.load_plan(path)


@pytest.mark.parametrize(
    ("name", "games", "tc"),
    [
        ("v1-fast-track-fast", 1000, "1+0.01"),
        ("v1-fast-track-stc", 200, "10+0.1"),
        ("v1-fast-track-ltc", 40, "60+0.6"),
    ],
)
def test_user_fast_track_keeps_exact_counts_and_serial_sprt_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, games: int, tc: str
) -> None:
    module = api()
    selected = module.profile(name)
    assert (selected.games, selected.tc, selected.sprt) == (games, tc, None)
    with pytest.raises(module.MatchError):
        module.profile(name, max_games=2)
    with pytest.raises(module.MatchError):
        module.profile(name, nodes=800)
    path, document = controlled_config(tmp_path, monkeypatch, name)
    document["concurrency"] = 4
    path.write_text(json.dumps(document))
    plan = module.load_plan(path)
    assert plan.selected.concurrency == 4
    assert plan.selected.games == games
    assert plan.selected.tc == tc
    document["concurrency"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(module.MatchError):
        module.load_plan(path)
    document["concurrency"] = 4
    document["engines"][1]["options"] = {**document["engines"][1]["options"], "Threads": "1"}
    path.write_text(json.dumps(document))
    with pytest.raises(module.MatchError, match="Selfplay"):
        module.load_plan(path)
