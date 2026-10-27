# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact mate proofs must handle adversarial defense and fail closed."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = "k7/8/1QK5/8/8/8/8/8 w - - 0 1"


def module() -> ModuleType:
    assert importlib.util.find_spec("mouse_tools.mate_fixtures") is not None, "Mate prover missing"
    return importlib.import_module("mouse_tools.mate_fixtures")


def test_mate_distance_requires_every_defense_and_rejects_stalemate() -> None:
    m = module()
    # Terminal losing side is in check; the other terminal is stalemate.
    graph = {
        ROOT: m.State(ROOT, "w", False, ("a1a2", "a1b1")),
        "fast": m.State("fast", "b", True, ()),
        "draw": m.State("draw", "b", False, ()),
    }

    class Oracle:
        def state(self, fen: str, move: str | None = None) -> object:
            return graph[{"a1a2": "fast", "a1b1": "draw"}[move]] if move else graph[fen]

    proof = m.Prover(Oracle(), maximum_states=100).prove(ROOT, 3)
    assert proof.mate == 1
    assert proof.winning_moves == ("a1a2",)
    assert proof.pv == ("a1a2",)


def test_defender_escape_refutes_apparent_mate_and_bound_is_not_success() -> None:
    m = module()
    graph = {
        ROOT: m.State(ROOT, "w", False, ("a1a2",)),
        "reply": m.State("reply", "b", True, ("h8g8", "h8h7")),
        "win": m.State("win", "w", False, ("a2a3",)),
        "escape": m.State("escape", "w", False, ()),
        "mate": m.State("mate", "b", True, ()),
    }
    children = {
        (ROOT, "a1a2"): "reply",
        ("reply", "h8g8"): "win",
        ("reply", "h8h7"): "escape",
        ("win", "a2a3"): "mate",
    }

    class Oracle:
        def state(self, fen: str, move: str | None = None) -> object:
            return graph[children[(fen, move)]] if move else graph[fen]

    assert m.Prover(Oracle(), maximum_states=100).prove(ROOT, 3).mate is None
    with pytest.raises(m.MateError, match="state budget"):
        m.Prover(Oracle(), maximum_states=1).prove(ROOT, 3)


@pytest.mark.parametrize(
    "fen",
    [
        "k7/8/8/8/8/8/Q7/7K w - - 0 1",  # Prior side is already checked.
        "k7/1K6/8/8/8/8/8/Q7 w - - 0 1",
        "k7/8/8/8/8/8/8/Q6K w K - 0 1",
        "k7/8/8/8/8/8/P7/Q6K w - - 0 1",
    ],
)
def test_candidates_reject_illegal_or_unsupported_state(fen: str) -> None:
    m = module()
    with pytest.raises(m.MateError):
        m.validate_position(fen)


def test_state_parser_requires_checkers_total_and_exact_divide() -> None:
    m = module()
    fen = "k7/8/1QK5/8/8/8/8/8 b - - 0 1"
    assert m.parse_state([f"Fen: {fen}", "Checkers: b6", "Nodes searched: 0"]).check
    for lines in (
        [f"Fen: {fen}", "Nodes searched: 0"],
        [f"Fen: {fen}", "Checkers: b6", "h8g8: 2", "Nodes searched: 2"],
        [f"Fen: {fen}", "Checkers: b6", "Nodes searched: 1"],
    ):
        with pytest.raises(m.MateError):
            m.parse_state(lines)


def test_timeout_and_malformed_oracle_leave_destination_unchanged(tmp_path: Path) -> None:
    m = module()
    destination = tmp_path / "corpus.json"
    destination.write_text("existing good corpus\n", encoding="utf-8")
    for body in ("import time; time.sleep(5)", "print('wrong', flush=True)"):
        script = tmp_path / "oracle.py"
        script.write_text(body, encoding="utf-8")
        with (
            pytest.raises(m.MateError),
            m.UciTransport((sys.executable, "-I", str(script)), timeout=0.1) as transport,
        ):
            m.write_corpus(destination, lambda: transport.exchange("uci", "uciok"))
        assert destination.read_text(encoding="utf-8") == "existing good corpus\n"


def test_hash_mismatch_never_launches_downloaded_program(tmp_path: Path) -> None:
    m = module()
    binary = tmp_path / "stockfish.exe"
    binary.write_bytes(b"not stockfish")
    with pytest.raises(m.MateError, match="SHA-256"):
        m.MateOracle(binary)


def test_committed_corpus_is_complete_distinct_and_well_formed() -> None:
    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/search/mates-v1"
    corpus = json.loads((root / "mates.json").read_text(encoding="utf-8"))
    m.validate_corpus(corpus)
    assert len(corpus["cases"]) == 18
    for n in (1, 2, 3):
        selected = [case for case in corpus["cases"] if case["mate"] == n]
        assert len(selected) == 6
        assert {case["fen"].split()[1] for case in selected} == {"w", "b"}


def test_captured_last_major_piece_is_a_valid_reachable_draw_position() -> None:
    module().validate_position("k7/8/8/8/8/8/8/7K w - - 0 1")


def test_original_candidates_include_blocked_major_piece_rays() -> None:
    m = module()
    values = m.candidates()
    assert len(values) == len(set(values))
    assert any("N" in fen.split()[0] for fen in values)
    assert any("R" in fen.split()[0] and "N" not in fen.split()[0] for fen in values)


@pytest.mark.parametrize(
    "text",
    [
        "Fen: k7/8/1QK5/8/8/8/8/8 b - - 0 1\nCheckers: junk\nNodes searched: 0",
        "Fen: k7/8/1QK5/8/8/8/8/8 b - - 0 1\nCheckers:\nCheckers:\nNodes searched: 0",
    ],
)
def test_corrupt_checker_fields_are_not_accepted_as_checkmate(text: str) -> None:
    m = module()
    with pytest.raises(m.MateError):
        m.parse_state(text.splitlines())


def test_excessive_oracle_line_is_bounded(tmp_path: Path) -> None:
    m = module()
    script = tmp_path / "oracle.py"
    script.write_text("print('x'*5000, flush=True)", encoding="utf-8")
    with (
        pytest.raises(m.MateError, match="bounds|malformed"),
        m.UciTransport((sys.executable, "-I", str(script)), timeout=2) as transport,
    ):
        transport.exchange("uci", "uciok")


def test_incomplete_transcript_cannot_produce_a_proof(tmp_path: Path) -> None:
    import gzip

    m = module()
    path = tmp_path / "states.gz"
    path.write_bytes(gzip.compress(b"", mtime=0))
    with pytest.raises(m.MateError, match="empty|missing"):
        m.RecordedOracle(path, m.sha256(path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("winning_moves", [[]]),
        ("winning_moves", []),
        ("mate", True),
        ("refuted_horizons", []),
        ("pv", ["a1a2"]),
    ],
)
def test_corpus_rejects_malformed_or_missing_proof_data(field: str, value: object) -> None:
    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/search/mates-v1"
    corpus = json.loads((root / "mates.json").read_text(encoding="utf-8"))
    case = next(row for row in corpus["cases"] if row["mate"] == 3)
    case[field] = value
    with pytest.raises(m.MateError):
        m.validate_corpus(corpus)


def test_every_committed_proof_replays_all_winning_roots_and_shallower_refutations() -> None:
    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/search/mates-v1"
    report = m.replay_corpus(root)
    assert report["cases_verified"] == 18
    assert report["mate_horizons_verified"] == 36
    assert report["oracle_execution"] is False


def test_recorded_oracle_reparses_raw_response_and_rejects_corruption(tmp_path: Path) -> None:
    import gzip

    m = module()
    fen = "k7/1Q6/2K5/8/8/8/8/8 b - - 1 1"
    value = {
        "query_fen": fen,
        "move": None,
        "state": {"fen": fen, "side": "b", "check": True, "moves": []},
        "response": [f"Fen: {fen}", "Checkers: b7", "Nodes searched: 0"],
    }
    path = tmp_path / "record.gz"
    path.write_bytes(gzip.compress((json.dumps(value) + "\n").encode(), mtime=0))
    assert m.RecordedOracle(path, m.sha256(path)).state(fen).check
    value["response"] = [f"Fen: {fen}", "Checkers: b7", "Nodes searched: 1"]
    path.write_bytes(gzip.compress((json.dumps(value) + "\n").encode(), mtime=0))
    with pytest.raises(m.MateError):
        m.RecordedOracle(path, m.sha256(path))


def test_malformed_divide_is_not_silently_ignored_as_zero_legal_moves() -> None:
    m = module()
    lines = ["Fen: k7/1Q6/2K5/8/8/8/8/8 b - - 1 1", "Checkers: b7", "a8a9: 1", "Nodes searched: 0"]
    with pytest.raises(m.MateError):
        m.parse_state(lines)


def test_handshake_requires_exact_version_and_mandatory_options() -> None:
    m = module()
    valid = [
        "id name Stockfish 17.1",
        "option name Threads type spin",
        "option name Hash type spin",
        "option name UCI_Chess960 type check",
        "uciok",
    ]
    m.validate_greeting(valid)
    for value in (
        valid[1:],
        valid + [valid[0]],
        valid[:-1],
        [line.replace("17.1", "17") for line in valid],
        [line for line in valid if not line.startswith("option name Hash ")],
    ):
        with pytest.raises(m.MateError):
            m.validate_greeting(value)


def test_exhausted_wall_budget_preserves_existing_output(tmp_path: Path) -> None:
    import time

    m = module()

    class SlowOracle:
        def state(self, fen: str, move: str | None = None) -> object:
            time.sleep(0.02)
            return m.State(ROOT, "w", False, ("a1a2",))

    destination = tmp_path / "proof.json"
    destination.write_text("old good output", encoding="utf-8")
    prover = m.Prover(SlowOracle(), maximum_states=100, maximum_seconds=0.001)
    with pytest.raises(m.MateError, match="wall-time"):
        m.write_corpus(destination, lambda: prover.prove(ROOT))
    assert destination.read_text(encoding="utf-8") == "old good output"


def test_epd_and_json_cannot_silently_disagree(tmp_path: Path) -> None:
    import shutil

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/search/mates-v1"
    target = tmp_path / "copied"
    shutil.copytree(root, target)
    path = target / "mates.epd"
    path.write_text(path.read_text(encoding="utf-8").replace("dm 1;", "dm 3;", 1), encoding="utf-8")
    provenance_path = target / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["files"]["mates.epd"] = m.sha256(path)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(m.MateError, match="EPD"):
        m.replay_corpus(target)


def test_cli_hash_failure_is_concise_nonzero_and_preserves_report(tmp_path: Path) -> None:
    import subprocess

    m = module()
    binary = tmp_path / "wrong.exe"
    binary.write_bytes(b"not the pinned official executable")
    destination = tmp_path / "report.json"
    destination.write_text("old verified report", encoding="utf-8")
    root = Path(__file__).resolve().parents[2] / "tests/data/search/mates-v1"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(m.__file__),
            "--oracle",
            str(binary),
            "--verify",
            str(root),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert not result.stdout
    assert "SHA-256" in result.stderr
    assert "Traceback" not in result.stderr
    assert destination.read_text(encoding="utf-8") == "old verified report"


@pytest.mark.parametrize("artifact", ["binary", "mates.json", "oracle-states.jsonl.gz"])
def test_mutation_during_proof_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    import shutil

    m = module()
    root = Path(__file__).resolve().parents[2] / "tests/data/search/mates-v1"
    suite = tmp_path / "suite"
    shutil.copytree(root, suite)
    binary = tmp_path / "oracle.exe"
    binary.write_bytes(b"frozen A")
    destination = tmp_path / "report.json"
    destination.write_text("old verified report", encoding="utf-8")
    recorded = m.RecordedOracle(
        suite / "oracle-states.jsonl.gz", m.sha256(suite / "oracle-states.jsonl.gz")
    )
    target = binary if artifact == "binary" else suite / artifact

    class MutatingOracle:
        changed = False
        queries = 0

        def __enter__(self) -> MutatingOracle:
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def state(self, fen: str, move: str | None = None) -> object:
            if not self.changed:
                target.write_bytes(target.read_bytes() + b" ")
                self.changed = True
            return recorded.state(fen, move)

    monkeypatch.setattr(m, "MateOracle", lambda *_: MutatingOracle())
    with pytest.raises(m.MateError, match="changed"):
        m.write_corpus(destination, lambda: m.verify_corpus(binary, suite))
    assert destination.read_text(encoding="utf-8") == "old verified report"


def test_reachable_clock91_is_valid_while_root_reserves_the_full_proof_horizon() -> None:
    m = module()
    root = "k7/8/2K5/8/5Q2/8/8/8 w - - 90 1"
    child = "k7/2K5/8/8/5Q2/8/8/8 b - - 91 1"
    m.validate_position(root)
    m.validate_position(child)
    m.validate_root(root)
    assert m._play(root, "c6c7") == child


@pytest.mark.parametrize("clock", [91, 95, 99, 100])
def test_proof_roots_cannot_silently_cross_the_rule50_horizon(clock: int) -> None:
    m = module()
    fen = f"k7/8/2K5/8/5Q2/8/8/8 w - - {clock} 1"

    class NoOracle:
        def state(self, fen: str, move: str | None = None) -> object:
            raise AssertionError("Unsafe proof root must fail before querying an oracle")

    with pytest.raises(m.MateError, match="clock"):
        m.Prover(NoOracle()).prove(fen)
