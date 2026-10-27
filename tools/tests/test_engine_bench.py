# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-corpus common-boundary evidence, with controlled UCI children only."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pytest

CORPUS = Path(__file__).resolve().parents[2] / "tests/data/bench/positions-v1.tsv"


def api() -> Any:
    return importlib.import_module("mouse_tools.engine_bench")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    script = tmp_path / "fake child.py"
    script.write_text("# controlled child\n", encoding="utf-8")
    source = tmp_path / "source.pb.gz"
    source.write_bytes(b"frozen source")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"frozen converted model")
    runtime = tmp_path / "runtime.dll"
    runtime.write_bytes(b"frozen runtime")
    source_file = {"path": str(source), "sha256": sha(source)}
    engines: dict[str, Any] = {}
    for role in ("mouse", "lc0"):
        own_model = model if role == "mouse" else source
        argv = [sys.executable, "-I", str(script), "--role=" + role]
        argv += (
            ["--weights", str(own_model), "--backend", "cuda", "--precision", "fp16"]
            if role == "mouse"
            else ["--weights=" + str(own_model), "--backend=cuda-fp16", "--config="]
        )
        engines[role] = {
            "argv": argv,
            "cwd": str(tmp_path),
            "executable_sha256": sha(Path(sys.executable)),
            "name_prefix": "Mouse " if role == "mouse" else "Lc0 ",
            "backend": "cuda" if role == "mouse" else "cuda-fp16",
            "precision": "fp16",
            "model": {"path": str(own_model), "sha256": sha(own_model)},
            "source_sha256": sha(source),
            "artifacts": [
                {"path": str(script), "sha256": sha(script), "role": "source"},
                {"path": str(runtime), "sha256": sha(runtime), "role": "runtime"},
            ],
            "options": {
                "NNCacheSize": "0",
                "MultiPV": "1",
                "Ponder": "false",
                "UCI_Chess960": "false",
            },
        }
    data: dict[str, Any] = {
        "schema": "MouseEngineBench1",
        "nodes": 800,
        "repetitions": 3,
        "order": ["mouse", "lc0", "lc0", "mouse", "mouse", "lc0"],
        "corpus": str(CORPUS),
        "corpus_sha256": sha(CORPUS),
        "source_net": source_file,
        "timeouts": {"startup_seconds": 300, "search_seconds": 30, "quit_seconds": 10},
        "oracle": {
            "argv": [sys.executable, "-I", str(script), "--role=oracle"],
            "cwd": str(tmp_path),
            "executable_sha256": sha(Path(sys.executable)),
            "source_commit": "03e27488f3d21d8ff4dbf3065603afa21dbd0ef3",
        },
        "engines": engines,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(data), encoding="utf-8")
    return config, data


class FakeUci:
    order: list[str] = []
    instances: list[FakeUci] = []
    bad = ""
    active = 0

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.role = next(x.removeprefix("--role=") for x in command if x.startswith("--role="))
        self.timeout = kwargs["timeout"]
        self.command = command
        self.queue: list[str] = []
        self.sent: list[str] = []
        self.received: list[str] = []
        self.exited = False
        self.process = object()
        self.searches = 0
        self.__class__.instances.append(self)

    def __enter__(self) -> FakeUci:
        assert self.__class__.active == 0
        self.__class__.active += 1
        self.__class__.order.append(self.role)
        return self

    def __exit__(self, *args: Any) -> None:
        self.__class__.active -= 1

    def send(self, text: str) -> None:
        self.sent.append(text)
        if text == "uci":
            self.queue += [
                "id name "
                + (
                    "Stockfish 17.1"
                    if self.role == "oracle"
                    else "Mouse fake"
                    if self.role == "mouse"
                    else "Lc0 fake"
                )
            ]
            self.queue += [
                "option name " + k + " type string default x"
                for k in ("NNCacheSize", "MultiPV", "Ponder", "UCI_Chess960")
            ]
            self.queue += ["uciok"]
        elif text == "isready":
            self.queue += ["readyok"]
        elif text == "go perft 1":
            self.queue += ["a2a3: 1", "Nodes searched: 1"]
        elif text == "go nodes 800":
            self.searches += 1
            nodes = 800 if self.role == "mouse" else 1025
            if self.bad == "nodes" and self.role == "mouse":
                nodes = 799
            if self.bad == "reference_under" and self.role == "lc0":
                nodes = 799
            self.queue += [f"info depth 1 nodes {nodes} time 999999 nps 999999999 pv a2a3"]
            if self.bad == "no_nodes":
                self.queue[-1] = "info depth 1 time 999999 nps 999999999 pv a2a3"
            self.queue += ["bestmove " + ("h1h8" if self.bad == "illegal" else "a2a3")]
        elif text == "quit":
            self.exited = True

    def line(self, **kwargs: Any) -> str:
        if not self.queue:
            raise RuntimeError("fake missing response")
        result = self.queue.pop(0)
        self.received.append(result)
        return result

    def close_stdin(self) -> None:
        pass

    def wait_exit(self) -> int:
        assert self.exited
        return 7 if self.bad == "late_exit" else 0

    def evidence(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "commands": self.sent,
            "exit_code": 7 if self.bad == "late_exit" else 0,
            "pumps_joined": True,
            "terminated_by_runner": not self.exited,
            "capture_complete": True,
            "stdout_text": "\n".join(self.received + self.queue),
            "stderr_text": "",
        }


def install(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = api()
    FakeUci.order, FakeUci.instances, FakeUci.bad, FakeUci.active = [], [], "", 0
    monkeypatch.setattr(module, "UciProcess", FakeUci)
    tick = itertools.count(1_000_000, 1_000_000)
    monkeypatch.setattr(module.time, "perf_counter_ns", lambda: next(tick))
    return module


def test_exact_schedule_full_history_warmups_common_clock_and_actual_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    config, _ = fixture(tmp_path)
    output = tmp_path / "report.json"
    report = module.run_benchmark(config, output)
    assert report["evidence_valid"] is True
    assert report["planned_searches"] == report["completed_searches"] == 192
    assert report["measured_searches"] == 96
    assert FakeUci.order == ["oracle", "mouse", "lc0", "lc0", "mouse", "mouse", "lc0"]
    assert all(x.searches == 32 for x in FakeUci.instances[1:])
    assert len(report["runs"]) == 6
    assert all(len(x["warmup"]) == len(x["measured"]) == 16 for x in report["runs"])
    assert report["ratio"] == pytest.approx(800 / 1025)
    assert report["engine_90_percent_met"] is False
    assert "backend_90_percent_met" not in report
    assert all(row["elapsed_ns"] == 2_000_000 for run in report["runs"] for row in run["measured"])
    for process in FakeUci.instances[1:]:
        positions = [x for x in process.sent if x.startswith("position ")]
        assert len(positions) == 32
        assert positions[1].endswith(" moves d2d4 d7d5 c2c4 e7e6 b1c3 g8f6")
        assert positions[:16] == positions[16:]
        assert process.sent[-1] == "quit"
    assert json.loads(output.read_text())["evidence_valid"] is True


@pytest.mark.parametrize("bad", ["illegal", "nodes", "reference_under", "no_nodes", "late_exit"])
def test_invalid_or_late_evidence_has_no_ratio_or_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    module = install(monkeypatch)
    FakeUci.bad = bad
    config, _ = fixture(tmp_path)
    output = tmp_path / "failed.json"
    with pytest.raises(module.EngineBenchError):
        module.run_benchmark(config, output)
    report = json.loads(output.read_text())
    assert report["evidence_valid"] is False
    assert "ratio" not in report
    assert "engine_90_percent_met" not in report
    assert FakeUci.active == 0


@pytest.mark.parametrize(
    "field,value", [("nodes", 801), ("repetitions", 2), ("order", ["mouse"] * 6), ("extra", 1)]
)
def test_fixed_method_cannot_be_reduced_or_reconfigured(
    tmp_path: Path, field: str, value: Any
) -> None:
    config, data = fixture(tmp_path)
    data[field] = value
    config.write_text(json.dumps(data))
    with pytest.raises(api().EngineBenchError):
        api().load_config(config)


def test_duplicate_keys_and_huge_nonfinite_deadlines_rejected(tmp_path: Path) -> None:
    config, data = fixture(tmp_path)
    config.write_text('{"schema":"MouseEngineBench1","schema":"MouseEngineBench1"}')
    with pytest.raises(api().EngineBenchError):
        api().load_config(config)
    for value in (True, float("nan"), 10**1000, 0, -1):
        data["timeouts"]["search_seconds"] = value
        config.write_text(json.dumps(data))
        with pytest.raises(api().EngineBenchError):
            api().load_config(config)


def test_hash_mutation_and_report_collision_fail_before_any_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    config, data = fixture(tmp_path)
    Path(data["source_net"]["path"]).write_bytes(b"changed")
    with pytest.raises(module.EngineBenchError, match="hash"):
        module.run_benchmark(config, tmp_path / "failed.json")
    assert FakeUci.order == []
    output = tmp_path / "existing.json"
    output.write_text("original")
    with pytest.raises(module.EngineBenchError):
        module.run_benchmark(config, output)
    assert output.read_text() == "original"


def test_cli_requires_isolated_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = api()
    monkeypatch.setattr(module, "_isolated", lambda: False)
    assert (
        module.main(["--config", str(tmp_path / "x.json"), "--output", str(tmp_path / "y.json")])
        == 1
    )


def test_declared_precision_must_be_bound_to_native_argv(tmp_path: Path) -> None:
    config, data = fixture(tmp_path)
    argv = data["engines"]["mouse"]["argv"]
    del argv[argv.index("--precision") :]
    config.write_text(json.dumps(data))
    with pytest.raises(api().EngineBenchError, match="precision"):
        api().load_config(config)


def test_oracle_must_also_quit_and_join_all_pumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    original = FakeUci.evidence

    def evidence(self: FakeUci) -> dict[str, Any]:
        result = original(self)
        if self.role == "oracle":
            result["pumps_joined"] = False
        return result

    monkeypatch.setattr(FakeUci, "evidence", evidence)
    config, _ = fixture(tmp_path)
    with pytest.raises(module.EngineBenchError, match="join"):
        module.run_benchmark(config, tmp_path / "failed.json")
    assert FakeUci.order == ["oracle"]


def test_duplicate_post_bestmove_cannot_hide_after_last_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if text == "go nodes 800":
            self.queue.append("bestmove a2a3")

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    with pytest.raises(module.EngineBenchError):
        module.run_benchmark(config, tmp_path / "failed.json")


def test_final_artifact_mutation_removes_all_ratio_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    config, data = fixture(tmp_path)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if text == "quit" and len(FakeUci.instances) == 7:
            Path(data["source_net"]["path"]).write_bytes(b"late changed")

    monkeypatch.setattr(FakeUci, "send", send)
    output = tmp_path / "failed.json"
    with pytest.raises(module.EngineBenchError, match="hash"):
        module.run_benchmark(config, output)
    report = json.loads(output.read_text())
    assert report["evidence_valid"] is False
    assert "ratio" not in report
    assert len(report["runs"]) == 6


def test_real_controlled_child_full_schedule_natural_quit_and_join(tmp_path: Path) -> None:
    config, data = fixture(tmp_path)
    script = Path(data["engines"]["mouse"]["argv"][2])
    script.write_text(
        """import sys
role=next(x.split('=',1)[1] for x in sys.argv if x.startswith('--role='))
for raw in sys.stdin:
 cmd=raw.strip()
 if cmd=='uci':
  names={'oracle':'Stockfish 17.1','mouse':'Mouse controlled','lc0':'Lc0 controlled'}
  print('id name '+names[role])
  for name in ['NNCacheSize','MultiPV','Ponder','UCI_Chess960']:
   print('option name '+name+' type string default x')
  print('uciok',flush=True)
 elif cmd=='isready': print('readyok',flush=True)
 elif cmd=='go perft 1': print('a2a3: 1\\nNodes searched: 1',flush=True)
 elif cmd=='go nodes 800': print('info nodes 800 nps 999999999 time 0\\nbestmove a2a3',flush=True)
 elif cmd=='quit': break
""",
        encoding="utf-8",
    )
    for engine in data["engines"].values():
        engine["artifacts"][0]["sha256"] = sha(script)
    data["timeouts"] = {"startup_seconds": 5, "search_seconds": 5, "quit_seconds": 5}
    config.write_text(json.dumps(data))
    report = api().run_benchmark(config, tmp_path / "real-child-report.json")
    assert report["evidence_valid"] is True
    for row in [report["oracle"], *report["runs"]]:
        assert row["process"]["exit_code"] == 0
        assert row["process"]["pumps_joined"] is True
        assert row["process"]["terminated_by_runner"] is False
        assert row["process"]["capture_complete"] is True


def test_nonmonotonic_measurement_clock_is_rejected_before_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    module = install(monkeypatch)
    monkeypatch.setattr(
        module.time,
        "get_clock_info",
        lambda name: SimpleNamespace(
            monotonic=False, adjustable=True, implementation="bad", resolution=0.001
        ),
    )
    config, _ = fixture(tmp_path)
    with pytest.raises(module.EngineBenchError, match="clock"):
        module.run_benchmark(config, tmp_path / "failed.json")
    assert FakeUci.order == []


def test_pinned_stockfish_banner_and_perft_blank_lines_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if self.role == "oracle" and text == "uci":
            self.queue.insert(0, "Stockfish 17.1 by the Stockfish developers (see AUTHORS file)")
        if self.role == "oracle" and text == "go perft 1":
            self.queue.insert(len(self.queue) - 1, "")
            self.queue.append("")

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    assert module.run_benchmark(config, tmp_path / "report.json")["evidence_valid"] is True


@pytest.mark.parametrize(
    "badline",
    [
        "info nodes 800 nodes 800",
        "info nodes 999999999999999999999999999",
        "info nodes -1",
        "info nodes 800 multipv 2",
        "info string error failed",
        "bestmove 0000",
    ],
)
def test_malformed_nodes_error_and_null_best_are_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, badline: str
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if text == "go nodes 800":
            self.queue.insert(0, badline)

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    with pytest.raises(module.EngineBenchError):
        module.run_benchmark(config, tmp_path / "failed.json")


@pytest.mark.parametrize("late", ["bestmove a2a3", "info string error late failure"])
def test_late_protocol_after_final_barrier_is_not_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, late: str
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if text == "quit" and self.role != "oracle":
            self.queue.append(late)

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    with pytest.raises(module.EngineBenchError, match="capture"):
        module.run_benchmark(config, tmp_path / "failed.json")


def test_actual_stockfish_author_separator_is_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if self.role == "oracle" and text == "uci":
            self.queue[1:1] = ["id author the Stockfish developers (see AUTHORS file)", ""]

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    report = module.run_benchmark(config, tmp_path / "report.json")
    assert report["evidence_valid"] is True
    assert report["completed_searches"] == 192


@pytest.mark.parametrize("role,blanks", [("oracle", 2), ("mouse", 1), ("lc0", 1)])
def test_stockfish_separator_does_not_allow_unknown_engine_or_duplicate_spacing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, blanks: int
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if self.role == role and text == "uci":
            self.queue[1:1] = ["id author the Stockfish developers (see AUTHORS file)"] + (
                [""] * blanks
            )

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    with pytest.raises(module.EngineBenchError, match="handshake"):
        module.run_benchmark(config, tmp_path / "failed.json")


STOCKFISH_PERFT_INFO = [
    "info string Available processors: 0-15",
    "info string Using 1 thread",
    "info string NNUE evaluation using nn-1c0000000000.nnue (133MiB, (22528, 3072, 15, 32, 1))",
    "info string NNUE evaluation using nn-37f18f62d772.nnue (6MiB, (22528, 128, 15, 32, 1))",
]


def test_actual_stockfish_pre_edge_diagnostics_are_bounded_and_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if self.role == "oracle" and text == "go perft 1":
            self.queue[0:0] = STOCKFISH_PERFT_INFO
            self.queue.insert(len(self.queue) - 1, "")
            self.queue.append("")
        if self.role == "oracle" and text == "setoption name Threads value 1":
            self.queue.append("info string Using 1 thread")

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    report = module.run_benchmark(config, tmp_path / "report.json")
    assert report["evidence_valid"] is True
    assert report["completed_searches"] == 192
    assert len(report["oracle"]["cases"]) == 16
    assert all(row["info"] == STOCKFISH_PERFT_INFO for row in report["oracle"]["cases"])
    assert sum(x.startswith("go perft ") for x in FakeUci.instances[0].sent) == 16


@pytest.mark.parametrize("phase", ["ready", "search", "late_capture"])
def test_actual_stockfish_uppercase_colon_errors_are_never_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        error = "info string ERROR: The network file was not loaded successfully."
        if self.role == "mouse" and (
            (phase == "ready" and text == "isready")
            or (phase == "search" and text == "go nodes 800")
        ):
            self.queue.insert(0, error)
        if self.role == "mouse" and phase == "late_capture" and text == "quit":
            self.queue.append(error)

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    output = tmp_path / "failed.json"
    with pytest.raises(module.EngineBenchError, match="error"):
        module.run_benchmark(config, output)
    report = json.loads(output.read_text())
    assert report["evidence_valid"] is False
    assert "ratio" not in report


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        (["unexpected oracle output"], []),
        (["info string unexpected oracle diagnostic"], []),
        ([STOCKFISH_PERFT_INFO[0]] * 2, []),
        ([], [STOCKFISH_PERFT_INFO[1]]),
        (["info string ERROR: network verification failed"], []),
        (["info string Using 2 threads"], []),
        (
            [
                *STOCKFISH_PERFT_INFO,
                "info string NNUE evaluation using third.nnue (1MiB, (1, 1, 1, 1, 1))",
            ],
            [],
        ),
        ([], ["a2a3: 1"]),
        ([], ["Nodes searched: 1"]),
        ([], ["h2h3: 2"]),
    ],
)
def test_perft_diagnostics_do_not_relax_edges_or_accept_unknown_duplicate_or_late_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: list[str], suffix: list[str]
) -> None:
    module = install(monkeypatch)
    original = FakeUci.send

    def send(self: FakeUci, text: str) -> None:
        original(self, text)
        if self.role == "oracle" and text == "go perft 1":
            self.queue[0:0] = prefix
            self.queue[len(self.queue) - 1 : len(self.queue) - 1] = suffix

    monkeypatch.setattr(FakeUci, "send", send)
    config, _ = fixture(tmp_path)
    output = tmp_path / "failed.json"
    with pytest.raises(module.EngineBenchError):
        module.run_benchmark(config, output)
    report = json.loads(output.read_text())
    assert report["evidence_valid"] is False
    assert not report["runs"]
    assert "ratio" not in report
