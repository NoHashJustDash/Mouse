# SPDX-License-Identifier: GPL-3.0-or-later
"""SPSA tuning of Mouse UCI options with fastchess game pairs.

Each step draws a random +/-1 flip per parameter, plays ``pairs_per_step`` game
pairs (both colours of one opening each) between theta + c_k*flip and
theta - c_k*flip, and moves theta by (a_k / c_k) * (wins - losses) * flip. The gains
follow the fishtest schedule: c_k = c / (k + 1)^gamma and a_k = a / (A + k + 1)^alpha,
with k the number of game pairs already played, A = 0.1 * total pairs, c chosen so that
c_k ends at ``c_end`` and a so that a_k ends at ``r_end * c_end^2``.

    python -I -m mouse_tools.spsa run --config C:/absolute/tune.json --output C:/absolute/run

The output directory holds ``config.json`` (the frozen plan), ``state.json`` (theta
after the last complete step) and ``history.jsonl`` (one line per step), plus each
step's fastchess log and PGN. Running again with the same plan and directory resumes
after the last complete step. A step whose match fails, or in which an engine
disconnects or stalls, is not counted: it is played again from the start, with the same
flips and openings, up to ``--attempts`` times in all, and the run stops with an error
once those are used up. Every failed attempt is recorded in ``failures.jsonl`` and its
files are kept as ``step-NNNNN.failed-K.*``, including fastchess's engine traffic log,
which is deleted after a step that succeeds.

A parameter marked ``"integer": true`` is sent as a whole number, for spin options, while theta
keeps its fraction so small steps still add up. The plan's ``composed`` maps an option to a
template, ``{"TimeManager": "legacy(slowmover={slowmover})"}``, for numbers an option carries
inside one string: the parameters a template names go in it and are not sent on their own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALPHA = 0.602
GAMMA = 0.101
RESULT = re.compile(r"^Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+),")
FAILURE = re.compile(r"disconnect|stalled|crash|illegal move", re.IGNORECASE)
# fastchess prints a per-player "Timeouts: N" / "Crashed: N" block after any time loss.
CRASH_COUNT = re.compile(r"^\s*Crashed: (\d+)$")
STEP_FILES = (".log", ".pgn", ".engines.log")


@dataclass(frozen=True)
class Parameter:
    name: str
    start: float
    minimum: float
    maximum: float
    c_end: float
    r_end: float
    integer: bool = False


@dataclass(frozen=True)
class Plan:
    fastchess: Path
    engine: Path
    options: dict[str, str]
    parameters: tuple[Parameter, ...]
    pairs: int
    pairs_per_step: int
    tc: str
    book: Path
    seed: int
    extra_arguments: tuple[str, ...]
    composed: dict[str, str] = field(default_factory=dict)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _template_fields(template: str) -> list[str]:
    return [name for _, name, _, _ in string.Formatter().parse(template) if name is not None]


def _check_composed(composed: dict[str, str], names: list[str], options: dict[str, str]) -> None:
    used: set[str] = set()
    for option, template in composed.items():
        _require(option not in options, f"{option}: a composed option cannot be a fixed option")
        _require(option not in names, f"{option}: a composed option cannot be a tuned parameter")
        for name in _template_fields(template):
            _require(name in names, f"{option}: template names an unknown parameter {name}")
            _require(name not in used, f"{name} is named by more than one template")
            used.add(name)


def load_plan(text: str) -> Plan:
    raw = json.loads(text)
    _require(isinstance(raw, dict), "SPSA plan must be a JSON object")
    parameters = []
    for item in raw["parameters"]:
        parameter = Parameter(
            name=str(item["name"]),
            start=float(item["start"]),
            minimum=float(item["min"]),
            maximum=float(item["max"]),
            c_end=float(item["c_end"]),
            r_end=float(item["r_end"]),
            integer=bool(item.get("integer", False)),
        )
        _require(
            parameter.minimum <= parameter.start <= parameter.maximum,
            f"{parameter.name}: start must lie within min..max",
        )
        _require(
            parameter.c_end > 0 and parameter.r_end > 0, f"{parameter.name}: gains must be > 0"
        )
        parameters.append(parameter)
    names = [parameter.name for parameter in parameters]
    _require(bool(names) and len(names) == len(set(names)), "SPSA needs distinct parameters")
    options = {str(key): str(value) for key, value in raw.get("options", {}).items()}
    _require(not set(options) & set(names), "A tuned parameter cannot also be a fixed option")
    composed = {str(key): str(value) for key, value in raw.get("composed", {}).items()}
    _check_composed(composed, names, options)
    plan = Plan(
        fastchess=Path(raw["fastchess"]),
        engine=Path(raw["engine"]),
        options=options,
        parameters=tuple(parameters),
        pairs=int(raw["pairs"]),
        pairs_per_step=int(raw["pairs_per_step"]),
        tc=str(raw["tc"]),
        book=Path(raw["book"]),
        seed=int(raw["seed"]),
        extra_arguments=tuple(str(item) for item in raw.get("extra_arguments", [])),
        composed=composed,
    )
    _require(plan.pairs_per_step > 0, "pairs_per_step must be positive")
    _require(
        plan.pairs > 0 and plan.pairs % plan.pairs_per_step == 0,
        "pairs must be a positive multiple of pairs_per_step",
    )
    for path in (plan.fastchess, plan.engine, plan.book):
        _require(path.is_absolute(), f"{path} must be an absolute path")
    return plan


def gains(parameter: Parameter, pairs: int, played: int) -> tuple[float, float]:
    """Return (a_k, c_k) after ``played`` of ``pairs`` game pairs."""
    stability = 0.1 * pairs
    c = parameter.c_end * pairs**GAMMA
    a = parameter.r_end * parameter.c_end**2 * (stability + pairs) ** ALPHA
    return a / (stability + played + 1) ** ALPHA, c / (played + 1) ** GAMMA


def clamp(parameter: Parameter, value: float) -> float:
    return min(parameter.maximum, max(parameter.minimum, value))


def flips(plan: Plan, step: int) -> list[int]:
    generator = random.Random(f"{plan.seed}:{step}")
    return [generator.choice((-1, 1)) for _ in plan.parameters]


def option_text(value: float) -> str:
    return f"{value:.6g}"


def parameter_text(parameter: Parameter, value: float) -> str:
    return str(round(value)) if parameter.integer else option_text(value)


def command(
    plan: Plan, plus: dict[str, float], minus: dict[str, float], step: int, directory: Path
) -> list[str]:
    in_template = {
        name for template in plan.composed.values() for name in _template_fields(template)
    }

    def engine(name: str, values: dict[str, float]) -> list[str]:
        texts = {item.name: parameter_text(item, values[item.name]) for item in plan.parameters}
        settings = [f"option.{key}={value}" for key, value in plan.options.items()]
        settings += [
            f"option.{key}={text}" for key, text in texts.items() if key not in in_template
        ]
        settings += [
            f"option.{key}={template.format(**texts)}" for key, template in plan.composed.items()
        ]
        return ["-engine", f"name={name}", f"cmd={plan.engine}", *settings]

    return [
        str(plan.fastchess),
        *engine("plus", plus),
        *engine("minus", minus),
        "-each",
        f"tc={plan.tc}",
        "proto=uci",
        "-openings",
        f"file={plan.book}",
        "format=epd",
        "order=random",
        "-srand",
        str(plan.seed * 1_000_003 + step),
        "-rounds",
        str(plan.pairs_per_step),
        "-games",
        "2",
        "-repeat",
        "-concurrency",
        "1",
        *plan.extra_arguments,
        "-log",
        f"file={directory / f'step-{step:05d}.engines.log'}",
        "level=info",
        "engine=true",
        "-pgnout",
        f"file={directory / f'step-{step:05d}.pgn'}",
    ]


def match_result(output: str) -> tuple[int, int, int]:
    """Wins, losses and draws of ``plus`` from fastchess's final report."""
    for line in output.splitlines():
        if (count := CRASH_COUNT.match(line)) and int(count.group(1)) == 0:
            continue
        if FAILURE.search(line):
            raise RuntimeError(f"Engine failure in an SPSA step: {line.strip()}")
    results = [match for line in output.splitlines() if (match := RESULT.match(line.strip()))]
    _require(bool(results), "fastchess printed no final result")
    games, wins, losses, draws = (int(value) for value in results[-1].groups())
    _require(games == wins + losses + draws, "fastchess result counts are inconsistent")
    return wins, losses, draws


def _set_aside(games: Path, step: int) -> None:
    """Rename the files an unfinished attempt at ``step`` left to ``step-NNNNN.failed-K.*``."""
    stem = f"step-{step:05d}"
    present = [games / f"{stem}{suffix}" for suffix in STEP_FILES]
    present = [path for path in present if path.exists()]
    if not present:
        return
    attempt = 1
    while any((games / f"{stem}.failed-{attempt}{suffix}").exists() for suffix in STEP_FILES):
        attempt += 1
    for path in present:
        path.replace(games / f"{stem}.failed-{attempt}{path.name.removeprefix(stem)}")


def _play(
    plan: Plan, plus: dict[str, float], minus: dict[str, float], step: int, games: Path
) -> tuple[int, int, int]:
    _set_aside(games, step)
    completed = subprocess.run(
        command(plan, plus, minus, step, games),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (games / f"step-{step:05d}.log").write_text(completed.stdout, encoding="utf-8")
    _require(completed.returncode == 0, f"fastchess exited {completed.returncode} at {step}")
    result = match_result(completed.stdout)
    (games / f"step-{step:05d}.engines.log").unlink(missing_ok=True)
    return result


def _plan_digest(text: str) -> str:
    return hashlib.sha256(json.dumps(json.loads(text), sort_keys=True).encode()).hexdigest()


def run(plan_text: str, output: Path, attempts: int = 1) -> dict[str, float]:
    _require(attempts >= 1, "attempts must be at least 1")
    plan = load_plan(plan_text)
    digest = _plan_digest(plan_text)
    state_path = output / "state.json"
    if state_path.exists():
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
        _require(state["plan_sha256"] == digest, "The output directory belongs to another plan")
    else:
        _require(not output.exists() or not any(output.iterdir()), f"{output} is not empty")
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(plan_text, encoding="utf-8")
        state = {
            "plan_sha256": digest,
            "played": 0,
            "theta": {parameter.name: parameter.start for parameter in plan.parameters},
        }
    games = output / "games"
    games.mkdir(exist_ok=True)
    while state["played"] < plan.pairs:
        played = int(state["played"])
        step = played // plan.pairs_per_step
        theta: dict[str, float] = state["theta"]
        flip = flips(plan, step)
        plus, minus, steps = {}, {}, {}
        for parameter, sign in zip(plan.parameters, flip, strict=True):
            _, c_k = gains(parameter, plan.pairs, played)
            plus[parameter.name] = clamp(parameter, theta[parameter.name] + c_k * sign)
            minus[parameter.name] = clamp(parameter, theta[parameter.name] - c_k * sign)
            steps[parameter.name] = c_k
        for attempt in range(1, attempts + 1):
            try:
                wins, losses, draws = _play(plan, plus, minus, step, games)
                break
            except (RuntimeError, ValueError) as error:
                failure = {"step": step, "attempt": attempt, "error": str(error)}
                with (output / "failures.jsonl").open("a", encoding="utf-8") as failures:
                    failures.write(json.dumps(failure) + "\n")
                print(
                    f"step {step + 1}: attempt {attempt}/{attempts} failed: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if attempt == attempts:
                    raise
        updated = {}
        for parameter, sign in zip(plan.parameters, flip, strict=True):
            a_k, c_k = gains(parameter, plan.pairs, played)
            change = a_k / c_k * (wins - losses) * sign
            updated[parameter.name] = clamp(parameter, theta[parameter.name] + change)
        record = {
            "step": step,
            "played": played,
            "flip": flip,
            "c_k": steps,
            "plus": plus,
            "minus": minus,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "theta": updated,
        }
        with (output / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(record) + "\n")
        state = {**state, "played": played + plan.pairs_per_step, "theta": updated}
        temporary = output / "state.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)
        values = ", ".join(f"{key}={option_text(value)}" for key, value in updated.items())
        total = plan.pairs // plan.pairs_per_step
        print(f"step {step + 1}/{total} +{wins} -{losses} ={draws}: {values}", flush=True)
    return dict(state["theta"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    tune = commands.add_parser("run", help="run or resume an SPSA plan")
    tune.add_argument("--config", type=Path, required=True)
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument(
        "--attempts", type=int, default=1, help="times a failing step is played (default 1)"
    )
    arguments = parser.parse_args(argv)
    final = run(arguments.config.read_text(encoding="utf-8"), arguments.output, arguments.attempts)
    print(json.dumps({key: option_text(value) for key, value in final.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
