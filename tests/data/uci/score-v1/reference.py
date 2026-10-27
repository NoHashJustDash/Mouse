# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline integrity audit and CPU-only replay of the pinned score observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

EXE_SHA256 = "e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b"
MANIFEST_SHA256 = "12e61ba1e2999d276103a95b8d9dfb78389b4609f70517fd8e50ca39c09b0b28"
OLD_ROOT = "<workspace>/.worktrees/search/.toolchain/uci-score-research"
SCORE = re.compile(r"\bscore (cp|mate) (-?\d+)(?: wdl (\d+) (\d+) (\d+))?")
CPU_OPTIONS = re.compile(
    r"--backend-opts=backend=random,random\(seed=0\),(?:record|replay)_file='[^']+'"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def arguments(directory: Path) -> list[str]:
    value = read_json(directory / "argv.json")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Invalid argv: {directory.name}")
    result: list[str] = value
    selected = [item for item in result if item.startswith("--backend=")]
    if len(selected) != 1 or selected[0] not in (
        "--backend=trivial",
        "--backend=recordreplay",
    ):
        raise ValueError("An explicit approved CPU backend is required")
    for required in (
        "--config=",
        "--weights=",
        "--threads=1",
        "--minibatch-size=1",
        "--max-prefetch=0",
        "--nncache=0",
        "--task-workers=0",
    ):
        if result.count(required) != 1:
            raise ValueError(f"Missing fixed CPU option: {required}")
    options = [item for item in result if item.startswith("--backend-opts=")]
    if len(options) != 1:
        raise ValueError("Exactly one backend-options argument is required")
    if selected[0] == "--backend=recordreplay" and not CPU_OPTIONS.fullmatch(
        options[0]
    ):
        raise ValueError(
            "Record/replay must explicitly use the seeded CPU random child"
        )
    if selected[0] == "--backend=trivial" and options[0] != "--backend-opts=":
        raise ValueError("Trivial controls require empty backend options")
    return result


def payload(directory: Path) -> tuple[str, int, tuple[int, ...], str]:
    text = (directory / "stdout.txt").read_text(encoding="utf-8")
    matches = SCORE.findall(text)
    best = [line for line in text.splitlines() if line.startswith("bestmove ")]
    if not matches or len(best) != 1:
        raise ValueError(f"Missing complete score/bestmove: {directory.name}")
    kind, value, win, draw, loss = matches[-1]
    wdl = tuple(int(item) for item in (win, draw, loss) if item)
    return kind, int(value), wdl, best[0]


def case_table(root: Path) -> str:
    data = read_json(root / "reference/score-cases-v2.json")
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("Invalid score cases")
    rows = data["cases"]
    if len(rows) != 28 or len({row["name"] for row in rows}) != 28:
        raise ValueError("The full 28-case baseline is required")
    lines = [
        "// SPDX-License-Identifier: GPL-3.0-or-later",
        "// Exact float32 inputs and literal official Lc0 outputs; generated from retained observations.",
    ]
    for row in rows:
        q = struct.unpack("<I", bytes.fromhex(row["q_bits"]))[0]
        d = struct.unpack("<I", bytes.fromhex(row["d_bits"]))[0]
        observed = row["observed"]
        for repeat in ("a", "b"):
            actual = payload(root / "reference" / f"case-{row['name']}-{repeat}")
            if actual[:3] != ("cp", observed["cp"], tuple(observed["wdl"])):
                raise ValueError(f"Case observation disagrees: {row['name']}")
        wdl = ", ".join(map(str, observed["wdl"]))
        lines.append(
            f'{{.name = "{row["name"]}", .q_bits = 0x{q:08x}U, .draw_bits = 0x{d:08x}U, .cp = {observed["cp"]}, .wdl = {{{wdl}}}}},'
        )
    return "\n".join(lines) + "\n"


def audit(root: Path) -> list[Path]:
    reference = root / "reference"
    manifest_path = reference / "evidence-manifest.json"
    if digest(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Pinned original evidence manifest changed")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid evidence manifest")
    files = manifest["files"]
    actual = {
        path.relative_to(reference).as_posix()
        for path in reference.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(files):
        raise ValueError("Original evidence file set changed")
    for name, identity in files.items():
        path = reference / name
        if not path.resolve().is_relative_to(reference.resolve()):
            raise ValueError("Unsafe evidence path")
        if (
            path.stat().st_size != identity["bytes"]
            or digest(path) != identity["sha256"]
        ):
            raise ValueError(f"Original evidence changed: {name}")
    if (root / "cases.inc").read_text(encoding="utf-8") != case_table(root):
        raise ValueError("C++ golden table differs from actual observations")
    sessions = sorted(path.parent for path in reference.glob("*/argv.json"))
    if len(sessions) != 72:
        raise ValueError("All 72 original search sessions are required")
    for directory in sessions:
        arguments(directory)
        if read_json(directory / "exit.json") != {"exit": 0}:
            raise ValueError(f"Failed original session: {directory.name}")
        payload(directory)
        initialized = re.findall(
            r"Creating backend \[([^]]+)\]",
            (directory / "stderr.txt").read_text(encoding="utf-8"),
        )
        if not set(initialized) <= {"random", "trivial", "recordreplay"}:
            raise ValueError("Unexpected backend in original transcript")
    return sessions


def run_session(
    argv: list[str], commands: list[str], directory: Path, cwd: Path
) -> None:
    directory.mkdir()
    (directory / "argv.json").write_text(
        json.dumps(argv, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "commands.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    stdout: list[str] = []
    stderr: list[str] = []
    lines: queue.Queue[str | None] = queue.Queue()

    def reader(stream: TextIO, retained: list[str], publish: bool) -> None:
        try:
            for line in stream:
                retained.append(line)
                if publish:
                    lines.put(line)
        finally:
            if publish:
                lines.put(None)

    with subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        assert (
            process.stdin is not None
            and process.stdout is not None
            and process.stderr is not None
        )
        workers = [
            threading.Thread(target=reader, args=(process.stdout, stdout, True)),
            threading.Thread(target=reader, args=(process.stderr, stderr, False)),
        ]
        for worker in workers:
            worker.start()
        try:
            for command in commands:
                process.stdin.write(command + "\n")
                process.stdin.flush()
                prefix = (
                    "uciok"
                    if command == "uci"
                    else "readyok"
                    if command == "isready"
                    else "bestmove "
                    if command.startswith("go ")
                    else None
                )
                if prefix is None:
                    continue
                deadline = time.monotonic() + 15
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"Lc0 waiting for {prefix}")
                    line = lines.get(timeout=remaining)
                    if line is None:
                        raise ValueError("Lc0 ended before completing the protocol")
                    if line.startswith(prefix):
                        break
            process.stdin.close()
            process.wait(timeout=10)
            if process.returncode != 0:
                raise ValueError(f"Lc0 exit {process.returncode}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            for worker in workers:
                worker.join()
            (directory / "stdout.txt").write_text("".join(stdout), encoding="utf-8")
            (directory / "stderr.txt").write_text("".join(stderr), encoding="utf-8")
            (directory / "exit.json").write_text(
                json.dumps({"exit": process.returncode}) + "\n", encoding="utf-8"
            )


def regenerate(root: Path, executable: Path, output: Path) -> None:
    sessions = audit(root)
    output = output.resolve()
    if output.is_relative_to(root.resolve()):
        raise ValueError("Regeneration output must be outside the pinned fixtures")
    if "'" in output.as_posix() or output.exists():
        raise ValueError(
            "Regeneration requires a new output directory without apostrophes"
        )
    if digest(executable) != EXE_SHA256:
        raise ValueError("Official Lc0 executable SHA256 mismatch")
    # Freeze plans before any invocation; validate/hash the original artifact set
    # again after execution. No automatic backend or network download is available.
    plans = [
        (
            directory,
            arguments(directory),
            (directory / "commands.txt").read_text(encoding="utf-8").splitlines(),
        )
        for directory in sessions
    ]
    plans.sort(key=lambda item: (item[0].name != "record-root", item[0].name))
    output.mkdir(parents=True)
    for replay in (root / "reference").glob("*.replay.bin"):
        shutil.copyfile(replay, output / replay.name)
    for directory, old_argv, commands in plans:
        argv = [str(executable.resolve())] + [
            item.replace(OLD_ROOT, output.as_posix()) for item in old_argv[1:]
        ]
        run_session(argv, commands, output / directory.name, output)
        if payload(directory) != payload(output / directory.name):
            raise ValueError(f"Binary replay changed: {directory.name}")
    if (output / "record-root.bin").read_bytes() != (
        root / "reference/record-root.bin"
    ).read_bytes():
        raise ValueError("Seeded CPU root record changed")
    if digest(executable) != EXE_SHA256:
        raise ValueError("Official Lc0 executable changed during regeneration")
    audit(root)
    report = {
        "schema": 1,
        "status": "passed",
        "sessions": len(plans),
        "exact_cases": 28,
        "repeats_per_case": 2,
        "gpu_executions": 0,
        "lc0_sha256": EXE_SHA256,
    }
    (output / "regeneration.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "regenerate"))
    parser.add_argument(
        "--fixtures", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--lc0", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.mode == "regenerate":
            if args.lc0 is None or args.output is None:
                raise ValueError("Regeneration requires --lc0 and --output")
            regenerate(args.fixtures.resolve(), args.lc0, args.output)
        else:
            audit(args.fixtures.resolve())
        print(json.dumps({"status": "passed", "sessions": 72, "cases": 28}))
        return 0
    except (
        OSError,
        ValueError,
        TimeoutError,
        queue.Empty,
        subprocess.SubprocessError,
        struct.error,
        KeyError,
        TypeError,
    ) as error:
        print(f"Score reference error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
