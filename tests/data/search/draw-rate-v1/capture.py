# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture lc0's draw-rate rescaling for chosen root W/D/L values.

Each case writes a one-record replay file (the score-v1 startpos record with its
Q and D replaced), runs the pinned official lc0 executable on its CPU
record/replay backend for ``go nodes 1`` with WDLDrawRateTarget,
WDLDrawRateReference and WDLMaxS set, and keeps the UCI score and the root node's
verbose WL/D. Those are lc0's internally stored (rescaled) values passed through
its display inverse, which at zero contempt leaves the width unchanged.

    python -I -B tests/data/search/draw-rate-v1/capture.py \
        --lc0 .toolchain/reference/lc0/lc0.exe --work .toolchain/draw-rate/<new dir>

Writes ``cases.json`` (inputs, argv, transcript excerpts) and ``cases.inc``
(literal float bits and observed integers) next to this script. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parents[1] / "uci" / "score-v1" / "reference" / "record-root.bin"
LC0_SHA256 = "e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b"
FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
SCORE = re.compile(r"^info depth 1 .* score cp (-?\d+) wdl (\d+) (\d+) (\d+) ")
NODE = re.compile(r"^info string node .*\(WL: +(-?\d\.\d{5})\) \(D: (\d\.\d{3})\)")

# (q, d, target, reference, max_s): balanced, favoured and winning roots across
# sharper and softer targets, two references, and a WDLMaxS cap that binds.
CASES = [
    (q, d, target, reference, max_s)
    for q, d in [
        (0.0, 0.5),
        (0.3, 0.4),
        (-0.45, 0.35),
        (0.6, 0.3),
        (0.85, 0.1),
        (0.1, 0.8),
    ]
    for target, reference, max_s in [
        (0.3, 0.5, 1.4),
        (0.75, 0.5, 1.4),
        (0.9, 0.62, 1.4),
        (0.6, 0.5, 0.5),
    ]
]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> str:
    return "0x" + struct.pack(">f", value).hex()


def template() -> tuple[int, list[float]]:
    data = TEMPLATE.read_bytes()
    key, count = struct.unpack_from("<Qi", data, 0)
    if len(data) != 12 + 4 * count:
        raise ValueError("score-v1 record-root.bin must hold exactly one record")
    return key, list(struct.unpack_from("<" + "f" * count, data, 12))


def run(
    lc0: Path, replay: Path, target: float, reference: float, max_s: float
) -> dict[str, object]:
    argv = [
        str(lc0),
        "--config=",
        "--backend=recordreplay",
        f"--backend-opts=backend=random,random(seed=0),replay_file='{replay.as_posix()}'",
        "--weights=",
        "--threads=1",
        "--minibatch-size=1",
        "--max-prefetch=0",
        "--nncache=0",
        "--task-workers=0",
        "--show-wdl",
        "--verbose-move-stats",
        f"--wdl-draw-rate-target={target}",
        f"--wdl-draw-rate-reference={reference}",
        f"--wdl-max-s={max_s}",
    ]
    # quit only after bestmove: lc0 abandons a running search when stdin delivers it early.
    commands = ["uci", "isready", "ucinewgame", f"position fen {FEN}", "go nodes 1"]
    stdout: list[str] = []
    with subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write("\n".join(commands) + "\n")
            process.stdin.flush()
            for line in process.stdout:
                stdout.append(line)
                if line.startswith("bestmove "):
                    break
            process.stdin.write("quit\n")
            process.stdin.close()
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.kill()
    if process.returncode != 0 or not stdout or not stdout[-1].startswith("bestmove "):
        raise ValueError(f"lc0 exit {process.returncode} before completing the search")
    score = [m for line in stdout if (m := SCORE.match(line))]
    node = [m for line in stdout if (m := NODE.match(line))]
    if len(score) != 1 or len(node) != 1:
        raise ValueError("expected one depth-1 score line and one root node line")
    return {
        "argv": [Path(argv[0]).name, *argv[1:3], "--backend-opts=<replay>", *argv[4:]],
        "score_line": score[0].string,
        "node_line": node[0].string,
        "cp": int(score[0][1]),
        "wdl": [int(score[0][2]), int(score[0][3]), int(score[0][4])],
        "wl": node[0][1],
        "d": node[0][2],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lc0", type=Path, required=True)
    parser.add_argument(
        "--work", type=Path, required=True, help="new directory for replay files"
    )
    options = parser.parse_args()
    digest = hashlib.sha256(options.lc0.read_bytes()).hexdigest()
    if digest != LC0_SHA256:
        raise SystemExit(
            f"lc0 executable SHA256 {digest} is not the pinned {LC0_SHA256}"
        )
    options.work.mkdir(parents=True)
    key, payload = template()
    rows = []
    for index, (q, d, target, reference, max_s) in enumerate(CASES):
        q32, d32 = f32(q), f32(d)
        record = [q32, d32, *payload[2:]]
        replay = options.work / f"case-{index:02d}.replay.bin"
        replay.write_bytes(
            struct.pack("<Qi", key, len(record))
            + struct.pack("<" + "f" * len(record), *record)
        )
        observed = run(options.lc0, replay, target, reference, max_s)
        rows.append(
            {
                "q": q,
                "d": d,
                "q_bits": bits(q32),
                "d_bits": bits(d32),
                "target": target,
                "reference": reference,
                "max_s": max_s,
                **observed,
            }
        )
    (HERE / "cases.json").write_text(
        json.dumps({"lc0_sha256": LC0_SHA256, "fen": FEN, "cases": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "// SPDX-License-Identifier: GPL-3.0-or-later",
        "// Generated by capture.py from the pinned lc0 executable; see cases.json.",
        "// {q bits, d bits, target, reference, max_s, cp, W, D, L, verbose WL, verbose D}",
    ]
    for row in rows:
        w, dr, lo = row["wdl"]
        lines.append(
            f"{{{row['q_bits']}U, {row['d_bits']}U, {row['target']}, {row['reference']}, {row['max_s']}, "
            f'{row["cp"]}, {w}, {dr}, {lo}, "{row["wl"]}", "{row["d"]}"}},'
        )
    (HERE / "cases.inc").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"captured {len(rows)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
