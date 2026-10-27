# Cache equality corpus v1

V1_PLAN 3.3/3.4 requires at least 200 positions from openings, middlegames and
endgames. `cache-equality-v1.tsv` contains 80 book positions, 80 middlegames and
40 endgames: 134 White-to-move and 66 Black-to-move roots. Histories contain
at most 195 plies. Every root is distinct ignoring the two FEN counters.

The source is the retained legal UCI transcript from the cancelled cache-v4
match. Its SHA-256 is
`940adb4b392e570c500e46a5deb92cf58d2f8c81c32dd071bc83c990e13ec20c`.
Opening positions are at ply 16 from the pinned `8moves_v3.pgn` book (its hash
is recorded in the TSV header). Middlegames are sampled at plies 35-105,
with at least 16 non-king pieces. Endgames have 4-10 non-king pieces and at
least 80 plies. Duplicate roots, fewer than two legal moves and rule-50
counters of 99 or more are excluded before the corpus is frozen.

FEN and legal-move counts were obtained by replaying every selected complete
UCI history through the existing official Stockfish 17.1 oracle; 281 queries
produced these 200 cases. This uses legal game-position facts, not match
results. The cancelled SPRT is not credited as a passed gate.

Run the same `mouse_cache_equivalence` harness linked against each cache
build, with the identical actual FP32 T1 model and TSV. It uses deterministic
compute, one thread, batch 8, a 2-million-entry cache and an 800-node limit,
with smart pruning disabled. Each root is searched cold and then again after
resetting visits while retaining its NN cache. Compare raw Q/draw/prior bits,
bestmove and all root-child visits. Rounded centipawn scores are insufficient.
The footer is written only after every case and the final model/corpus hash
audits pass. Timing and charged-byte fields are measurements, not equality
fields. This CPU gate does not establish GPU throughput or playing strength.

To compare retained captures, run the following Python 3.12 code from the
repository root with `python -I`. The two `stdout.txt` files are the native
harness outputs; `process.json` comes from the existing match capture utility.
An incomplete capture fails instead of accepting only its matching prefix.

```python
import hashlib
import json
from pathlib import Path

base = Path(".toolchain/cache-memory-v1")
runs = []
for name in ("before", "after"):
    folder = base / f"{name}-equality"
    process = json.loads((folder / "process.json").read_text(encoding="utf-8"))
    assert process["exit_code"] == 0 and process["error"] is None
    assert process["pumps_joined"] and not process["terminated_by_runner"]
    records = [json.loads(line) for line in
               (folder / "stdout.txt").read_text(encoding="utf-8").splitlines()]
    header, *cases, footer = records
    assert header["type"] == "header" and footer["type"] == "complete"
    assert header["schema"] == "MouseCacheEquality1"
    assert (header["nodes_per_search"], header["threads"], header["batch"],
            header["cache_capacity"], header["passes"]) == (800, 1, 8, 2000000, 2)
    assert header["positions"] == footer["positions"] >= 200
    assert len(cases) == footer["records"] == 2 * header["positions"]
    assert all(row["type"] == "case" for row in cases)
    assert len({row["id"] for row in cases}) == header["positions"]
    assert len({(row["id"], row["pass"]) for row in cases}) == len(cases)
    assert {row["pass"] for row in cases} == {0, 1}
    assert {row["group"] for row in cases} == {"book", "middle", "end"}
    assert footer["nodes"] == [sum(row["nodes"] for row in cases
                                   if row["pass"] == p) for p in (0, 1)]
    binary = base / f"{name}-bin"
    manifest = json.loads((binary / "SHA256.json").read_text(encoding="utf-8"))
    for file, digest in manifest.items():
        assert hashlib.sha256((binary / file).read_bytes()).hexdigest() == digest
    runs.append((header, cases, footer))
keys = ("id", "group", "pass", "bestmove", "nodes", "q_bits", "draw_bits",
        "roots", "cache_hits", "cache_misses")
left, right = runs
assert {k: v for k, v in left[0].items() if k != "git"} == {
    k: v for k, v in right[0].items() if k != "git"}
assert all(all(a[k] == b[k] for k in keys)
           for a, b in zip(left[1], right[1], strict=True))
print("PASS:", left[0]["positions"], "positions;", len(left[1]),
      "cold/warm records; bit-exact bestmove, visits, Q and root statistics")
for name, (_, _, footer) in zip(("before", "after"), runs, strict=True):
    print(name, "cold/warm NPS:", [round(n * 1e9 / t, 3) for n, t in
          zip(footer["nodes"], footer["elapsed_ns"], strict=True)])
```
