<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Classical input and policy references

These fixtures were produced by the official Lc0 v0.32.1 Windows binary, with
CPU `backend=trivial` and classical input format 1. The binary's SHA-256 is
`e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b`;
its source commit is `fd71a2d921b689c5f479d3227c3806c8e272d9c5`.
The supplied T1 weights have SHA-256
`bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406`.
The trivial backend does not evaluate those weights. These are binary input and
policy references; numerical T1 conversion evidence is recorded separately.

`cases.json` contains 27 original requests. `lc0-classical-v1/reference.json`
retains the exact process/UCI commands, binary/weights hashes, explicit FEN
history, move/index rows, and raw capture hashes. The `.policy.txt` files contain
actual verbose root statistics. Each `.v6.gz` holds the first unmodified
8,356-byte record from one completed selfplay game, recompressed with gzip
`mtime=0`. There are 81 records: all 27 positions with history fill `no`,
`fen_only`, and `always`. Full raw games and process logs remain in ignored
`build/lc0-reference/corpus-001`.

The requests cover both colors, asymmetric boards, orthodox and Chess960
castling (including stationary king/rook and swaps), all promotion suffixes,
capture promotions, en passant and pinned en passant, rule50, explicit opening
history, and a repetition cycle. `equivalent-start.*` adds one actual record
for semantic initial-board detection when empty ranks are spelled `44`.
Its separate provenance retains the exact command and original 45-record game
hash; the selected first record is checked by the regression suite.

The original corpus was captured twice with the same normalized input/policy
SHA-256:
`731775f798fae4634a05f49977da9b7c6ddd61fd1839e0bb222ccb7d0e47ee3f`.
This digest includes every case/fill's 112 mask/value pairs and sorted legal
move/index pairs. It excludes game trajectories, search statistics, timings,
random output directory names, and gzip headers. Those excluded outputs are
not claimed to be byte reproducible. Request file SHA-256:
`41803115f27a6629cafe841435ee7a5c41e712a3e9af6fba271c04f606fa814c`.
Tests verify these hashes, each transcript, and every selected raw record.

## APIs and conventions

`mouse_tools.encoding` is an original independent Python implementation.
`HistoryFrame(fen, repetitions=0)` takes an absolute validated FEN and an explicit
unsigned 32-bit Lc0 repetition count. Frames are chronological, oldest first;
the encoder uses the most recent eight. The caller owns repetition identity.
It must not silently substitute a FIDE repetition key for Lc0's raw/pseudo-EP
history identity.

`encode_classical(frames, input_format=1, fill="fen_only")` returns 112 immutable
`Plane(mask, value)` values. `Plane.dense()` expands A1-first squares. Every
history frame uses the current player's absolute color and rank orientation.
Rule50 is raw float32, not divided by 100. `no` allows the one inferred prior
board from the earliest raw en-passant target; filling repeats that inferred
board for subsequent missing slots. `fen_only` stops at the semantic white
initial board (counter values do not change board identity). Other formats,
unknown fill modes, malformed FENs, and nonalternating histories are rejected.

`policy_index(move, black=False)` and
`move_from_policy(index, legal_moves, black=False)` use king-to-rook castling
coordinates. The inverse requires legal board context. Black coordinates are
rank-flipped. There are 1,792 ordinary queen-ray/knight slots and 66 explicit
Q/R/B promotion slots; knight promotions use the ordinary geometry slot.
The real binary confirms `a7a8n=1401`, `a7a8q=1792`, `a7a8r=1793`,
`a7a8b=1794`, `e1h1=103`, and `e1a1=97`.

Behavior/schema sources, inspected at the pinned commit:

- [Encoder](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/encoder.cc)
- [Verbose policy rows](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc)
- [Training v6 layout](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/trainingdata/trainingdata_v6.h)
- [Training reader](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/trainingdata/reader.cc)

No upstream implementation or generated policy table is vendored here.

## Reproduce the binary fixtures

Install the tools with the verified development dependencies and invoke Python
in isolated mode. Run from the repository worktree, use new output paths, and
point `--engine`/`--weights` at the pinned files. The harness rehashes both before
execution, requires the pinned version and `trivial` backend, contains generated
files in the requested directory, completes each game, and rejects incomplete
or malformed output. It runs reference processes sequentially.

```powershell
python -I -m mouse_tools.reference_corpus --engine <workspace>/.toolchain/reference/lc0/lc0.exe --weights <workspace>/.toolchain/downloads/t1-256x10-distilled-swa-2432500.pb.gz --cases tests/data/nn/cases.json --output-dir build/lc0-reference/reproduced-new --raw-dir build/lc0-reference/raw-new
```

The retained second run used `build/lc0-reference/reproduced-v1` and
`build/lc0-reference/corpus-002`; it reproduced all 27 cases and the digest above.
Individual captures are available through `python -I -m mouse_tools.lc0_reference
policy|training`; `--moves` is UCI for policy and SAN for training PGN openings.
The exact selfplay arguments are retained per capture. In particular, selfplay
must not receive classic mode's `--config=` option: Lc0 reports unknown arguments
with exit code zero, so the harness also requires a complete training chunk.

## Cross-language and converter corpora

`mouse_tools.encoding_gate` validates exactly 10,000 C++ probe records against
the independent Python encoder and map. For every current FEN it also queries
the pinned Stockfish 17.1 process at depth one with Chess960 notation enabled.
There is no count cache. Every legal move/index and inverse and all 112 planes
must match. Missing, duplicate, malformed, or partial records fail with retained
JSON evidence. This 10,000-position comparison is distinct from the 27-position
actual-Lc0 fixture comparison above.

```powershell
build/windows-release/mouse_encoder_probe.exe .toolchain/encoder-corpus-new.jsonl
python -I -m mouse_tools.encoding_gate --probe .toolchain/encoder-corpus-new.jsonl --oracle .toolchain/reference/stockfish-17.1.exe --output build/encoding-gate-new.json
python -I -m mouse_tools.make_parity_inputs --probe .toolchain/encoder-corpus-new.jsonl --gate-report build/encoding-gate-new.json --output .toolchain/convert/parity-inputs-new.npy --provenance .toolchain/convert/parity-inputs-new.json
```

`make_parity_inputs` requires the source SHA and all fixed 10,000-position report
counts to agree. It selects the first 1,000 distinct complete FP32 tensors,
deduplicating exact little-endian bytes across all 112 planes. It continues
past duplicate records and fails if fewer than 1,000 distinct tensors exist.
There is no count or tolerance option and no padding. NPY shape is
`[1000,112,8,8]`, C order, float32, written with `allow_pickle=False`.
The provenance records selected IDs, source line numbers, tensor hashes, and
source/report/NPY hashes. Existing outputs and resolved or hard-link path
collisions are rejected. Both outputs are staged and exclusively published;
provenance is published last as the completion marker. Consumers require both.
An interrupted publication without provenance must not be treated as complete.

`gate-summary-v1.json` retains compact hashes/counts for the successful full
10,000-row run and the generated 1,000 distinct inputs. The full per-position
report and NPY remain ignored. The initial missing-oracle failure report is
retained separately. The converter's full CPU parity report records its own
model hashes, fixed absolute tolerance `1e-4`, and every output head's errors.
