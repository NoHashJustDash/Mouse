# Standalone search

This development checkpoint has a bounded, pipelined MCTS search and a standalone
analysis command. Full UCI play, fixed-node strength parity, complete self-play
and sanitizer acceptance are still pending. No release gate is implied.

For a converted FP32 T1 model:

```powershell
.\build\windows-release\mouse.exe search --weights nets/t1.onnx --backend cpu --precision fp32 --deterministic --nodes 800
.\build\windows-release\mouse.exe search --weights nets/t1.onnx --backend cpu --precision fp32 --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --moves "e2e4 e7e5" --movetime 1000
```

GPU commands use the explicit SDK directories, precision and cache configuration
documented in [INFERENCE.md](INFERENCE.md). A model or provider failure returns a
nonzero exit status and a diagnostic on stderr. Success emits one complete
`MouseSearch1` JSON object after validating the best move, completed PV and
unchanged model identity. Relative model and cache paths resolve against the
current directory. There is no artifact discovery or provider fallback.

The JSON includes build/model/provider/precision identities, a hash of the
canonical initial FEN, variant and complete explicit move sequence, final FEN,
node accounting, depth, selective depth, Q/draw, elapsed time, memory charge,
best move and PV. The `moves_left_utility` field is always a JSON boolean: it
reports whether optional MLH selection utility is enabled in this search. It
does not imply the model supplied that head or the confidence threshold fired.
A FEN root and a move sequence reaching the same board have
different history hashes. A rule-terminal root returns a null best move.
A root-only evaluation can choose a legal best move by prior while its completed
PV remains empty; it does not report that move as explored.

| Option | Default | Contract |
| --- | --- | --- |
| `--weights`, `--backend`, `--precision` | Required | Explicit ONNX model, cpu/cuda/tensorrt and fp32/fp16 |
| `--fen` | Start position | Strict FEN, at most 256 characters |
| `--moves` | Empty | Quoted consecutive legal UCI moves; at most 131072 plies / 1 MiB |
| `--chess960` | Off | King-to-rook castling notation; required for nonstandard rights |
| `--nodes` | 800 | Total completed root visits, including inherited visits |
| `--movetime` | Unset | Positive milliseconds, steady clock |
| `--depth` | Unset | First reached selective depth, 1..512 plies |
| `--threads` | 4 | 1..128 gather workers |
| `--batch` | 64 | 1..1024 leaves per fixed-capacity slot |
| `--buckets` | Derived | Increasing provider buckets covering the selected batch |
| `--ram-limit-mb` | 512 | 1..16384 MiB for managed tree, pipeline, worker and history storage |
| `--nncache-mb` | 0 | Separate exact-input NN cache budget |
| `--deterministic` | Off | One effective gather worker, fixed scheduling and ORT deterministic compute; CPU threads must be 1 |
| `--no-smart-pruning` | Off | Disable time-budget-based early pruning |
| `--no-moves-left-utility` | Off; utility enabled | Disable optional moves-left selection utility for controlled comparisons |
| `--cpuct` / `--cpuct-base` / `--cpuct-factor` | 1.745 / 38739 / 3.894 | Finite validated PUCT schedule |
| `--fpu-value` | 0.330 | Non-root reduction; root FPU is absolute +1 |
| `--policy-temperature` | 1.359 | Positive legal-policy softmax temperature |

An explicit time/depth limit removes the implicit 800-node cutoff; an explicit
node limit is retained with other limits. The first applicable limit wins.
Deterministic repeatability refers to the same model, runtime, configuration and
node/depth limits. Wall-time cutoffs and elapsed-time fields depend on scheduling.

Optional moves-left utility affects selection only when both parent and visited
child have complete valid MLH samples and `abs(parentQ) > 0.8`. Missing heads or
partial MLH samples leave Q+U selection unchanged. Known M samples back up with
one additional ply per parent; terminal M is zero. `--no-moves-left-utility`
restores Q+U selection even when the model supplies MLH. Final best-move ranking
still uses completed visits, then Q and stable prior/order. This candidate has
source/formula and synthetic-search tests; the unchanged 128/1024/8192-node real
mate gates and strength acceptance remain separate. The formula follows lc0's
[moves-left utility](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc).

The RAM charge includes fixed request/output slots, preallocated worker/history
capacities, tree storage, owner objects and allocation allowances. Borrowed
backend/session memory, the separately configured NN cache and OS thread stacks
are outside this search budget. Setup fails clearly if fixed costs cannot fit.
Re-rooting retains reachable statistics only when position, full NN history and
backend identities match; context-dependent repetition draws force a fresh tree.

The independent mate fixtures and their complete legal-oracle proofs live in
[tests/data/search/mates-v1](../tests/data/search/mates-v1/README.md). Actual search
node budgets and strength results must be measured and recorded separately.

Root diagnostics include each legal move, its completed visits, Q/draw and half-prior value. Unvisited moves report zero visits; these rows do not imply a completed continuation.
