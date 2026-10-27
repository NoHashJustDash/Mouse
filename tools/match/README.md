# Auditable fastchess matches

`mouse_tools.match` validates an explicit frozen plan, constructs one pinned
fastchess command, supervises its complete process tree, and audits retained
native evidence. Planning does not launch a process. A plan is not match proof.

```powershell
python -I -m mouse_tools.match plan --config C:/absolute/plan.json --output C:/absolute/new-plan-review
python -I -m mouse_tools.match run --config C:/absolute/plan.json --output C:/absolute/new-match-evidence
```

Use the normal installed `mouse-tools` package and the locked converter-enabled
Python environment. Both output directories must be new. Real match execution,
GPU scheduling, final artifact freeze, and review remain coordinated by the
parent project owner. Fixed-time profiles run exactly one game at once; fixed-node parity uses four.
Both engines and their pooled instances can reside on the GPU simultaneously,
so establish VRAM headroom before execution.

## Named profiles

| Profile | Complete games / color pairs | Limit | Book | Test |
|---|---:|---|---|---|
| `package-smoke` | 10 / 5 | 1+0.01 | Balanced | Package acceptance diagnostic only |
| `selfplay-fast` | 1000 / 500 | 1+0.01 | Balanced | Fixed-count selfplay safety |
| `uci-stress-fast` | 2000 / 1000 | 1+0.01 | Balanced | Fixed-count safety |
| `uci-stress-stc` | 500 / 250 | 10+0.1 | Balanced | Fixed-count safety |
| `tm-safety-fast` | 2000 / 1000 | 1+0.01 | Balanced | Fixed-count safety |
| `tm-safety-stc` | 2000 / 1000 | 10+0.1 | Balanced | Fixed-count safety |
| `tm-safety-ltc` | 2000 / 1000 | 60+0.6 | Balanced | Fixed-count safety |
| `parity-sprt` | Predeclared even `max_games` | Explicit `nodes` only | Balanced | Normalized SPRT -10,0 |
| `tm-sprt` | Predeclared even `max_games` | 10+0.1 | UHO | Normalized SPRT -5,0 |

`package-smoke` supplies the 10-game clean-machine/unzip/net check required by
[Engineering Standards §6](../../docs/ENGINEERING_STANDARDS.md#6-releases).
It requires two identical Mouse executable/model/option configurations, explicit
GPU `cuda|tensorrt` with `fp16`, `Threads=4`, `MinibatchSize=64`, and `NNCacheSize=0`.
Its five color pairs use the full strict capture/PGN/book/clock audit and accept
no count or limit override. Plan/report fields `acceptance_scope=package-smoke-only`
and `roadmap_gate_claim=false` keep this diagnostic separate from every roadmap
match gate. Ten games never count toward the unchanged 1000-game selfplay or
2000/500-game stress profiles. The project owner must independently establish
that the executable/net/runtime came from the clean extracted package; this
runner freezes those explicit input paths and hashes without claiming a clean
machine merely from a passing match.

Balanced is the unmodified official `8moves_v3.pgn`: 34,700 complete 16-ply
records. UHO is `UHO_Lichess_4852_v1.epd`: 2,632,036 records. Both books and the
Windows fastchess v1.8.2-alpha executable have pinned physical SHA-256 identities
in the module. They remain external inputs, not part of this repository; the books come from
[official-stockfish/books](https://github.com/official-stockfish/books).
The publisher's normalized SHA-384 acquisition checks are separate from the
unmodified physical bytes frozen for a match. No book is rewritten or shortened.

Fixed-count profiles reject limit/count overrides and require every planned
game. SPRT uses normalized Elo, alpha=beta=0.05, and complete color pairs. H0 is a
failed result; reaching the cap without a decision is `inconclusive` and exits
nonzero. H1 can pass only after every safety, evidence and identity check. These
profiles do not implement the separate release gauntlet or v1.0 strength gate.
`full_release_gate` is always false.

## Plan schema

The JSON object has schema `MouseMatchPlan1` and these exact required fields:

```json
{
  "schema": "MouseMatchPlan1",
  "profile": "selfplay-fast",
  "seed": 20261017,
  "fastchess": "C:/absolute/fastchess.exe",
  "book": "C:/absolute/8moves_v3.pgn",
  "engines": [
    {
      "name": "MouseA",
      "kind": "mouse",
      "executable": "C:/absolute/mouse.exe",
      "model": "C:/absolute/t1-fp16.onnx",
      "source_net": "C:/absolute/t1-256x10-distilled-swa-2432500.pb.gz",
      "args": [],
      "options": {}
    },
    {
      "name": "MouseB",
      "kind": "mouse",
      "executable": "C:/absolute/mouse.exe",
      "model": "C:/absolute/t1-fp16.onnx",
      "source_net": "C:/absolute/t1-256x10-distilled-swa-2432500.pb.gz",
      "args": [],
      "options": {}
    }
  ],
  "runtime_directories": ["C:/absolute/native-directory", "C:/absolute/fastchess-directory"],
  "source_roots": ["C:/absolute/frozen-engine-source"],
  "sha256": {},
  "idle_seconds": 600,
  "total_seconds": 86400,
  "max_output_bytes": 10737418240
}
```

The intentionally empty `options` and `sha256` objects above must be filled; this
schema illustration cannot execute. `max_games` is required only for SPRT;
`nodes` is additionally required for `parity-sprt`. Unknown fields are rejected.
The seed is fixed at 20261017. Paths must be absolute. Engine labels are distinct
stable identifiers; the first engine is the candidate Mouse. Selfplay requires
the same executable, model and options on both sides.

`sha256` maps each absolute path to its actual lowercase physical SHA-256. It
must cover fastchess, the opening book, both executables/models/source nets,
every direct `.dll`, `.dylib` or `.so*` file in every runtime directory, and every
recursive `.cpp`, `.h`, `.cc`, `.hpp` file below each source root. Required
runtime directories include both engine executable directories and fastchess's
own directory; each explicitly selected GPU SDK directory must also be present.
The config, interpreter, runner source and process-helper source are additionally
frozen automatically. Added/deleted inventory entries fail the final check.
Only a separately verified platform pin can enable another real fastchess
binary; the current gate pin is the official Windows x64 release.

All Phase 1 plans require the exact pinned T1 source hash
`bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406`.
Mouse ONNX metadata must bind that source and match the explicit precision;
external tensor storage is rejected. Lc0 loads the exact source file. Input
identities are frozen before ONNX loading and rechecked afterward and after
execution. Output directories cannot overlap immutable sources or runtimes.
Final build/source correspondence is an owner review responsibility: hashing a
source tree does not prove that a particular binary was built from it.

### Explicit engine configuration

Every engine requires string values for `Ponder=false`, `MultiPV=1`,
`UCI_Chess960=false`, `Threads`, and `NNCacheSize`. Timed/selfplay profiles use
`Threads=4`, `MinibatchSize=64`. A zero cache is an unambiguous matched setting;
nonzero Mouse cache is MiB while Lc0 uses entries.

Mouse has no extra argv. Set `WeightsFile` to its model, `MoveOverheadMs=0`,
`MinibatchSize` and `RamLimitMb`. Select `Backend=cpu, Precision=fp32` or an
explicit `cuda|tensorrt, fp16`. GPU plans also require `Device=0`, an absolute
`CacheDirectory`, and absolute existing semicolon-separated `GpuDllDirectories`
whose files are included in the declared runtime inventory. No PATH search or
provider fallback is requested by the runner. The child environment has an empty
PATH. The reviewed Mouse build must have MLH and terminal proof enabled; those
features are frozen source facts, not invented UCI options.

Lc0 requires the exact ordered `args=["--config=", "--show-hidden"]`, explicit
`WeightsFile`, and `Backend=cuda-fp16`. `--config=` disables external config
discovery; `--show-hidden` exposes the explicitly matched root-FPU options to
the strict advertised-option audit. Missing, reordered, duplicate or extra
arguments reject. This does not bypass option advertisement checks. For fixed-node parity Mouse uses its normal `Threads=4`, `MinibatchSize=64`,
`Deterministic=false`; pinned Lc0 declares `Threads=0`, `MinibatchSize=0` (auto).
These literal defaults were verified by CPU-only `uci`/`quit` catalogs and pinned
Lc0 classic/wrapper.cc; auto does not assert a guessed resolved worker/batch size.
Both engines use `NNCacheSize=0`, CPuct=1.745, CPuctBase=38739,
CPuctFactor=3.894, FpuValue=0.330, PolicyTemperature=1.359. Both engines explicitly set
`FpuStrategyAtRoot=same` and `FpuValueAtRoot=1`. The root uses the common
0.330 reduction under `same`; the root-only numeric value is ignored.
This follows the corrected root-FPU contract. Old absolute-root parity
plans are refused rather than silently reinterpreted. Lc0 additionally requires:

```text
SmartPruningFactor=0
FpuStrategy=reduction
FpuStrategyAtRoot=same
FpuValueAtRoot=1
HistoryFill=fen_only
MovesLeftMaxEffect=0.0345
MovesLeftThreshold=0.8
MovesLeftSlope=0.0027
MovesLeftConstantFactor=0
MovesLeftScaledFactor=1.6521
MovesLeftQuadraticFactor=-0.6521
```

These values follow pinned Lc0 v0.32.1 commit
`fd71a2d921b689c5f479d3227c3806c8e272d9c5`.
Numeric spelling differences such as `.33` and `0.330` compare exactly as decimal
values. Lc0 may report nominal nodes+1 even with minibatch1. Actual per-move PGN
nodes are retained; equal nominal caps do not assert identical visits.

## Execution and evidence

The command uses paired colors with default swap, seeded random openings,
concurrency1 for fixed-time profiles and concurrency4 for fixed-node parity,
strict mode, restart off, zero time margin and explicit startup/ping/newgame
timeouts of 300000ms. Fixed nodes never add a time/depth limit. No recovery,
quick mode, tablebase adjudication, ponder, retry, shortened count or fallback
can establish a pass. Fixed-time profiles continue to reject all adjudication.

Parity is bound to these exact native settings:

```text
-concurrency 4
-draw movenumber=34 movecount=8 score=10
-resign movecount=3 score=1000 twosided=true
-maxmoves 192
```

Pinned fastchess match.hpp:60/103 uses twice each native move count: draw
requires16 consecutive qualifying reported plies after its move34 threshold;
two-sided resign requires6 qualifying plies and a negative last moving side.
Opening insertion bypasses MaxMovesTracker (match.cpp:257–269), which updates
only after played engine moves (558–570), so16 book plies +2×192=400 total.
The auditor binds exact captured argv and serialized policy fields, permits only
`normal` or enabled `adjudication` PGN tags, reconciles the exact native reasons
`White wins by adjudication`, `Black wins by adjudication`, or
`Draw by adjudication`, and refuses impossible early/over400-ply outcomes.
This is source-bound native adjudication evidence, not an independent replay of
its score tracker. Syzygy and other outcome/error reasons remain refused.

`fixed_node_settings` records the literal options and native policy; each
`game_records` row retains its PGN `termination`, separate `native_reason`, and
all actual per-ply reported nodes. A nominal `go nodes=800` does not claim equal
visits or an invented numerical overshoot allowance for normal batched/auto
search. The unchanged normalized SPRT bounds are[-10,0], alpha=beta=.05, with a
predeclared even cap (the production plan uses20000); cap without a native
decision remains inconclusive. First valid native SPRT boundary and all late
completed/unpaired/cancelled/skipped tails remain separately reconciled.

Both process streams drain to disk. Finite idle/total watchdogs and output-byte
bounds kill/reap the process tree and join pumps; Windows uses a kill-on-close
Job object and POSIX uses its own process group. Live malformed game records or
error records stop the child. A timeout retains partial raw evidence and fails.
Cold-readiness deadlines remain outside game clocks. A positive watchdog is
mandatory because pure-node search disables fastchess's native move timeout.

`plan.json` records the exact argv and frozen inputs before a child starts.
Execution additionally retains stdout, stderr, protocol trace, full annotated
PGNs, native saved config/statistics, process exit and pump completion, and raw
SHA-256s. Audit reconciles native Started/Finished records, WDL/points/penta,
PGN results/termination, complete game/color pairs, book membership, exact saved
limits/options and UCI handshakes. It rejects abnormal termination, negative
remaining clock, nonfinite/missing annotations, duplicate/missing/extra games,
wrong openings/options, partial summaries, late nonzero exits and mutation.
Pinned fastchess supplies legal move handling; this tool does not claim an
independent chess legality or seeded std::shuffle implementation.

`report.json` is published atomically without clobbering earlier evidence.
Existing directories are refused. A final publication race retains the existing
report and emits a uniquely named failed report. `evidence_valid`, `complete`,
`safety_pass`, `sprt_decision`, `status` and `passed` remain separate. Even a
coherent native summary cannot pass with a missing PGN, handshake or final
`Finished match`. A plan has `native_execution=false`, `passed=false`.

## Paired statistics

Candidate score is based on its actual color in each game. Each pair's average
score is one of x={0,.25,.5,.75,1}; pentanomial counts correspond to those bins.
WL and DD are also retained separately. Let m be the empirical mean, v the
population variance of these paired averages, and P the number of pairs.
The report uses SE=sqrt(v/P), normal 95% score bounds m +/- 1.959963984540054 SE,
Elo=400 log10(m/(1-m)), and
nElo=(m-.5) * (800/ln(10))/sqrt(2v). Elo intervals transform the score bounds;
nElo intervals apply the same linear scaling. These are explicitly labelled
normal approximations with paired empirical variance. Zero variance or infinite
endpoints are represented by null/status rather than fabricated finite values.

This is an original implementation of documented formulas, informed by pinned
[elo_pentanomial.cpp](https://github.com/Disservin/fastchess/blob/f618e34540f94f4719ad3817950618dabe441318/app/src/matchmaking/elo/elo_pentanomial.cpp)
and [elo.hpp](https://github.com/Disservin/fastchess/blob/f618e34540f94f4719ad3817950618dabe441318/app/src/matchmaking/elo/elo.hpp).
Normalized SPRT decisions use the pinned native implementation and reconcile its
actual final decision, printed LLR/bounds and raw statistics. The harness does
not claim an independent GLRT calculation. No upstream source is vendored.

The pinned scheduler has a cap-path trap: its termination branch selects H1
for a non-H0 result when the final match counter is reached, including a
CONTINUE value. The harness therefore requires the actual printed LLR to be
consistent with the accepted boundary, accounting only for two-decimal printing.
An apparent H1 with interior LLR at the cap is failed, retained inconsistent
native evidence. `test_full_declared_sprt_cap_requires_actual_consistent_native_decision`
includes the full 1000-game H1/LLR0 rejection, H0 rejection, valid H1 and undecided
cap cases. See pinned
[roundrobin.cpp:151-156](https://github.com/Disservin/fastchess/blob/f618e34540f94f4719ad3817950618dabe441318/app/src/matchmaking/tournament/roundrobin/roundrobin.cpp#L151).

The V1 user instruction also permits explicit `"concurrency": 1` for the parity
SPRT. Omitting it retains the normal four-game default. Both choices
keep normal engine batching, node limits, SPRT bounds and native adjudication;
all protocol, scheduling and result audits bind to the chosen concurrency.
