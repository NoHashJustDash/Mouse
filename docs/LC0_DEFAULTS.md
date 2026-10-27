# Lc0 v0.32.1 defaults audit

Reference: official Windows lc0 v0.32.1, executable SHA256
`e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b`.
The official tag resolves to `fd71a2d921b689c5f479d3227c3806c8e272d9c5`.
Mouse follows lc0's search defaults, including the pending-child correction,
the legacy time manager and smart pruning, all re-implemented from lc0's source.
After a measured 1 ms time loss at the raw GUI clock cap, Mouse also accounts for
the cost of publishing its final output, without changing ordinary legacy
allocations or constants; a 96-case timing capture passed at the original 15 ms /
10 percent tolerance. This file lists each default and every place Mouse differs.
Strength results are in the README.

The actual reference commands were:

```text
lc0.exe --config= --show-hidden --help
lc0.exe --config= --show-hidden
stdin: uci
       quit
```

Both exited0 without initializing an inference backend. `--help` rounds real
numbers to two decimal places; the native UCI catalog above exposes the exact
defaults (for example CPuct1.745000 rather than the help display1.75).

| Option | lc0 default | Mouse equivalent | Same? | Evidence |
|---|---|---|---|---|
| NNCacheSize | 2,000,000 **entries** | 2,000,000 **entries**, allocated on demand | Yes, capacity; 200-position deterministic equality passed | `LegalCache`: 128-bit fingerprints cover namespace, all input bits and ordered legal indices. Only legal logits and WDL/MLH are retained. |
| OutOfOrderEval | true | Terminal and cached leaves back up while gathering in normal mode | Partial; deterministic mode keeps batch barriers | `gather_request` / `apply_cached`; source `ProcessPickedTask` and `GatherMinibatch`. Cache-only gathers continue without empty provider calls. |
| MaxOutOfOrderEvalsFactor | 2.400000 | Fixed truncation of 2.4 times effective minibatch | Default implemented; no UCI override | Source `classic/search.h`, `max_out_of_order_`. |
| MaxCollisionEvents | 917 | Fixed 917-event cap per gather | Mouse enforces requested cap | Pinned source declares this option but does not consume its getter in `search.cc`; Mouse bounds reservation storage with it. |
| MaxCollisionVisits | 80000 | Fixed 80000-visit maximum | Default implemented | Weighted ancestor reservations are cancelled, never evaluated or counted as real visits. |
| MaxCollisionVisitsScalingStart | 28 | 28 | Yes | `collisions.cpp`; lc0 CalculateCollisionsLeft. |
| MaxCollisionVisitsScalingEnd | 145000 | 145000 | Yes | Same. |
| MaxCollisionVisitsScalingPower | 1.250000 | 1.25 | Yes | Same; minimum of tree size and remaining-playout estimate. |
| Pending-edge PUCT eligibility | Pending children remain scored; a selected pending leaf collides | Stage1 skipped pending children; 89476a5 retains eligibility | Corrected; fresh Stage3.1 H1 on11da0ea | lc0 `classic/node.h` EdgeAndNode::GetU and `node.cc` TryStartScoreUpdate. Controlled regression RED:19 zero-prior alternatives each received1 visit. Fix:130 search/session cases pass on Windows and TSan. |
| MaxPrefetch | 32 | Trial removed after measured regression | No | Source port and its regression passed, but five-position geometric mean fell .450 to .406; no prefetch retained. |
| StickyEndgames | true | Terminal proof propagation, ancestor value replacement, and proven-win/ordinary/proven-loss final ordering | Partial; corrected terminal-choice gate H1 on11da0ea | `proof.cpp`, `search.cpp::refresh_proof_locked`/`backup_locked` and `selection.cpp::prefer_final_move`. Actual lc0 chose mate2visits over stalemate16; committed terminal_choice_v1 golden and retained-tree regression. Mouse compares established mate strategy upper bounds, not exact DTM; proven draw distance is not stored, so those ties retain visits/Q/prior. lc0 has additional bounds/statistic adjustments. |
| Tree reuse | Enabled by verified continuation/history | `advance`, verified full root/history identity; unrelated roots reset | Partial | `SearchEngine::advance` and `set_position`. Mouse discards the retained subtree when contextual draw evidence cannot survive; lc0 selectively reverts twofold terminal visits with EnsureNodeTwoFoldCorrectForDepth. |
| SmartPruningFactor | 1.330000 | Configurable1.33; source division and conservative NPS forecast | Source port implemented; gates open | `engine/search/pruning.cpp`; startup200ms after first evaluation,300-node bias, remaining-time/node minimum; factor0 disables. Native regression was RED at172ms despite factor0, now honors the300ms deadline. |
| SmartPruningMinimumBatches | 0 | Configurable0; source startup/early-win/resignability ordering | Source port implemented; gates open | Native option name verified in `classic/stoppers/common.cc`; `search/pruning.cpp` and fake-clock tests. Root visit statistics enumerate ALL legal edges as lc0 does, even with searchmoves. |
| TimeManager | legacy | Stateful native curve, first-book-move bonus and saved-time bank, default slowmover/immediate-use1 | Source port implemented; gates open | `engine/uci/time.cpp`;96 actual-lc0 allocation captures within logger resolution, bank/reset/movetime regressions. Actual269730a,d5f666f,and corrected b324e2f captures each passed96/96 at the original tolerance; the [0,5] gainer was stopped and accepted at 114 games. |
| MoveOverheadMs | 200 | 30 | No | Both native option catalogs. Gate matches explicitly select0 for both; this is a documented delivery reserve, not search tuning. |
| MovesLeftMaxEffect | 0.034500 | 0.0345 | Yes, default constant | `engine/search/moves_left.cpp`; existing source-based utility regressions. |
| MovesLeftThreshold | 0.800000 | 0.8 | Yes, default constant | Same. |
| MovesLeftSlope | 0.002700 | 0.0027 | Yes, default constant | Same. |
| MovesLeftConstantFactor | 0.000000 | 0 | Yes, default constant | Same. |
| MovesLeftScaledFactor | 1.652100 | 1.6521 | Yes, default constant | Same. |
| MovesLeftQuadraticFactor | -0.652100 | -0.6521 | Yes, default constant | Same. |
| CPuct / CPuctAtRoot | 1.745000 /1.745000 | 1.745 for both | Yes, defaults | `engine/search/selection.h`; root-own-params is false in lc0. Formula parity is tested; lc0 FastLog and Mouse log1p are not represented as bit-identical. |
| CPuctBase / CPuctBaseAtRoot | 38739.000000 /38739.000000 | 38739 for both | Yes, defaults | Same. |
| CPuctFactor / CPuctFactorAtRoot | 3.894000 /3.894000 | 3.894 for both | Yes, defaults | Same. |
| RootHasOwnCpuctParams | false | Shared parameters at root | Yes | Native catalog; selection configuration. |
| FpuStrategy / FpuValue | reduction /0.330000 | reduction /0.330 | Yes, defaults | `engine/search/selection.cpp::first_play_urgency`. |
| FpuStrategyAtRoot / FpuValueAtRoot | same /1.000000 | same /1.0 | Yes, defaults | Same: `same` inherits ordinary reduction; the root override is used only for an explicit root strategy. |
| PolicyTemperature | 1.359000 | 1.359 | Yes, default | `engine/search/prior.cpp`; legal-policy parity against actual lc0 was checked independently. |
| Threads / MinibatchSize | 0 /0, backend auto (CUDA: 2 search workers; batch 2 x SM count, 96 on RTX 5070) | 2 /0 (auto: 2 x SM count on GPU, 64 on CPU) | MinibatchSize: Yes (same rule, network_cuda.cc). Threads: Mouse threads are gather workers, not lc0 search workers; 2 measured best (2026-10-18 sweep) | Sweep 2026-10-18: batch 64 0.847x lc0, 96 0.950x, 96+2 threads 0.986-0.991x (idle machine, a714fff). |

The original 256 MiB raw cache held 28,876 entries. Its replacement stores
unmodified legal logits, WDL and optional moves-left output. All physical
1,858-logit evaluations are validated before extracting legal values. Cache hits
publish the same legal output without synthesizing a full NNOutput.

Pinned lc0 `src/neural/memcache.cc::ComputeEvalPositionHash` calls
`pos.pos.back().Hash()`. Despite its nearby comment, `src/chess/position.cc`
includes the current board AND repetition count. It omits earlier history and
the rule-50 counter. Mouse retains full encoded-input identity by design; a transposition with different encoded history cannot reuse a value.
This is a semantic difference, not just a storage implementation difference.
Sources: https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/memcache.cc
and https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/chess/position.cc.

The first 2,000,000-entry implementation preallocated about 8.03 GB. Mouse replaced it with on-demand entries, variable legal payloads and a
128-bit fingerprint. The key covers every input mask and float bit (including
signed zero and empty-plane values), ordered legal indices and the immutable
model/provider namespace. Unlike the former implementation, this relies on
fingerprints rather than full key comparison; collision risk is probabilistic.
New-entry allocation may fail explicitly and preserves existing entries. Hits
and same-key updates retain their allocation-free contract. Clearing releases
entries but retains hash buckets. Capacity is unchanged and outside RamLimitMb.

The cache change was verified by deterministic output equality over 200
positions at 800 nodes, plus memory and speed measurements, with CPU inference
for both builds. Both captures completed: 200 positions, 400 cold/warm records, bit-exact
bestmove, visits, Q and root statistics. CPU private commit after exactly 20,000
visits was 8,838,119,424 -> 796,471,296 bytes (only smart pruning disabled).
No Elo gain is claimed for the memory change.
The existing CPU hit benchmark was run in three interleaved process pairs.
Median legal ns/row, old to new: batch 1, 749.780 to 1639.320; batch 8,
732.166 to 1664.640; batch 64, 737.296 to 1688.240. This is a lookup-cost
regression, not a measure of search speed.

Search parity with lc0 was tested with fixed-node SPRTs (bounds [-10, 0]). The
first normal-batched run failed after 150 games; after the pending-edge
correction it passed after 648 games (101 wins, 488 draws, 59 losses, LLR 2.95).
The proven-terminal final-choice correction then passed a fresh run: 480 games,
79 wins, 362 draws, 39 losses, LLR 2.95. The legacy time manager's elapsed-time
check passed 96/96. The time-management match against the previous build was
stopped early at 114 games: +58.5 +/- 37.4 Elo, zero time losses, pentanomial
[0, 8, 26, 19, 4].

Pinned primary sources: [lc0 search](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/search.cc),
[parameters](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/params.cc),
[legacy TM](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/stoppers/legacy.cc),
[stoppers](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/stoppers/stoppers.cc),
[node selection/update](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/search/classic/node.cc).
Legal-cache pattern: [lc0 memcache](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/neural/memcache.cc).
