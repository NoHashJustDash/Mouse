# Fixed standalone bench corpus v1

`positions-v1.tsv` contains 16 original hand-authored analysis cases. No positions,
benchmark lists or implementation were copied from Lc0 or another engine. The
standard opening move sequences are independently replayed by Mouse's legal move
parser. Other FENs are legal analysis setups, not claims of a reconstructed game.

Each non-comment row has exactly three tab-separated fields: stable case ID,
initial six-field FEN, and chronological standard-UCI moves (possibly empty).
Empty move fields retain their trailing tab. The production command embeds these
same immutable records; native tests compare every record to this tracked file.
The corpus version is `MouseBenchPositions1`. Its canonical SHA-256 is:

```text
24f1e92c3d40b903dbbb6952b6bbfa330640342a45701471216a6db4ebdf4515
```

Hash UTF-8 `MouseBenchPositions1\n` followed by every data row, in order, with one
LF after each row. Comments and physical CRLF/LF differences do not enter the
canonical hash. Changing cases/order/history requires a new reviewed corpus
version; there is no CLI corpus path or case-count override.

Coverage includes start position, queen-pawn development, white/black castling,
white/black en passant, white/black promotion, pawnless material, black pawn
endgame, halfmove clock99, and complete opening/repetition/pawn-push/capture
histories. Tests check actual castle/EP/promotion availability, pawnlessness,
clock and repeated-history state, rather than relying on the IDs alone. The
full-history cases are not replaced by their final FENs: NN input and input SHA
retain the complete explicit history. All 16 roots are nonterminal.

After main/CMake integration, example usage is:

```powershell
mouse bench --weights nets/t1.onnx --backend cpu --precision fp32 --deterministic --nodes 800
```

Weights/provider/precision are explicit. Default nodes per case is800 (12,800
completed visits total); `--nodes` selects another recorded fixed budget.
`--fen`, `--moves`, `--movetime`, `--depth`, `--chess960`, `--positions` and
`--corpus` are rejected. Standard search/backend configuration is reused.
Insufficient RAM, NN failure, bad output, illegal result/PV, unexpected early
stop or any model/backend identity change fails the whole command; no successful
signature is printed from a partial case set.

Every case starts with a fresh tree, full Position/EncodingHistory and fresh
bounded exact-input NN cache. Backend session construction and its normal warmup
occur once, separately from measured search. Per-case `setup_ns` covers FEN and
history replay, cache/tree construction and root setup. `search_ns` measures only
`SearchEngine::run`, including its worker startup and drain; validation and owner
destruction occur afterward. Aggregate nps is actual completed nodes divided by
the sum of these measured search intervals. Immutable AttackTables construction,
identity audits, setup, validation and cleanup are included in `bench_wall_ns`.
`backend_setup_ns` records the separately excluded session creation/warmup.
`command_wall_ns` adds parsing/path resolution/backend setup to bench wall, up to
the completed report; report formatting/stdout I/O and final backend destruction
are outside that field. No measured component is silently added to search nps.

Success emits one complete `MouseBench1` JSON object containing build/runtime,
model/provider/device/cache/config identities, corpus/input hashes, histories,
per-case counts/Q/draw/best move/legal PV, setup/search timing and managed RAM,
followed by a final `nodes N nps M` signature. The signature is omitted if report
publication fails. OS stream failures can leave partial bytes and return nonzero;
case/evaluation failures publish no report bytes at all.

This corpus and injected native tests provide functional evidence only. Actual T1
measurements and a matching actual-Lc0 corpus/config/cache/warm/timing comparison
remain parent-owned gates. No throughput ratio, speedup or Elo claim is implied.