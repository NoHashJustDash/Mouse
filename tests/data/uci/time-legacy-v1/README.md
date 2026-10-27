# Actual lc0 v0.32.1 legacy allocations

`allocations.inc` contains96 native logged allocations, captured with the
publisher-verified official Windows lc0 v0.32.1 binary SHA256
`e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b`.
Source tag resolves to `fd71a2d921b689c5f479d3227c3806c8e272d9c5`.

Exact capture command on this machine:

```powershell
<workspace>/.venv/Scripts/python.exe -I .toolchain/v1-tm/collect-reference-allocations.py
```

The script uses the existing bounded `mouse_tools.uci_gate.UciProcess` and runs:

```text
lc0.exe --config= --show-hidden --backend=random --backend-opts=uniform=true,delay=1 --threads=1 --minibatch-size=1 --nncache=0 --move-overhead=0 --smart-pruning-factor=0 --weights=<workspace>/.toolchain/downloads/t1-256x10-distilled-swa-2432500.pb.gz --logfile=<absolute reference-allocation-v1.log>
```

After `isready`, each row sends `ucinewgame`, the starting-board FEN below,
then `go nodes 1 wtime <clock> btime <clock> winc <inc> binc <inc>` and optional
`movestogo 40`. Angle-bracket values come from the grid below.
The fixed node count is ONLY for the allocation-log probe. This is not the
elapsed-time gate and does not claim to measure strength or use the GPU.

The Cartesian grid is clocks1000/10000/60000/300000ms, increments0/1%/10%,
movestogo absent/40, and internal plies1/20/60/120, in that order. Each row is
`clock,increment,movestogo-or0,internal-ply,native-logged-ms`. The native logger
prints6 significant digits; the scalar unit allows at most1ms against that
printed value, capped by the remaining clock. No elapsed-time tolerance changed.

Pinned v0.32.1 `Position::FromFen` passes the FEN fullmove field directly into
its internal `ply_count_`; supplied UCI moves increment that once per ply.
The actual source and a native allocation probe confirmed this quirk. Thus FEN
`rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 <internal-ply>`
selects the requested native time-curve point. Deriving standard2*(fullmove-1)
would disagree with the actual reference. Mouse tracks base fullmove plus the
number of supplied moves for this source-compatible time context.

The native96-case probe exited0 with joined transport and96 allocation logs.
Additional overhead30ms probes produced209.314/897.059/942.45/9150(capped3000)/
917.745/928.725/931.47/179.412ms, used in the explicit active-side/unit fixtures.

The separate real elapsed gate uses `python -I -m mouse_tools.time_parity capture`
with kindlc0 or mouse. Both disable smart pruning and subtract zero overhead.
The reference uses its CPU-only random backend; Mouse uses the actual T1FP32
CPU model. This isolates timing without taking the GPU away from the parity SPRT.
Both play through their real UCI frontends and native clock machinery; no node
limit is set in this gate. Each complete table needs all96 normally exited,
joined samples and stable binary/model hashes. Incomplete captures retain only
`.pending` files. `time_parity compare` enforces max15ms/10% per row, unchanged
from V1_PLAN. The elapsed table is still being collected; this fixture alone
is not its approval or a completed TM gate.
