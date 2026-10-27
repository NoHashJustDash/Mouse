# Pinned default UCI score fixtures

These are original Mouse probes and generated outputs of the official Lc0
v0.32.1 executable. No upstream implementation source is vendored here.
The exact original research directory is retained under `reference/`:403 files
plus its pinned manifest, including all72 full argv/command/stdout/stderr/exit
transcripts, replay inputs, the original research scripts and the source URL/hash
manifest. Archived scripts are byte-preserved research evidence; use the new
portable `reference.py` for maintained audit/regeneration.

- Source commit: `fd71a2d921b689c5f479d3227c3806c8e272d9c5`.
- Official Windows executable SHA256:
  `e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b`.
- Publisher-verified GPU release ZIP SHA256:
  `8d0ce17676eb15e303bea9e790742d31c94ce5d24107f6187adc58e820f6d2f7`.
- Original evidence manifest SHA256:
  `12e61ba1e2999d276103a95b8d9dfb78389b4609f70517fd8e50ca39c09b0b28`.

`cases.inc` contains28 exact float32 Q/D bit patterns and literal observed cp/WDL
integers from `reference/score-cases-v2.json`; every case was run twice. The C++
test consumes these literal observations rather than computing its expectations
from the formatter. The offline audit verifies this table against both raw
transcripts and the pinned complete original file manifest. It rejects missing,
added, modified or shortened evidence. The controls retain explicit WDL_mu,
legacy centipawn, raw W-L, forced mates for both colours, and default WDL visibility.

Run these commands from the repository root with Python3.12 (standard library only):

```powershell
python -I -B tests/data/uci/score-v1/reference.py audit
python -I -B tests/data/uci/score-v1/test_reference.py
python -I -B tests/data/uci/score-v1/reference.py regenerate --lc0 <workspace>/.toolchain/reference/lc0/lc0.exe --output .toolchain/score/new-reference-run
```

Regeneration requires the exact pinned official executable and a new output
directory outside these fixtures. It never downloads files or selects an
automatic/GPU backend. It reuses the original exact replay inputs and all72
protocol plans. Record/replay explicitly selects both the CPU random parent and
`random(seed=0)` child; mate controls select CPU trivial. One thread, minibatch1,
no prefetch/cache/task workers, empty weights/config and the original score-option
controls are preserved. The protocol waits for uciok/readyok/bestmove, enforces
bounded waits, kills and joins on failure, and retains transcripts even on error.
It compares every cp/mate/WDL/bestmove payload exactly and verifies identical
seeded root-record bytes plus executable/fixture identities before and after.
Timing fields are diagnostic and are not compared. No tolerance is used.

The default mapping intentionally differs from the legacy90*tan formula, and
inverse reporting rescaling still runs at the default ratio1/difference0. cp uses
truncation toward zero; integer WDL uses half-away rounding and the pinned
negative-draw correction. The +/-Q half ties are asymmetric. Endpoints ordinary
Q=+/-1 emit cp+/-12800; the formatter does not invent mate distances. Callers
identify proven terminals separately. See the score research/implementation
handoff for equations, source references, numerical discontinuities and evidence.
