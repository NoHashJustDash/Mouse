# Original UCI protocol-v1 contract

Mouse-owned normative expected transcripts, authored from the approved v0.5 UCI
design. This is not fabricated binary output or an upstream transcript. Each JSON
step records an exact command, stable literal response, or strict constraint on
volatile native output. Raw actual stdout/stderr, sent bytes, exit status and
artifact hashes are captured only by a real run, separately in its report.

The fixed 21 cases cover every required command and option, readiness during
search, finite limits, publication, infinite/ponder, stop activation eight times,
quit/EOF, invalid input/history, one-MiB framing, deterministic repeat, Chess960,
terminal roots and fatal inference while stdin remains open. No case may be
removed or resampled. The optional explicit failure model is required to claim
complete protocol-v1 coverage; otherwise the report is incomplete and cannot pass.

Only exact tracked tiny `valid.onnx` / `search-nonfinite.onnx` fixture hashes are
accepted. Every inference request explicitly selects CPU/fp32, one CPU/search
thread, deterministic batch one, 64 MiB tree RAM and NN cache zero. The bare
startup cases exercise no-argument UCI and missing weights without loading a
backend. They set CPU/fp32 before readiness/search when needed. No T1, GPU or
allocation-stress graph is executed.

Node-limited protocol checks record actual nodes between one and their requested
maximum; normal UCI smart pruning can finish early. These checks do not establish
the separate unchanged 128/1024/8192-node exact mate gate. Time/NPS are retained
and bounded but are excluded from deterministic equality. Info fields, score cp,
WDL permille, PV coordinate grammar, restricted roots, MultiPV uniqueness, and
exactly one bestmove per accepted search are checked. Full independent PV
legality/self-play/match/sanitizer gates remain separate required evidence.

Commands and placeholders are expanded by the original stdlib-only
`python -I -m mouse_tools.uci_gate --engine <exe> --weights <valid.onnx>
--failure-weights <search-nonfinite.onnx> --output <new.json>` runner.
No captured result is created by editing this contract.

The senior-corrected root-FPU contract adds two typed options in source order,
immediately after FpuValue: FpuStrategyAtRoot (combo: same, reduction,
absolute; default same) and FpuValueAtRoot (finite string-real; default1).
The handshake transaction explicitly sets same/1 and re-reads both.
All original21 case IDs and every other case remain unchanged. These are
normative additions, not captured native output. Older21-case reports
retain their original source/contract identities and are not rewritten.

The finite-limit case explicitly sets MoveOverheadMs0: its original movetime25
then retains25ms usable after the native legacy port started subtracting the
configured reserve. The ponder-hit case waits for initial model readiness before
starting ponder; its existing isready DURING ponder is retained. These cases
require evaluated PVs, so cold-start emergency publication is checked separately
by the first-go/load-blocking/deadline session regressions. All21 cases, original
clock values, node bounds, PV/evaluation assertions and eight stop races remain.
