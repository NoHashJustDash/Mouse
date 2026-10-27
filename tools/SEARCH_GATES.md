# Search gates

The mate runner executes every original case from `tests/data/search/mates-v1`.
It is separate from fixture generation and does not use Stockfish search or a
Mouse test Backend. Install the worktree tools package normally in the locked
Python 3.12 environment, and run downloaded-file readers with `python -I`.

```powershell
New-Item -ItemType Directory -Force .toolchain/search-gate | Out-Null
python -I -m mouse_tools.search_gate mates --engine build/windows-release/mouse.exe --weights nets/t1.onnx --cases tests/data/search/mates-v1 --backend cpu --precision fp32 --mate1-nodes 128 --mate2-nodes 1024 --mate3-nodes 8192 --batch 8 --ram-limit-mb 64 --timeout 900 --output .toolchain/search-gate/mates-cpu-new.json
```

Use a new report filename for every attempt. CPU requires fp32. For CUDA/TRT,
select `--backend cuda` or `--backend tensorrt`, the intended precision and actual
model, plus repeated absolute `--gpu-dll-dir` arguments and an explicit absolute
`--cache-dir`. CUDA fp16 requires the genuine half graph enforced by the native
Backend. For example, with cuDNN and TensorRT extracted under `C:/NVIDIA`:

```text
C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin
C:/NVIDIA/cudnn-windows-x86_64-9.8.0.87_cuda12-archive/bin
C:/NVIDIA/TensorRT-10.9.0.34/lib
```

The child receives an empty PATH without changing the parent's environment. The
app-local runtime is mandatory; the tool hashes every app-local runtime library
and every library in the selected SDK directories. It never retries a failed
provider or selects another backend. Native Backend validates the actual ONNX
interface/metadata/compute precision; the gate checks the reported exact model
SHA, selected provider and precision against the request.

## Gate contract

The original seven corpus files are pinned to their accepted SHA-256 values.
No subset, extra corpus member, altered best-move set or changed EPD is accepted.
All 18 positions, six per exact mate distance and both colors, remain required.
EPD and JSON must correspond exactly; proof and provenance are the original
independently verified corpus. Node-budget defaults remain 128/1024/8192. Any
explicit alternative budget is recorded in every row and is a different
experiment; it cannot supersede a failed required budget without review.

Every command explicitly selects search threads 1, CPU threads 1, deterministic
mode, batch 8 by default, RAM 64 MiB by default, NN cache 0, exact node budget, fixed
bucket list, no smart pruning, and the pinned cpuct/FPU/temperature parameters.
There is no time/depth limit. `stop_reason` must be `nodes`, with
`nodes == new_playouts == requested budget` and `inherited_visits == 0`.
Memory, time, depth, terminal, external-stop or pruning results fail the gate.
The full 18 cases continue after ordinary per-case failures, without retry.
Preflight failures and cancellation are clearly incomplete failed evidence.

MouseSearch1 must be one complete strict UTF-8 JSON object with every original
field; duplicate keys, unknown fields, nonfinite constants/numbers, oversized or
excessively nested data fail. Version/git, model/input hashes, FEN, deterministic
configuration, RAM charge, numeric bounds, bestmove and complete PV statistics
are checked. Optional `root_moves` is retained and, when present, must list every
unique legal move with finite bounded visits/Q/draw/prior. Optional
`moves_left_utility` must be boolean true for this command's default mode; the
runner does not silently disable that utility or retry a failing mate in another
mode. Legacy output omitting either optional field remains supported.

The selected best move must belong to the original complete root set that
forces mate within the **minimal** fixture horizon. A legal slower mating line
still fails that case. The runner independently checks pawnless PV geometry,
blocking, captures and own-king safety; its legal sets are tested against the
pinned real recorded Stockfish states. Each result records whether its PV ends
in opponent checkmate, even when the selected root move fails the minimal-horizon
criterion. A single terminal PV is not an exhaustive search certificate.
`search_proved_forced_mate` remains false: MouseSearch1 does not expose such a
certificate. A passed report means all 18 root moves match the independent exact
proof within the recorded budgets, not that every v0.4 gate has passed.

## Evidence and failure handling

Engine, model, all corpus files, runtime/SDK libraries and producing Python
sources are frozen before the first corpus load or child execution and checked
after all cases. Runtime and corpus inventory additions/removals also fail.
Reports identify the frozen hashes. Mutation invalidates every pass label while
retaining `observed_passed` and all raw diagnostics; a replacement file's identity
is never substituted into a successful report.

Each row retains its command, timeout, return code, exact captured stdout/stderr,
capture hashes, original JSON, nodes/budget, best/PV/Q/draw/elapsed statistics,
winning-root check and terminal-PV diagnostic. Pipe caps are 64 KiB stdout and 1 MiB
stderr. Reaching a cap is an explicit failure, not a shortened successful result.
Timeout/cancellation kills and reaps the child; bounded drain failures cannot
pass. The per-child timeout defaults to 900 s and can be set explicitly up to one
day for a slow startup without removing positions or lowering a node budget.

Publication uses a flushed/fsynced temporary file plus atomic no-clobber hard
link, requiring a filesystem that supports hard links. Existing reports are
never overwritten, including a report created during the run. Failure details
are kept in the requested new report, or in a unique `.failed-<id>.json` sibling
if the requested name already exists or races. Invalid destinations cannot
replace input files; a filesystem that cannot publish evidence returns failure
rather than switching to a non-atomic writer.

Exit codes: 0 for full 18/18 success, 1 for failure, 2 for invalid CLI/non-isolated
invocation, 130 for interruption. A report's `status`, `passed` and
`hashes_unchanged` must all agree with a complete 18-case result before it is used
as evidence. Unit tests launch only controlled Python children and fake native
transports; their synthetic GPU-labelled replies do not execute GPU code and do
not establish a real CPU/GPU gate.
