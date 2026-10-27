<!-- SPDX-License-Identifier: GPL-3.0-or-later -->
# Chess core acceptance gates

Run from the repository root with the installed `mouse-tools` package and
Python 3.12 in isolated mode. The module refuses a CLI invocation without `-I`.
The development interpreter below is explicitly a development environment;
official Python acceptance remains a separate v0.0 requirement.

After changing the package source, reinstall it in the environment being used:

```powershell
.\.toolchain\python-development\Scripts\python.exe -I -m pip install --no-deps --no-build-isolation ./tools
```

## Full perft gate

```powershell
.\.toolchain\python-development\Scripts\python.exe -I -m mouse_tools.chess_gate perft --engine build/windows-release/mouse.exe --suite tests/data/perft.epd --output build/gates/perft.json
```

Every stored depth runs in its own Mouse process, including the deep counts.
The tracked suite currently contains 126 distinct positions and 392 depth
checks; its independent oracle and regeneration commands are documented in
`tests/data/README.md`. There is no depth cap or subset/full switch. A suite
must have at least 100 distinct positions, Chess960 coverage, and all eight
mandatory counts from the six spec positions. Missing or altered mandatory
counts fail before the first engine command.

This runner protects the approved Mouse baseline: all 126 positions and all 392
depth/count facts. Candidate contracts identify 86 positions and their required
depths; the package also pins 40 generated playout position hashes and depths.
A normalized digest protects every baseline node count:
`a88ad185c25e9abca7026a4e35fd088d17fc4694be4baf8e6861412021380d5f`.
The digest hashes sorted `(position_key(FEN), variant, depth, nodes)` records
serialized as ASCII JSON with separators `(",", ":")`. Position keys ignore
clocks and castling-right order. Comments, row order and identifiers may change;
additional positions and depths are allowed and every stored depth still runs.
Missing baseline depths, removed playouts and changed counts fail before running
the engine. JSON identifies contract `mouse-v0.1-approved-392-v1`.

The 392-fact contract describes this approved fixture, not a new universal
roadmap requirement. Use `mouse_tools.perft_reference verify` for independently
sourced suites. A replaced or shortened baseline is a different gate and must
be reported as such.

Commands use `mouse perft <depth> --fen <fen> [--chess960]`. Each must exit zero
and emit exactly one result line of the form `nodes N time_ms M nps K`; the
node count must equal its EPD fact exactly. Missing output, duplicate totals,
malformed output, wrong counts and timeouts fail the gate. `--timeout` changes
the per-command time limit (default 120 seconds), never the required work.
EPD depths must be 1 through 128, matching the engine limit; counts and FEN
counters must fit unsigned 64-bit integers (fullmove is positive). Decimal
tokens are length-checked before integer conversion. Malformed or oversized
input produces a failed report rather than a Python conversion traceback.

The JSON records the engine path/SHA-256, its captured `--version`, suite
path/SHA-256, every command and its stdout/stderr/exit code/elapsed time,
expected and observed counts, and the number of completed checks. Updates are
atomic after each completed check and at termination. Engine and suite hashes
are checked again at completion. Preserve this file with the benchmark record;
engine-reported `time_ms` and `nps` are perft timings, whereas command elapsed
time also includes process startup. Existing output is replaced when explicitly
selected, so use a new filename for each evidence run.

## Million complete game fuzz gate

Freeze `mouse_tests.exe` and its adjacent runtime DLLs into a dedicated run
directory before starting. On Windows a running executable cannot be replaced;
the frozen copy lets builds continue and preserves the tested binary identity.
Keep the copied files together using the configured application-local runtime
layout. Do not use the live build executable for a background gate while
rebuilding it.

```powershell
.\.toolchain\python-development\Scripts\python.exe -I -m mouse_tools.chess_gate fuzz --tests build/fuzz-run/mouse_tests.exe --workers 6 --output-dir build/gates/fuzz-001
```

The output directory must be new; the runner refuses to overwrite earlier fuzz
evidence. `--workers` is 1 through 16. Each worker is a separate subprocess
running Catch2 filter `[property-long]` with the compact reporter. On Windows,
workers use `CREATE_NO_WINDOW`. No thread invokes Catch2 assertions.

The C++ gate fixes the total to **1,000,000 complete random games**. There is no
production game-count override. Shard `i` of `n` receives environment variables
`MOUSE_FUZZ_SHARD_INDEX=i` and `MOUSE_FUZZ_SHARD_COUNT=n` and owns game IDs
`floor(1000000*i/n)` through `floor(1000000*(i+1)/n)-1`. The C++ test seeds each
game independently and emits its completion marker to process stdout:

```text
MOUSE_FUZZ_RESULT total=1000000 shard=<i> shards=<n> begin=<first> end=<exclusive> completed=<count> plies=<count>
```

A passing shard must exit zero and emit exactly one correctly formed marker
with the fixed total, its assigned index/count/range and all assigned games
completed. Every range must match the partition above; duplicated, overlapping,
partial, missing or inconsistent results fail. The sum of validated completed
games must be exactly one million, and the test binary hash must be unchanged.
The default per-shard timeout is 86,400 seconds; `--timeout` can change it.

`result.json` records the test executable SHA-256, exact shard commands and
ranges, statuses, exit codes, elapsed times and completed counts. Each shard
retains `shard-NN.stdout.log`, `shard-NN.stderr.log` and Catch2's
`shard-NN.catch.log`. The marker is parsed from process stdout separately from
Catch2's reporter file. `reported` holds parsed marker fields, including partial
counts, when available; `completed` and root `completed_games` count only
validated passing shards. An interrupted shard may have completed games that
have not emitted a final marker, so zero validated games does not assert that
no work occurred.

On any failure, timeout or Ctrl+C, the runner terminates and reaps all children
it started, preserves logs, and writes a failed/interrupted result. Partial
progress is never a pass. Exit codes are 0 for a complete pass, 1 for failure,
and 130 for interruption. A forced OS kill or power loss cannot produce a
final result; the last atomic checkpoint remains `running` and is not evidence
of a pass.

## Runner verification

```powershell
.\.toolchain\python-development\Scripts\python.exe -I -m pytest tools
.\.toolchain\python-development\Scripts\python.exe -I -m ruff check --no-cache tools
.\.toolchain\python-development\Scripts\python.exe -I -m mypy --strict --config-file tools/pyproject.toml tools
```

Protocol tests use explicitly test-only child scripts or an injected command
runner. They exercise failures and fixed million-game completion markers
without running a shortened product gate. Passing those tests establishes the
runner's behavior; actual perft/fuzz gate passes require the real binaries and
retained evidence. Sanitizers and baseline speed are separate v0.1 obligations.
