# Mouse

Mouse is a UCI chess engine that plays with Lc0's neural network
**T1-256x10-distilled-swa-2432500**. It is written from scratch in C++20 and runs
the network through ONNX Runtime on an NVIDIA GPU (TensorRT or CUDA) or on the CPU.

Version 1.0.0 aims to play the T1 network the way lc0 does: the same Monte Carlo
tree search with lc0's default parameters, smart pruning and legacy time manager,
implemented independently and checked against lc0 itself.

Mouse was built with the help of AI coding assistants; see
[Made with AI assistance](#made-with-ai-assistance) and [Credits](#credits).

## Strength

All results are from one machine (GeForce RTX 5070, Ryzen 7 7800X3D), playing one game
at a time with pondering off and Mouse at its default settings.

**Against a rated reference engine.** Motor 0.9.0 is rated 3694 ± 9 on the
[CCRL Blitz list](https://computerchess.org.uk/404/) (single CPU). Motor ran single
threaded with 128 MB hash. The openings were the balanced `8moves_v3` book, each
played once with each colour. Games were adjudicated as drawn after move 40 once
both engines scored within ±10 cp for 8 moves, and as won once both agreed on at
least 1000 cp for 4 moves.

| Time control | Games | Mouse vs Motor | Elo difference | Rating estimate |
|---|---|---|---|---|
| 10+0.1 | 100 | +20 −4 =76 | +56 ± 27 | 3750 ± 28 |
| 60+0.6 | 60 | +6 −3 =51 | +17 ± 34 | 3711 ± 35 |
| 180+2 | 22 of 40 planned | +3 −0 =19 | +48 ± 47 | 3742 ± 48 |

The rating estimate is Motor's CCRL rating plus the measured difference. It is our
own estimate, not a CCRL rating: CCRL plays on different hardware, scales its time
control to it, and rates against many opponents. Use it as a reference point.

Margins are 95% confidence intervals. At 60+0.6 the margin includes zero and at
180+2 it only just excludes it, so the longer games show Mouse at least at
Motor's level but not by how much. The 180+2 match was stopped after 22 of 40
planned games to make the release; Mouse had not lost a game.

**Against lc0 with the same network.** lc0 v0.32.1 running T1, 600 games at
10+0.1, UHO openings played once with each colour, both at default settings:

| Match | Result |
|---|---|
| Mouse vs lc0 | **−59.6 ± 12.3 Elo** (+70 −172 =358) |
| Search speed | 0.89x lc0's nodes per second (geometric mean of five positions) |

Mouse is weaker than lc0 with the same network. The main known cause: when the
search tree contains a repeated position, Mouse discards the whole tree before its
next move instead of keeping it as lc0 does. In self-play that happened before
about 40% of searches. Mouse also does not yet skip root moves that can no longer
become best before time runs out. Both are being fixed for the next release.

## Made with AI assistance

Most of Mouse's code, tests and documentation were written by AI coding
assistants: Anthropic's Claude, OpenAI's Codex, Google's Gemini and Zhipu AI's
GLM. The project's author directed the work: set the goals, made the design
decisions, and ran the tests and matches. Changes written by one assistant were
reviewed by another.

Nothing was taken on an assistant's word. The move generator is checked against
reference perft counts, the network input encoding and policy map are checked
position by position against lc0, and each release is played against lc0 and a
rated engine before it ships (results above). Mouse is still a young engine, so
please report anything that looks wrong.

## Credits

- [Lc0 (Leela Chess Zero)](https://lczero.org/): the T1 network Mouse plays with,
  and the design Mouse follows: the network weights format, input encoding and
  policy map, and the search (PUCT selection, default parameters, moves-left
  utility, smart pruning and time manager). Mouse's code is its own
  implementation; where it follows lc0, the comments cite the lc0 source file
  (v0.32.1 or commit `fd71a2d`). The reference fixtures under `tests/data/nn/` were produced with
  lc0, and lc0 v0.32.1 is the engine Mouse is measured against. Lc0 is licensed
  under the GPL-3.0, like Mouse. Mouse is not affiliated with or endorsed by the
  Lc0 project.
- [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) by DeepMind:
  the approach behind both lc0 and Mouse, a Monte Carlo tree search guided by a
  neural network.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) by Microsoft (MIT
  licence), bundled with the release.
- [tl::expected](https://github.com/TartanLlama/expected) by Sy Brand (CC0),
  compiled into Mouse.
- [Catch2](https://github.com/catchorg/Catch2) (Boost Software Licence), used by
  the unit tests.
- NVIDIA CUDA, cuDNN and TensorRT, installed separately and not bundled.
- The Python tools use [NumPy](https://numpy.org/), [ONNX](https://onnx.ai/),
  [protobuf](https://protobuf.dev/) and ONNX Runtime, installed from PyPI.
- [Stockfish](https://stockfishchess.org/) 17.1, used only to verify the perft
  node counts in the move-generator tests. Some test positions come from
  Stockfish's perft tests and the
  [Chessprogramming wiki](https://www.chessprogramming.org/Perft_Results).
  Mouse contains no Stockfish code.
- [Motor](https://github.com/martinnovaak/motor) by Martin Novák, the rated
  reference engine for the strength estimate.
- [fastchess](https://github.com/Disservin/fastchess) by Disservin, used to run
  the matches, with Stefan Pohl's UHO openings and the `8moves_v3` book from
  [official-stockfish/books](https://github.com/official-stockfish/books).

## Quick start (Windows x64, NVIDIA GPU)

1. Install the NVIDIA driver, CUDA 12.8, cuDNN 9.8 and TensorRT 10.9
   (versions and links in [docs/TOOLCHAIN.md](docs/TOOLCHAIN.md)).
2. Unzip `mouse-1.0.0-windows-x64.zip`.
3. Download and convert the T1 network (needs Python 3.12). From the unzipped folder:

   ```powershell
   py -3.12 -m venv .venv
   .venv/Scripts/python -I -m pip install --only-binary=:all: --require-hashes -r tools/requirements-converter-windows-cpu.lock
   .venv/Scripts/python -I -m pip install --no-build-isolation --no-deps ./tools
   .venv/Scripts/python -I -m mouse_tools.setup_net --directory nets
   ```

   The download checks the file's size and SHA-256 before converting it.
4. Add `mouse.exe` to your chess GUI as a UCI engine. Set `WeightsFile` to the
   converted `.onnx` file and `GpuDllDirectories` to your CUDA `bin`, cuDNN `bin`
   and TensorRT `lib` folders, separated by semicolons.

The first start builds a TensorRT engine for your GPU, which takes about two
minutes; later starts take a few seconds. The network and NVIDIA libraries are not
included in the download.

## Windows build

Install system Git for Windows, python.org Python 3.12, and Visual Studio 2022
Build Tools with Desktop development with C++, a Windows SDK, CMake >= 3.28,
and Ninja. From a clean clone in a directory outside OneDrive, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build.ps1
```

The build script acquires the pinned vcpkg baseline and official ORT SDK, checks
download hashes, builds, and runs CPU tests. It does not install system tools.
CUDA compilation is not part of the engine build; neither NVCC nor MSVC 14.38
is required. See [toolchain](docs/TOOLCHAIN.md) for the verified GPU stack.

For an already initialized developer environment:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release
ctest --preset windows-release
.\build\windows-release\mouse.exe --version
```

`--version` prints the actual loaded ORT version alongside the version, Git
revision, build type, and compiler. A missing runtime or any version other than
1.22.0 fails clearly. The ORT runtime and provider libraries are copied beside
the executable. Windows loading excludes the current directory and PATH.
GPU dependencies must be supplied through explicitly registered SDK directories;
the tracked GPU test documents and exercises this mechanism.


## Chess core commands

```powershell
.\build\windows-release\mouse.exe perft 6
.\build\windows-release\mouse.exe perft divide 3 --fen "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
cmake --preset windows-pext
cmake --build --preset windows-pext
ctest --preset windows-pext
```

Magic sliding attacks are the default. The PEXT build requires BMI2 and fails
clearly on an unsupported CPU. Chess960 positions with nonstandard castling
rights require `--chess960`; divide then uses king-to-rook move tokens.
Malformed FEN, options and unsupported depths return errors. Full perft and
million-game commands are documented in [tools/CHESS_GATES.md](tools/CHESS_GATES.md).

## T1 conversion and inspection

Install the platform converter lock and package as documented in
[tools/CONVERSION.md](tools/CONVERSION.md), then run:

```powershell
.\.venv\Scripts\python.exe -I tools/fetch_net.py
.\.venv\Scripts\python.exe -I -m mouse_tools.convert nets/t1-256x10-distilled-swa-2432500.pb.gz nets/t1.onnx --sha256 bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406
.\build\windows-release\mouse.exe describe-net nets/t1.onnx
```

The engine checks metadata and the actual dynamic tensor interface before
accepting a model. Inspection uses an explicit CPU session and does no inference.
The converter supports the actual T1 attention architecture: 20,198,188 learned
parameters, ten encoder blocks, smolgen, policy, WDL and moves-left heads.
Unsupported architectures or encodings fail clearly. Input/history/policy
reference commands and provenance are in [tests/data/nn/README.md](tests/data/nn/README.md).

## Inference

Explicit CPU, CUDA and TensorRT evaluation, full batch throughput sweeps, runtime
configuration and the fixed50 parity command are documented in
[docs/INFERENCE.md](docs/INFERENCE.md). FP16 exports use `--precision fp16`.

## Play through UCI

Convert the downloaded source net to the precision you intend to run. For example,
create a GPU model after installing the locked converter environment:

```powershell
.\.venv\Scripts\python.exe -I -m mouse_tools.convert nets/t1-256x10-distilled-swa-2432500.pb.gz nets/t1-fp16.onnx --sha256 bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406 --precision fp16
.\build\windows-release\mouse.exe uci --weights nets/t1-fp16.onnx --backend tensorrt --precision fp16 --gpu-dll-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin" --gpu-dll-dir "<cuDNN bin>" --gpu-dll-dir "<TensorRT lib>" --cache-dir "<absolute writable cache directory>"
```

Use the actual SDK directories from your installation. All three explicit GPU
directories and the ONNX model must be valid. The engine reports errors instead
of selecting another model, precision or provider. TensorRT builds its disk cache
on first load; allow up to five minutes for initial readiness in a GUI or runner.
It reports cold-build progress as `info string` messages. On the RTX 5070 with
ORT 1.22.0, TensorRT 10.9 and T1 FP16, a fresh cache took 119.39 seconds; the
verified warm EPContext cache reached `readyok` in 4.19 seconds. These measured
times depend on hardware and system load. Cache identity changes require a new
cold build. Keep `context.onnx`, engine and timing payloads together in their
cache namespace; missing or altered files fail explicitly. The model path is
still required and validated on every load. A first `go` includes any remaining
loading time in its move deadline, so complete `isready` before starting a game
when possible.

For CPU analysis, use the FP32 conversion from the earlier example:

```powershell
.\build\windows-release\mouse.exe uci --weights nets/t1.onnx --backend cpu --precision fp32 --batch 1
```

CPU analysis uses a one-position batch to keep accepted inference work small.
Timed bestmove publication is independent of an accepted provider call. The
owned worker drains that call and joins it at shutdown; late failures are fatal.
Shutdown can therefore wait for a provider that has not yet returned.
The publisher accounts for measured output and wake latency before the raw GUI
clock expires, including with MoveOverheadMs=0. Ordinary legacy search allocations
stay intact when that clock has room; this does not change legacy TM constants.

The default cache is per user: Windows uses %LOCALAPPDATA%/Mouse/cache.
Linux uses $XDG_CACHE_HOME/Mouse/cache, or $HOME/.cache/Mouse/cache when XDG is
unset. An explicit absolute CacheDirectory overrides this default. The Windows
TensorRT cache path must fit the pinned runtime's filename limits; a rejected
path reports the limit and a shorter ASCII path remedy before GPU startup.

A GUI may launch Mouse with no arguments and set its options. Model loading
starts on the owned worker when weights are configured, and restarts when a
backend option changes. Search options retain the loaded model. `isready` waits
for the latest configuration; a first `go` without it still uses the move's
original deadline. If no evaluated choice exists by then, Mouse prints
`info string warning: emergency move without evaluation` and uses deterministic
capture/check/promotion/central-square ordering. Normal decisions prefer visits,
Q, priors and evaluated reused continuations. Set WeightsFile and the
provider/precision before analysis. GpuDllDirectories
takes semicolon-separated absolute SDK directories. Paths containing spaces are
ordinary UCI string values and do not need embedded quotes. MultiPV and WDL
output are configurable; Chess960 uses standard UCI_Chess960 king-to-rook moves.

The standalone benchmark runs the fixed original 16-position corpus, starting
a fresh tree and NN result cache for each case:

```powershell
.\build\windows-release\mouse.exe bench --weights nets/t1.onnx --backend cpu --precision fp32 --deterministic --nodes 800
```

It emits a complete JSON report followed by nodes N nps M after every case
succeeds. See [bench corpus and timing](tests/data/bench/README.md).
[UCI process tests](tests/data/uci/protocol-v1/README.md) document the golden
protocol contract.

## UCI options

This table is generated from the executable's actual uci declarations. Defaults
do not autodiscover a model. Regenerate/check it after rebuilding:

```powershell
.\.venv\Scripts\python.exe -I -m mouse_tools.uci_options --binary build/windows-release/mouse.exe --readme README.md
.\.venv\Scripts\python.exe -I -m mouse_tools.uci_options --binary build/windows-release/mouse.exe --readme README.md --check
```

<!-- BEGIN MOUSE UCI OPTIONS -->
Generated from Mouse (Zulfeqar Malik) via `uci` (build version omitted).

| Option | Type | Default | Range / choices |
| --- | --- | --- | --- |
| WeightsFile | string | &lt;empty&gt; | — |
| Backend | combo | tensorrt | cpu; cuda; tensorrt |
| Precision | combo | fp16 | fp32; fp16 |
| Threads | spin | 2 | 1 … 128 |
| MinibatchSize | spin | 0 | 0 … 1024 |
| NNCacheSize | spin | 2000000 | 0 … 16000000 |
| CacheHistoryLength | spin | 0 | 0 … 7 |
| RamLimitMb | spin | 512 | 1 … 16384 |
| CPuct | string | 1.745 | — |
| CPuctBase | string | 38739 | — |
| CPuctFactor | string | 3.894 | — |
| FpuValue | string | 0.33 | — |
| FpuStrategyAtRoot | combo | same | same; reduction; absolute |
| FpuValueAtRoot | string | 1 | — |
| PolicyTemperature | string | 1.359 | — |
| MultiPV | spin | 1 | 1 … 256 |
| LogSearchStats | check | false | — |
| UCI\_ShowWDL | check | false | — |
| UCI\_Chess960 | check | false | — |
| Ponder | check | false | — |
| MoveOverheadMs | spin | 30 | 0 … 60000 |
| SmartPruningFactor | string | 1.33 | — |
| SmartPruningMinimumBatches | spin | 0 | 0 … 10000 |
| GpuDllDirectories | string | &lt;empty&gt; | — |
| CacheDirectory | string | per-user cache | — |
| Device | spin | 0 | 0 … 1023 |
| CpuThreads | spin | 1 | 1 … 128 |
| WorkspaceMb | spin | 1024 | 1 … 16384 |
| Deterministic | check | false | — |
<!-- END MOUSE UCI OPTIONS -->

`NNCacheSize` counts entries, with a default capacity of 2,000,000. Entries
are allocated on demand, outside `RamLimitMb`; startup does not reserve storage
for the full capacity. Set a smaller count to bound retained evaluations or 0
to disable the cache. Each entry stores only the legal-move float32 logits,
WDL and optional moves-left value. Temperature is applied later.

A 128-bit fingerprint covers the model/provider namespace, every encoded input
bit and the ordered legal policy indices. Fingerprints replace complete input
storage; a hash collision is possible. Allocation
failure is reported explicitly, without reducing capacity or switching providers.
Hits and updates of existing entries allocate nothing; inserting new entries may
allocate. Clearing releases entries while retaining index buckets for reuse.
A local CPU probe measured 795,533,312 bytes of private commit after
`go nodes 20000` with the default cache and tree limits (default smart pruning
stopped at 16,060 visits). With pruning disabled to force all 20,000 visits,
private commit was 796,471,296 bytes. These are CPU measurements; GPU providers
add their own model, workspace and runtime memory.
The diagnostic `--nncache-mb` flag retains its byte-budget meaning; in UCI startup
arguments it is converted to a conservative legal-cache entry capacity.


## Tools and verification

Python installation and hash-locked commands are in [tools/README.md](tools/README.md).
Run downloaded-file tooling with isolated Python:

```powershell
.\.venv\Scripts\python.exe -I tools/fetch_net.py
```

The downloader verifies the pinned SHA-256, including existing files, and replaces
the destination atomically only after verification. The pin is a locally observed
hash from the official Lc0 download, not a publisher-signed checksum.

See the [contribution guide](CONTRIBUTING.md).

`LogSearchStats` emits one `info string searchstats` after each search drains.
Gather and NN-wait durations can overlap provider work; backup includes cached
and terminal evaluations. `gpu_idle_ms` is wall time minus synchronous provider
call time (`idle_basis host_provider_gaps`), not a CUDA kernel timestamp measurement.
GPU backends pad to the next configured batch bucket. TensorRT uses a separate
fixed-shape session and CUDA graph for each bucket, with persistent device IO
and pinned host staging buffers. Changing the backend layout creates a new
verified engine-cache namespace; cold construction builds every bucket.
