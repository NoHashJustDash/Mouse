# Changelog

All notable changes to Mouse are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and Semantic Versioning.

## [Unreleased]

## [2.0.0] - 2026-11-18

Mouse now ships with the M2.1 network and a faster engine. With M2.1, Mouse 2.0.0 beat
lc0 v0.32.1 running T1 by +33.1 ± 13.8 Elo in an SPRT (632 games at 10+0.1 on one
machine). That result measures the engine and the network together.

### Changed

- The network is now M2.1, which replaces M1.5. It has M1.5's shape (19.9M parameters,
  8.4M used per square) and was trained longer, on BT4 labels for all of its Lc0 data.
  M2.0, its first 750k steps, beat M1.5 by +27.2 ± 12.4 Elo, and M2.1 beat M2.0 by
  +31.9 ± 13.4 Elo.
- The release ZIP ships the network as `M2.1.onnx`.
- Legal moves are found from the checking and pinned pieces, where every candidate move
  was made and unmade. The moves and their order are unchanged; `perft` is 5.5 times as
  fast.
- Release builds use link-time optimisation (`MOUSE_ENABLE_IPO`).
- The GPU backends are built for up to eight batch sizes (1, 8, up to five geometric steps
  and the maximum), where there were up to four, so a batch is padded less. The first
  start builds about twice as many TensorRT engines: 438 seconds on an RTX 5070, against
  218 in 1.1.0. Engine caches from 1.1.0 are rebuilt once.
- At the default `CacheHistoryLength` of 0, the evaluation cache key leaves out the
  50-move counter, as lc0's does. The same position at another count is now a cache hit,
  so a search with the cache is no longer bit-identical to one without it.
- Proven positions are handled as lc0's sticky endgames are: a proven position below the
  root takes visits with its exact value and no network evaluation, a move into a
  position the opponent has won gets prior 0, and a newly proven position corrects what
  its ancestors already received from it. Mate distances below a proven position are no
  longer refined, so a reported mate can be longer than the shortest one.
- Each network row's legal moves are mapped to policy indices once, not twice, and
  expansion seeds are checked for duplicates in linear time.
- With M2.1, the engine searches 22% more positions in the same time than 1.1.0's.

### Fixed

- A move whose time allowed no further search was held until its deadline. It is now
  played at once, and the unused time stays on the clock.

### Added

- The network's architecture and training code are published separately as
  [Mouse-MoE](https://github.com/NoHashJustDash/Mouse-MoE).
- `setup-gpu.ps1`, in the release ZIP: downloads the CUDA, cuDNN and TensorRT libraries
  Mouse needs from NVIDIA's servers into a `gpu` folder beside `mouse.exe` and checks their
  SHA-256 hashes. No NVIDIA account or installer is needed.
- Mouse uses the `gpu` folder beside `mouse.exe` when `GpuDllDirectories` is empty.
- `mouse doctor`: reports the NVIDIA driver, the library folder, any missing library and
  the GPU, and says what to do about each problem.
- The packager can make a DirectML release ZIP for AMD, Intel and other DirectX 12 GPUs,
  with M2.1 as plugin-free FP32 and FP16 exports ([docs/DIRECTML.md](docs/DIRECTML.md)).
  2.0.0 has no DirectML download; one is planned for 2.0.1.

## [1.1.0] - 2026-11-07

Mouse now ships with its own network, M1.5.

### Added

- The M1.5 network, with the `mouse_moe_trt` TensorRT plugin that runs it.
- Syzygy tablebase probing, vendored from Lc0.
- Lc0's `selfplay` and `dag-preview` modes, and its `policyhead` and `valuehead` modes.
- Lc0's gather, parallel picking, solid tree, contempt, score, temperature, noise, time and
  logging options.
- A DirectML backend for AMD, Intel and other DirectX 12 GPUs, in separate builds.
- `onnx2leela`, which wraps an ONNX network as a Leela weights file.
- An additional permission under GPL section 7 for combining Mouse with NVIDIA's CUDA, cuDNN
  and TensorRT libraries, in `docs/NVIDIA_PERMISSION.md`.
- `docs/NETWORK.md`: the network's licence, the Lc0 training-data notice and the method used
  to relabel that data.

### Changed

- CPuct, FpuValue, PolicyTemperature, WDLDrawRateTarget and WDLDrawRateReference now default
  to values tuned for M1.5, which beat lc0's values by +45.7 ± 15.5 Elo in an SPRT.
- Missing TensorRT engines are built in a helper process.

### Fixed

- Mouse kept discarding its tree before about 40% of searches when the tree held a repeated
  position; it now keeps the tree as lc0 does.
- Root moves that can no longer become best are pruned as lc0 prunes them.
- 1 ms waits now work when a GUI starts Mouse without a window.
- Batch sizing no longer charges each row a whole root path.
- Every evaluated row is published after a halted gather.

## [1.0.0] - 2026-10-27

First public release: Mouse plays Lc0's T1-256x10-distilled-swa-2432500 network
with lc0's search defaults, on NVIDIA GPUs (TensorRT, CUDA) or the CPU.

### Added

- Windows GPU toolchain inventory, compatibility checks, and provenance record.
- C++20 CMake/vcpkg foundation, app-local ORT binding and loaded-version CLI.
- Tracked CPU/GPU/CLI regressions, portable CI, hash-locked Python tools and verified T1 downloader.

- Independent magic/PEXT bitboard chess core, reversible state and legal Chess960 move generation.
- Validating FEN, repetition/game outcomes, perft/divide CLI, sourced 126-position perft suite and full million-game runner.

- Independent T1 attention/smolgen converter with strict downloaded-file validation and hash-locked CPU Torch tools.
- Classical 112-plane history encoder, exact 1858 policy mapping, actual Lc0 reference fixtures and full parity runners.
- ONNX metadata/tensor validation and the describe-net inspection command.

- Explicit serialized ORT CPU/CUDA/TensorRT inference, pinned I/O, validated disk caches and exact bounded NN result cache.
- Genuine FP16 conversion, deterministic hashed evaluation transport, fixed50 parity tooling and end-to-end backendbench.

- Bounded compact MCTS tree, half priors, exact draw rules, tree reuse and pipelined leaf gathering.
- Deterministic standalone search, explicit node/time/depth controls and full input/model identities.
- Original mate-in-1/2/3 corpus with complete independently replayed legal-oracle proofs.

- Complete asynchronous UCI, bounded cancellable input, transactional options/history,
  actual Lc0 WDL_mu score parity, pondering, root restrictions and baseline clocks.
- Fixed full-history bench, isolated native UCI conformance and generated option documentation.
- Verified optional moves-left utility and exact terminal proof propagation that
  preserves real visits and exploration while respecting proven outcomes in final selection.
- Audited complete backend throughput evidence and cold/warm CUDA startup measurements.
- Persisted TensorRT EPContext models, startup phase diagnostics and cold-build progress.
- Lc0 v0.32.1 legacy time management and smart pruning based on the reference source,
  with native elapsed-time comparison tables.
- On-demand NN caching at the unchanged default capacity of 2,000,000 entries,
  with 128-bit input fingerprints and variable legal-logit payloads. Cache
  allocations are accounted separately from the search-tree limit.

### Fixed

- Keep pending children eligible for PUCT selection when resolving batch collisions.
- Prefer established wins over ordinary moves or draws, and ordinary moves over
  established losses, in final move selection; keep bestmove, PV and MultiPV consistent.
- Reserve measured output and wake latency before the raw GUI clock cap while
  retaining ordinary legacy allocations when the remaining clock permits them.

- Load configured models on the owned worker before search; first-go loading
  shares the move deadline and obsolete configuration loads are retired safely.
- Retire new-game visits by generation without reallocating the drained tree arena.
- Choose emergency moves from completed visits, priors and reused evidence;
  explicitly warn when only deterministic chess ordering is available.

- Enforce timed UCI publication independently of synchronous inference completion,
  with one owned worker and a bounded pending turn; preserve late fatal errors.
- Add opt-in UCI debug host timestamps for inference and turn boundaries.
- Repair Linux compiler prerequisites, exact binary CLI identity reads, optional
  serialization checks, fatal UBSan policy and isolated Python interruption tests.

- Recover timed search after transient provider stalls using recent physical root evaluations; preserve drain reserves across cache hits and reused roots.
- Preserve original score-evidence bytes across Git exports.
- Reject malformed UTF8 option paths and suppress bestmove after fatal output/provider errors.
- Keep cancellation across search activation and safely accept input after bestmove publication.
- Stage pinned ORT libraries beside relocated executable outputs and repair missing payloads.
