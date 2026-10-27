# Changelog

All notable changes to Mouse are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and Semantic Versioning.

## [Unreleased]

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
