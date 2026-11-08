# Vendored code

## lc0_syzygy

`syzygy.cc` and `syzygy.h` come from Leela Chess Zero, commit
`fd71a2d921b689c5f479d3227c3806c8e272d9c5`, `src/syzygy/`, licensed GPL-3.0-or-later with the
additional permission for NVIDIA libraries stated in their headers. Lc0's file in turn derives from
Ronald de Man's Syzygy probing code and Stockfish. Copyright headers are kept as they were.

Changes from upstream:

- The includes point at `board_shim.h`, which presents Mouse's chess types under the names the file
  expects. The tablebase logic itself is unchanged.
