# Perft suite provenance

`perft.epd` contains **126 distinct positions and 392 required depth checks**.
The complete local and CI gate runs every stored `Dn` value; a shallow run does
not satisfy this suite. No depth from the six positions in
[`docs/PHASE1_SPEC.md`](../../docs/PHASE1_SPEC.md) was shortened or omitted.

Each noncomment line contains six FEN fields followed by:

```text
; id "unique_identifier"; variant standard|chess960; D1 count; D2 count; ...;
```

Clocks remain in the FEN, but uniqueness ignores both clocks and castling-rights
ordering. The orthodox initial arrangement is excluded from the generated
Chess960 starts to avoid duplicating the standard start position under another
castling notation. Identifiers describe the source/coverage group:

| Group | Positions | Coverage |
|---|---:|---|
| `spec_*` | 6 | Every exact spec FEN and required depth/count, plus every shallower depth |
| `special_*` | 20 | Both colors; legal and pinned illegal EP, EP check evasions, all promotion pieces, capture promotions, both castling sides, single/double check, blocks, pins, mate, stalemate, high rule-50 counter |
| `stockfish_frc_*` | 4 | Independently published Chess960 regression facts |
| `frc_start_*` | 24 | Deterministic Scharnagl-numbered legal initial arrays |
| `frc_castle_*` | 32 | Both colors; stationary king/rook, king-rook swap, overlapping origins/destinations, long crossing paths, edge origins |
| `playout_*` | 40 | Deterministic legal playout positions, both colors, sampled every third ply from ply 8 |

The stripped Chess960 cases contain one castling rook and the two kings. The
opposing king is distant and outside the rook's file. During generation each
case must include its expected king-to-rook castling move in Stockfish's D1 root
divide. EP and promotion cases likewise check required or forbidden moves before
any fixture is written. Random playout moves are selected only from Stockfish's
legal root divide. The tool validates FEN shape; the Mouse C++ tests separately
validate all FENs with Mouse's own parser and legal move generation.

## Independent reference

Reference: **Stockfish 17.1**, official Windows x64 BMI2 release binary, from
[the official release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_17.1).
The tag resolves to source commit `03e27488f3d21d8ff4dbf3065603afa21dbd0ef3`.
The executable prints `id name Stockfish 17.1`; `Threads` is explicitly set to 1.
Every Chess960 query explicitly enables `UCI_Chess960`; standard queries disable
it. There is no engine or data fallback.

- Archive: [stockfish-windows-x86-64-bmi2.zip](https://github.com/official-stockfish/Stockfish/releases/download/sf_17.1/stockfish-windows-x86-64-bmi2.zip)
- Archive SHA-256, locally observed: `8034b0a0725bcf19796c3695b8794698b15bc2168edacde06ab32fec23cc709d`
- Executable SHA-256, locally observed and enforced before execution: `8f8c79837ddf2d0b7baccf555769778dd6a7e22bec823814fe19ea8bc37ff39f`
- GitHub's official release API reports no asset digest for this release and
  lists no checksum asset. These local hashes pin the observed artifact; they
  are not described as publisher checksums. The original API JSON, archive and
  executable stay in ignored `.toolchain/reference/stockfish-17.1/`.

The four `stockfish_frc_*` facts are from lines 72-75 of the
[official `sf_17.1` perft tests](https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_17.1/tests/perft.sh):
D2 = 1,438; D3 = 37,340; D4 = 2,237,725; D4 = 2,098,209, respectively.
All four were checked against the downloaded reference before writing this
suite. The six spec facts were also checked rather than copied blindly.
Position/count facts are cited; no Stockfish source code is vendored. The
reference executable is a private test dependency and is not shipped with Mouse.

## Regeneration and verification

From the chess-core worktree root, after downloading and extracting the exact
archive into the ignored path above:

```powershell
python -I tools/src/mouse_tools/perft_reference.py generate --engine .toolchain/reference/stockfish-17.1/stockfish/stockfish-windows-x86-64-bmi2.exe
# Generated 126 distinct positions; every stored depth verified
python -I tools/src/mouse_tools/perft_reference.py verify --engine .toolchain/reference/stockfish-17.1/stockfish/stockfish-windows-x86-64-bmi2.exe
# Verified 126 positions, 392 depth checks
```

The tool is standard-library-only and runs directly with isolated Python; it
also works as `python -I -m mouse_tools.perft_reference` after normal package
installation. It enforces the executable hash and UCI identity, validates output
by comparing each root divide sum with its total, rejects missing/duplicate
facts and positions, and fails on a timeout or nonconforming oracle response.
Generation rejects identical resolved or hard-linked suite/provenance destinations
before constructing the oracle or querying it, so metadata cannot overwrite the EPD.
Decimal inputs are bounded before integer conversion: depths are 1 through 128,
and counters/node counts fit unsigned 64-bit integers (fullmove is positive).
Oversized or malformed tokens raise a clear reference-input error.

Generation writes the fixtures only after every requested count, known spec
fact, and special-move assertion passes. The generator uses no count cache. One
persistent Stockfish process is reused, and every stored depth gets a separate
`position fen ...`, `d`, `go perft N` query. Neither generation nor verification
reads counts back as oracle results.

`perft-provenance.json` records the exact seed **20261017**, command template,
reference identity and hashes, selected known facts, all legal playout move
histories, and the resulting EPD SHA-256. Initial generation and a second
regeneration produced the same EPD hash:
`910feba480429df63c0348b15ee5e158467f5762be16ff98378e88ecbdfa62c6`.

Initial generation on 2026-10-17 used the explicitly permitted cached CPython
3.12.14 in this worktree's ignored `.toolchain/python-development` environment.
The command wrapper reported **2.0185292 seconds** wall time, including interpreter
and reference startup/exit, legal playout queries, all 392 depth queries, writing
both fixture files, and displaying the log tail. This is provenance for that
uncached run, not a Mouse speed benchmark or official-interpreter acceptance.
The complete initial output is in ignored `.toolchain/perft-generate.log`.
A later independent `verify` command checked all 392 stored values again.

## Approved gate contract

`mouse_tools.chess_gate perft` protects all 392 facts in this tracked baseline,
including every shallower depth and all 40 generated playout positions. Its
package manifest combines the 86 candidate position/depth contracts with 40
normalized playout position SHA-256/depth pairs and a digest over every required
`(position_key(FEN), variant, depth, nodes)` record. The sorted records use ASCII
JSON with separators `(",", ":")`; the normalized fact SHA-256 is
`a88ad185c25e9abca7026a4e35fd088d17fc4694be4baf8e6861412021380d5f`.
Package tests reproduce that digest and the 40-position manifest from this EPD.

Comments, row order, identifiers, clocks and castling-right order may change;
additional positions/depths are allowed, and all stored depths execute. Existing
baseline facts cannot be deleted or changed while retaining the approved gate
identity. A different independently sourced suite can use the reference
`verify` command above; replacing the milestone baseline requires Senior
Developer review. This contract preserves the approved fixture and does not
change the roadmap's general minimum of 100 positions.