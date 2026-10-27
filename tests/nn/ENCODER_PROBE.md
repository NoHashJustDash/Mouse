# Encoder corpus probe

`mouse_encoder_probe <new-output.jsonl>` is a test utility. It always emits
exactly 10,000 positions; there is no count, subset, or resume option. It uses
protocol version 1 and `std::mt19937_64` seed `0x4d4f555345020001`. Fixed edge
histories precede deterministic legal random walks, including both colors,
Chess960 king/rook overlap, all promotion choices in legal move sets, raw and
geometrically normalized en-passant, pinned EP repetition, long histories, and
rule-50 counters.

Build and generate from the repository's developer shell:

```powershell
. .\scripts\Enter-DeveloperShell.ps1
cmake --build --preset local-net-release --target mouse_encoder_probe
build/windows-release/mouse_encoder_probe.exe build/encoder-corpus-v1.jsonl
```

The destination must not exist. Creation is exclusive: rerunning against an
existing file fails without changing it. A failure after creation leaves an
incomplete file for inspection and prints no success marker. The utility never
removes a destination. Accept output only when exit status is zero and stdout
contains `MOUSE_ENCODER_PROBE_RESULT version=1 ... positions=10000 ...`; JSONL
row count and IDs must also be independently checked. The marker includes seed,
white/black counts, and the number of fixed edge positions. Errors go to stderr.

Each line is one JSON object:

```text
{"id":0,"frames":[{"fen":"...","repetitions":0}],"planes":[[mask,value],...],"moves":[["e1h1",index],...]}
```

- IDs are contiguous integers 0 through 9999.
- `frames` contains the last one through eight chronological frames. FENs use
  absolute board/color coordinates and actual counters. Each FEN EP field equals
  the captured frame EP: raw at a FEN root, geometric capturability after an
  appended move. Castling uses KQkq where orthodox and rook-file letters where
  required. Repetition counts are uint32 prior occurrences from the full game,
  not recomputed from the truncated eight-frame view.
- `planes` contains all 112 classical planes as `[uint64 decimal mask, float
  value]`. Values have `float::max_digits10` precision and locale-independent
  decimal syntax. The final side to move determines color/rank orientation for
  every historical piece plane. Encoding uses input format 1 and default
  `HistoryFill::FenOnly`.
- `moves` contains the complete current legal set. UCI always uses Chess960
  king-to-rook castling notation, including orthodox castles. Indices are the
  1858-slot `PolicyMap`; a duplicate index is a generation failure.

`EncodingHistory` follows pinned Lc0 v0.32.1 `src/chess/board.cc` (raw FEN EP and
geometric double-push EP) and `src/chess/position.cc` (input repetition). It does
not use the FIDE-normalized Position key. The pinned-EP cycle is included so
independent consumers can distinguish these identities. Exact reference commit:
`fd71a2d921b689c5f479d3227c3806c8e272d9c5`. No Lc0 source is vendored.

This probe reports inputs and legal policy indices, not neural output parity.
The Python mirror gate independently reconstructs and checks every row; real
Lc0 binary verification remains a separate gate.
