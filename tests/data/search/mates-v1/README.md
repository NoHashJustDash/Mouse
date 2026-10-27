# Original exact mate corpus, version 1

These 18 positions were constructed by Mouse's own coordinate enumeration, not
copied from a puzzle collection. There are six positions for each exact mate
distance 1, 2, and 3: KQK, KRK, and KQNK with an own knight blocking the queen's
rank-eight ray, each with both colors to move. Black cases are a rank reflection
plus color exchange of the independently proved white construction. All 18
board/side/castling/en-passant identities are distinct.

`mates.epd` uses six FEN fields followed by `id`, `variant standard`, `dm N`, and
`bm` with every root move that forces mate within that **minimal** mate horizon.
`mates.json` also records one legal PV against a longest surviving defense, all
refuted shallower mate horizons, and cumulative proof-state statistics. A root
move that only forces a slower mate is outside this `bm` set. No Mouse node
budgets have been guessed; `mouse_node_budgets` is null until measured by the
separate production-search gate.

## Independent proof

The original stdlib tool is `tools/src/mouse_tools/mate_fixtures.py`. Its private
RNG seed is 20261017. It enumerates a losing king on a8, attacking kings on c6,
b6, c5, d5, d6, and shuffled queen/rook squares. A second enumeration places a
queen on d8..h8 with an own knight between it and a8. It selects the first exact
proof in each material/distance category and proves the reflected color case
separately. Neither Mouse search nor Stockfish search scores choose moves.

The exact pinned official Stockfish 17.1 executable supplies legal move lists,
checkers and board transitions via `position fen ... [moves ...]`, `d`, and
`go perft 1` in one persistent single-thread CPU process. The tool verifies the
binary SHA-256 before launch and after all queries, the exact UCI identity and
required options, every returned FEN against an original coordinate transition,
and checkers against independent piece-attack geometry. It rejects adjacent
kings and an already checked non-side-to-move king before querying Stockfish.
Only pawnless positions with no castling/en-passant and rule-50 clock <=90 are
supported. Capturing the last major piece correctly produces a reachable
king-versus-king draw branch.

The finite proof is an attacker OR / defender AND tree. Checkmate of the opponent
succeeds; stalemate, attacker checkmate, or a live horizon leaf fails. The prover
checks complete root move sets at 1, 3, then 5 plies, stopping at the first
successful horizon. Every smaller horizon has therefore been refuted. A defender
can refute a branch with one escape; otherwise every legal defense must succeed.
Memoization uses board/side/rights/EP, with no persistent or prior-run cache. Five
plies cannot reach rule-50 from these clocks. A repeated position cannot create
an artificial mate in a finite proof. PV length is exactly 2*N-1 and its terminal
checkmate is separately replayed.

`oracle-states.jsonl.gz` preserves every real query, complete raw `d`/perft reply,
and normalized state, with deterministic gzip mtime. Unit tests hash-check it,
reparse all replies, recompute transitions/check geometry, and reprove all 18
positions and all 36 horizons. This is recorded-oracle replay, not a fresh binary
execution. `uci-handshake.txt` is the actual greeting. `provenance.json` pins the
binary, generator and shared legal-response parser source, artifact hashes,
source release/commit, exact UCI commands, interpreter identity, wall time and
cold-query counts. The binary is private cached tooling and is not vendored.

## Regeneration and fresh verification

Install the worktree tools package in the locked development environment first.
Every downloaded-file reader must run isolated:

```powershell
python -I -m mouse_tools.mate_fixtures --oracle <workspace>/.worktrees/net-conversion/.toolchain/reference/stockfish-17.1.exe --output .toolchain/mates/regenerated-v1
python -I -m mouse_tools.mate_fixtures --oracle <workspace>/.worktrees/net-conversion/.toolchain/reference/stockfish-17.1.exe --verify tests/data/search/mates-v1 --output .toolchain/mates/live-verify-v1.json
python -m pytest tools/tests/test_mate_fixtures.py
```

Generation requires a new destination directory. Verification writes its report
only after all proofs and before/after identities pass, using atomic replacement.
Malformed data, missing states, oracle exit/timeout, or exhausted proof budgets
fail loudly and preserve an existing report. Default bounds: 200,000 distinct
states, 300 seconds per prover, 10 seconds per oracle response, 4,096 characters
per response line, 512 lines per response, and 64 MiB decompressed transcript.
There is no reduced-depth/full-gate substitution.

The final handoff records the exact development-interpreter commands and results.
This corpus does not establish Mouse search node budgets, neural mate solving,
SPRT, self-play, sanitizers, or official Python acceptance.
