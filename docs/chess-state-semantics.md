# Repetition and en-passant semantics

Mouse retains the raw validated en-passant square for FEN. Its repetition key
includes en passant only when the side to move has a legal capture, including
king-safety and pinned-pawn checks. This follows the legal-move equality in
[FIDE article 9.2.3](https://handbook.fide.com/chapter/E012023).
The incremental and full recomputation paths share this precise contract.

The pinned Lc0 v0.32.1 board representation differs. Its double-push update
retains an EP flag when an adjacent opposing pawn could capture geometrically;
it does not check that pawn's king safety at that step. Its hash includes that
encoded pawn/EP state. Its FEN load path also retains validated raw EP state.
References: [board.cc](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/chess/board.cc#L680)
and [board.h](https://github.com/LeelaChessZero/lc0/blob/fd71a2d921b689c5f479d3227c3806c8e272d9c5/src/chess/board.h#L102).

This investigation does not claim Lc0 hash equivalence. Neural history encoding
must reproduce the pinned Lc0 input behavior independently, using raw state and
history where required. A legal repetition key is not a substitute for an input
tensor cache key. v0.2 binary parity tests remain required for these semantics.

Regression coverage includes capturable, absent and pinned en-passant,
recomputed/incremental keys, FEN round-trips, irreversible moves and undo.
The parser also reconstructs the asserted preceding double-push occupancy to
reject pre-existing attacks that make the EP history impossible.
