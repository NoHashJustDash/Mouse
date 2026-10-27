// SPDX-License-Identifier: GPL-3.0-or-later
// Observed official lc0 v0.32.1, SHA256
// e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b.
// Source: fd71a2d9 src/search/classic/search.cc:725-821.
// CPU random backend uniform=true,delay=1; threads1/batch1/cache0;
// SmartPruningFactor0, MaxCollisionEvents1/Visits1, MaxPrefetch0.
// FEN:7k/8/5KQ1/8/8/8/8/8 w - - 0 1
// go nodes16 searchmoves f6e5 ->reported17nodes,stalemate16visits.
// go nodes18 searchmoves g6g7 ->reported19nodes,mate2visits.
// go nodes1 (filter removed,no fresh visit) ->bestmove g6g7.
// Native verbose: f6e5 N16,WL0,D1,M0,T; g6g7 N2,WL1,D0,M0,T.
// Prior is omitted: its native printed value is rounded and irrelevant to
// the proven-outcome comparison. Child proof is the opponent's viewpoint.
constexpr mouse::search::FinalMoveStatistics kNativeTerminalDraw{
    .visits = 16, .q = 0, .child_proof = {.outcome = mouse::search::ProofOutcome::Draw}};
constexpr mouse::search::FinalMoveStatistics kNativeTerminalMate{
    .visits = 2, .q = 1, .child_proof = {.outcome = mouse::search::ProofOutcome::Loss}};
