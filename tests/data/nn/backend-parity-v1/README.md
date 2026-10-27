# Fixed 50-history backend parity inputs

These are original Mouse chess histories and numeric inputs, GPL-3.0-or-later.
The set is frozen for v0.3; its count, membership and tolerance cannot be changed
by a harness command-line option. `cases.json` has SHA256
`e581f042ab3918d1d87f82106e58b0e553841b6787c11ab90344a13310b61be5`.
`planes.bin` has SHA256
`77d172aa20ef60267631bc8f632a6370370eb7a85f87dea2df266eb31fbf7d01`.

All 27 named histories from `../lc0-classical-v1/reference.json` come first,
unchanged. Each case's packed planes are decoded from its genuine official
Lc0 v0.32.1 `fen_only.v6.gz` training record. Legal move/index rows come from
that binary's complete verbose root rows, already cross-checked with the training
record's complete legal indices. They cover startpos, both sides, clock/history
fill, asymmetric Black input, rule50=99, castling before/after, promotions and
capture promotions of both colors, legal/pinned EP of both colors, Chess960
stationary/swap/queenside castling of both colors, check, pawnless and repetition.
No upstream implementation or lookup table is copied.

The remaining 23 tensors come in source order from the accepted 1000-distinct
input corpus, after rejecting tensors equal to any named edge or earlier selected
input. Selected source IDs: 1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,21,22,23,24,
25,26,27. The complete source corpus passed the independent 10000-position
C++/Python plane and uncached Stockfish legal-move/index gate. These extra cases
retain complete chronological histories, including eight-frame histories, and
all original packed planes and complete legal move rows. All 50 complete
float32 tensors are distinct. There are 736 legal move rows in total.

`provenance.json` pins all source artifacts and 27 training chunks. Its sources:

- Actual-Lc0 edge manifest SHA256
  `5dee45734358db88d6818a59ae334e41936d27d88f52d49beb889606205fe865`;
  official Lc0 commit `fd71a2d921b689c5f479d3227c3806c8e272d9c5`.
- Accepted 10000-history JSONL SHA256
  `5df515c842a956b477d94e066d07695fde83b0ffe724f6631fc477d236b604c8`.
- Accepted 1000-tensor NPY SHA256
  `e09f29c803a5cb896c150e3055b1770490a9696561c223b226e42a22cb856632`.
- Selection metadata SHA256
  `3d7c69b4f410cf921c9d5fb4295297f4379fdc81a2992175fc3ee271ca179107`.

Reproduce into a new directory with the approved cached source artifacts:

```powershell
python -I -m mouse_tools.make_backend_parity_fixture --edges tests/data/nn/lc0-classical-v1 --probe <workspace>/.worktrees/net-conversion/.toolchain/encoder-corpus-project-v1.jsonl --inputs <workspace>/.worktrees/net-conversion/.toolchain/convert/parity-inputs-1000-v1.npy --metadata <workspace>/.worktrees/net-conversion/.toolchain/convert/parity-inputs-1000-v1.json --output .toolchain/convert/backend-parity-reproduced
```

All three generated artifacts have been reproduced byte-for-byte. The source
hashes are fixed in the generator; a changed source or duplicate named edge is
an explicit error, never an occasion to drop a case.

`planes.bin` is exactly 67216 bytes. It follows the native transport: ASCII
`MOUSEIN1`, uint32LE count50, uint32LE planes112, then each row's 112 pairs of
uint64LE mask and float32LE value. Bits expand A1-first to `[112,8,8]` with the
value where the bit is set and zero elsewhere. No host struct alignment is used.

The CPU oracle is the preserved independently produced official Lc0 FP32 ONNX
SHA256 `0e2699c19b617781d4fbe3d00286756b125836a30c0658afc620e197bec35d04`.
The harness evaluates it directly each run with CPUExecutionProvider only and
fallback disabled. It compares native outputs by legal-move masked softmax at
temperature1, absolute probability difference<=0.005 and WDL difference<=0.005.
MLH and raw policy logits are diagnostic only. All 50 rows must be evaluated;
there is no resampling, case filtering or tolerance override. A CPU run cannot
satisfy the GPU gate; both CUDA FP16 and TensorRT FP16 results are required.
