# Draw-rate rescaling fixtures (lc0 parity)

`cases.json` and `cases.inc` hold 24 observations from the pinned official lc0 v0.32.1
executable (SHA256 `e32164ceb85ab128e6fe5e02d34cf17608d638335ae7d519f82a58c372e9666b`,
source commit `fd71a2d921b689c5f479d3227c3806c8e272d9c5`). No lc0 code is vendored.

Each case replaces Q and D in the score-v1 startpos replay record
(`tests/data/uci/score-v1/reference/record-root.bin`) and runs `go nodes 1` on lc0's CPU
record/replay backend with `--wdl-draw-rate-target`, `--wdl-draw-rate-reference` and
`--wdl-max-s` set. The grid covers six root W/D/L values and four settings: softer and
sharper targets, two references, and a WDLMaxS of 0.5 that caps the width.

lc0 rescales the evaluation as it enters the tree, and then reports it through its display
inverse, which at zero contempt keeps the width. The UCI `score cp`/`wdl` and the
verbose root `WL`/`D` therefore show the stored, rescaled values.
`tests/search/draw_rate_test.cpp` ("Draw-rate rescaling matches the lc0 executable") checks
Mouse's `rescale_wdl` + `format_score` against all of them exactly.

Regenerate from the repository root (Python 3.12, standard library only) into a new
work directory:

```powershell
python -I -B tests/data/search/draw-rate-v1/capture.py --lc0 .toolchain/reference/lc0/lc0.exe --work .toolchain/draw-rate/<new dir>
```
