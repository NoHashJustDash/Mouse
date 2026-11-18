# Mouse on AMD, Intel and other DirectX 12 GPUs (DirectML)

The DirectML build of Mouse runs its network through ONNX Runtime's DirectML
provider, so it works on any GPU with a DirectX 12 driver: AMD Radeon, Intel Arc
and integrated graphics, and NVIDIA. On NVIDIA GPUs the CUDA/TensorRT build is much
faster; use DirectML there only if that build cannot run.

Built with the `windows-dml` CMake preset after `scripts/stage-directml-sdk.ps1` has
assembled the pinned SDK: ONNX Runtime 1.22.0 (DirectML package) and DirectML 1.15.4.
Its ONNX Runtime has no CUDA or TensorRT provider, so this build offers the `directml`
and `cpu` backends only, and `directml` is the default.

Mouse 2.0.0 has no DirectML download. One is planned for 2.0.1, with the two M2.1
networks named under Setup below; the `M2.1.onnx` in the NVIDIA download runs only on
TensorRT. Until then, build from source as described above and use an Lc0 network such
as T1.

## Requirements

- 64-bit Windows 10 version 1803 or newer, or Windows 11.
- A GPU with a current DirectX 12 driver.
- The files of the package kept together: `mouse.exe`, `onnxruntime.dll`,
  `onnxruntime_providers_shared.dll` and `DirectML.dll`. Mouse refuses to start a
  DirectML session if Windows loads any other `DirectML.dll`, such as the older one
  in System32.

## Setup

1. Unzip `mouse-<version>-windows-x64-directml.zip`. Nothing else needs installing.
2. Run `mouse.exe doctor` to confirm the GPU is found.
3. Add `mouse.exe` to your chess GUI as a UCI engine.
4. Set `WeightsFile` to the full path of one of the two M2.1 networks in the folder:
   - `M2.1.onnx` (FP32) works with the default options.
   - `M2.1-fp16.onnx` (FP16) is faster. Set `Precision` to `fp16` with it.

`Precision` must match the network: `fp32` (the default) for an FP32 network, `fp16` for
an FP16 one. DirectML does not convert precision, so a mismatch fails with a message
that says so.

## Speed and accuracy

Measured on a GeForce RTX 5070 with `backendbench`, in network evaluations per second:

| Network | DirectML | TensorRT build, same card |
|---|---|---|
| `M2.1-fp16.onnx` | 8,901 | 33,816 |
| `M2.1.onnx` (FP32) | 5,842 | |

On that card DirectML is about 3.8 times slower than the TensorRT build. Speeds on AMD
and Intel GPUs have not been measured: run `backendbench` (below) to see yours. On the
CPU alone (`Backend` set to `cpu`), M2.1 ran at about 80 evaluations per second on a
Ryzen 7 7800X3D, which plays legal chess but weakly.

Both networks give the same answers as the reference: on 1,024 held-out positions the
FP32 network chose the same top move as the CPU provider every time, and the FP16 network
99.2% of the time (mean policy KL 1.9e-4, largest win/draw/loss difference 0.034).

`MinibatchSize` 0 (the default) uses a fixed batch of 128 on DirectML, because DirectML
reports no processor count to size it from (128 was fastest of 64, 128 and 256 on the
GPU it was measured on). Other GPUs may prefer another value; measure with
`backendbench` below.

## Choosing the GPU

`Device` counts the hardware GPUs in Windows' high-performance order, so `0` is the
fastest one: on a laptop with integrated and discrete graphics, `0` is the discrete GPU
even when the display runs on the integrated one. An ordinal past the last GPU fails
and lists the GPUs it found. To see which GPU is used and how fast it is:

```
mouse.exe backendbench --weights <net.onnx> --backend directml --precision fp32 --batches 50
```

The `device_identity` line names the GPU and its driver version.

`mouse.exe doctor` is the quick check: it confirms `DirectML.dll` is in place, names the
GPU that `Device` 0 resolves to, and says what to do if either is missing.

## UCI options

<!-- BEGIN MOUSE UCI OPTIONS -->
Generated from Mouse (Zulfeqar Malik) via `uci` (build version omitted).

| Option | Type | Default | Range / choices |
| --- | --- | --- | --- |
| WeightsFile | string | &lt;empty&gt; | — |
| Backend | combo | directml | cpu; directml |
| Precision | combo | fp32 | fp32; fp16 |
| Threads | spin | 2 | 1 … 128 |
| MinibatchSize | spin | 0 | 0 … 1024 |
| NNCacheSize | spin | 2000000 | 0 … 16000000 |
| CacheHistoryLength | spin | 0 | 0 … 7 |
| RamLimitMb | spin | 512 | 1 … 16384 |
| CPuct | string | 1.26 | — |
| CPuctBase | string | 38739 | — |
| CPuctFactor | string | 3.894 | — |
| FpuValue | string | 0.323 | — |
| FpuStrategyAtRoot | combo | same | same; reduction; absolute |
| FpuValueAtRoot | string | 1 | — |
| PolicyTemperature | string | 1.352 | — |
| MovesLeftMaxEffect | string | 0.0345 | — |
| MovesLeftThreshold | string | 0.8 | — |
| MovesLeftSlope | string | 0.0027 | — |
| MovesLeftConstantFactor | string | 0 | — |
| MovesLeftScaledFactor | string | 1.6521 | — |
| MovesLeftQuadraticFactor | string | -0.6521 | — |
| WDLDrawRateTarget | string | 0.779 | — |
| WDLDrawRateReference | string | 0.75 | — |
| WDLMaxS | string | 1.4 | — |
| MultiPV | spin | 1 | 1 … 256 |
| LogSearchStats | check | false | — |
| UCI\_ShowWDL | check | false | — |
| UCI\_Chess960 | check | false | — |
| Ponder | check | false | — |
| MoveOverheadMs | spin | 30 | 0 … 60000 |
| SmartPruningFactor | string | 1.33 | — |
| SmartPruningMinimumBatches | spin | 0 | 0 … 10000 |
| GpuDllDirectories | string | &lt;empty&gt; | — |
| CacheDirectory | string | per-user cache | — |
| Device | spin | 0 | 0 … 1023 |
| CpuThreads | spin | 1 | 1 … 128 |
| WorkspaceMb | spin | 1024 | 1 … 16384 |
| Deterministic | check | false | — |
| MaxCollisionEvents | spin | 917 | 1 … 65536 |
| MaxCollisionVisits | spin | 80000 | 1 … 100000000 |
| MaxCollisionVisitsScalingStart | spin | 28 | 1 … 100000 |
| MaxCollisionVisitsScalingEnd | spin | 145000 | 0 … 100000000 |
| MaxCollisionVisitsScalingPower | string | 1.25 | — |
| OutOfOrderEval | check | true | — |
| MaxOutOfOrderEvalsFactor | string | 2.4 | — |
| MaxPrefetch | spin | 32 | 0 … 1024 |
| IdlingMinimumWork | spin | 0 | 0 … 10000 |
| ThreadIdlingThreshold | spin | 1 | 0 … 128 |
| MinimumProcessingWork | spin | 20 | 2 … 100000 |
| MinimumPerTaskProcessing | spin | 8 | 1 … 100000 |
| MinimumPickingWork | spin | 1 | 1 … 100000 |
| MinimumRemainingPickingWork | spin | 20 | 0 … 100000 |
| SolidTreeThreshold | spin | 100 | 1 … 2000000000 |
| Temperature | string | 0 | — |
| TempDecayMoves | spin | 0 | 0 … 640 |
| TempDecayDelayMoves | spin | 0 | 0 … 100 |
| TempCutoffMove | spin | 0 | 0 … 1000 |
| TempEndgame | string | 0 | — |
| TempValueCutoff | string | 100 | — |
| TempVisitOffset | string | 0 | — |
| DirichletNoiseEpsilon | string | 0 | — |
| DirichletNoiseAlpha | string | 0.3 | — |
| TaskWorkers | spin | -1 | -1 … 128 |
| MaxConcurrentSearchers | spin | 1 | 0 … 128 |
| SearchSpinBackoff | check | false | — |
| TwoFoldDraws | check | true | — |
| StickyEndgames | check | true | — |
| FpuStrategy | combo | reduction | reduction; absolute |
| HistoryFill | combo | fen\_only | no; fen\_only; always |
| DrawScore | string | 0 | — |
| NodesPerSecondLimit | string | 0 | — |
| Contempt | string | &lt;empty&gt; | — |
| ContemptMode | combo | play | play; white\_side\_analysis; black\_side\_analysis; disable |
| ContemptMaxValue | string | 420 | — |
| WDLContemptAttenuation | string | 1 | — |
| WDLBookExitBias | string | 0.65 | — |
| WDLCalibrationElo | string | 0 | — |
| WDLEvalObjectivity | string | 1 | — |
| UCI\_Opponent | string | &lt;empty&gt; | — |
| UCI\_RatingAdv | string | 0 | — |
| ScoreType | combo | WDL\_mu | centipawn; centipawn\_with\_drawscore; centipawn\_2019; centipawn\_2018; win\_percentage; Q; W-L; WDL\_mu |
| PerPVCounters | check | false | — |
| UCI\_ShowMovesLeft | check | false | — |
| VerboseMoveStats | check | false | — |
| LogFile | string | &lt;empty&gt; | — |
| ConfigFile | string | mouse.config | — |
| TimeManager | string | legacy | — |
| BackendOptions | string | &lt;empty&gt; | — |
| SyzygyPath | string | &lt;empty&gt; | — |
| SyzygyFastPlay | check | false | — |
<!-- END MOUSE UCI OPTIONS -->

## Licences

Mouse is GPL-3.0-or-later. The package includes ONNX Runtime (MIT, `licenses/ONNXRuntime-LICENSE`
and `licenses/ThirdPartyNotices.txt`) and Microsoft DirectML, which Microsoft licenses for
distribution with applications (`licenses/DirectML-LICENSE.txt`,
`licenses/DirectML-ThirdPartyNotices.txt`).
