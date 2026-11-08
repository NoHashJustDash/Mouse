# Mouse on AMD, Intel and other DirectX 12 GPUs (DirectML)

The DirectML build of Mouse runs its network through ONNX Runtime's DirectML
provider, so it works on any GPU with a DirectX 12 driver: AMD Radeon, Intel Arc
and integrated graphics, and NVIDIA. On NVIDIA GPUs the CUDA/TensorRT build is much
faster; use DirectML there only if that build cannot run.

Built with the `windows-dml` CMake preset after `scripts/stage-directml-sdk.ps1` has
assembled the pinned SDK: ONNX Runtime 1.22.0 (DirectML package) and DirectML 1.15.4.
Its ONNX Runtime has no CUDA or TensorRT provider, so this build offers the `directml`
and `cpu` backends only, and `directml` is the default.

## Requirements

- 64-bit Windows 10 version 1803 or newer, or Windows 11.
- A GPU with a current DirectX 12 driver.
- The files of the package kept together: `mouse.exe`, `onnxruntime.dll`,
  `onnxruntime_providers_shared.dll` and `DirectML.dll`. Mouse refuses to start a
  DirectML session if Windows loads any other `DirectML.dll`, such as the older one
  in System32.

## Setup

1. Add `mouse.exe` to your chess GUI as a UCI engine.
2. Set `WeightsFile` to the `.onnx` network.
3. Set `Precision` to match the network: `fp32` (the default) for an FP32 network,
   `fp16` for an FP16 one. DirectML does not convert precision, so a mismatch fails
   with a message that says so.

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
| CPuct | string | 1.745 | — |
| CPuctBase | string | 38739 | — |
| CPuctFactor | string | 3.894 | — |
| FpuValue | string | 0.33 | — |
| FpuStrategyAtRoot | combo | same | same; reduction; absolute |
| FpuValueAtRoot | string | 1 | — |
| PolicyTemperature | string | 1.359 | — |
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
<!-- END MOUSE UCI OPTIONS -->

## Licences

Mouse is GPL-3.0-or-later. The package includes ONNX Runtime (MIT, `licenses/ONNXRuntime-LICENSE`
and `licenses/ThirdPartyNotices.txt`) and Microsoft DirectML, which Microsoft licenses for
distribution with applications (`licenses/DirectML-LICENSE.txt`,
`licenses/DirectML-ThirdPartyNotices.txt`).
