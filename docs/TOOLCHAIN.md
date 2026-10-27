# Mouse toolchain

The versions Mouse is built and tested with. Other versions may work, but these
are the ones the release was verified on.

## Build tools (Windows x64)

| Component | Version | Notes |
|---|---|---|
| Visual Studio 2022 Build Tools | 17.14 (MSVC 14.44) | "Desktop development with C++" workload and a Windows SDK |
| Windows SDK | 10.0.26100.0 | |
| CMake | ≥ 3.28 (tested 3.31.6) | The copy bundled with VS 2022 works |
| Ninja | 1.12.1 | The copy bundled with VS 2022 works |
| Git for Windows | any recent | |
| Python | 3.12 (python.org) | Only for the tools: net download, conversion, match runner |
| vcpkg | baseline `a5f736224e85b32e25a8ffc52562333180396642` | Fetched by `scripts/build.ps1`; provides Catch2 3.16.0 and tl-expected 1.3.1 |

No CUDA compiler is needed: Mouse has no custom CUDA kernels.

## GPU runtime

| Component | Version | Source |
|---|---|---|
| NVIDIA driver | ≥ 570.65 (tested 617.42) | |
| CUDA Toolkit | 12.8 Update 1 | [NVIDIA CUDA 12.8.1](https://developer.nvidia.com/cuda-12-8-1-download-archive) |
| cuDNN | 9.8.0.87 for CUDA 12 | [cuDNN 9.8 archive](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip) |
| TensorRT | 10.9.0.34 for CUDA 12.8 | [NVIDIA TensorRT 10.x downloads](https://developer.nvidia.com/tensorrt/download/10x) (NVIDIA account required) |
| ONNX Runtime GPU | 1.22.0 | Bundled with the release; [ORT 1.22.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0) |

Tested GPU: GeForce RTX 5070 (Blackwell, compute capability 12.0). ONNX Runtime
1.22 pairs with TensorRT 10.9 and CUDA 12.x, per its
[TensorRT provider requirements](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements).

The NVIDIA libraries are not bundled. Mouse does not search `PATH` or the current
directory for them: pass their directories through the `GpuDllDirectories` UCI
option (see the README).

## Download hashes

SHA-256 of the files Mouse was verified with:

| File | Bytes | SHA-256 |
|---|---:|---|
| `t1-256x10-distilled-swa-2432500.pb.gz` | 37,118,673 | `bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406` |
| `onnxruntime-win-x64-gpu-1.22.0.zip` | 312,740,706 | `5b5241716b2628c1ab5e79ee620be767531021149ee68f30fc46c16263fb94dd` |
| `cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip` | 675,349,654 | `d8a23705e3884b137b7e05449fb2b61bfa524e7cfc3fda80743d633f423c6ce4` |
| `TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip` | 1,845,842,538 | `c2758eb60191f01a47b24f54700e5463f577ebe129cd18fe835d0aa9f1e1a16d` |
| `lc0-v0.32.1-windows-gpu-nvidia-cuda12.zip` (reference engine for testing) | 581,970,338 | `8d0ce17676eb15e303bea9e790742d31c94ce5d24107f6187adc58e820f6d2f7` |
| `fastchess-windows-x86-64.zip` (match runner, v1.8.2-alpha) | 7,261,969 | `007ce550dc809510b34622a1aeb43fd443d40284045bef5c1bc39c8941114143` |

`tools/fetch_net.py` checks the T1 file's size and SHA-256 before conversion.
The T1 network comes from [Lc0's storage](https://storage.lczero.org/files/networks-contrib/).

## Python tools

The tools install from hash-locked requirement files in `tools/`:

- `requirements-converter-windows-cpu.lock`: net download and conversion (CPU PyTorch 2.14.1, ONNX 1.18.0, NumPy 2.2.6, ORT 1.22.0)
- `requirements-converter-windows-gpu.lock`: the same with ORT GPU and TensorRT 10.9.0.34
- `requirements-converter-linux-cpu.lock`: Linux CPU conversion
- `requirements-dev.lock`, `requirements-cpu.lock`, `requirements-windows-gpu.lock`: development and tests

Install with `pip install --only-binary=:all: --require-hashes -r <lock>`.
