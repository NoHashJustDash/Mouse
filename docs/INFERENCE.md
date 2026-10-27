# Inference commands

The engine uses ONNX files with the metadata and tensor contract documented in
[CONVERSION.md](../tools/CONVERSION.md). Select the provider and precision
explicitly. CPU inference accepts FP32. CUDA FP16 requires a genuine FP16 compute
graph; export one with `--precision fp16`. TensorRT enables its FP16 builder
option explicitly. Inputs and outputs remain float32 in both artifacts.

The native backend serializes session, binding and scratch use. Calls own their
output storage, and failed calls leave it unchanged. GPU inference uses actual
pinned host allocations, I/O binding and synchronization before publishing
outputs. CPU fallback is disabled for GPU sessions. Missing providers, unsupported
graphs, invalid metadata, allocation failures and corrupt caches are errors.

On Windows, pinned ORT/provider DLLs are staged beside the executable. Supply
absolute CUDA, cuDNN and TensorRT SDK library directories explicitly; the engine
registers those directories without relying on PATH. A build without CUDA SDK
headers supports CPU inference and rejects GPU requests clearly. No NVCC or CUDA
import library is required. Linux dependency loading uses explicit absolute SDK
libraries and keeps their dependency closure alive through session destruction.
Linux GPU execution remains unverified locally.

## End-to-end throughput

From a configured Windows build, a full sweep is:

```powershell
.\build\windows-release\mouse.exe backendbench --weights C:/nets/t1-fp16.onnx --backend cuda --precision fp16 --gpu-dll-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin" --gpu-dll-dir C:/sdk/cudnn/bin --gpu-dll-dir C:/sdk/tensorrt/lib --start-batch-size 1 --max-batch-size 1024 --batch-step 1 --batches 100
.\build\windows-release\mouse.exe backendbench --weights C:/nets/t1-fp16.onnx --backend tensorrt --precision fp16 --gpu-dll-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin" --gpu-dll-dir C:/sdk/cudnn/bin --gpu-dll-dir C:/sdk/tensorrt/lib --cache-dir C:/cache/mouse-trt --start-batch-size 1 --max-batch-size 1024 --batch-step 1 --batches 100
```

Replace example SDK directories with the actual installed library directories.
Run these commands serially, with no other GPU work. Defaults are exactly the
full 1 through 1024 sweep, step 1, 100 timed evaluations per batch size.
A smaller range is a diagnostic, and cannot substitute for that measurement.
The CPU provider is available for CI diagnostics with `--precision fp32`.

Each row reports the actual physical batch, timed completed positions/second,
elapsed time, session initialization and one additional untimed warm-up.
Timing includes buffer expansion, binding, upload, inference, download,
synchronization, output validation and publication. Result caching is disabled.
CUDA specializes each batch size in its own session outside timed evaluation;
CPU and TensorRT share a dynamic session with actual requested N. Constructor
warm-up is included in initialization time. Record process startup separately
when measuring cold versus warm CUDA JIT or TensorRT cache behavior.

The header records build version/hash/compiler, ORT version, model digest,
device/driver/CUDA/TensorRT identity and configuration. Model bytes are pinned
before construction, checked against every session and checked again before the
best-result summary. Partial output from a failed sweep is not a passing result.

TensorRT caches use a namespace derived from actual model bytes, GPU UUID/SM,
runtime/provider versions and binaries, precision, workspace and profile options.
A process-held lock protects construction; successful warm-up publishes a
checksum manifest. Incomplete, mismatched or corrupt caches fail explicitly.

## Deterministic evaluation transport

`backend-eval` reads a bounded file and writes binary stdout only after every
chunk succeeds. Errors go to stderr with a nonzero exit code:

```powershell
.\build\windows-release\mouse.exe backend-eval --weights C:/nets/t1.onnx --backend cpu --precision fp32 --inputs tests/data/nn/backend-parity-v1/planes.bin --buckets 1,8,32,64 --nncache-mb 0
```

Buckets are increasing, distinct integers from 1 through 1024. CUDA pads to the
smallest configured bucket that fits a request; padding never returns as an
extra result. CPU/TensorRT execute actual N up to the largest bucket. Every
configured bucket is warmed during construction. The parity runner validates
all 50 complete rows, including their legal move mappings:

```powershell
python -I -m mouse_tools.backend_parity --engine build/windows-release/mouse.exe --weights C:/nets/t1-fp16.onnx --reference C:/reference/t1-lc0-f32.onnx --cases tests/data/nn/backend-parity-v1/cases.json --inputs tests/data/nn/backend-parity-v1/planes.bin --backend cuda --precision fp16 --buckets 1,8,32,64 --gpu-dll-dir "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin" --gpu-dll-dir C:/sdk/cudnn/bin --gpu-dll-dir C:/sdk/tensorrt/lib
```

The official Lc0 reference is pinned by digest. Legal policy softmax uses
temperature 1. Policy probability and WDL absolute errors must each be at most
0.005. CPU reports are diagnostics and do not satisfy either GPU gate.

Input wire format: ASCII `MOUSEIN1`, uint32 little-endian row count (1..1024),
uint32 plane count (112), followed by each row's 112 (uint64 mask, float32 value)
pairs. Every value must be finite; exact length is required. Each row is 1344
bytes. Empty masks retain their value bits in cache identity.

Output: ASCII `MOUSEOU1`, uint32 row count, uint32 width (1862), 64 ASCII
lowercase hexadecimal model SHA256 bytes, 64 input SHA256 bytes, then float32
little-endian rows of 1858 raw policy logits, W/D/L and moves-left. Every policy
value must be finite; WDL values must be finite in [0,1] with positive sum;
moves-left must be finite and nonnegative. No failed chunk publishes a prefix.

The optional bounded LRU keys all 112 masks and exact float32 bits, with complete
equality after hashing, within one immutable backend identity. Its byte charge
includes control, copied identity, nodes and index capacities. Zero disables
storage. It preallocates all valid-call working memory, deduplicates repeated
misses and keeps staged hits valid through eviction.

## Options

| Option | Accepted values / default |
|---|---|
| `--weights` | Required ONNX path |
| `--backend` | Required cpu, cuda or tensorrt |
| `--precision` | Required fp32 or fp16 |
| `--gpu-dll-dir` | Repeatable existing absolute SDK library directory |
| `--cache-dir` | Writable absolute TensorRT cache root (created when absent), required for TensorRT |
| `--device` | 0..1023, default 0 |
| `--cpu-threads` | 1..128, default 1 |
| `--workspace-mb` | 1..16384, default 1024 |
| `--inputs` | Required input transport path for backend-eval |
| `--buckets` | backend-eval: increasing distinct 1..1024; default 1,8,32,64 |
| `--nncache-mb` | backend-eval: 0..16384, default 0 |
| `--start-batch-size` | backendbench: 1..1024, default 1 |
| `--max-batch-size` | backendbench: start..1024, default 1024 |
| `--batch-step` | backendbench: 1..256, default 1 |
| `--batches` | backendbench: 1..1000000, default 100 |

There is no UCI search loop in this checkpoint. UCI option integration belongs
to the later search/protocol milestones.
