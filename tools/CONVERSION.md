# T1 conversion

Install Python 3.12 and the matching CPU converter lock before installing the
package. These locks are separate from the engine GPU environment:

```powershell
python -m pip install --require-hashes -r tools/requirements-converter-windows-cpu.lock
python -m pip install --no-deps --no-build-isolation ./tools
```

On Linux x64, use `tools/requirements-converter-linux-cpu.lock` instead. Its
PyTorch wheel requires glibc 2.28 or later. Both locks pin the official CPU wheel
by direct URL and publisher SHA256; the `converter` extra records direct package
requirements, while the lock pins the complete converter/development environment.
For Windows `-GpuPython`, use `requirements-converter-windows-gpu.lock`: it pairs CPU Torch with GPU ORT and the accepted TensorRT SDK wheel, and excludes CPU ORT. Use the matching lock followed by `--no-deps` package installation; do not combine CPU/GPU ORT extras or platform locks.

Run downloaded-file readers in isolated mode:

```powershell
python -I -m mouse_tools.convert nets/T1-256x10-distilled-swa-2432500.pb.gz nets/t1.onnx --sha256 bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406
```

`python -I tools/convert.py` exposes the same command. Existing outputs require
`--overwrite`; publication is atomic and failed conversion preserves the existing
artifact. JSON output includes both file hashes, the actual learned parameter
count, raw/effective formats, explicit compatibility upgrades, and export warnings.
The converter never invokes Lc0 and never runs GPU inference.

## Supported architecture

This converter supports global LINEAR16, classical112 input, a gated PE_MAP
attention body with per-block smolgen and squared-ReLU FFNs, Mish embedding/head
activations, attention policy, WDL and moves-left v1. Every tensor length and
attention dimension is checked. No convolution, batchnorm, relative-position
attention, multihead head-selection, dense positional embedding or other layer
encoding is claimed supported; known unsupported combinations fail with a reason.
The actual T1 contains 274 learned tensors / 20,198,188 parameters.

The actual legacy network enum4 with encoder weights is explicitly upgraded to
attention6. Its shared smolgen triggers the pinned Lc0 loader's Swish/squared-ReLU
overrides. Only an absent input-embedding field receives PE_MAP; a present zero is
rejected. Raw presence is retained in provenance. No history-fill default is
inferred from the input format.

The parser bounds compressed input to 64 MiB and decompressed input to 128 MiB,
checks gzip integrity, varints, field lengths/types, singular duplicates, enum
presence, finite float32 quantization ranges and all tensor shapes. Unknown
optional fields are skipped only after valid wire boundaries; no upstream proto,
generated parser or lookup table is vendored.

## Export contract

FP32, ONNX IR8 / opset17, self-contained tensor data, dynamic `batch`:

| Name | Shape | Meaning |
|---|---|---|
| `planes` | `[batch,112,8,8]` | Already encoded float32 planes |
| `policy` | `[batch,1858]` | Raw logits; no policy softmax |
| `wdl` | `[batch,3]` | Softmax probabilities in W,D,L order |
| `mlh` | `[batch,1]` | Nonnegative moves-left output |

Required metadata keys and values:

| Key | Value |
|---|---|
| `mouse.schema_version` | `1` |
| `mouse.converter_version` | `0.2.0-dev` |
| `mouse.source_sha256` | SHA256 of the original compressed source bytes |
| `mouse.input_format` | `1` |
| `mouse.network_format` | `6` |
| `mouse.raw_network_format` | `4` for legacy T1, `6` for explicit supported attention |
| `mouse.policy_format` | `3` |
| `mouse.value_format` | `2` |
| `mouse.moves_left_format` | `1` |
| `mouse.policy_mapping` | `lc0-1858-v1` |
| `mouse.learned_parameter_count` | Actual decoded learned element count |

The consumer must inspect actual tensor names/types/shapes as well as metadata.
Classical input encoding and history fill are separate contracts; conversion does
not choose or reconstruct board histories. The policy map has 1792 ordinary
geometric entries plus 66 explicit Q/R/B promotions; knight promotion uses its
ordinary coordinate entry. Castling uses king-to-rook coordinates.

## Independent numerical parity

First obtain the reference with the official Lc0 v0.32.1 `leela2onnx` command
(`lc0 leela2onnx --input=<t1 .pb.gz> --output=<reference .onnx>`). Pass an NPY file containing **at least 1000 distinct real
encoded positions** with dtype float32 and shape `[N,112,8,8]`; the upstream corpus
helper records its provenance and distinctness. The parity runner evaluates every
row, does not generate its own reference, checks its explicitly supplied SHA256,
uses only CPUExecutionProvider with fallback disabled, and reports all three heads:

```powershell
python -I -m mouse_tools.convert.parity nets/t1.onnx .toolchain/[redacted-reference]/t1-lc0-f32.onnx .toolchain/parity-planes.npy --reference-sha256 0e2699c19b617781d4fbe3d00286756b125836a30c0658afc620e197bec35d04
```

The absolute tolerance is fixed at `1e-4`. `--batch-size` controls memory, not count.
`--diagnostic` allows a short debugging corpus and always reports
`gate_satisfied=false`. `--disable-optimizations` helps isolate graph differences;
the normal gate uses optimized CPU execution. This FP32 conversion check does not
satisfy the later GPU/FP16 engine parity gate.

## Primary sources and dependencies

Behavioral facts are pinned to [Lc0 fd71a2d9](https://github.com/LeelaChessZero/lc0/tree/fd71a2d921b689c5f479d3227c3806c8e272d9c5),
particularly `src/neural/loader.cc`, `src/utils/weights_adapter.cc`,
`src/neural/onnx/converter.cc` and `src/neural/encoder.cc`, and
[lczero-common b326b154 schema](https://github.com/LeelaChessZero/lczero-common/blob/b326b154221a6eb91977bdccf11d6f89e8547875/proto/net.proto).
The implementation and geometric table generation are original.

PyTorch 2.14.1+cpu is pinned from the [official CPU index](https://download.pytorch.org/whl/cpu/torch/).
Its [2.14 export documentation](https://docs.pytorch.org/docs/2.14/onnx.html) supports
explicit `dynamo=False`; legacy-export deprecation warnings are retained in JSON
provenance. NumPy 2.2.6, ONNX 1.18.0 and ORT CPU 1.22.0 were checked with an independent
CPU round-trip before converter implementation. Every lock wheel was hashed against
its official PyPI release JSON or the PyTorch CPU index. Linux wheel metadata was
resolved with Linux markers independently; Linux execution remains a separate
acceptance step where no local Linux runtime is available.
