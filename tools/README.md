# Mouse Python tools

Python 3.12 is required. Install the package normally so isolated mode can import
it; scripts never modify `sys.path`. Run downloaded-file readers with `python -I`.

Create an environment using python.org Python (`py -3.12 -m venv .venv` on
Windows), activate it, and install exactly one platform lock for the complete
Phase 1 tooling and test suite:

```powershell
# Windows x64 CPU tools and converter:
python -I -m pip install --only-binary=:all: --require-hashes -r tools/requirements-converter-windows-cpu.lock
# Linux x64 uses requirements-converter-linux-cpu.lock instead.
python -I -m pip install --no-deps --no-build-isolation ./tools
python -I -m pip check
python -I -m pytest tools
python -I -m ruff check tools
python -I -m mypy --strict --config-file tools/pyproject.toml tools
```

The converter locks pin CPU Torch 2.14.1+cpu by an official direct wheel URL,
ONNX 1.18.0, NumPy 2.2.6, ORT 1.22.0, and all development/build dependencies.
Linux requires glibc 2.28 or later. Every wheel is hash-pinned from official PyPI
release metadata or the PyTorch CPU index; `--only-binary=:all:` forbids source
builds. The bootstrap script selects the Windows CPU converter lock by default.

For Windows GPU tools, use `requirements-converter-windows-gpu.lock` in a
separate environment. This combines CPU Torch with GPU ORT and TensorRT,
and excludes CPU ORT. The two ORT distributions export the same Python module;
do not install both or combine platform locks/extras. `scripts/build.ps1 -GpuPython`
selects the combined GPU lock.

TensorRT 10.9.0.34 comes from the accepted NVIDIA SDK archive
`TensorRT-10.9.0.34.Windows.win10.cuda-12.8.zip`, at
`python/tensorrt-10.9.0.34-cp312-none-win_amd64.whl`. Its SHA-256 is
`5585b4b1832ac0dc5c6d62f0a02314661ab3f88196f1e7e188c3df8378e2cfb6`.
The archive provenance is in [TOOLCHAIN.md](../docs/TOOLCHAIN.md). With the
accepted wheels in `.toolchain/downloads/python-wheels`, an offline install is:

```powershell
python -I -m pip install --no-index --find-links .toolchain/downloads/python-wheels --only-binary=:all: --require-hashes -r tools/requirements-converter-windows-gpu.lock
python -I -m pip install --no-deps --no-build-isolation ./tools
```

The earlier `requirements-dev.lock`, `requirements-cpu.lock` and
`requirements-windows-gpu.lock` remain unchanged for foundation reproducibility.
They do not contain Torch and cannot run the complete converter test suite.
See [CONVERSION.md](CONVERSION.md) for supported architecture, metadata and
numerical parity commands.

## Download the Phase 1 net

```powershell
python -I tools/fetch_net.py --destination nets
# Equivalent after package installation:
python -I -m mouse_tools.fetch_net --destination nets
```

The CLI uses the exact official Lc0 storage URL for
`t1-256x10-distilled-swa-2432500.pb.gz`. The pinned SHA-256 is
`bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406`, and the
expected length is 37,118,673 bytes. This hash records the accepted downloaded
artifact; Lc0 does not publish a checksum for this file. Existing files are
validated before reuse. A corrupt existing file fails clearly; `--force`
explicitly requests replacement. Transfers use a separate file in the target
directory and replace the destination only after length and digest validation.
An interruption or failed checksum leaves an existing destination intact and
removes the partial file. No alternate net or URL is tried.

## Reproduce the SDK test fixture

`tests/data/toolchain.onnx` is a Mouse-authored MatMul/Add/Relu graph, not a chess
network or vendored upstream source. It has input `input` of shape `[4,32]` and
output `output` of shape `[4,16]`, ONNX opset 17 and IR version 10. Generate it with:

```powershell
python -I -m mouse_tools.make_toolchain_fixture --output tools/tests/data/toolchain.onnx
```

Weights at flat index `i` are `((i % 17) - 8) / 16`, bias at index `j` is
`j / 16`, and test input at flat index `i` is `i / 16`. The output is
`Relu(input @ weights + bias)`. The CPU tests compare all 64 outputs against an
independent literal reference captured at toolchain acceptance. The native
GPU-tagged tests consume the same graph. These tests prove SDK execution only;
they do not prove T1 conversion, Lc0 parity, chess correctness or throughput.
