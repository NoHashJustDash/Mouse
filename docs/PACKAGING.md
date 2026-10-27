# Release packaging

Use a no-Git source export with its own configured Windows Release build.
The source and output folders must be separate. Set the exact intended version
at configure time, for example `-DMOUSE_VERSION=1.0.0-rc.1`. Do not set a Git hash
for an export that has no commit yet. Initialize the public repository afterward.

From a VS 2022 x64 developer shell with the installed tools package:

```powershell
python -I -m mouse_tools.package --source C:/build/mouse-source --build C:/build/mouse-source/build/windows-release --version 1.0.0-rc.1 --output C:/build/mouse-release
```

The command rebuilds `mouse` in that source folder, runs CMake install, checks the
exact eleven installed files, verifies pinned ORT and license hashes, validates
PE imports, and smoke-tests `--version` and `--help` from ZIP extraction. It emits
the Windows x64 ZIP, corresponding-source ZIP, SHA256SUMS, and manifest.json.
Source snapshots, configuration, built and installed binaries remain hash-bound
through publication. Existing output is immutable. This validates artifacts,
not chess strength or tournament release gates; `release_gate` remains false.
No network weights or NVIDIA runtime DLLs are included. Four Microsoft ONNX
Runtime DLLs are bundled, including its CUDA and TensorRT provider adapters.
Install the documented NVIDIA dependencies separately.

After extracting the binary package, install the tools dependencies and module:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -I -m pip install --only-binary=:all: --require-hashes -r tools/requirements-converter-windows-cpu.lock
.venv/Scripts/python -I -m pip install --no-build-isolation --no-deps ./tools
.venv/Scripts/python -I -m mouse_tools.setup_net --directory nets
```

The download validates the pinned byte count and SHA-256, then conversion verifies
the same source SHA-256 and produces FP16 ONNX. Failed downloads never proceed to
conversion. Existing ONNX output is not overwritten implicitly.
