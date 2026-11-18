# SPDX-License-Identifier: GPL-3.0-or-later
"""scripts/setup-gpu.ps1 fills the gpu folder from verified archives, or fails loudly.

The tests run the real script against a small stand-in manifest whose archives are
already in ``-ArchiveDirectory``, so nothing is downloaded.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY / "scripts" / "setup-gpu.ps1"
MANIFEST = REPOSITORY / "scripts" / "gpu-packages.json"

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell script")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_archive(path: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def stage(
    tmp_path: Path, *, wrong_archive_hash: bool = False, wrong_file_hash: bool = False
) -> tuple[Path, Path]:
    """A copy of the script with a two-package manifest and both archives present."""
    work = tmp_path / "mouse"
    work.mkdir()
    shutil.copy(SCRIPT, work / "setup-gpu.ps1")
    archives = tmp_path / "archives"
    archives.mkdir()
    first = {"pkg-a/bin/alpha.dll": b"alpha library", "pkg-a/bin/unused.dll": b"not wanted"}
    second = {"pkg-b/lib/beta.dll": b"beta library"}
    make_archive(archives / "a.zip", first)
    make_archive(archives / "b.zip", second)
    a_bytes = (archives / "a.zip").read_bytes()
    b_bytes = (archives / "b.zip").read_bytes()
    manifest = {
        "schema": 1,
        "licenses": [{"name": "Example licence", "url": "https://example.invalid/licence"}],
        "packages": [
            {
                "name": "Package A",
                "url": "https://example.invalid/a.zip",
                "archive": "a.zip",
                "size": len(a_bytes),
                "sha256": "0" * 64 if wrong_archive_hash else sha256(a_bytes),
                "files": [{"entry": "pkg-a/bin/alpha.dll"}],
            },
            {
                "name": "Package B",
                "url": "https://example.invalid/b.zip",
                "archive": "b.zip",
                "size": len(b_bytes),
                "files": [
                    {
                        "entry": "pkg-b/lib/beta.dll",
                        "sha256": "1" * 64 if wrong_file_hash else sha256(b"beta library"),
                    }
                ],
            },
        ],
    }
    (work / "gpu-packages.json").write_text(json.dumps(manifest), encoding="utf-8")
    return work, archives


def run(work: Path, archives: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(work / "setup-gpu.ps1"),
            "-ArchiveDirectory",
            str(archives),
            "-SkipGpuCheck",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_installs_only_the_listed_files_into_the_gpu_folder(tmp_path: Path) -> None:
    work, archives = stage(tmp_path)
    result = run(work, archives, "-AcceptNvidiaLicenses")
    assert result.returncode == 0, result.stdout + result.stderr
    gpu = work / "gpu"
    assert sorted(p.name for p in gpu.iterdir()) == ["alpha.dll", "beta.dll"]
    assert (gpu / "alpha.dll").read_bytes() == b"alpha library"
    assert (gpu / "beta.dll").read_bytes() == b"beta library"
    # The user's own archives are left where they were.
    assert (archives / "a.zip").is_file()


def test_refuses_without_licence_acceptance_and_installs_nothing(tmp_path: Path) -> None:
    work, archives = stage(tmp_path)
    result = run(work, archives)
    assert result.returncode != 0
    assert "https://example.invalid/licence" in result.stdout + result.stderr
    assert "AcceptNvidiaLicenses" in result.stdout + result.stderr
    assert not (work / "gpu").exists()


def test_an_archive_with_the_wrong_hash_stops_the_setup(tmp_path: Path) -> None:
    work, archives = stage(tmp_path, wrong_archive_hash=True)
    result = run(work, archives, "-AcceptNvidiaLicenses")
    assert result.returncode != 0
    assert "SHA-256" in result.stdout + result.stderr
    assert not (work / "gpu" / "alpha.dll").exists()


def test_a_library_with_the_wrong_hash_is_not_installed(tmp_path: Path) -> None:
    work, archives = stage(tmp_path, wrong_file_hash=True)
    result = run(work, archives, "-AcceptNvidiaLicenses")
    assert result.returncode != 0
    assert "beta.dll" in result.stdout + result.stderr
    assert not (work / "gpu" / "beta.dll").exists()


def test_a_second_run_finds_everything_installed(tmp_path: Path) -> None:
    work, archives = stage(tmp_path)
    assert run(work, archives, "-AcceptNvidiaLicenses").returncode == 0
    # With the archives gone, a second run can only succeed by seeing the installed files.
    shutil.rmtree(archives)
    archives.mkdir()
    result = run(work, archives, "-AcceptNvidiaLicenses")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "already installed" in result.stdout


def test_the_shipped_manifest_lists_what_the_engine_loads() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    names = {Path(f["entry"]).name for p in manifest["packages"] for f in p["files"]}
    # The libraries a fresh TensorRT engine build and run were verified with (2026-11-16).
    assert names == {
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cufft64_11.dll",
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_ops64_9.dll",
        "nvinfer_10.dll",
        "nvinfer_builder_resource_10.dll",
        "nvinfer_plugin_10.dll",
        "nvonnxparser_10.dll",
    }
    for package in manifest["packages"]:
        assert package["url"].startswith("https://developer.download.nvidia.com/")
        assert package["size"] > 0
        # Every package is verified: by its archive hash, or by a hash on each file.
        assert "sha256" in package or all("sha256" in f for f in package["files"])


def test_downloads_a_missing_archive_with_powershell_redirecting_output(tmp_path: Path) -> None:
    """The real download path, against a local server, run as `setup-gpu.ps1 *> log`.

    Regression: with PowerShell redirecting the script's streams, Windows PowerShell 5.1
    turned curl's progress output into a terminating error and the setup died at the
    first download.
    """
    import functools
    import http.server
    import threading

    work, archives = stage(tmp_path)
    served = tmp_path / "served"
    served.mkdir()
    shutil.move(str(archives / "a.zip"), served / "a.zip")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log = tmp_path / "setup.log"
    try:
        manifest_path = work / "gpu-packages.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["packages"][0]["url"] = f"http://127.0.0.1:{server.server_port}/a.zip"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        command = (
            f"& '{work / 'setup-gpu.ps1'}' -ArchiveDirectory '{archives}' -SkipGpuCheck "
            f"-AcceptNvidiaLicenses *> '{log}'"
        )
        subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
    text = log.read_text(encoding="utf-16") if log.exists() else "<no log>"
    assert (work / "gpu" / "alpha.dll").is_file(), text
    assert (work / "gpu" / "alpha.dll").read_bytes() == b"alpha library"
    assert (archives / "a.zip").read_bytes() == (served / "a.zip").read_bytes()
    assert not (archives / "a.zip.part").exists()
    assert "Done." in text
