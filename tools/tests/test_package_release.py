# SPDX-License-Identifier: GPL-3.0-or-later
"""Release packaging must bind an explicit source tree to its own CMake build."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
from test_package import fixture, sha

from mouse_tools import package


def release_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    _, document = fixture(tmp_path, monkeypatch)
    source = tmp_path / "source tree"
    source.mkdir()
    with zipfile.ZipFile(document["source_zip"]) as archive:
        for name in archive.namelist():
            target = source / name.removeprefix("Mouse-source/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    for name in ("tools/setup_net.py", "tools/src/mouse_tools/setup_net.py"):
        (source / name).write_text("# authored source fixture\n", encoding="utf-8")
    build = source / "build/windows-release"
    build.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source.as_posix()}\n"
        "CMAKE_BUILD_TYPE:STRING=Release\nMOUSE_VERSION:STRING=1.0.0-rc.1\n"
        "MOUSE_GIT_HASH:STRING=\n",
        encoding="utf-8",
    )
    stage = Path(document["install_stage"])
    shutil.copyfile(stage / "mouse.exe", build / "mouse.exe")

    def run(command: list[str], cwd: Path) -> None:
        if command[1] == "--install":
            shutil.copytree(stage, Path(command[-1]))

    monkeypatch.setattr(package, "_release_command", run, raising=False)

    def smoke(executable: Path, flag: str) -> list[str]:
        result = (
            "Mouse 1.0.0-rc.1\ngit: unknown\nbuild: Release\n"
            "compiler: authored child\nONNX Runtime: 1.22.0\n"
            if flag == "--version"
            else "Usage: mouse uci --version --help\n"
        )
        return [sys.executable, "-I", "-c", f"print({result!r},end='')"]

    monkeypatch.setattr(package, "_smoke_command", smoke)
    return source, build


def test_release_packages_corresponding_source_and_checked_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, build = release_fixture(tmp_path, monkeypatch)
    output = tmp_path / "release"
    result = package.produce_release(source, build, "1.0.0-rc.1", output)
    assert result["status"] == "candidate_passed"
    assert result["release_gate"] is False
    assert result["source_provenance"] == "Rebuilt from the hash-bound source folder"
    assert result["commit"] == "unknown"
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ")
        assert sha((output / name).read_bytes()) == expected
    with zipfile.ZipFile(output / "mouse-1.0.0-rc.1-windows-x64.zip") as archive:
        assert "tools/fetch_net.py" in archive.namelist()
        assert "tools/setup_net.py" in archive.namelist()
        assert "tools/src/mouse_tools/setup_net.py" in archive.namelist()
        assert "nvcuda.dll" not in archive.namelist()
    with zipfile.ZipFile(output / "mouse-1.0.0-rc.1-source.zip") as archive:
        assert archive.comment == b"unknown"
        assert "Mouse-source/engine/main.cpp" in archive.namelist()
    manifest = json.loads((output / "manifest.json").read_text())
    assert str(source) not in json.dumps(manifest)


@pytest.mark.parametrize(
    "version", ["0.5.0-dev", "1.0", "01.0.0", "1.0.0-rc.0", "1.0.0-rc.01", "1.0.0+local", "unknown"]
)
def test_release_refuses_non_release_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    source, build = release_fixture(tmp_path, monkeypatch)
    with pytest.raises(package.PackageError, match="release version"):
        package.produce_release(source, build, version, tmp_path / "release")
    assert not (tmp_path / "release").exists()


@pytest.mark.parametrize("mutation", ["source", "version", "debug", "extra", "changed", "binary"])
def test_release_fails_closed_on_mismatched_or_changed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    source, build = release_fixture(tmp_path, monkeypatch)
    cache = build / "CMakeCache.txt"
    if mutation == "source":
        cache.write_text(cache.read_text().replace(source.as_posix(), tmp_path.as_posix()))
    elif mutation == "version":
        cache.write_text(cache.read_text().replace("1.0.0-rc.1", "1.0.0"))
    elif mutation == "debug":
        cache.write_text(cache.read_text().replace("Release", "Debug"))
    elif mutation == "extra":
        (tmp_path / "clean install with spaces/nvcuda.dll").write_bytes(b"forbidden")
    else:
        original = package._release_command

        def change(command: list[str], cwd: Path) -> None:
            original(command, cwd)
            if command[1] == "--install":
                target = (
                    source / "engine/main.cpp" if mutation == "changed" else build / "mouse.exe"
                )
                target.write_bytes(b"changed after build")

        monkeypatch.setattr(package, "_release_command", change)
    with pytest.raises(package.PackageError):
        package.produce_release(source, build, "1.0.0-rc.1", tmp_path / "release")
    assert not (tmp_path / "release/manifest.json").exists()


def test_setup_net_pins_download_and_converter_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mouse_tools import fetch_net, setup_net
    from mouse_tools.convert import cli

    calls: list[list[str]] = []

    def capture(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(fetch_net, "main", capture)
    monkeypatch.setattr(cli, "main", capture)
    monkeypatch.setattr(setup_net, "_isolated", lambda: True)
    assert setup_net.main(["--directory", str(tmp_path)]) == 0
    assert calls[0] == ["--destination", str(tmp_path)]
    assert calls[1][-4:] == ["--sha256", fetch_net.T1_SHA256, "--precision", "fp16"]


def test_setup_net_stops_on_failed_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from mouse_tools import fetch_net, setup_net
    from mouse_tools.convert import cli

    monkeypatch.setattr(fetch_net, "main", lambda argv: 1)
    monkeypatch.setattr(cli, "main", lambda argv: pytest.fail("conversion after bad download"))
    monkeypatch.setattr(setup_net, "_isolated", lambda: True)
    assert setup_net.main(["--directory", str(tmp_path)]) == 1


@pytest.mark.parametrize(
    "generated", ["mouse_tools.egg-info", "build", "dist", ".venv", ".toolchain"]
)
def test_release_ignores_generated_directories_in_source_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated: str
) -> None:
    source, _ = release_fixture(tmp_path, monkeypatch)
    expected = package._release_inventory(source)
    directory = source / "tools/src" / generated
    directory.mkdir()
    (directory / "generated.txt").write_text("generated local metadata", encoding="utf-8")
    assert package._release_inventory(source) == expected
