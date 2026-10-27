# SPDX-License-Identifier: GPL-3.0-or-later
"""Package hash-bound development inputs or a configured release source export.

Development plans consume prebuilt evidence. Release exports are rebuilt and
verified before packaging. Both interfaces produce local archives and manifests.
PE layout facts follow Microsoft's PE/COFF specification; the decoder is original.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import cast

from . import uci_gate

ORT_SHA256 = {
    "onnxruntime.dll": ("8c934c01f34702f60de665273876dbe9f70fcbbe0f7e0f292840ac0c13daa338"),
    "onnxruntime_providers_shared.dll": (
        "3b53c353cd52a7be926beb289277b79de1e659a1dc3bb74b24c99a6a7296d9d9"
    ),
    "onnxruntime_providers_cuda.dll": (
        "0f32c09da925ec58c650a0d72ccc950a73db570cae0adeffb3d07165419b8bd1"
    ),
    "onnxruntime_providers_tensorrt.dll": (
        "9466afbb9a1cae0594e6f78353c1aa37e283d16386a20d0fe9573ab95c780652"
    ),
}
LICENSE_SHA256 = {
    "LICENSE": ("3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"),
    "licenses/ONNXRuntime-LICENSE": (
        "c250d6278f0b47a6439fb7592b08b58a55eb9f535aa49a1db63211c3f982b674"
    ),
    "licenses/ThirdPartyNotices.txt": (
        "e00f828e0a33de591a355ae6606d2625f5758da7d2c844db7821c9dd3e3647b6"
    ),
    "licenses/tl-expected-LICENSE": (
        "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
    ),
}
STAGE_FILES = frozenset(
    {"mouse.exe", "README.md", "CHANGELOG.md"} | set(ORT_SHA256) | set(LICENSE_SHA256)
)
SUPPORT_REQUIRED = frozenset(
    {
        "tools/fetch_net.py",
        "tools/convert.py",
        "tools/README.md",
        "tools/CONVERSION.md",
        "tools/LICENSE",
        "tools/pyproject.toml",
        "tools/requirements-converter-windows-cpu.lock",
        "tools/src/mouse_tools/__init__.py",
        "tools/src/mouse_tools/fetch_net.py",
        "tools/src/mouse_tools/convert/cli.py",
        "docs/TOOLCHAIN.md",
    }
)
SOURCE_REQUIRED = SUPPORT_REQUIRED | frozenset(
    {
        "LICENSE",
        "README.md",
        "CHANGELOG.md",
        "CMakeLists.txt",
        "CMakePresets.json",
        "vcpkg.json",
        "engine/main.cpp",
    }
)
MAX_FILE = 512 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
MAX_MEMBERS = 10000
MAX_SMALL_TEST_FIXTURE = 128 * 1024
MAX_COMPRESSED_TEST_DATA = 16 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}")
# Unrelated projects are rejected by the closed SOURCE_DIRECTORIES allowlist.
FORBIDDEN = {".git", ".toolchain", "nets", ".venv", "__pycache__", "node_modules"}
SOURCE_DIRECTORIES = frozenset({"engine", "tests", "tools", "docs", "cmake", "scripts", ".github"})
SOURCE_ROOT_FILES = frozenset(
    {
        "AGENTS.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "README.md",
        "CHANGELOG.md",
        "CMakeLists.txt",
        "CMakePresets.json",
        "vcpkg.json",
        "vcpkg-configuration.json",
        ".gitignore",
        ".gitattributes",
        ".clang-format",
        ".clang-tidy",
    }
)
COMPILED = {
    ".exe",
    ".dll",
    ".lib",
    ".obj",
    ".o",
    ".a",
    ".so",
    ".dylib",
    ".pdb",
    ".pyd",
    ".pyc",
    ".whl",
    ".bin",
    ".pch",
    ".ilk",
    ".exp",
}
WINDOWS = frozenset(
    [
        "kernel32",
        "kernelbase",
        "ntdll",
        "user32",
        "gdi32",
        "advapi32",
        "ole32",
        "oleaut32",
        "combase",
        "shell32",
        "shlwapi",
        "ws2_32",
        "wsock32",
        "rpcrt4",
        "secur32",
        "crypt32",
        "bcrypt",
        "bcryptprimitives",
        "version",
        "dbghelp",
        "psapi",
        "winmm",
        "winhttp",
        "wininet",
        "setupapi",
        "iphlpapi",
        "userenv",
        "wtsapi32",
        "normaliz",
        "imm32",
        "powrprof",
        "avrt",
        "wintrust",
        "dxgi",
        "dxcore",
        "d3d11",
        "d3d12",
        "d3dcompiler_47",
        "dwrite",
    ]
)
PREREQUISITES = re.compile(
    r"(?:api-ms-win-(?:core|crt)-[a-z0-9-]+-l[0-9]+-[0-9]+-[0-9]+|"
    r"ext-ms-win-(?:ntuser|kernel|gdi|advapi32|shell)-[a-z0-9-]+-l[0-9]+-[0-9]+-[0-9]+|"
    r"ucrtbase|concrt140|vcomp140|vcruntime140(?:_[0-9]+)?|"
    r"msvcp140(?:_[0-9]+|_atomic_wait|_codecvt_ids)?|nvcuda|"
    r"cudart64_12|cublas(?:lt)?64_12|cufft64_11|curand64_10|"
    r"cudnn(?:_(?:adv|cnn|graph|ops|heuristic|engines_precompiled|engines_runtime_compiled))?64_9|"
    r"nvinfer(?:_plugin|_dispatch|_lean)?_10|"
    r"nvonnxparser_10|nvrtc64_[0-9]+_[0-9]+|nvrtc-builtins64_[0-9]+|nvjitlink_[0-9]+_[0-9]+)\.dll"
)


class PackageError(ValueError):
    """Unsafe, incomplete, changed, or unapproved candidate input/evidence."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise PackageError(message)


def safe_name(value: str) -> str:
    require(isinstance(value, str) and 0 < len(value) <= 240, "Invalid archive name length")
    parts = value.split("/")
    require(
        all(
            re.fullmatch(r"[A-Za-z0-9_+.-]+", p)
            and p not in (".", "..")
            and not p.endswith((".", " "))
            for p in parts
        ),
        "Unsafe archive path",
    )
    for part in parts:
        require(part.lower() not in FORBIDDEN, "Forbidden Phase 1 archive path")
        stem = part.split(".", 1)[0].upper()
        require(
            not re.fullmatch(r"CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9]", stem), "Reserved Windows path"
        )
    return value


def regular(path: Path) -> None:
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode)
        and not info.st_nlink < 1
        and not (getattr(info, "st_file_attributes", 0) & 0x400),
        "Require regular non-reparse file",
    )
    require(info.st_size <= MAX_FILE, "Artifact exceeds512MiB bound")


def digest(path: Path) -> str:
    regular(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 1024 * 1024))
            require(block, "Artifact truncated while hashing")
            h.update(block)
            remaining -= len(block)
        require(not stream.read(1), "Artifact grew while hashing")
    return h.hexdigest()


def object_value(value: object) -> dict[str, object]:
    require(
        isinstance(value, dict) and all(isinstance(k, str) for k in value), "Require JSON object"
    )
    return cast(dict[str, object], value)


def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in items:
        require(key not in value, "Duplicate JSON key")
        value[key] = item
    return value


def constant(value: str) -> object:
    raise PackageError(f"Nonfinite JSON number:{value}")


def absolute(value: object) -> Path:
    require(
        isinstance(value, str) and value and not any(c in value for c in "\x00\r\n"),
        "Invalid absolute path",
    )
    path = Path(cast(str, value))
    require(path.is_absolute(), "Require absolute paths")
    return path.absolute()


def hash_map(value: object) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in object_value(value).items():
        safe_name(key)
        require(isinstance(item, str) and HEX.fullmatch(item), "Invalid SHA256 entry")
        require(key.lower() not in {p.lower() for p in result}, "Case-aliased path")
        result[key] = cast(str, item)
    return result


def parse_pe(data: bytes) -> dict[str, object]:
    """Bounded PE32+ AMD64 normal/delay DLL imports, including RVA translation."""
    require(64 <= len(data) <= MAX_FILE and data[:2] == b"MZ", "Invalid DOS header")

    def unpack(fmt: str, offset: int) -> tuple[int, ...]:
        size = struct.calcsize(fmt)
        require(0 <= offset <= len(data) - size, "Truncated PE structure")
        return cast(tuple[int, ...], struct.unpack_from(fmt, data, offset))

    header = unpack("<I", 60)[0]
    require(header >= 64 and data[header : header + 4] == b"PE\0\0", "Invalid PE signature")
    machine, count = unpack("<HH", header + 4)
    optional_size = unpack("<H", header + 20)[0]
    optional = header + 24
    require(
        machine == 0x8664 and 1 <= count <= 96 and 240 <= optional_size <= 4096,
        "Require PE32+ AMD64",
    )
    require(
        unpack("<H", optional)[0] == 0x20B and optional + optional_size <= len(data),
        "Invalid PE32+ optional header",
    )
    image_base = unpack("<Q", optional + 24)[0]
    headers = unpack("<I", optional + 60)[0]
    require(
        0 < headers <= len(data) and unpack("<I", optional + 108)[0] >= 14, "Invalid PE directories"
    )
    sections = []
    for i in range(count):
        section = optional + optional_size + 40 * i
        virtual_size, va, size, raw = unpack("<IIII", section + 8)
        require(
            raw + size <= len(data) and va + max(virtual_size, size) <= 2**32,
            "Invalid PE section bounds",
        )
        sections.append((va, max(virtual_size, size), raw, size))

    def translate(rva: int, size: int) -> int:
        require(0 <= rva < 2**32 and size > 0, "Invalid PE RVA")
        matches = [
            (raw + rva - va)
            for va, span, raw, length in sections
            if va <= rva and rva + size <= va + span and rva - va + size <= length
        ]
        if rva + size <= headers:
            matches.append(rva)
        require(len(matches) == 1 and matches[0] + size <= len(data), "Unmapped/ambiguous PE RVA")
        return matches[0]

    def dll_name(rva: int) -> str:
        raw = bytearray()
        for i in range(261):
            byte = data[translate(rva + i, 1)]
            if byte == 0:
                break
            raw.append(byte)
        else:
            raise PackageError("Unterminated import name")
        try:
            name = raw.decode("ascii").lower()
        except UnicodeDecodeError as error:
            raise PackageError("Non-ASCII DLL import") from error
        require(re.fullmatch(r"[a-z0-9_.-]+\.dll", name), "Invalid DLL import name")
        return name

    imports: set[str] = set()
    for directory, width in ((1, 20), (13, 32)):
        rva, size = unpack("<II", optional + 112 + directory * 8)
        if not rva and not size:
            continue
        require(rva and width <= size <= 128 * 1024, "Invalid PE import directory")
        for i in range(min(size // width, 4096)):
            offset = translate(rva + i * width, width)
            descriptor = unpack("<" + "I" * (width // 4), offset)
            if not any(descriptor):
                break
            if directory == 13:
                require(descriptor[0] in (0, 1), "Unknown delay import attributes")
                name_rva = descriptor[1] if descriptor[0] == 1 else descriptor[1] - image_base
            else:
                name_rva = descriptor[3]
            imports.add(dll_name(name_rva))
        else:
            raise PackageError("Unterminated PE import descriptors")
    return {"machine": "AMD64", "format": "PE32+", "imports": sorted(imports)}


def stage_inventory(root: Path) -> set[str]:
    require(
        root.is_dir()
        and not root.is_symlink()
        and not (getattr(root.lstat(), "st_file_attributes", 0) & 0x400),
        "Require explicit non-reparse install stage",
    )
    found: set[str] = set()
    for path in root.iterdir():
        if path.name == "licenses":
            info = path.lstat()
            require(
                stat.S_ISDIR(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
                "Reparse license directory",
            )
            children = list(path.iterdir())
            require(len(children) == 3, "Exact license inventory required")
            for child in children:
                name = "licenses/" + child.name
                require(name in STAGE_FILES, "Unexpected license file")
                regular(child)
                found.add(name)
        else:
            require(
                path.name in STAGE_FILES, "Unexpected stage payload; never scan unknown directories"
            )
            regular(path)
            found.add(path.name)
    require(found == STAGE_FILES, "Exact11-file CMake install stage required")
    return found


@dataclass(frozen=True)
class Plan:
    config: Path
    stage: Path
    stage_hashes: Mapping[str, str]
    source: Path
    source_sha256: str
    source_root: str
    support: Mapping[str, str]
    fixtures: Mapping[str, str]
    version: str
    commit: str
    timeout: float
    identities: Mapping[Path, str]
    dependencies: Mapping[str, dict[str, object]]
    interpreter_alias: Path
    interpreter_target: Path
    release: bool = False
    release_source: Path | None = None
    release_files: Mapping[str, str] | None = None

    def verify(self) -> None:
        if self.release_source is not None:
            require(
                _release_inventory(self.release_source) == self.release_files,
                "Release source inventory changed",
            )
        require(
            self.interpreter_alias.resolve(strict=True) == self.interpreter_target,
            "Interpreter alias changed after plan freeze",
        )
        require(stage_inventory(self.stage) == set(self.stage_hashes), "Stage inventory changed")
        for name, expected in self.stage_hashes.items():
            require(digest(self.stage / name) == expected, f"Stage artifact changed:{name}")
        for path, expected in self.identities.items():
            require(digest(path) == expected, f"Frozen artifact changed:{path}")


def _archive_inventory(plan: Plan) -> dict[str, str]:
    hashes: dict[str, str] = {}
    aliases: set[str] = set()
    directories: dict[str, str] = {}
    total = 0
    fixtures: set[str] = set()
    with zipfile.ZipFile(plan.source) as archive:
        require(
            archive.comment == plan.commit.encode("ascii"),
            "Source archive clean commit comment differs",
        )
        members = archive.infolist()
        require(0 < len(members) <= MAX_MEMBERS, "Source archive member bound")
        prefix = plan.source_root + "/"
        for member in members:
            full = safe_name(member.filename.rstrip("/"))
            require(
                full == plan.source_root or full.startswith(prefix), "Source archive root mismatch"
            )
            require(full.lower() not in aliases, "Duplicate/case-aliased ZIP entry")
            aliases.add(full.lower())
            for ancestor in (PurePosixPath(full), *PurePosixPath(full).parents):
                spelling = str(ancestor)
                if spelling == ".":
                    continue
                require(
                    directories.get(spelling.lower(), spelling) == spelling,
                    "Case-aliased archive directory",
                )
                directories[spelling.lower()] = spelling
            if full != plan.source_root:
                relative = full.removeprefix(prefix)
                top = relative.split("/", 1)[0]
                require(
                    relative in SOURCE_ROOT_FILES
                    or (top in SOURCE_DIRECTORIES and ("/" in relative or member.is_dir())),
                    "Source ZIP path is outside the Phase1 allowlist",
                )
            mode = member.external_attr >> 16
            require(
                stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR),
                "ZIP symlink/special file refused",
            )
            require(
                not member.flag_bits & 1
                and member.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "Unsupported encrypted/compressed ZIP member",
            )
            if member.is_dir():
                require(stat.S_IFMT(mode) in (0, stat.S_IFDIR), "Directory type mismatch")
                continue
            require(
                stat.S_IFMT(mode) in (0, stat.S_IFREG) and full.startswith(prefix),
                "ZIP file type mismatch",
            )
            name = full.removeprefix(prefix)
            suffix = PurePosixPath(name).suffix.lower()
            binary_data = suffix == ".bin"
            network_fixture = name.lower().endswith((".onnx", ".pb", ".pb.gz"))
            compressed_data = name.lower().endswith((".v6.gz", ".jsonl.gz"))
            if network_fixture or compressed_data or binary_data:
                bound = MAX_COMPRESSED_TEST_DATA if compressed_data else MAX_SMALL_TEST_FIXTURE
                # These are opaque, hash-bound test inputs. The producer never
                # decompresses their inner payload or executes fixture content.
                require(
                    name.startswith(("tests/data/", "tools/tests/data/"))
                    and name in plan.fixtures
                    and member.file_size <= bound,
                    "Production net or unbound test fixture refused",
                )
                fixtures.add(name)
            elif suffix in (".zip", ".gz", ".tar", ".7z"):
                raise PackageError("Nested/archive payload in source")
            # .bin is exempt only after the explicit test-data location, hash
            # declaration and 128 KiB bound above. Other compiled forms remain
            # forbidden, even if a caller declares their hash as a fixture.
            require(
                (suffix not in COMPILED or binary_data)
                and not re.search(r"\.so(?:\.|$)", name.lower()),
                "Compiled binary in source archive",
            )
            require(0 <= member.file_size <= MAX_FILE, "Source member size bound")
            total += member.file_size
            require(total <= MAX_TOTAL, "Source expanded size bound")
            h = hashlib.sha256()
            with archive.open(member) as stream:
                consumed = 0
                while block := stream.read(1024 * 1024):
                    consumed += len(block)
                    require(consumed <= member.file_size, "Source member expansion changed")
                    h.update(block)
                require(consumed == member.file_size, "Truncated source member")
            hashes[name] = h.hexdigest()
    require(
        not any(
            str(parent) in hashes
            for name in hashes
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        ),
        "File/directory alias in source archive",
    )
    require(hashes.keys() >= SOURCE_REQUIRED, "Incomplete corresponding Phase1 source")
    require(
        fixtures == set(plan.fixtures) and all(hashes[n] == h for n, h in plan.fixtures.items()),
        "Test fixture hash inventory differs",
    )
    required = SUPPORT_REQUIRED | {
        n for n in hashes if n.startswith("tools/") and not n.startswith("tools/tests/")
    }
    require(
        required <= plan.support.keys(), "Required tooling/locks absent from explicit support map"
    )
    require(
        all(
            n.startswith(("tools/", "docs/")) and hashes.get(n) == h
            for n, h in plan.support.items()
        ),
        "Explicit support hash differs from source",
    )
    for name in ("LICENSE", "README.md", "CHANGELOG.md"):
        require(
            hashes[name] == plan.stage_hashes[name],
            "Staged docs/license do not correspond to source",
        )
    return hashes


def archive_inventory(plan: Plan) -> dict[str, str]:
    try:
        return _archive_inventory(plan)
    except (zipfile.BadZipFile, zlib.error, EOFError) as error:
        raise PackageError(f"Invalid source ZIP/archive:{error}") from error


def load_plan(config: Path) -> Plan:
    config = config.absolute()
    config_hash = digest(config)
    require(config.stat().st_size <= 1024 * 1024, "Config exceeds1MiB")
    row = object_value(
        json.loads(
            config.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    )
    require(
        set(row)
        == {
            "schema",
            "version",
            "commit",
            "install_stage",
            "stage_sha256",
            "source_zip",
            "source_sha256",
            "source_root",
            "support_sha256",
            "test_fixture_sha256",
            "timeout_seconds",
        },
        "Unknown/missing packaging plan fields",
    )
    require(row["schema"] == "MousePackagePlan1", "Unknown packaging schema")
    version, commit = str(row["version"]), str(row["commit"])
    require(
        re.fullmatch(r"0\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)-dev", version),
        "Only development candidates; RC/release requires a validated gate ledger",
    )
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Require clean40-hex commit identity")
    stage, source = absolute(row["install_stage"]), absolute(row["source_zip"])
    stage_hashes = hash_map(row["stage_sha256"])
    require(set(stage_hashes) == STAGE_FILES, "Exact stage hash map required")
    require(
        all(stage_hashes.get(n) == h for n, h in ORT_SHA256.items()),
        "Require pinned ORT1.22.0 DLLs",
    )
    require(
        all(stage_hashes.get(n) == h for n, h in LICENSE_SHA256.items()),
        "Require exact GPL/ORT MIT/notices/tl license payload",
    )
    source_hash = row["source_sha256"]
    require(isinstance(source_hash, str) and HEX.fullmatch(source_hash), "Invalid source ZIP hash")
    root = safe_name(str(row["source_root"]))
    require("/" not in root, "Require one explicit source archive root")
    timeout = row["timeout_seconds"]
    require(
        type(timeout) in (int, float)
        and 0 < cast(float, timeout) <= 60
        and math.isfinite(cast(float, timeout)),
        "Finite smoke deadline required in(0,60]",
    )
    identities = {config: config_hash, source: cast(str, source_hash)}
    # The running interpreter is a trusted process identity, not a supplied
    # packaging artifact. Unix venv launchers are normally symlinks; freeze both
    # their canonical executable bytes and their original alias resolution.
    interpreter_alias = Path(sys.executable).absolute()
    interpreter_target = interpreter_alias.resolve(strict=True)
    for own in (
        Path(__file__).absolute(),
        Path(uci_gate.__file__).absolute(),
        interpreter_target,
    ):
        identities[own] = digest(own)
    plan = Plan(
        config,
        stage,
        stage_hashes,
        source,
        cast(str, source_hash),
        root,
        hash_map(row["support_sha256"]),
        hash_map(row["test_fixture_sha256"]),
        version,
        commit,
        float(cast(float, timeout)),
        identities,
        {},
        interpreter_alias,
        interpreter_target,
    )
    plan.verify()  # Freeze before ZIP decompression and PE parsing.
    archive_inventory(plan)
    dependencies = {}
    for name in ("mouse.exe", *ORT_SHA256):
        parsed = parse_pe((stage / name).read_bytes())
        dependencies[name] = parsed
        for imported in cast(list[str], parsed["imports"]):
            require(
                imported in ORT_SHA256
                or imported.removesuffix(".dll") in WINDOWS
                or PREREQUISITES.fullmatch(imported),
                f"Unapproved external import dependency:{name}->{imported}",
            )
    plan.verify()
    return replace(plan, dependencies=dependencies)


def publish(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # The no-clobber link commits this artifact.
    finally:
        # A leftover temporary name cannot undo a successful publication.
        with suppress(OSError):
            Path(temporary).unlink(missing_ok=True)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _smoke_command(executable: Path, flag: str) -> list[str]:
    return [str(executable), flag]


def smoke(executable: Path, flag: str, timeout: float, log: Path) -> dict[str, object]:
    owned = uci_gate.UciProcess(
        _smoke_command(executable, flag),
        cwd=executable.parent,
        timeout=timeout,
        stdout_limit=256 * 1024,
        stderr_limit=65536,
    )
    failure: str | None = None
    code: int | None = None
    try:
        with owned:
            owned.close_stdin()
            code = owned.wait_exit()
    except KeyboardInterrupt:
        failure = "Interrupted smoke; candidate cannot pass"
    except (OSError, ValueError) as error:
        failure = str(error)
    evidence = owned.evidence() if hasattr(owned, "process") else {"command": owned.command}
    evidence["failure"] = failure
    publish(log, json_bytes(evidence))
    require(
        not failure
        and code == 0
        and evidence.get("pumps_joined")
        and evidence.get("capture_complete")
        and not evidence.get("stderr_bytes"),
        f"Smoke failed:{flag}:{failure or code}",
    )
    return evidence


def build_zip(payload: Path, target: Path, hashes: Mapping[str, str]) -> None:
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(hashes):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | (0o755 if name == "mouse.exe" else 0o644)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with (payload / name).open("rb") as source, archive.open(info, "w") as dest:
                shutil.copyfileobj(source, dest, 1024 * 1024)


def produce(config: Path, output: Path) -> dict[str, object]:
    return _produce(load_plan(config), output)


def _produce(plan: Plan, output: Path) -> dict[str, object]:
    output = absolute(str(output))
    # Inspect the supplied ancestry before resolving aliases, so junctions/symlinks
    # cannot disappear through canonicalization. No output exists at this point.
    for ancestor in (output, *output.parents):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        require(
            not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
            "Reparse output path refused",
        )
    output = output.resolve(strict=False)
    stage = plan.stage.resolve(strict=True)
    inputs = [p.resolve(strict=True) for p in plan.identities]
    require(not output.exists(), "Existing candidate evidence is immutable")
    require(
        not (output.is_relative_to(stage) or stage.is_relative_to(output))
        and all(not p.is_relative_to(output) and not output.is_relative_to(p) for p in inputs),
        "Output overlaps immutable input",
    )
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    work.mkdir()
    result: dict[str, object] = {
        "schema": "MouseLocalCandidate1",
        "status": "failed",
        "release_gate": False,
        "version": plan.version,
        "commit": plan.commit,
        "onnx_runtime": "1.22.0",
        "errors": [],
        "input_sha256": ({} if plan.release else {str(p): h for p, h in plan.identities.items()}),
        "stage_sha256": dict(plan.stage_hashes),
        "pe_dependencies": dict(plan.dependencies),
        "source_provenance": (
            "Rebuilt from the hash-bound source folder"
            if plan.release
            else "Owner hash-bound git-archive comment; no independent Git graph authentication"
        ),
        "smoke": [],
    }
    errors = cast(list[str], result["errors"])
    try:
        plan.verify()
        source_hashes = archive_inventory(plan)
        result["source_files"] = source_hashes
        payload = work / "payload"
        payload.mkdir()
        copied = {**plan.stage_hashes, **plan.support}
        for name in plan.stage_hashes:
            destination = payload / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (plan.stage / name).open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        with zipfile.ZipFile(plan.source) as archive:
            for name in plan.support:
                target = payload / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    archive.open(plan.source_root + "/" + name) as member_source,
                    target.open("xb") as dst,
                ):
                    shutil.copyfileobj(member_source, dst, 1024 * 1024)
        require(all(digest(payload / n) == h for n, h in copied.items()), "Copied payload differs")
        result["copied_sha256"] = copied
        binary_name = f"mouse-{plan.version}-windows-x64.zip"
        source_name = f"mouse-{plan.version}-source.zip"
        binary = work / binary_name
        build_zip(payload, binary, copied)
        binary_sha256 = digest(binary)
        result["binary_zip_sha256"] = binary_sha256
        extracted = work / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(binary) as archive:
            require(set(archive.namelist()) == set(copied), "Candidate ZIP inventory differs")
            for member in archive.infolist():
                name = safe_name(member.filename)
                target = extracted / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as extract_source, target.open("xb") as dst:
                    shutil.copyfileobj(extract_source, dst, 1024 * 1024)

        def verify_copies() -> None:
            result["candidate_hashes_unchanged"] = False
            require(digest(binary) == binary_sha256, "Frozen candidate ZIP changed")
            for name, h in copied.items():
                require(
                    digest(extracted / name) == h and digest(payload / name) == h,
                    "Extracted/copied payload changed",
                )
            result["candidate_hashes_unchanged"] = True

        verify_copies()
        plan.verify()
        rows = cast(list[dict[str, object]], result["smoke"])
        for flag in ("--version", "--help"):
            row = smoke(
                extracted / "mouse.exe", flag, plan.timeout, work / ("smoke-" + flag[2:] + ".json")
            )
            rows.append({k: v for k, v in row.items() if k != "command"} if plan.release else row)
            text = cast(str, row["stdout_text"])
            if flag == "--version":
                lines = text.splitlines()
                require(
                    len(lines) == 5
                    and lines[0] == "Mouse " + plan.version
                    and (
                        (plan.release and plan.commit == "unknown" and lines[1] == "git: unknown")
                        or (
                            re.fullmatch(r"git: [0-9a-f]{7,40}", lines[1])
                            and plan.commit.startswith(lines[1].removeprefix("git: "))
                        )
                    )
                    and lines[2] == "build: Release"
                    and lines[3].startswith("compiler: ")
                    and lines[4] == "ONNX Runtime: 1.22.0",
                    "Version/clean commit/ORT smoke identity mismatch",
                )
            else:
                require(
                    text.startswith("Usage: mouse ")
                    and "uci" in text
                    and "--version" in text
                    and "--help" in text,
                    "Help smoke output mismatch",
                )
            plan.verify()
            verify_copies()
        plan.verify()
        verify_copies()
        artifacts = {binary_name: binary_sha256, source_name: plan.source_sha256}
        result.update(status="candidate_passed", hashes_unchanged=True, artifacts=artifacts)
        manifest = json_bytes(result)
        sums = "".join(
            f"{h}  {n}\n"
            for n, h in sorted(
                {**artifacts, "manifest.json": hashlib.sha256(manifest).hexdigest()}.items()
            )
        )
        # All validation precedes publication; the success manifest is always last.
        publish(output / binary_name, binary.read_bytes())
        publish(output / source_name, plan.source.read_bytes())
        publish(output / "SHA256SUMS", sums.encode("ascii"))
        require(
            all(digest(output / name) == h for name, h in artifacts.items()),
            "Published archive hash changed before success manifest",
        )
        require(
            (output / "SHA256SUMS").read_bytes() == sums.encode("ascii"),
            "Published checksum list changed",
        )
        plan.verify()
        verify_copies()
        publish(output / "manifest.json", manifest)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error, EOFError) as error:
        errors.append(f"{type(error).__name__}:{error}")
        result.update(status="failed", release_gate=False)
        try:
            plan.verify()
            result["hashes_unchanged"] = result.get("candidate_hashes_unchanged", True)
        except (OSError, ValueError) as changed:
            errors.append(str(changed))
            result["hashes_unchanged"] = False
        publish(output / "failure.json", json_bytes(result))
    return result


def _release_command(command: list[str], cwd: Path) -> None:
    """Build/install only the explicit tree; a failing tool never yields a package."""
    try:
        subprocess.run(command, cwd=cwd, check=True, timeout=1800)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise PackageError(f"Release build/install failed: {error}") from error


def _release_inventory(source: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    roots = [source / name for name in sorted(SOURCE_ROOT_FILES | SOURCE_DIRECTORIES)]
    for root in roots:
        if not root.exists():
            continue
        pending = [root]
        while pending:
            path = pending.pop()
            info = path.lstat()
            require(
                not stat.S_ISLNK(info.st_mode)
                and not (getattr(info, "st_file_attributes", 0) & 0x400),
                "Reparse source path refused",
            )
            if path.is_dir():
                if path.name in {
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    "build",
                    "dist",
                    ".venv",
                    ".toolchain",
                } or path.name.endswith(".egg-info"):
                    continue
                pending.extend(sorted(path.iterdir()))
            else:
                name = safe_name(path.relative_to(source).as_posix())
                hashes[name] = digest(path)
                require(len(hashes) <= MAX_MEMBERS, "Source member count bound")
    require(hashes.keys() >= SOURCE_REQUIRED, "Incomplete corresponding source folder")
    return hashes


def produce_release(source: Path, build: Path, version: str, output: Path) -> dict[str, object]:
    """Package a configured release source tree without asserting tournament gates.

    This path is deliberately separate from the development JSON-plan interface.
    A pre-Git archive has an honest unknown commit; source content hashes bind it.
    """
    require(
        re.fullmatch(
            r"[1-9][0-9]*\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?:-rc\.[1-9][0-9]*)?", version
        ),
        "Require strict release version X.Y.Z or X.Y.Z-rc.N (X,N positive)",
    )
    source, build, output = source.absolute(), build.absolute(), output.absolute()
    for path in (source, build, output):
        for ancestor in (path, *path.parents):
            if ancestor.exists():
                info = ancestor.lstat()
                require(
                    not stat.S_ISLNK(info.st_mode)
                    and not (getattr(info, "st_file_attributes", 0) & 0x400),
                    "Reparse release path refused",
                )
    source, build, output = source.resolve(), build.resolve(), output.resolve()
    require(build.is_relative_to(source), "Build must belong to the supplied source folder")
    require(
        not output.exists()
        and not output.is_relative_to(source)
        and not source.is_relative_to(output),
        "Release output must be new and outside the source folder",
    )
    require(not (source / ".git").exists(), "Use an exported source folder without Git metadata")
    cache = build / "CMakeCache.txt"
    cache_hash = digest(cache)
    fields = dict(
        re.findall(r"^([A-Za-z_][A-Za-z0-9_]*):[^=\n]+=([^\n]*)$", cache.read_text(), re.M)
    )
    require(
        Path(fields.get("CMAKE_HOME_DIRECTORY", "")).resolve() == source,
        "CMake build belongs to a different source folder",
    )
    require(fields.get("CMAKE_BUILD_TYPE") == "Release", "Require a Release CMake build")
    require(fields.get("MOUSE_VERSION") == version, "CMake release version mismatch")
    require(fields.get("MOUSE_GIT_HASH", "") in ("", "unknown"), "Archive commit override refused")
    hashes = _release_inventory(source)
    require(
        {"tools/setup_net.py", "tools/src/mouse_tools/setup_net.py"} <= hashes.keys(),
        "Release requires the checksum-verified network setup script",
    )
    _release_command(
        ["cmake", "--build", str(build), "--target", "mouse", "--parallel", "2"], source
    )
    require(digest(cache) == cache_hash, "CMake configuration changed during build")
    executable = build / "mouse.exe"
    binary_hash = digest(executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mouse-release-", dir=output.parent) as temporary:
        work = Path(temporary)
        stage = work / "stage"
        _release_command(["cmake", "--install", str(build), "--prefix", str(stage)], source)
        stage_hashes = {name: digest(stage / name) for name in stage_inventory(stage)}
        require(
            stage_hashes["mouse.exe"] == binary_hash, "Installed binary differs from source build"
        )
        require(digest(executable) == binary_hash, "Source build binary changed during install")
        require(_release_inventory(source) == hashes, "Source folder changed during build/install")
        require(all(stage_hashes[n] == h for n, h in ORT_SHA256.items()), "Require pinned ORT DLLs")
        require(
            all(stage_hashes[n] == h for n, h in LICENSE_SHA256.items()), "License hash differs"
        )
        archived = work / "source.zip"
        with zipfile.ZipFile(archived, "x", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.comment = b"unknown"
            for name in sorted(hashes):
                member = zipfile.ZipInfo("Mouse-source/" + name, (1980, 1, 1, 0, 0, 0))
                member.create_system = 3
                member.external_attr = (stat.S_IFREG | 0o644) << 16
                member.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(member, (source / name).read_bytes())
        fixtures = {
            n: h
            for n, h in hashes.items()
            if n.lower().endswith((".onnx", ".pb", ".pb.gz", ".v6.gz", ".jsonl.gz", ".bin"))
        }
        support = {
            n: h
            for n, h in hashes.items()
            if n in SUPPORT_REQUIRED
            or (n.startswith("tools/") and not n.startswith("tools/tests/"))
        }
        identities = {source / n: h for n, h in hashes.items()}
        identities.update({cache: cache_hash, executable: binary_hash, archived: digest(archived)})
        interpreter_alias = Path(sys.executable).absolute()
        interpreter_target = interpreter_alias.resolve(strict=True)
        for own in (
            Path(__file__).absolute(),
            Path(uci_gate.__file__).absolute(),
            interpreter_target,
        ):
            identities[own] = digest(own)
        plan = Plan(
            cache,
            stage,
            stage_hashes,
            archived,
            digest(archived),
            "Mouse-source",
            support,
            fixtures,
            version,
            "unknown",
            30.0,
            identities,
            {},
            interpreter_alias,
            interpreter_target,
            release=True,
            release_source=source,
            release_files=hashes,
        )
        plan.verify()
        require(archive_inventory(plan) == hashes, "Archived source differs from source folder")
        dependencies = {}
        for name in ("mouse.exe", *ORT_SHA256):
            parsed = parse_pe((stage / name).read_bytes())
            dependencies[name] = parsed
            for imported in cast(list[str], parsed["imports"]):
                require(
                    imported in ORT_SHA256
                    or imported.removesuffix(".dll") in WINDOWS
                    or PREREQUISITES.fullmatch(imported),
                    f"Unapproved external import dependency:{name}->{imported}",
                )
        return _produce(replace(plan, dependencies=dependencies), output)


def _isolated() -> bool:
    return bool(sys.flags.isolated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--source", type=Path)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(_isolated(), "Run with isolated Python: python -I -m mouse_tools.package")
        if args.source is not None:
            require(
                args.build is not None and args.version is not None,
                "Release requires --build and --version",
            )
            result = produce_release(args.source, args.build, args.version, args.output)
        else:
            require(args.build is None and args.version is None, "Build/version require --source")
            result = produce(args.config, args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error, EOFError) as error:
        print(f"Packaging error:{error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "candidate_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
