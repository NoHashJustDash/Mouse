# SPDX-License-Identifier: GPL-3.0-or-later
"""Candidate packaging uses authored PE/ZIP bytes and real controlled children."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import struct
import sys
import threading
import zipfile
from pathlib import Path
from typing import Any

import pytest


def api() -> Any:
    try:
        return importlib.import_module("mouse_tools.package")
    except ModuleNotFoundError:
        pytest.fail("Production packaging producer is missing")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pe(imports: tuple[str, ...] = ("KERNEL32.dll",), *, delay: bool = False) -> bytes:
    data = bytearray(4096)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 128)
    data[128:132] = b"PE\x00\x00"
    struct.pack_into("<HHIIIHH", data, 132, 0x8664, 1, 0, 0, 0, 240, 0x2022)
    optional = 152
    struct.pack_into("<H", data, optional, 0x20B)
    struct.pack_into("<Q", data, optional + 24, 0x180000000)
    struct.pack_into("<I", data, optional + 60, 512)
    struct.pack_into("<I", data, optional + 108, 16)
    table, width, directory = (0x200, 32, 13) if delay else (0x200, 20, 1)
    struct.pack_into(
        "<II", data, optional + 112 + directory * 8, 0x1000, width * (len(imports) + 1)
    )
    section = optional + 240
    data[section : section + 8] = b".rdata\x00\x00"
    struct.pack_into("<IIII", data, section + 8, 3584, 0x1000, 3584, 512)
    for i, name in enumerate(imports):
        offset = 0x400 + i * 256
        rva = 0x1000 + offset - 512
        if delay:
            struct.pack_into("<II", data, table + i * width, 1, rva)
        else:
            struct.pack_into("<I", data, table + i * width + 12, rva)
        raw = name.encode() + b"\x00"
        data[offset : offset + len(raw)] = raw
    return bytes(data)


@pytest.mark.parametrize("delay", [False, True])
def test_pe32plus_amd64_normal_and_delay_imports_are_read_independently(delay: bool) -> None:
    result = api().parse_pe(pe(("KERNEL32.dll", "nvcuda.dll"), delay=delay))
    assert result["machine"] == "AMD64"
    assert result["imports"] == ["kernel32.dll", "nvcuda.dll"]


@pytest.mark.parametrize(
    "mutation", ["dos", "signature", "machine", "optional", "truncated", "rva", "terminator"]
)
def test_bad_pe_never_becomes_a_valid_payload(mutation: str) -> None:
    data = bytearray(pe())
    if mutation == "dos":
        data[0] = 0
    elif mutation == "signature":
        data[128] = 0
    elif mutation == "machine":
        struct.pack_into("<H", data, 132, 0x14C)
    elif mutation == "optional":
        struct.pack_into("<H", data, 152, 0x10B)
    elif mutation == "truncated":
        data = data[:200]
    elif mutation == "rva":
        struct.pack_into("<I", data, 0x20C, 0xFFFF0000)
    else:
        data[0x214:0x228] = data[0x200:0x214]
    with pytest.raises(api().PackageError):
        api().parse_pe(bytes(data))


@pytest.mark.parametrize(
    "name",
    [
        "../bad",
        "/root",
        "C:/net",
        "a\\b",
        "a//b",
        "a/./b",
        "a/../b",
        "CON.txt",
        "a. ",
        "a:stream",
        "a\x00b",
        ".toolchain/x",
        ".git/config",
    ],
)
def test_archive_names_reject_platform_aliases_and_forbidden_content(name: str) -> None:
    with pytest.raises(api().PackageError):
        api().safe_name(name)


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    module = api()
    stage = tmp_path / "clean install with spaces"
    stage.mkdir()
    stage_data = {
        "mouse.exe": pe(("onnxruntime.dll", "KERNEL32.dll")),
        **{name: pe() for name in module.ORT_SHA256},
        "LICENSE": b"GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n",
        "README.md": b"Mouse candidate; use mouse uci --weights net.onnx\n",
        "CHANGELOG.md": b"# Changelog\n0.5.0-dev\n",
        "licenses/ONNXRuntime-LICENSE": b"MIT License\nCopyright Microsoft Corporation\n",
        "licenses/ThirdPartyNotices.txt": b"ONNX Runtime third party notices\n",
        "licenses/tl-expected-LICENSE": b"CC0 1.0 Universal\n",
    }
    for name, value in stage_data.items():
        path = stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    monkeypatch.setattr(
        module, "ORT_SHA256", {name: sha(stage_data[name]) for name in module.ORT_SHA256}
    )
    monkeypatch.setattr(
        module, "LICENSE_SHA256", {name: sha(stage_data[name]) for name in module.LICENSE_SHA256}
    )
    entries = {name: b"original testable source\n" for name in module.SOURCE_REQUIRED}
    for name in ("LICENSE", "README.md", "CHANGELOG.md"):
        entries[name] = stage_data[name]
    support = {name: sha(entries[name]) for name in module.SUPPORT_REQUIRED}
    archive = tmp_path / "clean phase1.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.comment = b"a" * 40
        for name, value in entries.items():
            zipped.writestr("Mouse-source/" + name, value)
    document = dict(
        schema="MousePackagePlan1",
        version="0.5.0-dev",
        commit="a" * 40,
        install_stage=str(stage),
        stage_sha256={name: sha(value) for name, value in stage_data.items()},
        source_zip=str(archive),
        source_sha256=sha(archive.read_bytes()),
        source_root="Mouse-source",
        support_sha256=support,
        test_fixture_sha256={},
        timeout_seconds=2,
    )
    config = tmp_path / "package plan.json"
    config.write_text(json.dumps(document), encoding="utf-8")
    return config, document


def rewrite(config: Path, document: dict[str, Any]) -> None:
    config.write_text(json.dumps(document), encoding="utf-8")


def change_zip(
    config: Path,
    document: dict[str, Any],
    entries: list[tuple[str, bytes]],
    *,
    comment: bytes | None = None,
) -> None:
    path = Path(document["source_zip"])
    with zipfile.ZipFile(path) as zipped:
        old = [(info.filename, zipped.read(info)) for info in zipped.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipped:
        zipped.comment = comment if comment is not None else b"a" * 40
        for name, data in old + entries:
            zipped.writestr(name, data)
    document["source_sha256"] = sha(path.read_bytes())
    rewrite(config, document)


def test_exact_clean_plan_binds_stage_source_and_support_without_checkout_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = fixture(tmp_path, monkeypatch)
    plan = api().load_plan(config)
    plan.verify()
    # Standard Unix venvs expose Python through a symlink. Freeze the real
    # interpreter bytes and its alias, while supplied artifacts stay non-reparse.
    interpreter = Path(sys.executable).resolve(strict=True)
    assert plan.identities[interpreter] == sha(interpreter.read_bytes())
    assert plan.interpreter_alias == Path(sys.executable).absolute()
    assert plan.interpreter_target == interpreter
    assert plan.version == "0.5.0-dev"
    assert len(plan.stage_hashes) == 11


@pytest.mark.parametrize(
    "mutation",
    [
        "rc",
        "release",
        "dirty",
        "unknown_key",
        "hash",
        "ort",
        "extra_stage",
        "missing_license",
        "source_comment",
        "support",
        "source_doc",
    ],
)
def test_invalid_identity_inventory_and_labels_are_refused_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    module = api()
    config, document = fixture(tmp_path, monkeypatch)
    if mutation == "rc":
        document["version"] = "1.0.0-rc.1"
    elif mutation == "release":
        document["version"] = "1.0.0"
    elif mutation == "dirty":
        document["commit"] += "-dirty"
    elif mutation == "unknown_key":
        document["allowed_external_dlls"] = ["untrusted.dll"]
    elif mutation == "hash":
        document["stage_sha256"]["mouse.exe"] = "0" * 64
    elif mutation == "ort":
        path = Path(document["install_stage"]) / "onnxruntime.dll"
        path.write_bytes(pe(("USER32.dll",)))
        document["stage_sha256"][path.name] = sha(path.read_bytes())
    elif mutation == "extra_stage":
        (Path(document["install_stage"]) / "secret.pb.gz").write_bytes(b"weights")
    elif mutation == "missing_license":
        (Path(document["install_stage"]) / "licenses/ThirdPartyNotices.txt").unlink()
    elif mutation == "source_comment":
        change_zip(config, document, [], comment=b"b" * 40)
    elif mutation == "support":
        document["support_sha256"][next(iter(document["support_sha256"]))] = "0" * 64
    else:
        path = Path(document["install_stage"]) / "README.md"
        path.write_bytes(b"different staged source docs")
        document["stage_sha256"][path.name] = sha(path.read_bytes())
    rewrite(config, document)
    with pytest.raises(module.PackageError):
        module.load_plan(config)


@pytest.mark.parametrize(
    "name",
    [
        "Mouse-source/unrelated-project/private.py",
        "Mouse-source/.git/config",
        "Mouse-source/engine/a.dll",
        "Mouse-source/nets/t1.pb.gz",
        "Mouse-source/tools/PYPROJECT.toml",
        "../escape",
        "Mouse-source/.toolchain/a",
    ],
)
def test_source_zip_rejects_hidden_or_alias_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    change_zip(config, document, [(name, b"not permitted")])
    with pytest.raises(api().PackageError):
        api().load_plan(config)


def test_unapproved_external_import_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    path = Path(document["install_stage"]) / "mouse.exe"
    path.write_bytes(pe(("untrusted.dll",)))
    document["stage_sha256"]["mouse.exe"] = sha(path.read_bytes())
    rewrite(config, document)
    with pytest.raises(api().PackageError, match="import|depend"):
        api().load_plan(config)


def smoke_child(monkeypatch: pytest.MonkeyPatch, *, extra: str = "", wrong: bool = False) -> None:
    def command(executable: Path, flag: str) -> list[str]:
        version = (
            "Mouse 0.5.0-dev\ngit: "
            + "a" * 12
            + "\nbuild: Release\ncompiler: authored child\nONNX Runtime: 1.22.0\n"
        )
        if wrong:
            version = version.replace("1.22.0", "1.21.0")
        result = (
            version
            if flag == "--version"
            else "Usage: mouse [uci [explicit backend/search options]]\nmouse --version | --help\n"
        )
        return [
            sys.executable,
            "-I",
            "-c",
            "import os,sys,time;assert os.environ['PATH']=='';"
            + extra
            + f"print({result!r},end='')",
        ]

    monkeypatch.setattr(api(), "_smoke_command", command)


def test_complete_package_is_deterministic_no_clobber_and_smoked_from_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    smoke_child(monkeypatch)
    output = tmp_path / "candidate one"
    result = api().produce(config, output)
    assert result["status"] == "candidate_passed"
    assert result["release_gate"] is False
    assert len(result["smoke"]) == 2
    assert result["hashes_unchanged"] is True
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "candidate_passed"
    assert (output / "SHA256SUMS").is_file()
    assert manifest["commit"] == "a" * 40
    assert manifest["onnx_runtime"] == "1.22.0"
    zip_path = output / "mouse-0.5.0-dev-windows-x64.zip"
    with zipfile.ZipFile(zip_path) as zipped:
        assert "mouse.exe" in zipped.namelist()
        assert set(document["support_sha256"]) <= set(zipped.namelist())
        assert not any("nvcuda" in name or "nets/" in name for name in zipped.namelist())
    other = tmp_path / "candidate two"
    api().produce(config, other)
    assert zip_path.read_bytes() == (other / zip_path.name).read_bytes()
    original = (output / "manifest.json").read_bytes()
    with pytest.raises(api().PackageError):
        api().produce(config, output)
    assert (output / "manifest.json").read_bytes() == original


@pytest.mark.parametrize(
    "mutation", ["timeout", "wrong_ort", "stage_changed", "nonzero", "unexpected_stderr"]
)
def test_failed_smoke_retains_logs_but_never_publishes_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    extra = ""
    if mutation == "timeout":
        extra = "time.sleep(10);"
        document["timeout_seconds"] = 0.15
        rewrite(config, document)
    elif mutation == "stage_changed":
        path = Path(document["install_stage"]) / "mouse.exe"
        extra = f"from pathlib import Path;Path({str(path)!r}).write_bytes(b'changed');"
    elif mutation == "nonzero":
        extra = "sys.exit(7);"
    elif mutation == "unexpected_stderr":
        extra = "sys.stderr.write('unexpected error');"
    smoke_child(monkeypatch, extra=extra, wrong=mutation == "wrong_ort")
    before = {thread.ident for thread in threading.enumerate()}
    result = api().produce(config, tmp_path / "failed")
    assert result["status"] == "failed"
    assert result["errors"]
    assert not (tmp_path / "failed/manifest.json").exists()
    assert (tmp_path / "failed/failure.json").is_file()
    assert list((tmp_path / "failed/work").glob("smoke-*.json"))
    assert {thread.ident for thread in threading.enumerate()} == before


def test_cli_refuses_nonisolated_python_before_loading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(api(), "_isolated", lambda: False)
    assert (
        api().main(["--config", str(tmp_path / "missing"), "--output", str(tmp_path / "output")])
        == 1
    )
    assert "-I" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


def test_zip_symlink_is_rejected_before_extracting_any_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    path = Path(document["source_zip"])
    with zipfile.ZipFile(path, "a") as archive:
        member = zipfile.ZipInfo("Mouse-source/tools/linked.py")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        archive.writestr(member, b"../../outside")
    document["source_sha256"] = sha(path.read_bytes())
    rewrite(config, document)
    with pytest.raises(api().PackageError, match="symlink|special"):
        api().load_plan(config)


def test_file_directory_zip_alias_is_rejected_as_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    change_zip(
        config,
        document,
        [("Mouse-source/docs/alias", b"file"), ("Mouse-source/docs/alias/inside.md", b"nested")],
    )
    with pytest.raises(api().PackageError, match="alias|directory"):
        api().load_plan(config)


def test_final_output_hash_is_rechecked_before_success_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, _ = fixture(tmp_path, monkeypatch)
    smoke_child(monkeypatch)
    original = module.publish

    def corrupt_after_publish(path: Path, data: bytes) -> None:
        original(path, data)
        if path.name.endswith("windows-x64.zip"):
            with path.open("ab") as stream:
                stream.write(b"altered output")

    monkeypatch.setattr(module, "publish", corrupt_after_publish)
    result = module.produce(config, tmp_path / "mutated output")
    assert result["status"] == "failed"
    assert not (tmp_path / "mutated output/manifest.json").exists()


def test_windows_external_import_families_do_not_admit_unrelated_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, document = fixture(tmp_path, monkeypatch)
    path = Path(document["install_stage"]) / "mouse.exe"
    path.write_bytes(
        pe(
            (
                "api-ms-win-core-file-l1-1-0.dll",
                "vcruntime140_1.dll",
                "cudart64_12.dll",
                "nvinfer_10.dll",
            ),
            delay=True,
        )
    )
    document["stage_sha256"]["mouse.exe"] = sha(path.read_bytes())
    rewrite(config, document)
    plan = module.load_plan(config)
    assert "nvinfer_10.dll" in plan.dependencies["mouse.exe"]["imports"]


@pytest.mark.parametrize("name", ["tests/data/test.onnx", "tools/tests/data/test.pb.gz"])
def test_only_explicit_small_test_network_fixtures_are_allowed_in_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    data = b"original small test fixture"
    document["test_fixture_sha256"] = {name: sha(data)}
    change_zip(config, document, [("Mouse-source/" + name, data)])
    api().load_plan(config)
    document["test_fixture_sha256"][name] = "0" * 64
    rewrite(config, document)
    with pytest.raises(api().PackageError):
        api().load_plan(config)


@pytest.mark.parametrize(
    "name", ["tests/data/too-large.onnx", "tests/data/too-large.pb", "tests/data/too-large.pb.gz"]
)
def test_unbound_and_oversized_test_fixtures_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    data = b"x" * (128 * 1024 + 1)
    document["test_fixture_sha256"] = {name: sha(data)}
    change_zip(config, document, [("Mouse-source/" + name, data)])
    with pytest.raises(api().PackageError):
        api().load_plan(config)


def test_interrupt_retains_smoke_evidence_and_joins_owned_pumps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, _ = fixture(tmp_path, monkeypatch)
    smoke_child(monkeypatch)

    def interrupt(_self: Any) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(module.uci_gate.UciProcess, "wait_exit", interrupt)
    try:
        result = module.produce(config, tmp_path / "interrupted")
    except KeyboardInterrupt:
        pytest.fail("Interrupted child evidence escaped without a failed report")
    assert result["status"] == "failed"
    assert (tmp_path / "interrupted/work/smoke-version.json").is_file()
    assert not (tmp_path / "interrupted/manifest.json").exists()


@pytest.mark.parametrize("name", [".aws/credentials", "private/copy.txt", "build/snapshot.txt"])
def test_source_archive_uses_explicit_phase1_top_level_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    change_zip(config, document, [("Mouse-source/" + name, b"outside Phase1 source whitelist")])
    with pytest.raises(api().PackageError, match="allowlist|whitelist"):
        api().load_plan(config)


@pytest.mark.parametrize("import_name", ["api-ms-win-malware.dll", "cudnn_custom64_9.dll"])
def test_unrecognized_windows_or_nvidia_prefixed_library_is_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, import_name: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    path = Path(document["install_stage"]) / "mouse.exe"
    path.write_bytes(pe((import_name,)))
    document["stage_sha256"]["mouse.exe"] = sha(path.read_bytes())
    rewrite(config, document)
    with pytest.raises(api().PackageError, match="import|depend"):
        api().load_plan(config)


def test_corrupt_deflate_stream_becomes_a_concise_packaging_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    path = Path(document["source_zip"])
    with zipfile.ZipFile(path) as archive:
        member = archive.infolist()[0]
    raw = bytearray(path.read_bytes())
    names, extras = struct.unpack_from("<HH", raw, member.header_offset + 26)
    offset = member.header_offset + 30 + names + extras
    raw[offset] = 0xFF  # invalid DEFLATE block type, not just a CRC mismatch
    path.write_bytes(raw)
    document["source_sha256"] = sha(raw)
    rewrite(config, document)
    with pytest.raises(api().PackageError, match="ZIP|archive"):
        api().load_plan(config)


def test_smoke_cannot_replace_the_untested_work_zip_after_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "zip changed during smoke"
    binary = output / "work/mouse-0.5.0-dev-windows-x64.zip"
    extra = (
        "import zipfile;"
        f"p={str(binary)!r};z=zipfile.ZipFile(p);"
        "items={i.filename:z.read(i) for i in z.infolist()};z.close();"
        "items['mouse.exe']=b'untested replacement executable';"
        "w=zipfile.ZipFile(p,'w');[w.writestr(n,b) for n,b in items.items()];w.close();"
    )
    smoke_child(monkeypatch, extra=extra)
    result = module.produce(config, output)
    assert result["status"] == "failed"
    assert not (output / "manifest.json").exists()
    assert (output / "failure.json").is_file()
    assert result["binary_zip_sha256"] != sha(binary.read_bytes())
    assert result["hashes_unchanged"] is False


def test_arbitrarily_large_json_timeout_has_a_concise_cli_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = api()
    config, document = fixture(tmp_path, monkeypatch)
    document["timeout_seconds"] = 10**400
    rewrite(config, document)
    with pytest.raises(module.PackageError, match="deadline"):
        module.load_plan(config)
    assert module.main(["--config", str(config), "--output", str(tmp_path / "unused")]) == 1
    assert "Packaging error:" in capsys.readouterr().err
    assert not (tmp_path / "unused").exists()


def test_success_manifest_link_is_the_commit_point_despite_temp_cleanup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "late cleanup failure"
    smoke_child(monkeypatch)
    original = Path.unlink
    raised = False

    def fail_once(path: Path, missing_ok: bool = False) -> None:
        nonlocal raised
        if not raised and path.name.startswith(".publish-") and (output / "manifest.json").exists():
            raised = True
            raise PermissionError("injected temporary-name cleanup failure after commit")
        original(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_once)
    result = module.produce(config, output)
    assert raised
    assert result["status"] == "candidate_passed"
    assert json.loads((output / "manifest.json").read_text())["status"] == "candidate_passed"
    assert not (output / "failure.json").exists()


def test_output_dotdot_alias_refuses_before_creating_any_immutable_stage_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, document = fixture(tmp_path, monkeypatch)
    stage = Path(document["install_stage"])
    alias = tmp_path / "ordinary alias directory"
    alias.mkdir()
    output = alias / ".." / stage.name / "candidate"
    module.load_plan(config).verify()
    refused = False
    try:
        module.produce(config, output)
    except module.PackageError:
        refused = True
    assert not (stage / "candidate").exists()
    assert refused


def test_corresponding_source_preserves_original_contributing_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    config, document = fixture(tmp_path, monkeypatch)
    original = b"# Contributing to Mouse\nOriginal Phase1 project instructions.\n"
    change_zip(config, document, [("Mouse-source/CONTRIBUTING.md", original)])
    plan = module.load_plan(config)
    plan.verify()
    inventory = module.archive_inventory(plan)
    assert inventory.keys() >= module.SOURCE_REQUIRED | {"CONTRIBUTING.md"}
    assert inventory["CONTRIBUTING.md"] == sha(original)
    with zipfile.ZipFile(plan.source) as archived:
        assert archived.read("Mouse-source/CONTRIBUTING.md") == original


@pytest.mark.parametrize("missing", [None, "vcpkg.json", "CMakeLists.txt", "engine/main.cpp"])
def test_manifest_only_pinned_source_retains_manifest_and_requires_real_build_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    module = api()
    config, document = fixture(tmp_path, monkeypatch)
    manifest = (
        json.dumps(
            {
                "name": "mouse",
                "version-semver": "0.0.0",
                "builtin-baseline": "a5f736224e85b32e25a8ffc52562333180396642",
                "dependencies": ["catch2", "tl-expected"],
            },
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    source = Path(document["source_zip"])
    with zipfile.ZipFile(source) as archived:
        entries = [
            (name, manifest if name == "Mouse-source/vcpkg.json" else archived.read(name))
            for name in archived.namelist()
            if name != "Mouse-source/vcpkg-configuration.json" and name != f"Mouse-source/{missing}"
        ]
    with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archived:
        archived.comment = b"a" * 40
        for name, contents in entries:
            archived.writestr(name, contents)
    document["source_sha256"] = sha(source.read_bytes())
    rewrite(config, document)
    if missing is not None:
        with pytest.raises(module.PackageError, match="Incomplete corresponding Phase1 source"):
            module.load_plan(config)
        return
    plan = module.load_plan(config)
    inventory = module.archive_inventory(plan)
    assert "vcpkg-configuration.json" not in inventory
    assert inventory["vcpkg.json"] == sha(manifest)
    assert inventory.keys() >= {"vcpkg.json", "CMakeLists.txt", "engine/main.cpp"}
    with zipfile.ZipFile(plan.source) as archived:
        assert archived.read("Mouse-source/vcpkg.json") == manifest


@pytest.mark.parametrize(
    "name", ["tests/data/nn/parity.v6.gz", "tests/data/search/oracle-states.jsonl.gz"]
)
def test_declared_compressed_test_data_is_retained_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    data = gzip.compress(b"original compressed test data\n", mtime=0)
    document["test_fixture_sha256"] = {name: sha(data)}
    change_zip(config, document, [("Mouse-source/" + name, data)])
    plan = api().load_plan(config)
    assert api().archive_inventory(plan)[name] == sha(data)
    document["test_fixture_sha256"][name] = "0" * 64
    rewrite(config, document)
    with pytest.raises(api().PackageError, match="fixture hash"):
        api().load_plan(config)


@pytest.mark.parametrize(
    ("name", "declared", "oversized"),
    [
        ("tests/data/nn/unbound.v6.gz", False, False),
        ("docs/outside.jsonl.gz", True, False),
        ("tests/data/nested.zip", True, False),
        ("tests/data/unknown.gz", True, False),
        ("tests/data/too-large.jsonl.gz", True, True),
    ],
)
def test_compressed_test_data_keeps_location_declaration_type_and_actual_size_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    declared: bool,
    oversized: bool,
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    data = gzip.compress(b"x" * (16 * 1024 * 1024 if oversized else 20), compresslevel=0, mtime=0)
    if oversized:
        assert len(data) > 16 * 1024 * 1024
    if declared:
        document["test_fixture_sha256"] = {name: sha(data)}
    change_zip(config, document, [("Mouse-source/" + name, data)])
    with pytest.raises(api().PackageError, match="fixture|Nested/archive"):
        api().load_plan(config)


def test_declared_binary_test_data_is_retained_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    name = "tests/data/nn/planes.bin"
    data = b"MOUSEIN1" + struct.pack("<II", 1, 112) + bytes(112 * 12)
    document["test_fixture_sha256"] = {name: sha(data)}
    change_zip(config, document, [("Mouse-source/" + name, data)])
    plan = api().load_plan(config)
    assert api().archive_inventory(plan)[name] == sha(data)
    document["test_fixture_sha256"][name] = "0" * 64
    rewrite(config, document)
    with pytest.raises(api().PackageError, match="fixture hash"):
        api().load_plan(config)


@pytest.mark.parametrize(
    ("name", "declared", "oversized"),
    [
        ("tests/data/unbound.bin", False, False),
        ("docs/outside.bin", True, False),
        ("tests/data/too-large.bin", True, True),
        ("tests/data/forbidden.dll", True, False),
        ("tests/data/forbidden.obj", True, False),
    ],
)
def test_binary_fixture_exception_keeps_data_bounds_and_compiled_payload_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    declared: bool,
    oversized: bool,
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    data = b"x" * (128 * 1024 + 1) if oversized else pe()
    if declared:
        document["test_fixture_sha256"] = {name: sha(data)}
    change_zip(config, document, [("Mouse-source/" + name, data)])
    with pytest.raises(api().PackageError, match="fixture|Compiled binary"):
        api().load_plan(config)


def test_interpreter_alias_resolution_cannot_change_after_plan_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    config, _ = fixture(tmp_path, monkeypatch)
    plan = api().load_plan(config)
    alias = tmp_path / "different interpreter"
    alias.write_bytes(b"different interpreter bytes")
    changed = replace(plan, interpreter_alias=alias)
    with pytest.raises(api().PackageError, match="Interpreter alias changed"):
        changed.verify()


@pytest.mark.parametrize(
    "relative",
    ["unrelated-project/private.py", "unrelated_project/private.py", "research/private.py"],
)
def test_source_allowlist_rejects_every_unrelated_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    config, document = fixture(tmp_path, monkeypatch)
    change_zip(config, document, [("Mouse-source/" + relative, b"private local work")])
    with pytest.raises(api().PackageError, match="outside the Phase1 allowlist"):
        api().load_plan(config)
