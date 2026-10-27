# SPDX-License-Identifier: GPL-3.0-or-later
"""Code-driven UCI documentation, including real bounded child handshakes."""

from __future__ import annotations

import importlib
import os
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# Normative 25-option fixture: original 23-option native capture below was made
# without a model/provider on 2026-10-17; the two root declarations are authored
# from the reviewed options.cpp contract in exact source order. This expanded
# fixture is expected output, not a new captured native transcript.
# mouse.exe SHA256 b4145c22346ec424bd037addca63eb2deb7466becbd29eb2356ef580e9ca7090.
OBSERVED = """id name Mouse 0.5.0-dev
id author Zulfeqar Malik
option name WeightsFile type string default <empty>
option name Backend type combo default tensorrt var cpu var cuda var tensorrt
option name Precision type combo default fp16 var fp32 var fp16
option name Threads type spin default 4 min 1 max 128
option name MinibatchSize type spin default 64 min 1 max 1024
option name NNCacheSize type spin default 256 min 0 max 16384
option name RamLimitMb type spin default 512 min 1 max 16384
option name CPuct type string default 1.745
option name CPuctBase type string default 38739
option name CPuctFactor type string default 3.894
option name FpuValue type string default 0.33
option name FpuStrategyAtRoot type combo default same var same var reduction var absolute
option name FpuValueAtRoot type string default 1
option name PolicyTemperature type string default 1.359
option name MultiPV type spin default 1 min 1 max 256
option name UCI_ShowWDL type check default false
option name UCI_Chess960 type check default false
option name Ponder type check default false
option name MoveOverheadMs type spin default 30 min 0 max 60000
option name GpuDllDirectories type string default <empty>
option name CacheDirectory type string default .cache/mouse
option name Device type spin default 0 min 0 max 1023
option name CpuThreads type spin default 1 min 1 max 128
option name WorkspaceMb type spin default 1024 min 1 max 16384
option name Deterministic type check default false
uciok
"""
BEGIN = "<!-- BEGIN MOUSE UCI OPTIONS -->"
END = "<!-- END MOUSE UCI OPTIONS -->"


def api() -> Any:
    try:
        return importlib.import_module("mouse_tools.uci_options")
    except ModuleNotFoundError:
        pytest.fail("UCI options documentation generator missing")


def observed() -> Any:
    return api().parse_handshake(OBSERVED.splitlines())


def document(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text(f"before\n{BEGIN}\nold\n{END}\nafter\n", encoding="utf-8")
    return path


def child(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


def engine_code(*, suffix: str = "", exit_code: int = 0) -> str:
    return (
        "import sys; assert sys.stdin.readline() == 'uci\\n'; "
        f"sys.stdout.write({OBSERVED!r}); sys.stdout.flush(); "
        "assert sys.stdin.readline() == 'quit\\n'; "
        f"{suffix}; raise SystemExit({exit_code})"
    )


def test_normative_fixture_keeps_all_25_typed_options_and_source_order() -> None:
    result = observed()
    assert result.engine_name == "Mouse 0.5.0-dev"
    assert result.author == "Zulfeqar Malik"
    assert result.transcript == tuple(OBSERVED.splitlines())
    assert len(result.options) == 25
    by_name = {option.name: option for option in result.options}
    assert list(by_name)[:3] == ["WeightsFile", "Backend", "Precision"]
    assert by_name["WeightsFile"].default == "<empty>"
    assert by_name["Backend"].choices == ("cpu", "cuda", "tensorrt")
    assert by_name["Backend"].default == "tensorrt"
    assert by_name["Precision"].choices == ("fp32", "fp16")
    assert by_name["Threads"].kind == "spin"
    assert by_name["Threads"].minimum == 1
    assert by_name["Threads"].maximum == 128
    assert by_name["Threads"].default == "4"
    assert by_name["CPuct"].kind == "string"
    assert by_name["CPuct"].default == "1.745"
    assert list(by_name)[10:14] == [
        "FpuValue",
        "FpuStrategyAtRoot",
        "FpuValueAtRoot",
        "PolicyTemperature",
    ]
    assert by_name["FpuStrategyAtRoot"].kind == "combo"
    assert by_name["FpuStrategyAtRoot"].default == "same"
    assert by_name["FpuStrategyAtRoot"].choices == ("same", "reduction", "absolute")
    assert by_name["FpuValueAtRoot"].kind == "string"
    assert by_name["FpuValueAtRoot"].default == "1"
    assert by_name["Deterministic"].kind == "check"
    assert by_name["Deterministic"].default == "false"
    assert by_name["MoveOverheadMs"].maximum == 60000
    with pytest.raises(AttributeError):
        result.options[0].name = "changed"


def test_future_option_is_documented_without_a_fixed_option_count() -> None:
    lines = OBSERVED.splitlines()
    lines.insert(-1, "option name Future capability type button")
    result = api().parse_handshake(lines)
    assert len(result.options) == 26
    assert result.options[-1].kind == "button"
    assert result.options[-1].default is None
    assert "Future capability" in api().render_markdown(result)


@pytest.mark.parametrize(
    "line",
    [
        "option name Enabled type check default true",
        "option name Offset type spin default -2 min -4 max 8",
        "option name 模型 file type string default C:/网络 folder/a.onnx",
        "option name Empty type string default ",
        "option name Variant type combo default slow mode var fast var slow mode",
        "option name Clear Hash type button",
    ],
)
def test_all_five_uci_option_kinds_parse_complete_declarations(line: str) -> None:
    assert api().parse_option(line).name


@pytest.mark.parametrize(
    "line",
    [
        "option name  type check default true",
        "option name Bad type unknown default 1",
        "option name Bad type check default True",
        "option name Bad type check default false extra",
        "option name Bad type spin default 1 min 2 max 3",
        "option name Bad type spin default 2 min 3 max 1",
        "option name Bad type spin default 1 min 0",
        "option name Bad type spin default 1 min 0 max 2 max 3",
        "option name Bad type spin default nan min 0 max 3",
        "option name Bad type spin default 1 min 0 max " + "9" * 21,
        "option name Bad type combo default x var y",
        "option name Bad type combo default x var x var x",
        "option name Bad type combo default x var ",
        "option name Bad type combo default x",
        "option name Bad type button default ignored",
        "option name Bad type string",
        "option name Bad type string default bad\x00value",
        "option name Bad\nInjected type string default value",
        "option name Bad\u202eName type string default value",
        "info string option name Bad type check default true",
    ],
)
def test_malformed_option_declarations_never_become_documentation(line: str) -> None:
    with pytest.raises(api().UciOptionsError):
        api().parse_option(line)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lines: lines[:-1],
        lambda lines: ["id name ", *lines[1:]],
        lambda lines: [lines[0], "id author ", *lines[2:]],
        lambda lines: [*lines[:-1], lines[2], "uciok"],
        lambda lines: [*lines[:-1], lines[2].replace("WeightsFile", "weightsfile"), "uciok"],
        lambda lines: [*lines, "uciok"],
        lambda lines: [*lines[:-1], "info string failed", "uciok"],
        lambda lines: [*lines[:2], "", *lines[2:]],
        lambda lines: ["id name Mouse", "id author Author", "uciok"],
    ],
)
def test_incomplete_duplicate_or_unexpected_protocol_is_rejected(mutation: Any) -> None:
    with pytest.raises(api().UciOptionsError):
        api().parse_handshake(mutation(OBSERVED.splitlines()))


def test_parser_bounds_do_not_allow_unbounded_counts_or_lines() -> None:
    module = api()
    lines = ["id name Mouse", "id author Author"]
    lines += [f"option name item{i} type button" for i in range(257)]
    with pytest.raises(module.UciOptionsError, match="bound"):
        module.parse_handshake([*lines, "uciok"])
    with pytest.raises(module.UciOptionsError, match="bound"):
        module.parse_option("option name Long type string default " + "x" * 8192)


def test_markdown_escapes_untrusted_protocol_and_retains_unicode() -> None:
    result = api().parse_handshake(
        [
            "id name Mouse <unsafe> | name",
            "id author Author & friends",
            "option name A|B`_* type string default <empty> & [网格] \\path",
            "option name Signed type spin default -2 min -4 max 8",
            "option name Modes type combo default fast var fast var slow mode",
            "option name Reset type button",
            "uciok",
        ]
    )
    table = api().render_markdown(result)
    assert "&lt;unsafe&gt; \\| name" in table
    assert "Author &amp; friends" in table
    assert "A\\|B\\`\\_\\*" in table
    assert "&lt;empty&gt; &amp; \\[网格\\] \\\\path" in table
    assert "-4 … 8" in table
    assert "fast; slow mode" in table
    assert "| Reset | button | — | — |" in table
    assert table.endswith("\n")


def test_atomic_update_preserves_utf8_crlf_and_all_bytes_outside_markers(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    prefix = f"# 网格\r\nkeep|*\r\n{BEGIN}\r\n".encode()
    suffix = f"{END}\r\nuntouched café".encode()
    path.write_bytes(prefix + b"stale\r\n" + suffix)
    table = api().render_markdown(observed())
    assert api().update_document(path, table, check=True)
    assert path.read_bytes() == prefix + b"stale\r\n" + suffix
    assert api().update_document(path, table)
    assert path.read_bytes() == prefix + table.replace("\n", "\r\n").encode() + suffix
    assert not api().update_document(path, table)
    assert not api().update_document(path, table, check=True)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["README.md"]


@pytest.mark.parametrize(
    "text",
    [
        "no markers",
        f"{END}\nold\n{BEGIN}\n",
        f"{BEGIN}\n{BEGIN}\n{END}\n",
        f"{BEGIN}\n{END}\n{END}\n",
        f"prefix {BEGIN}\nold\n{END}\n",
        f"{BEGIN}\nold\n{END} suffix\n",
    ],
)
def test_ambiguous_missing_or_nonstandalone_markers_leave_file_unchanged(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "README.md"
    path.write_text(text, encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(api().UciOptionsError, match="marker"):
        api().update_document(path, api().render_markdown(observed()))
    assert path.read_bytes() == before


def test_invalid_utf8_and_oversized_document_fail_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    for original in (b"\xff", b"x" * (4 * 1024 * 1024 + 1)):
        path.write_bytes(original)
        with pytest.raises(api().UciOptionsError):
            api().update_document(path, api().render_markdown(observed()))
        assert path.read_bytes() == original


def test_failed_atomic_replace_preserves_original_and_removes_owned_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path = document(tmp_path)
    original = path.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        raise OSError("controlled replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="controlled"):
        module.update_document(path, module.render_markdown(observed()))
    assert path.read_bytes() == original
    assert sorted(item.name for item in tmp_path.iterdir()) == ["README.md"]


def test_concurrent_document_change_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = api()
    path = document(tmp_path)
    original_fsync = os.fsync

    def changed_during_flush(handle: int) -> None:
        original_fsync(handle)
        path.write_text("a newer owner edit", encoding="utf-8")

    monkeypatch.setattr(module.os, "fsync", changed_during_flush)
    with pytest.raises(module.UciOptionsError, match="changed"):
        module.update_document(path, module.render_markdown(observed()))
    assert path.read_text(encoding="utf-8") == "a newer owner edit"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["README.md"]


def test_real_process_sends_only_uci_then_quit_and_keeps_exact_transcript() -> None:
    result = api().capture_handshake(child(engine_code(suffix="pass")), timeout=2)
    assert result == observed()


@pytest.mark.parametrize(
    ("code", "match"),
    [
        (engine_code(suffix="sys.stderr.write('fatal diagnostic')"), "stderr"),
        (engine_code(suffix="pass", exit_code=3), "exit"),
        (engine_code(suffix="print('unexpected',flush=True)"), "protocol|handshake"),
        ("print('id name Missing',flush=True)", "EOF|handshake"),
        ("print('x'*9000,flush=True)", "bound"),
        ("import sys;sys.stdout.buffer.write(b'bad\\xff\\n');sys.stdout.flush()", "UTF-8"),
    ],
)
def test_real_process_failures_publish_no_option_table(code: str, match: str) -> None:
    with pytest.raises(api().UciOptionsError, match=match):
        api().capture_handshake(child(code), timeout=2)


def test_handshake_deadline_reaps_and_joins_without_reader_leaks() -> None:
    before = {thread.ident for thread in threading.enumerate()}
    started = time.monotonic()
    with pytest.raises(api().UciOptionsError, match="deadline"):
        api().capture_handshake(child("import time;time.sleep(60)"), timeout=0.15)
    assert time.monotonic() - started < 3
    assert {thread.ident for thread in threading.enumerate()} == before


def test_cli_updates_checks_and_reports_errors_without_partial_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = api()
    path = document(tmp_path)
    binary = tmp_path / "mouse.exe"
    binary.write_bytes(b"CLI wiring test; real child transport covered separately")
    commands: list[list[str]] = []

    def capture(command: list[str]) -> Any:
        commands.append(command)
        return observed()

    monkeypatch.setattr(module, "capture_handshake", capture)
    arguments = ["--binary", str(binary), "--readme", str(path)]
    assert module.main([*arguments, "--check"]) == 1
    assert "stale" in capsys.readouterr().err
    assert "old\n" in path.read_text(encoding="utf-8")
    assert module.main(arguments) == 0
    assert module.main([*arguments, "--check"]) == 0
    assert commands == [[str(binary.resolve())]] * 3
    path.write_text("missing markers", encoding="utf-8")
    assert module.main(arguments) == 1
    output = capsys.readouterr()
    assert "marker" in output.err
    assert path.read_text(encoding="utf-8") == "missing markers"


def test_readonly_document_fails_before_allocating_a_temporary(tmp_path: Path) -> None:
    module = api()
    path = document(tmp_path)
    original = path.read_bytes()
    original_mode = path.stat().st_mode
    os.chmod(path, stat.S_IREAD)
    try:
        with pytest.raises(module.UciOptionsError, match="read.only"):
            module.update_document(path, module.render_markdown(observed()))
        assert path.read_bytes() == original
        assert sorted(item.name for item in tmp_path.iterdir()) == ["README.md"]
    finally:
        os.chmod(path, original_mode)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_capture_deadlines_fail_before_process_launch(timeout: float) -> None:
    with pytest.raises(api().UciOptionsError, match="deadline"):
        api().capture_handshake(child("raise SystemExit(99)"), timeout=timeout)


def test_per_user_cache_docs_are_stable_across_profiles_without_masking_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = api()
    key = "LOCALAPPDATA" if sys.platform == "win32" else "XDG_CACHE_HOME"
    roots = (
        ("C:/Users/A/AppData/Local", "D:/Profiles/B/Local")
        if sys.platform == "win32"
        else ("/home/a/.cache", "/home/b/.cache")
    )
    rendered = []
    for root in roots:
        monkeypatch.setenv(key, root)
        raw = OBSERVED.replace(
            "default .cache/mouse",
            "default " + root + "/Mouse/cache",
        )
        table = module.render_markdown(module.parse_handshake(raw.splitlines()))
        assert "| CacheDirectory | string | per-user cache |" in table
        assert root not in table
        rendered.append(table)
    assert rendered[0] == rendered[1]
    override = OBSERVED.replace("default .cache/mouse", "default /explicit/portable-cache")
    assert "/explicit/portable-cache" in module.render_markdown(
        module.parse_handshake(override.splitlines())
    )
    monkeypatch.delenv(key)
    old_default = OBSERVED.replace("default .cache/mouse", "default " + roots[0] + "/Mouse/cache")
    assert roots[0] in module.render_markdown(module.parse_handshake(old_default.splitlines()))


def test_home_cache_docs_only_normalize_the_actual_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api().sys, "platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/native-user")
    raw = OBSERVED.replace("default .cache/mouse", "default /home/native-user/.cache/Mouse/cache")
    assert "| CacheDirectory | string | per-user cache |" in api().render_markdown(
        api().parse_handshake(raw.splitlines())
    )


@pytest.mark.parametrize(
    "version",
    [
        "0.5.0-dev",
        "1.0.0-rc.1",
        "1.0.0",
        "unknown",
        "0.0.0",
        "1.2.3-alpha.0",
        "1.2.3-01a",
        "1.2.3--",
        "1.2.3+001.build",
        "1.2.3-rc.0+build.01",
    ],
)
def test_mouse_build_versions_share_only_documentation_provenance(
    version: str, tmp_path: Path
) -> None:
    module = api()
    raw = OBSERVED.replace("id name Mouse 0.5.0-dev", "id name Mouse " + version, 1)
    result = module.parse_handshake(raw.splitlines())
    table = module.render_markdown(result)
    expected_header = "Generated from Mouse (Zulfeqar Malik) via `uci` (build version omitted)."
    assert table.splitlines()[0] == expected_header
    assert table == module.render_markdown(observed())
    assert result.engine_name == "Mouse " + version
    assert result.transcript == tuple(raw.splitlines())
    path = document(tmp_path)
    module.update_document(path, module.render_markdown(observed()))
    assert not module.update_document(path, table, check=True)


@pytest.mark.parametrize("version", ["0.5.0-dev", "1.0.0-rc.1", "1.0.0", "unknown"])
def test_real_child_capture_preserves_versioned_identity_and_transcript(version: str) -> None:
    module = api()
    raw = OBSERVED.replace("id name Mouse 0.5.0-dev", "id name Mouse " + version, 1)
    code = engine_code(suffix="pass").replace(repr(OBSERVED), repr(raw))
    result = module.capture_handshake(child(code), timeout=2)
    assert result.engine_name == "Mouse " + version
    assert result.transcript == tuple(raw.splitlines())
    assert result.options == observed().options
    assert module.render_markdown(result).splitlines()[0] == (
        "Generated from Mouse (Zulfeqar Malik) via `uci` (build version omitted)."
    )


@pytest.mark.parametrize(
    "engine_name",
    [
        "Mouse",
        "OtherMouse 1.0.0",
        "mouse 1.0.0",
        "MOUSE 1.0.0",
        "Mouse  1.0.0",
        "Mouse 1.0.0 ",
        "Mouse 1.0.0 extra",
        "Mouse unknown extra",
        "Mouse UNKNOWN",
        "unknown",
        "Mouse 01.0.0",
        "Mouse 1.00.0",
        "Mouse 1.0.00",
        "Mouse 1.0",
        "Mouse v1.0.0",
        "Mouse 1.0.0-01",
        "Mouse 1.0.0-rc.01",
        "Mouse 1.0.0-rc..1",
        "Mouse 1.0.0-",
        "Mouse 1.0.0+",
        "Mouse 1.0.0+build..1",
        "Mouse 1.0.0+build_1",
    ],
)
def test_unrecognized_engine_identity_remains_literal_and_stale(
    engine_name: str, tmp_path: Path
) -> None:
    module = api()
    raw = OBSERVED.replace("id name Mouse 0.5.0-dev", "id name " + engine_name, 1)
    result = module.parse_handshake(raw.splitlines())
    table = module.render_markdown(result)
    assert "(build version omitted)" not in table.splitlines()[0]
    assert result.engine_name == engine_name
    assert result.transcript == tuple(raw.splitlines())
    path = document(tmp_path)
    module.update_document(path, module.render_markdown(observed()))
    before = path.read_bytes()
    assert module.update_document(path, table, check=True)
    assert path.read_bytes() == before


def test_recognized_mouse_keeps_actual_escaped_author_and_author_changes_stale(
    tmp_path: Path,
) -> None:
    module = api()
    raw = OBSERVED.replace("id author Zulfeqar Malik", "id author Other & [team]", 1)
    result = module.parse_handshake(raw.splitlines())
    table = module.render_markdown(result)
    assert table.splitlines()[0] == (
        "Generated from Mouse (Other &amp; \\[team\\]) via `uci` (build version omitted)."
    )
    assert result.author == "Other & [team]"
    path = document(tmp_path)
    module.update_document(path, module.render_markdown(observed()))
    assert module.update_document(path, table, check=True)


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        (
            "option name CPuct type string default 1.745",
            "option name NewCPuct type string default 1.745",
        ),
        (
            "option name CPuct type string default 1.745",
            "option name CPuct type check default true",
        ),
        (
            "option name CPuct type string default 1.745",
            "option name CPuct type string default 1.746",
        ),
        ("default 4 min 1 max 128", "default 4 min 1 max 129"),
        ("var cpu var cuda var tensorrt", "var cuda var cpu var tensorrt"),
        (
            (
                "option name WeightsFile type string default <empty>\n"
                "option name Backend type combo default tensorrt var cpu var cuda var tensorrt"
            ),
            (
                "option name Backend type combo default tensorrt var cpu var cuda var tensorrt\n"
                "option name WeightsFile type string default <empty>"
            ),
        ),
    ],
    ids=["name", "type", "default", "range", "choice-order", "declaration-order"],
)
def test_build_provenance_normalization_preserves_strict_option_staleness(
    original: str, changed: str, tmp_path: Path
) -> None:
    module = api()
    raw = OBSERVED.replace("Mouse 0.5.0-dev", "Mouse 1.0.0-rc.1", 1).replace(original, changed, 1)
    table = module.render_markdown(module.parse_handshake(raw.splitlines()))
    path = document(tmp_path)
    module.update_document(path, module.render_markdown(observed()))
    before = path.read_bytes()
    assert module.update_document(path, table, check=True)
    assert path.read_bytes() == before
