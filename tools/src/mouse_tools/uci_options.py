# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate bounded, code-driven UCI option tables without loading a network.

Only ``uci`` and ``quit`` are sent. The existing UCI transport owns subprocess
cleanup; parsing intentionally does not import its fixed Mouse option catalog.
One caller owns an update. An optimistic content check rejects intervening edits;
concurrent writers still require external serialization around the final replace.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import ntpath
import os
import posixpath
import re
import stat
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from mouse_tools.uci_gate import UciGateError, UciProcess

BEGIN_MARKER = "<!-- BEGIN MOUSE UCI OPTIONS -->"
END_MARKER = "<!-- END MOUSE UCI OPTIONS -->"
MAXIMUM_OPTIONS = 256
MAXIMUM_LINE_BYTES = 8192
MAXIMUM_TRANSCRIPT_BYTES = 512 * 1024
MAXIMUM_DOCUMENT_BYTES = 4 * 1024 * 1024
_INTEGER = r"-?(?:0|[1-9][0-9]{0,19})"
_SPIN = re.compile(rf"default ({_INTEGER}) min ({_INTEGER}) max ({_INTEGER})")
_KINDS = frozenset(("check", "spin", "combo", "button", "string"))
# Keep identity recognition aligned with cmake/ValidateBuildIdentity.cmake.
_SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


class UciOptionsError(ValueError):
    """Malformed, stale, incomplete or unbounded generated documentation."""


@dataclass(frozen=True)
class UciOption:
    name: str
    kind: str
    default: str | None
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class Handshake:
    engine_name: str
    author: str
    options: tuple[UciOption, ...]
    transcript: tuple[str, ...]


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.isprintable() and value:
        raise UciOptionsError("Protocol text contains a control or nonprinting character")
    try:
        size = len(value.encode("utf-8", errors="strict"))
    except UnicodeError as error:
        raise UciOptionsError("Protocol text is not strict UTF-8") from error
    if size > MAXIMUM_LINE_BYTES:
        raise UciOptionsError("Protocol line exceeded its byte bound")


def _integer(token: str) -> int:
    value = int(token)
    if not -(1 << 63) <= value <= (1 << 64) - 1:
        raise UciOptionsError("Spin integer exceeded its 64-bit bound")
    return value


def parse_option(line: str) -> UciOption:
    """Validate one complete declaration, retaining exact names/default text."""
    _text(line)
    if not line.startswith("option name "):
        raise UciOptionsError("Expected a complete option name/type declaration")
    name, separator, tail = line.removeprefix("option name ").partition(" type ")
    if not separator or not name or name != name.strip():
        raise UciOptionsError("Option declaration has a missing or ambiguous name/type")
    kind, separator, attributes = tail.partition(" ")
    if kind not in _KINDS:
        raise UciOptionsError("Unknown UCI option type")
    if kind == "button":
        if separator:
            raise UciOptionsError("Button option cannot declare a default or attributes")
        return UciOption(name, kind, None)
    if not attributes.startswith("default "):
        raise UciOptionsError("Non-button option requires an explicit default")
    default = attributes.removeprefix("default ")
    if kind == "check":
        if default not in ("true", "false"):
            raise UciOptionsError("Check default must be true or false")
    elif kind == "spin":
        matched = _SPIN.fullmatch(attributes)
        if matched is None:
            raise UciOptionsError("Spin requires one integer default/min/max declaration")
        value, minimum, maximum = map(_integer, matched.groups())
        if not minimum <= value <= maximum:
            raise UciOptionsError("Spin default is outside its ordered min/max range")
        return UciOption(name, kind, matched[1], minimum, maximum)
    elif kind == "combo":
        parts = default.split(" var ")
        default, choices = parts[0], tuple(parts[1:])
        if (
            not choices
            or any(not choice or choice != choice.strip() for choice in choices)
            or len(set(choices)) != len(choices)
            or default not in choices
        ):
            raise UciOptionsError("Combo requires unique choices containing its default")
        return UciOption(name, kind, default, choices=choices)
    return UciOption(name, kind, default)


def parse_handshake(lines: Sequence[str]) -> Handshake:
    """Require complete identity, unique typed declarations and one final uciok."""
    if not 4 <= len(lines) <= MAXIMUM_OPTIONS + 3:
        raise UciOptionsError("Incomplete handshake or option count exceeded bound")
    for line in lines:
        _text(line)
    if sum(len(line.encode("utf-8")) + 1 for line in lines) > MAXIMUM_TRANSCRIPT_BYTES:
        raise UciOptionsError("Handshake transcript exceeded byte bound")
    if (
        not lines[0].startswith("id name ")
        or not lines[0].removeprefix("id name ").strip()
        or not lines[1].startswith("id author ")
        or not lines[1].removeprefix("id author ").strip()
        or lines[-1] != "uciok"
    ):
        raise UciOptionsError("Incomplete or unexpected UCI identity/handshake protocol")
    options = tuple(parse_option(line) for line in lines[2:-1])
    if len({option.name.casefold() for option in options}) != len(options):
        raise UciOptionsError("Duplicate UCI option name")
    return Handshake(
        lines[0].removeprefix("id name "),
        lines[1].removeprefix("id author "),
        options,
        tuple(lines),
    )


def _binary_hash(path: Path) -> str:
    if not path.is_file():
        raise UciOptionsError("Native executable must be a regular file")
    digest = hashlib.sha256()
    remaining = 512 * 1024 * 1024
    with path.open("rb") as source:
        while block := source.read(min(65536, remaining + 1)):
            remaining -= len(block)
            if remaining < 0:
                raise UciOptionsError("Native executable exceeded byte bound")
            digest.update(block)
    return digest.hexdigest()


def capture_handshake(command: Sequence[str], *, timeout: float = 10) -> Handshake:
    """Capture one native handshake with bounded I/O/deadlines and owned cleanup.

    Timeout bounds each transport operation; an additional absolute handshake
    deadline prevents endless valid slow-drip declarations. No isready/go or
    option commands can initialize a model/provider. Binary changes reject.
    """
    try:
        if not command:
            raise UciOptionsError("A native executable command is required")
        binary = Path(command[0]).resolve(strict=True)
        before = _binary_hash(binary)
        with UciProcess(
            [str(binary), *command[1:]],
            timeout=timeout,
            line_limit=MAXIMUM_LINE_BYTES,
            stdout_limit=MAXIMUM_TRANSCRIPT_BYTES,
            stderr_limit=65536,
            queue_limit=MAXIMUM_OPTIONS + 8,
        ) as process:
            deadline = time.monotonic() + timeout
            process.send("uci")
            count = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise UciOptionsError("UCI handshake deadline exceeded")
                line = process.line(timeout=remaining)
                count += 1
                if count > MAXIMUM_OPTIONS + 3:
                    raise UciOptionsError("UCI handshake option count exceeded bound")
                if line == "uciok":
                    break
            process.send("quit")
            code = process.wait_exit()
            if code != 0:
                raise UciOptionsError(f"Native handshake exited with code {code}")
            if process.stderr:
                diagnostic = bytes(process.stderr[:2048]).decode("utf-8", errors="replace")
                raise UciOptionsError(f"Native handshake wrote stderr: {diagnostic}")
            raw = bytes(process.stdout)
        if _binary_hash(binary) != before:
            raise UciOptionsError("Native executable changed during handshake")
        if not raw.endswith(b"\n"):
            raise UciOptionsError("Incomplete UCI handshake protocol line")
        lines = [part.removesuffix(b"\r").decode("utf-8") for part in raw[:-1].split(b"\n")]
        return parse_handshake(lines)
    except (OSError, UnicodeError, UciGateError) as error:
        raise UciOptionsError(f"UCI handshake failed: {error}") from error


def _cell(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return re.sub(r"([\\`*_\[\]|])", r"\\\1", escaped)


def _display_default(option: UciOption) -> str:
    if option.default is None:
        return "\u2014"
    # Native handshakes retain the actual resolved path. Documentation compares
    # only the verified environment default using the platform's lexical path
    # rules, so profiles/OSs share a table while explicit overrides stay literal.
    if option.name == "CacheDirectory" and option.kind == "string":
        if sys.platform == "win32":
            root = os.environ.get("LOCALAPPDATA", "")
            expected = ntpath.join(root, "Mouse", "cache")
            if ntpath.isabs(root) and ntpath.normcase(ntpath.normpath(option.default)) == (
                ntpath.normcase(ntpath.normpath(expected))
            ):
                return "per-user cache"
        else:
            root = os.environ.get("XDG_CACHE_HOME", "")
            if not root:
                home = os.environ.get("HOME", "")
                root = posixpath.join(home, ".cache") if posixpath.isabs(home) else ""
            expected = posixpath.join(root, "Mouse", "cache")
            if posixpath.isabs(root) and posixpath.normpath(option.default) == posixpath.normpath(
                expected
            ):
                return "per-user cache"
    return option.default


def _provenance(handshake: Handshake) -> str:
    """Omit only a verified Mouse build version in documentation, retaining raw UCI."""
    version = handshake.engine_name.removeprefix("Mouse ")
    matched = _SEMVER.fullmatch(version)
    prerelease = matched["prerelease"] if matched is not None else None
    recognized = handshake.engine_name.startswith("Mouse ") and (
        version == "unknown"
        or matched is not None
        and not any(re.fullmatch(r"0[0-9]+", part) for part in (prerelease or "").split("."))
    )
    if recognized:
        return (
            f"Generated from Mouse ({_cell(handshake.author)}) via `uci` (build version omitted)."
        )
    return f"Generated from {_cell(handshake.engine_name)} ({_cell(handshake.author)}) via `uci`."


def render_markdown(handshake: Handshake) -> str:
    """Render validated records in native declaration order, escaping cell text."""
    lines = [
        _provenance(handshake),
        "",
        "| Option | Type | Default | Range / choices |",
        "| --- | --- | --- | --- |",
    ]
    for option in handshake.options:
        bounds = "—"
        if option.kind == "spin":
            bounds = f"{option.minimum} … {option.maximum}"
        elif option.kind == "combo":
            bounds = "; ".join(option.choices)
        default = _display_default(option)
        cells = (option.name, option.kind, default, bounds)
        lines.append("| " + " | ".join(map(_cell, cells)) + " |")
    return "\n".join(lines) + "\n"


def _document(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise UciOptionsError("README target must be an existing regular nonsymlink file")
    with path.open("rb") as source:
        raw = source.read(MAXIMUM_DOCUMENT_BYTES + 1)
    if len(raw) > MAXIMUM_DOCUMENT_BYTES:
        raise UciOptionsError("README exceeded document byte bound")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise UciOptionsError("README is not strict UTF-8") from error
    return raw


def update_document(path: Path, table: str, *, check: bool = False) -> bool:
    """Return whether replacement is needed; check=True never mutates the file.

    Exactly one standalone marker pair must already exist. Preserve every byte
    outside its interior, plus the beginning marker's LF/CRLF convention. A
    sibling temporary is flushed, permission-matched and atomically replaced;
    only that owned temporary is removed on failures.
    """
    original = _document(path)
    begin, end = BEGIN_MARKER.encode("ascii"), END_MARKER.encode("ascii")
    if original.count(begin) != 1 or original.count(end) != 1:
        raise UciOptionsError("README requires exactly one fixed marker pair")
    start = re.search(rb"(?m)^" + re.escape(begin) + rb"(\r?\n)", original)
    finish = re.search(rb"(?m)^" + re.escape(end) + rb"(?:\r?\n|$)", original)
    if start is None or finish is None or start.end() > finish.start():
        raise UciOptionsError("README markers must be standalone and correctly ordered")
    if BEGIN_MARKER in table or END_MARKER in table or not table.endswith("\n"):
        raise UciOptionsError("Generated table has an invalid marker/newline boundary")
    interior = table.replace("\n", start[1].decode("ascii")).encode("utf-8")
    replacement = original[: start.end()] + interior + original[finish.start() :]
    if len(replacement) > MAXIMUM_DOCUMENT_BYTES:
        raise UciOptionsError("Generated README exceeded document byte bound")
    if replacement == original:
        return False
    if check:
        return True
    temporary: Path | None = None
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if not mode & stat.S_IWUSR:
            raise UciOptionsError("README is read-only; refusing to replace it")
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(replacement)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        if _document(path) != original:
            raise UciOptionsError("README changed during generation; refusing to overwrite")
        os.replace(temporary, path)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument(
        "--check", action="store_true", help="fail on stale content without writing"
    )
    args = parser.parse_args(argv)
    try:
        binary = args.binary.resolve(strict=True)
        handshake = capture_handshake([str(binary)])
        changed = update_document(args.readme, render_markdown(handshake), check=args.check)
        if args.check and changed:
            print("UCI options documentation is stale", file=sys.stderr)
            return 1
        action = "updated" if changed else "current"
        print(f"UCI options documentation {action}: {len(handshake.options)} options")
        return 0
    except (OSError, UciOptionsError) as error:
        print(f"UCI options documentation: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("UCI options documentation interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
