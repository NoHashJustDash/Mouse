# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded original CPU-only UCI process conformance with immutable evidence."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import os
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import BinaryIO, Self, cast

MAXIMUM_LINE_BYTES = 1024 * 1024
MOVE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


class UciGateError(ValueError):
    """Incomplete, malformed, unbounded, or changed protocol evidence."""


def _positive(value: float, name: str, maximum: float = 300) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= maximum:
        raise UciGateError(f"{name} must be finite in (0,{maximum}]")


def creation_flags() -> int:
    """Use a hidden Windows child; other platforms have no Windows creation flags."""
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _windows_kernel() -> ctypes.CDLL:
    if sys.platform == "win32":
        return ctypes.WinDLL("kernel32", use_last_error=True)
    raise UciGateError("Windows job ownership requires Windows")


class _WindowsJob:
    """Kill-on-close native job owns descendants and their inherited pipe handles."""

    def __init__(self) -> None:
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [
                ("per_process", ctypes.c_int64),
                ("per_job", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("minimum", ctypes.c_size_t),
                ("maximum", ctypes.c_size_t),
                ("active", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class Counters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64)
                for name in ("read_ops", "write_ops", "other_ops", "read", "write", "other")
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", Counters),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process", ctypes.c_size_t),
                ("peak_job", ctypes.c_size_t),
            ]

        self.kernel = _windows_kernel()
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = cast(int | None, self.kernel.CreateJobObjectW(None, None))
        if not self.handle:
            raise UciGateError("Cannot create child process job")
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, Windows SDK ABI.
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise UciGateError("Cannot configure kill-on-close child process job")

    def assign(self, pid: int) -> None:
        handle = self.kernel.OpenProcess(0x0101, False, pid)  # SET_QUOTA | TERMINATE.
        if not handle:
            raise UciGateError("Cannot open owned child for job assignment")
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, handle):
                raise UciGateError("Cannot assign child to kill-on-close process job")
        finally:
            self.kernel.CloseHandle(handle)

    def close(self) -> None:
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class UciProcess:
    """One owned process tree, three bounded joined pumps, no daemon readers.

    Only the driver reads line(); pumps own native streams. The writer queue and
    acknowledged writes make even a child that refuses stdin subject to a deadline.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = 10,
        cwd: Path | None = None,
        line_limit: int = 65536,
        stdout_limit: int = 4 * 1024 * 1024,
        stderr_limit: int = 1024 * 1024,
        queue_limit: int = 2048,
    ) -> None:
        _positive(timeout, "deadline")
        if not command or any(not isinstance(value, str) or "\x00" in value for value in command):
            raise UciGateError("A complete native child command is required")
        for value, maximum in (
            (line_limit, 65536),
            (stdout_limit, 8 * 1024 * 1024),
            (stderr_limit, 2 * 1024 * 1024),
            (queue_limit, 4096),
        ):
            if type(value) is not int or not 1 <= value <= maximum:
                raise UciGateError("Invalid bounded transport limits")
        self.command = list(command)
        self.timeout = timeout
        self.cwd = cwd
        self.line_limit = line_limit
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit
        self.lines: queue.Queue[str] = queue.Queue(maxsize=queue_limit)
        self.writes: queue.Queue[tuple[bytes, threading.Event]] = queue.Queue(maxsize=8)
        self.failure: str | None = None
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.stdout_done = threading.Event()
        self.stderr_done = threading.Event()
        self.stdin_done = threading.Event()
        self.eof_requested = threading.Event()
        self.threads: list[threading.Thread] = []
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.stdin = bytearray()
        self.submitted: list[dict[str, object]] = []
        self.submitted_bytes = 0
        self.process: subprocess.Popen[bytes]
        self.job: _WindowsJob | None = None
        self.started = 0.0
        self.finished = 0.0
        self.closed_cleanly = False
        self.terminated_by_runner = False

    def _fail(self, message: str) -> None:
        with self.lock:
            if self.failure is None:
                self.failure = message

    def _check(self) -> None:
        if self.failure is not None:
            raise UciGateError(self.failure)

    def _pump(self, stream: BinaryIO, *, stdout: bool) -> None:
        pending = bytearray()
        data = self.stdout if stdout else self.stderr
        limit = self.stdout_limit if stdout else self.stderr_limit
        done = self.stdout_done if stdout else self.stderr_done
        try:
            while True:
                block = stream.read(4096)
                if not block:
                    if stdout and pending:
                        self._fail("unterminated stdout protocol line")
                    return
                room = max(0, limit - len(data))
                data.extend(block[:room])
                if len(block) > room:
                    self._fail(
                        "stdout capture exceeded bound"
                        if stdout
                        else "stderr capture exceeded bound"
                    )
                    return
                if not stdout:
                    continue
                pending.extend(block)
                while b"\n" in pending:
                    index = pending.index(10)
                    raw = bytes(pending[:index])
                    del pending[: index + 1]
                    if len(raw) > self.line_limit:
                        self._fail("stdout line exceeded bound")
                        return
                    try:
                        line = raw.removesuffix(b"\r").decode("utf-8", errors="strict")
                    except UnicodeError:
                        self._fail("stdout protocol is not strict UTF-8")
                        return
                    try:
                        self.lines.put_nowait(line)
                    except queue.Full:
                        self._fail("stdout line queue exceeded bound")
                        return
                if len(pending) > self.line_limit:
                    self._fail("stdout line exceeded bound")
                    return
        except (OSError, ValueError) as error:
            if not self.closed.is_set():
                self._fail(f"native output read failed: {error}")
        finally:
            done.set()

    def _writer(self, stream: BinaryIO) -> None:
        try:
            while not self.closed.is_set():
                try:
                    data, acknowledged = self.writes.get(timeout=0.02)
                except queue.Empty:
                    if self.eof_requested.is_set():
                        stream.close()
                        return
                    continue
                offset = 0
                try:
                    while offset < len(data):
                        written = stream.write(data[offset : offset + 65536])
                        if not written:
                            raise OSError("native stdin accepted no bytes")
                        self.stdin.extend(data[offset : offset + written])
                        offset += written
                    stream.flush()
                finally:
                    acknowledged.set()
        except (OSError, ValueError) as error:
            if not self.closed.is_set():
                self._fail(f"native stdin write failed: {error}")
        finally:
            self.stdin_done.set()

    def __enter__(self) -> Self:
        self.started = time.monotonic()
        self.job = _WindowsJob() if sys.platform == "win32" else None
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env={**os.environ, "PATH": ""},
                bufsize=0,
                creationflags=creation_flags(),
                start_new_session=sys.platform != "win32",
            )
            if self.job is not None:
                self.job.assign(self.process.pid)
            assert self.process.stdin and self.process.stdout and self.process.stderr
            for thread in (
                threading.Thread(target=self._writer, args=(self.process.stdin,)),
                threading.Thread(
                    target=self._pump, args=(self.process.stdout,), kwargs={"stdout": True}
                ),
                threading.Thread(
                    target=self._pump, args=(self.process.stderr,), kwargs={"stdout": False}
                ),
            ):
                thread.start()
                self.threads.append(thread)
            return self
        except BaseException:
            self.close()
            raise

    def send(self, line: str, *, newline: bool = True) -> None:
        data = line.encode("utf-8") + (b"\n" if newline else b"")
        if (
            len(data) > MAXIMUM_LINE_BYTES + 2
            or self.submitted_bytes + len(data) > 4 * 1024 * 1024
            or len(self.submitted) >= 4096
        ):
            raise UciGateError("stdin command/case bound exceeded")
        if self.eof_requested.is_set():
            raise UciGateError("Cannot write after stdin EOF")
        self._check()
        self.submitted_bytes += len(data)
        self.submitted.append(
            {
                "text": line + ("\n" if newline else ""),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "at_seconds": time.monotonic() - self.started,
            }
        )
        acknowledged = threading.Event()
        deadline = time.monotonic() + self.timeout
        try:
            self.writes.put((data, acknowledged), timeout=self.timeout)
        except queue.Full as error:
            raise UciGateError("stdin queue deadline exceeded") from error
        while not acknowledged.wait(min(0.02, max(0, deadline - time.monotonic()))):
            self._check()
            if time.monotonic() >= deadline:
                raise UciGateError("stdin write deadline exceeded")
        self._check()

    def close_stdin(self) -> None:
        self.eof_requested.set()
        if not self.stdin_done.wait(self.timeout):
            self._check()
            raise UciGateError("stdin close deadline exceeded")
        self._check()

    def line(self, *, timeout: float | None = None) -> str:
        duration = self.timeout if timeout is None else timeout
        _positive(duration, "line deadline")
        deadline = time.monotonic() + duration
        while True:
            self._check()
            try:
                return self.lines.get(timeout=min(0.02, max(0, deadline - time.monotonic())))
            except queue.Empty:
                if self.stdout_done.is_set():
                    self._check()
                    raise UciGateError("Unexpected EOF before required protocol response") from None
                if time.monotonic() >= deadline:
                    raise UciGateError("protocol response deadline exceeded") from None

    def wait_exit(self) -> int:
        deadline = time.monotonic() + self.timeout
        while self.process.poll() is None:
            self._check()
            if time.monotonic() >= deadline:
                raise UciGateError("process exit deadline exceeded")
            time.sleep(0.01)
        for event in (self.stdout_done, self.stderr_done):
            if not event.wait(max(0, deadline - time.monotonic())):
                raise UciGateError("output drain deadline exceeded")
        self._check()
        return cast(int, self.process.returncode)

    def close(self) -> None:
        if hasattr(self, "process") and self.process.poll() is None:
            self.terminated_by_runner = True
        self.closed.set()
        if self.job is not None:
            self.job.close()
        if hasattr(self, "process"):
            if sys.platform != "win32":
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
            if self.process.poll() is None:
                self.process.kill()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._fail("owned child did not reap after kill")
        for thread in self.threads:
            thread.join(timeout=2)
            if thread.is_alive():
                self._fail("owned transport pump did not join after child kill")
        if hasattr(self, "process"):
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                if stream is not None:
                    stream.close()
        self.finished = time.monotonic()
        self.closed_cleanly = not any(thread.is_alive() for thread in self.threads)

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
        if kind is None:
            self._check()

    def evidence(self) -> dict[str, object]:
        result: dict[str, object] = {
            "command": self.command,
            "pid": self.process.pid,
            "commands": self.submitted,
            "exit_code": self.process.returncode,
            "error": self.failure,
            "elapsed_seconds": (self.finished or time.monotonic()) - self.started,
            "pumps_joined": self.closed_cleanly,
            "terminated_by_runner": self.terminated_by_runner,
            "capture_complete": self.failure is None
            and self.stdout_done.is_set()
            and self.stderr_done.is_set(),
            "stdin_eof_requested": self.eof_requested.is_set(),
        }
        for name, raw in (("stdin", self.stdin), ("stdout", self.stdout), ("stderr", self.stderr)):
            result[name + "_text"] = raw.decode("utf-8", errors="backslashreplace")
            result[name + "_base64"] = base64.b64encode(raw).decode("ascii")
            result[name + "_sha256"] = hashlib.sha256(raw).hexdigest()
            result[name + "_bytes"] = len(raw)
        return result


def _integer(token: str, minimum: int = 0, maximum: int = (1 << 64) - 1) -> int:
    if re.fullmatch(r"-?[0-9]{1,20}", token) is None:
        raise UciGateError("Invalid finite bounded protocol integer")
    number = int(token)
    if not minimum <= number <= maximum:
        raise UciGateError("Protocol integer exceeds allowed bounds")
    return number


def parse_bestmove(line: str) -> str:
    fields = line.split(" ")
    if (
        len(fields) != 2
        or fields[0] != "bestmove"
        or (fields[1] != "0000" and MOVE.fullmatch(fields[1]) is None)
    ):
        raise UciGateError("Malformed complete bestmove line")
    return fields[1]


def parse_info(line: str) -> dict[str, object]:
    """Strict Mouse info schema, preserving timing while checking every token."""
    fields = line.split(" ")
    required = ("depth", "seldepth", "nodes", "nps", "time", "hashfull", "tbhits", "multipv")
    if not fields or fields.pop(0) != "info":
        raise UciGateError("Expected structured info line")
    result: dict[str, object] = {}
    for key in required:
        if len(fields) < 2 or fields.pop(0) != key:
            raise UciGateError("Missing, duplicate, or reordered info field")
        result[key] = _integer(
            fields.pop(0),
            1 if key == "multipv" else 0,
            1000 if key == "hashfull" else 512 if key in ("depth", "seldepth") else (1 << 64) - 1,
        )
    if len(fields) < 3 or fields[:2] != ["score", "cp"]:
        raise UciGateError("Expected explicit score cp, never an inferred mate score")
    result["score_cp"] = _integer(fields[2], -(1 << 31), (1 << 31) - 1)
    del fields[:3]
    if fields and fields[0] == "wdl":
        if len(fields) < 4:
            raise UciGateError("Incomplete WDL info")
        result["wdl"] = [_integer(token, 0, 1000) for token in fields[1:4]]
        if sum(cast(list[int], result["wdl"])) != 1000:
            raise UciGateError("WDL permille must sum exactly to 1000")
        del fields[:4]
    pv: list[str] = []
    if fields:
        if (
            fields.pop(0) != "pv"
            or not fields
            or any(MOVE.fullmatch(move) is None for move in fields)
        ):
            raise UciGateError("Malformed PV or unknown trailing info fields")
        pv = fields
    result["pv"] = pv
    if (
        result["tbhits"] != 0
        or cast(int, result["depth"]) > cast(int, result["seldepth"])
        or cast(int, result["depth"]) != len(pv)
    ):
        raise UciGateError("Inconsistent depth/PV or unsupported tablebase claim")
    return result


# Exact tiny original CPU fixtures. This CLI cannot accidentally execute T1/OOM.
VALID_MODEL_SHA256 = "3102463d573789b3da22e19e8b76faa3c50a724b89f83b6a9d5e4aa49fe7fdf2"
FAILURE_MODEL_SHA256 = "762400c217c36eaa4395bbb0d1b45fc158d9ffc9a0b0f04e6ad2aad871a7d7ff"
GOLDEN_SHA256 = "8ec818dd8707e1510360c5c62cd12d4236f9046d33b469517673b308a69c4a53"
README_SHA256 = "308b96332a211215a006e3a10d6566fb179b429207db2b3cb2f2b2f9be289894"

OPTION_TYPES: Mapping[str, tuple[str, int | None, int | None]] = MappingProxyType(
    {
        "WeightsFile": ("string", None, None),
        "Backend": ("combo", None, None),
        "Precision": ("combo", None, None),
        "Threads": ("spin", 1, 128),
        "MinibatchSize": ("spin", 0, 1024),  # 0 = automatic (2 x GPU SM count)
        "NNCacheSize": ("spin", 0, 16000000),
        "CacheHistoryLength": ("spin", 0, 7),
        "RamLimitMb": ("spin", 1, 16384),
        "CPuct": ("string", None, None),
        "CPuctBase": ("string", None, None),
        "CPuctFactor": ("string", None, None),
        "FpuValue": ("string", None, None),
        "FpuStrategyAtRoot": ("combo", None, None),
        "FpuValueAtRoot": ("string", None, None),
        "PolicyTemperature": ("string", None, None),
        "MultiPV": ("spin", 1, 256),
        "UCI_ShowWDL": ("check", None, None),
        "UCI_Chess960": ("check", None, None),
        "Ponder": ("check", None, None),
        "LogSearchStats": ("check", None, None),
        "MoveOverheadMs": ("spin", 0, 60000),
        "SmartPruningFactor": ("string", None, None),
        "SmartPruningMinimumBatches": ("spin", 0, 10000),
        "GpuDllDirectories": ("string", None, None),
        "CacheDirectory": ("string", None, None),
        "Device": ("spin", 0, 1023),
        "CpuThreads": ("spin", 1, 128),
        "WorkspaceMb": ("spin", 1, 16384),
        "Deterministic": ("check", None, None),
    }
)
REAL_OPTIONS = frozenset(
    (
        "CPuct",
        "CPuctBase",
        "CPuctFactor",
        "FpuValue",
        "FpuValueAtRoot",
        "PolicyTemperature",
        "SmartPruningFactor",
    )
)


def validate_handshake(
    lines: Sequence[str],
    expected: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    if (
        len(lines) != len(OPTION_TYPES) + 3
        or not re.fullmatch(r"id name Mouse [0-9]+\.[0-9]+\.[0-9]+[0-9A-Za-z._+\-]*", lines[0])
        or lines[1] != "id author Zulfeqar Malik"
        or lines[-1] != "uciok"
    ):
        raise UciGateError("Incomplete, duplicate, or unexpected UCI identity/options transcript")
    result: dict[str, dict[str, object]] = {}
    for line in lines[2:-1]:
        match = re.fullmatch(r"option name (\w+) type (\w+) default (.*)", line)
        if match is None or match[1] not in OPTION_TYPES or match[1] in result:
            raise UciGateError("Unknown, duplicate, or malformed option declaration")
        name, kind, value = match.groups()
        required_kind, minimum, maximum = OPTION_TYPES[name]
        if kind != required_kind:
            raise UciGateError("Option type differs from the fixed protocol contract")
        if kind == "spin":
            bound = re.fullmatch(r"([0-9]+) min ([0-9]+) max ([0-9]+)", value)
            if bound is None or int(bound[2]) != minimum or int(bound[3]) != maximum:
                raise UciGateError("Option spin bounds differ from the fixed contract")
            assert minimum is not None and maximum is not None
            _integer(bound[1], minimum, maximum)
            value = bound[1]
        elif kind == "check" and value not in ("true", "false"):
            raise UciGateError("Option check default must be a strict boolean")
        elif kind == "combo":
            variants = {
                "Backend": " var cpu var cuda var tensorrt",
                "Precision": " var fp32 var fp16",
                "FpuStrategyAtRoot": " var same var reduction var absolute",
            }[name]
            if not value.endswith(variants):
                raise UciGateError("Option combo variants differ from the fixed contract")
            value = value.removesuffix(variants)
            if value not in variants.split(" var ")[1:]:
                raise UciGateError("Combo default is not one of the explicit variants")
        if name in REAL_OPTIONS:
            try:
                number = Decimal(value)
            except InvalidOperation as error:
                raise UciGateError("Malformed finite real option default") from error
            if not number.is_finite():
                raise UciGateError("Nonfinite real option default")
        if name in expected:
            same = (
                Decimal(value) == Decimal(expected[name])
                if name in REAL_OPTIONS
                else value == expected[name]
            )
            if not same:
                raise UciGateError(f"Option {name} did not retain the requested transaction")
        result[name] = {"type": kind, "default": value}
    if set(result) != set(OPTION_TYPES):
        raise UciGateError("Missing required option declaration")
    return result


@dataclass(frozen=True)
class Case:
    identifier: str
    startup: str
    steps: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class Spec:
    cases: tuple[Case, ...]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise UciGateError("Duplicate golden JSON key")
        result[key] = value
    return result


def load_spec(root: Path) -> Spec:
    if set(path.name for path in root.iterdir()) != {"expectations.json", "README.md"}:
        raise UciGateError("Original golden directory inventory changed")
    for name, digest in (("expectations.json", GOLDEN_SHA256), ("README.md", README_SHA256)):
        path = root / name
        if not path.is_file() or path.stat().st_size > 128 * 1024 or _sha256(path) != digest:
            raise UciGateError("Original golden contract hash changed")
    value = json.loads(
        (root / "expectations.json").read_text(encoding="utf-8"), object_pairs_hook=_unique
    )
    if not isinstance(value, dict) or value.get("schema") != 1 or value.get("cpu_only") is not True:
        raise UciGateError("Malformed fixed golden schema")
    cases = []
    for row in value["cases"]:
        if not isinstance(row, dict) or set(row) != {"id", "startup", "steps"}:
            raise UciGateError("Malformed golden case")
        if row["startup"] not in ("configured", "bare", "failure") or not isinstance(
            row["steps"], list
        ):
            raise UciGateError("Malformed golden startup/steps")
        cases.append(Case(str(row["id"]), str(row["startup"]), tuple(row["steps"])))
    if len(cases) != 21 or len({case.identifier for case in cases}) != 21:
        raise UciGateError("Original golden 21-case inventory changed")
    return Spec(tuple(cases))


@dataclass(frozen=True)
class GateConfig:
    engine: Path
    weights: Path
    failure_weights: Path | None = None
    spec_dir: Path = Path("tests/data/uci/protocol-v1")
    timeout: float = 10
    case_timeout: float = 60

    def validate(self) -> None:
        _positive(self.timeout, "step deadline")
        _positive(self.case_timeout, "case deadline", 900)
        if self.case_timeout < self.timeout:
            raise UciGateError("Case deadline must include a complete step deadline")
        if not self.engine.is_file() or not self.weights.is_file():
            raise UciGateError("Explicit native engine and tiny CPU fixture files are required")


def _libraries(root: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            (
                path.resolve()
                for path in root.iterdir()
                if path.is_file()
                and (path.name.lower().endswith((".dll", ".dylib")) or ".so" in path.name)
            ),
            key=str,
        )
    )
    if len(paths) > 128 or not any("onnxruntime" in path.name.lower() for path in paths):
        raise UciGateError("Explicit app-local ONNX Runtime libraries are required")
    return paths


@dataclass(frozen=True)
class Artifacts:
    config: GateConfig
    hashes: dict[str, str]
    libraries: tuple[Path, ...]

    @classmethod
    def freeze(cls, config: GateConfig) -> Artifacts:
        config.validate()
        libraries = _libraries(config.engine.resolve().parent)
        models = [(config.weights, VALID_MODEL_SHA256)]
        if config.failure_weights is not None:
            models.append((config.failure_weights, FAILURE_MODEL_SHA256))
        for path, _expected in models:
            if not path.is_file() or path.stat().st_size > 128 * 1024:
                raise UciGateError("Only the exact tiny tracked CPU fixture models are permitted")
        paths = [
            config.engine,
            config.weights,
            *libraries,
            config.spec_dir / "expectations.json",
            config.spec_dir / "README.md",
            Path(__file__),
            Path(sys.executable),
        ]
        if config.failure_weights is not None:
            paths.append(config.failure_weights)
        if config.engine.stat().st_size > 256 * 1024 * 1024 or any(
            path.stat().st_size > 512 * 1024 * 1024 for path in libraries
        ):
            raise UciGateError("Native artifact bound exceeded")
        # Freeze each unique identity once, before validating/loading it. In
        # particular do not validate model A, then re-freeze a replacement B.
        hashes: dict[str, str] = {}
        for path in paths:
            name = str(path.resolve())
            if name not in hashes:
                hashes[name] = _sha256(path)
        for path, expected in models:
            if hashes[str(path.resolve())] != expected:
                raise UciGateError("Only the exact tiny tracked CPU fixture models are permitted")
        frozen = cls(config, hashes, libraries)
        load_spec(config.spec_dir)
        frozen.verify()
        return frozen

    def verify(self) -> None:
        if _libraries(self.config.engine.resolve().parent) != self.libraries:
            raise UciGateError("Native runtime library inventory changed")
        for name, digest in self.hashes.items():
            path = Path(name)
            if not path.is_file() or _sha256(path) != digest:
                raise UciGateError(f"Frozen artifact changed: {name}")
        load_spec(self.config.spec_dir)


def _command(config: GateConfig, case: Case) -> list[str]:
    command = [str(config.engine.resolve())]
    if case.startup == "bare":
        return command
    model = config.failure_weights if case.startup == "failure" else config.weights
    if model is None:
        raise UciGateError("Explicit failure fixture is required for fatal-inference coverage")
    return command + [
        "uci",
        "--weights",
        model.resolve().as_posix(),
        "--backend",
        "cpu",
        "--precision",
        "fp32",
        "--threads",
        "1",
        "--batch",
        "1",
        "--cpu-threads",
        "1",
        "--nncache-mb",
        "0",
        "--ram-limit-mb",
        "64",
        "--buckets",
        "1",
        "--deterministic",
    ]


class Driver:
    def __init__(self, process: UciProcess, config: GateConfig) -> None:
        self.process = process
        self.config = config
        self.deadline = time.monotonic() + config.case_timeout
        self.active = False
        self.previous_active = False
        self.infos: list[dict[str, object]] = []
        self.searches: list[dict[str, object]] = []
        self.errors: list[str] = []
        self.handshakes: list[dict[str, dict[str, object]]] = []
        self.last_command = ""

    def _line(self, deadline: float) -> str:
        remaining = min(deadline, self.deadline) - time.monotonic()
        if remaining <= 0:
            raise UciGateError("protocol step deadline exceeded")
        return self.process.line(timeout=remaining)

    def _info(self, line: str) -> None:
        if not self.active:
            raise UciGateError("Unsolicited structured info outside an active search")
        info = parse_info(line)
        if cast(int, info["time"]) > self.config.case_timeout * 1000:
            raise UciGateError("Reported search time exceeds the complete case deadline")
        self.infos.append(info)

    def _expect(self, target: str, *, error: str | None = None) -> None:
        deadline = time.monotonic() + self.process.timeout
        while True:
            line = self._line(deadline)
            if error is not None and line.startswith("info string error "):
                if error.casefold() not in line.casefold() or len(line) <= len(
                    "info string error "
                ):
                    raise UciGateError(f"Wrong error for {self.last_command!r}: {line}")
                self.errors.append(line)
                self.active = self.previous_active
                return
            if line == target and error is None:
                return
            if line.startswith("info depth "):
                self._info(line)
                continue
            raise UciGateError(f"Unexpected output while awaiting {target!r}: {line!r}")

    def _best(self, expected: dict[str, object]) -> None:
        deadline = time.monotonic() + self.process.timeout
        while True:
            line = self._line(deadline)
            if line.startswith("info depth "):
                self._info(line)
                continue
            move = parse_bestmove(line)
            break
        allowed = cast(list[str], expected["allowed"])
        terminal = expected["terminal"] is True
        if move not in allowed or (move == "0000") != terminal:
            raise UciGateError("Bestmove violates the exact legal/restricted golden roots")
        if not self.active or not self.infos:
            raise UciGateError("Bestmove is duplicated or lacks a completed info report")
        final: dict[int, dict[str, object]] = {}
        for info in self.infos:
            if ("wdl" in info) != (expected["wdl"] is True):
                raise UciGateError("WDL option did not control every structured info line")
            pv = cast(list[str], info["pv"])
            if pv and pv[0] not in allowed:
                raise UciGateError("A root PV escaped the requested searchmoves restriction")
            if not terminal and cast(int, info["nodes"]) < 1:
                raise UciGateError("Nonterminal info lacks a real evaluated root")
            maximum = expected["nodes_max"]
            if maximum is not None and cast(int, info["nodes"]) > cast(int, maximum):
                raise UciGateError("Reported nodes exceeded the requested limit")
            final[cast(int, info["multipv"])] = info
        count = cast(int, expected["multipv"])
        if set(final) != set(range(1, count + 1)):
            raise UciGateError("Incomplete or unexpected MultiPV index set")
        roots = [cast(list[str], final[index]["pv"]) for index in range(1, count + 1)]
        root_only = (
            final[1]["nodes"] == 1
            and final[1]["depth"] == 0
            and final[1]["seldepth"] == 0
            and not roots[0]
        )
        # The approved UCI design permits a real root evaluation without a
        # visited child after immediate stop or an already-expired tiny clock.
        # It reports depth zero; inventing a child PV would misstate evaluation.
        if not terminal and not root_only and (not roots[0] or roots[0][0] != move):
            raise UciGateError("Bestmove differs from the final principal line")
        nonempty = [pv[0] for pv in roots if pv]
        if len(nonempty) != len(set(nonempty)):
            raise UciGateError("MultiPV repeats a root continuation")
        self.searches.append(
            {
                "bestmove": move,
                "final_info": [final[i] for i in sorted(final)],
                "all_info": list(self.infos),
            }
        )
        self.active = False

    def _quiet(self, seconds: float) -> None:
        end = min(self.deadline, time.monotonic() + seconds)
        while time.monotonic() < end:
            self.process._check()
            try:
                line = self.process.lines.get(timeout=max(0, min(0.02, end - time.monotonic())))
            except queue.Empty:
                if self.process.stdout_done.is_set():
                    raise UciGateError(
                        "Premature EOF during required held-search interval"
                    ) from None
                continue
            self._info(line)

    def step(self, action: dict[str, object]) -> None:
        if time.monotonic() >= self.deadline:
            raise UciGateError("Complete case deadline exceeded")
        self.process.timeout = min(self.config.timeout, self.deadline - time.monotonic())
        if "send" in action or "oversize" in action:
            text = (
                "x" * (MAXIMUM_LINE_BYTES + 1)
                if "oversize" in action
                else cast(str, action["send"])
            )
            text = text.replace("{weights}", self.config.weights.resolve().as_posix())
            self.last_command = text
            self.previous_active = self.active
            if text.startswith("go ") and not self.active:
                self.active = True
                self.infos = []
            self.process.send(text, newline=action.get("newline", True) is True)
        elif "handshake" in action:
            lines: list[str] = []
            deadline = time.monotonic() + self.process.timeout
            while not lines or lines[-1] != "uciok":
                lines.append(self._line(deadline))
                if len(lines) > 64:
                    raise UciGateError("Oversized option handshake")
            expected = {
                key: value.replace("{weights}", self.config.weights.resolve().as_posix())
                for key, value in cast(dict[str, str], action["handshake"]).items()
            }
            self.handshakes.append(validate_handshake(lines, expected))
        elif "expect" in action:
            self._expect(cast(str, action["expect"]))
        elif "error" in action:
            self._expect("one explicit error", error=cast(str, action["error"]))
        elif "best" in action:
            self._best(cast(dict[str, object], action["best"]))
        elif "quiet" in action:
            self._quiet(cast(float, action["quiet"]))
        elif "eof" in action:
            self.process.close_stdin()
        elif "repeat_equal" in action:
            if len(self.searches) < 2:
                raise UciGateError("Missing complete deterministic repetitions")
            stable = []
            for search in self.searches[-2:]:
                rows = cast(list[dict[str, object]], search["final_info"])
                stable.append(
                    (
                        search["bestmove"],
                        [
                            {key: value for key, value in row.items() if key not in ("time", "nps")}
                            for row in rows
                        ],
                    )
                )
            if stable[0] != stable[1]:
                raise UciGateError(
                    "Deterministic complete search fields changed across repetitions"
                )
        elif "exit" in action:
            code = self.process.wait_exit()
            expected_code = action["exit"]
            if (expected_code == "nonzero" and not 0 < code < 128) or (
                expected_code != "nonzero" and code != expected_code
            ):
                raise UciGateError(
                    f"Unexpected native exit status {code}, expected {expected_code}"
                )
            while not self.process.lines.empty():
                self._info(self.process.lines.get_nowait())
            if self.process.stderr and expected_code != "nonzero":
                raise UciGateError("Unexpected native stderr in a successful protocol case")
        else:
            raise UciGateError("Unknown fixed golden action")


class _InterruptedCase(KeyboardInterrupt):
    def __init__(self, report: dict[str, object]) -> None:
        super().__init__()
        self.report = report


def run_case(config: GateConfig, case: Case) -> dict[str, object]:
    result: dict[str, object] = {
        "id": case.identifier,
        "passed": False,
        "completed": False,
        "steps_planned": len(case.steps),
        "steps_completed": 0,
    }
    process = UciProcess(
        _command(config, case), timeout=config.timeout, cwd=config.engine.resolve().parent
    )
    driver = Driver(process, config)
    interrupted = False
    try:
        with process:
            for index, action in enumerate(case.steps):
                driver.step(action)
                result["steps_completed"] = index + 1
            result["stdin_eof_before_exit"] = process.eof_requested.is_set()
            result["passed"] = True
    except KeyboardInterrupt:
        interrupted = True
        result["passed"] = False
        result["error"] = "Interrupted; incomplete protocol evidence"
    except Exception as error:
        result["passed"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    result["completed"] = not interrupted and result["steps_completed"] == len(case.steps)
    result["searches"] = driver.searches
    result["errors_observed"] = driver.errors
    result["handshakes"] = driver.handshakes
    if hasattr(process, "process"):
        result["process"] = process.evidence()
    if interrupted:
        raise _InterruptedCase(result)
    return result


def _publish(output: Path, report: dict[str, object]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = output
    if destination.exists():
        destination = output.with_name(output.stem + ".failed-" + uuid.uuid4().hex + ".json")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=output.name + ".",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            report["passed"] = False
            report["status"] = "failed"
            cast(list[str], report["errors"]).append(
                "Report destination was created during execution"
            )
            temporary.unlink()
            temporary = None
            return _publish(output, report)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_gate(config: GateConfig, output: Path) -> dict[str, object]:
    protected = [
        config.engine,
        config.weights,
        config.spec_dir / "expectations.json",
        config.spec_dir / "README.md",
        Path(__file__),
        Path(sys.executable),
    ]
    if config.failure_weights is not None:
        protected.append(config.failure_weights)
    if output.resolve() in {path.resolve() for path in protected}:
        raise UciGateError("Report destination cannot replace an input artifact")
    report: dict[str, object] = {
        "schema": 1,
        "gate": "mouse-uci-protocol-v1-tiny-cpu",
        "status": "failed",
        "passed": False,
        "backend": "cpu",
        "precision": "fp32",
        "python": sys.version,
        "coverage_complete": False,
        "hashes_unchanged": False,
        "planned_cases": 21,
        "completed_cases": 0,
        "attempted_cases": 0,
        "cases": [],
        "artifacts": {},
        "errors": [],
        "full_release_gate": False,
    }
    rows = cast(list[dict[str, object]], report["cases"])
    errors = cast(list[str], report["errors"])
    snapshot: Artifacts | None = None
    try:
        if output.exists():
            raise UciGateError("Existing report is immutable; choose a new output path")
        snapshot = Artifacts.freeze(config)
        if output.resolve() in {path.resolve() for path in snapshot.libraries}:
            raise UciGateError("Report destination cannot replace a runtime input artifact")
        report["artifacts"] = dict(snapshot.hashes)
        spec = load_spec(config.spec_dir)
        for case in spec.cases:
            if case.startup == "failure" and config.failure_weights is None:
                rows.append(
                    {
                        "id": case.identifier,
                        "passed": False,
                        "completed": False,
                        "not_run": True,
                        "error": "Explicit failure fixture absent; incomplete required coverage",
                    }
                )
                continue
            rows.append(run_case(config, case))
        report["coverage_complete"] = all(row.get("completed", True) for row in rows)
    except _InterruptedCase as error:
        rows.append(error.report)
        report["status"] = "interrupted"
        errors.append("Interrupted; incomplete protocol suite")
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        errors.append("Interrupted; incomplete protocol suite")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    if snapshot is not None:
        try:
            snapshot.verify()
            report["hashes_unchanged"] = True
        except (OSError, UciGateError) as error:
            errors.append(f"Artifact verification: {error}")
            for row in rows:
                row["observed_passed"] = row["passed"]
                row["passed"] = False
    report["attempted_cases"] = sum(row.get("not_run") is not True for row in rows)
    report["completed_cases"] = sum(row.get("completed", True) is True for row in rows)
    report["passed"] = bool(
        len(rows) == 21
        and report["coverage_complete"]
        and report["hashes_unchanged"]
        and not errors
        and all(row["passed"] for row in rows)
    )
    if report["passed"]:
        report["status"] = "passed"
    path = _publish(output, report)
    report["report_path"] = str(path.resolve())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--failure-weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-dir", type=Path, default=Path("tests/data/uci/protocol-v1"))
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--case-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if not sys.flags.isolated:
        print("UCI gate requires python -I", file=sys.stderr)
        return 2
    try:
        report = run_gate(
            GateConfig(
                args.engine,
                args.weights,
                args.failure_weights,
                args.spec_dir,
                args.timeout,
                args.case_timeout,
            ),
            args.output,
        )
    except (OSError, ValueError) as error:
        print(f"UCI gate error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "completed_cases", "planned_cases", "report_path")
            }
        )
    )
    if not report["passed"]:
        print(
            json.dumps(
                {
                    "errors": report["errors"],
                    "failed_cases": [
                        {
                            key: row.get(key)
                            for key in ("id", "error", "steps_completed", "steps_planned")
                        }
                        for row in cast(list[dict[str, object]], report["cases"])
                        if not row["passed"]
                    ],
                }
            ),
            file=sys.stderr,
        )
    return 130 if report["status"] == "interrupted" else 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
