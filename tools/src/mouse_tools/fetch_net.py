# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch the pinned Phase 1 network without publishing unverified bytes."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from email.message import Message
from http.client import HTTPException
from pathlib import Path
from typing import IO, Protocol, cast
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

T1_FILENAME = "t1-256x10-distilled-swa-2432500.pb.gz"
T1_URL = "https://storage.lczero.org/files/networks-contrib/" + T1_FILENAME
T1_SHA256 = "bc27a6cae8ad36f2b9a80a6ad9dabb0d6fda25b1e7f481a79bc359e14f563406"
T1_SIZE = 37_118_673
CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """The requested download or existing file could not be verified."""


class Readable(Protocol):
    def read(self, size: int = -1) -> bytes:
        """Read binary bytes from the transfer."""


class Transport(Protocol):
    def __call__(self, url: str, timeout: float) -> AbstractContextManager[Readable]:
        """Open a transfer; the context manager owns its lifetime."""


class _NoRedirect(HTTPRedirectHandler):
    # The CLI requests one exact official URL. A redirect must be reviewed rather
    # than quietly fetching a different host or downgrading HTTPS to HTTP.
    def redirect_request(
        self, req: Request, fp: IO[bytes], code: int, msg: str, headers: Message, newurl: str
    ) -> Request | None:
        raise DownloadError(f"Redirect refused for pinned URL {req.full_url}: {newurl}")


@contextmanager
def _https_transport(url: str, timeout: float) -> Iterator[Readable]:
    opener = build_opener(_NoRedirect())
    request = Request(url, headers={"User-Agent": "Mouse/0.0.0"})
    with opener.open(request, timeout=timeout) as response:
        yield cast(Readable, response)


def _validate_request(url: str, sha256: str, expected_size: int, timeout: float) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
        secure = (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and (port is None or port > 0)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except ValueError as error:
        raise DownloadError(f"Invalid HTTPS URL: {url}") from error
    if not secure:
        raise DownloadError(f"Download source must be an HTTPS URL without credentials: {url}")
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise DownloadError("Expected SHA-256 must contain exactly 64 lowercase hexadecimal digits")
    if expected_size <= 0:
        raise DownloadError("Expected file size must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise DownloadError("Download timeout must be finite and positive")


def _verify_existing(path: Path, sha256: str, expected_size: int) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise DownloadError(
            f"File size mismatch for {path}: expected {expected_size}, got {actual_size}; "
            "use --force to download a replacement"
        )
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != sha256:
        raise DownloadError(
            f"SHA-256 mismatch for {path}: expected {sha256}, got {actual_sha256}; "
            "use --force to download a replacement"
        )


def download_verified(
    destination: Path,
    *,
    url: str,
    sha256: str,
    expected_size: int,
    transport: Transport = _https_transport,
    force: bool = False,
    timeout: float = 30.0,
) -> Path:
    """Verify or atomically replace a file using one explicitly selected source.

    Concurrent invocations may each download a separate temporary file. Each
    replacement is independently verified, so readers see only a complete file.
    The caller must provide a trusted digest and size; the CLI pins both to T1.
    """
    _validate_request(url, sha256, expected_size, timeout)
    temporary_path: Path | None = None
    try:
        if destination.exists() and not force:
            _verify_existing(destination, sha256, expected_size)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        received = 0
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            with transport(url, timeout) as source:
                while chunk := source.read(CHUNK_SIZE):
                    received += len(chunk)
                    if received > expected_size:
                        raise DownloadError(
                            f"Download size exceeds expected {expected_size} bytes: {url}"
                        )
                    target.write(chunk)
                    digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
        if received != expected_size:
            raise DownloadError(
                f"Download size mismatch: expected {expected_size}, got {received} bytes: {url}"
            )
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != sha256:
            raise DownloadError(
                f"SHA-256 mismatch for {url}: expected {sha256}, got {actual_sha256}"
            )
        os.replace(temporary_path, destination)
        return destination
    except (OSError, URLError, HTTPException) as error:
        raise DownloadError(f"Failed to fetch or validate {destination}: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify the pinned Lc0 T1 network")
    parser.add_argument(
        "--destination", type=Path, default=Path("nets"), help="network directory (default: nets)"
    )
    parser.add_argument("--force", action="store_true", help="explicitly replace an existing file")
    arguments = parser.parse_args(argv)
    if not sys.flags.isolated:
        parser.error("downloaded files must be read with python -I tools/fetch_net.py")
    try:
        path = download_verified(
            arguments.destination / T1_FILENAME,
            url=T1_URL,
            sha256=T1_SHA256,
            expected_size=T1_SIZE,
            force=arguments.force,
        )
    except DownloadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Verified {path} SHA256 {T1_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
