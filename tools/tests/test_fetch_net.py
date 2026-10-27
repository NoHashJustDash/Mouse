# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for verified downloads; all streams and files are real."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import pytest

from mouse_tools.fetch_net import DownloadError, download_verified

DATA = b"verified network contents\n"
SHA256 = "bcb26d48fb7d2b9c72d71494c9fe7131259f0925b6ac12dc80bf6f441581d456"
URL = "https://storage.lczero.org/files/networks-contrib/test.pb.gz"


class StreamTransport:
    """Supply a real binary stream without access to the public network."""

    def __init__(self, data: bytes = DATA) -> None:
        self.data = data

    @contextmanager
    def __call__(self, url: str, timeout: float) -> Iterator[BinaryIO]:
        assert url == URL
        assert timeout > 0
        with io.BytesIO(self.data) as stream:
            yield stream


class BrokenStream(io.BytesIO):
    def read(self, size: int | None = -1) -> bytes:
        if self.tell() > 0:
            raise OSError("connection interrupted")
        return super().read(5 if size is None or size < 0 else min(size, 5))


@contextmanager
def interrupted_transport(url: str, timeout: float) -> Iterator[BinaryIO]:
    assert url == URL
    assert timeout > 0
    with BrokenStream(DATA) as stream:
        yield stream


def test_verified_download_publishes_complete_file(tmp_path: Path) -> None:
    target = tmp_path / "nets" / "network.pb.gz"
    result = download_verified(
        target, url=URL, sha256=SHA256, expected_size=len(DATA), transport=StreamTransport()
    )
    assert result == target
    assert target.read_bytes() == DATA
    assert list(target.parent.iterdir()) == [target]


def test_existing_valid_file_does_not_access_network(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    target.write_bytes(DATA)
    result = download_verified(
        target, url=URL, sha256=SHA256, expected_size=len(DATA), transport=interrupted_transport
    )
    assert result == target
    assert target.read_bytes() == DATA


def test_existing_corrupt_file_fails_without_replacement(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    target.write_bytes(b"corruption")
    with pytest.raises(DownloadError, match="SHA-256|size"):
        download_verified(
            target, url=URL, sha256=SHA256, expected_size=len(DATA), transport=StreamTransport()
        )
    assert target.read_bytes() == b"corruption"


def test_wrong_hash_is_not_published(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    with pytest.raises(DownloadError, match="SHA-256"):
        download_verified(
            target, url=URL, sha256="0" * 64, expected_size=len(DATA), transport=StreamTransport()
        )
    assert list(tmp_path.iterdir()) == []


def test_wrong_size_is_not_published(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    with pytest.raises(DownloadError, match="size"):
        download_verified(
            target, url=URL, sha256=SHA256, expected_size=len(DATA) + 1, transport=StreamTransport()
        )
    assert list(tmp_path.iterdir()) == []


def test_interrupted_refresh_preserves_existing_verified_file(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    target.write_bytes(DATA)
    with pytest.raises(DownloadError, match="interrupted"):
        download_verified(
            target,
            url=URL,
            sha256=SHA256,
            expected_size=len(DATA),
            transport=interrupted_transport,
            force=True,
        )
    assert target.read_bytes() == DATA
    assert list(tmp_path.iterdir()) == [target]


def test_bad_refresh_preserves_existing_verified_file(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    target.write_bytes(DATA)
    with pytest.raises(DownloadError, match="SHA-256"):
        download_verified(
            target,
            url=URL,
            sha256=SHA256,
            expected_size=len(DATA),
            transport=StreamTransport(b"x" * len(DATA)),
            force=True,
        )
    assert target.read_bytes() == DATA
    assert list(tmp_path.iterdir()) == [target]


def test_force_replaces_corruption_only_after_verification(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    target.write_bytes(b"corruption")
    download_verified(
        target,
        url=URL,
        sha256=SHA256,
        expected_size=len(DATA),
        transport=StreamTransport(),
        force=True,
    )
    assert target.read_bytes() == DATA


@pytest.mark.parametrize(
    "url", ["http://example.com/net", "file:///tmp/net", "https://u:p@example.com/net"]
)
def test_insecure_source_is_rejected(tmp_path: Path, url: str) -> None:
    with pytest.raises(DownloadError, match="HTTPS"):
        download_verified(
            tmp_path / "network",
            url=url,
            sha256=SHA256,
            expected_size=len(DATA),
            transport=StreamTransport(),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("sha256", ["", "z" * 64, "0" * 63])
def test_invalid_hash_is_rejected_before_io(tmp_path: Path, sha256: str) -> None:
    with pytest.raises(DownloadError, match="SHA-256"):
        download_verified(
            tmp_path / "network",
            url=URL,
            sha256=sha256,
            expected_size=len(DATA),
            transport=StreamTransport(),
        )
    assert list(tmp_path.iterdir()) == []


def test_fixture_digest_is_independent() -> None:
    assert hashlib.sha256(DATA).hexdigest() == SHA256


def test_existing_same_size_corruption_fails_sha256(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    target.write_bytes(b"x" * len(DATA))
    with pytest.raises(DownloadError, match="SHA-256"):
        download_verified(
            target, url=URL, sha256=SHA256, expected_size=len(DATA), transport=StreamTransport()
        )
    assert target.read_bytes() == b"x" * len(DATA)


def test_oversized_transfer_is_not_published(tmp_path: Path) -> None:
    target = tmp_path / "network.pb.gz"
    with pytest.raises(DownloadError, match="size"):
        download_verified(
            target,
            url=URL,
            sha256=SHA256,
            expected_size=len(DATA),
            transport=StreamTransport(DATA + b"extra"),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("size", [0, -1])
def test_nonpositive_size_is_rejected_before_io(tmp_path: Path, size: int) -> None:
    with pytest.raises(DownloadError, match="size"):
        download_verified(
            tmp_path / "network",
            url=URL,
            sha256=SHA256,
            expected_size=size,
            transport=StreamTransport(),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_io(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(DownloadError, match="timeout"):
        download_verified(
            tmp_path / "network",
            url=URL,
            sha256=SHA256,
            expected_size=len(DATA),
            transport=StreamTransport(),
            timeout=timeout,
        )
    assert list(tmp_path.iterdir()) == []


def test_invalid_https_port_is_rejected_before_io(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="HTTPS"):
        download_verified(
            tmp_path / "network",
            url="https://storage.lczero.org:invalid/net",
            sha256=SHA256,
            expected_size=len(DATA),
            transport=StreamTransport(),
        )
    assert list(tmp_path.iterdir()) == []
