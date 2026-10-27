# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise urllib's actual request headers and redirects on loopback HTTP."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mouse_tools.fetch_net import DownloadError, Readable, _https_transport, download_verified

DATA = b"verified network contents\n"
SHA256 = "bcb26d48fb7d2b9c72d71494c9fe7131259f0925b6ac12dc80bf6f441581d456"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/network")
            self.end_headers()
            return
        if self.headers.get("User-Agent") != "Mouse/0.0.0":
            self.send_error(403, "An identified Mouse request is required")
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(DATA)))
        self.end_headers()
        self.wfile.write(DATA)

    def log_message(self, format: str, *args: object) -> None:
        # Test responses are asserted directly; HTTP access logs would hide failures.
        return


@pytest.fixture
def server_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
        assert not worker.is_alive()


def test_default_transport_identifies_mouse_to_download_host(server_url: str) -> None:
    with _https_transport(server_url + "/network", 5) as stream:
        assert stream.read() == DATA


def test_redirect_does_not_replace_verified_file(tmp_path: Path, server_url: str) -> None:
    @contextmanager
    def local_transport(url: str, timeout: float) -> Iterator[Readable]:
        with _https_transport(server_url + "/redirect", timeout) as stream:
            yield stream

    target = tmp_path / "network.pb.gz"
    target.write_bytes(DATA)
    with pytest.raises(DownloadError, match="Redirect refused"):
        download_verified(
            target,
            url="https://storage.lczero.org/files/networks-contrib/test.pb.gz",
            sha256=SHA256,
            expected_size=len(DATA),
            transport=local_transport,
            force=True,
        )
    assert target.read_bytes() == DATA
    assert list(tmp_path.iterdir()) == [target]
