# SPDX-License-Identifier: GPL-3.0-or-later
"""Download the pinned network and convert verified bytes to FP16 ONNX."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import fetch_net
from .convert import cli


def _isolated() -> bool:
    return bool(sys.flags.isolated)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("nets"))
    args = parser.parse_args(argv)
    if not _isolated():
        parser.error("Run with python -I -m mouse_tools.setup_net")
    directory = args.directory
    status = fetch_net.main(["--destination", str(directory)])
    if status:
        return status
    return cli.main(
        [
            str(directory / fetch_net.T1_FILENAME),
            str(directory / "t1-fp16.onnx"),
            "--sha256",
            fetch_net.T1_SHA256,
            "--precision",
            "fp16",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
