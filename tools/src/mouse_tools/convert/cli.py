# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated command-line boundary for downloaded network input."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    if not sys.flags.isolated:
        print("network conversion requires python -I -m mouse_tools.convert", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(
        description="Convert supported Lc0 T1 attention weights to ONNX"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", help="required compressed-source SHA256, when supplied")
    parser.add_argument(
        "--overwrite", action="store_true", help="atomically replace an existing output"
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="graph compute precision; inputs and outputs remain float32 (default: fp32)",
    )
    arguments = parser.parse_args(argv)
    from .export import convert_network
    from .wire import ConvertError

    try:
        report = convert_network(
            arguments.source,
            arguments.output,
            expected_sha256=arguments.sha256,
            overwrite=arguments.overwrite,
            precision=arguments.precision,
        )
    except ConvertError as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), indent=2))
    return 0
