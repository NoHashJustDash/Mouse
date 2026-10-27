# SPDX-License-Identifier: GPL-3.0-or-later
"""The reference generator is a stdlib CLI usable with isolated Python."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_perft_reference_cli_help() -> None:
    script = Path(__file__).parents[1] / "src" / "mouse_tools" / "perft_reference.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "Stockfish" in result.stdout


def test_perft_reference_cli_requires_isolation(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "src" / "mouse_tools" / "perft_reference.py"
    result = subprocess.run(
        [sys.executable, str(script), "generate", "--engine", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert "python -I" in result.stderr


def test_generate_rejects_resolved_destination_collision_before_opening_oracle(
    tmp_path: Path,
) -> None:
    script = Path(__file__).parents[1] / "src" / "mouse_tools" / "perft_reference.py"
    destination = tmp_path / "same-output"
    destination.write_bytes(b"preserve existing data")
    alias = tmp_path / "unused" / ".." / destination.name
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "generate",
            "--engine",
            str(tmp_path / "missing"),
            "--suite",
            str(destination),
            "--provenance",
            str(alias),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert "destinations must differ" in result.stderr
    assert "Traceback" not in result.stderr
    assert destination.read_bytes() == b"preserve existing data"
