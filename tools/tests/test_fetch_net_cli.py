# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the installed script from isolated and non-isolated interpreters."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_requires_isolation_before_reading_files(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "fetch_net.py"
    result = subprocess.run(
        [sys.executable, str(script), "--destination", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "python -I" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_installed_cli_runs_in_isolated_mode() -> None:
    script = Path(__file__).resolve().parents[1] / "fetch_net.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "T1" in result.stdout
