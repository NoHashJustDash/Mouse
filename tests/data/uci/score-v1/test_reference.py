# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the score-evidence audit as an isolated subprocess, without Lc0/GPU."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class ReferenceTests(unittest.TestCase):
    def run_audit(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(ROOT / "reference.py"),
                "audit",
                "--fixtures",
                str(root),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def test_complete_pinned_fixture_set_audits_without_reference_binary(self) -> None:
        result = self.run_audit(ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"sessions": 72', result.stdout)
        self.assertIn('"cases": 28', result.stdout)

    def test_observation_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixtures"
            shutil.copytree(ROOT, root)
            path = root / "reference/case-mu_pos-a/stdout.txt"
            text = path.read_text(encoding="utf-8")
            self.assertIn("score cp 79", text)
            path.write_text(
                text.replace("score cp 79", "score cp 78"), encoding="utf-8"
            )
            result = self.run_audit(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Original evidence changed", result.stderr)

    def test_shortened_evidence_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixtures"
            shutil.copytree(ROOT, root)
            shutil.rmtree(root / "reference/case-mu_pos-b")
            result = self.run_audit(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("file set changed", result.stderr)

    def test_cpp_golden_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fixtures"
            shutil.copytree(ROOT, root)
            path = root / "cases.inc"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace("563, 0, 437", "562, 0, 438"), encoding="utf-8"
            )
            result = self.run_audit(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("C++ golden table differs", result.stderr)

    def test_unpinned_executable_is_rejected_before_output_or_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "untrusted.exe"
            executable.write_bytes(b"not a trusted executable")
            output = root / "output"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(ROOT / "reference.py"),
                    "regenerate",
                    "--lc0",
                    str(executable),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("executable SHA256 mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_regeneration_cannot_write_inside_pinned_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = root / "fixtures"
            shutil.copytree(ROOT, fixtures)
            executable = root / "untrusted.exe"
            executable.write_bytes(b"not invoked")
            output = fixtures / "reference/new-output"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(ROOT / "reference.py"),
                    "regenerate",
                    "--fixtures",
                    str(fixtures),
                    "--lc0",
                    str(executable),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("outside the pinned fixtures", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
