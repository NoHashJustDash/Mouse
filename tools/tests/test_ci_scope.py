# SPDX-License-Identifier: GPL-3.0-or-later
"""Routing never suppresses runtime suites for a runtime change."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["tools/src/mouse_tools/package.py"], (False, False, True)),
        (["engine/search/search.cpp"], (True, True, False)),
        (["engine/search/search.cpp", "tools/src/mouse_tools/package.py"], (True, True, True)),
        (["CMakeLists.txt"], (True, True, True)),
        ([".github/workflows/ci.yml"], (True, True, True)),
        (["tests/data/new.json"], (True, True, True)),
        (["README.md"], (True, True, True)),
        (["docs/notes.md"], (False, False, False)),
        (["unknown"], (True, True, True)),
    ],
)
def test_affected_suites(paths: list[str], expected: tuple[bool, bool, bool]) -> None:
    script = Path(__file__).resolve().parents[2] / ".github/scripts/ci_scope.py"
    spec = importlib.util.spec_from_file_location("ci_scope", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.affected(paths)
    assert (result["runtime"], result["lint"], result["python"]) == expected
