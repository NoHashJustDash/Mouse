# SPDX-License-Identifier: GPL-3.0-or-later
"""Choose affected CI suites; unknown paths conservatively select every suite."""

from __future__ import annotations

import argparse
from pathlib import Path


PACKAGE_ONLY = {
    "tools/src/mouse_tools/package.py",
    "tools/src/mouse_tools/setup_net.py",
    "tools/setup_net.py",
    "tools/tests/test_package.py",
    "tools/tests/test_package_release.py",
}


def affected(paths: list[str]) -> dict[str, bool]:
    result = {"runtime": False, "lint": False, "python": False}
    for path in paths:
        if path in PACKAGE_ONLY:
            result["python"] = True
        elif path.startswith(("engine/", "tests/")) and path.endswith((".cpp", ".h")):
            result["runtime"] = result["lint"] = True
        elif path.startswith("docs/") and path.endswith(".md"):
            continue
        else:
            return dict.fromkeys(result, True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = args.paths.read_bytes().decode("utf-8").split("\0")
    result = affected([path for path in paths if path])
    with args.output.open("a", encoding="utf-8") as output:
        for name, enabled in result.items():
            output.write(f"{name}={str(enabled).lower()}\n")


if __name__ == "__main__":
    main()
