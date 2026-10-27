# SPDX-License-Identifier: GPL-3.0-or-later
"""Entry script for the normally installed Mouse tools package."""

from mouse_tools.convert.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
