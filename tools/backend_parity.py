# SPDX-License-Identifier: GPL-3.0-or-later
"""Installed-package entry point; downloaded-file readers require python -I."""

from mouse_tools.backend_parity import main

if __name__ == "__main__":
    raise SystemExit(main())
