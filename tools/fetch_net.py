# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility entry point; install tools/ and run with python -I."""

from mouse_tools.fetch_net import main

if __name__ == "__main__":
    raise SystemExit(main())
