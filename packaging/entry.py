"""Frozen-application entry point.

PyInstaller executes its entry script as a top-level ``__main__`` module, not as
``dosubs.app``, so the relative imports inside :mod:`dosubs.app` fail there. This
launcher imports the package properly, which is what a frozen build needs; the
development entry point stays ``python -m dosubs.app``.
"""

from __future__ import annotations

import sys

from dosubs.app import main

if __name__ == "__main__":
    sys.exit(main())
