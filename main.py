"""Convenience launcher for Aura."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add the src directory to sys.path for local execution."""
    src_path = Path(__file__).resolve().parent / "src"
    src_str = str(src_path)
    if src_path.is_dir() and src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_path()

from aura.main import main  # noqa: E402  (import after path setup)


if __name__ == "__main__":
    main()
