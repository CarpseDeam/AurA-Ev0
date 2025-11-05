"""Utilities for scanning project directories."""

from __future__ import annotations

import os
from typing import Any


def scan_directory(path: str, max_depth: int = 2) -> dict[str, Any]:
    """Return a simple structure describing files and directories."""
    if not path:
        raise ValueError("Path must be provided.")
    root = os.path.abspath(path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory does not exist: {root}")
    files: list[str] = []
    directories: list[str] = []
    for current_path, dirnames, filenames in os.walk(root):
        depth = current_path[len(root) :].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
        for dirname in dirnames:
            directories.append(os.path.relpath(os.path.join(current_path, dirname), root))
        for filename in filenames:
            files.append(os.path.relpath(os.path.join(current_path, filename), root))
    return {"files": sorted(files), "directories": sorted(directories)}
