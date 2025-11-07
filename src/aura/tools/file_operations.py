"""File operation tools for direct file manipulation.

This module provides tools for creating, modifying, and deleting files,
enabling direct code manipulation capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def create_file(path: str, content: str) -> str:
    """Create a new file with the provided content.

    Args:
        path: Path to file (relative or absolute)
        content: File contents to write

    Returns:
        Success or error message
    """
    LOGGER.info("🔧 TOOL CALLED: create_file(%s)", path)
    try:
        target = Path(path)
        if not target.is_absolute():
            target = Path.cwd() / target

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

        size = len(content.encode("utf-8"))
        LOGGER.info("✓ Created file: %s (%d bytes)", path, size)
        return f"Successfully created '{path}' ({size} bytes)"
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to create file %s: %s", path, exc)
        return f"Error creating '{path}': {exc}"


def modify_file(path: str, old_content: str, new_content: str) -> str:
    """Modify an existing file by replacing old content with new content.

    Args:
        path: Path to file (relative or absolute)
        old_content: Content to find and replace
        new_content: Content to replace with

    Returns:
        Success or error message
    """
    LOGGER.info("🔧 TOOL CALLED: modify_file(%s)", path)
    try:
        target = Path(path)
        if not target.is_absolute():
            target = Path.cwd() / target

        if not target.exists():
            return f"Error: file '{path}' does not exist."

        current = target.read_text(encoding="utf-8")
        if old_content not in current:
            return f"Error: old_content not found in '{path}'."

        updated = current.replace(old_content, new_content)
        target.write_text(updated, encoding="utf-8")

        LOGGER.info("✓ Modified file: %s", path)
        return f"Successfully modified '{path}'"
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to modify file %s: %s", path, exc)
        return f"Error modifying '{path}': {exc}"


def delete_file(path: str) -> str:
    """Delete an existing file.

    Args:
        path: Path to file (relative or absolute)

    Returns:
        Success or error message
    """
    LOGGER.info("🔧 TOOL CALLED: delete_file(%s)", path)
    try:
        target = Path(path)
        if not target.is_absolute():
            target = Path.cwd() / target

        if not target.exists():
            return f"Error: file '{path}' does not exist."

        target.unlink()
        LOGGER.info("✓ Deleted file: %s", path)
        return f"Successfully deleted '{path}'"
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to delete file %s: %s", path, exc)
        return f"Error deleting '{path}': {exc}"
