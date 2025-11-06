"""File system-related tool functions for Aura.

This module contains tools for reading, listing, and searching files.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def read_project_file(path: str) -> str:
    """Return the contents of a project file.

    Args:
        path: Path to file (relative or absolute)

    Returns:
        File contents as string, or error message
    """
    LOGGER.info("🔧 TOOL CALLED: read_project_file(%s)", path)
    try:
        target = Path(path)
        if not target.is_absolute():
            target = Path.cwd() / target
        if not target.exists():
            return f"Error: file '{path}' does not exist."
        return target.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to read project file %s: %s", path, exc)
        return f"Error reading '{path}': {exc}"


def list_project_files(directory: str = ".", extension: str = ".py") -> list[str]:
    """List project files matching the given extension.

    Args:
        directory: Directory to search in (default: ".")
        extension: File extension to filter by (default: ".py")

    Returns:
        Sorted list of relative file paths
    """
    LOGGER.info("🔧 TOOL CALLED: list_project_files(%s)", directory)
    try:
        base = Path(directory)
        if not base.is_absolute():
            base = Path.cwd() / base
        if not base.exists():
            return []
        suffix = extension if extension.startswith(".") else f".{extension}"
        files = [
            _relative_to_cwd(path) for path in base.rglob(f"*{suffix}") if path.is_file()
        ]
        return sorted(files)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "Failed to list project files in %s with extension %s: %s",
            directory,
            extension,
            exc,
        )
        return []


def search_in_files(
    pattern: str,
    directory: str = ".",
    file_extension: str = ".py",
) -> dict[str, object]:
    """Search the codebase for a pattern and return matching lines.

    Args:
        pattern: The search term or regex pattern
        directory: Directory to search in (default: ".")
        file_extension: File extension to filter by (default: ".py")

    Returns:
        Dictionary with "matches" key containing list of matches.
        Each match has: file, line_number, content
    """
    LOGGER.info("🔧 TOOL CALLED: search_in_files(%s)", pattern)
    try:
        base = Path(directory)
        if not base.is_absolute():
            base = Path.cwd() / base

        if not base.exists():
            LOGGER.error("Directory does not exist: %s", directory)
            return {"matches": []}

        suffix = file_extension if file_extension.startswith(".") else f".{file_extension}"
        matches = []

        for file_path in base.rglob(f"*{suffix}"):
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                for line_num, line in enumerate(content.split("\n"), start=1):
                    if pattern.lower() in line.lower():
                        matches.append(
                            {
                                "file": _relative_to_cwd(file_path),
                                "line_number": line_num,
                                "content": line.strip(),
                            }
                        )

                        if len(matches) >= 50:
                            LOGGER.info("Search hit 50 match limit")
                            return {"matches": matches}

            except (UnicodeDecodeError, PermissionError) as exc:
                LOGGER.debug("Skipping file %s: %s", file_path, exc)
                continue

        LOGGER.info("Search found %d matches for pattern: %s", len(matches), pattern)
        return {"matches": matches}

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to search files: %s", exc)
        return {"matches": []}


def read_multiple_files(file_paths: list[str]) -> dict[str, str]:
    """Read multiple project files at once.

    Args:
        file_paths: List of file paths to read

    Returns:
        Dictionary mapping file paths to their contents
        Example: {"file1.py": "content...", "file2.py": "content..."}
    """
    LOGGER.info("🔧 TOOL CALLED: read_multiple_files(%s)", file_paths)
    if not file_paths:
        return {}

    results = {}
    for path in file_paths:
        try:
            target = Path(path)
            if not target.is_absolute():
                target = Path.cwd() / target

            if not target.exists():
                results[path] = f"Error: file '{path}' does not exist."
                continue

            if not target.is_file():
                results[path] = f"Error: '{path}' is not a file."
                continue

            content = target.read_text(encoding="utf-8")
            results[path] = content

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read file %s: %s", path, exc)
            results[path] = f"Error reading '{path}': {exc}"

    LOGGER.info(
        "Read %d files: %d successful",
        len(file_paths),
        sum(1 for v in results.values() if not v.startswith("Error")),
    )
    return results


def _relative_to_cwd(path: Path) -> str:
    """Return a path relative to the current working directory when possible.

    Args:
        path: Path to make relative

    Returns:
        Relative path string, or absolute path if not relative to cwd
    """
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)
