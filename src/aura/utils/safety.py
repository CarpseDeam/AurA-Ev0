"""Safety utilities to prevent self-modification."""

from __future__ import annotations

from pathlib import Path


def is_safe_working_directory(working_dir: str, app_source_dir: str) -> tuple[bool, str]:
    """Check if working directory is safe (not Aura's source)."""
    work_path = Path(working_dir).resolve()
    app_path = Path(app_source_dir).resolve()

    try:
        work_path.relative_to(app_path)
        return False, "Cannot use Aura's source directory as working directory"
    except ValueError:
        pass

    try:
        app_path.relative_to(work_path)
        return False, "Cannot use parent of Aura's source directory"
    except ValueError:
        pass

    return True, ""

