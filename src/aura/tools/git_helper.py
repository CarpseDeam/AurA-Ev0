"""Git helper utilities used by Aura orchestration."""

from __future__ import annotations

import logging
import subprocess
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


class GitHelper:
    """Wraps common git operations with error handling."""

    def __init__(self, working_dir: str) -> None:
        if not working_dir:
            raise ValueError("Working directory must be provided.")
        self._working_dir = working_dir

    def commit(self, message: str, files: Optional[List[str]] = None) -> bool:
        """Stage the provided files (or everything) and create a commit."""
        if not message:
            LOGGER.error("Cannot commit without a message.")
            return False
        targets = self._normalize_files(files)
        if not self._run(["git", "add", *targets]):
            return False
        return self._commit(message)

    def push(self, remote: str = "origin", branch: str = "main") -> bool:
        """Push local commits to the specified remote and branch."""
        return self._run(["git", "push", remote, branch])

    def get_status(self) -> str:
        """Return a short status summary for display."""
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self._working_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            LOGGER.error("Failed to get git status: %s", result.stderr.strip())
            return "git status failed"
        return result.stdout.strip()

    def _commit(self, message: str) -> bool:
        """Execute commit and handle common failure cases."""
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self._working_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "nothing to commit" in stderr.lower():
                LOGGER.info("Git commit skipped: %s", stderr)
            else:
                LOGGER.error("Git commit failed: %s", stderr or result.stdout.strip())
            return False
        return True

    def _normalize_files(self, files: Optional[List[str]]) -> List[str]:
        if not files:
            return ["."]
        cleaned = []
        for entry in files:
            if not entry:
                continue
            candidate = entry.strip()
            if candidate.endswith(" (updated)"):
                candidate = candidate[: -len(" (updated)")]
            cleaned.append(candidate)
        return cleaned or ["."]

    def _run(self, command: List[str]) -> bool:
        """Run a git command and log on failure."""
        result = subprocess.run(
            command,
            cwd=self._working_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            LOGGER.error(
                "Git command failed (%s): %s",
                " ".join(command),
                result.stderr.strip() or result.stdout.strip(),
            )
            return False
        return True
