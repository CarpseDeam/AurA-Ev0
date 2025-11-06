"""Git-related tool functions for Aura.

This module contains tools for git operations like status, commit, push, and diff.
"""

from __future__ import annotations

import logging
import os
import subprocess

LOGGER = logging.getLogger(__name__)


def get_git_status() -> str:
    """Return the short git status for the current repository.

    Returns:
        Git status output or "clean" if no changes, or error message
    """
    LOGGER.info("🔧 TOOL CALLED: get_git_status")
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "git status failed"
        LOGGER.error("git status failed: %s", error)
        return f"Error: {error}"
    return result.stdout.strip() or "clean"


def git_commit(message: str) -> str:
    """Commit all changes with the given message.

    Args:
        message: Commit message

    Returns:
        Success message or error message
    """
    LOGGER.info("🔧 TOOL CALLED: git_commit(%s)", message)
    if not message or not message.strip():
        return "Error: commit message cannot be empty"

    add_result = subprocess.run(
        ["git", "add", "."],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        error = add_result.stderr.strip() or "git add failed"
        LOGGER.error("git add failed: %s", error)
        return f"Error staging files: {error}"

    commit_result = subprocess.run(
        ["git", "commit", "-m", message.strip()],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        error = (
            commit_result.stderr.strip()
            or commit_result.stdout.strip()
            or "git commit failed"
        )
        LOGGER.error("git commit failed: %s", error)
        return f"Error committing: {error}"

    output = commit_result.stdout.strip()
    LOGGER.info("Committed successfully: %s", message)
    return f"✅ Committed successfully: {message}\n{output}"


def git_push(remote: str = "origin", branch: str = "main") -> str:
    """Push commits to the remote repository.

    Args:
        remote: Remote name (default: "origin")
        branch: Branch name (default: "main")

    Returns:
        Success message or error message
    """
    LOGGER.info("🔧 TOOL CALLED: git_push(%s/%s)", remote, branch)
    result = subprocess.run(
        ["git", "push", remote, branch],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "git push failed"
        LOGGER.error("git push failed: %s", error)
        return f"Error pushing to {remote}/{branch}: {error}"

    output = result.stdout.strip()
    LOGGER.info("Pushed successfully to %s/%s", remote, branch)
    return f"✅ Pushed successfully to {remote}/{branch}\n{output}"


def git_diff(file_path: str = "", staged: bool = False) -> str:
    """Show git diff for changes in the repository.

    Args:
        file_path: Optional specific file to show diff for
        staged: If True, show staged changes; otherwise show unstaged (default: False)

    Returns:
        String containing the diff output, or empty string if no changes
    """
    LOGGER.info("🔧 TOOL CALLED: git_diff(%s)", file_path)
    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if file_path:
            cmd.append(file_path)

        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            error = result.stderr.strip() or "git diff failed"
            LOGGER.error("git diff failed: %s", error)
            return f"Error: {error}"

        output = result.stdout.strip()
        if not output:
            return ""

        LOGGER.info("Git diff retrieved: %d characters", len(output))
        return output

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to get git diff: %s", exc)
        return f"Error getting diff: {exc}"
