"""Git-related tool functions for Aura.

This module contains tools for git operations like status, commit, push, and diff.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from git import GitCommandError, Repo
except ImportError:  # pragma: no cover
    GitCommandError = Exception  # type: ignore[misc,assignment]
    Repo = None  # type: ignore[assignment]

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


def git_commit(message: str = "") -> str:
    """Commit all changes with the given message.

    If message is not provided or is empty, a specialized local AI will automatically
    generate a conventional commit message based on the staged changes.

    Args:
        message: Commit message (optional). If omitted, auto-generates using local AI.

    Returns:
        Success message or error message
    """
    LOGGER.info("🔧 TOOL CALLED: git_commit(%s)", message or "(auto-generate)")

    # Stage all changes first
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

    # Auto-generate message if not provided
    if not message or not message.strip():
        LOGGER.info("No commit message provided; auto-generating via local AI...")

        # Get the staged diff
        diff_result = subprocess.run(
            ["git", "diff", "--staged"],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
        )

        if diff_result.returncode != 0:
            error = diff_result.stderr.strip() or "git diff failed"
            LOGGER.error("git diff --staged failed: %s", error)
            return f"Error getting staged changes: {error}"

        staged_diff = diff_result.stdout.strip()
        if not staged_diff:
            return "Error: No staged changes to commit"

        # Import here to avoid circular dependency
        from aura.tools.local_agent_tools import generate_commit_message

        message = generate_commit_message(staged_diff)

        # Check if generation failed
        if message.startswith("Error:"):
            return message

        LOGGER.info("Auto-generated commit message: %s", message[:100])

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


def git_blame(file_path: str, line_number: int) -> dict[str, Any]:
    """Return author and commit metadata for a specific file line."""
    LOGGER.info("?? TOOL CALLED: git_blame(%s:%s)", file_path, line_number)
    if line_number <= 0:
        return {"error": "Line number must be greater than zero."}

    repo = _load_repo()
    if isinstance(repo, dict):
        return repo

    target_path = _resolve_repo_path(repo, file_path)
    if target_path is None:
        return {"error": f"File '{file_path}' is outside the repository."}

    rel = os.fspath(target_path)
    rel = rel.replace("\\", "/")

    try:
        blame_data = repo.blame("HEAD", rel)
    except GitCommandError as exc:  # pragma: no cover
        LOGGER.error("git blame failed: %s", exc)
        return {"error": f"git blame failed: {exc}"}

    counter = 0
    for commit, lines in blame_data:
        for line in lines:
            counter += 1
            if counter == line_number:
                return {
                    "file": rel,
                    "line": line_number,
                    "author": commit.author.name,
                    "email": commit.author.email,
                    "commit": commit.hexsha,
                    "summary": commit.summary,
                    "committed_datetime": commit.committed_datetime.isoformat(),
                    "context": line.rstrip("\n"),
                }

    return {"error": f"Line {line_number} is beyond the end of {file_path}."}


def create_new_branch(branch_name: str, start_point: str = "HEAD") -> dict[str, Any]:
    """Create and check out a new git branch based on start_point."""
    LOGGER.info("?? TOOL CALLED: create_new_branch(%s)", branch_name)
    if not branch_name or branch_name.strip() == "":
        return {"success": False, "error": "Branch name cannot be empty."}

    repo = _load_repo()
    if isinstance(repo, dict):
        return repo

    branch_name = branch_name.strip()
    existing_names = {head.name for head in repo.branches}
    if branch_name in existing_names:
        return {"success": False, "error": f"Branch '{branch_name}' already exists."}

    try:
        new_branch = repo.create_head(branch_name, start_point)
        new_branch.checkout()
        return {
            "success": True,
            "branch": branch_name,
            "start_point": start_point,
        }
    except GitCommandError as exc:  # pragma: no cover
        LOGGER.error("Failed to create branch %s: %s", branch_name, exc)
        return {"success": False, "error": f"Failed to create branch: {exc}"}


def _load_repo() -> Repo | dict[str, Any]:
    if Repo is None:
        return {"error": "GitPython is not installed. Install it with: pip install GitPython"}
    try:
        return Repo(Path.cwd(), search_parent_directories=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load git repository: %s", exc)
        return {"error": f"Failed to open git repository: {exc}"}


def _resolve_repo_path(repo: Repo, file_path: str) -> Path | None:
    root = Path(repo.working_tree_dir or Path.cwd()).resolve()
    candidate = Path(file_path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve().relative_to(root)
    except ValueError:
        return None
