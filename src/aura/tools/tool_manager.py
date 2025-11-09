"""Sandboxed access layer for Aura's filesystem tools."""

from __future__ import annotations

import ast
import fnmatch
import json
import logging
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

try:  # Optional dependency; fallback matcher used if unavailable.
    from pathspec import PathSpec
except ImportError:  # pragma: no cover - pathspec is optional at runtime.
    PathSpec = None  # type: ignore[assignment]

try:
    from git import GitCommandError, Repo
except ImportError:  # pragma: no cover
    GitCommandError = Exception  # type: ignore[misc,assignment]
    Repo = None  # type: ignore[assignment]

from aura.utils.file_filter import load_gitignore_patterns

LOGGER = logging.getLogger(__name__)

STDLIB_MODULES = getattr(
    sys,
    "stdlib_module_names",
    {
        "os",
        "sys",
        "ast",
        "logging",
        "pathlib",
        "typing",
        "json",
        "datetime",
        "collections",
        "functools",
        "itertools",
        "re",
        "math",
        "random",
        "time",
        "io",
        "subprocess",
        "threading",
        "multiprocessing",
        "unittest",
        "pytest",
    },
)


class ToolManager:
    """Provide workspace-scoped file system utilities."""

    def __init__(self, workspace_dir: str) -> None:
        resolved = Path(workspace_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
        self.workspace_dir = resolved
        self._gitignore_mtime: float | None = None
        self._gitignore_patterns: list[str] = []
        self._gitignore_spec: PathSpec | None = None
        self._gitignore_fallback_matchers: list[tuple[str, bool]] = []
        self._gitignore_warning_logged = False
        self._ensure_gitignore_state()
        LOGGER.info("ToolManager workspace set to %s", self.workspace_dir)

    def update_workspace(self, workspace_dir: str) -> None:
        """Re-point the ToolManager at a new workspace directory."""
        resolved = Path(workspace_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
        if resolved == self.workspace_dir:
            LOGGER.debug("ToolManager workspace already set to %s; skipping update", resolved)
            return
        self.workspace_dir = resolved
        self._reset_gitignore_cache()
        LOGGER.info("ToolManager workspace updated to %s", self.workspace_dir)

    # ------------------------------------------------------------------ #
    # File operation helpers
    # ------------------------------------------------------------------ #
    def create_file(self, path: str, content: str) -> str:
        """Create a file within the workspace."""
        LOGGER.info("🔧 TOOL CALLED: create_file(%s)", path)
        try:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            size = len(content.encode("utf-8"))
            LOGGER.info("✅ Created file: %s (%d bytes)", target, size)
            return f"Successfully created '{path}' ({size} bytes)"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to create file %s: %s", path, exc)
            return f"Error creating '{path}': {exc}"

    def modify_file(self, path: str, old_content: str, new_content: str) -> str:
        """Replace content within a workspace file."""
        LOGGER.info("🔧 TOOL CALLED: modify_file(%s)", path)
        try:
            target = self._resolve_path(path)

            if not target.exists():
                return f"Error: file '{path}' does not exist."

            current = target.read_text(encoding="utf-8")
            if old_content not in current:
                return f"Error: old_content not found in '{path}'."
            updated = current.replace(old_content, new_content)
            target.write_text(updated, encoding="utf-8")
            LOGGER.info("✅ Modified file: %s", target)
            return f"Successfully modified '{path}'"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to modify file %s: %s", path, exc)
            return f"Error modifying '{path}': {exc}"

    def replace_file_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
    ) -> str:
        """Replace a block of lines using explicit line numbers."""
        LOGGER.info(
            "🔧 TOOL CALLED: replace_file_lines(%s, start=%s, end=%s)",
            path,
            start_line,
            end_line,
        )
        try:
            start = int(start_line)
            end = int(end_line)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("replace_file_lines invalid line numbers: %s", exc)
            return "Error: start_line and end_line must be integers."

        if start < 1 or end < start:
            return "Error: start_line must be >= 1 and end_line must be >= start_line."

        try:
            target = self._resolve_path(path)
            if not target.exists():
                return f"Error: file '{path}' does not exist."

            contents = target.read_text(encoding="utf-8")
            lines = contents.splitlines(keepends=True)
            total_lines = len(lines)
            if end > total_lines:
                return (
                    f"Error: file '{path}' has only {total_lines} "
                    f"line(s); cannot replace through line {end}."
                )

            start_index = start - 1
            replaced_block = "".join(lines[start_index : end])
            before = "".join(lines[:start_index])
            after = "".join(lines[end:])

            replacement = new_content or ""
            updated_contents = before + replacement + after
            target.write_text(updated_contents, encoding="utf-8")

            replaced_lines = end - start + 1
            message = (
                f"Replaced lines {start}-{end} ({replaced_lines} line(s)) in '{path}'."
            )
            LOGGER.info(message)
            if not replacement.endswith("\n") and replacement:
                LOGGER.debug(
                    "replace_file_lines inserted content without trailing newline for %s",
                    path,
                )
            LOGGER.debug(
                "replace_file_lines replaced block:\n%s\nwith:\n%s",
                replaced_block,
                replacement,
            )
            return message
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "Failed to replace lines %s-%s in %s: %s", start_line, end_line, path, exc
            )
            return f"Error replacing lines {start_line}-{end_line} in '{path}': {exc}"

    def delete_file(self, path: str) -> str:
        """Delete a workspace file."""
        LOGGER.info("🔧 TOOL CALLED: delete_file(%s)", path)
        try:
            target = self._resolve_path(path)
            if not target.exists():
                return f"Error: file '{path}' does not exist."
            target.unlink()
            LOGGER.info("✅ Deleted file: %s", target)
            return f"Successfully deleted '{path}'"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to delete file %s: %s", path, exc)
            return f"Error deleting '{path}': {exc}"

    # ------------------------------------------------------------------ #
    # File system helpers
    # ------------------------------------------------------------------ #
    def read_project_file(self, path: str) -> str:
        """Return file contents if the target resides inside the workspace."""
        LOGGER.info("🔧 TOOL CALLED: read_project_file(%s)", path)
        try:
            target = self._resolve_path(path)
            if not target.exists():
                LOGGER.warning("read_project_file missing path: %s", target)
                return f"Error: file '{path}' does not exist."

            LOGGER.debug("Reading file at %s", target)
            return target.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read file %s: %s", path, exc)
            return f"Error reading '{path}': {exc}"

    def list_project_files(self, directory: str = ".", extension: str = ".py") -> list[str]:
        """List files that match the provided extension within the workspace."""
        LOGGER.info("🔧 TOOL CALLED: list_project_files(%s, %s)", directory, extension)
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                LOGGER.warning("list_project_files base missing: %s", base)
                return []

            self._ensure_gitignore_state()
            base_relative = self._relative_path(base)
            if self._is_path_ignored(base_relative, is_dir=True):
                LOGGER.info(
                    "list_project_files skipping %s because it is ignored by .gitignore",
                    base_relative or ".",
                )
                return []

            suffix = extension if extension.startswith(".") else f".{extension}"
            LOGGER.debug(
                "Scanning %s for *%s (workspace=%s)", base, suffix, self.workspace_dir
            )
            files: list[str] = []
            for path in self._iter_workspace_files(base):
                if suffix and path.suffix != suffix:
                    continue
                files.append(self._relative_path(path))

            LOGGER.info(
                "list_project_files returning %d paths from %s", len(files), base
            )
            return sorted(files)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to list project files in %s: %s", directory, exc)
            return [f"Error listing files in '{directory}': {exc}"]

    def search_in_files(
        self,
        pattern: str,
        directory: str = ".",
        file_extension: str = ".py",
    ) -> dict[str, Any]:
        """Search for a case-insensitive pattern within workspace files.

        Args:
            pattern: Search pattern (case-insensitive)
            directory: Directory to search in (default: ".")
            file_extension: File extension filter (default: ".py")

        Returns:
            Dictionary with format: {"matches": [{"file": str, "line_number": int, "content": str}], "total": int}
        """
        LOGGER.info("🔧 TOOL CALLED: search_in_files(%s)", pattern)
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                return {"matches": [], "total": 0}

            suffix = file_extension if file_extension.startswith(".") else f".{file_extension}"
            matches = []
            lowered = pattern.lower()

            for file_path in base.rglob(f"*{suffix}"):
                if not file_path.is_file():
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError) as exc:
                    LOGGER.debug("Skipping unreadable file %s: %s", file_path, exc)
                    continue

                for line_num, line in enumerate(content.splitlines(), start=1):
                    if lowered in line.lower():
                        matches.append(
                            {
                                "file": self._relative_path(file_path),
                                "line_number": line_num,
                                "content": line.strip(),
                            }
                        )
                        if len(matches) >= 50:
                            LOGGER.info("Search hit 50 match limit")
                            return {"matches": matches, "total": len(matches), "truncated": True}

            LOGGER.info("Search found %d matches for pattern: %s", len(matches), pattern)
            return {"matches": matches, "total": len(matches)}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to search files for pattern %s: %s", pattern, exc)
            return {"matches": [], "total": 0, "error": f"Error searching for '{pattern}': {exc}"}

    def read_multiple_files(self, file_paths: list[str]) -> dict[str, Any]:
        """Read multiple files and return structured results.

        Args:
            file_paths: List of file paths to read

        Returns:
            Dictionary with file paths as keys and content/error info as values
            Example: {"file1.py": {"success": True, "content": "..."}, "file2.py": {"success": False, "error": "..."}}
        """
        LOGGER.info("🔧 TOOL CALLED: read_multiple_files(%s)", file_paths)
        try:
            if not file_paths:
                return {"error": "No files specified"}

            results = {}
            for user_path in file_paths:
                try:
                    target = self._resolve_path(user_path)
                except PermissionError as exc:
                    LOGGER.warning(
                        "Denied read_multiple_files outside workspace: %s | workspace=%s",
                        exc,
                        self.workspace_dir,
                    )
                    results[user_path] = {"success": False, "error": str(exc)}
                    continue

                if not target.exists():
                    LOGGER.warning("read_multiple_files missing path: %s", target)
                    results[user_path] = {"success": False, "error": "file does not exist"}
                    continue

                if not target.is_file():
                    LOGGER.warning("read_multiple_files non-file path: %s", target)
                    results[user_path] = {"success": False, "error": "not a file"}
                    continue

                try:
                    LOGGER.debug("Reading multiple file entry: %s", target)
                    content = target.read_text(encoding="utf-8")
                    results[user_path] = {"success": True, "content": content}
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to read %s: %s", user_path, exc)
                    results[user_path] = {"success": False, "error": str(exc)}

            return results
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read multiple files: %s", exc)
            return {"error": f"Failed to read files: {exc}"}

    # ------------------------------------------------------------------ #
    # Git operations
    # ------------------------------------------------------------------ #
    def get_git_status(self) -> str:
        """Return the short git status for the current repository.

        Returns:
            Git status output or "clean" if no changes, or error message
        """
        LOGGER.info("🔧 TOOL CALLED: get_git_status")
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workspace_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "git status failed"
            LOGGER.error("git status failed: %s", error)
            return f"Error: {error}"
        return result.stdout.strip() or "clean"

    def git_commit(self, message: str = "") -> str:
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
            cwd=self.workspace_dir,
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
                cwd=self.workspace_dir,
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
            cwd=self.workspace_dir,
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

    def git_push(self, remote: str = "origin", branch: str = "main") -> str:
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
            cwd=self.workspace_dir,
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

    def git_diff(self, file_path: str = "", staged: bool = False) -> str:
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
                cwd=self.workspace_dir,
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

    def git_blame(self, file_path: str, line_number: int) -> dict[str, Any]:
        """Return author and commit metadata for a specific file line.

        Returns:
            Dictionary with blame info or error
        """
        LOGGER.info("🔧 TOOL CALLED: git_blame(%s:%s)", file_path, line_number)
        if line_number <= 0:
            return {"error": "Line number must be greater than zero."}

        repo = self._load_repo()
        if isinstance(repo, dict):
            return repo

        target_path = self._resolve_repo_path(repo, file_path)
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

    def create_new_branch(self, branch_name: str, start_point: str = "HEAD") -> dict[str, Any]:
        """Create and check out a new git branch based on start_point.

        Returns:
            Dictionary with success status and details
        """
        LOGGER.info("🔧 TOOL CALLED: create_new_branch(%s)", branch_name)
        if not branch_name or branch_name.strip() == "":
            return {"success": False, "error": "Branch name cannot be empty."}

        repo = self._load_repo()
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

    def _load_repo(self):
        """Load git repository starting from workspace directory. Returns Repo object or dict with error."""
        if Repo is None:
            return {"error": "GitPython is not installed. Install it with: pip install GitPython"}
        try:
            return Repo(self.workspace_dir, search_parent_directories=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load git repository: %s", exc)
            return {"error": f"Failed to open git repository: {exc}"}

    def _resolve_repo_path(self, repo: Repo, file_path: str):
        """Resolve file path relative to repository root. Returns Path or None."""
        root = Path(repo.working_tree_dir or self.workspace_dir).resolve()
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        try:
            return candidate.resolve().relative_to(root)
        except ValueError:
            return None

    # ------------------------------------------------------------------ #
    # Python tool operations
    # ------------------------------------------------------------------ #
    def run_tests(self, test_path: str = "tests/", verbose: bool = False) -> dict[str, Any]:
        """Run pytest on the codebase and return test results.

        Args:
            test_path: Path to tests directory or file (default: "tests/")
            verbose: Enable verbose output (default: False)

        Returns:
            Dictionary with keys: passed, failed, duration, output
        """
        LOGGER.info("🔧 TOOL CALLED: run_tests(%s)", test_path)
        try:
            cmd = ["pytest", test_path]
            if verbose:
                cmd.append("-v")
            cmd.extend(["--tb=short", "-q"])

            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                check=False,
                capture_output=True,
                text=True,
            )

            output = result.stdout + result.stderr

            passed = 0
            failed = 0
            duration = 0.0

            for line in output.split("\n"):
                if "passed" in line or "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "passed" in part and i > 0:
                            try:
                                passed = int(parts[i - 1])
                            except (ValueError, IndexError):
                                pass
                        if "failed" in part and i > 0:
                            try:
                                failed = int(parts[i - 1])
                            except (ValueError, IndexError):
                                pass
                if "seconds" in line or "s" in line:
                    import re

                    match = re.search(r"(\d+\.?\d*)\s*s", line)
                    if match:
                        duration = float(match.group(1))

            LOGGER.info("Tests completed: passed=%d, failed=%d", passed, failed)
            return {
                "passed": passed,
                "failed": failed,
                "duration": duration,
                "output": output.strip(),
            }

        except FileNotFoundError:
            LOGGER.error("pytest is not installed or not found in PATH")
            return {
                "passed": 0,
                "failed": 0,
                "duration": 0.0,
                "output": "Error: pytest is not installed. Install with: pip install pytest",
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to run tests: %s", exc)
            return {
                "passed": 0,
                "failed": 0,
                "duration": 0.0,
                "output": f"Error running tests: {exc}",
            }

    def lint_code(self, file_paths: list[str], directory: str = ".") -> dict[str, Any]:
        """Run pylint to catch errors and code quality issues.

        Args:
            file_paths: List of specific files to lint (empty list to scan directory)
            directory: Directory to lint if file_paths is empty (default: ".")

        Returns:
            Dictionary with keys: errors (list), warnings (list), score (float), output (str)
        """
        LOGGER.info("🔧 TOOL CALLED: lint_code(%s)", file_paths or directory)
        try:
            cmd = ["pylint"]

            if file_paths and len(file_paths) > 0:
                cmd.extend(file_paths)
            else:
                base = Path(directory)
                if not base.is_absolute():
                    base = self.workspace_dir / base

                if base.exists():
                    py_files = [str(f) for f in base.rglob("*.py") if f.is_file()]
                    if not py_files:
                        return {
                            "errors": [],
                            "warnings": [],
                            "score": 10.0,
                            "output": "No Python files found to lint.",
                        }
                    cmd.extend(py_files[:20])
                else:
                    return {
                        "errors": [],
                        "warnings": [],
                        "score": 0.0,
                        "output": f"Error: directory '{directory}' does not exist.",
                    }

            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                check=False,
                capture_output=True,
                text=True,
            )

            output = result.stdout + result.stderr

            errors = []
            warnings = []
            score = 0.0

            for line in output.split("\n"):
                line_lower = line.lower()
                if ": error:" in line_lower or ": e" in line_lower:
                    errors.append(line.strip())
                elif ": warning:" in line_lower or ": w" in line_lower:
                    warnings.append(line.strip())
                elif "your code has been rated at" in line_lower:
                    import re

                    match = re.search(r"rated at ([\d.]+)/", line)
                    if match:
                        score = float(match.group(1))

            if "No module named" in output or "not found" in output.lower():
                LOGGER.error("pylint is not installed or not found in PATH")
                return {
                    "errors": ["pylint is not installed. Install with: pip install pylint"],
                    "warnings": [],
                    "score": 0.0,
                    "output": output.strip(),
                }

            LOGGER.info(
                "Linting completed: errors=%d, warnings=%d, score=%.2f",
                len(errors),
                len(warnings),
                score,
            )
            return {
                "errors": errors[:20],
                "warnings": warnings[:20],
                "score": score,
                "output": output.strip(),
            }

        except FileNotFoundError:
            LOGGER.error("pylint is not installed or not found in PATH")
            return {
                "errors": ["pylint is not installed. Install with: pip install pylint"],
                "warnings": [],
                "score": 0.0,
                "output": "Error: pylint not found",
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to lint code: %s", exc)
            return {
                "errors": [str(exc)],
                "warnings": [],
                "score": 0.0,
                "output": f"Error linting code: {exc}",
            }

    def install_package(self, package: str, version: str = "") -> str:
        """Install a Python package using pip.

        Args:
            package: Package name to install (required)
            version: Optional version constraint (e.g., ">=1.0.0")

        Returns:
            Success or error message as a string
        """
        LOGGER.info("🔧 TOOL CALLED: install_package(%s)", package)
        if not package or not package.strip():
            return "Error: package name cannot be empty"

        try:
            package_spec = package.strip()
            if version:
                package_spec = f"{package_spec}{version}"

            cmd = ["pip", "install", package_spec, "--break-system-packages"]

            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                error = (
                    result.stderr.strip() or result.stdout.strip() or "pip install failed"
                )
                LOGGER.error("pip install failed for %s: %s", package_spec, error)
                return f"Error installing {package_spec}: {error}"

            output = result.stdout.strip()
            LOGGER.info("Package installed successfully: %s", package_spec)
            return f"✅ Successfully installed {package_spec}\n{output}"

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to install package %s: %s", package, exc)
            return f"Error installing package: {exc}"

    def format_code(
        self,
        file_paths: list[str], directory: str = "."
    ) -> dict[str, Any]:
        """Format Python code using Black formatter.

        Args:
            file_paths: List of specific files to format (empty list to format directory)
            directory: Directory to format if file_paths is empty (default: ".")

        Returns:
            Dictionary with keys: formatted (count), errors (list), message (summary)
        """
        LOGGER.info("🔧 TOOL CALLED: format_code(%s)", file_paths or directory)
        try:
            cmd = ["black"]

            if file_paths and len(file_paths) > 0:
                cmd.extend(file_paths)
            else:
                cmd.append(directory)

            result = subprocess.run(
                cmd,
                cwd=self.workspace_dir,
                check=False,
                capture_output=True,
                text=True,
            )

            output = result.stdout + result.stderr

            formatted_count = output.count("reformatted")
            if "reformatted" not in output and result.returncode == 0:
                formatted_count = 0

            errors = []
            if result.returncode != 0:
                if "No module named" in output or "not found" in output:
                    errors.append("Black is not installed. Install with: pip install black")
                else:
                    errors.append(output.strip())

            message = output.strip() if output.strip() else "No files needed formatting"

            LOGGER.info(
                "Code formatting completed: formatted=%d, errors=%d",
                formatted_count,
                len(errors),
            )
            return {
                "formatted": formatted_count,
                "errors": errors,
                "message": message,
            }

        except FileNotFoundError:
            LOGGER.error("Black is not installed or not found in PATH")
            return {
                "formatted": 0,
                "errors": ["Black is not installed. Install with: pip install black"],
                "message": "Error: Black formatter not found",
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to format code: %s", exc)
            return {
                "formatted": 0,
                "errors": [str(exc)],
                "message": f"Error formatting code: {exc}",
            }

    def get_function_definitions(self, file_path: str) -> dict[str, Any]:
        """Extract function signatures from a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary with 'functions' key containing list of function definitions
            Example: {"functions": [{"name": "generate_password", "params": ["length", "use_numbers"], "line": 5, "docstring": "..."}]}
        """
        LOGGER.info("🔧 TOOL CALLED: get_function_definitions(%s)", file_path)
        try:
            target = Path(file_path)
            if not target.is_absolute():
                target = self.workspace_dir / target

            if not target.exists():
                LOGGER.error("File does not exist: %s", file_path)
                return {"functions": [], "error": "File does not exist"}

            if not target.suffix == ".py":
                LOGGER.error("File is not a Python file: %s", file_path)
                return {"functions": [], "error": "File is not a Python file"}

            content = target.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(target))

            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    for arg in node.args.args:
                        params.append(arg.arg)

                    docstring = ast.get_docstring(node)

                    functions.append(
                        {
                            "name": node.name,
                            "params": params,
                            "line": node.lineno,
                            "docstring": docstring or "",
                        }
                    )

            LOGGER.info("Extracted %d function definitions from %s", len(functions), file_path)
            return {"functions": functions, "total": len(functions)}

        except SyntaxError as exc:
            LOGGER.error("Syntax error in file %s: %s", file_path, exc)
            return {"functions": [], "error": f"Syntax error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to extract function definitions from %s: %s", file_path, exc)
            return {"functions": [], "error": str(exc)}

    def get_cyclomatic_complexity(self, file_path: str) -> dict[str, Any]:
        """Calculate cyclomatic complexity metrics for the provided Python file.

        Returns:
            Dictionary with complexity metrics
        """
        LOGGER.info("🔧 TOOL CALLED: get_cyclomatic_complexity(%s)", file_path)
        try:
            from radon.complexity import cc_visit
        except ImportError:
            return {
                "error": "Radon is not installed. Install it with: pip install radon",
            }

        target = self._resolve_python_path(file_path)
        if not target.exists():
            return {"error": f"File does not exist: {file_path}"}

        try:
            content = target.read_text(encoding="utf-8")
            blocks = cc_visit(content)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to compute complexity for %s: %s", file_path, exc)
            return {"error": f"Failed to compute complexity: {exc}"}

        entries = []
        complexities = []
        for block in blocks:
            name = f"{block.classname}.{block.name}" if getattr(block, "classname", None) else block.name
            entry = {
                "name": name,
                "complexity": block.complexity,
                "rank": getattr(block, "rank", ""),
                "lineno": getattr(block, "lineno", None),
                "endline": getattr(block, "endline", None),
                "is_method": bool(getattr(block, "classname", None)),
                "is_async": bool(getattr(block, "is_async", False)),
            }
            entries.append(entry)
            complexities.append(block.complexity)

        summary = {
            "count": len(entries),
            "max": max(complexities) if complexities else 0,
            "min": min(complexities) if complexities else 0,
            "average": round(statistics.mean(complexities), 2) if complexities else 0,
            "high_complexity": [item for item in entries if item.get("rank") in {"D", "E", "F"}],
        }

        return {
            "file": str(target),
            "results": entries,
            "summary": summary,
        }

    def generate_test_file(
        self,
        source_file: str,
        tests_root: str = "tests",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Create or extend a pytest test file with stubs for public callables in source_file.

        Returns:
            Dictionary with generation results
        """
        LOGGER.info("🔧 TOOL CALLED: generate_test_file(%s)", source_file)
        source_path = self._resolve_python_path(source_file)
        if not source_path.exists():
            return {"success": False, "error": f"Source file does not exist: {source_file}"}
        if source_path.suffix != ".py":
            return {"success": False, "error": "Source file must be a Python module."}

        try:
            content = source_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(source_path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to parse %s: %s", source_file, exc)
            return {"success": False, "error": f"Failed to parse source file: {exc}"}

        symbols = self._collect_public_callables(tree)
        if not symbols["functions"] and not symbols["methods"]:
            return {"success": False, "error": "No public callables found to generate tests for."}

        module_parts = self._module_parts_from_source(source_path)
        module_path = ".".join(module_parts) if module_parts else source_path.stem
        tests_directory = self._resolve_tests_root(tests_root)
        destination = self._compute_test_destination(module_parts, tests_directory, source_path)
        stubs = self._build_test_stubs(symbols)

        header = self._build_test_header(module_path, symbols)
        new_content = header + "\n\n" + "\n\n".join(block for _, block in stubs) + "\n"

        if destination.exists() and not overwrite:
            existing = destination.read_text(encoding="utf-8")
            missing_blocks = [
                (name, block)
                for name, block in stubs
                if f"def test_{name}" not in existing
            ]
            if not missing_blocks:
                return {
                    "success": True,
                    "created": False,
                    "path": str(destination),
                    "message": "Test file already contains stubs for all public callables.",
                }
            updated = existing.rstrip() + "\n\n" + "\n\n".join(block for _, block in missing_blocks) + "\n"
            destination.write_text(updated, encoding="utf-8")
            return {
                "success": True,
                "created": False,
                "path": str(destination),
                "added_tests": [name for name, _ in missing_blocks],
            }

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(new_content, encoding="utf-8")
        return {
            "success": True,
            "created": True,
            "path": str(destination),
            "tests_created": [name for name, _ in stubs],
        }

    def _resolve_python_path(self, path_like: str) -> Path:
        """Resolve Python file path relative to workspace."""
        target = Path(path_like)
        if not target.is_absolute():
            target = self.workspace_dir / target
        return target.resolve()

    def _module_parts_from_source(self, source_path: Path) -> list[str]:
        """Extract module path parts from source file."""
        try:
            relative = source_path.resolve().relative_to(self.workspace_dir)
            module_path = relative.with_suffix("")
            parts = list(module_path.parts)
            if parts and parts[0] == "src":
                parts = parts[1:]
            return parts or [source_path.with_suffix("").name]
        except ValueError:
            return [source_path.with_suffix("").name]

    def _collect_public_callables(self, tree: ast.AST) -> dict[str, Any]:
        """Collect public functions and methods from AST."""
        functions = []
        methods = []

        for node in tree.body:  # type: ignore[union-attr]
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and not child.name.startswith("_"):
                        methods.append((node.name, child.name))

        return {
            "functions": sorted(set(functions)),
            "methods": sorted(set(methods)),
        }

    def _resolve_tests_root(self, root: str) -> Path:
        """Resolve tests root directory."""
        tests_root = Path(root)
        if not tests_root.is_absolute():
            tests_root = self.workspace_dir / tests_root
        return tests_root.resolve()

    def _compute_test_destination(self, module_parts: list[str], tests_root: Path, source_path: Path) -> Path:
        """Compute destination path for test file."""
        if module_parts:
            module_name = module_parts[-1]
            sub_path = Path(*module_parts[:-1]) if len(module_parts) > 1 else Path()
            destination_dir = tests_root / sub_path
        else:
            module_name = source_path.stem
            destination_dir = tests_root
        return destination_dir / f"test_{module_name}.py"

    def _build_test_header(self, module_path: str, symbols: dict[str, Any]) -> str:
        """Build test file header with imports."""
        imports = sorted(
            set(symbols["functions"]) | {cls for cls, _ in symbols["methods"]}
        )
        lines = [
            f'"""Auto-generated pytest stubs for {module_path}."""',
            "",
            "import pytest",
        ]
        if module_path:
            if imports:
                lines.append(f"from {module_path} import {', '.join(imports)}")
            else:
                lines.append(f"import {module_path}")
        return "\n".join(lines).strip()

    def _build_test_stubs(self, symbols: dict[str, Any]) -> list[tuple[str, str]]:
        """Build test stub code for functions and methods."""
        stubs = []
        for func in symbols["functions"]:
            body = "\n".join(
                [
                    f"def test_{func}():",
                    f'    """Auto-generated stub for {func}."""',
                    '    assert False, "TODO: implement test"',
                ]
            )
            stubs.append((func, body))

        for cls, method in symbols["methods"]:
            stub_name = f"{cls}_{method}"
            qualified = f"{cls}.{method}"
            body = "\n".join(
                [
                    f"def test_{stub_name}():",
                    f'    """Auto-generated stub for {qualified}."""',
                    '    assert False, "TODO: implement test"',
                ]
            )
            stubs.append((stub_name, body))

        return stubs

    # ------------------------------------------------------------------ #
    # Symbol analysis tools
    # ------------------------------------------------------------------ #
    def find_definition(self, symbol_name: str, search_directory: str = ".") -> dict[str, Any]:
        """Find where a symbol (class/function/variable) is defined.

        Args:
            symbol_name: Name of the symbol to search for
            search_directory: Directory to search recursively (default ".")

        Returns:
            Dictionary with keys: found, file, line, type, signature, docstring, context
        """
        LOGGER.info("🔧 TOOL CALLED: find_definition(symbol_name=%s)", symbol_name)
        search_path = self._resolve_directory(search_directory)
        if not search_path.exists():
            return {"found": False, "error": f"Directory does not exist: {search_directory}"}

        for py_file in search_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(py_file))
                lines = content.splitlines()
                for node in ast.walk(tree):
                    result = self._match_definition_node(node, symbol_name, py_file, lines)
                    if result:
                        LOGGER.debug("Found %s in %s at line %s", symbol_name, result['file'], result['line'])
                        return result
            except Exception as exc:
                LOGGER.warning("Failed to parse %s: %s", py_file, exc)

        LOGGER.debug("Symbol %s not found in %s", symbol_name, search_directory)
        return {"found": False, "error": f"Symbol '{symbol_name}' not found"}

    def _match_definition_node(self, node: ast.AST, symbol_name: str, file_path: Path, lines: list[str]):
        """Match AST node against symbol name. Returns dict or None."""
        if isinstance(node, ast.ClassDef) and node.name == symbol_name:
            return self._create_def_result(file_path, node, "class", lines)
        elif isinstance(node, ast.FunctionDef) and node.name == symbol_name:
            return self._create_def_result(file_path, node, "function", lines)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    return self._create_def_result(file_path, node, "variable", lines)
        return None

    def _create_def_result(self, file_path: Path, node: ast.AST, def_type: str, lines: list[str]) -> dict[str, Any]:
        """Create definition result dictionary."""
        line_num = node.lineno
        start, end = max(0, line_num - 4), min(len(lines), line_num + 3)
        return {
            "found": True, "file": str(file_path), "line": line_num, "type": def_type,
            "signature": lines[line_num - 1].strip() if line_num <= len(lines) else "",
            "docstring": ast.get_docstring(node) or "", "context": lines[start:end]
        }

    def find_usages(self, symbol_name: str, search_directory: str = ".") -> dict[str, Any]:
        """Find all usages of a symbol in Python files.

        Args:
            symbol_name: Name of the symbol to search for
            search_directory: Directory to search recursively (default ".")

        Returns:
            Dictionary with keys: total_usages, files_count, usages (list of usage dicts)
        """
        LOGGER.info("🔧 TOOL CALLED: find_usages(symbol_name=%s)", symbol_name)
        search_path = self._resolve_directory(search_directory)
        if not search_path.exists():
            return {"total_usages": 0, "files_count": 0, "error": f"Directory does not exist: {search_directory}"}

        all_usages = []
        files_with_usages = set()

        for py_file in search_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(py_file))
                lines = content.splitlines()
                for node in ast.walk(tree):
                    usage_type = self._classify_usage_type(node, symbol_name)
                    if usage_type and hasattr(node, 'lineno') and node.lineno <= len(lines):
                        all_usages.append({
                            "file": str(py_file), "line": node.lineno,
                            "context": lines[node.lineno - 1].strip(), "usage_type": usage_type
                        })
                        files_with_usages.add(str(py_file))
                    if len(all_usages) >= 100:
                        break
            except Exception as exc:
                LOGGER.warning("Failed to parse %s: %s", py_file, exc)
            if len(all_usages) >= 100:
                break

        LOGGER.debug("Found %d usages of %s in %d files", len(all_usages), symbol_name, len(files_with_usages))
        return {"total_usages": len(all_usages), "files_count": len(files_with_usages), "usages": all_usages[:100]}

    def _classify_usage_type(self, node: ast.AST, symbol_name: str) -> str:
        """Classify how a symbol is being used. Returns usage type string or empty string."""
        if isinstance(node, ast.ImportFrom) and any(alias.name == symbol_name for alias in node.names):
            return "import"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == symbol_name:
            return "call"
        elif isinstance(node, ast.Attribute) and node.attr == symbol_name:
            return "attribute"
        elif isinstance(node, ast.Name) and node.id == symbol_name:
            return "reference"
        return ""

    def get_imports(self, file_path: str) -> dict[str, Any]:
        """Extract and categorize all imports from a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary with keys: stdlib, third_party, local, import_details
        """
        LOGGER.info("🔧 TOOL CALLED: get_imports(file_path=%s)", file_path)
        path = self._resolve_path(file_path)
        if not path.exists():
            return {"error": f"File does not exist: {file_path}"}

        try:
            content = path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=file_path)
            stdlib_imports, third_party_imports, local_imports, import_details = [], [], [], []

            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    details = self._parse_import_node(node)
                    import_details.append(details)
                    module = details["module"]
                    if self._is_local_import(module):
                        local_imports.append(module)
                    elif self._is_stdlib_module(module):
                        stdlib_imports.append(module)
                    else:
                        third_party_imports.append(module)

            LOGGER.debug("Extracted %d imports from %s", len(import_details), file_path)
            return {
                "stdlib": sorted(set(stdlib_imports)), "third_party": sorted(set(third_party_imports)),
                "local": sorted(set(local_imports)), "import_details": import_details
            }
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", file_path, exc)
            return {"error": f"Failed to parse file: {exc}"}

    def _parse_import_node(self, node) -> dict[str, Any]:
        """Parse import node into structured data. Accepts ast.Import or ast.ImportFrom."""
        if isinstance(node, ast.Import):
            return {
                "line": node.lineno, "module": node.names[0].name if node.names else "",
                "names": [alias.name for alias in node.names],
                "alias": node.names[0].asname if node.names and node.names[0].asname else None, "type": "import"
            }
        return {
            "line": node.lineno, "module": node.module or "", "names": [alias.name for alias in node.names],
            "alias": None, "type": "from_import"
        }

    def _is_local_import(self, module: str) -> bool:
        """Check if module is a local import."""
        return module.startswith(".") or module.startswith("src.") or module.startswith("aura.")

    def _is_stdlib_module(self, module: str) -> bool:
        """Check if module is from standard library."""
        return module.split(".")[0] in STDLIB_MODULES

    def get_dependency_graph(self, symbol_name: str, search_directory: str = ".") -> dict[str, Any]:
        """Build a lightweight dependency graph for a symbol across the project.

        Returns:
            Dictionary with dependency graph data
        """
        LOGGER.info("🔧 TOOL CALLED: get_dependency_graph(%s)", symbol_name)
        search_path = self._resolve_directory(search_directory)
        if not search_path.exists():
            return {"error": f"Directory does not exist: {search_directory}"}

        target = self._locate_symbol(symbol_name, search_path)
        if not target:
            return {"error": f"Symbol '{symbol_name}' not found in {search_directory}"}

        node, source_path, lines = target
        dependencies = self._collect_symbol_dependencies(node, source_path, lines)
        dependents = self._collect_symbol_dependents(symbol_name, search_path, limit=150)

        return {
            "symbol": symbol_name,
            "defined_in": str(source_path),
            "line": getattr(node, "lineno", None),
            "type": node.__class__.__name__,
            "dependencies": dependencies,
            "dependents": dependents,
            "summary": {
                "dependency_count": len(dependencies),
                "dependents_count": len(dependents),
            },
        }

    def get_class_hierarchy(self, class_name: str, search_directory: str = ".") -> dict[str, Any]:
        """Return inheritance details for a class, including parents and subclasses.

        Returns:
            Dictionary with class hierarchy data
        """
        LOGGER.info("🔧 TOOL CALLED: get_class_hierarchy(%s)", class_name)
        search_path = self._resolve_directory(search_directory)
        if not search_path.exists():
            return {"error": f"Directory does not exist: {search_directory}"}

        class_map = {}
        children_map = defaultdict(list)

        for py_file in search_path.rglob("*.py"):
            content, tree = self._read_ast(py_file)
            if not tree or content is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [self._expression_to_name(base) for base in node.bases if self._expression_to_name(base)]
                    info = {
                        "name": node.name,
                        "file": str(py_file),
                        "line": node.lineno,
                        "bases": bases,
                        "docstring": ast.get_docstring(node) or "",
                    }
                    class_map[node.name] = info
                    for base in bases:
                        children_map[base].append(node.name)

        if class_name not in class_map:
            return {"error": f"Class '{class_name}' not found in {search_directory}"}

        ancestors = self._collect_ancestors(class_name, class_map)
        descendants = self._collect_descendants(class_name, children_map)

        return {
            "class": class_name,
            "defined_in": class_map[class_name]["file"],
            "line": class_map[class_name]["line"],
            "bases": class_map[class_name]["bases"],
            "ancestors": ancestors,
            "descendants": descendants,
            "hierarchy": self._build_hierarchy_branch(class_name, class_map, children_map),
        }

    def safe_rename_symbol(
        self,
        file_path: str,
        symbol_name: str,
        new_name: str,
        project_root: str = "",
    ) -> dict[str, Any]:
        """Perform a project-wide, refactor-aware rename using Rope.

        Args:
            file_path: Path to the file containing the symbol
            symbol_name: Name of the symbol to rename
            new_name: New name for the symbol
            project_root: Project root directory (empty string to use workspace)

        Returns:
            Dictionary with rename results
        """
        LOGGER.info("🔧 TOOL CALLED: safe_rename_symbol(%s -> %s)", symbol_name, new_name)
        if not new_name or not new_name.isidentifier():
            return {"success": False, "error": "New name must be a valid identifier."}

        try:
            from rope.base import project as rope_project
            from rope.base.exceptions import RopeError
            from rope.refactor.rename import Rename
        except ImportError:
            return {
                "success": False,
                "error": "Rope is not installed. Install it with: pip install rope",
            }

        target_path = self._resolve_path(file_path)
        if not target_path.exists():
            return {"success": False, "error": f"File does not exist: {file_path}"}

        try:
            source = target_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(target_path))
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Failed to parse file: {exc}"}

        target_node = self._find_symbol_node(tree, symbol_name)
        if not target_node:
            return {"success": False, "error": f"Symbol '{symbol_name}' not found in {file_path}"}

        offset = self._calculate_offset(source, getattr(target_node, "lineno", 1), getattr(target_node, "col_offset", 0))
        root = Path(project_root).resolve() if project_root else self.workspace_dir

        proj: rope_project.Project | None = None
        try:
            proj = rope_project.Project(str(root))
            relative = str(target_path.resolve().relative_to(root))
        except ValueError:
            if proj:
                proj.close()
            return {"success": False, "error": f"File {file_path} is outside the project root {root}"}

        try:
            resource = proj.find_resource(relative)
            rename_refactor = Rename(proj, resource, offset)
            changes = rename_refactor.get_changes(new_name)
            proj.do(changes)
            changed = [res.path for res in changes.get_changed_resources()]
            return {
                "success": True,
                "message": f"Renamed {symbol_name} to {new_name}",
                "files_updated": changed,
            }
        except RopeError as exc:
            LOGGER.error("Rope rename failed: %s", exc)
            return {"success": False, "error": f"Rename failed: {exc}"}
        finally:
            if proj:
                try:
                    proj.close()
                except Exception:  # noqa: BLE001
                    pass

    def _locate_symbol(self, symbol_name: str, search_path: Path):
        """Locate symbol definition in search path. Returns tuple or None."""
        for py_file in search_path.rglob("*.py"):
            content, tree = self._read_ast(py_file)
            if not tree or content is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_name:
                    return node, py_file, content.splitlines()
        return None

    def _collect_symbol_dependencies(self, node: ast.AST, file_path: Path, lines: list[str]) -> list[dict[str, Any]]:
        """Collect dependencies for a symbol."""
        dependencies = []
        seen = set()

        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = self._expression_to_name(base)
                if base_name:
                    key = (base_name, "base_class", node.lineno)
                    if key not in seen:
                        dependencies.append(
                            {
                                "name": base_name,
                                "kind": "base_class",
                                "file": str(file_path),
                                "line": node.lineno,
                            }
                        )
                        seen.add(key)

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._expression_to_name(child.func)
                if call_name:
                    lineno = getattr(child, "lineno", None)
                    key = (call_name, "call", lineno or 0)
                    if key not in seen:
                        dependencies.append(
                            {
                                "name": call_name,
                                "kind": "call",
                                "file": str(file_path),
                                "line": lineno,
                                "context": self._line_text(lines, lineno),
                            }
                        )
                        seen.add(key)
            elif isinstance(child, ast.Attribute):
                attr_name = child.attr
                lineno = getattr(child, "lineno", None)
                key = (attr_name, "attribute", lineno or 0)
                if key not in seen and attr_name:
                    dependencies.append(
                        {
                            "name": attr_name,
                            "kind": "attribute",
                            "file": str(file_path),
                            "line": lineno,
                            "context": self._line_text(lines, lineno),
                        }
                    )
                    seen.add(key)

        return dependencies

    def _collect_symbol_dependents(self, symbol_name: str, search_path: Path, limit: int = 150) -> list[dict[str, Any]]:
        """Collect symbols that depend on this symbol."""
        dependents = []
        seen_locations = set()

        for py_file in search_path.rglob("*.py"):
            content, tree = self._read_ast(py_file)
            if not tree or content is None:
                continue
            lines = content.splitlines()

            for node in ast.walk(tree):
                if self._node_references_symbol(node, symbol_name):
                    lineno = getattr(node, "lineno", None)
                    location = (str(py_file), lineno or 0)
                    if location in seen_locations:
                        continue
                    seen_locations.add(location)
                    dependents.append(
                        {
                            "file": str(py_file),
                            "line": lineno,
                            "context": self._line_text(lines, lineno),
                            "usage_type": self._classify_usage_type(node, symbol_name),
                        }
                    )
                    if len(dependents) >= limit:
                        return dependents
        return dependents

    def _read_ast(self, py_file: Path) -> tuple:
        """Read and parse Python file to AST. Returns (content_str, ast_tree) or (None, None)."""
        try:
            content = py_file.read_text(encoding="utf-8")
            return content, ast.parse(content, filename=str(py_file))
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Skipping %s: %s", py_file, exc)
            return None, None

    def _expression_to_name(self, expr: ast.AST) -> str:
        """Convert AST expression to name string. Returns name or empty string."""
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            value = self._expression_to_name(expr.value)
            return f"{value}.{expr.attr}" if value else expr.attr
        if isinstance(expr, ast.Subscript):
            return self._expression_to_name(expr.value)
        if isinstance(expr, ast.Call):
            return self._expression_to_name(expr.func)
        return ""

    def _collect_ancestors(self, class_name: str, class_map: dict[str, Any], visited: set | None = None) -> list[str]:
        """Collect all ancestor classes."""
        visited = visited or set()
        visited.add(class_name)
        ancestors = []

        bases = class_map.get(class_name, {}).get("bases", [])
        for base in bases:
            ancestors.append(base)
            if base in class_map and base not in visited:
                ancestors.extend(self._collect_ancestors(base, class_map, visited))
        return ancestors

    def _collect_descendants(
        self,
        class_name: str,
        children_map: dict[str, Any],
        visited: set | None = None,
    ) -> list[str]:
        """Collect all descendant classes."""
        visited = visited or set()
        visited.add(class_name)
        descendants = []

        for child in children_map.get(class_name, []):
            descendants.append(child)
            if child not in visited:
                descendants.extend(self._collect_descendants(child, children_map, visited))
        return descendants

    def _build_hierarchy_branch(
        self,
        class_name: str,
        class_map: dict[str, Any],
        children_map: dict[str, Any],
        visited: set | None = None,
    ) -> dict[str, Any]:
        """Build hierarchical tree structure."""
        visited = visited or set()
        visited.add(class_name)
        info = class_map.get(class_name, {})
        return {
            "name": class_name,
            "file": info.get("file"),
            "line": info.get("line"),
            "bases": info.get("bases", []),
            "children": [
                self._build_hierarchy_branch(child, class_map, children_map, visited)
                for child in children_map.get(class_name, [])
                if child not in visited
            ],
        }

    def _find_symbol_node(self, tree: ast.AST, symbol_name: str):
        """Find AST node for symbol. Returns ast.AST node or None."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_name:
                return node
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol_name:
                        return node
        return None

    def _calculate_offset(self, source: str, line: int, column: int) -> int:
        """Calculate character offset from line/column."""
        lines = source.splitlines(keepends=True)
        line_index = max(0, line - 1)
        prior = sum(len(lines[i]) for i in range(min(line_index, len(lines))))
        return prior + column

    def _line_text(self, lines: list[str], lineno: int | None) -> str:
        """Get text of a specific line."""
        if not lineno or lineno <= 0 or lineno > len(lines):
            return ""
        return lines[lineno - 1].strip()

    def _node_references_symbol(self, node: ast.AST, symbol_name: str) -> bool:
        """Check if node references a symbol."""
        if isinstance(node, ast.Name):
            return node.id == symbol_name
        if isinstance(node, ast.Attribute):
            return node.attr == symbol_name
        if isinstance(node, ast.Call):
            ref_name = self._expression_to_name(node.func)
            return ref_name == symbol_name
        if isinstance(node, ast.ImportFrom):
            return any(alias.name == symbol_name or alias.asname == symbol_name for alias in node.names)
        if isinstance(node, ast.Import):
            return any(alias.name.split(".")[-1] == symbol_name for alias in node.names)
        return False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_path(self, user_path: str) -> Path:
        """Resolve a user-supplied path relative to the workspace."""
        candidate = Path(user_path).expanduser() if user_path else Path(".")
        if not candidate.is_absolute():
            candidate = self.workspace_dir / candidate
        resolved = candidate.resolve(strict=False)
        if not self._is_within_workspace(resolved):
            message = (
                f"Access to '{user_path}' is outside the workspace directory "
                f"({self.workspace_dir})."
            )
            raise PermissionError(message)
        LOGGER.debug("Resolved %s -> %s (workspace=%s)", user_path, resolved, self.workspace_dir)
        return resolved

    def _resolve_directory(self, directory: str) -> Path:
        """Resolve directory paths while honoring the workspace sandbox."""
        return self._resolve_path(directory or ".")

    def _relative_path(self, path: Path) -> str:
        """Return a path relative to the workspace when possible."""
        try:
            return path.relative_to(self.workspace_dir).as_posix()
        except ValueError:
            return path.as_posix()

    def _is_within_workspace(self, path: Path) -> bool:
        """Return True when the path lives inside the workspace tree."""
        try:
            path.relative_to(self.workspace_dir)
            return True
        except ValueError:
            return False

    def _reset_gitignore_cache(self) -> None:
        """Clear cached gitignore data when the workspace changes."""
        self._gitignore_mtime = None
        self._gitignore_patterns = []
        self._gitignore_spec = None
        self._gitignore_fallback_matchers = []
        self._gitignore_warning_logged = False
        self._ensure_gitignore_state()

    def _ensure_gitignore_state(self) -> None:
        """Ensure .gitignore patterns are loaded and compiled."""
        gitignore_path = self.workspace_dir / ".gitignore"
        if not gitignore_path.exists():
            self._gitignore_mtime = None
            self._gitignore_patterns = []
            self._gitignore_spec = None
            self._gitignore_fallback_matchers = []
            return

        try:
            mtime = gitignore_path.stat().st_mtime
        except OSError as exc:  # noqa: BLE001
            LOGGER.debug("Unable to stat .gitignore: %s", exc)
            return

        if self._gitignore_mtime == mtime:
            return

        patterns = load_gitignore_patterns(str(self.workspace_dir))
        self._gitignore_mtime = mtime
        self._gitignore_patterns = patterns

        spec = self._compile_gitignore_spec(patterns)
        self._gitignore_spec = spec
        if spec is None:
            self._gitignore_fallback_matchers = self._build_fallback_gitignore_matchers(
                patterns
            )
        else:
            self._gitignore_fallback_matchers = []

    def _compile_gitignore_spec(self, patterns: Sequence):
        """Return a compiled PathSpec for .gitignore patterns when available. Returns PathSpec or None."""
        if not patterns:
            return None
        if PathSpec is None:
            if patterns and not self._gitignore_warning_logged:
                LOGGER.warning(
                    "pathspec is not installed; falling back to basic .gitignore matching."
                )
                self._gitignore_warning_logged = True
            return None

        try:
            return PathSpec.from_lines("gitwildmatch", patterns)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to compile .gitignore via pathspec: %s", exc)
            return None

    def _build_fallback_gitignore_matchers(
        self,
        patterns: Sequence[str],
    ) -> list[tuple[str, bool]]:
        """Build a simple glob-based matcher for environments without pathspec."""
        matchers: list[tuple[str, bool]] = []
        for raw in patterns:
            if not raw:
                continue
            is_negated = raw.startswith("!")
            pattern = raw[1:] if is_negated else raw
            if not pattern:
                continue
            anchored = pattern.startswith("/")
            cannon = pattern.lstrip("/")
            directory_only = cannon.endswith("/")
            cannon = cannon.rstrip("/")
            if not cannon:
                continue

            base_patterns = [cannon]
            if directory_only:
                base_patterns.append(f"{cannon}/**")

            for candidate in base_patterns:
                normalized = candidate.replace("\\", "/")
                if not anchored and not normalized.startswith("**/"):
                    normalized = f"**/{normalized}"
                matchers.append((normalized, is_negated))

        return matchers

    def _is_path_ignored(self, relative_path: str, is_dir: bool) -> bool:
        """Return True if a relative path should be ignored by .gitignore rules."""
        if not relative_path or relative_path in {".", ""}:
            return False
        if not self._gitignore_patterns and not self._gitignore_spec:
            return False

        candidate = relative_path.replace("\\", "/")
        spec = self._gitignore_spec
        if spec is not None:
            if spec.match_file(candidate):
                return True
            if is_dir:
                dir_candidate = candidate if candidate.endswith("/") else f"{candidate}/"
                if spec.match_file(dir_candidate):
                    return True
            return False

        if not self._gitignore_fallback_matchers:
            return False

        targets = {candidate}
        if is_dir:
            targets.add(f"{candidate}/")

        ignored = False
        for pattern, is_negated in self._gitignore_fallback_matchers:
            if any(fnmatch.fnmatchcase(target, pattern) for target in targets):
                ignored = not is_negated
        return ignored

    def _iter_workspace_files(self, root: Path) -> Iterator[Path]:
        """Yield workspace files beneath `root`, respecting .gitignore rules."""
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except (OSError, PermissionError) as exc:
                LOGGER.debug("Skipping directory %s: %s", current, exc)
                continue

            for entry in entries:
                try:
                    resolved = entry.resolve(strict=False)
                except OSError:
                    resolved = entry

                if not self._is_within_workspace(resolved):
                    LOGGER.debug("Skipping path outside workspace bounds: %s", entry)
                    continue

                relative = self._relative_path(entry)
                is_dir = entry.is_dir()
                if self._is_path_ignored(relative, is_dir=is_dir):
                    LOGGER.debug("Skipping .gitignore-matched path: %s", relative)
                    continue

                if is_dir:
                    stack.append(entry)
                    continue

                yield entry
