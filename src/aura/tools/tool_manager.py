"""Sandboxed access layer for Aura's filesystem tools."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

try:  # Optional dependency; fallback matcher used if unavailable.
    from pathspec import PathSpec
except ImportError:  # pragma: no cover - pathspec is optional at runtime.
    PathSpec = None  # type: ignore[assignment]

from aura.utils.file_filter import load_gitignore_patterns

LOGGER = logging.getLogger(__name__)


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
    ) -> dict[str, object]:
        """Search for a case-insensitive pattern within workspace files."""
        LOGGER.info("🔧 TOOL CALLED: search_in_files(%s)", pattern)
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                return {"matches": []}

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
                            return {"matches": matches}

            LOGGER.info("Search found %d matches for pattern: %s", len(matches), pattern)
            return {"matches": matches}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to search files for pattern %s: %s", pattern, exc)
            return {"matches": [], "error": f"Error searching for '{pattern}': {exc}"}

    def read_multiple_files(self, file_paths: list[str]) -> dict[str, str]:
        """Read multiple files and return a mapping of path → contents/error.

        Unexpected failures yield {"__error__": "..."} for easier debugging.
        """
        LOGGER.info("?? TOOL CALLED: read_multiple_files(%s)", file_paths)
        try:
            if not file_paths:
                return {}

            results: dict[str, str] = {}
            for user_path in file_paths:
                try:
                    target = self._resolve_path(user_path)
                except PermissionError as exc:
                    LOGGER.warning(
                        "Denied read_multiple_files outside workspace: %s | workspace=%s",
                        exc,
                        self.workspace_dir,
                    )
                    results[user_path] = f"Error reading '{user_path}': {exc}"
                    continue

                if not target.exists():
                    LOGGER.warning("read_multiple_files missing path: %s", target)
                    results[user_path] = f"Error: file '{user_path}' does not exist."
                    continue

                if not target.is_file():
                    LOGGER.warning("read_multiple_files non-file path: %s", target)
                    results[user_path] = f"Error: '{user_path}' is not a file."
                    continue

                try:
                    LOGGER.debug("Reading multiple file entry: %s", target)
                    results[user_path] = target.read_text(encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to read %s: %s", user_path, exc)
                    results[user_path] = f"Error reading '{user_path}': {exc}"

            return results
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read multiple files: %s", exc)
            return {"__error__": f"Error reading files: {exc}"}

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

    def _compile_gitignore_spec(self, patterns: Sequence[str]) -> PathSpec | None:
        """Return a compiled PathSpec for .gitignore patterns when available."""
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
