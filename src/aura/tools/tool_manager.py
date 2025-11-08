"""Sandboxed access layer for Aura's filesystem tools."""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class ToolManager:
    """Provide workspace-scoped file system utilities."""

    def __init__(self, workspace_dir: str) -> None:
        resolved = Path(workspace_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
        self.workspace_dir = resolved
        LOGGER.info("ToolManager workspace set to %s", self.workspace_dir)

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
        except PermissionError as exc:
            LOGGER.warning("Denied create_file outside workspace: %s", exc)
            return f"Error creating '{path}': {exc}"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to create file %s: %s", path, exc)
            return f"Error creating '{path}': {exc}"

    def modify_file(self, path: str, old_content: str, new_content: str) -> str:
        """Replace content within a workspace file."""
        LOGGER.info("🔧 TOOL CALLED: modify_file(%s)", path)
        try:
            target = self._resolve_path(path)
        except PermissionError as exc:
            LOGGER.warning("Denied modify_file outside workspace: %s", exc)
            return f"Error modifying '{path}': {exc}"

        if not target.exists():
            return f"Error: file '{path}' does not exist."

        try:
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

    def delete_file(self, path: str) -> str:
        """Delete a workspace file."""
        LOGGER.info("🔧 TOOL CALLED: delete_file(%s)", path)
        try:
            target = self._resolve_path(path)
        except PermissionError as exc:
            LOGGER.warning("Denied delete_file outside workspace: %s", exc)
            return f"Error deleting '{path}': {exc}"

        if not target.exists():
            return f"Error: file '{path}' does not exist."

        try:
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
        except PermissionError as exc:
            LOGGER.warning(
                "Denied read_project_file outside workspace: %s | workspace=%s",
                exc,
                self.workspace_dir,
            )
            return f"Error reading '{path}': {exc}"

        if not target.exists():
            LOGGER.warning("read_project_file missing path: %s", target)
            return f"Error: file '{path}' does not exist."

        try:
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
        except PermissionError as exc:
            LOGGER.warning(
                "Denied list_project_files outside workspace: %s | workspace=%s",
                exc,
                self.workspace_dir,
            )
            return [f"Error: {exc}"]

        if not base.exists():
            LOGGER.warning("list_project_files base missing: %s", base)
            return []

        suffix = extension if extension.startswith(".") else f".{extension}"
        LOGGER.debug(
            "Scanning %s for *%s (workspace=%s)", base, suffix, self.workspace_dir
        )
        files = [
            self._relative_path(path)
            for path in base.rglob(f"*{suffix}")
            if path.is_file() and self._is_within_workspace(path)
        ]
        LOGGER.info(
            "list_project_files returning %d paths from %s", len(files), base
        )
        return sorted(files)

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
        except PermissionError as exc:
            LOGGER.warning(
                "Denied search_in_files outside workspace: %s | workspace=%s",
                exc,
                self.workspace_dir,
            )
            return {"matches": [], "error": str(exc)}

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

    def read_multiple_files(self, file_paths: list[str]) -> dict[str, str]:
        """Read multiple files and return a mapping of path → contents/error."""
        LOGGER.info("🔧 TOOL CALLED: read_multiple_files(%s)", file_paths)
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
