"""Sandboxed access layer for Aura's filesystem tools."""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Optional dependency; fallback matcher used if unavailable.
    from pathspec import PathSpec
except ImportError:  # pragma: no cover - pathspec is optional at runtime.
    PathSpec = None  # type: ignore[assignment]

from aura.exceptions import FileVerificationError
from aura.models import FileVerificationLog
from aura.tools.git_tools import GitTools
from aura.tools.python_tools import PythonTools
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

ASSET_TYPE_EXTENSIONS: dict[str, set[str]] = {
    "meshes": {".fbx", ".obj"},
    "textures": {".png", ".jpg", ".jpeg"},
    "sounds": {".wav", ".mp3"},
}

ASSET_TYPE_ALIASES: dict[str, str] = {
    "mesh": "meshes",
    "meshes": "meshes",
    "model": "meshes",
    "models": "meshes",
    "texture": "textures",
    "textures": "textures",
    "image": "textures",
    "images": "textures",
    "sound": "sounds",
    "sounds": "sounds",
    "audio": "sounds",
}

ASSET_EXTENSION_LOOKUP: dict[str, str] = {
    extension: asset_type
    for asset_type, extensions in ASSET_TYPE_EXTENSIONS.items()
    for extension in extensions
}

# Binary file extensions that should not be read as UTF-8 text
BINARY_EXTENSIONS: set[str] = {
    # 3D Models & Meshes
    ".fbx", ".blend", ".obj", ".gltf", ".glb", ".dae", ".3ds", ".max", ".ma", ".mb",
    # Images & Textures
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tga", ".tiff", ".webp", ".ico", ".psd",
    ".exr", ".hdr", ".dds", ".ktx", ".astc",
    # Audio
    ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus",
    # Video
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
    # Executables & Libraries
    ".exe", ".dll", ".so", ".dylib", ".a", ".lib", ".pyd",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Binary Data
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    # Game-specific
    ".unity3d", ".unitypackage", ".pak", ".asset", ".bundle",
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
}

# Mapping of file extensions to human-readable descriptions
BINARY_FILE_DESCRIPTIONS: dict[str, str] = {
    ".fbx": "FBX 3D Model",
    ".blend": "Blender 3D Project",
    ".obj": "Wavefront OBJ 3D Model",
    ".gltf": "glTF 3D Model (JSON)",
    ".glb": "glTF 3D Model (Binary)",
    ".png": "PNG Image",
    ".jpg": "JPEG Image",
    ".jpeg": "JPEG Image",
    ".wav": "WAV Audio",
    ".mp3": "MP3 Audio",
    ".ogg": "Ogg Vorbis Audio",
    ".mp4": "MP4 Video",
    ".zip": "ZIP Archive",
    ".exe": "Windows Executable",
    ".dll": "Dynamic Link Library",
    ".so": "Shared Object Library",
    ".pdf": "PDF Document",
    ".db": "Database File",
    ".sqlite": "SQLite Database",
    ".unity3d": "Unity Asset Bundle",
    ".pak": "Packed Game Assets",
}

DEFAULT_FILTERED_DIR_NAMES: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    ".claude",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".nox",
    ".tox",
    "aura-workspace",
}

DEFAULT_FILTERED_FILE_NAMES: set[str] = {
    ".DS_Store",
    "Thumbs.db",
}

DEFAULT_FILTERED_FILE_SUFFIXES: set[str] = {
    ".pyc",
}


class VerificationConfig:
    """Configuration for smart file verification with mode options.

    Modes:
    - "always": Verify all operations (default legacy behavior, slowest)
    - "smart": Smart verification - verify high-risk operations only (recommended)
    - "never": Skip verification (fastest but unsafe, not recommended)
    """

    # File extensions that should always be verified (critical code files)
    CRITICAL_EXTENSIONS = {
        ".py",     # Python source
        ".gd",     # GDScript
        ".tscn",   # Godot scene
        ".godot",  # Godot resource
        ".tres",   # Godot resource
        ".json",   # Configuration files
        ".yaml",   # Configuration files
        ".yml",    # Configuration files
        ".toml",   # Configuration files
    }

    # File extensions that can skip verification (assets, media, etc.)
    SKIP_VERIFICATION_EXTENSIONS = {
        # Images & Textures
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tga", ".tiff", ".webp", ".ico",
        ".psd", ".exr", ".hdr", ".dds", ".ktx", ".astc",
        # Audio
        ".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus",
        # Video
        ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
        # 3D Models & Meshes
        ".fbx", ".blend", ".obj", ".gltf", ".glb", ".dae", ".3ds",
        # Archives
        ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz",
        # Binary files
        ".exe", ".dll", ".so", ".dylib", ".a", ".lib",
        # Fonts
        ".ttf", ".otf", ".woff", ".woff2", ".eot",
    }

    def __init__(self, mode: str = "smart"):
        """Initialize verification config.

        Args:
            mode: Verification mode ("always", "smart", or "never")
        """
        # Check environment variable first
        env_mode = os.getenv("AURA_VERIFICATION_MODE", "").lower()
        if env_mode in ("always", "smart", "never"):
            self.mode = env_mode
        elif mode in ("always", "smart", "never"):
            self.mode = mode
        else:
            LOGGER.warning(
                "Invalid verification mode '%s', defaulting to 'smart'",
                mode,
            )
            self.mode = "smart"

        LOGGER.info("File verification mode: %s", self.mode)

    def should_verify(
        self,
        operation: str,
        file_path: str,
    ) -> bool:
        """Determine if a file operation should be verified.

        Args:
            operation: Operation type ("CREATE", "MODIFY", "DELETE")
            file_path: Path to the file being operated on

        Returns:
            True if verification should be performed, False to skip
        """
        # Never mode - skip all verification (unsafe)
        if self.mode == "never":
            return False

        # Always mode - verify everything (legacy behavior)
        if self.mode == "always":
            return True

        # Smart mode - apply intelligent logic
        extension = Path(file_path).suffix.lower()

        # Always verify MODIFY operations (highest risk)
        if operation == "MODIFY":
            return True

        # Always verify critical file extensions
        if extension in self.CRITICAL_EXTENSIONS:
            return True

        # Skip verification for asset files (low risk)
        if extension in self.SKIP_VERIFICATION_EXTENSIONS:
            LOGGER.debug(
                "Skipping verification for asset file: %s (extension: %s)",
                file_path,
                extension,
            )
            return False

        # Default: verify unknown file types to be safe
        return True


class ToolManager:
    """Provide workspace-scoped file system utilities."""

    def __init__(self, workspace_dir: str, verification_mode: str = "smart") -> None:
        resolved = Path(workspace_dir).expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Workspace directory does not exist: {workspace_dir}")
        self.workspace_dir = resolved
        self.git_tools = GitTools(self.workspace_dir)
        self.python_tools = PythonTools(self, STDLIB_MODULES)
        self.verification_config = VerificationConfig(mode=verification_mode)
        self._gitignore_mtime: float | None = None
        self._gitignore_patterns: list[str] = []
        self._gitignore_spec: PathSpec | None = None
        self._gitignore_fallback_matchers: list[tuple[str, bool]] = []
        self._gitignore_warning_logged = False
        self._default_respect_gitignore: bool = False
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
        self.git_tools.update_workspace(resolved)
        self._reset_gitignore_cache()
        LOGGER.info("ToolManager workspace updated to %s", self.workspace_dir)

    def respect_gitignore(self, enabled: bool = True) -> str:
        """Toggle whether file operations should respect .gitignore rules by default.

        This is a configuration tool for the Analyst to control gitignore behavior
        across subsequent file discovery operations. When disabled (default), all files
        including gitignored assets (meshes, textures, etc.) will be visible.

        :param enabled: When True, respect .gitignore rules. When False, show all files.
        :return: Status message confirming the new state.
        """
        LOGGER.info("ðŸ”§ TOOL CALLED: respect_gitignore(enabled=%s)", enabled)
        try:
            self._default_respect_gitignore = enabled
            state = "enabled" if enabled else "disabled"
            message = f"Gitignore filtering {state}. File operations will {'respect' if enabled else 'ignore'} .gitignore rules by default."
            LOGGER.info("âœ… respect_gitignore state updated: %s", state)
            return message
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to update respect_gitignore state: %s", exc)
            return f"Error updating gitignore state: {exc}"

    # ------------------------------------------------------------------ #
    # Verification helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _digest_content(payload: str | bytes | None) -> str | None:
        """Return a SHA-256 digest for the provided payload."""
        if payload is None:
            return None
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _record_verification_result(
        self,
        *,
        phase: str,
        operation: str,
        target: Path,
        expected: str | bytes | None,
        actual: str | bytes | None,
        success: bool,
        details: str | None = None,
    ) -> None:
        """Persist verification telemetry without interrupting tool execution."""
        rel_path = self._relative_path(target)
        try:
            FileVerificationLog.record(
                phase=phase,
                operation=operation,
                file_path=rel_path,
                expected_digest=self._digest_content(expected),
                actual_digest=self._digest_content(actual),
                success=success,
                details=details,
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "Failed to persist verification log for %s (phase=%s, op=%s)",
                rel_path,
                phase,
                operation,
                exc_info=True,
            )

    def _verify_written_file(
        self,
        *,
        operation: str,
        target: Path,
        expected_content: str,
    ) -> None:
        """Ensure the on-disk file precisely matches the expected content."""
        try:
            actual_content = target.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._record_verification_result(
                phase="write",
                operation=operation,
                target=target,
                expected=expected_content,
                actual=None,
                success=False,
                details=f"Unable to read file for verification: {exc}",
            )
            raise FileVerificationError(
                f"{operation} verification failed for '{self._relative_path(target)}'.",
                {"file": self._relative_path(target), "error": str(exc)},
            ) from exc

        success = actual_content == expected_content
        self._record_verification_result(
            phase="write",
            operation=operation,
            target=target,
            expected=expected_content,
            actual=actual_content,
            success=success,
            details=None if success else "File contents differ from expected payload.",
        )
        if not success:
            raise FileVerificationError(
                f"{operation} verification failed for '{self._relative_path(target)}'.",
                {"file": self._relative_path(target)},
            )

    def _verify_deletion(self, target: Path) -> None:
        """Confirm that a file was permanently removed."""
        exists = target.exists()
        actual_bytes = None
        details = None
        if exists:
            try:
                actual_bytes = target.read_bytes()
            except Exception as exc:  # noqa: BLE001
                details = f"Unable to inspect leftover file: {exc}"

        self._record_verification_result(
            phase="delete",
            operation="DELETE",
            target=target,
            expected=None,
            actual=actual_bytes,
            success=not exists,
            details=details if details else (None if not exists else "File still exists after delete."),
        )
        if exists:
            raise FileVerificationError(
                f"DELETE verification failed for '{self._relative_path(target)}'.",
                {"file": self._relative_path(target)},
            )

    # ------------------------------------------------------------------ #
    # File operation helpers
    # ------------------------------------------------------------------ #
    def create_file(self, path: str, content: str) -> str:
        """Create a file within the workspace."""
        LOGGER.info("ðŸ"§ TOOL CALLED: create_file(%s)", path)
        try:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if content is None:
                raise FileVerificationError(
                    "CREATE operations must include file content.",
                    {"file": path},
                )
            target.write_text(content, encoding="utf-8")

            # Smart verification - only verify if config says so
            if self.verification_config.should_verify("CREATE", path):
                self._verify_written_file(operation="CREATE", target=target, expected_content=content)
            else:
                LOGGER.debug("Skipping verification for CREATE operation on %s", path)

            size = len(content.encode("utf-8"))
            LOGGER.info("âœ… Created file: %s (%d bytes)", target, size)
            return f"Successfully created '{path}' ({size} bytes)"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to create file %s: %s", path, exc)
            raise

    def modify_file(self, path: str, old_content: str, new_content: str) -> str:
        """Replace content within a workspace file."""
        LOGGER.info("✏️ TOOL CALLED: modify_file(%s)", path)
        try:
            target = self._resolve_path(path)

            if not target.exists():
                raise FileVerificationError(
                    f"File '{path}' does not exist.",
                    {"file": path},
                )

            if new_content is None:
                raise FileVerificationError(
                    "MODIFY operations must include replacement content.",
                    {"file": path},
                )

            if not old_content:
                raise FileVerificationError(
                    "MODIFY operations must include old_content for auditing.",
                    {"file": path},
                )

            current = target.read_text(encoding="utf-8")
            if old_content not in current:
                raise FileVerificationError(
                    "old_content not found in target file.",
                    {"file": path},
                )
            updated = current.replace(old_content, new_content)
            target.write_text(updated, encoding="utf-8")

            # Smart verification - only verify if config says so
            if self.verification_config.should_verify("MODIFY", path):
                self._verify_written_file(operation="MODIFY", target=target, expected_content=updated)
            else:
                LOGGER.debug("Skipping verification for MODIFY operation on %s", path)

            LOGGER.info("✅ Modified file: %s", target)
            return f"Successfully modified '{path}'"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to modify file %s: %s", path, exc)
            raise

    def replace_file_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
    ) -> str:
        """Replace a block of lines using explicit line numbers."""
        LOGGER.info(
            "🪚 TOOL CALLED: replace_file_lines(%s, start=%s, end=%s)",
            path,
            start_line,
            end_line,
        )
        try:
            start = int(start_line)
            end = int(end_line)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("replace_file_lines invalid line numbers: %s", exc)
            raise ValueError("start_line and end_line must be integers.") from exc

        if start < 1 or end < start:
            raise ValueError("start_line must be >= 1 and end_line must be >= start_line.")

        try:
            target = self._resolve_path(path)
            if not target.exists():
                raise FileVerificationError(
                    f"File '{path}' does not exist.",
                    {"file": path},
                )

            contents = target.read_text(encoding="utf-8")
            lines = contents.splitlines(keepends=True)
            total_lines = len(lines)
            if end > total_lines:
                raise ValueError(
                    f"File '{path}' has only {total_lines} line(s); cannot replace through line {end}."
                )

            start_index = start - 1
            replaced_block = "".join(lines[start_index:end])
            before = "".join(lines[:start_index])
            after = "".join(lines[end:])

            replacement = new_content or ""
            updated_contents = before + replacement + after
            target.write_text(updated_contents, encoding="utf-8")

            # Smart verification - only verify if config says so
            if self.verification_config.should_verify("MODIFY", path):
                self._verify_written_file(
                    operation="MODIFY",
                    target=target,
                    expected_content=updated_contents,
                )
            else:
                LOGGER.debug("Skipping verification for MODIFY operation on %s", path)

            replaced_lines = end - start + 1
            message = (
                f"Replaced lines {start}-{end} ({replaced_lines} line(s)) in '{path}'."
            )
            LOGGER.info(message)
            if not replacement.endswith("\n") and replacement:
                LOGGER.debug("replace_file_lines inserted content without trailing newline for %s", path)
            LOGGER.debug("replace_file_lines replaced block:\n%s\nwith:\n%s", replaced_block, replacement)
            return message
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "Failed to replace lines %s-%s in %s: %s", start_line, end_line, path, exc
            )
            raise

    def delete_file(self, path: str) -> str:
        """Delete a workspace file."""
        LOGGER.info("🗑️ TOOL CALLED: delete_file(%s)", path)
        try:
            target = self._resolve_path(path)
            if not target.exists():
                raise FileVerificationError(
                    f"File '{path}' does not exist.",
                    {"file": path},
                )
            target.unlink()

            # Smart verification - only verify if config says so
            if self.verification_config.should_verify("DELETE", path):
                self._verify_deletion(target)
            else:
                LOGGER.debug("Skipping verification for DELETE operation on %s", path)

            LOGGER.info("✅ Deleted file: %s", target)
            return f"Successfully deleted '{path}'"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to delete file %s: %s", path, exc)
            raise

    def create_godot_material(
        self,
        path: str,
        albedo: str | None = None,
        normal: str | None = None,
        roughness: str | None = None,
        metallic: str | None = None,
        emission: str | None = None,
    ) -> str:
        """Create a Godot 4.x StandardMaterial3D .tres file."""
        LOGGER.info("ðŸ”§ TOOL CALLED: create_godot_material(%s)", path)
        try:
            target = self._resolve_path(path)
            if not target.name.endswith(".tres"):
                return f"Error: Material path '{path}' must end with .tres"

            textures = {
                "albedo_texture": albedo,
                "normal_map_texture": normal,
                "roughness_texture": roughness,
                "metallic_texture": metallic,
                "emission_texture": emission,
            }

            ext_resources = []
            resource_lines = []
            load_steps = 1
            texture_id = 1

            for prop, tex_path in textures.items():
                if tex_path:
                    try:
                        resolved_tex = self._resolve_path(tex_path)
                        if not resolved_tex.exists():
                            return f"Error: Texture path '{tex_path}' for '{prop}' does not exist."

                        uid = f"uid://{uuid.uuid4().hex[:12]}"
                        relative_tex_path = self._relative_path(resolved_tex)

                        ext_resources.append(
                            f'[ext_resource type="Texture2D" uid="{uid}" path="res://{relative_tex_path}" id="{texture_id}"]'
                        )
                        resource_lines.append(f'{prop} = ExtResource("{texture_id}")')
                        texture_id += 1
                        load_steps += 1
                    except PermissionError as exc:
                        return f"Error resolving texture path '{tex_path}': {exc}"


            tres_content = [
                f'[gd_resource type="StandardMaterial3D" load_steps={load_steps} format=3 uid="uid://{uuid.uuid4().hex[:12]}"]',
                "",
            ]
            tres_content.extend(ext_resources)
            tres_content.append("")
            tres_content.append("[resource]")
            tres_content.extend(resource_lines)

            final_content = "\n".join(tres_content) + "\n"

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(final_content, encoding="utf-8")
            size = len(final_content.encode("utf-8"))
            LOGGER.info("âœ… Created Godot material: %s (%d bytes)", target, size)
            return f"Successfully created Godot material '{path}' ({size} bytes)"

        except Exception as exc:
            LOGGER.exception("Failed to create Godot material %s: %s", path, exc)
            return f"Error creating Godot material '{path}': {exc}"

    # ------------------------------------------------------------------ #
    # File system helpers
    # ------------------------------------------------------------------ #
    def read_project_file(self, path: str) -> str:
        """Return file contents if the target resides inside the workspace."""
        LOGGER.info("ðŸ”§ TOOL CALLED: read_project_file(%s)", path)
        try:
            target = self._resolve_path(path)
            if not target.exists():
                LOGGER.warning("read_project_file missing path: %s", target)
                return f"Error: file '{path}' does not exist."

            # Check if this is a known binary file format
            extension = target.suffix.lower()
            if extension in BINARY_EXTENSIONS:
                LOGGER.info("Detected binary file format: %s", extension)
                return self._format_binary_file_metadata(target, extension)

            # Try to read as UTF-8 text, but catch binary files we didn't recognize
            LOGGER.debug("Reading file at %s", target)
            try:
                return target.read_text(encoding="utf-8")
            except UnicodeDecodeError as decode_err:
                LOGGER.warning(
                    "File %s appears to be binary (UnicodeDecodeError): %s",
                    target,
                    decode_err,
                )
                return self._format_binary_file_metadata(target, extension, is_unknown=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read file %s: %s", path, exc)
            return f"Error reading '{path}': {exc}"

    def list_project_files(
        self,
        directory: str = ".",
        extension: str = ".py",
        respect_gitignore: bool | None = None,
    ) -> dict[str, Any]:
        """List files that match the provided extension within the workspace.

        :param directory: Directory (relative to the workspace) to search in.
        :param extension: File extension filter (defaults to Python files).
        :param respect_gitignore: When True, apply .gitignore rules during traversal.
                                   When None, uses the default set by respect_gitignore tool.
        """
        # Use stored default if not explicitly provided
        if respect_gitignore is None:
            respect_gitignore = self._default_respect_gitignore

        LOGGER.info(
            "ðŸ”§ TOOL CALLED: list_project_files(directory=%s, extension=%s, respect_gitignore=%s)",
            directory,
            extension,
            respect_gitignore,
        )
        response: dict[str, Any] = {
            "directory": directory,
            "extension": extension,
            "respect_gitignore": respect_gitignore,
            "files": [],
            "count": 0,
        }
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                message = f"Directory does not exist: {directory}"
                LOGGER.warning("list_project_files base missing: %s", base)
                response["error"] = message
                return response

            self._ensure_gitignore_state()
            if respect_gitignore:
                base_relative = self._relative_path(base)
                if self._is_path_ignored(base_relative, is_dir=True):
                    message = f"Directory '{directory}' is ignored by .gitignore rules."
                    LOGGER.info(
                        "list_project_files skipping %s because it is ignored by .gitignore",
                        base_relative or ".",
                    )
                    response["error"] = message
                    return response

            suffix = extension if not extension or extension.startswith(".") else f".{extension}"
            LOGGER.debug("Scanning %s for *%s (workspace=%s)", base, suffix or "*", self.workspace_dir)
            files: list[str] = []
            try:
                for path in self._iter_workspace_files(base, respect_gitignore=respect_gitignore):
                    if suffix and path.suffix != suffix:
                        continue
                    files.append(self._relative_path(path))
            except Exception as exc:  # noqa: BLE001
                LOGGER.error(
                    "list_project_files traversal failed for %s (extension=%s, respect_gitignore=%s): %s",
                    base,
                    suffix or "*",
                    respect_gitignore,
                    exc,
                    exc_info=True,
                )
                response["error"] = f"Error listing files in '{directory}': {exc}"
                return response

            files.sort()
            response["files"] = files
            response["count"] = len(files)
            LOGGER.info(
                "list_project_files returning %d paths from %s (extension=%s)",
                response["count"],
                base,
                suffix or "*",
            )
            return response
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to list project files in %s: %s", directory, exc)
            response["error"] = f"Error listing files in '{directory}': {exc}"
            return response

    def search_in_files(
        self,
        pattern: str,
        directory: str = ".",
        file_extension: str = ".py",
    ) -> dict:
        """Search for a case-insensitive pattern within workspace files.

        Args:
            pattern: Search pattern (case-insensitive)
            directory: Directory to search in (default: ".")
            file_extension: File extension filter (default: ".py")

        Returns:
            Dictionary with format: {"matches": [{"file": str, "line_number": int, "content": str}], "total": int}
        """
        LOGGER.info("ðŸ”§ TOOL CALLED: search_in_files(%s)", pattern)
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

    def read_multiple_files(self, file_paths_json: str) -> dict:
        """Read multiple files and return structured results.

        Args:
            file_paths_json: A JSON string representing a list of file paths to read.

        Returns:
            Dictionary with file paths as keys and content/error info as values.
        """
        LOGGER.info("ðŸ”§ TOOL CALLED: read_multiple_files(%s)", file_paths_json)
        try:
            file_paths = json.loads(file_paths_json)
            if not isinstance(file_paths, list):
                return {"error": "Input must be a JSON array of strings."}
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON provided for file paths: {exc}"}

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
    # Context analysis helpers
    # ------------------------------------------------------------------ #
    def get_project_structure(self, directory: str = ".", max_depth: int = 2) -> dict[str, Any]:
        """Summarize the project layout for a given directory."""
        if isinstance(max_depth, str):
            try:
                max_depth = int(max_depth)
            except ValueError:
                LOGGER.warning("Invalid max_depth value '%s'; defaulting to 2", max_depth)
                max_depth = 2
        LOGGER.info(
            "ðŸ”§ TOOL CALLED: get_project_structure(directory=%s, max_depth=%s)",
            directory,
            max_depth,
        )
        summary: dict[str, Any] = {
            "root": directory,
            "directories": [],
            "files": [],
            "max_depth": max_depth,
        }
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                summary["error"] = f"Directory does not exist: {directory}"
                return summary

            stack: list[tuple[Path, int]] = [(base, 0)]
            while stack:
                current, depth = stack.pop()
                try:
                    entries = sorted(current.iterdir(), key=lambda entry: entry.name.lower())
                except (OSError, PermissionError) as exc:
                    LOGGER.debug("Unable to read %s: %s", current, exc)
                    continue

                rel_current = self._relative_path(current)
                if depth > 0 and rel_current:
                    summary["directories"].append(rel_current.replace("\\", "/"))

                if depth >= max_depth:
                    continue

                for entry in entries:
                    rel = self._relative_path(entry)
                    is_dir = entry.is_dir()
                    if self._is_path_ignored(rel, is_dir=is_dir):
                        continue
                    if is_dir:
                        stack.append((entry, depth + 1))
                    else:
                        summary["files"].append(rel.replace("\\", "/"))

            summary["directories"].sort()
            summary["files"].sort()
            summary["root"] = self._relative_path(base) or "."
            summary["total_items"] = len(summary["directories"]) + len(summary["files"])
            return summary
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to build project structure for %s: %s", directory, exc)
            summary["error"] = str(exc)
            return summary

    # ------------------------------------------------------------------ #
    # Asset discovery helpers
    # ------------------------------------------------------------------ #
    def verify_asset_paths(self, paths: Sequence[str] | str | None) -> dict[str, Any]:
        """Check whether each provided asset path exists within the workspace."""
        response: dict[str, Any] = {
            "paths": {},
            "existing": [],
            "missing": [],
            "requested": 0,
        }

        provided: list[str] = []
        if paths is None:
            response["error"] = "No asset paths provided."
            return response
        if isinstance(paths, str):
            provided = [paths]
        else:
            provided = [str(path) for path in paths if path is not None]

        normalized_inputs = [entry.strip() for entry in provided if entry.strip()]
        response["requested"] = len(normalized_inputs)
        LOGGER.info("?? TOOL CALLED: verify_asset_paths(count=%s)", response["requested"])
        if not normalized_inputs:
            response["error"] = "No asset paths provided."
            return response

        errors: dict[str, str] = {}
        for raw in normalized_inputs:
            key = raw.replace("\\", "/")
            exists = False
            try:
                resolved = self._resolve_path(raw)
                key = self._relative_path(resolved) or key
                exists = resolved.exists()
            except PermissionError as exc:
                LOGGER.warning("verify_asset_paths denied for %s: %s", raw, exc)
                errors[key] = str(exc)
            except OSError as exc:
                LOGGER.debug("verify_asset_paths error for %s: %s", raw, exc)
                errors[key] = str(exc)

            response["paths"][key] = bool(exists)
            if exists:
                response["existing"].append(key)
            else:
                response["missing"].append(key)

        response["existing"].sort()
        response["missing"].sort()
        if errors:
            response["errors"] = errors
        return response

    def list_project_assets(
        self,
        project_root: str = ".",
        subdirectory: str | None = None,
        respect_gitignore: bool | None = None,
    ) -> dict[str, Any]:
        """Return a categorized view of asset files beneath a project directory.

        :param project_root: Root directory to start searching from.
        :param subdirectory: Optional subdirectory within project_root.
        :param respect_gitignore: When True, apply .gitignore rules during traversal.
                                   When None, uses the default set by respect_gitignore tool.
        """
        # Use stored default if not explicitly provided
        if respect_gitignore is None:
            respect_gitignore = self._default_respect_gitignore

        LOGGER.info(
            "?? TOOL CALLED: list_project_assets(project_root=%s, subdirectory=%s, respect_gitignore=%s)",
            project_root,
            subdirectory,
            respect_gitignore,
        )
        response: dict[str, Any] = {
            "project_root": project_root or ".",
            "subdirectory": subdirectory,
            "scanned_path": project_root or ".",
            "respect_gitignore": respect_gitignore,
            "assets": {asset_type: [] for asset_type in ASSET_TYPE_EXTENSIONS},
            "counts": {asset_type: 0 for asset_type in ASSET_TYPE_EXTENSIONS},
            "total_assets": 0,
        }

        try:
            project_path = self._resolve_directory(project_root or ".")
            if not project_path.exists():
                response["error"] = f"Directory does not exist: {project_root or '.'}"
                return response

            target_path = project_path
            subdirectory_clean = (subdirectory or "").strip()
            if subdirectory_clean:
                combined = Path(project_root or ".") / subdirectory_clean
                target_path = self._resolve_directory(str(combined))

            if not target_path.exists():
                response["error"] = f"Directory does not exist: {subdirectory_clean or project_root or '.'}"
                response["scanned_path"] = self._relative_path(target_path) or "."
                return response

            response["project_root"] = self._relative_path(project_path) or "."
            response["scanned_path"] = self._relative_path(target_path) or "."

            for file_path in self._iter_workspace_files(target_path, respect_gitignore=respect_gitignore):
                if not file_path.is_file():
                    continue
                asset_type = self._classify_asset_extension(file_path.suffix)
                if not asset_type:
                    continue
                try:
                    stats = file_path.stat()
                except OSError as exc:
                    LOGGER.debug("Unable to stat %s: %s", file_path, exc)
                    continue

                entry = {
                    "path": self._relative_path(file_path),
                    "name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": stats.st_size,
                }
                modified = self._format_timestamp(stats.st_mtime)
                if modified:
                    entry["last_modified"] = modified

                response["assets"][asset_type].append(entry)

            for asset_type, items in response["assets"].items():
                items.sort(key=lambda item: item["path"])
                response["counts"][asset_type] = len(items)

            response["total_assets"] = sum(response["counts"].values())
            return response
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to list project assets: %s", exc)
            response["error"] = str(exc)
            return response

    def search_assets_by_pattern(
        self,
        pattern: str,
        file_type: str | None = None,
        directory: str = ".",
        respect_gitignore: bool | None = None,
    ) -> dict[str, Any]:
        """Search for asset files that match a glob pattern.

        :param pattern: Glob pattern to match asset file names/paths.
        :param file_type: Optional filter for specific asset type (meshes, textures, sounds).
        :param directory: Directory to search within.
        :param respect_gitignore: When True, apply .gitignore rules during traversal.
                                   When None, uses the default set by respect_gitignore tool.
        """
        # Use stored default if not explicitly provided
        if respect_gitignore is None:
            respect_gitignore = self._default_respect_gitignore

        response: dict[str, Any] = {
            "pattern": pattern,
            "file_type": file_type,
            "directory": directory,
            "respect_gitignore": respect_gitignore,
            "matches": [],
            "count": 0,
        }
        normalized_pattern = (pattern or "").strip()
        LOGGER.info(
            "?? TOOL CALLED: search_assets_by_pattern(pattern=%s, file_type=%s, directory=%s, respect_gitignore=%s)",
            normalized_pattern,
            file_type,
            directory,
            respect_gitignore,
        )

        if not normalized_pattern:
            response["error"] = "Search pattern is required."
            return response

        normalized_type = self._normalize_asset_type_name(file_type)
        if file_type and normalized_type is None:
            response["error"] = (
                "Unsupported file_type. Expected one of: meshes, textures, sounds."
            )
            return response

        try:
            base = self._resolve_directory(directory or ".")
            if not base.exists():
                response["error"] = f"Directory does not exist: {directory or '.'}"
                return response

            matches: list[dict[str, Any]] = []
            pattern_lower = normalized_pattern.lower()
            for file_path in self._iter_workspace_files(base, respect_gitignore=respect_gitignore):
                if not file_path.is_file():
                    continue

                asset_type = self._classify_asset_extension(file_path.suffix)
                if asset_type is None:
                    continue
                if normalized_type and asset_type != normalized_type:
                    continue

                rel_path = self._relative_path(file_path)
                rel_posix = rel_path.replace("\\", "/")
                if not (
                    fnmatch.fnmatch(rel_posix, normalized_pattern)
                    or fnmatch.fnmatch(rel_posix.lower(), pattern_lower)
                    or fnmatch.fnmatch(file_path.name, normalized_pattern)
                    or fnmatch.fnmatch(file_path.name.lower(), pattern_lower)
                ):
                    continue

                try:
                    stats = file_path.stat()
                except OSError as exc:
                    LOGGER.debug("Unable to stat %s: %s", file_path, exc)
                    continue

                entry = {
                    "path": rel_posix,
                    "extension": file_path.suffix.lower(),
                    "size_bytes": stats.st_size,
                    "type": asset_type,
                }
                modified = self._format_timestamp(stats.st_mtime)
                if modified:
                    entry["last_modified"] = modified
                matches.append(entry)

            matches.sort(key=lambda item: item["path"])
            response["matches"] = matches
            response["count"] = len(matches)
            if normalized_type:
                response["resolved_type"] = normalized_type
            return response
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to search assets with pattern %s: %s", pattern, exc)
            response["error"] = str(exc)
            return response

    def get_asset_metadata(self, asset_path: str) -> dict[str, Any]:
        """Return file metadata for a single asset path."""
        LOGGER.info("?? TOOL CALLED: get_asset_metadata(%s)", asset_path)
        normalized = (asset_path or "").strip()
        response: dict[str, Any] = {
            "path": normalized,
            "exists": False,
            "file_size": None,
            "extension": None,
            "last_modified": None,
            "full_path": None,
            "relative_path": None,
        }

        if not normalized:
            response["error"] = "Asset path is required."
            return response

        try:
            resolved = self._resolve_path(normalized)
        except PermissionError as exc:
            response["error"] = str(exc)
            return response

        response["relative_path"] = self._relative_path(resolved)
        response["full_path"] = str(resolved)
        response["extension"] = resolved.suffix.lower()

        if not resolved.exists():
            response["error"] = f"Asset not found: {normalized}"
            return response

        try:
            stats = resolved.stat()
        except OSError as exc:  # noqa: BLE001
            LOGGER.exception("Unable to stat asset %s: %s", asset_path, exc)
            response["error"] = f"Unable to read metadata: {exc}"
            return response

        response["exists"] = True
        response["file_size"] = stats.st_size
        response["last_modified"] = self._format_timestamp(stats.st_mtime)
        asset_type = self._classify_asset_extension(resolved.suffix)
        if asset_type:
            response["type"] = asset_type
        return response

    def detect_duplicate_code(self, min_lines: int = 5) -> dict[str, Any]:
        """Detect repeated code blocks across Python files."""

        return self.python_tools.detect_duplicate_code(min_lines=min_lines)

    def check_naming_conventions(self, directory: str = ".") -> dict[str, Any]:
        """Identify functions and classes that violate basic naming rules."""

        return self.python_tools.check_naming_conventions(directory=directory)

    def analyze_type_hints(self, directory: str = ".") -> dict[str, Any]:
        """Report functions missing parameter or return annotations."""

        return self.python_tools.analyze_type_hints(directory=directory)

    def inspect_docstrings(self, directory: str = ".", include_private: bool = False) -> dict[str, Any]:
        """List modules, classes, and functions missing docstrings."""

        return self.python_tools.inspect_docstrings(directory=directory, include_private=include_private)

    def get_function_signatures(self, file_path: str) -> dict[str, Any]:
        """Return simplified function signatures for a Python file."""

        return self.python_tools.get_function_signatures(file_path=file_path)

    def find_unused_imports(self, file_path: str) -> dict[str, Any]:
        """Detect unused imports in a Python file."""

        return self.python_tools.find_unused_imports(file_path=file_path)

    def get_code_metrics(self, directory: str = ".") -> dict[str, Any]:
        """Return aggregate code metrics for the target directory."""

        return self.python_tools.get_code_metrics(directory=directory)

    # ------------------------------------------------------------------ #
    # Git operations
    # ------------------------------------------------------------------ #
    def get_git_status(self) -> str:
        """Return the short git status for the current repository.

        Returns:
            Git status output or "clean" if no changes, or error message
        """
        return self.git_tools.get_git_status()

    def git_commit(self, message: str = "") -> str:
        """Commit all changes with the given message.

        If message is not provided or is empty, a specialized local AI will automatically
        generate a conventional commit message based on the staged changes.

        Args:
            message: Commit message (optional). If omitted, auto-generates using local AI.

        Returns:
            Success message or error message
        """
        return self.git_tools.git_commit(message)

    def git_push(self, remote: str = "origin", branch: str = "main") -> str:
        """Push commits to the remote repository.

        Args:
            remote: Remote name (default: "origin")
            branch: Branch name (default: "main")

        Returns:
            Success message or error message
        """
        return self.git_tools.git_push(remote=remote, branch=branch)

    def git_diff(self, file_path: str = "", staged: bool = False) -> str:
        """Show git diff for changes in the repository.

        Args:
            file_path: Optional specific file to show diff for
            staged: If True, show staged changes; otherwise show unstaged (default: False)

        Returns:
            String containing the diff output, or empty string if no changes
        """
        return self.git_tools.git_diff(file_path=file_path, staged=staged)

    def git_blame(self, file_path: str, line_number: int) -> dict:
        """Return author and commit metadata for a specific file line.

        Returns:
            Dictionary with blame info or error
        """
        return self.git_tools.git_blame(file_path=file_path, line_number=line_number)

    def create_new_branch(self, branch_name: str, start_point: str = "HEAD") -> dict:
        """Create and check out a new git branch based on start_point.

        Returns:
            Dictionary with success status and details
        """
        return self.git_tools.create_new_branch(branch_name=branch_name, start_point=start_point)

    # ------------------------------------------------------------------ #
    # Python tool operations
    # ------------------------------------------------------------------ #
    def run_tests(self, test_path: str = "tests/", verbose: bool = False) -> dict:
        """Run pytest on the codebase and return test results.

        Args:
            test_path: Path to tests directory or file (default: "tests/")
            verbose: Enable verbose output (default: False)

        Returns:
            Dictionary with keys: passed, failed, duration, output
        """

        return self.python_tools.run_tests(test_path=test_path, verbose=verbose)

    def lint_code(self, file_paths_json: str, directory: str = ".") -> dict:
        """Run pylint to catch errors and code quality issues.

        Args:
            file_paths_json: A JSON string representing a list of specific files to lint.
            directory: Directory to lint if file_paths_json is empty (default: ".")

        Returns:
            Dictionary with keys: errors (list), warnings (list), score (float), output (str)
        """

        return self.python_tools.lint_code(file_paths_json=file_paths_json, directory=directory)

    def install_package(self, package: str, version: str = "") -> str:
        """Install a Python package using pip.

        Args:
            package: Package name to install (required)
            version: Optional version constraint (e.g., ">=1.0.0")

        Returns:
            Success or error message as a string
        """

        return self.python_tools.install_package(package=package, version=version)

    def format_code(
        self,
        file_paths_json: str, directory: str = "."
    ) -> dict:
        """Format Python code using Black formatter.

        Args:
            file_paths_json: A JSON string representing a list of specific files to format.
            directory: Directory to format if file_paths_json is empty (default: ".")

        Returns:
            Dictionary with keys: formatted (count), errors (list), message (summary)
        """

        return self.python_tools.format_code(file_paths_json=file_paths_json, directory=directory)

    def get_function_definitions(self, file_path: str) -> dict:
        """Extract function signatures from a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary with 'functions' key containing list of function definitions
            Example: {"functions": [{"name": "generate_password", "params": ["length", "use_numbers"], "line": 5, "docstring": "..."}]}
        """

        return self.python_tools.get_function_definitions(file_path=file_path)

    def get_cyclomatic_complexity(self, file_path: str) -> dict:
        """Calculate cyclomatic complexity metrics for the provided Python file.

        Returns:
            Dictionary with complexity metrics
        """

        return self.python_tools.get_cyclomatic_complexity(file_path=file_path)

    def generate_test_file(
        self,
        source_file: str,
        tests_root: str = "tests",
        overwrite: bool = False,
    ) -> dict:
        """Create or extend a pytest test file with stubs for public callables in source_file.

        Returns:
            Dictionary with generation results
        """

        return self.python_tools.generate_test_file(source_file=source_file, tests_root=tests_root, overwrite=overwrite)








    # ------------------------------------------------------------------ #
    # Symbol analysis tools
    # ------------------------------------------------------------------ #
    def find_definition(self, symbol_name: str, search_directory: str = ".") -> dict:
        """Find where a symbol (class/function/variable) is defined.

        Args:
            symbol_name: Name of the symbol to search for
            search_directory: Directory to search recursively (default ".")

        Returns:
            Dictionary with keys: found, file, line, type, signature, docstring, context
        """

        return self.python_tools.find_definition(symbol_name=symbol_name, search_directory=search_directory)



    def find_usages(self, symbol_name: str, search_directory: str = ".") -> dict:
        """Find all usages of a symbol in Python files.

        Args:
            symbol_name: Name of the symbol to search for
            search_directory: Directory to search recursively (default ".")

        Returns:
            Dictionary with keys: total_usages, files_count, usages (list of usage dicts)
        """

        return self.python_tools.find_usages(symbol_name=symbol_name, search_directory=search_directory)


    def get_imports(self, file_path: str) -> dict:
        """Extract and categorize all imports from a Python file.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary with keys: stdlib, third_party, local, import_details
        """

        return self.python_tools.get_imports(file_path=file_path)




    def get_dependency_graph(self, symbol_name: str, search_directory: str = ".") -> dict:
        """Build a lightweight dependency graph for a symbol across the project.

        Returns:
            Dictionary with dependency graph data
        """

        return self.python_tools.get_dependency_graph(symbol_name=symbol_name, search_directory=search_directory)

    def get_class_hierarchy(self, class_name: str, search_directory: str = ".") -> dict:
        """Return inheritance details for a class, including parents and subclasses.

        Returns:
            Dictionary with class hierarchy data
        """

        return self.python_tools.get_class_hierarchy(class_name=class_name, search_directory=search_directory)

    def safe_rename_symbol(
        self,
        file_path: str,
        symbol_name: str,
        new_name: str,
        project_root: str = "",
    ) -> dict:
        """Perform a project-wide, refactor-aware rename using Rope.

        Args:
            file_path: Path to the file containing the symbol
            symbol_name: Name of the symbol to rename
            new_name: New name for the symbol
            project_root: Project root directory (empty string to use workspace)

        Returns:
            Dictionary with rename results
        """

        return self.python_tools.safe_rename_symbol(file_path=file_path, symbol_name=symbol_name, new_name=new_name, project_root=project_root)













    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _format_timestamp(self, timestamp: float | None) -> str | None:
        """Return an ISO-8601 timestamp string for the provided epoch value."""
        if timestamp is None:
            return None
        try:
            return (
                datetime.fromtimestamp(timestamp, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (OSError, OverflowError, ValueError) as exc:
            LOGGER.debug("Unable to format timestamp %s: %s", timestamp, exc)
            return None

    def _normalize_asset_type_name(self, asset_type: str | None) -> str | None:
        """Normalize asset type aliases to canonical keys."""
        if not asset_type:
            return None
        normalized = asset_type.strip().lower()
        if not normalized:
            return None
        return ASSET_TYPE_ALIASES.get(normalized)

    def _classify_asset_extension(self, extension: str | None) -> str | None:
        """Return the asset category for a file extension."""
        if not extension:
            return None
        return ASSET_EXTENSION_LOOKUP.get(extension.lower())

    def _format_binary_file_metadata(
        self,
        file_path: Path,
        extension: str,
        is_unknown: bool = False,
    ) -> str:
        """Format metadata for a binary file that cannot be read as UTF-8 text.

        :param file_path: Path to the binary file
        :param extension: File extension (e.g., '.fbx')
        :param is_unknown: True if this is an unknown binary format detected via UnicodeDecodeError
        :return: JSON-formatted string with file metadata
        """
        stat = file_path.stat()
        size_bytes = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        # Human-readable file size
        size_human = self._format_file_size(size_bytes)

        # Get file type description
        if extension in BINARY_FILE_DESCRIPTIONS:
            file_type = BINARY_FILE_DESCRIPTIONS[extension]
        elif is_unknown:
            file_type = f"Unknown Binary File ({extension or 'no extension'})"
        else:
            file_type = f"Binary File ({extension or 'no extension'})"

        # Determine helpful context message based on file type
        if extension in {".fbx", ".blend", ".obj", ".gltf", ".glb", ".dae"}:
            context = "This is a 3D mesh/model file used by game engines like Godot or Unity."
        elif extension in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tga", ".webp"}:
            context = "This is an image/texture file used for graphics and visual assets."
        elif extension in {".wav", ".mp3", ".ogg", ".flac"}:
            context = "This is an audio file used for sound effects or music."
        elif extension in {".mp4", ".avi", ".mov", ".webm"}:
            context = "This is a video file."
        elif extension in {".zip", ".tar", ".gz", ".7z", ".rar"}:
            context = "This is a compressed archive file containing other files."
        elif extension in {".exe", ".dll", ".so", ".dylib"}:
            context = "This is a compiled executable or library file."
        elif is_unknown:
            context = "File could not be decoded as UTF-8 text - it appears to be binary data."
        else:
            context = "Binary file - content not readable as text."

        metadata = {
            "type": "binary_file",
            "extension": extension or "(no extension)",
            "file_type": file_type,
            "size_bytes": size_bytes,
            "size_human": size_human,
            "last_modified": modified.isoformat(),
            "message": f"Binary file - content not readable as text. {context}",
            "is_unknown_binary": is_unknown,
        }

        return json.dumps(metadata, indent=2)

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.

        :param size_bytes: Size in bytes
        :return: Human-readable string (e.g., '239.9 KB', '1.2 MB')
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

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

    def _iter_workspace_files(
        self,
        root: Path,
        *,
        respect_gitignore: bool = True,
    ) -> Iterator[Path]:
        """Yield workspace files beneath `root`, optionally respecting .gitignore rules."""
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
                if self._is_filtered_entry(relative, is_dir=is_dir):
                    LOGGER.debug("Skipping filtered path: %s", relative)
                    continue

                if respect_gitignore and self._is_path_ignored(relative, is_dir=is_dir):
                    LOGGER.debug("Skipping .gitignore-matched path: %s", relative)
                    continue

                if is_dir:
                    stack.append(entry)
                    continue

                yield entry

    def _is_filtered_entry(self, relative_path: str, *, is_dir: bool) -> bool:
        """Return True for paths that should always be filtered out."""
        if not relative_path or relative_path in {".", ""}:
            return False

        normalized = relative_path.strip("/")
        if not normalized:
            return False

        parts = [part for part in normalized.split("/") if part]
        if not parts:
            return False

        # Filter when any parent directory is in the blocked set.
        parent_segments = parts if is_dir else parts[:-1]
        if any(segment in DEFAULT_FILTERED_DIR_NAMES for segment in parent_segments):
            return True

        current_name = parts[-1]
        if current_name in DEFAULT_FILTERED_DIR_NAMES:
            return True

        if is_dir:
            return False

        if current_name in DEFAULT_FILTERED_FILE_NAMES:
            return True

        suffix = Path(current_name).suffix.lower()
        return suffix in DEFAULT_FILTERED_FILE_SUFFIXES
