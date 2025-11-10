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
        LOGGER.info("🔧 TOOL CALLED: respect_gitignore(enabled=%s)", enabled)
        try:
            self._default_respect_gitignore = enabled
            state = "enabled" if enabled else "disabled"
            message = f"Gitignore filtering {state}. File operations will {'respect' if enabled else 'ignore'} .gitignore rules by default."
            LOGGER.info("✅ respect_gitignore state updated: %s", state)
            return message
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to update respect_gitignore state: %s", exc)
            return f"Error updating gitignore state: {exc}"

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
        LOGGER.info("🔧 TOOL CALLED: create_godot_material(%s)", path)
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
            LOGGER.info("✅ Created Godot material: %s (%d bytes)", target, size)
            return f"Successfully created Godot material '{path}' ({size} bytes)"

        except Exception as exc:
            LOGGER.exception("Failed to create Godot material %s: %s", path, exc)
            return f"Error creating Godot material '{path}': {exc}"

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
            "🔧 TOOL CALLED: list_project_files(directory=%s, extension=%s, respect_gitignore=%s)",
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

    def read_multiple_files(self, file_paths_json: str) -> dict:
        """Read multiple files and return structured results.

        Args:
            file_paths_json: A JSON string representing a list of file paths to read.

        Returns:
            Dictionary with file paths as keys and content/error info as values.
        """
        LOGGER.info("🔧 TOOL CALLED: read_multiple_files(%s)", file_paths_json)
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
            "🔧 TOOL CALLED: get_project_structure(directory=%s, max_depth=%s)",
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
        LOGGER.info("🔧 TOOL CALLED: detect_duplicate_code(min_lines=%s)", min_lines)
        duplicates: dict[str, dict[str, Any]] = {}
        try:
            python_files = [
                path for path in self._iter_workspace_files(self.workspace_dir) if path.suffix == ".py"
            ]
            for path in python_files:
                try:
                    source = path.read_text(encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Unable to read %s while detecting duplicates: %s", path, exc)
                    continue
                try:
                    tree = ast.parse(source, filename=str(path), type_comments=True)
                except SyntaxError as exc:
                    LOGGER.debug("Skipping %s due to syntax error: %s", path, exc)
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        continue
                    segment = ast.get_source_segment(source, node)
                    if not segment:
                        continue
                    normalized_lines = [line.strip() for line in segment.splitlines() if line.strip()]
                    line_count = len(normalized_lines)
                    if line_count < min_lines:
                        continue
                    normalized = "\n".join(normalized_lines)
                    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    payload = duplicates.setdefault(
                        digest,
                        {
                            "preview": "\n".join(normalized_lines[: min(8, line_count)]),
                            "line_count": line_count,
                            "occurrences": [],
                        },
                    )
                    payload["occurrences"].append(
                        {
                            "file": self._relative_path(path),
                            "symbol": getattr(node, "name", None),
                            "start_line": getattr(node, "lineno", None),
                            "end_line": getattr(node, "end_lineno", None),
                        }
                    )

            groups = [
                {
                    "preview": data["preview"],
                    "line_count": data["line_count"],
                    "occurrences": data["occurrences"],
                }
                for data in duplicates.values()
                if len(data["occurrences"]) > 1
            ]
            return {"total_groups": len(groups), "duplicates": groups}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to detect duplicate code: %s", exc)
            return {"total_groups": 0, "duplicates": [], "error": str(exc)}

    def check_naming_conventions(self, directory: str = ".") -> dict[str, Any]:
        """Identify functions and classes that violate basic naming rules."""
        LOGGER.info("🔧 TOOL CALLED: check_naming_conventions(%s)", directory)
        snake_case = re.compile(r"^[a-z_][a-z0-9_]*$")
        pascal_case = re.compile(r"^[A-Z][A-Za-z0-9]+$")
        issues: dict[str, list[dict[str, Any]]] = {
            "functions": [],
            "classes": [],
        }
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                return {"error": f"Directory does not exist: {directory}", **issues}
            python_files = [path for path in self._iter_workspace_files(base) if path.suffix == ".py"]

            for path in python_files:
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except SyntaxError as exc:
                    LOGGER.debug("Skipping %s due to syntax error: %s", path, exc)
                    continue

                for node in ast.walk(tree):
                    rel_path = self._relative_path(path)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = node.name
                        if name.startswith("_") or snake_case.match(name) or name.startswith("__"):
                            continue
                        issues["functions"].append(
                            {"name": name, "file": rel_path, "line": node.lineno}
                        )
                    elif isinstance(node, ast.ClassDef):
                        if pascal_case.match(node.name):
                            continue
                        issues["classes"].append(
                            {"name": node.name, "file": rel_path, "line": node.lineno}
                        )

            return {
                "invalid_functions": issues["functions"],
                "invalid_classes": issues["classes"],
                "checked_files": len(python_files),
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to validate naming conventions: %s", exc)
            return {"error": str(exc), "invalid_functions": [], "invalid_classes": [], "checked_files": 0}

    def analyze_type_hints(self, directory: str = ".") -> dict[str, Any]:
        """Report functions missing parameter or return annotations."""
        LOGGER.info("🔧 TOOL CALLED: analyze_type_hints(%s)", directory)
        findings: list[dict[str, Any]] = []
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                return {"error": f"Directory does not exist: {directory}", "missing_annotations": []}

            python_files = [path for path in self._iter_workspace_files(base) if path.suffix == ".py"]
            for path in python_files:
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), type_comments=True)
                except SyntaxError as exc:
                    LOGGER.debug("Skipping %s for type hint analysis: %s", path, exc)
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    missing_params: list[str] = []
                    for arg in getattr(node.args, "posonlyargs", []):
                        if arg.annotation is None:
                            missing_params.append(arg.arg)
                    for arg in node.args.args:
                        if arg.arg in {"self", "cls"}:
                            continue
                        if arg.annotation is None:
                            missing_params.append(arg.arg)
                    for arg in node.args.kwonlyargs:
                        if arg.annotation is None:
                            missing_params.append(arg.arg)
                    vararg = node.args.vararg
                    if vararg and vararg.annotation is None:
                        missing_params.append(f"*{vararg.arg}")
                    kwarg = node.args.kwarg
                    if kwarg and kwarg.annotation is None:
                        missing_params.append(f"**{kwarg.arg}")

                    missing_return = node.returns is None
                    if missing_params or missing_return:
                        findings.append(
                            {
                                "function": node.name,
                                "file": self._relative_path(path),
                                "line": node.lineno,
                                "missing_parameters": missing_params,
                                "missing_return": missing_return,
                            }
                        )

            return {"missing_annotations": findings, "checked_files": len(python_files)}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to analyze type hints: %s", exc)
            return {"error": str(exc), "missing_annotations": [], "checked_files": 0}

    def inspect_docstrings(self, directory: str = ".", include_private: bool = False) -> dict[str, Any]:
        """List modules, classes, and functions missing docstrings."""
        LOGGER.info(
            "🔧 TOOL CALLED: inspect_docstrings(directory=%s, include_private=%s)",
            directory,
            include_private,
        )
        missing: list[dict[str, Any]] = []
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                return {"error": f"Directory does not exist: {directory}", "missing": []}

            python_files = [path for path in self._iter_workspace_files(base) if path.suffix == ".py"]
            for path in python_files:
                try:
                    source = path.read_text(encoding="utf-8")
                    tree = ast.parse(source, filename=str(path))
                except SyntaxError as exc:
                    LOGGER.debug("Skipping %s for docstring inspection: %s", path, exc)
                    continue

                module_doc = ast.get_docstring(tree)
                if not module_doc:
                    missing.append({"type": "module", "file": self._relative_path(path), "line": 1})

                for node in ast.walk(tree):
                    name = getattr(node, "name", "")
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if not include_private and name.startswith("_"):
                            continue
                        docstring = ast.get_docstring(node)
                        if docstring:
                            continue
                        missing.append(
                            {
                                "type": "function" if not isinstance(node, ast.ClassDef) else "class",
                                "name": name,
                                "file": self._relative_path(path),
                                "line": node.lineno,
                            }
                        )

            return {"missing": missing, "checked_files": len(python_files)}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to inspect docstrings: %s", exc)
            return {"error": str(exc), "missing": [], "checked_files": 0}

    def get_function_signatures(self, file_path: str) -> dict[str, Any]:
        """Return simplified function signatures for a Python file."""
        LOGGER.info("🔧 TOOL CALLED: get_function_signatures(%s)", file_path)
        details = self.get_function_definitions(file_path)
        functions = details.get("functions") or []
        signatures = []
        for entry in functions:
            params = entry.get("params") or []
            signature = f"{entry.get('name')}({', '.join(params)})"
            signatures.append(
                {
                    "name": entry.get("name"),
                    "signature": signature,
                    "line": entry.get("line"),
                    "docstring": entry.get("docstring", ""),
                }
            )
        response: dict[str, Any] = {"signatures": signatures, "total": len(signatures)}
        if details.get("error"):
            response["error"] = details["error"]
        return response

    def find_unused_imports(self, file_path: str) -> dict[str, Any]:
        """Detect unused imports in a Python file."""
        LOGGER.info("🔧 TOOL CALLED: find_unused_imports(%s)", file_path)
        try:
            target = self._resolve_python_path(file_path)
        except Exception as exc:  # noqa: BLE001
            return {"unused": [], "error": str(exc)}

        if not target.exists():
            return {"unused": [], "error": f"File does not exist: {file_path}"}

        try:
            source = target.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(target), type_comments=True)
        except SyntaxError as exc:
            return {"unused": [], "error": f"Syntax error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to analyze imports for %s: %s", file_path, exc)
            return {"unused": [], "error": str(exc)}

        imported_names: dict[str, dict[str, Any]] = {}

        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
                for alias in node.names:
                    local = alias.asname or alias.name.split(".")[0]
                    imported_names[local] = {
                        "module": alias.name,
                        "line": node.lineno,
                    }

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # type: ignore[override]
                if node.module == "*" or any(alias.name == "*" for alias in node.names):
                    return
                for alias in node.names:
                    local = alias.asname or alias.name
                    imported_names[local] = {
                        "module": f"{node.module}.{alias.name}" if node.module else alias.name,
                        "line": node.lineno,
                    }

        class UsageVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.used: set[str] = set()

            def visit_Name(self, node: ast.Name) -> None:  # type: ignore[override]
                if isinstance(node.ctx, ast.Load):
                    self.used.add(node.id)
                self.generic_visit(node)

        ImportVisitor().visit(tree)
        usage = UsageVisitor()
        usage.visit(tree)

        unused = [
            {"name": name, "module": meta["module"], "line": meta["line"]}
            for name, meta in imported_names.items()
            if name not in usage.used
        ]
        return {"unused": unused, "total_imports": len(imported_names)}

    def get_code_metrics(self, directory: str = ".") -> dict[str, Any]:
        """Return aggregate code metrics for the target directory."""
        LOGGER.info("🔧 TOOL CALLED: get_code_metrics(%s)", directory)
        try:
            base = self._resolve_directory(directory)
            if not base.exists():
                return {"error": f"Directory does not exist: {directory}"}

            python_files = [path for path in self._iter_workspace_files(base) if path.suffix == ".py"]
            total_lines = 0
            function_count = 0
            class_count = 0
            todo_comments = 0

            for path in python_files:
                try:
                    content = path.read_text(encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("Skipping %s for code metrics: %s", path, exc)
                    continue
                total_lines += len(content.splitlines())
                todo_comments += sum(1 for line in content.splitlines() if "TODO" in line.upper())

                try:
                    tree = ast.parse(content, filename=str(path))
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_count += 1
                    elif isinstance(node, ast.ClassDef):
                        class_count += 1

            average_lines = (
                round(total_lines / len(python_files), 2) if python_files else 0
            )
            return {
                "python_files": len(python_files),
                "total_lines": total_lines,
                "average_lines_per_file": average_lines,
                "function_count": function_count,
                "class_count": class_count,
                "todo_comments": todo_comments,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to compute code metrics for %s: %s", directory, exc)
            return {"error": str(exc)}

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

    def git_blame(self, file_path: str, line_number: int) -> dict:
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

    def create_new_branch(self, branch_name: str, start_point: str = "HEAD") -> dict:
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
    def run_tests(self, test_path: str = "tests/", verbose: bool = False) -> dict:
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

    def lint_code(self, file_paths_json: str, directory: str = ".") -> dict:
        """Run pylint to catch errors and code quality issues.

        Args:
            file_paths_json: A JSON string representing a list of specific files to lint.
            directory: Directory to lint if file_paths_json is empty (default: ".")

        Returns:
            Dictionary with keys: errors (list), warnings (list), score (float), output (str)
        """
        LOGGER.info("🔧 TOOL CALLED: lint_code(%s)", file_paths_json or directory)
        file_paths = []
        if file_paths_json:
            try:
                file_paths = json.loads(file_paths_json)
                if not isinstance(file_paths, list):
                    return {"errors": ["Input must be a JSON array of strings."], "warnings": [], "score": 0.0}
            except json.JSONDecodeError as exc:
                return {"errors": [f"Invalid JSON: {exc}"], "warnings": [], "score": 0.0}

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
        file_paths_json: str, directory: str = "."
    ) -> dict:
        """Format Python code using Black formatter.

        Args:
            file_paths_json: A JSON string representing a list of specific files to format.
            directory: Directory to format if file_paths_json is empty (default: ".")

        Returns:
            Dictionary with keys: formatted (count), errors (list), message (summary)
        """
        LOGGER.info("🔧 TOOL CALLED: format_code(%s)", file_paths_json or directory)
        file_paths = []
        if file_paths_json:
            try:
                file_paths = json.loads(file_paths_json)
                if not isinstance(file_paths, list):
                    return {"formatted": 0, "errors": ["Input must be a JSON array of strings."]}
            except json.JSONDecodeError as exc:
                return {"formatted": 0, "errors": [f"Invalid JSON: {exc}"]}

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

    def get_function_definitions(self, file_path: str) -> dict:
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

    def get_cyclomatic_complexity(self, file_path: str) -> dict:
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
    ) -> dict:
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
    def find_definition(self, symbol_name: str, search_directory: str = ".") -> dict:
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

    def find_usages(self, symbol_name: str, search_directory: str = ".") -> dict:
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

    def get_imports(self, file_path: str) -> dict:
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

    def get_dependency_graph(self, symbol_name: str, search_directory: str = ".") -> dict:
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

    def get_class_hierarchy(self, class_name: str, search_directory: str = ".") -> dict:
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
