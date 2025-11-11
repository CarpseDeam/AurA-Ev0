# src/aura/tools/godot_tools.py
"""Godot Engine utilities exposed as analyst tools."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

logger = logging.getLogger(__name__)

_EXT_RESOURCE_PATTERN = re.compile(r'ExtResource\(\s*"([^"]+)"\s*\)')
_SUB_RESOURCE_PATTERN = re.compile(r'SubResource\(\s*"([^"]+)"\s*\)')


@dataclass
class SectionProperty:
    name: str
    value_lines: list[str]


@dataclass
class SectionEntry:
    kind: Literal["property", "raw"]
    data: SectionProperty | str


@dataclass
class TSCNSection:
    section_type: str
    attributes: dict[str, str]
    entries: list[SectionEntry] = field(default_factory=list)
    raw_header: str | None = None
    header_dirty: bool = False
    leading_lines: list[str] = field(default_factory=list)

    def render_header(self) -> str:
        if not self.header_dirty and self.raw_header:
            return self.raw_header
        attr_str = " ".join(f"{key}={value}" for key, value in self.attributes.items())
        return f"[{self.section_type}{(' ' + attr_str) if attr_str else ''}]"

    def get_property(self, name: str) -> SectionProperty | None:
        for entry in self.entries:
            if entry.kind == "property" and entry.data.name == name:
                return entry.data
        return None

    def set_property(self, name: str, value_lines: list[str]) -> None:
        existing = self.get_property(name)
        if existing:
            existing.value_lines = value_lines
            return
        self.entries.append(SectionEntry(kind="property", data=SectionProperty(name, value_lines)))


@dataclass
class ParsedScene:
    path: Path
    sections: list[TSCNSection]
    trailing_lines: list[str] = field(default_factory=list)


@dataclass
class SceneNode:
    name: str
    node_type: str
    parent: str
    path: str
    attributes: dict[str, str]
    properties: dict[str, str]
    section: TSCNSection


def _get_agent_workspace() -> Path | None:
    """Return the agent's configured working directory, if available."""
    try:
        from aura.state import get_app_state  # Local import to avoid Qt dependency at import.
    except Exception:  # noqa: BLE001 - fallback when state module is unavailable.
        return None

    app_state = get_app_state()
    working_dir = getattr(app_state, "working_directory", "") if app_state else ""
    if not working_dir:
        return None

    try:
        return Path(working_dir).expanduser().resolve()
    except OSError as exc:
        logger.warning("Unable to resolve working directory %s: %s", working_dir, exc)
    return None


def _resolve_scene_file(scene_path: str) -> Path:
    """Resolve a Godot scene path so it resides within the active workspace."""
    normalized = (scene_path or "").strip()
    if not normalized:
        raise ValueError("Scene path must be provided.")
    if normalized.lower().startswith("res://"):
        normalized = normalized[6:]
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("Scene path must be provided.")

    candidate = Path(normalized).expanduser()
    workspace = _get_agent_workspace()
    base_dir = workspace or Path.cwd()

    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_dir / candidate).resolve()

    if workspace:
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                f"Scene path '{scene_path}' is outside the working directory: {workspace}"
            ) from exc
    return resolved


def _resolve_workspace_path(relative_path: str) -> Path | None:
    """Resolve a workspace-relative path without enforcing file existence."""
    normalized = (relative_path or "").strip()
    if not normalized:
        return None
    if normalized.lower().startswith("res://"):
        normalized = normalized[6:]
    workspace = _get_agent_workspace()
    base_dir = workspace or Path.cwd()
    candidate = (base_dir / normalized).resolve()
    if workspace:
        try:
            candidate.relative_to(workspace)
        except ValueError:
            return None
    return candidate


def _split_property_value(value: str) -> list[str]:
    return value.splitlines() or [value]


def _strip_quotes(value: str | None) -> str:
    if not value:
        return ""
    trimmed = value.strip()
    if len(trimmed) >= 2 and trimmed[0] == '"' and trimmed[-1] == '"':
        return trimmed[1:-1]
    return trimmed


def _normalize_scene_parent(value: str | None) -> str:
    if not value or value == ".":
        return "."
    if value.startswith("/root/"):
        return value[len("/root/") :]
    if value.startswith("./"):
        return value[2:]
    return value


def _normalize_input_node_path(path: str | None, root_name: str | None) -> str:
    if not path:
        return "."
    candidate = path.strip()
    if candidate in {".", "/", "/root"}:
        return "."
    if root_name and candidate == root_name:
        return "."
    if candidate.startswith("/root/"):
        candidate = candidate[len("/root/") :]
    if candidate.startswith("./"):
        candidate = candidate[2:]
    return candidate or "."


def _looks_like_godot_expression(value: str) -> bool:
    tokens = (
        "Vector",
        "Transform",
        "Basis",
        "Quat",
        "Color",
        "NodePath",
        "ExtResource",
        "SubResource",
        "Rect2",
        "Plane",
        "Packed",
        "RID",
        "Resource",
    )
    return (
        not value
        or value.startswith(tokens)
        or value.startswith("{")
        or value.startswith("[")
        or value.startswith(("\"", "'"))
        or (value.endswith(")") and "(" in value)
        or value in {"true", "false", "null"}
    )


def _quote_godot_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )
    return f'"{escaped}"'


def _format_dict_key(key: Any) -> str:
    if isinstance(key, str):
        return _quote_godot_string(key)
    return str(key)


def _format_inline_value(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if _looks_like_godot_expression(stripped):
            return stripped
        return _quote_godot_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        inner = ", ".join(_format_inline_value(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, Mapping):
        inner = ", ".join(
            f"{_format_dict_key(key)}: {_format_inline_value(val)}" for key, val in value.items()
        )
        return f"{{{inner}}}"
    return _quote_godot_string(str(value))


def _format_property_value(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('"""') and stripped.count('"""') >= 2:
            return stripped.splitlines()
        if "\n" in value and not stripped.startswith('"""'):
            lines = ["\"\"\""]
            lines.extend(value.splitlines())
            lines.append("\"\"\"")
            return lines
        if _looks_like_godot_expression(stripped):
            return _split_property_value(stripped)
        return [_quote_godot_string(value)]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        inner = ", ".join(_format_inline_value(item) for item in value)
        return [f"[{inner}]"]
    if isinstance(value, Mapping):
        inner = ", ".join(
            f"{_format_dict_key(key)}: {_format_inline_value(val)}" for key, val in value.items()
        )
        return [f"{{{inner}}}"]
    return [_quote_godot_string(str(value))]


def _parse_section_header(header_line: str) -> tuple[str, dict[str, str]]:
    stripped = header_line.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        raise ValueError(f"Invalid TSCN section header: {header_line}")
    inner = stripped[1:-1].strip()
    if " " in inner:
        section_type, attr_str = inner.split(" ", 1)
    else:
        section_type, attr_str = inner, ""
    attributes: dict[str, str] = {}
    if attr_str:
        for match in re.finditer(r'(\w+)=(".*?"|\S+)', attr_str):
            attributes[match.group(1)] = match.group(2)
    return section_type, attributes


def _parse_tscn_file(scene_path: Path) -> ParsedScene:
    text = scene_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    sections: list[TSCNSection] = []
    trailing_buffer: list[str] = []
    current_section: TSCNSection | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and not stripped.startswith("[["):
            section_type, attributes = _parse_section_header(stripped)
            current_section = TSCNSection(
                section_type=section_type,
                attributes=attributes,
                raw_header=stripped,
                leading_lines=trailing_buffer,
            )
            trailing_buffer = []
            sections.append(current_section)
            i += 1
            continue

        if current_section is None:
            trailing_buffer.append(line)
            i += 1
            continue

        if "=" in line and not stripped.startswith(";"):
            key_part, value_part = line.split("=", 1)
            key = key_part.strip()
            value = value_part.strip()
            value_lines = [value]
            if value.startswith('"""') and not (value.endswith('"""') and value != '"""'):
                while i + 1 < len(lines):
                    i += 1
                    continuation = lines[i]
                    value_lines.append(continuation)
                    if continuation.strip().endswith('"""'):
                        break
            current_section.entries.append(
                SectionEntry(kind="property", data=SectionProperty(name=key, value_lines=value_lines))
            )
        else:
            current_section.entries.append(SectionEntry(kind="raw", data=line))
        i += 1

    return ParsedScene(path=scene_path, sections=sections, trailing_lines=trailing_buffer)


def _serialize_tscn(parsed: ParsedScene) -> str:
    output_lines: list[str] = []
    for section in parsed.sections:
        if section.leading_lines:
            output_lines.extend(section.leading_lines)
        output_lines.append(section.render_header())
        for entry in section.entries:
            if entry.kind == "raw":
                output_lines.append(entry.data)
                continue
            prop = entry.data
            first_line = prop.value_lines[0] if prop.value_lines else ""
            output_lines.append(f"{prop.name} = {first_line}")
            if len(prop.value_lines) > 1:
                output_lines.extend(prop.value_lines[1:])
    if parsed.trailing_lines:
        output_lines.extend(parsed.trailing_lines)
    serialized = "\n".join(output_lines)
    if not serialized.endswith("\n"):
        serialized += "\n"
    return serialized


def _write_scene(parsed: ParsedScene) -> None:
    serialized = _serialize_tscn(parsed)
    parsed.path.write_text(serialized, encoding="utf-8")


def _collect_section_properties(section: TSCNSection) -> dict[str, str]:
    properties: dict[str, str] = {}
    for entry in section.entries:
        if entry.kind != "property":
            continue
        properties[entry.data.name] = "\n".join(entry.data.value_lines)
    return properties


def _build_nodes(parsed: ParsedScene) -> tuple[list[SceneNode], dict[str, SceneNode], str | None]:
    nodes: list[SceneNode] = []
    node_map: dict[str, SceneNode] = {}
    root_name: str | None = None

    for section in parsed.sections:
        if section.section_type != "node":
            continue
        name = _strip_quotes(section.attributes.get("name"))
        node_type = _strip_quotes(section.attributes.get("type"))
        parent_raw = section.attributes.get("parent")
        parent = _normalize_scene_parent(parent_raw)

        if root_name is None:
            path = "."
            parent = "."
            root_name = name or "Root"
        else:
            if parent in {".", ""}:
                path = name
            else:
                path = f"{parent}/{name}"

        node = SceneNode(
            name=name or "",
            node_type=node_type or "",
            parent=parent or ".",
            path=path or ".",
            attributes={key: _strip_quotes(value) for key, value in section.attributes.items()},
            properties=_collect_section_properties(section),
            section=section,
        )
        nodes.append(node)

        path_key = node.path or "."
        node_map[path_key] = node
        if node.path == ".":
            node_map["."] = node
            node_map[node.name] = node
        else:
            node_map[f"./{node.path}"] = node

    return nodes, node_map, root_name


def _build_node_tree(nodes: Sequence[SceneNode]) -> dict[str, Any] | None:
    lookup: dict[str, dict[str, Any]] = {}
    root_payload: dict[str, Any] | None = None

    for node in nodes:
        payload = {
            "name": node.name,
            "type": node.node_type,
            "path": node.path,
            "parent_path": node.parent,
            "attributes": node.attributes,
            "properties": node.properties,
            "children": [],
        }
        lookup_key = node.path if node.path != "." else "."
        lookup[lookup_key] = payload
        if node.path == ".":
            root_payload = payload

    for node in nodes:
        if node.path == ".":
            continue
        parent_key = node.parent or "."
        parent_payload = lookup.get(parent_key)
        if parent_payload:
            parent_payload["children"].append(lookup[node.path])

    return root_payload


def _collect_ext_resources(parsed: ParsedScene) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for section in parsed.sections:
        if section.section_type != "ext_resource":
            continue
        resource = {
            "attributes": {key: _strip_quotes(value) for key, value in section.attributes.items()},
            "properties": _collect_section_properties(section),
        }
        resources.append(resource)
    return resources


def _collect_ext_resources_map(parsed: ParsedScene) -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for section in parsed.sections:
        if section.section_type != "ext_resource":
            continue
        identifier = _strip_quotes(section.attributes.get("id"))
        resources[identifier] = {
            "path": _strip_quotes(section.attributes.get("path")),
            "type": _strip_quotes(section.attributes.get("type")),
        }
    return resources


def _collect_sub_resources(parsed: ParsedScene) -> list[dict[str, Any]]:
    resources: list[dict[str, Any]] = []
    for section in parsed.sections:
        if section.section_type != "sub_resource":
            continue
        resources.append(
            {
                "attributes": {key: _strip_quotes(value) for key, value in section.attributes.items()},
                "properties": _collect_section_properties(section),
            }
        )
    return resources


def _collect_sub_resources_map(parsed: ParsedScene) -> dict[str, dict[str, Any]]:
    sub_resources: dict[str, dict[str, Any]] = {}
    for section in parsed.sections:
        if section.section_type != "sub_resource":
            continue
        identifier = _strip_quotes(section.attributes.get("id"))
        sub_resources[identifier] = {"type": _strip_quotes(section.attributes.get("type", ""))}
    return sub_resources


def _collect_resource_references(parsed: ParsedScene) -> tuple[set[str], set[str]]:
    ext_refs: set[str] = set()
    sub_refs: set[str] = set()

    def _scan_text(text: str) -> None:
        ext_refs.update(match.group(1) for match in _EXT_RESOURCE_PATTERN.finditer(text))
        sub_refs.update(match.group(1) for match in _SUB_RESOURCE_PATTERN.finditer(text))

    for section in parsed.sections:
        for value in section.attributes.values():
            _scan_text(value)
        for entry in section.entries:
            if entry.kind != "property":
                continue
            joined = "\n".join(entry.data.value_lines)
            _scan_text(joined)

    return ext_refs, sub_refs


def _load_scene(scene_path: str) -> ParsedScene:
    """Load a scene file, searching common locations if not found directly."""
    # Keep track of attempted paths for error reporting
    attempted_paths: list[Path] = []
    last_exception: Exception | None = None

    # Try to resolve and parse the path directly first
    try:
        resolved = _resolve_scene_file(scene_path)
        attempted_paths.append(resolved)
        # Try to parse even if doesn't exist (for test compatibility with mocked files)
        try:
            return _parse_tscn_file(resolved)
        except (FileNotFoundError, OSError) as parse_exc:
            last_exception = parse_exc
            # File doesn't exist at this location, continue to search
    except ValueError as exc:
        last_exception = exc
        # Path resolution failed (outside workspace), will try workspace-relative search below

    # If direct resolution didn't work, search common locations
    workspace = _get_agent_workspace()
    base_dir = workspace or Path.cwd()

    # Extract just the filename if a path was provided
    scene_name = Path(scene_path).name
    if not scene_name.endswith('.tscn'):
        scene_name += '.tscn'

    # List of common scene locations to search
    search_paths = [
        base_dir / scene_path,  # Direct relative to workspace
        base_dir / scene_name,  # Just filename at root
        base_dir / "scenes" / scene_name,  # In scenes/ folder
        base_dir / "new-game-project" / "scenes" / scene_name,  # In new-game-project/scenes/
    ]

    # Also search any */scenes/ directories
    try:
        scenes_dirs = list(base_dir.glob("*/scenes/"))
        for scenes_dir in scenes_dirs:
            search_paths.append(scenes_dir / scene_name)
    except Exception:  # noqa: BLE001
        pass  # Ignore glob errors

    # Try each search path, attempting to parse the file
    for candidate in search_paths:
        # Skip if already tried
        if candidate in attempted_paths:
            continue

        try:
            resolved_candidate = candidate.resolve()
            attempted_paths.append(resolved_candidate)

            # Only try to parse if file exists (skip for efficiency)
            if resolved_candidate.exists() and resolved_candidate.suffix == '.tscn':
                logger.info("Found scene at: %s", resolved_candidate)
                return _parse_tscn_file(resolved_candidate)
        except (FileNotFoundError, OSError, ValueError) as exc:
            last_exception = exc
            continue

    # File not found anywhere - provide helpful error message
    unique_paths = list(dict.fromkeys(str(p) for p in attempted_paths[:5]))  # Show first 5 unique locations
    searched_locations = "\n  - ".join(unique_paths)
    error_detail = f": {last_exception}" if last_exception else ""
    raise ValueError(
        f"Scene file '{scene_path}' not found{error_detail}. Searched locations:\n  - {searched_locations}"
    )


def read_godot_scene(scene_path: str) -> dict[str, Any]:
    """Parse a .tscn scene into a structured JSON-ready representation."""
    try:
        parsed = _load_scene(scene_path)
        nodes, _, _ = _build_nodes(parsed)
        if not nodes:
            return {"success": False, "error": "Scene file did not define any nodes."}
        root = _build_node_tree(nodes)
        if not root:
            return {"success": False, "error": "Unable to determine scene root node."}
        return {
            "success": True,
            "scene": {
                "path": str(parsed.path),
                "root": root,
                "ext_resources": _collect_ext_resources(parsed),
                "sub_resources": _collect_sub_resources(parsed),
            },
        }
    except ValueError as exc:
        logger.warning("Failed to read Godot scene %s: %s", scene_path, exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read Godot scene %s", scene_path)
        return {"success": False, "error": f"Failed to parse scene: {exc}"}


def read_godot_scene_tree(scene_path: str) -> dict[str, Any]:
    """Backward-compatible alias for read_godot_scene."""
    result = read_godot_scene(scene_path)
    if result.get("success"):
        result["scene_tree"] = result.pop("scene")
    return result


def add_godot_node(
    scene_path: str,
    parent_node_path: str,
    node_name: str,
    node_type: str,
    properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert a new node under the requested parent."""
    properties = properties or {}
    try:
        parsed = _load_scene(scene_path)
        nodes, node_map, root_name = _build_nodes(parsed)
        if not nodes:
            return {"success": False, "error": "Scene file did not define any nodes."}
        normalized_parent = _normalize_input_node_path(parent_node_path, root_name)
        parent = node_map.get(normalized_parent)
        if not parent:
            return {"success": False, "error": f"Parent node '{parent_node_path}' was not found."}
        sanitized_name = (node_name or "").strip()
        if not sanitized_name:
            return {"success": False, "error": "node_name must be provided."}
        sanitized_type = (node_type or "").strip()
        if not sanitized_type:
            return {"success": False, "error": "node_type must be provided."}
        new_path = sanitized_name if parent.path == "." else f"{parent.path}/{sanitized_name}"
        if new_path in node_map:
            return {
                "success": False,
                "error": f"A node already exists at path '{new_path}'.",
            }
        parent_attr = '"."' if parent.path == "." else f'"{parent.path}"'
        new_section = TSCNSection(
            section_type="node",
            attributes={
                "name": f'"{sanitized_name}"',
                "type": f'"{sanitized_type}"',
                "parent": parent_attr,
            },
            raw_header=None,
            header_dirty=True,
            leading_lines=[""],
        )
        for key, value in properties.items():
            value_lines = _format_property_value(value)
            new_section.entries.append(
                SectionEntry(kind="property", data=SectionProperty(name=key, value_lines=value_lines))
            )
        parsed.sections.append(new_section)
        _write_scene(parsed)
        return {
            "success": True,
            "node_path": new_path,
            "message": f"Added node '{sanitized_name}' under '{parent_node_path or '.'}'.",
        }
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to add node to %s", scene_path)
        return {"success": False, "error": f"Failed to add node: {exc}"}


def modify_godot_node_property(
    scene_path: str,
    node_path: str,
    property_name: str,
    new_value: Any,
) -> dict[str, Any]:
    """Update or add a property on an existing node."""
    try:
        parsed = _load_scene(scene_path)
        nodes, node_map, root_name = _build_nodes(parsed)
        if not nodes:
            return {"success": False, "error": "Scene file did not define any nodes."}
        normalized_path = _normalize_input_node_path(node_path, root_name)
        target = node_map.get(normalized_path)
        if not target:
            return {"success": False, "error": f"Node '{node_path}' was not found."}
        sanitized_property = (property_name or "").strip()
        if not sanitized_property:
            return {"success": False, "error": "property_name must be provided."}
        value_lines = _format_property_value(new_value)
        target.section.set_property(sanitized_property, value_lines)
        _write_scene(parsed)
        return {
            "success": True,
            "node_path": target.path,
            "property": sanitized_property,
            "message": f"Updated '{sanitized_property}' on node '{target.path}'.",
        }
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to modify node %s in %s", node_path, scene_path)
        return {"success": False, "error": f"Failed to modify node: {exc}"}


def validate_godot_scene(scene_path: str) -> dict[str, Any]:
    """Validate structure and resource references for a .tscn scene."""
    errors: list[str] = []
    try:
        parsed = _load_scene(scene_path)
        if not parsed.sections or parsed.sections[0].section_type != "gd_scene":
            errors.append("Scene is missing the [gd_scene] header.")
        nodes, node_map, _ = _build_nodes(parsed)
        root_nodes = [node for node in nodes if node.path == "."]
        if len(root_nodes) != 1:
            errors.append("Scene must contain exactly one root node.")
        for node in nodes:
            if node.path == ".":
                continue
            if node.parent not in node_map:
                errors.append(f"Node '{node.path}' references missing parent '{node.parent}'.")
        ext_resources = _collect_ext_resources_map(parsed)
        sub_resources = _collect_sub_resources_map(parsed)
        ext_refs, sub_refs = _collect_resource_references(parsed)
        for identifier in ext_refs:
            resource = ext_resources.get(identifier)
            if not resource:
                errors.append(f"ExtResource('{identifier}') is referenced but not declared.")
                continue
            resource_path = resource.get("path", "")
            if resource_path.startswith("res://"):
                resolved = _resolve_workspace_path(resource_path)
                if resolved and not resolved.exists():
                    errors.append(
                        f"ExtResource('{identifier}') points to missing file '{resource_path}'."
                    )
        for identifier in sub_refs:
            if identifier not in sub_resources:
                errors.append(f"SubResource('{identifier}') is referenced but not declared.")
        if errors:
            return {"success": False, "errors": errors}
        return {"success": True, "valid": True, "message": "Scene is valid."}
    except ValueError as exc:
        errors.append(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to validate Godot scene %s", scene_path)
        errors.append(f"Failed to validate scene: {exc}")
    return {"success": False, "errors": errors}


def get_project_godot_config() -> dict[str, Any]:
    """Parse basic sections of the project.godot file located at the workspace root."""
    try:
        with open("project.godot", "r", encoding="utf-8") as file:
            content = file.read()

        config: dict[str, Any] = {}
        autoloads: dict[str, str] = {}
        autoload_pattern = re.compile(r"\[autoload\]\s*([^[]+)", re.MULTILINE)
        autoload_match = autoload_pattern.search(content)
        if autoload_match:
            for line in autoload_match.group(1).strip().splitlines():
                if "=" not in line:
                    continue
                name, path = line.split("=", 1)
                autoloads[name.strip()] = path.strip().strip('"')
        config["autoloads"] = autoloads

        input_map: dict[str, Any] = {}
        input_map_pattern = re.compile(r"\[input\]\s*([^[]+)", re.MULTILINE)
        input_match = input_map_pattern.search(content)
        if input_match:
            actions = re.findall(r'"([^"]+)"=\{', input_match.group(1))
            input_map["actions"] = sorted(set(actions))
        config["input_map"] = input_map

        return {"success": True, "config": config}
    except FileNotFoundError:
        return {"success": False, "error": "project.godot file not found in the workspace root."}
    except Exception as exc:  # noqa: BLE001
        logger.error("Error parsing project.godot file: %s", exc)
        return {"success": False, "error": f"Failed to parse project.godot: {exc}"}


def create_godot_script(path: str, class_name: str, extends: str, template: str) -> dict[str, Any]:
    """
    Generates a GDScript file with the correct syntax and template.

    Args:
        path: The path to create the script file.
        class_name: The name of the class (optional).
        extends: The node type the script extends from.
        template: The template to use ('basic', 'character', 'singleton').

    Returns:
        A dictionary indicating success or an error message.
    """
    templates = {
        "basic": """
extends {extends}
{class_name_line}

# Called when the node enters the scene tree for the first time.
func _ready():
\tpass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
\tpass
""",
        "character": """
extends {extends}
{class_name_line}

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")


func _physics_process(delta):
\tif not is_on_floor():
\t\tvelocity.y += gravity * delta

\tif Input.is_action_just_pressed("ui_accept") and is_on_floor():
\t\tvelocity.y = JUMP_VELOCITY

\tvar direction = Input.get_axis("ui_left", "ui_right")
\tif direction:
\t\tvelocity.x = direction * SPEED
\telse:
\t\tvelocity.x = move_toward(velocity.x, 0, SPEED)

\tmove_and_slide()
""",
        "singleton": """
extends {extends}
{class_name_line}

# This script is an autoload singleton.

func _ready():
    print("Singleton is ready!")

func my_global_function():
    pass
""",
    }

    if template not in templates:
        return {
            "success": False,
            "error": f"Invalid template '{template}'. Available templates: {list(templates.keys())}",
        }

    script_content = templates[template]
    class_name_line = f"class_name {class_name}" if class_name else ""

    content = script_content.format(extends=extends, class_name_line=class_name_line).strip()

    try:
        script_path = Path(path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(content, encoding="utf-8")
        return {"success": True, "message": f"Script created at {path}"}
    except Exception as exc:  # noqa: BLE001
        logger.error("Error creating script file at %s: %s", path, exc)
        return {"success": False, "error": f"Failed to write script: {exc}"}


__all__ = [
    "read_godot_scene",
    "read_godot_scene_tree",
    "add_godot_node",
    "modify_godot_node_property",
    "validate_godot_scene",
    "get_project_godot_config",
    "create_godot_script",
]
