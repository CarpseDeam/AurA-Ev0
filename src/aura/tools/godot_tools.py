# src/aura/tools/godot_tools.py

"""
Tools for interacting with Godot Engine projects.

This module provides functionality for reading Godot project configuration,
parsing scene files, and generating GDScript boilerplate code.
"""

import logging
import re
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


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


def read_godot_scene_tree(scene_path: str) -> dict:
    """
    Parses a .tscn file and returns a JSON representation of the node hierarchy.

    Args:
        scene_path: The path to the .tscn file.

    Returns:
        A dictionary representing the scene tree or an error message.
    """
    try:
        resolved_scene = _resolve_scene_file(scene_path)
    except ValueError as exc:
        return {"error": str(exc)}

    try:
        with open(resolved_scene, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.splitlines()

        if not lines or not lines[0].startswith('[gd_scene '):
            return {"error": "Not a valid .tscn file."}

        nodes = {}
        node_order = []  # Track the order nodes appear in the file

        for line in lines:
            line = line.strip()

            # Look for [node ...] sections
            if line.startswith('[node '):
                # Extract attributes from the node section header
                name_match = re.search(r'name="([^"]+)"', line)
                type_match = re.search(r'type="([^"]+)"', line)
                parent_match = re.search(r'parent="([^"]+)"', line)

                if name_match and type_match:
                    node_name = name_match.group(1)
                    node_type = type_match.group(1)
                    parent_path = parent_match.group(1) if parent_match else "."

                    # Create node entry
                    node_data = {
                        "name": node_name,
                        "type": node_type,
                        "parent_path": parent_path,
                        "children": []
                    }
                    nodes[node_name] = node_data
                    node_order.append(node_name)

        # Build the tree hierarchy
        root_nodes = {}

        for node_name in node_order:
            node_data = nodes[node_name]
            parent_path = node_data["parent_path"]

            # Root node has parent "."
            if parent_path == ".":
                # This is a root node
                root_nodes[node_name] = node_data
            else:
                # Find the parent node by matching the path
                # The parent_path can be "." for direct children of root
                # or "ParentName" or "ParentName/GrandParent" for nested nodes
                parent_name = parent_path.split('/')[-1] if '/' in parent_path else parent_path

                # If parent_path is just ".", it's a child of the first root
                if parent_path == ".":
                    if node_order:
                        first_root = node_order[0]
                        if first_root in nodes:
                            nodes[first_root]["children"].append(node_data)
                elif parent_name in nodes:
                    nodes[parent_name]["children"].append(node_data)
                else:
                    # If we can't find the parent, add as root
                    root_nodes[node_name] = node_data

            # Remove parent_path from the final output as it's just for building the tree
            node_data.pop("parent_path", None)

        return {"success": True, "scene_tree": root_nodes}

    except FileNotFoundError:
        return {"error": f"Scene file not found at {resolved_scene}"}
    except Exception as e:
        logger.error(f"Error parsing scene file {resolved_scene}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


def get_project_godot_config() -> dict:
    """
    Reads and parses the project.godot file from the workspace root.

    Returns:
        A dictionary with input map and autoload singletons, or an error message.
    """
    try:
        with open('project.godot', 'r', encoding='utf-8') as f:
            content = f.read()

        config = {}
        
        autoloads = {}
        autoload_pattern = re.compile(r'[autoload]\n([^\n]+)', re.DOTALL)
        autoload_match = autoload_pattern.search(content)
        if autoload_match:
            autoload_block = autoload_match.group(1)
            for line in autoload_block.strip().split('\n'):
                if '=' in line:
                    name, path = line.split('=', 1)
                    name = name.strip()
                    path = path.strip().strip('"')
                    autoloads[name] = path
        config['autoloads'] = autoloads

        input_map = {}
        input_map_pattern = re.compile(r'[input]\n([^\n]+)', re.DOTALL)
        input_map_match = input_map_pattern.search(content)
        if input_map_match:
            input_map_block = input_map_match.group(1)
            action_pattern = re.compile(r'"([^"]+)"={{')
            actions = action_pattern.findall(input_map_block)
            input_map['actions'] = list(set(actions))

        config['input_map'] = input_map

        return {"success": True, "config": config}

    except FileNotFoundError:
        return {"error": "project.godot file not found in the workspace root."}
    except Exception as e:
        logger.error(f"Error parsing project.godot file: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


def create_godot_script(path: str, class_name: str, extends: str, template: str) -> dict:
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
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
""",
        "character": """
extends {extends}
{class_name_line}

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")


func _physics_process(delta):
	if not is_on_floor():
		velocity.y += gravity * delta

	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	var direction = Input.get_axis("ui_left", "ui_right")
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)

	move_and_slide()
""",
        "singleton": """
extends {extends}
{class_name_line}

# This script is an autoload singleton.

func _ready():
    print("Singleton is ready!")

func my_global_function():
    pass
"""
    }

    if template not in templates:
        return {"error": f"Invalid template '{template}'. Available templates: {list(templates.keys())}"}

    script_content = templates[template]
    class_name_line = f"class_name {class_name}" if class_name else ""
    
    content = script_content.format(
        extends=extends,
        class_name_line=class_name_line
    ).strip()

    try:
        script_path = Path(path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {"success": True, "message": f"Script created at {path}"}
    except Exception as e:
        logger.error(f"Error creating script file at {path}: {e}")
        return {"error": f"An unexpected error occurred while writing the file: {e}"}