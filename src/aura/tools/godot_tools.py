# src/aura/tools/godot_tools.py

"""
Tools for interacting with Godot Engine projects.

This module provides functionality for reading Godot project configuration,
parsing scene files, and generating GDScript boilerplate code.
"""

import re
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

def read_godot_scene_tree(scene_path: str) -> dict:
    """
    Parses a .tscn file and returns a JSON representation of the node hierarchy.

    Args:
        scene_path: The path to the .tscn file.

    Returns:
        A dictionary representing the scene tree or an error message.
    """
    try:
        with open(scene_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic validation
        if not content.startswith('[gd_scene '):
            return {"error": "Not a valid .tscn file."}

        nodes = {}
        node_pattern = re.compile(r'[node name="([^"]+)" type="([^"]+)"(?: parent="([^"]+)")?]')
        script_pattern = re.compile(r'script = ExtResource\("([^"]+)"\)')

        # Split content by [node] sections to process them individually
        node_sections = content.split('[node ')
        
        for section in node_sections[1:]: # Skip the header part
            full_section = '[node ' + section
            match = node_pattern.search(full_section)
            if not match:
                continue

            name, type, parent = match.groups()
            parent = parent if parent else "." # Root nodes have parent="."

            script_match = script_pattern.search(full_section)
            script_path = script_match.group(1) if script_match else None

            nodes[name] = {
                "type": type,
                "parent": parent,
                "script": script_path,
                "children": []
            }

        # Build the hierarchy
        tree = {}
        for name, data in nodes.items():
            if data['parent'] != '.' and data['parent'] in nodes:
                nodes[data['parent']]['children'].append(nodes[name])
            elif data['parent'] == '.':
                tree[name] = data
        
        # Clean up children from the top-level nodes that have been moved
        final_tree = {}
        for name, data in nodes.items():
            if data['parent'] == '.':
                final_tree[name] = data

        return {"success": True, "scene_tree": final_tree}

    except FileNotFoundError:
        return {"error": f"Scene file not found at {scene_path}"}
    except Exception as e:
        logger.error(f"Error parsing scene file {scene_path}: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


def get_project_godot_config() -> dict:
    """
    Reads and parses the project.godot file from the workspace root.

    Returns:
        A dictionary with input map and autoload singletons, or an error message.
    """
    try:
        # Assuming project.godot is in the root of the workspace.
        # This might need to be adjusted if the project is in a subdirectory.
        with open('project.godot', 'r', encoding='utf-8') as f:
            content = f.read()

        config = {}
        
        # Extract autoload singletons
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

        # Extract input map
        input_map = {}
        input_map_pattern = re.compile(r'[input]\n([^\n]+)', re.DOTALL)
        input_map_match = input_map_pattern.search(content)
        if input_map_match:
            input_map_block = input_map_match.group(1)
            # This is a simplification. Godot's input map is more complex.
            # This regex just looks for action names.
            action_pattern = re.compile(r'"([^"]+)"=\{')
            actions = action_pattern.findall(input_map_block)
            input_map['actions'] = list(set(actions)) # Get unique actions

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

# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")


func _physics_process(delta):
	# Add the gravity.
	if not is_on_floor():
		velocity.y += gravity * delta

	# Handle Jump.
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
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

# Add your global functions here.
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
