"""
A utility for dynamically generating Anthropic tool schemas from Python callables.
"""
import inspect
import re
from typing import Callable, Dict, Any

def build_anthropic_tool_schema(tool: Callable) -> Dict[str, Any]:
    """
    Dynamically builds a JSON schema for a given Python tool (callable)
    that is compatible with the Anthropic API.

    Args:
        tool: The Python tool (function or method) to generate the schema for.

    Returns:
        A dictionary representing the JSON schema for the tool.
    """
    signature = inspect.signature(tool)
    docstring = inspect.getdoc(tool) or ""

    # Parse the docstring
    docstring_lines = docstring.strip().split('\n')
    description = docstring_lines[0] if docstring_lines else ""
    param_descriptions = {}
    for line in docstring_lines[1:]:
        line = line.strip()
        if line.startswith(":param"):
            match = re.match(r':param\s+([^:]+):\s+(.*)', line)
            if match:
                name, desc = match.groups()
                param_descriptions[name.strip()] = desc.strip()


    properties = {}
    required = []

    type_mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    for name, param in signature.parameters.items():
        param_type = "string"  # Default to string
        if param.annotation is not inspect.Parameter.empty:
            annotation_str = str(param.annotation)
            if "dict" in annotation_str.lower():
                param_type = "object"
            elif "list" in annotation_str.lower():
                param_type = "array"
            else:
                try:
                    param_type = type_mapping.get(param.annotation.__name__, "string")
                except AttributeError:
                    param_type = "string"  # Fallback for complex types without __name__

        properties[name] = {
            "type": param_type,
            "description": param_descriptions.get(name, "")
        }
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "name": tool.__name__,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
