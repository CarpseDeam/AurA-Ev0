"""
A utility for dynamically generating Anthropic tool schemas from Python callables.
"""
import inspect
import re
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel


def build_anthropic_tool_schema(tool: Callable, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Dynamically builds a JSON schema for a given Python tool (callable)
    that is compatible with the Anthropic API.

    Args:
        tool: The Python tool (function or method) to generate the schema for.
        name: Optional explicit tool name to use in the schema output.

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
                param_name_from_doc, desc = match.groups()
                param_descriptions[param_name_from_doc.strip()] = desc.strip()


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

    for param_name, param in signature.parameters.items():
        param_type = "string"  # Default to string
        if param.annotation is not inspect.Parameter.empty:
            annotation_str = str(param.annotation)
            if "dict" in annotation_str.lower():
                param_type = "object"
            elif "list" in annotation_str.lower():
                param_type = "array"
            else:
                try:
                    # Try to get the type from __name__ attribute
                    param_type = type_mapping.get(param.annotation.__name__, "string")
                except AttributeError:
                    # If annotation is a string (from stringized annotations), try direct mapping
                    if isinstance(param.annotation, str):
                        param_type = type_mapping.get(param.annotation, "string")
                    else:
                        param_type = "string"  # Fallback for complex types without __name__

        properties[param_name] = {
            "type": param_type,
            "description": param_descriptions.get(param_name, "")
        }
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    tool_name = name if name is not None else tool.__name__

    return {
        "name": tool_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def build_pydantic_tool_schema(
    model: Type[BaseModel],
    name: str,
    description: str,
    *,
    additional_required: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Build an Anthropic tool schema directly from a Pydantic model.

    This ensures the tool schema exactly matches the Pydantic validation,
    so Claude sees the same schema that will be used for validation.

    Args:
        model: The Pydantic model class to generate the schema from.
        name: The tool name to use in the schema.
        description: The tool description.
        additional_required: Additional field names to mark as required,
            useful for fields that are validated by model validators
            but have default values in the schema.

    Returns:
        A dictionary representing the JSON schema for the tool.
    """
    # Get the JSON schema from the Pydantic model
    model_schema = model.model_json_schema()

    # Convert Pydantic JSON schema to Anthropic tool schema format
    properties = {}
    required = []

    # Extract properties and required fields
    if "properties" in model_schema:
        properties = model_schema["properties"]

    if "required" in model_schema:
        required = list(model_schema["required"])

    # Add additional required fields
    if additional_required:
        for field_name in additional_required:
            if field_name not in required and field_name in properties:
                required.append(field_name)

    # Process nested definitions if they exist
    definitions = model_schema.get("$defs", {})

    # Replace references with actual definitions
    def resolve_refs(schema_obj: Any) -> Any:
        """Recursively resolve $ref references in the schema."""
        if isinstance(schema_obj, dict):
            if "$ref" in schema_obj:
                ref_path = schema_obj["$ref"]
                # Extract the definition name from the reference
                # e.g., "#/$defs/FileOperation" -> "FileOperation"
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in definitions:
                        return resolve_refs(definitions[def_name])
                return schema_obj
            else:
                return {k: resolve_refs(v) for k, v in schema_obj.items()}
        elif isinstance(schema_obj, list):
            return [resolve_refs(item) for item in schema_obj]
        else:
            return schema_obj

    properties = resolve_refs(properties)

    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
