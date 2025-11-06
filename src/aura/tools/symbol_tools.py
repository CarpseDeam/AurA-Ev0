"""Symbol resolution tools for analyzing Python code structure using AST parsing."""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

STDLIB_MODULES = getattr(sys, 'stdlib_module_names', {
    'os', 'sys', 'ast', 'logging', 'pathlib', 'typing', 'json', 'datetime',
    'collections', 'functools', 'itertools', 're', 'math', 'random', 'time',
    'io', 'subprocess', 'threading', 'multiprocessing', 'unittest', 'pytest'
})


def find_definition(symbol_name: str, search_directory: str = ".") -> dict[str, Any]:
    """Find where a symbol (class/function/variable) is defined.

    Args:
        symbol_name: Name of the symbol to search for
        search_directory: Directory to search recursively (default ".")

    Returns:
        Dictionary with keys: found, file, line, type, signature, docstring, context
    """
    LOGGER.info(f"🔧 TOOL CALLED: find_definition(symbol_name={symbol_name})")
    search_path = Path(search_directory).resolve()
    if not search_path.exists():
        return {"found": False, "error": f"Directory does not exist: {search_directory}"}

    for py_file in search_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            lines = content.splitlines()
            for node in ast.walk(tree):
                result = _match_definition_node(node, symbol_name, py_file, lines)
                if result:
                    LOGGER.debug(f"Found {symbol_name} in {result['file']} at line {result['line']}")
                    return result
        except Exception as exc:
            LOGGER.warning(f"Failed to parse {py_file}: {exc}")

    LOGGER.debug(f"Symbol {symbol_name} not found in {search_directory}")
    return {"found": False, "error": f"Symbol '{symbol_name}' not found"}


def _match_definition_node(node: ast.AST, symbol_name: str, file_path: Path, lines: list[str]) -> dict[str, Any] | None:
    if isinstance(node, ast.ClassDef) and node.name == symbol_name:
        return _create_result(file_path, node, "class", lines)
    elif isinstance(node, ast.FunctionDef) and node.name == symbol_name:
        return _create_result(file_path, node, "function", lines)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == symbol_name:
                return _create_result(file_path, node, "variable", lines)
    return None


def _create_result(file_path: Path, node: ast.AST, def_type: str, lines: list[str]) -> dict[str, Any]:
    line_num = node.lineno
    start, end = max(0, line_num - 4), min(len(lines), line_num + 3)
    return {
        "found": True, "file": str(file_path), "line": line_num, "type": def_type,
        "signature": lines[line_num - 1].strip() if line_num <= len(lines) else "",
        "docstring": ast.get_docstring(node) or "", "context": lines[start:end]
    }


def find_usages(symbol_name: str, search_directory: str = ".") -> dict[str, Any]:
    """Find all usages of a symbol in Python files.

    Args:
        symbol_name: Name of the symbol to search for
        search_directory: Directory to search recursively (default ".")

    Returns:
        Dictionary with keys: total_usages, files_count, usages (list of usage dicts)
    """
    LOGGER.info(f"🔧 TOOL CALLED: find_usages(symbol_name={symbol_name})")
    search_path = Path(search_directory).resolve()
    if not search_path.exists():
        return {"total_usages": 0, "files_count": 0, "error": f"Directory does not exist: {search_directory}"}

    all_usages: list[dict[str, Any]] = []
    files_with_usages = set()

    for py_file in search_path.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
            lines = content.splitlines()
            for node in ast.walk(tree):
                usage_type = _classify_usage(node, symbol_name)
                if usage_type and hasattr(node, 'lineno') and node.lineno <= len(lines):
                    all_usages.append({
                        "file": str(py_file), "line": node.lineno,
                        "context": lines[node.lineno - 1].strip(), "usage_type": usage_type
                    })
                    files_with_usages.add(str(py_file))
                if len(all_usages) >= 100:
                    break
        except Exception as exc:
            LOGGER.warning(f"Failed to parse {py_file}: {exc}")
        if len(all_usages) >= 100:
            break

    LOGGER.debug(f"Found {len(all_usages)} usages of {symbol_name} in {len(files_with_usages)} files")
    return {"total_usages": len(all_usages), "files_count": len(files_with_usages), "usages": all_usages[:100]}


def _classify_usage(node: ast.AST, symbol_name: str) -> str | None:
    if isinstance(node, ast.ImportFrom) and any(alias.name == symbol_name for alias in node.names):
        return "import"
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == symbol_name:
        return "call"
    elif isinstance(node, ast.Attribute) and node.attr == symbol_name:
        return "attribute"
    elif isinstance(node, ast.Name) and node.id == symbol_name:
        return "reference"
    return None


def get_imports(file_path: str) -> dict[str, Any]:
    """Extract and categorize all imports from a Python file.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        Dictionary with keys: stdlib, third_party, local, import_details
    """
    LOGGER.info(f"🔧 TOOL CALLED: get_imports(file_path={file_path})")
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File does not exist: {file_path}"}

    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=file_path)
        stdlib_imports, third_party_imports, local_imports, import_details = [], [], [], []

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                details = _parse_import_node(node)
                import_details.append(details)
                module = details["module"]
                if _is_local_import(module):
                    local_imports.append(module)
                elif _is_stdlib_module(module):
                    stdlib_imports.append(module)
                else:
                    third_party_imports.append(module)

        LOGGER.debug(f"Extracted {len(import_details)} imports from {file_path}")
        return {
            "stdlib": sorted(set(stdlib_imports)), "third_party": sorted(set(third_party_imports)),
            "local": sorted(set(local_imports)), "import_details": import_details
        }
    except Exception as exc:
        LOGGER.warning(f"Failed to parse {file_path}: {exc}")
        return {"error": f"Failed to parse file: {exc}"}


def _parse_import_node(node: ast.Import | ast.ImportFrom) -> dict[str, Any]:
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


def _is_local_import(module: str) -> bool:
    return module.startswith(".") or module.startswith("src.") or module.startswith("aura.")


def _is_stdlib_module(module: str) -> bool:
    return module.split(".")[0] in STDLIB_MODULES
