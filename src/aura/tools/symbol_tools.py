"""Symbol resolution tools for analyzing Python code structure using AST parsing."""

from __future__ import annotations

import ast
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def get_dependency_graph(symbol_name: str, search_directory: str = ".") -> dict[str, Any]:
    """Build a lightweight dependency graph for a symbol across the project."""
    LOGGER.info("?? TOOL CALLED: get_dependency_graph(%s)", symbol_name)
    search_path = Path(search_directory).resolve()
    if not search_path.exists():
        return {"error": f"Directory does not exist: {search_directory}"}

    target = _locate_symbol(symbol_name, search_path)
    if not target:
        return {"error": f"Symbol '{symbol_name}' not found in {search_directory}"}

    node, source_path, lines = target
    dependencies = _collect_symbol_dependencies(node, source_path, lines)
    dependents = _collect_symbol_dependents(symbol_name, search_path, limit=150)

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


def get_class_hierarchy(class_name: str, search_directory: str = ".") -> dict[str, Any]:
    """Return inheritance details for a class, including parents and subclasses."""
    LOGGER.info("?? TOOL CALLED: get_class_hierarchy(%s)", class_name)
    search_path = Path(search_directory).resolve()
    if not search_path.exists():
        return {"error": f"Directory does not exist: {search_directory}"}

    class_map: dict[str, dict[str, Any]] = {}
    children_map: dict[str, list[str]] = defaultdict(list)

    for py_file in search_path.rglob("*.py"):
        content, tree = _read_ast(py_file)
        if not tree or content is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [_expression_to_name(base) for base in node.bases if _expression_to_name(base)]
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

    ancestors = _collect_ancestors(class_name, class_map)
    descendants = _collect_descendants(class_name, children_map)

    return {
        "class": class_name,
        "defined_in": class_map[class_name]["file"],
        "line": class_map[class_name]["line"],
        "bases": class_map[class_name]["bases"],
        "ancestors": ancestors,
        "descendants": descendants,
        "hierarchy": _build_hierarchy_branch(class_name, class_map, children_map),
    }


def safe_rename_symbol(
    file_path: str,
    symbol_name: str,
    new_name: str,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Perform a project-wide, refactor-aware rename using Rope."""
    LOGGER.info("?? TOOL CALLED: safe_rename_symbol(%s -> %s)", symbol_name, new_name)
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

    target_path = Path(file_path)
    if not target_path.is_absolute():
        target_path = Path.cwd() / target_path
    if not target_path.exists():
        return {"success": False, "error": f"File does not exist: {file_path}"}

    try:
        source = target_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target_path))
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Failed to parse file: {exc}"}

    target_node = _find_symbol_node(tree, symbol_name)
    if not target_node:
        return {"success": False, "error": f"Symbol '{symbol_name}' not found in {file_path}"}

    offset = _calculate_offset(source, getattr(target_node, "lineno", 1), getattr(target_node, "col_offset", 0))
    root = Path(project_root).resolve() if project_root else Path.cwd()

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


def _locate_symbol(symbol_name: str, search_path: Path) -> tuple[ast.AST, Path, list[str]] | None:
    for py_file in search_path.rglob("*.py"):
        content, tree = _read_ast(py_file)
        if not tree or content is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_name:
                return node, py_file, content.splitlines()
    return None


def _collect_symbol_dependencies(node: ast.AST, file_path: Path, lines: list[str]) -> list[dict[str, Any]]:
    dependencies: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()

    if isinstance(node, ast.ClassDef):
        for base in node.bases:
            base_name = _expression_to_name(base)
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
            call_name = _expression_to_name(child.func)
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
                            "context": _line_text(lines, lineno),
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
                        "context": _line_text(lines, lineno),
                    }
                )
                seen.add(key)

    return dependencies


def _collect_symbol_dependents(symbol_name: str, search_path: Path, limit: int = 150) -> list[dict[str, Any]]:
    dependents: list[dict[str, Any]] = []
    seen_locations: set[tuple[str, int]] = set()

    for py_file in search_path.rglob("*.py"):
        content, tree = _read_ast(py_file)
        if not tree or content is None:
            continue
        lines = content.splitlines()

        for node in ast.walk(tree):
            if _node_references_symbol(node, symbol_name):
                lineno = getattr(node, "lineno", None)
                location = (str(py_file), lineno or 0)
                if location in seen_locations:
                    continue
                seen_locations.add(location)
                dependents.append(
                    {
                        "file": str(py_file),
                        "line": lineno,
                        "context": _line_text(lines, lineno),
                        "usage_type": _classify_usage(node, symbol_name),
                    }
                )
                if len(dependents) >= limit:
                    return dependents
    return dependents


def _read_ast(py_file: Path) -> tuple[str | None, ast.AST | None]:
    try:
        content = py_file.read_text(encoding="utf-8")
        return content, ast.parse(content, filename=str(py_file))
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Skipping %s: %s", py_file, exc)
        return None, None


def _expression_to_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        value = _expression_to_name(expr.value)
        return f"{value}.{expr.attr}" if value else expr.attr
    if isinstance(expr, ast.Subscript):
        return _expression_to_name(expr.value)
    if isinstance(expr, ast.Call):
        return _expression_to_name(expr.func)
    return None


def _collect_ancestors(class_name: str, class_map: dict[str, dict[str, Any]], visited: set[str] | None = None) -> list[str]:
    visited = visited or set()
    visited.add(class_name)
    ancestors: list[str] = []

    bases = class_map.get(class_name, {}).get("bases", [])
    for base in bases:
        ancestors.append(base)
        if base in class_map and base not in visited:
            ancestors.extend(_collect_ancestors(base, class_map, visited))
    return ancestors


def _collect_descendants(
    class_name: str,
    children_map: dict[str, list[str]],
    visited: set[str] | None = None,
) -> list[str]:
    visited = visited or set()
    visited.add(class_name)
    descendants: list[str] = []

    for child in children_map.get(class_name, []):
        descendants.append(child)
        if child not in visited:
            descendants.extend(_collect_descendants(child, children_map, visited))
    return descendants


def _build_hierarchy_branch(
    class_name: str,
    class_map: dict[str, dict[str, Any]],
    children_map: dict[str, list[str]],
    visited: set[str] | None = None,
) -> dict[str, Any]:
    visited = visited or set()
    visited.add(class_name)
    info = class_map.get(class_name, {})
    return {
        "name": class_name,
        "file": info.get("file"),
        "line": info.get("line"),
        "bases": info.get("bases", []),
        "children": [
            _build_hierarchy_branch(child, class_map, children_map, visited)
            for child in children_map.get(class_name, [])
            if child not in visited
        ],
    }


def _find_symbol_node(tree: ast.AST, symbol_name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol_name:
            return node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    return node
    return None


def _calculate_offset(source: str, line: int, column: int) -> int:
    lines = source.splitlines(keepends=True)
    line_index = max(0, line - 1)
    prior = sum(len(lines[i]) for i in range(min(line_index, len(lines))))
    return prior + column


def _line_text(lines: list[str], lineno: int | None) -> str:
    if not lineno or lineno <= 0 or lineno > len(lines):
        return ""
    return lines[lineno - 1].strip()


def _node_references_symbol(node: ast.AST, symbol_name: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == symbol_name
    if isinstance(node, ast.Attribute):
        return node.attr == symbol_name
    if isinstance(node, ast.Call):
        ref_name = _expression_to_name(node.func)
        return ref_name == symbol_name
    if isinstance(node, ast.ImportFrom):
        return any(alias.name == symbol_name or alias.asname == symbol_name for alias in node.names)
    if isinstance(node, ast.Import):
        return any(alias.name.split(".")[-1] == symbol_name for alias in node.names)
    return False
