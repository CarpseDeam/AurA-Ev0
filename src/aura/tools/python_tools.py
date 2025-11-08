"""Python-specific tool functions for Aura.

This module contains tools for testing, linting, formatting, and analyzing Python code.
"""

from __future__ import annotations

import ast
import logging
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def run_tests(test_path: str = "tests/", verbose: bool = False) -> dict[str, object]:
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
            cwd=os.getcwd(),
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


def lint_code(file_paths: list[str] | None = None, directory: str = ".") -> dict[str, object]:
    """Run pylint to catch errors and code quality issues.

    Args:
        file_paths: Optional list of specific files to lint
        directory: Directory to lint if file_paths not provided (default: ".")

    Returns:
        Dictionary with keys: errors (list), warnings (list), score (float), output (str)
    """
    LOGGER.info("🔧 TOOL CALLED: lint_code(%s)", file_paths or directory)
    try:
        cmd = ["pylint"]

        if file_paths:
            cmd.extend(file_paths)
        else:
            base = Path(directory)
            if not base.is_absolute():
                base = Path.cwd() / base

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
            cwd=os.getcwd(),
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


def install_package(package: str, version: str = "") -> str:
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
            cwd=os.getcwd(),
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
    file_paths: list[str] | None = None, directory: str = "."
) -> dict[str, object]:
    """Format Python code using Black formatter.

    Args:
        file_paths: Optional list of specific files to format
        directory: Directory to format if file_paths not provided (default: ".")

    Returns:
        Dictionary with keys: formatted (count), errors (list), message (summary)
    """
    LOGGER.info("🔧 TOOL CALLED: format_code(%s)", file_paths or directory)
    try:
        cmd = ["black"]

        if file_paths:
            cmd.extend(file_paths)
        else:
            cmd.append(directory)

        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
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


def get_function_definitions(file_path: str) -> list[dict[str, object]]:
    """Extract function signatures from a Python file.

    Args:
        file_path: Path to the Python file to analyze

    Returns:
        List of dictionaries with keys: name, params, line, docstring
        Example: [{"name": "generate_password", "params": ["length", "use_numbers"], "line": 5}]
    """
    LOGGER.info("🔧 TOOL CALLED: get_function_definitions(%s)", file_path)
    try:
        target = Path(file_path)
        if not target.is_absolute():
            target = Path.cwd() / target

        if not target.exists():
            LOGGER.error("File does not exist: %s", file_path)
            return []

        if not target.suffix == ".py":
            LOGGER.error("File is not a Python file: %s", file_path)
            return []

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
        return functions

    except SyntaxError as exc:
        LOGGER.error("Syntax error in file %s: %s", file_path, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to extract function definitions from %s: %s", file_path, exc)
        return []


def get_cyclomatic_complexity(file_path: str) -> dict[str, Any]:
    """Calculate cyclomatic complexity metrics for the provided Python file."""
    LOGGER.info("?? TOOL CALLED: get_cyclomatic_complexity(%s)", file_path)
    try:
        from radon.complexity import cc_visit
    except ImportError:
        return {
            "error": "Radon is not installed. Install it with: pip install radon",
        }

    target = _resolve_python_path(file_path)
    if not target.exists():
        return {"error": f"File does not exist: {file_path}"}

    try:
        content = target.read_text(encoding="utf-8")
        blocks = cc_visit(content)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to compute complexity for %s: %s", file_path, exc)
        return {"error": f"Failed to compute complexity: {exc}"}

    entries: list[dict[str, Any]] = []
    complexities: list[float] = []
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
    source_file: str,
    tests_root: str = "tests",
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create or extend a pytest test file with stubs for public callables in source_file.
    """
    LOGGER.info("?? TOOL CALLED: generate_test_file(%s)", source_file)
    source_path = _resolve_python_path(source_file)
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

    symbols = _collect_public_callables(tree)
    if not symbols["functions"] and not symbols["methods"]:
        return {"success": False, "error": "No public callables found to generate tests for."}

    module_parts = _module_parts_from_source(source_path)
    module_path = ".".join(module_parts) if module_parts else source_path.stem
    tests_directory = _resolve_tests_root(tests_root)
    destination = _compute_test_destination(module_parts, tests_directory, source_path)
    stubs = _build_test_stubs(symbols)

    header = _build_test_header(module_path, symbols)
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


def _resolve_python_path(path_like: str) -> Path:
    target = Path(path_like)
    if not target.is_absolute():
        target = Path.cwd() / target
    return target.resolve()


def _module_parts_from_source(source_path: Path) -> list[str]:
    project_root = Path.cwd()
    try:
        relative = source_path.resolve().relative_to(project_root)
        module_path = relative.with_suffix("")
        parts = list(module_path.parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        return parts or [source_path.with_suffix("").name]
    except ValueError:
        return [source_path.with_suffix("").name]


def _collect_public_callables(tree: ast.AST) -> dict[str, list[Any]]:
    functions: list[str] = []
    methods: list[tuple[str, str]] = []

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


def _resolve_tests_root(root: str) -> Path:
    tests_root = Path(root)
    if not tests_root.is_absolute():
        tests_root = Path.cwd() / tests_root
    return tests_root.resolve()


def _compute_test_destination(module_parts: list[str], tests_root: Path, source_path: Path) -> Path:
    if module_parts:
        module_name = module_parts[-1]
        sub_path = Path(*module_parts[:-1]) if len(module_parts) > 1 else Path()
        destination_dir = tests_root / sub_path
    else:
        module_name = source_path.stem
        destination_dir = tests_root
    return destination_dir / f"test_{module_name}.py"


def _build_test_header(module_path: str, symbols: dict[str, list[Any]]) -> str:
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


def _build_test_stubs(symbols: dict[str, list[Any]]) -> list[tuple[str, str]]:
    stubs: list[tuple[str, str]] = []
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
