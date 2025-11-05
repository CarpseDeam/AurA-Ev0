"""Python-specific tool functions for Aura.

This module contains tools for testing, linting, formatting, and analyzing Python code.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def run_tests(test_path: str = "tests/", verbose: bool = False) -> dict[str, object]:
    """Run pytest on the codebase and return test results.

    Args:
        test_path: Path to tests directory or file (default: "tests/")
        verbose: Enable verbose output (default: False)

    Returns:
        Dictionary with keys: passed, failed, duration, output
    """
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
