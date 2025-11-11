"""Python-focused analysis helpers extracted from ToolManager."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from .tool_manager import ToolManager


LOGGER = logging.getLogger(__name__)

__all__ = ["PythonTools"]


class PythonTools:
    """Encapsulate Python analysis/manipulation helpers for ToolManager."""

    def __init__(self, manager: "ToolManager", stdlib_modules) -> None:
        self._manager = manager
        self._stdlib_modules = set(stdlib_modules)

    @property
    def workspace_dir(self) -> Path:
        return self._manager.workspace_dir

    def _resolve_directory(self, directory: str) -> Path:
        return self._manager._resolve_directory(directory)

    def _resolve_path(self, path_like: str) -> Path:
        return self._manager._resolve_path(path_like)

    def _relative_path(self, path: Path) -> str:
        return self._manager._relative_path(path)

    def _iter_workspace_files(self, root: Path, *, respect_gitignore: bool = True) -> Iterator[Path]:
        return self._manager._iter_workspace_files(root, respect_gitignore=respect_gitignore)

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
        return module.split(".")[0] in self._stdlib_modules

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
