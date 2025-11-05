"""Conversational interface for Aura with Gemini streaming."""

from __future__ import annotations

import ast
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterator, Mapping

import google.generativeai as genai
from aura.agents import PythonCoderAgent, SessionContext

LOGGER = logging.getLogger(__name__)


class SessionContextManager:
    """Manage shared session context between tool calls."""

    def __init__(self) -> None:
        self._context: list[str] = []
        self._lock = Lock()

    def get_context(self) -> tuple[str, ...]:
        """Return stored context entries."""
        with self._lock:
            return tuple(self._context)

    def add_entry(self, entry: str) -> None:
        """Append a context entry if it is non-empty."""
        sanitized = entry.strip()
        if not sanitized:
            return
        with self._lock:
            self._context.append(sanitized)

    def clear(self) -> None:
        """Remove all stored context entries."""
        with self._lock:
            self._context.clear()


_SESSION_CONTEXT_MANAGER: SessionContextManager | None = None


def get_session_context_manager() -> SessionContextManager:
    """Return the singleton session context manager."""
    global _SESSION_CONTEXT_MANAGER
    if _SESSION_CONTEXT_MANAGER is None:
        _SESSION_CONTEXT_MANAGER = SessionContextManager()
    return _SESSION_CONTEXT_MANAGER


AURA_SYSTEM_PROMPT = (
    "You are Aura, an AI orchestrator with personality. You help developers build "
    "clean code by breaking requests into focused sessions.\n\n"
    "Traits:\n"
    "- Enthusiastic but not annoying\n"
    "- Speaks like a helpful senior dev\n"
    '- Casual language ("let\'s", "we\'re gonna", "looks good")\n'
    '- Celebrates wins ("Nice!", "Boom!")\n'
    "- Honest about challenges\n"
    "- Explains technical choices clearly\n"
    "- NO corporate speak, NO robot language\n\n"
    "You use specialized tools to accomplish tasks. Your key tools are:\n"
    "- execute_python_session: Generates and executes Python code to build features\n"
    "- clear_session_context: Clears session history when starting fresh work\n"
    "- read_project_file: Reads existing project files to understand the codebase\n"
    "- read_multiple_files: Reads multiple files at once for better context\n"
    "- list_project_files: Lists files in the project to discover what exists\n"
    "- git_commit: Commits changes to version control\n"
    "- git_push: Pushes commits to the remote repository\n"
    "- get_git_status: Checks the current git status\n"
    "- git_diff: Shows what changed before committing\n"
    "- run_tests: Runs pytest to verify code works correctly\n"
    "- lint_code: Catches errors and quality issues before running code\n"
    "- search_in_files: Finds code patterns and function signatures in the codebase\n"
    "- get_function_definitions: Extracts exact function signatures from files\n"
    "- install_package: Installs Python dependencies with pip\n"
    "- format_code: Auto-formats code using Black formatter\n\n"
    "You decide when to use each tool based on the user's request. When discussing plans "
    "or results, speak naturally in first person.\n\n"
    "IMPORTANT: Before calling functions from other files, use get_function_definitions to "
    "extract their exact signatures. This prevents parameter name mismatches.\n\n"
    "Example:\n"
    'Bad: "The system will now create the user model"\n'
    'Good: "Let me build that user model for you"\n'
)


def read_project_file(path: str) -> str:
    """Return the contents of a project file."""
    try:
        target = Path(path)
        if not target.is_absolute():
            target = Path.cwd() / target
        if not target.exists():
            return f"Error: file '{path}' does not exist."
        return target.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to read project file %s: %s", path, exc)
        return f"Error reading '{path}': {exc}"


def list_project_files(directory: str = ".", extension: str = ".py") -> list[str]:
    """List project files matching the given extension."""
    try:
        base = Path(directory)
        if not base.is_absolute():
            base = Path.cwd() / base
        if not base.exists():
            return []
        suffix = extension if extension.startswith(".") else f".{extension}"
        files = [_relative_to_cwd(path) for path in base.rglob(f"*{suffix}") if path.is_file()]
        return sorted(files)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception(
            "Failed to list project files in %s with extension %s: %s",
            directory,
            extension,
            exc,
        )
        return []

def _relative_to_cwd(path: Path) -> str:
    """Return a path relative to the current working directory when possible."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def get_git_status() -> str:
    """Return the short git status for the current repository."""
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "git status failed"
        LOGGER.error("git status failed: %s", error)
        return f"Error: {error}"
    return result.stdout.strip() or "clean"


def git_commit(message: str) -> str:
    """Commit all changes with the given message."""
    if not message or not message.strip():
        return "Error: commit message cannot be empty"

    add_result = subprocess.run(
        ["git", "add", "."],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if add_result.returncode != 0:
        error = add_result.stderr.strip() or "git add failed"
        LOGGER.error("git add failed: %s", error)
        return f"Error staging files: {error}"

    commit_result = subprocess.run(
        ["git", "commit", "-m", message.strip()],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        error = commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed"
        LOGGER.error("git commit failed: %s", error)
        return f"Error committing: {error}"

    output = commit_result.stdout.strip()
    LOGGER.info("Committed successfully: %s", message)
    return f"✅ Committed successfully: {message}\n{output}"


def git_push(remote: str = "origin", branch: str = "main") -> str:
    """Push commits to the remote repository."""
    result = subprocess.run(
        ["git", "push", remote, branch],
        cwd=os.getcwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "git push failed"
        LOGGER.error("git push failed: %s", error)
        return f"Error pushing to {remote}/{branch}: {error}"

    output = result.stdout.strip()
    LOGGER.info("Pushed successfully to %s/%s", remote, branch)
    return f"✅ Pushed successfully to {remote}/{branch}\n{output}"


def execute_python_session(session_prompt: str, working_directory: str) -> dict[str, object]:
    """Run a Python coder session using the local project."""
    LOGGER.info(
        "execute_python_session called: prompt_length=%d, working_directory=%s",
        len(session_prompt),
        working_directory,
    )
    context_manager = get_session_context_manager()

    try:
        agent = PythonCoderAgent(api_key=os.getenv("GEMINI_API_KEY", ""))
    except ValueError as exc:
        LOGGER.error("Failed to create PythonCoderAgent: %s", exc)
        return {
            "success": False,
            "summary": "",
            "files_created": [],
            "files_modified": [],
            "errors": [str(exc)],
        }

    try:
        working_dir = Path(working_directory) if working_directory else Path.cwd()
        project_files = list_project_files(str(working_dir))
        LOGGER.debug("Found %d project files in %s", len(project_files), working_dir)

        context = SessionContext(
            working_dir=working_dir,
            session_prompt=session_prompt,
            previous_work=context_manager.get_context(),
            project_files=project_files,
        )

        result = agent.execute_session(context)

        if result.success:
            files = list(result.files_created) + list(result.files_modified)
            ordered_files = list(dict.fromkeys(files))
            files_section = ", ".join(ordered_files) if ordered_files else "none"
            summary_text = (result.summary or "").strip() or "No summary provided"
            context_manager.add_entry(f"Session: {summary_text} | Files: {files_section}")

        LOGGER.info(
            "Session completed: success=%s, files_created=%d, files_modified=%d",
            result.success,
            len(result.files_created),
            len(result.files_modified),
        )

        return {
            "success": result.success,
            "summary": result.summary,
            "files_created": list(result.files_created),
            "files_modified": list(result.files_modified),
            "commands_run": list(result.commands_run),
            "output_lines": list(result.output_lines),
            "errors": list(result.errors),
            "duration_seconds": result.duration_seconds,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Python coding session failed: %s", exc)
        return {
            "success": False,
            "summary": "",
            "files_created": [],
            "files_modified": [],
            "commands_run": [],
            "output_lines": [],
            "errors": [f"Session execution failed: {exc}"],
        }


def clear_session_context() -> str:
    """Clear accumulated session context for a fresh start."""
    manager = get_session_context_manager()
    manager.clear()
    LOGGER.info("Session context cleared.")
    return "✅ Session context cleared. Ready for a new project!"


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


def git_diff(file_path: str = "", staged: bool = False) -> str:
    """Show git diff for changes in the repository.

    Args:
        file_path: Optional specific file to show diff for
        staged: If True, show staged changes; otherwise show unstaged (default: False)

    Returns:
        String containing the diff output, or empty string if no changes
    """
    try:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if file_path:
            cmd.append(file_path)

        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            error = result.stderr.strip() or "git diff failed"
            LOGGER.error("git diff failed: %s", error)
            return f"Error: {error}"

        output = result.stdout.strip()
        if not output:
            return ""

        LOGGER.info("Git diff retrieved: %d characters", len(output))
        return output

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to get git diff: %s", exc)
        return f"Error getting diff: {exc}"


def search_in_files(
    pattern: str,
    directory: str = ".",
    file_extension: str = ".py",
) -> dict[str, object]:
    """Search the codebase for a pattern and return matching lines.

    Args:
        pattern: The search term or regex pattern
        directory: Directory to search in (default: ".")
        file_extension: File extension to filter by (default: ".py")

    Returns:
        Dictionary with "matches" key containing list of matches.
        Each match has: file, line_number, content
    """
    try:
        base = Path(directory)
        if not base.is_absolute():
            base = Path.cwd() / base

        if not base.exists():
            LOGGER.error("Directory does not exist: %s", directory)
            return {"matches": []}

        suffix = file_extension if file_extension.startswith(".") else f".{file_extension}"
        matches = []

        for file_path in base.rglob(f"*{suffix}"):
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                for line_num, line in enumerate(content.split("\n"), start=1):
                    if pattern.lower() in line.lower():
                        matches.append({
                            "file": _relative_to_cwd(file_path),
                            "line_number": line_num,
                            "content": line.strip(),
                        })

                        if len(matches) >= 50:
                            LOGGER.info("Search hit 50 match limit")
                            return {"matches": matches}

            except (UnicodeDecodeError, PermissionError) as exc:
                LOGGER.debug("Skipping file %s: %s", file_path, exc)
                continue

        LOGGER.info("Search found %d matches for pattern: %s", len(matches), pattern)
        return {"matches": matches}

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to search files: %s", exc)
        return {"matches": []}


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
            error = result.stderr.strip() or result.stdout.strip() or "pip install failed"
            LOGGER.error("pip install failed for %s: %s", package_spec, error)
            return f"Error installing {package_spec}: {error}"

        output = result.stdout.strip()
        LOGGER.info("Package installed successfully: %s", package_spec)
        return f"✅ Successfully installed {package_spec}\n{output}"

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to install package %s: %s", package, exc)
        return f"Error installing package: {exc}"


def format_code(file_paths: list[str] | None = None, directory: str = ".") -> dict[str, object]:
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

        LOGGER.info("Code formatting completed: formatted=%d, errors=%d", formatted_count, len(errors))
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

                functions.append({
                    "name": node.name,
                    "params": params,
                    "line": node.lineno,
                    "docstring": docstring or "",
                })

        LOGGER.info("Extracted %d function definitions from %s", len(functions), file_path)
        return functions

    except SyntaxError as exc:
        LOGGER.error("Syntax error in file %s: %s", file_path, exc)
        return []
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to extract function definitions from %s: %s", file_path, exc)
        return []


def read_multiple_files(file_paths: list[str]) -> dict[str, str]:
    """Read multiple project files at once.

    Args:
        file_paths: List of file paths to read

    Returns:
        Dictionary mapping file paths to their contents
        Example: {"file1.py": "content...", "file2.py": "content..."}
    """
    if not file_paths:
        return {}

    results = {}
    for path in file_paths:
        try:
            target = Path(path)
            if not target.is_absolute():
                target = Path.cwd() / target

            if not target.exists():
                results[path] = f"Error: file '{path}' does not exist."
                continue

            if not target.is_file():
                results[path] = f"Error: '{path}' is not a file."
                continue

            content = target.read_text(encoding="utf-8")
            results[path] = content

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read file %s: %s", path, exc)
            results[path] = f"Error reading '{path}': {exc}"

    LOGGER.info("Read %d files: %d successful", len(file_paths), sum(1 for v in results.values() if not v.startswith("Error")))
    return results


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

        LOGGER.info("Linting completed: errors=%d, warnings=%d, score=%.2f", len(errors), len(warnings), score)
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


@dataclass
class ChatMessage:
    """Represents a single chat message in the conversation history."""

    role: str
    content: str


@dataclass
class ChatService:
    """Streams conversational replies from Gemini with Aura's personality."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "gemini-2.5-pro"
    _history: list[ChatMessage] = field(default_factory=list, init=False)
    _client_configured: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate API key and configure client."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        genai.configure(api_key=self.api_key)
        self._client_configured = True

    def clear_session_context(self) -> None:
        """Reset stored tool session context."""
        get_session_context_manager().clear()
        LOGGER.info("ChatService cleared session context.")

    def send_message(self, message: str) -> Iterator[str]:
        """Send a message and yield the streaming response."""
        if not message:
            raise ValueError("Message must be a non-empty string.")
        self._history.append(ChatMessage(role="user", content=message))
        LOGGER.debug("Sending chat message: %s", message)

        model = genai.GenerativeModel(
            self.model,
            system_instruction=AURA_SYSTEM_PROMPT,
            tools=[
                read_project_file,
                list_project_files,
                get_git_status,
                git_commit,
                git_push,
                execute_python_session,
                clear_session_context,
                run_tests,
                git_diff,
                search_in_files,
                install_package,
                format_code,
                get_function_definitions,
                read_multiple_files,
                lint_code,
            ],
        )

        chat_history = [
            {"role": msg.role, "parts": [msg.content]}
            for msg in self._history
            if msg.role != "system"
        ]

        response = model.generate_content(
            chat_history,
            stream=True,
        )
        collected = []
        try:
            for chunk in response:
                text = chunk.text or ""
                if text:
                    collected.append(text)
                    yield text
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Chat streaming failed: %s", exc)
            yield "[Aura] Uh-oh, something went sideways talking to Gemini."
            return
        combined = "".join(collected)
        if combined:
            self._history.append(ChatMessage(role="model", content=combined))

    def get_history(self) -> list[Mapping[str, str]]:
        """Return the chat history including system prompt."""
        return [{"role": entry.role, "content": entry.content} for entry in self._history]

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self._history.clear()