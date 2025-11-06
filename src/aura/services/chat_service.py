"""Conversational interface for Aura with Gemini streaming."""

from __future__ import annotations

import logging
import os
from functools import wraps
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Iterator, Mapping

import google.generativeai as genai
from aura.agents import PythonCoderAgent, SessionContext
from aura.tools import (
    format_code,
    get_function_definitions,
    get_git_status,
    git_commit,
    git_diff,
    git_push,
    install_package,
    lint_code,
    list_project_files,
    read_multiple_files,
    read_project_file,
    run_tests,
    search_in_files,
)

LOGGER = logging.getLogger(__name__)

# Preserve original tool implementations before wrapping them with high-visibility logging.
_ORIG_READ_PROJECT_FILE = read_project_file
_ORIG_LIST_PROJECT_FILES = list_project_files
_ORIG_SEARCH_IN_FILES = search_in_files
_ORIG_GET_FUNCTION_DEFINITIONS = get_function_definitions
_ORIG_READ_MULTIPLE_FILES = read_multiple_files
_ORIG_GET_GIT_STATUS = get_git_status
_ORIG_GIT_COMMIT = git_commit
_ORIG_GIT_PUSH = git_push
_ORIG_GIT_DIFF = git_diff
_ORIG_RUN_TESTS = run_tests
_ORIG_LINT_CODE = lint_code
_ORIG_FORMAT_CODE = format_code
_ORIG_INSTALL_PACKAGE = install_package


@wraps(_ORIG_READ_PROJECT_FILE)
def read_project_file(path: str) -> str:
    LOGGER.warning("🚨 TOOL_CALL read_project_file path=%s", path)
    return _ORIG_READ_PROJECT_FILE(path)


@wraps(_ORIG_LIST_PROJECT_FILES)
def list_project_files(directory: str = ".", extension: str = ".py") -> list[str]:
    LOGGER.warning("🚨 TOOL_CALL list_project_files directory=%s extension=%s", directory, extension)
    return _ORIG_LIST_PROJECT_FILES(directory=directory, extension=extension)


@wraps(_ORIG_SEARCH_IN_FILES)
def search_in_files(
    pattern: str, directory: str = ".", file_extension: str = ".py"
) -> dict[str, object]:
    LOGGER.warning("🚨 TOOL_CALL search_in_files pattern=%s directory=%s extension=%s", pattern, directory, file_extension)
    return _ORIG_SEARCH_IN_FILES(pattern=pattern, directory=directory, file_extension=file_extension)


@wraps(_ORIG_GET_FUNCTION_DEFINITIONS)
def get_function_definitions(file_path: str) -> list[dict[str, object]]:
    LOGGER.warning("🚨 TOOL_CALL get_function_definitions file_path=%s", file_path)
    return _ORIG_GET_FUNCTION_DEFINITIONS(file_path)


@wraps(_ORIG_READ_MULTIPLE_FILES)
def read_multiple_files(file_paths: list[str]) -> dict[str, str]:
    LOGGER.warning("🚨 TOOL_CALL read_multiple_files file_count=%d paths=%s", len(file_paths), file_paths)
    return _ORIG_READ_MULTIPLE_FILES(file_paths)


@wraps(_ORIG_GET_GIT_STATUS)
def get_git_status() -> str:
    LOGGER.warning("🚨 TOOL_CALL get_git_status")
    return _ORIG_GET_GIT_STATUS()


@wraps(_ORIG_GIT_COMMIT)
def git_commit(message: str) -> str:
    LOGGER.warning("🚨 TOOL_CALL git_commit message=%s", message)
    return _ORIG_GIT_COMMIT(message)


@wraps(_ORIG_GIT_PUSH)
def git_push(remote: str = "origin", branch: str = "main") -> str:
    LOGGER.warning("🚨 TOOL_CALL git_push remote=%s branch=%s", remote, branch)
    return _ORIG_GIT_PUSH(remote=remote, branch=branch)


@wraps(_ORIG_GIT_DIFF)
def git_diff(file_path: str = "", staged: bool = False) -> str:
    LOGGER.warning("🚨 TOOL_CALL git_diff file_path=%s staged=%s", file_path or "<all>", staged)
    return _ORIG_GIT_DIFF(file_path=file_path, staged=staged)


@wraps(_ORIG_RUN_TESTS)
def run_tests(
    test_path: str = "tests/", verbose: bool = False
) -> dict[str, object]:
    LOGGER.warning("🚨 TOOL_CALL run_tests test_path=%s verbose=%s", test_path, verbose)
    return _ORIG_RUN_TESTS(test_path=test_path, verbose=verbose)


@wraps(_ORIG_LINT_CODE)
def lint_code(
    file_paths: list[str] | None = None, directory: str = "."
) -> dict[str, object]:
    LOGGER.warning("🚨 TOOL_CALL lint_code file_paths=%s directory=%s", file_paths, directory)
    return _ORIG_LINT_CODE(file_paths=file_paths, directory=directory)


@wraps(_ORIG_FORMAT_CODE)
def format_code(
    file_paths: list[str] | None = None, directory: str = "."
) -> dict[str, object]:
    LOGGER.warning("🚨 TOOL_CALL format_code file_paths=%s directory=%s", file_paths, directory)
    return _ORIG_FORMAT_CODE(file_paths=file_paths, directory=directory)


@wraps(_ORIG_INSTALL_PACKAGE)
def install_package(package: str, version: str = "") -> str:
    LOGGER.warning("🚨 TOOL_CALL install_package package=%s version=%s", package, version)
    return _ORIG_INSTALL_PACKAGE(package=package, version=version)


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
    "WORKFLOW - ALWAYS follow these steps:\n"
    "1. UNDERSTAND: Before building anything new:\n"
    "   - Call search_in_files() to see what already exists\n"
    "   - Call get_function_definitions() to see exact function signatures\n"
    "   - Call read_project_file() or read_multiple_files() for implementation details\n"
    "2. PLAN: Break the work into focused sessions based on what you learned\n"
    "3. BUILD: Execute sessions one at a time\n"
    "4. VERIFY: After each session, optionally run tests or check code quality\n\n"
    "IMPORTANT: Before calling functions from other files, use get_function_definitions to "
    "extract their exact signatures. This prevents parameter name mismatches.\n\n"
    "Example:\n"
    'Bad: "The system will now create the user model"\n'
    'Good: "Let me build that user model for you"\n'
)




def execute_python_session(session_prompt: str, working_directory: str) -> dict[str, object]:
    """Run a Python coder session using the local project."""
    LOGGER.warning("🚨 TOOL_CALL execute_python_session working_directory=%s prompt_chars=%d", working_directory, len(session_prompt))
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
    LOGGER.warning("🚨 TOOL_CALL clear_session_context invoked")
    manager = get_session_context_manager()
    manager.clear()
    LOGGER.info("Session context cleared.")
    return "✅ Session context cleared. Ready for a new project!"




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
                if chunk.function_calls:
                    for fc in chunk.function_calls:
                        args_summary = ", ".join([f"{k}={v}" for k, v in fc.args.items()])
                        yield f"TOOL_CALL::{fc.name}::{args_summary}"

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
