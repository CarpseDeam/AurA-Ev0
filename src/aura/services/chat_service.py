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
    "CRITICAL MANDATORY REQUIREMENTS - TOOL USAGE IS NOT OPTIONAL:\n\n"
    "YOU MUST USE TOOLS BEFORE GENERATING ANY CODE. This is not a suggestion.\n\n"
    "REQUIRED WORKFLOW - Follow this EXACT order every single time:\n\n"
    "1. DISCOVERY PHASE (MANDATORY - Do this FIRST):\n"
    "   - ALWAYS call list_project_files() to see what files exist\n"
    "   - ALWAYS call search_in_files() to find relevant code patterns\n"
    "   - If you skip this step, you are FAILING\n\n"
    "2. UNDERSTANDING PHASE (MANDATORY - Do this SECOND):\n"
    "   - ALWAYS call get_function_definitions() for ANY file you will modify or call\n"
    "   - ALWAYS call read_project_file() to see implementation details\n"
    "   - NEVER guess at function signatures - ALWAYS verify with tools\n"
    "   - If you guess instead of checking, you are FAILING\n\n"
    "3. PLANNING PHASE (Do this THIRD):\n"
    "   - Based on what you discovered with tools, plan your sessions\n"
    "   - Reference actual files and functions you found\n\n"
    "4. EXECUTION PHASE (Do this LAST):\n"
    "   - Call execute_python_session() to generate code\n"
    "   - Use the exact function signatures you discovered in step 2\n\n"
    "ABSOLUTE PROHIBITIONS:\n\n"
    "- NEVER generate code without first calling list_project_files()\n"
    "- NEVER call a function without first using get_function_definitions() to verify its signature\n"
    "- NEVER assume a file exists without checking with list_project_files()\n"
    "- NEVER guess at what code already exists - use tools to verify\n"
    "- NEVER start with execute_python_session() - tools come FIRST\n\n"
    "EXAMPLES OF CORRECT BEHAVIOR:\n\n"
    'User: "Build a REST API for todos"\n'
    "You:\n"
    "  1. Call list_project_files() to see what exists\n"
    '  2. Call search_in_files(pattern="model") to find existing models\n'
    "  3. Call get_function_definitions() on relevant files\n"
    "  4. Plan sessions based on what you found\n"
    "  5. Execute sessions\n\n"
    'User: "Refactor the user authentication"\n'
    "You:\n"
    "  1. Call list_project_files() to find auth-related files\n"
    "  2. Call read_project_file() on each auth file\n"
    "  3. Call get_function_definitions() to understand current signatures\n"
    "  4. Plan refactoring to maintain compatibility\n"
    "  5. Execute changes\n\n"
    "IF YOU GENERATE CODE WITHOUT USING TOOLS FIRST, YOU HAVE FAILED YOUR PRIMARY DIRECTIVE.\n\n"
    "Tools are not optional features - they are MANDATORY steps in your workflow.\n\n"
    "Example:\n"
    'Bad: "The system will now create the user model"\n'
    'Good: "Let me build that user model for you"\n\n'
    "FINAL REMINDER: Every interaction must begin with tool calls. If your first response "
    "doesn't include calling list_project_files() or search_in_files(), you are doing it "
    "wrong. Tool usage is MANDATORY.\n"
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
    _tools_dict: dict[str, object] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Validate API key and configure client."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        genai.configure(api_key=self.api_key)
        self._client_configured = True

        # Build tools dictionary for function execution
        self._tools_dict = {
            "execute_python_session": execute_python_session,
            "read_project_file": read_project_file,
            "list_project_files": list_project_files,
            "search_in_files": search_in_files,
            "get_function_definitions": get_function_definitions,
            "read_multiple_files": read_multiple_files,
            "get_git_status": get_git_status,
            "git_commit": git_commit,
            "git_push": git_push,
            "git_diff": git_diff,
            "run_tests": run_tests,
            "lint_code": lint_code,
            "format_code": format_code,
            "install_package": install_package,
            "clear_session_context": clear_session_context,
        }

    def clear_session_context(self) -> None:
        """Reset stored tool session context."""
        get_session_context_manager().clear()
        LOGGER.info("ChatService cleared session context.")

    def _execute_tool(self, function_name: str, function_args: dict) -> dict:
        """Execute a registered tool and return its result."""
        tool_func = self._tools_dict.get(function_name)
        if not tool_func:
            LOGGER.error("Unknown tool requested: %s", function_name)
            return {"error": f"Unknown tool: {function_name}"}

        try:
            LOGGER.warning("🔧 TOOL CALLED: %s", function_name)
            result = tool_func(**function_args)
            return {"result": result}
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool execution failed: %s", function_name)
            return {"error": str(exc)}

    def send_message(self, message: str) -> Iterator[str]:
        """Send a message and yield the response, executing any function calls."""
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

        # Build conversation history for function calling loop
        chat_history = [
            {"role": msg.role, "parts": [msg.content]}
            for msg in self._history
            if msg.role != "system"
        ]

        # Multi-turn conversation loop for function calling
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        try:
            while iteration < max_iterations:
                iteration += 1
                LOGGER.debug("Function calling loop iteration %d", iteration)

                # Use non-streaming generate_content for function calling
                response = model.generate_content(chat_history, stream=False)

                # Check if response has function calls
                if not response.candidates:
                    LOGGER.warning("No candidates in response")
                    yield "[Aura] I didn't get a proper response from Gemini."
                    return

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    LOGGER.warning("No content parts in response")
                    yield "[Aura] Got an empty response from Gemini."
                    return

                # Check first part for function call
                first_part = candidate.content.parts[0]
                if hasattr(first_part, "function_call") and first_part.function_call:
                    # Model wants to call a function
                    function_call = first_part.function_call
                    function_name = function_call.name
                    function_args = dict(function_call.args) if function_call.args else {}

                    LOGGER.info("Model requested function call: %s with args: %s", function_name, function_args)

                    # Notify user about the tool call
                    args_summary = ", ".join([f"{k}={v}" for k, v in function_args.items()])
                    yield f"🔧 Calling tool: {function_name}({args_summary})\n\n"

                    # Execute the function
                    result = self._execute_tool(function_name, function_args)

                    # Append model's function call to conversation history
                    chat_history.append({"role": "model", "parts": [first_part]})

                    # Append function response to conversation history
                    function_response_part = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=function_name,
                            response=result
                        )
                    )
                    chat_history.append({"role": "user", "parts": [function_response_part]})

                    # Continue the loop to get next response from model
                    continue

                else:
                    # No function call - we got the final text response
                    try:
                        final_text = response.text
                        if final_text:
                            self._history.append(ChatMessage(role="model", content=final_text))
                            yield final_text
                        else:
                            yield "[Aura] Got an empty text response."
                        return
                    except Exception as text_exc:  # noqa: BLE001
                        LOGGER.exception("Failed to extract text from response: %s", text_exc)
                        yield "[Aura] Something went wrong getting the response text."
                        return

            # If we hit max iterations
            LOGGER.warning("Hit max function calling iterations (%d)", max_iterations)
            yield "[Aura] I got stuck in a loop calling tools. Let me know if you want to try again."

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Chat with function calling failed: %s", exc)
            yield "[Aura] Uh-oh, something went sideways talking to Gemini."
            return

    def get_history(self) -> list[Mapping[str, str]]:
        """Return the chat history including system prompt."""
        return [{"role": entry.role, "content": entry.content} for entry in self._history]

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self._history.clear()
