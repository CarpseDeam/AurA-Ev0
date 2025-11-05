"""Conversational interface for Aura with Gemini streaming."""

from __future__ import annotations

import logging
import os
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
