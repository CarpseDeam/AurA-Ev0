"""Conversational interface for Aura with Gemini streaming."""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Mapping

import google.generativeai as genai
from aura.agents import PythonCoderAgent, SessionContext

LOGGER = logging.getLogger(__name__)

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
    "You coordinate CLI coding agents but users talk to YOU. When discussing plans "
    "or results, speak naturally in first person.\n\n"
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


def list_project_files(directory: str = ".", extension: str = ".py") -> List[str]:
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

    # Stage all changes
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

    # Commit with message
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
    try:
        agent = PythonCoderAgent(api_key=os.getenv("GEMINI_API_KEY", ""))
    except ValueError as exc:
        return {
            "success": False,
            "summary": "",
            "files_created": [],
            "files_modified": [],
            "errors": [str(exc)],
        }

    context = SessionContext(
        working_dir=Path(working_directory) if working_directory else Path.cwd(),
        session_prompt=session_prompt,
        previous_work=(),
        project_files=(),
    )
    result = agent.execute_session(context)
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
    _history: List[ChatMessage] = field(default_factory=list, init=False)
    _client_configured: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate API key and configure client."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        genai.configure(api_key=self.api_key)
        self._client_configured = True

    def send_message(self, message: str) -> Iterator[str]:
        """Send a message and yield the streaming response."""
        if not message:
            raise ValueError("Message must be a non-empty string.")
        self._history.append(ChatMessage(role="user", content=message))
        LOGGER.debug("Sending chat message: %s", message)

        # Pass system prompt via system_instruction parameter
        model = genai.GenerativeModel(
            self.model,
            system_instruction=AURA_SYSTEM_PROMPT,
            tools=[read_project_file, list_project_files, get_git_status, git_commit, git_push, execute_python_session],
        )

        # Build chat history, excluding system messages (Gemini doesn't accept them)
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

    def get_history(self) -> List[Mapping[str, str]]:
        """Return the chat history including system prompt."""
        return [{"role": entry.role, "content": entry.content} for entry in self._history]

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self._history.clear()
