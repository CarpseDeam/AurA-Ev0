"""Simplified conversation orchestrator for Aura.

This module replaces the previous multi-session orchestration pipeline with a
lightweight conversational flow that streams responses from the ChatService
directly to the UI.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from aura import config
from aura.exceptions import AuraConfigurationError, AuraExecutionError
from aura.services.chat_service import ChatService

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionResult:
    """Outcome details for a single conversation turn."""

    session_name: str
    exit_code: int
    duration_seconds: float
    files_created: List[str]
    success: bool


@dataclass(frozen=True)
class ConversationSession:
    """Metadata describing the active conversation turn."""

    name: str
    prompt: str


@dataclass(frozen=True)
class _ConversationOutcome:
    """Internal helper capturing the result of a conversation turn."""

    response: str
    duration_seconds: float
    success: bool


class _ConversationWorker(QObject):
    """Background worker that performs a single ChatService call."""

    chunk_emitted = Signal(str)
    finished = Signal(_ConversationOutcome)
    failed = Signal(str)

    def __init__(self, chat_service: ChatService, prompt: str) -> None:
        super().__init__()
        self._chat_service = chat_service
        self._prompt = prompt

    def run(self) -> None:
        """Execute the chat request and stream chunks to listeners."""
        started = time.perf_counter()
        streamed = {"value": False}

        def _on_chunk(chunk: str) -> None:
            streamed["value"] = True
            self.chunk_emitted.emit(chunk)

        try:
            response = self._chat_service.send_message(self._prompt, on_chunk=_on_chunk)
            if not streamed["value"] and response:
                self.chunk_emitted.emit(f"{config.STREAM_PREFIX}{response}")
                self.chunk_emitted.emit(f"{config.STREAM_PREFIX}\n")

            duration = time.perf_counter() - started
            success = not response.strip().lower().startswith("error:")
            outcome = _ConversationOutcome(
                response=response,
                duration_seconds=duration,
                success=success,
            )
            self.finished.emit(outcome)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Conversation worker failed")
            self.failed.emit(str(exc))


class Orchestrator(QObject):
    """Coordinate a conversational turn between the user and ChatService."""

    planning_started = Signal()
    plan_ready = Signal(object)
    session_started = Signal(int, object)
    session_output = Signal(str)
    session_complete = Signal(int, object)
    all_sessions_complete = Signal()
    error_occurred = Signal(str)
    progress_update = Signal(str)

    def __init__(
        self,
        chat_service: ChatService,
        working_dir: str,
        agent_path: str,
        *,
        use_background_thread: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not chat_service:
            raise AuraConfigurationError("Chat service is required for orchestration.")

        resolved = Path(working_dir).resolve()
        if not resolved.is_dir():
            raise AuraConfigurationError(
                "Working directory does not exist.",
                context={"path": str(resolved)},
            )

        self._chat_service = chat_service
        self._working_dir = resolved
        self._agent_path = agent_path
        self._use_background_thread = use_background_thread
        self._history: List[Tuple[str, str]] = []
        self._thread: QThread | None = None
        self._worker: _ConversationWorker | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def execute_goal(self, goal: str) -> None:
        """Execute a single conversational turn for the provided goal."""
        try:
            self._validate_environment()
        except AuraConfigurationError as exc:
            LOGGER.error("Cannot execute goal: %s", exc)
            self.error_occurred.emit(str(exc))
            return

        sanitized = (goal or "").strip()
        if not sanitized:
            self.error_occurred.emit("Goal must be provided.")
            return

        if self._thread is not None and self._thread.isRunning():
            self.error_occurred.emit("A conversation is already in progress.")
            return

        session = ConversationSession(name="Conversation", prompt=sanitized)
        LOGGER.info(
            "Conversation started | goal_preview=%s | working_dir=%s",
            sanitized[:80],
            self._working_dir,
        )
        self.planning_started.emit()
        self.progress_update.emit("Starting conversation...")
        self.session_started.emit(0, session)

        prompt = self._build_prompt(sanitized)
        if self._use_background_thread:
            self._start_background_conversation(prompt)
        else:
            self._run_conversation(prompt)

    def update_agent_path(self, agent_path: str) -> None:
        """Update the remembered CLI agent path (used for manual runs)."""
        self._agent_path = agent_path

    def update_working_directory(self, path: str) -> None:
        """Update the working directory used when building prompts."""
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise AuraConfigurationError(
                "Working directory does not exist.",
                context={"path": str(resolved)},
            )
        self._working_dir = resolved

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    @property
    def history(self) -> Tuple[Tuple[str, str], ...]:
        """Return the accumulated conversation history."""
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_background_conversation(self, prompt: str) -> None:
        """Execute the conversation on a background QThread."""
        self._thread = QThread(self)
        self._worker = _ConversationWorker(self._chat_service, prompt)
        self._worker.moveToThread(self._thread)

        session = ConversationSession(name="Conversation", prompt=prompt)
        self._worker.chunk_emitted.connect(self.session_output.emit)
        self._worker.finished.connect(
            lambda outcome: self._finalize_conversation(session, outcome)
        )
        self._worker.failed.connect(self._handle_worker_error)

        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _run_conversation(self, prompt: str) -> None:
        """Execute the conversation synchronously (useful for testing)."""
        started = time.perf_counter()
        streamed = {"value": False}

        def _on_chunk(chunk: str) -> None:
            streamed["value"] = True
            self.session_output.emit(chunk)

        session = ConversationSession(name="Conversation", prompt=prompt)
        try:
            response = self._chat_service.send_message(prompt, on_chunk=_on_chunk)
            if not streamed["value"] and response:
                self.session_output.emit(f"{config.STREAM_PREFIX}{response}")
                self.session_output.emit(f"{config.STREAM_PREFIX}\n")

            duration = time.perf_counter() - started
            success = not response.strip().lower().startswith("error:")
            outcome = _ConversationOutcome(
                response=response,
                duration_seconds=duration,
                success=success,
            )
            self._finalize_conversation(session, outcome)
        except Exception as exc:  # noqa: BLE001
            error = AuraExecutionError(
                "Unable to contact Gemini. Verify GEMINI_API_KEY and your connection, then try again.",
                context={"detail": str(exc)},
            )
            LOGGER.exception("Conversation execution failed")
            self.error_occurred.emit(str(error))

    def _finalize_conversation(
        self,
        session: ConversationSession,
        outcome: _ConversationOutcome,
    ) -> None:
        """Update history, emit completion signals, and reset state."""
        self._history.append(("user", session.prompt))
        self._history.append(("assistant", outcome.response.strip()))

        exit_code = 0 if outcome.success else 1
        result = SessionResult(
            session_name=session.name,
            exit_code=exit_code,
            duration_seconds=outcome.duration_seconds,
            files_created=[],
            success=outcome.success,
        )

        if not outcome.success:
            message = outcome.response.strip() or "Conversation failed."
            self.error_occurred.emit(message)

        self.session_complete.emit(0, result)
        self.all_sessions_complete.emit()
        status_message = "Conversation complete" if outcome.success else "Conversation failed"
        self.progress_update.emit(status_message)
        LOGGER.info(
            "Conversation finished | success=%s | duration=%.2fs",
            outcome.success,
            outcome.duration_seconds,
        )

        if self._thread and self._thread.isRunning():
            self._thread.quit()

    def _handle_worker_error(self, message: str) -> None:
        """Handle errors raised by the background worker."""
        LOGGER.error("Conversation worker failed: %s", message)
        self.error_occurred.emit(
            message
            or "Unable to contact Gemini. Verify GEMINI_API_KEY and your connection, then try again."
        )
        self.all_sessions_complete.emit()
        self.progress_update.emit("Conversation failed")
        if self._thread and self._thread.isRunning():
            self._thread.quit()

    def _cleanup_thread(self) -> None:
        """Release worker/thread resources after completion."""
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def _build_prompt(self, goal: str) -> str:
        """Build the conversational prompt including prior turns."""
        if not self._history:
            return goal

        lines: List[str] = []
        for role, text in self._history:
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {text}")
        lines.append(f"User: {goal}")
        return "\n\n".join(lines)

    def _validate_environment(self) -> None:
        """Ensure the working directory remains valid."""
        if not self._working_dir.is_dir():
            raise AuraConfigurationError(
                "Workspace no longer exists. Select a valid working directory to continue.",
                context={"issue": "working_directory_missing", "path": str(self._working_dir)},
            )
        if self._agent_path:
            agent = Path(self._agent_path)
            if not agent.exists():
                raise AuraConfigurationError(
                    "Configured agent executable cannot be found. Please update your agent settings.",
                    context={"issue": "agent_missing", "path": str(agent)},
                )
            if not os.access(agent, os.X_OK):
                raise AuraConfigurationError(
                    "Configured agent is not executable. Please reconfigure the agent path.",
                    context={"issue": "agent_not_executable", "path": str(agent)},
                )
