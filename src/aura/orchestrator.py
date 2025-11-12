"""Simplified conversation orchestrator for Aura's single-agent workflow."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from aura import config
from aura.exceptions import AuraConfigurationError, AuraExecutionError
from aura.models import Conversation, Message, MessageRole, Project
from aura.services.local_summarizer_service import LocalSummarizerService
from aura.services.simple_agent_service import AgentTool, SingleAgentService
from aura.state import AppState
from aura.tools.anthropic_tool_builder import build_anthropic_tool_schema
from aura.tools.local_agent_tools import generate_commit_message
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)

RUN_START_MESSAGE = "? Kickstarting Aura run..."
WORKSPACE_CHECK_MESSAGE = "?? Validating workspace..."
AGENT_START_MESSAGE = "?? Claude agent: investigating and executing..."
SUCCESS_MESSAGE = "? Conversation complete"
FAILURE_MESSAGE = "? Conversation failed"


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


class _SingleAgentWorker(QObject):
    """Background worker that streams a SingleAgentService run."""

    chunk_emitted = Signal(str)
    finished = Signal(_ConversationOutcome)
    failed = Signal(str)

    def __init__(
        self,
        agent: SingleAgentService,
        goal: str,
        tools: List[AgentTool],
    ) -> None:
        super().__init__()
        self._agent = agent
        self._goal = goal
        self._tools = tools

    def run(self) -> None:
        """Execute the single-agent flow and stream chunks to listeners."""
        started = time.perf_counter()
        generator = self._agent.execute_task(self._goal, self._tools)
        collected: List[str] = []
        try:
            while True:
                try:
                    chunk = next(generator)
                except StopIteration as stop:
                    response = stop.value if stop.value is not None else "".join(collected)
                    outcome = _ConversationOutcome(
                        response=response.strip(),
                        duration_seconds=time.perf_counter() - started,
                        success=True,
                    )
                    self.finished.emit(outcome)
                    return
                if not chunk:
                    continue
                collected.append(chunk)
                self.chunk_emitted.emit(f"{config.STREAM_PREFIX}{chunk}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Single-agent worker failed")
            self.failed.emit(str(exc))


class Orchestrator(QObject):
    """Coordinate a conversational turn between the user and SingleAgentService."""

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
        app_state: AppState,
        *,
        single_agent: SingleAgentService,
        use_background_thread: bool = True,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self.app_state = app_state
        resolved = Path(app_state.working_directory).resolve()
        if not resolved.is_dir():
            raise AuraConfigurationError(
                "Working directory does not exist.",
                context={"path": str(resolved)},
            )

        self._working_dir = resolved
        self._single_agent = single_agent
        self._use_background_thread = use_background_thread
        self._history: List[Tuple[str, str]] = []
        self._thread: QThread | None = None
        self._worker: _SingleAgentWorker | None = None
        self._tool_manager = ToolManager(str(self._working_dir))
        self._agent_tools_cache: List[AgentTool] | None = None
        self._conversation_lock = threading.Lock()

        endpoint = (self.app_state.local_model_endpoint or "").strip()
        self._summarizer: LocalSummarizerService | None = None
        if endpoint:
            self._summarizer = LocalSummarizerService(endpoint=endpoint)
        else:
            LOGGER.debug("Local summarizer endpoint not configured; history summarization disabled.")

    def execute_goal(self, goal: str) -> None:
        """Execute a single conversational turn for the provided goal."""
        self.progress_update.emit(RUN_START_MESSAGE)

        try:
            self.progress_update.emit(WORKSPACE_CHECK_MESSAGE)
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
        self.session_started.emit(0, session)

        conversation_id = self._ensure_active_conversation()
        self.load_conversation_history(conversation_id)

        formatted_goal = self._build_prompt(sanitized)
        tools = self._build_agent_tools()
        self._single_agent.active_conversation_id = conversation_id

        if self._use_background_thread:
            self._start_single_agent_execution(session, formatted_goal, tools)
        else:
            self._run_single_agent_execution(session, formatted_goal, tools)

    def update_working_directory(self, path: str) -> None:
        """Update the working directory used when building prompts."""
        resolved = Path(path).resolve()
        if not resolved.is_dir():
            raise AuraConfigurationError(
                "Working directory does not exist.",
                context={"path": str(resolved)},
            )
        LOGGER.info("Orchestrator updating working directory to %s", resolved)
        self._working_dir = resolved
        if hasattr(self, "_tool_manager") and self._tool_manager is not None:
            self._tool_manager.update_workspace(str(self._working_dir))
        else:
            self._tool_manager = ToolManager(str(self._working_dir))
        self._agent_tools_cache = None
        LOGGER.info("ToolManager bound to workspace %s", self._tool_manager.workspace_dir)

    def reset_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()

    def _ensure_active_conversation(self) -> int:
        """Ensure a conversation exists so intermediate artifacts can be stored."""
        conversation_id = self.app_state.current_conversation_id
        if conversation_id is not None:
            return conversation_id
        with self._conversation_lock:
            conversation_id = self.app_state.current_conversation_id
            if conversation_id is None:
                project_id = self.app_state.current_project_id
                conv = Conversation.create(project_id=project_id)
                conversation_id = conv.id
                self.app_state.set_current_conversation(conversation_id)
                LOGGER.info("Created conversation %s for new execution", conversation_id)
        return conversation_id

    @property
    def history(self) -> Tuple[Tuple[str, str], ...]:
        """Return the accumulated conversation history."""
        return tuple(self._history)

    def _build_agent_tools(self) -> List[AgentTool]:
        """Construct tool schemas for the single agent."""
        if self._agent_tools_cache is not None:
            return self._agent_tools_cache

        handlers = {
            "list_project_files": self._tool_manager.list_project_files,
            "search_in_files": self._tool_manager.search_in_files,
            "read_project_file": self._tool_manager.read_project_file,
            "read_multiple_files": self._tool_manager.read_multiple_files,
            "get_function_definitions": self._tool_manager.get_function_definitions,
            "run_tests": self._tool_manager.run_tests,
            "lint_code": self._tool_manager.lint_code,
            "format_code": self._tool_manager.format_code,
            "install_package": self._tool_manager.install_package,
            "get_git_status": self._tool_manager.get_git_status,
            "git_commit": self._tool_manager.git_commit,
            "git_push": self._tool_manager.git_push,
            "git_diff": self._tool_manager.git_diff,
            "generate_commit_message": generate_commit_message,
            "find_definition": self._tool_manager.find_definition,
            "find_usages": self._tool_manager.find_usages,
            "get_imports": self._tool_manager.get_imports,
            "create_file": self._tool_manager.create_file,
            "modify_file": self._tool_manager.modify_file,
            "replace_file_lines": self._tool_manager.replace_file_lines,
            "delete_file": self._tool_manager.delete_file,
        }

        tools: List[AgentTool] = []
        for name, handler in handlers.items():
            schema = build_anthropic_tool_schema(handler, name=name)
            tools.append(AgentTool(name=name, handler=handler, schema=schema))

        self._agent_tools_cache = tools
        return tools

    def _start_single_agent_execution(
        self,
        session: ConversationSession,
        goal: str,
        tools: List[AgentTool],
    ) -> None:
        """Execute the single-agent flow on a background QThread."""
        self.progress_update.emit(AGENT_START_MESSAGE)
        self._thread = QThread(self)
        self._worker = _SingleAgentWorker(self._single_agent, goal, tools)
        self._worker.moveToThread(self._thread)
        self._worker.chunk_emitted.connect(self.session_output.emit)
        self._worker.finished.connect(
            lambda outcome: self._finalize_conversation(session, outcome)
        )
        self._worker.failed.connect(self._handle_worker_error)
        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _run_single_agent_execution(
        self,
        session: ConversationSession,
        goal: str,
        tools: List[AgentTool],
    ) -> None:
        """Execute the single-agent flow synchronously (useful for testing)."""
        self.progress_update.emit(AGENT_START_MESSAGE)
        started = time.perf_counter()
        collected: List[str] = []
        generator = self._single_agent.execute_task(goal, tools)
        try:
            while True:
                try:
                    chunk = next(generator)
                except StopIteration as stop:
                    response = stop.value if stop.value is not None else "".join(collected)
                    outcome = _ConversationOutcome(
                        response=response.strip(),
                        duration_seconds=time.perf_counter() - started,
                        success=True,
                    )
                    self._finalize_conversation(session, outcome)
                    return
                if not chunk:
                    continue
                collected.append(chunk)
                self.session_output.emit(f"{config.STREAM_PREFIX}{chunk}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Single-agent execution failed")
            error = AuraExecutionError(
                "Single-agent execution failed. Verify API keys and connectivity.",
                context={"detail": str(exc)},
            )
            self._handle_worker_error(str(error))

    def _finalize_conversation(
        self,
        session: ConversationSession,
        outcome: _ConversationOutcome,
    ) -> None:
        """Update history, emit completion signals, and reset state."""
        self._history.append((MessageRole.USER, session.prompt))
        self._history.append((MessageRole.ASSISTANT, outcome.response.strip()))
        self._persist_messages_to_db(session.prompt, outcome.response.strip())

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
        status_message = SUCCESS_MESSAGE if outcome.success else FAILURE_MESSAGE
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
        self.error_occurred.emit(message or "Single-agent execution failed.")
        self.all_sessions_complete.emit()
        self.progress_update.emit(FAILURE_MESSAGE)
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

    def _summarize_history(self, messages: List[Tuple[str, str]]) -> str:
        """Summarize historical messages using the local summarizer service."""
        if not self._summarizer or not messages:
            return ""

        try:
            try:
                return asyncio.run(self._summarizer.summarize_conversation(messages))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._summarizer.summarize_conversation(messages)
                    )
                finally:
                    loop.close()
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to summarize conversation history: %s", exc)
            return ""

    def _build_prompt(self, goal: str) -> str:
        """Build the conversational prompt, summarizing old history if needed."""
        SUMMARY_THRESHOLD = 30  # 15 pairs of messages
        RECENT_MESSAGES_COUNT = 10  # 5 pairs of messages

        lines: List[str] = []
        summary = ""
        project_instructions = ""
        history = list(self._history)
        recent_history: List[Tuple[str, str]] = history

        project_id = self.app_state.current_project_id
        if project_id:
            project = Project.get_by_id(project_id)
            if project and project.custom_instructions:
                project_instructions = project.custom_instructions

        history_length = len(history)
        if history_length > SUMMARY_THRESHOLD:
            if not self._summarizer:
                LOGGER.debug(
                    "History length %s exceeds threshold but no summarizer is configured.",
                    history_length,
                )
            else:
                split_index = max(history_length - RECENT_MESSAGES_COUNT, 0)
                old_history = history[:split_index]
                recent_history = history[split_index:] or history
                if old_history:
                    LOGGER.info("Summarizing %s old messages...", len(old_history))
                    raw_summary = self._summarize_history(old_history)
                    cleaned_summary = raw_summary.strip()
                    if cleaned_summary and not cleaned_summary.lower().startswith("error:"):
                        summary = cleaned_summary
                        LOGGER.info("Summarization complete.")
                    else:
                        if cleaned_summary:
                            LOGGER.warning(
                                "Summarization returned an error. Falling back to full history: %s",
                                cleaned_summary,
                            )
                        else:
                            LOGGER.warning(
                                "Summarization unavailable. Using full conversation history."
                            )
                        summary = ""
                        recent_history = history
                else:
                    recent_history = history

        if project_instructions:
            lines.append("Project Instructions:")
            lines.append(project_instructions)
            lines.append("\n---")

        if summary:
            lines.append("Here is a summary of the conversation so far:")
            lines.append(summary)
            lines.append("\nContinuing the conversation:")

        for role, text in recent_history:
            if role == MessageRole.USER:
                prefix = "User"
            elif role == MessageRole.ASSISTANT:
                prefix = "Assistant"
            else:
                prefix = role.title() if role else "Message"
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

    def load_conversation_history(self, conversation_id: int) -> None:
        """Load conversation history from database into memory."""
        try:
            conv = Conversation.get_by_id(conversation_id)
            if conv:
                self._history = conv.get_history()
                LOGGER.info(
                    "Loaded %s messages from conversation %s",
                    len(self._history),
                    conversation_id,
                )
            else:
                LOGGER.warning("Conversation %s not found", conversation_id)
                self._history = []
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load conversation history: %s", exc)
            self._history = []

    def _persist_messages_to_db(self, user_message: str, assistant_message: str) -> None:
        """Persist user and assistant messages to the database."""
        try:
            conversation_id = self.app_state.current_conversation_id

            if conversation_id is None:
                with self._conversation_lock:
                    conversation_id = self.app_state.current_conversation_id
                    if conversation_id is None:
                        project_id = self.app_state.current_project_id
                        conv = Conversation.create(project_id=project_id)
                        conversation_id = conv.id
                        self.app_state.set_current_conversation(conversation_id)
                        LOGGER.info("Created new conversation %s", conversation_id)

            Message.create(conversation_id, MessageRole.USER, user_message)
            Message.create(conversation_id, MessageRole.ASSISTANT, assistant_message)

            conv = Conversation.get_by_id(conversation_id)
            if conv and not conv.title:
                conv.generate_title_from_first_message()

            LOGGER.info("Persisted messages to conversation %s", conversation_id)

        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to persist messages to database: %s", exc)
            self.error_occurred.emit(
                "Warning: Failed to save conversation history. Your responses will not persist."
            )


__all__ = ["ConversationSession", "Orchestrator", "SessionResult"]
