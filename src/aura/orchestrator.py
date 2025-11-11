"""Simplified conversation orchestrator for Aura.

This module replaces the previous multi-session orchestration pipeline with a
lightweight conversational flow that streams responses from the ChatService
directly to the UI.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from aura import config
from aura.exceptions import AuraConfigurationError, AuraExecutionError
from aura.services.chat_service import ChatService
from aura.services.analyst_agent_service import AnalystAgentService
from aura.services.executor_agent_service import ExecutorAgentService
from aura.services.local_summarizer_service import LocalSummarizerService
from aura.state import AppState
from aura.tools.tool_manager import ToolManager
from aura.models import Conversation, ExecutionPlan, Message, MessageRole, Project

LOGGER = logging.getLogger(__name__)

RUN_START_MESSAGE = "⚡ Kickstarting Aura run..."
WORKSPACE_CHECK_MESSAGE = "📁 Validating workspace..."
ANALYST_START_MESSAGE = "🧠 Analyst agent: drafting plan..."
EXECUTOR_START_MESSAGE = "⚙️ Executor agent: dispatching micro-actions..."
CONTEXT_BUILD_MESSAGE = "🧱 Compiling workspace context..."
MICRO_AGENT_START_MESSAGE = "🤖 Micro-agent: engaging tools..."
SUCCESS_MESSAGE = "✅ Conversation complete"
FAILURE_MESSAGE = "❌ Conversation failed"

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

    def __init__(
        self,
        chat_service: ChatService,
        prompt: str,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._chat_service = chat_service
        self._prompt = prompt
        self._progress_callback = progress_callback

    def _notify_progress(self, message: str) -> None:
        if not message or self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Progress callback failed for conversation worker", exc_info=True)

    def run(self) -> None:
        """Execute the chat request and stream chunks to listeners."""
        started = time.perf_counter()
        streamed = {"value": False}
        self._notify_progress(MICRO_AGENT_START_MESSAGE)

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
        except Exception as exc:
            LOGGER.exception("Conversation worker failed")
            self.failed.emit(str(exc))


class _TwoAgentWorker(QObject):
    """Background worker for two-agent execution flow."""

    chunk_emitted = Signal(str)
    finished = Signal(_ConversationOutcome)
    failed = Signal(str)

    def __init__(
        self,
        analyst_agent: AnalystAgentService,
        executor_agent: ExecutorAgentService,
        goal: str,
        conversation_id: int | None,
        *,
        conversation_history: List[dict] | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._analyst_agent = analyst_agent
        self._executor_agent = executor_agent
        self._goal = goal
        self._conversation_id = conversation_id
        self._conversation_history = conversation_history or []
        self._progress_callback = progress_callback

    def _notify_progress(self, message: str) -> None:
        if not message or self._progress_callback is None:
            return
        try:
            self._progress_callback(message)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Progress callback failed for two-agent worker", exc_info=True)

    def run(self) -> None:
        """Execute two-agent flow: analysis then execution."""
        started = time.perf_counter()
        streamed = {"value": False}

        def _on_chunk(chunk: str) -> None:
            streamed["value"] = True
            self.chunk_emitted.emit(chunk)

        try:
            self._notify_progress(ANALYST_START_MESSAGE)
            plan_or_error = self._analyst_agent.analyze_and_plan(
                self._goal,
                on_chunk=_on_chunk,
                conversation_id=self._conversation_id,
                conversation_history=self._conversation_history,
            )

            if isinstance(plan_or_error, str):
                error_msg = plan_or_error or "Analysis failed"
                self.failed.emit(error_msg)
                return

            execution_plan: ExecutionPlan = plan_or_error
            self._notify_progress(EXECUTOR_START_MESSAGE)
            result = self._executor_agent.execute_plan(
                execution_plan,
                on_chunk=_on_chunk,
                conversation_id=self._conversation_id,
            )

            duration = time.perf_counter() - started
            success = not result.strip().lower().startswith("error:")
            outcome = _ConversationOutcome(
                response=result,
                duration_seconds=duration,
                success=success,
            )
            self.finished.emit(outcome)

        except Exception as exc:
            LOGGER.exception("Two-agent worker failed")
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
        app_state: AppState,
        *,
        api_key: str | None = None,
        analyst_api_key: str | None = None,
        executor_api_key: str | None = None,
        chat_service: ChatService | None = None,
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
        self._use_background_thread = use_background_thread
        self._history: List[Tuple[str, str]] = []
        self._thread: QThread | None = None
        self._worker: _ConversationWorker | _TwoAgentWorker | None = None
        self._tool_manager = ToolManager(str(self._working_dir))
        self._conversation_lock = threading.Lock()

        endpoint = (self.app_state.local_model_endpoint or "").strip()
        self._summarizer: LocalSummarizerService | None = None
        if endpoint:
            self._summarizer = LocalSummarizerService(endpoint=endpoint)
        else:
            LOGGER.debug("Local summarizer endpoint not configured; history summarization disabled.")

        effective_analyst_key = analyst_api_key or api_key
        effective_executor_key = executor_api_key

        if chat_service is not None:
            if hasattr(chat_service, "tool_manager"):
                chat_service.tool_manager = self._tool_manager
            self._chat_service = chat_service
        else:
            if not effective_analyst_key:
                raise AuraConfigurationError(
                    "Analyst API key (ANTHROPIC_API_KEY) is required to initialize the chat service.",
                    context={"env_var": "ANTHROPIC_API_KEY"},
                )
            self._chat_service = ChatService(
                api_key=effective_analyst_key,
                tool_manager=self._tool_manager,
                model_name=self.app_state.analyst_model,
            )

        self._analyst_agent: AnalystAgentService | None = None
        self._executor_agent: ExecutorAgentService | None = None

        if effective_analyst_key and effective_executor_key:
            self._analyst_agent = AnalystAgentService(
                api_key=effective_analyst_key,
                tool_manager=self._tool_manager,
                investigation_model=self.app_state.analyst_investigation_model,
                planning_model=self.app_state.analyst_planning_model,
            )
            self._executor_agent = ExecutorAgentService(
                api_key=effective_executor_key,
                tool_manager=self._tool_manager,
                model_name=self.app_state.executor_model,
            )

    def execute_goal(self, goal: str) -> None:
        """Execute a single conversational turn for the provided goal."""
        # IMMEDIATE progress update
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

        conversation_id: int | None = None
        if self._analyst_agent and self._executor_agent:
            conversation_id = self._ensure_active_conversation()
            # Load conversation history from database so Analyst has context
            self.load_conversation_history(conversation_id)

        if self._analyst_agent and self._executor_agent:
            if self._use_background_thread:
                self._start_two_agent_execution(session, sanitized, conversation_id)
            else:
                self._run_two_agent_execution(session, sanitized, conversation_id)
        else:
            self.progress_update.emit(CONTEXT_BUILD_MESSAGE)
            prompt = self._build_prompt(sanitized)
            if self._use_background_thread:
                self._start_background_conversation(session, prompt)
            else:
                self._run_conversation(session, prompt)

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
        if hasattr(self._chat_service, "tool_manager"):
            self._chat_service.tool_manager = self._tool_manager
        if self._analyst_agent:
            self._analyst_agent.tool_manager = self._tool_manager
        if self._executor_agent:
            self._executor_agent.tool_manager = self._tool_manager
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

    def _prepare_analyst_history(self) -> List[dict]:
        """Prepare conversation history in Claude API message format for Analyst.

        Converts internal history format (role, text) tuples to Claude API format
        with dictionaries containing 'role' and 'content' keys.

        Returns:
            List of message dictionaries suitable for Claude API
        """
        analyst_history = []
        for role, text in self._history:
            # Convert role to Claude API format (user/assistant)
            api_role = role.lower() if role else "user"
            analyst_history.append({
                "role": api_role,
                "content": [{"type": "text", "text": text}],
            })
        return analyst_history

    @property
    def history(self) -> Tuple[Tuple[str, str], ...]:
        """Return the accumulated conversation history."""
        return tuple(self._history)

    @property
    def chat_service(self) -> ChatService:
        """Expose the managed ChatService instance."""
        return self._chat_service

    def _start_two_agent_execution(
        self,
        session: ConversationSession,
        goal: str,
        conversation_id: int | None,
    ) -> None:
        """Execute two-agent flow on a background QThread."""
        # Prepare conversation history for Analyst
        analyst_history = self._prepare_analyst_history()

        self._thread = QThread(self)
        self._worker = _TwoAgentWorker(
            self._analyst_agent,
            self._executor_agent,
            goal,
            conversation_id,
            conversation_history=analyst_history,
            progress_callback=self.progress_update.emit,
        )
        self._worker.moveToThread(self._thread)
        self._worker.chunk_emitted.connect(self.session_output.emit)
        self._worker.finished.connect(
            lambda outcome: self._finalize_conversation(session, outcome)
        )
        self._worker.failed.connect(self._handle_worker_error)

        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _run_two_agent_execution(
        self,
        session: ConversationSession,
        goal: str,
        conversation_id: int | None,
    ) -> None:
        """Execute two-agent flow synchronously."""
        started = time.perf_counter()
        streamed = {"value": False}

        # Prepare conversation history for Analyst
        analyst_history = self._prepare_analyst_history()

        def _on_chunk(chunk: str) -> None:
            streamed["value"] = True
            self.session_output.emit(chunk)

        try:
            self.progress_update.emit(ANALYST_START_MESSAGE)
            plan_or_error = self._analyst_agent.analyze_and_plan(
                goal,
                on_chunk=_on_chunk,
                conversation_id=conversation_id,
                conversation_history=analyst_history,
            )

            if isinstance(plan_or_error, str):
                error_msg = plan_or_error or "Analysis failed"
                self.error_occurred.emit(error_msg)
                outcome = _ConversationOutcome(
                    response=error_msg,
                    duration_seconds=time.perf_counter() - started,
                    success=False,
                )
                self._finalize_conversation(session, outcome)
                return

            execution_plan: ExecutionPlan = plan_or_error
            self.progress_update.emit(EXECUTOR_START_MESSAGE)
            result = self._executor_agent.execute_plan(
                execution_plan,
                on_chunk=_on_chunk,
                conversation_id=conversation_id,
            )

            duration = time.perf_counter() - started
            success = not result.strip().lower().startswith("error:")
            outcome = _ConversationOutcome(
                response=result,
                duration_seconds=duration,
                success=success,
            )
            self._finalize_conversation(session, outcome)

        except Exception as exc:
            error = AuraExecutionError(
                "Two-agent execution failed. Verify API keys and connectivity.",
                context={"detail": str(exc)},
            )
            LOGGER.exception("Two-agent execution failed")
            self.error_occurred.emit(str(error))

    def _start_background_conversation(
        self,
        session: ConversationSession,
        prompt: str,
    ) -> None:
        """Execute the conversation on a background QThread."""
        self._thread = QThread(self)
        self._worker = _ConversationWorker(
            self._chat_service,
            prompt,
            progress_callback=self.progress_update.emit,
        )
        self._worker.moveToThread(self._thread)
        self._worker.chunk_emitted.connect(self.session_output.emit)
        self._worker.finished.connect(
            lambda outcome: self._finalize_conversation(session, outcome)
        )
        self._worker.failed.connect(self._handle_worker_error)

        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _run_conversation(
        self,
        session: ConversationSession,
        prompt: str,
    ) -> None:
        """Execute the conversation synchronously (useful for testing)."""
        started = time.perf_counter()
        streamed = {"value": False}

        def _on_chunk(chunk: str) -> None:
            streamed["value"] = True
            self.session_output.emit(chunk)

        try:
            self.progress_update.emit(MICRO_AGENT_START_MESSAGE)
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
        except Exception as exc:
            error = AuraExecutionError(
                "Unable to contact the analyst agent. Verify ANTHROPIC_API_KEY and your connection, then try again.",
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
        self._history.append((MessageRole.USER, session.prompt))
        self._history.append((MessageRole.ASSISTANT, outcome.response.strip()))
        # Persist to database
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
        self.error_occurred.emit(
            message
            or "Unable to contact the analyst agent. Verify ANTHROPIC_API_KEY and your connection, then try again."
        )
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
        except Exception as exc:
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

        # Get project-specific instructions
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
        """
        Load conversation history from database into memory.

        Args:
            conversation_id: Conversation ID to load
        """
        try:
            conv = Conversation.get_by_id(conversation_id)
            if conv:
                self._history = conv.get_history()
                LOGGER.info(f"Loaded {len(self._history)} messages from conversation {conversation_id}")
            else:
                LOGGER.warning(f"Conversation {conversation_id} not found")
                self._history = []
        except Exception as e:
            LOGGER.error(f"Failed to load conversation history: {e}")
            self._history = []

    def _persist_messages_to_db(self, user_message: str, assistant_message: str) -> None:
        """
        Persist user and assistant messages to the database.

        Args:
            user_message: User's message content
            assistant_message: Assistant's response content
        """
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
                        LOGGER.info(f"Created new conversation {conversation_id}")

            # Save messages
            Message.create(conversation_id, MessageRole.USER, user_message)
            Message.create(conversation_id, MessageRole.ASSISTANT, assistant_message)

            # Auto-generate title from first message if needed
            conv = Conversation.get_by_id(conversation_id)
            if conv and not conv.title:
                conv.generate_title_from_first_message()

            LOGGER.info(f"Persisted messages to conversation {conversation_id}")

        except Exception as e:
            LOGGER.error(f"Failed to persist messages to database: {e}")
            self.error_occurred.emit("Warning: Failed to save conversation history. Your responses will not persist.")

