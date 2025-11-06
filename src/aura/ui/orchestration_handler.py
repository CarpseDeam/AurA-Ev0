"""Business-logic handler for orchestrator events."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from src.aura import config
from src.aura.events import Event, EventType
from src.aura.orchestrator import SessionResult
from src.aura.services.planning_service import Session, SessionPlan
from src.aura.state import AppState
from src.aura.ui.output_panel import OutputPanel
from src.aura.ui.status_bar_manager import StatusBarManager


class OrchestrationHandler(QObject):
    """Routes orchestrator events to the output panel and status bar."""

    request_input_enabled = Signal(bool)
    request_input_focus = Signal()

    def __init__(
        self,
        output_panel: OutputPanel,
        status_manager: StatusBarManager,
        app_state: AppState,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._output_panel = output_panel
        self._status_manager = status_manager
        self._app_state = app_state
        self._last_error_message: Optional[str] = None

    def handle_planning_started(self) -> None:
        """Reset state and display planning message."""
        self._app_state.set_current_plan(None)
        self._set_running_state()
        self.request_input_enabled.emit(False)
        self._output_panel.display_thinking("Analyzing request and planning sessions...")

    def handle_plan_ready(self, plan: SessionPlan) -> None:
        """Render the plan details and re-enable input."""
        # Use duck typing instead of isinstance check since Qt signals may not preserve exact type
        if plan is not None and hasattr(plan, 'sessions') and hasattr(plan, 'total_estimated_minutes'):
            self._app_state.set_current_plan(plan)
            self._display_plan(plan)
        else:
            self._app_state.set_current_plan(None)
            self._output_panel.display_error("Received invalid plan data.")

        self.request_input_enabled.emit(True)
        self.request_input_focus.emit()

    def handle_session_started(self, index: int, session: Session) -> None:
        """Display headers for a new session."""
        self._set_running_state()
        total = len(self._app_state.current_plan.sessions) if self._app_state.current_plan else "?"
        name = getattr(session, "name", "Unknown session")

        self._output_panel.display_output("")  # Spacer
        self._output_panel.display_output(
            f"▶ Session {index + 1}/{total}: {name}",
            config.COLORS.accent,
        )

    def handle_session_output(self, text: str) -> None:
        """Relay streaming session output to the transcript."""
        if text is None:
            return

        raw_text = text if isinstance(text, str) else str(text)

        if raw_text.startswith(config.STREAM_PREFIX):
            chunk = raw_text[len(config.STREAM_PREFIX) :]
            self._output_panel.display_stream_chunk(chunk, config.COLORS.agent_output)
            return

        if not raw_text:
            return

        stripped_text = raw_text.strip()

        if stripped_text.startswith("TOOL_CALL::"):
            try:
                _, tool_name, args_summary = stripped_text.split("::", 2)
                self._output_panel.display_tool_call(tool_name, args_summary)
            except ValueError:
                self._output_panel.display_output(raw_text)  # Fallback
        elif stripped_text.startswith("⋯"):
            self._output_panel.display_thinking(stripped_text[1:].strip())
        elif stripped_text.startswith("▶"):
            self._output_panel.display_output(stripped_text, config.COLORS.accent)
        elif stripped_text.startswith("✓"):
            self._output_panel.display_success(stripped_text[1:].strip())
        elif stripped_text.startswith("+"):
            action, path = stripped_text[1:].strip().split(maxsplit=1)
            self._output_panel.display_file_operation(action, path)
        elif stripped_text.startswith("~"):
            action, path = stripped_text[1:].strip().split(maxsplit=1)
            self._output_panel.display_file_operation(action, path)
        else:
            self._output_panel.display_output(raw_text)

    def handle_session_complete(self, index: int, result: SessionResult) -> None:
        """Summarize the result of a completed session."""
        if result is None:
            return

        duration = getattr(result, "duration_seconds", 0.0)
        success = getattr(result, "success", False)

        if success:
            self._output_panel.display_success(f"Session complete in {duration:.1f}s")
        else:
            self._output_panel.display_error(f"Session failed in {duration:.1f}s")

    def handle_all_complete(self) -> None:
        """Mark orchestration as complete."""
        self._app_state.set_current_plan(None)
        self._set_completed_state()
        self._output_panel.display_output("")  # Spacer
        self._output_panel.display_success("All sessions complete")
        self.request_input_enabled.emit(True)
        self.request_input_focus.emit()

    def handle_error(self, error: str) -> None:
        """Surface orchestration errors."""
        self._last_error_message = error
        self._set_error_state()
        self._output_panel.display_error(error)
        self.request_input_enabled.emit(True)
        self.request_input_focus.emit()
        self._app_state.set_current_plan(None)

    def handle_background_event(self, event: Event) -> None:
        """Process events forwarded from the background event bus."""
        if not isinstance(event, Event):
            return

        if event.type is EventType.SESSION_OUTPUT:
            text = str(event.data.get("text", ""))
            if not text:
                return

            # Strip STREAM prefix if present (for streaming output)
            if text.startswith(config.STREAM_PREFIX):
                chunk = text[len(config.STREAM_PREFIX):]
                self._output_panel.display_stream_chunk(chunk, config.COLORS.agent_output)
                return

            stripped_text = text.strip()
            if not stripped_text:
                return

            if stripped_text.startswith("TOOL_CALL::"):
                try:
                    _, tool_name, args_summary = stripped_text.split("::", 2)
                    self._output_panel.display_tool_call(tool_name, args_summary)
                except ValueError:
                    self._output_panel.display_output(stripped_text)  # Fallback
            else:
                self._output_panel.display_output(stripped_text)

        elif event.type is EventType.ERROR:
            error = str(event.data.get("error", "")).strip()
            if not error:
                return
            if error == self._last_error_message:
                self._last_error_message = None
                return
            self._output_panel.display_error(error)
            self._last_error_message = None

    def format_plan(self, plan: SessionPlan) -> List[str]:
        """Return a formatted plan summary for logging or testing."""
        lines: List[str] = [
            "Session Plan",
            f"Total estimate: {plan.total_estimated_minutes} minutes",
        ]
        for idx, session in enumerate(plan.sessions, start=1):
            deps = ", ".join(session.dependencies) if session.dependencies else "None"
            lines.append(f"{idx}. {session.name} ({session.estimated_minutes} min)")
            lines.append(f"   Dependencies: {deps}")
        if plan.reasoning:
            lines.append(f"Reasoning: {plan.reasoning}")
        return lines

    def _display_plan(self, plan: SessionPlan) -> None:
        """Display the structured plan in the output panel."""
        self._output_panel.display_output("")  # Spacer
        self._output_panel.display_output(
            "┌ Session Plan",
            config.COLORS.accent,
        )

        if plan.reasoning:
            self._output_panel.display_output(
                f"├─ Reasoning: {plan.reasoning}", config.COLORS.agent_output
            )

        self._output_panel.display_output(
            f"├─ Sessions: {len(plan.sessions)}", config.COLORS.agent_output
        )
        self._output_panel.display_output(
            f"└─ Estimated time: ~{plan.total_estimated_minutes} minutes",
            config.COLORS.agent_output,
        )

        self._output_panel.display_output("")  # Spacer
        self._output_panel.display_success("Type 'start' when ready to begin building.")

    def _set_ready_state(self) -> None:
        """Set the ready state through the status manager."""
        self._status_manager.update_status("Ready", config.COLORS.text, persist=True)

    def _set_running_state(self) -> None:
        """Set the running state through the status manager."""
        self._status_manager.update_status("Running...", config.COLORS.accent, persist=True)

    def _set_completed_state(self) -> None:
        """Set the completed state through the status manager."""
        self._status_manager.update_status("Completed", config.COLORS.success, persist=True)

    def _set_error_state(self) -> None:
        """Set the error state through the status manager."""
        self._status_manager.update_status("Error", config.COLORS.error, persist=True)
