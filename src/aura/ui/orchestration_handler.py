"""Business-logic handler for orchestrator events."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from aura import config
from aura.events import Event, EventType
from aura.orchestrator import SessionResult
from aura.state import AppState
from aura.ui.output_panel import OutputPanel
from aura.ui.status_bar_manager import StatusBarManager

LOGGER = logging.getLogger(__name__)


def _log_handler_recv(message: str) -> None:
    """Log message receipt at handler for debugging duplicates.

    Args:
        message: The message being received
    """
    # Create unique ID from message content
    msg_id = hashlib.md5(message.encode()).hexdigest()[:8]

    # Truncate message for logging
    msg_preview = message[:50].replace('\n', '\\n')

    LOGGER.info(f"HANDLER_RECV [ID:{msg_id}]: {msg_preview}")


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
        try:
            self._set_running_state()
            self.request_input_enabled.emit(False)
            self._output_panel.display_thinking("Analyzing request...")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to handle planning start")
            self._output_panel.display_error("Unable to display planning status. Check logs for details.")

    def handle_session_started(self, index: int, session) -> None:
        """Display headers for a new session."""
        try:
            self._set_running_state()
            name = getattr(session, "name", "Conversation")

            self._output_panel.display_output("")  # Spacer
            self._output_panel.display_output(
                f"▶ {name}",
                config.COLORS.accent,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to handle session_started")
            self._output_panel.display_error("Unable to start the session. See aura.log for details.")

    def handle_session_output(self, text: str) -> None:
        """Relay streaming session output to the transcript."""
        try:
            if text is None:
                return

            raw_text = text if isinstance(text, str) else str(text)

            # Log message receipt for debugging duplicates
            _log_handler_recv(raw_text)

            if raw_text.startswith(config.STREAM_PREFIX):
                chunk = raw_text[len(config.STREAM_PREFIX) :]
                self._output_panel.display_stream_chunk(chunk, config.COLORS.agent_output)
                return

            if not raw_text:
                return

            stripped_text = raw_text.strip()

            if stripped_text.startswith("TOOL_CALL::"):
                self._handle_tool_call_message(stripped_text)
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
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to render session output")
            self._output_panel.display_error("Unable to display part of the response. See logs for details.")

    def handle_session_complete(self, index: int, result: SessionResult) -> None:
        """Summarize the result of a completed session."""
        try:
            if result is None:
                return

            duration = getattr(result, "duration_seconds", 0.0)
            success = getattr(result, "success", False)

            if success:
                self._output_panel.display_success(f"Response complete in {duration:.1f}s")
            else:
                self._output_panel.display_error(f"Response failed in {duration:.1f}s")
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to summarize session result")
            self._output_panel.display_error("Unable to summarize the session outcome. See logs for details.")

    def handle_all_complete(self) -> None:
        """Mark orchestration as complete."""
        try:
            self._set_completed_state()
            self._output_panel.display_output("")  # Spacer
            self._output_panel.display_success("Conversation finished")
            self.request_input_enabled.emit(True)
            self.request_input_focus.emit()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to finalize conversation")
            self._output_panel.display_error("Conversation finished with issues. See logs for details.")

    def handle_error(self, error: str) -> None:
        """Surface orchestration errors."""
        try:
            self._last_error_message = error
            self._set_error_state()
            user_message = self._format_user_error(error)
            self._output_panel.display_error(user_message)
            self.request_input_enabled.emit(True)
            self.request_input_focus.emit()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to handle orchestration error")
            self._output_panel.display_error("An unexpected error occurred. See aura.log for details.")
            self.request_input_enabled.emit(True)
            self.request_input_focus.emit()

    def handle_background_event(self, event: Event) -> None:
        """Process events forwarded from the background event bus."""
        try:
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
                    self._handle_tool_call_message(stripped_text)
                else:
                    self._output_panel.display_output(stripped_text)

            elif event.type is EventType.ERROR:
                error = str(event.data.get("error", "")).strip()
                if not error:
                    return
                if error == self._last_error_message:
                    self._last_error_message = None
                    return
                self._output_panel.display_error(self._format_user_error(error))
                self._last_error_message = None
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to handle background event")
            self._output_panel.display_error("Background event processing failed. See aura.log for details.")

    def _handle_tool_call_message(self, payload: str) -> None:
        """Parse a TOOL_CALL message and render it appropriately."""
        try:
            _, tool_name, raw_args = payload.split("::", 2)
        except ValueError:
            self._output_panel.display_output(payload)
            return

        parsed_args = self._parse_tool_args(raw_args)
        if self._render_file_tool_call(tool_name, parsed_args):
            return
        if self._render_read_tool_call(tool_name, parsed_args):
            return

        if isinstance(parsed_args, (dict, list)):
            args_display = json.dumps(parsed_args, ensure_ascii=False, indent=2)
        else:
            args_display = str(parsed_args)

        if len(args_display) > 240:
            args_display = f"{args_display[:237]}..."

        self._output_panel.display_tool_call(tool_name, args_display)

    def _render_file_tool_call(self, tool_name: str, args: Any) -> bool:
        """Render structured file operations when sufficient data is available."""
        if tool_name not in {"create_file", "modify_file", "delete_file"}:
            return False
        if not isinstance(args, dict):
            return False

        path = str(args.get("path") or args.get("file_path") or "")
        if not path:
            return False

        if tool_name == "create_file":
            content = str(args.get("content", ""))
            self._output_panel.display_file_creation(path, content)
            return True

        if tool_name == "modify_file":
            content = (
                args.get("new_content")
                or args.get("content")
                or args.get("diff")
                or args.get("patch")
                or ""
            )
            self._output_panel.display_file_modification(path, str(content))
            return True

        if tool_name == "delete_file":
            self._output_panel.display_file_deletion(path)
            return True

        return False

    def _render_read_tool_call(self, tool_name: str, args: Any) -> bool:
        """Render read/list operations with clearer descriptions."""
        if tool_name not in {"read_project_file", "list_project_files"}:
            return False

        if not isinstance(args, dict):
            return False

        target = str(
            args.get("path")
            or args.get("file")
            or args.get("directory")
            or args.get("root")
            or ""
        )

        if tool_name == "read_project_file":
            message = f"Reading {target or 'file'}"
        else:
            message = f"Listing files in {target or '.'}"

        self._output_panel.display_tool_call(tool_name, message)
        return True

    @staticmethod
    def _parse_tool_args(raw_args: str) -> Any:
        """Best-effort parsing of tool arguments from the agent stream."""
        candidate = raw_args.strip()
        if not candidate:
            return {}
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return candidate

    def _format_user_error(self, error: str) -> str:
        """Return a user-friendly error message with actionable guidance."""
        if not error:
            return "Something went wrong. Check aura.log for details."
        lowered = error.lower()
        if "gemini" in lowered or "api key" in lowered:
            return f"{error} Please verify that GEMINI_API_KEY is set."
        if "workspace" in lowered or "working directory" in lowered:
            return f"{error} Choose a valid working directory and retry."
        return error

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
