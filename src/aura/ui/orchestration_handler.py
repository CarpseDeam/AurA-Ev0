"""Business-logic handler for orchestrator events."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal, QTimer

from aura import config
from aura.events import Event, EventType
from aura.orchestrator import SessionResult
from aura.state import AppState
from aura.ui.output_panel import OutputPanel
from aura.ui.status_bar_manager import StatusBarManager

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolAnnouncement:
    """Represents a single-line summary for an incoming tool call."""

    icon: str
    action: str
    target: str
    color: str
    meta: str = ""

    def render(self) -> str:
        """Return the fully formatted line for display."""
        core = f"{self.icon} {self.action}"
        if self.target:
            core = f"{core}: {self.target}"
        return f"{core} {self.meta}".strip()


GENERIC_TOOL_ICON = "⚙️"
TOOL_BATCH_WINDOW_MS = 160
TOOL_PREVIEW_LIMIT = 3


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
        self._pending_tool_announcements: list[ToolAnnouncement] = []
        self._tool_batch_timer = QTimer(self)
        self._tool_batch_timer.setSingleShot(True)
        self._tool_batch_timer.timeout.connect(self._flush_tool_announcements)

    def handle_planning_started(self) -> None:
        """Reset state and display planning message."""
        try:
            self._set_running_state()
            self.request_input_enabled.emit(False)
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

            stripped = raw_text.strip()
            if not stripped:
                return

            if stripped.startswith("TOOL_CALL::"):
                self._handle_tool_call_message(stripped)
                return

            if self._pending_tool_announcements:
                self._flush_tool_announcements()
            self._render_text_with_diffs(raw_text)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to render session output")
            self._output_panel.display_error("Unable to display part of the response. See logs for details.")

    def handle_session_complete(self, index: int, result: SessionResult) -> None:
        """Summarize the result of a completed session."""
        try:
            self._flush_tool_announcements()
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
            self._flush_tool_announcements()
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
            self._flush_tool_announcements()
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

                self._render_text_with_diffs(text)

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
        announcement = self._build_tool_announcement(tool_name, parsed_args)
        if announcement:
            self._queue_tool_announcement(announcement)
        if self._render_file_tool_call(tool_name, parsed_args):
            return
        if self._render_read_tool_call(tool_name, parsed_args):
            return

        summary = self._summarize_tool_args(parsed_args)
        line = f"⚙️ {tool_name}: {summary}" if summary else f"⚙️ {tool_name}"
        self._output_panel.display_output(line, config.COLORS.secondary)

    def _render_text_with_diffs(self, text: str) -> None:
        """Split streamed text around diff fences and render accordingly."""
        if not text:
            return
        for segment, is_diff in self._split_diff_segments(text):
            if is_diff:
                diff_body = segment.strip("\n")
                if diff_body:
                    self._output_panel.display_diff_block(diff_body)
                continue
            self._render_standard_output(segment)

    def _split_diff_segments(self, text: str) -> list[tuple[str, bool]]:
        """Return a list of (segment, is_diff) tuples for a payload."""
        if not text:
            return []
        marker = "```diff"
        fence = "```"
        segments: list[tuple[str, bool]] = []
        cursor = 0
        length = len(text)
        while cursor < length:
            start = text.find(marker, cursor)
            if start == -1:
                trailing = text[cursor:]
                if trailing:
                    segments.append((trailing, False))
                break
            if start > cursor:
                segments.append((text[cursor:start], False))
            close = text.find(fence, start + len(marker))
            if close == -1:
                segments.append((text[start:], False))
                break
            diff_body = text[start + len(marker):close]
            if diff_body.startswith("\n"):
                diff_body = diff_body[1:]
            segments.append((diff_body, True))
            cursor = close + len(fence)
        return segments or [(text, False)]

    def _render_standard_output(self, text: str) -> None:
        """Render non-tool output as simple paragraphs."""
        if not text:
            return
        stripped = text.strip()
        if not stripped:
            return
        if stripped.startswith("?"):
            stripped = stripped[1:].strip() or stripped
        self._output_panel.display_output(stripped)

    def _render_file_tool_call(self, tool_name: str, args: Any) -> bool:
        """Render concise summaries for file create/modify/delete tools."""
        if tool_name not in {"create_file", "modify_file", "delete_file"}:
            return False
        if not isinstance(args, dict):
            return False

        path = self._extract_path_argument(args) or "workspace file"

        if tool_name == "create_file":
            return True

        if tool_name == "modify_file":
            old_content = args.get("old_content") or ""
            new_content = args.get("new_content") or args.get("content") or ""
            diff_text = self._build_diff_from_contents(path, old_content, new_content)
            if not diff_text:
                fallback = args.get("diff") or args.get("patch")
                diff_text = str(fallback or "")
            if diff_text:
                self._output_panel.display_edit_block(path, diff_text)
            return True

        self._output_panel.display_file_deletion(path)
        return True

    def _render_read_tool_call(self, tool_name: str, args: Any) -> bool:
        """Render single-line status for read/list operations."""
        read_tools = {"read_project_file", "read_multiple_files", "read_file"}
        list_tools = {"list_project_files"}
        if tool_name not in read_tools | list_tools:
            return False

        return True

    @staticmethod
    def _extract_path_argument(args: Any) -> str:
        """Return a path/directory argument from a parsed tool payload."""
        if not isinstance(args, dict):
            return ""
        for key in ("path", "file_path", "file", "target", "directory", "root"):
            value = args.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def _count_lines(content: str) -> int:
        """Return the number of newline-delimited lines in text."""
        if not content:
            return 0
        return len(content.splitlines())

    @staticmethod
    def _format_file_size(byte_count: int) -> str:
        """Return a human-readable file size label."""
        if byte_count <= 0:
            return "0 B"
        units = ("B", "KB", "MB", "GB")
        size = float(byte_count)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{int(size)} B"

    @staticmethod
    def _summarize_tool_args(args: Any) -> str:
        """Return a compact JSON-ish summary suitable for a single line."""
        if args is None:
            return ""
        if isinstance(args, (dict, list)):
            summary = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        else:
            summary = str(args)
        return summary if len(summary) <= 160 else f"{summary[:157]}..."

    def _build_tool_announcement(self, tool_name: str, args: Any) -> ToolAnnouncement | None:
        """Return a ToolAnnouncement describing the requested action."""
        if not tool_name:
            return None

        color = config.COLORS.tool_call
        icon = GENERIC_TOOL_ICON
        action = "Running"
        meta = ""
        path_hint = ""
        if isinstance(args, dict):
            path_hint = self._extract_path_argument(args)

        read_tools = {"read_project_file", "read_multiple_files", "read_file"}
        list_tools = {"list_project_files"}

        if tool_name == "create_file":
            action = "Creating"
            icon = "🆕"
            content = str((args or {}).get("content", ""))
            size_label = self._format_file_size(len(content.encode("utf-8")))
            line_count = self._count_lines(content)
            if line_count and size_label:
                meta = f"({line_count} lines • {size_label})"
            elif line_count:
                meta = f"({line_count} lines)"
            elif size_label:
                meta = f"({size_label})"
        elif tool_name == "modify_file":
            action = "Modifying"
            icon = "✏️"
        elif tool_name == "delete_file":
            action = "Deleting"
            icon = "🗑️"
        elif tool_name in read_tools:
            action = "Reading"
            icon = "📖"
        elif tool_name in list_tools:
            action = "Listing"
            icon = "📂"
        else:
            path_hint = path_hint or tool_name

        target = self._shorten_target(path_hint) if path_hint else tool_name
        return ToolAnnouncement(icon=icon, action=action, target=target, color=color, meta=meta)

    def _queue_tool_announcement(self, announcement: ToolAnnouncement) -> None:
        """Display a live status line for a tool action."""
        if not announcement:
            return
        self._pending_tool_announcements.append(announcement)
        if self._tool_batch_timer.isActive():
            self._tool_batch_timer.stop()
        self._tool_batch_timer.start(TOOL_BATCH_WINDOW_MS)

    def _flush_tool_announcements(self) -> None:
        """Flush queued tool announcements with optional batching."""
        if not self._pending_tool_announcements:
            return
        if self._tool_batch_timer.isActive():
            self._tool_batch_timer.stop()
        pending = self._pending_tool_announcements
        self._pending_tool_announcements = []
        if len(pending) == 1:
            announcement = pending[0]
            self._output_panel.display_output(announcement.render(), announcement.color)
            return

        preview = ", ".join(
            f"{item.icon} {item.action}" for item in pending[:TOOL_PREVIEW_LIMIT]
        )
        if len(pending) > TOOL_PREVIEW_LIMIT:
            preview = f"{preview}, …" if preview else "…"

        summary = f"{GENERIC_TOOL_ICON} Running {len(pending)} tools..."
        if preview:
            summary = f"{summary} ({preview})"
        self._output_panel.display_output(summary, config.COLORS.tool_call)

    @staticmethod
    def _shorten_target(value: str, limit: int = 60) -> str:
        """Truncate long targets so they fit on a single line."""
        if not value:
            return ""
        if len(value) <= limit:
            return value
        return f"...{value[-(limit - 3):]}"

    def _build_diff_from_contents(
        self,
        path: str,
        old_content: Any,
        new_content: Any,
    ) -> str:
        """Return a unified diff between two text blobs."""
        old_text = old_content if isinstance(old_content, str) else str(old_content or "")
        new_text = new_content if isinstance(new_content, str) else str(new_content or "")
        if not old_text and not new_text:
            return ""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}" if path else "a/file",
                tofile=f"b/{path}" if path else "b/file",
                lineterm="",
            )
        )
        return "\n".join(diff_lines)

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
