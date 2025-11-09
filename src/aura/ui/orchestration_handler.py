"""Business-logic handler for orchestrator events."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal, QTimer

from aura import config
from aura.events import (
    ExecutionComplete,
    FileOperation,
    PhaseTransition,
    StatusUpdate,
    StreamingChunk,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from aura.orchestrator import SessionResult
from aura.state import AppState
from aura.ui.output_panel import OutputPanel
from aura.ui.status_bar_manager import StatusBarManager
from aura.ui.task_list_display import TaskListDisplay, TaskStatus

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


class VerbosityLevel(str, Enum):
    """Represents how much streaming output to surface."""

    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"

    @classmethod
    def from_value(cls, value: Optional[str]) -> "VerbosityLevel":
        """Map persisted values (including None) onto a valid enum."""
        if isinstance(value, cls):
            return value
        normalized = (value or "").lower()
        for level in cls:
            if level.value == normalized:
                return level
        return cls.NORMAL


STREAM_SUMMARY_WINDOW_MS = 1200
STREAM_SUMMARY_MAX_LEN = 220
IGNORED_CHUNK_PHRASES = {
    "thinking",
    "analysis",
    "analyzing",
    "considering options",
    "reasoning",
    "pondering",
}


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
        self._verbosity = VerbosityLevel.NORMAL
        self._chunk_buffer: list[str] = []
        self._chunk_buffer_source: Optional[str] = None
        self._chunk_buffer_timer = QTimer(self)
        self._chunk_buffer_timer.setSingleShot(True)
        self._chunk_buffer_timer.timeout.connect(self._flush_chunk_buffer)
        self._in_thinking_block = False
        self._last_chunk_summary: str = ""
        self._task_list_display = TaskListDisplay()
        self._active_task_id: Optional[str] = None
        self._task_list_enabled = True
        self._pending_task_updates: list[tuple[str, TaskStatus]] = []
        self._task_update_timer = QTimer(self)
        self._task_update_timer.setSingleShot(True)
        self._task_update_timer.timeout.connect(self._flush_task_updates)

    @property
    def verbosity(self) -> VerbosityLevel:
        """Return the currently active verbosity level."""
        return self._verbosity

    def set_verbosity(self, level: VerbosityLevel) -> None:
        """Update verbosity and reset streaming buffers."""
        new_level = VerbosityLevel.from_value(level)
        if new_level == self._verbosity:
            return
        self._verbosity = new_level
        self._reset_chunk_buffer()

    def set_task_list_enabled(self, enabled: bool) -> None:
        """Enable or disable task list display.

        Args:
            enabled: Whether to enable task list display
        """
        self._task_list_enabled = enabled
        if not enabled:
            self._task_list_display.clear()

    def handle_planning_started(self) -> None:
        """Reset state and display planning message."""
        try:
            self._reset_chunk_buffer()
            self._task_list_display.clear()
            self._set_running_state()
            self.request_input_enabled.emit(False)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to handle planning start")
            self._output_panel.display_error("Unable to display planning status. Check logs for details.")

    def handle_session_started(self, index: int, session) -> None:
        """Display headers for a new session."""
        try:
            self._reset_chunk_buffer()
            self._task_list_display.clear()  # Clear task list for new session
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

    def handle_feedback_event(self, event: object) -> None:
        """Process typed feedback events from the event bus."""
        try:
            if isinstance(event, StreamingChunk):
                self._handle_streaming_chunk_event(event)
                return

            if isinstance(event, ToolCallStarted):
                self._handle_tool_call_event(event)
                return

            if isinstance(event, ToolCallCompleted):
                self._handle_tool_call_completed_event(event)
                return

            if isinstance(event, ToolCallFailed):
                self._handle_tool_call_failure_event(event)
                return

            if isinstance(event, FileOperation):
                self._handle_file_operation_event(event)
                return

            if isinstance(event, PhaseTransition):
                self._handle_phase_transition_event(event)
                return

            if isinstance(event, StatusUpdate):
                self._handle_status_update_event(event)
                return

            if isinstance(event, ExecutionComplete):
                self._handle_execution_complete_event(event)
                return
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to handle feedback event")
            self._output_panel.display_error("Background event processing failed. See aura.log for details.")

    def _handle_streaming_chunk_event(self, event: StreamingChunk) -> None:
        """Filter and display streaming chunks according to verbosity settings."""
        text = (event.text or "").rstrip("\r")
        if not text:
            return

        # Parse text for task structures (if task list is enabled and not minimal verbosity)
        if self._task_list_enabled and self._verbosity is not VerbosityLevel.MINIMAL:
            new_groups = self._task_list_display.parse_text_for_tasks(text, event.source)
            if new_groups:
                groups = self._task_list_display.get_all_groups()
                if groups:
                    self._output_panel.display_task_list(groups)

        if self._verbosity is VerbosityLevel.VERBOSE:
            _log_handler_recv(text)
            self._output_panel.display_stream_chunk(text, config.COLORS.agent_output)
            return

        filtered = self._strip_hidden_blocks(text)
        if self._verbosity is VerbosityLevel.MINIMAL:
            # Still update hidden block tracking even when suppressing all text.
            return

        normalized = self._normalize_chunk(filtered)
        if not normalized:
            return
        self._chunk_buffer_source = event.source or self._chunk_buffer_source
        self._chunk_buffer.append(normalized)
        total_chars = sum(len(segment) for segment in self._chunk_buffer)
        if event.is_final or total_chars >= STREAM_SUMMARY_MAX_LEN:
            self._flush_chunk_buffer()
            return
        self._chunk_buffer_timer.start(STREAM_SUMMARY_WINDOW_MS)

    def _handle_tool_call_message(self, payload: str) -> None:
        """Parse a TOOL_CALL message and render it appropriately."""
        try:
            _, tool_name, raw_args = payload.split("::", 2)
        except ValueError:
            self._output_panel.display_output(payload)
            return

        parsed_args = self._parse_tool_args(raw_args)
        self._render_tool_call(tool_name, parsed_args)

    def _handle_tool_call_event(self, event: ToolCallStarted) -> None:
        """Render tool call feedback emitted from the event bus."""
        self._flush_chunk_buffer()
        parsed_args = self._normalize_parameters(event.parameters)

        # Create or update task for this tool call
        if self._task_list_enabled and self._verbosity is not VerbosityLevel.MINIMAL:
            task_id = self._task_list_display.create_task_from_tool_call(
                event.tool_name or "tool",
                parsed_args,
            )
            if task_id:
                # Mark task as in-progress
                self._active_task_id = task_id
                self._queue_task_update(task_id, TaskStatus.IN_PROGRESS)

                # Display updated task list
                groups = self._task_list_display.get_all_groups()
                if groups:
                    self._output_panel.display_task_list(groups)

        self._render_tool_call(event.tool_name or "tool", parsed_args)

    def _handle_tool_call_completed_event(self, event: ToolCallCompleted) -> None:
        """Surface a concise completion line for every tool call."""
        self._flush_chunk_buffer()
        tool = event.tool_name or "tool"

        # Update task status to completed
        if self._task_list_enabled and self._verbosity is not VerbosityLevel.MINIMAL:
            updated = self._task_list_display.update_task_status_by_tool(
                tool,
                TaskStatus.COMPLETED,
            )
            if updated:
                # Display updated task list
                groups = self._task_list_display.get_all_groups()
                if groups:
                    self._output_panel.display_task_list(groups)

        summary = self._summarize_tool_result(event.result)
        if summary:
            message = f"[OK] {tool} completed: {summary}"
        else:
            message = f"[OK] {tool} completed"
        self._output_panel.display_output(message, config.COLORS.success)

    def _handle_tool_call_failure_event(self, event: ToolCallFailed) -> None:
        """Always display tool failures, regardless of verbosity."""
        self._flush_chunk_buffer()
        tool = event.tool_name or "tool"

        # Update task status to failed
        if self._task_list_enabled and self._verbosity is not VerbosityLevel.MINIMAL:
            updated = self._task_list_display.update_task_status_by_tool(
                tool,
                TaskStatus.FAILED,
            )
            if updated:
                # Display updated task list
                groups = self._task_list_display.get_all_groups()
                if groups:
                    self._output_panel.display_task_list(groups)

        error = (event.error or "Tool call failed.").strip()
        message = f"[FAIL] {tool} failed: {error}"
        self._output_panel.display_output(message, config.COLORS.error)

    def _handle_phase_transition_event(self, event: PhaseTransition) -> None:
        """Highlight state transitions across the orchestration pipeline."""
        self._flush_chunk_buffer()
        previous = (event.from_phase or "unknown").strip() or "unknown"
        current = (event.to_phase or "unknown").strip() or "unknown"

        # Clear task list on major phase transitions
        if self._task_list_enabled and previous != current:
            self._task_list_display.clear()

        message = f"-> Phase: {previous} -> {current}"
        self._output_panel.display_output(message, config.COLORS.accent)

    def _render_tool_call(self, tool_name: str, parsed_args: Any) -> None:
        """Shared rendering pipeline for tool call events."""
        announcement = self._build_tool_announcement(tool_name, parsed_args)
        if announcement:
            self._queue_tool_announcement(announcement)
        if self._render_file_tool_call(tool_name, parsed_args):
            return
        if self._render_read_tool_call(tool_name, parsed_args):
            return

        summary = self._summarize_tool_args(parsed_args)
        line = f"?? {tool_name}: {summary}" if summary else f"?? {tool_name}"
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
        if tool_name not in {"create_file", "modify_file", "replace_file_lines", "delete_file"}:
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
        if tool_name == "replace_file_lines":
            diff_text = self._build_line_range_diff(
                path,
                args.get("start_line"),
                args.get("end_line"),
                args.get("new_content"),
            )
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
        elif tool_name in {"modify_file", "replace_file_lines"}:
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

    def _build_line_range_diff(
        self,
        path: str,
        start_line: Any,
        end_line: Any,
        new_content: Any,
    ) -> str:
        """Return a pseudo-diff for replace_file_lines tool calls."""
        content = new_content if isinstance(new_content, str) else str(new_content or "")
        try:
            start = int(start_line)
            end = int(end_line)
            if start < 1 or end < start:
                raise ValueError
        except Exception:
            start = end = None

        diff_lines: list[str] = []
        if start is not None and end is not None:
            span = end - start + 1
            diff_lines.append(f"@@ {start},{span} @@")
        header = f"# Replacing lines {start_line}-{end_line} in {path}" if start_line and end_line else ""
        if header:
            diff_lines.append(header)

        if content:
            for line in content.splitlines():
                diff_lines.append(f"+{line}")
        else:
            diff_lines.append("- <content removed>")
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

    @staticmethod
    def _normalize_parameters(parameters: object) -> dict[str, Any]:
        """Normalize event parameters into a dict for rendering helpers."""
        if parameters is None:
            return {}
        if isinstance(parameters, dict):
            args = parameters.get("args")
            kwargs = parameters.get("kwargs")
            if isinstance(kwargs, dict):
                merged = dict(kwargs)
                if args is not None:
                    merged.setdefault("args", args)
                return merged
            return dict(parameters)
        return {"value": parameters}

    def _strip_hidden_blocks(self, text: str) -> str:
        """Remove segments enclosed in hidden tags (e.g., <thinking>)."""
        if not text:
            return ""
        lowered = text.lower()
        cursor = 0
        output: list[str] = []
        while cursor < len(text):
            if self._in_thinking_block:
                closing_index = None
                closing_token = ""
                for tag in ("thinking", "analysis"):
                    token = f"</{tag}>"
                    idx = lowered.find(token, cursor)
                    if idx != -1 and (closing_index is None or idx < closing_index):
                        closing_index = idx
                        closing_token = token
                if closing_index is None:
                    # Entire remainder is hidden.
                    return "".join(output)
                cursor = closing_index + len(closing_token)
                self._in_thinking_block = False
                continue

            start_index = None
            start_token_end = None
            for tag in ("thinking", "analysis"):
                token = f"<{tag}"
                idx = lowered.find(token, cursor)
                if idx != -1 and (start_index is None or idx < start_index):
                    start_index = idx
                    end = lowered.find(">", idx)
                    start_token_end = end + 1 if end != -1 else None
            if start_index is None:
                output.append(text[cursor:])
                break
            output.append(text[cursor:start_index])
            if start_token_end is None:
                # Tag started but not finished in this chunk.
                self._in_thinking_block = True
                break
            cursor = start_token_end
            self._in_thinking_block = True
        cleaned = "".join(output)
        cleaned = re.sub(r"</(thinking|analysis)>", "", cleaned, flags=re.IGNORECASE)
        return cleaned

    def _normalize_chunk(self, text: str) -> str:
        """Collapse whitespace and discard repetitive reasoning lines."""
        normalized = re.sub(r"\s+", " ", text or "").strip()
        if not normalized:
            return ""
        normalized = normalized.lstrip("-•").strip()
        lowered = normalized.lower()
        if lowered in IGNORED_CHUNK_PHRASES:
            return ""
        if lowered.startswith("analysis:"):
            normalized = normalized[len("analysis:") :].strip().capitalize()
        return normalized

    def _flush_chunk_buffer(self) -> None:
        """Emit any buffered streaming summary."""
        if not self._chunk_buffer:
            self._chunk_buffer_timer.stop()
            self._chunk_buffer_source = None
            return
        summary = self._summarize_chunk_buffer()
        self._chunk_buffer = []
        self._chunk_buffer_timer.stop()
        source = self._format_source_label(self._chunk_buffer_source)
        self._chunk_buffer_source = None
        if not summary:
            return
        self._last_chunk_summary = summary
        self._output_panel.display_output(f"[{source}] {summary}", config.COLORS.agent_output)

    def _summarize_chunk_buffer(self) -> str:
        """Create a single line summary from buffered chunks."""
        summary = re.sub(r"\s+", " ", " ".join(self._chunk_buffer)).strip()
        if not summary:
            return ""
        if summary == self._last_chunk_summary:
            return ""
        if len(summary) > STREAM_SUMMARY_MAX_LEN:
            summary = summary[:STREAM_SUMMARY_MAX_LEN].rstrip() + "..."
        return summary

    def _format_source_label(self, source: Optional[str]) -> str:
        """Return a friendly display label for a streaming source."""
        if not source:
            return "Analyst"
        normalized = source.replace("_", " ").strip()
        if not normalized:
            return "Analyst"
        return normalized.title()

    def _summarize_tool_result(self, result: Any) -> str:
        """Convert tool results into a short one-line summary."""
        if result is None:
            return ""
        if isinstance(result, str):
            summary = result.strip()
        else:
            try:
                summary = json.dumps(result, default=str)
            except TypeError:
                summary = str(result)
        summary = summary.splitlines()[0].strip()
        if len(summary) > 160:
            summary = summary[:160].rstrip() + "..."
        return summary

    def _reset_chunk_buffer(self) -> None:
        """Clear any buffered streaming text and reset filters."""
        self._chunk_buffer.clear()
        self._chunk_buffer_timer.stop()
        self._chunk_buffer_source = None
        self._in_thinking_block = False

    def _handle_file_operation_event(self, event: FileOperation) -> None:
        """Render file operation completions."""
        self._flush_chunk_buffer()
        operation = (event.operation or "").lower()
        filepath = event.filepath or ""
        if not operation or not filepath:
            return
        verb_map = {
            "create_file": "Created",
            "modify_file": "Modified",
            "replace_file_lines": "Modified",
            "delete_file": "Deleted",
            "read_project_file": "Read",
        }
        icon_map = {
            "create_file": "🆕",
            "modify_file": "✏️",
            "replace_file_lines": "✏️",
            "delete_file": "🗑️",
            "read_project_file": "📖",
        }
        verb = verb_map.get(operation, operation.replace("_", " ").title())
        icon = icon_map.get(operation, "??")
        color = config.COLORS.success
        if operation == "delete_file":
            color = config.COLORS.error
        elif operation.startswith("read"):
            color = config.COLORS.secondary
        message = f"{icon} {verb}: {filepath}"
        self._output_panel.display_output(message, color)

    def _handle_status_update_event(self, event: StatusUpdate) -> None:
        """Display status updates from services."""
        self._flush_chunk_buffer()
        message = event.message or "Status update"
        phase = (event.phase or "").lower()
        color = config.COLORS.accent
        if "error" in phase:
            color = config.COLORS.error
        elif "complete" in phase or "success" in phase:
            color = config.COLORS.success
        self._status_manager.update_status(message, color, persist=True)

    def _handle_execution_complete_event(self, event: ExecutionComplete) -> None:
        """Show execution summaries when agents finish."""
        self._flush_chunk_buffer()
        summary = (event.summary or "").strip()
        success = event.success is not False
        color = config.COLORS.success if success else config.COLORS.error
        message = summary or ("Execution complete." if success else "Execution failed.")
        self._output_panel.display_output(message, color)
        if not success:
            self._last_error_message = message

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

    def _queue_task_update(self, task_id: str, status: TaskStatus) -> None:
        """Queue a task status update for batched processing.

        Args:
            task_id: The task ID to update
            status: The new status
        """
        self._pending_task_updates.append((task_id, status))
        if not self._task_update_timer.isActive():
            self._task_update_timer.start(50)  # 50ms batch window

    def _flush_task_updates(self) -> None:
        """Flush pending task status updates."""
        if not self._pending_task_updates:
            return

        # Apply all pending updates
        for task_id, status in self._pending_task_updates:
            self._task_list_display.update_task_status(task_id, status)

        self._pending_task_updates.clear()

        # Re-render task list once with all updates
        groups = self._task_list_display.get_all_groups()
        if groups:
            self._output_panel.display_task_list(groups)
