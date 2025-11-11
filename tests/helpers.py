"""Reusable helpers for Aura's automated tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence


class RecordingOutputPanel:
    """Test double that records output panel invocations."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, Any]] = []

    def display_output(self, text: str, color: str | None = None, font_size: int | None = None) -> None:
        self.messages.append(("output", text, color, font_size))

    def display_stream_chunk(self, text: str, color: str | None = None) -> None:
        self.messages.append(("stream", text, color))

    def display_error(self, text: str) -> None:
        self.messages.append(("error", text, None))

    def display_success(self, text: str) -> None:
        self.messages.append(("success", text, None))

    def display_task_list(self, groups: Any) -> None:  # noqa: ANN401 - flexible for tests
        self.messages.append(("tasks", groups, None))

    def display_diff_block(self, diff: str) -> None:
        self.messages.append(("diff", diff, None))

    def display_edit_block(self, path: str, diff: str) -> None:
        self.messages.append(("edit", path, diff))

    def display_file_deletion(self, path: str) -> None:
        self.messages.append(("delete", path, None))


@dataclass
class RecordingStatusManager:
    """Minimal status manager that keeps track of the latest message."""

    updates: List[tuple[str, str, bool]] = field(default_factory=list)

    def update_status(self, message: str, color: str, persist: bool = False) -> None:
        self.updates.append((message, color, persist))


@dataclass
class SimpleChatService:
    """Stub ChatService that returns canned responses."""

    response_text: str
    tool_manager: Any
    streamed_chunks: Sequence[str] | None = None

    def send_message(
        self,
        user_message: str,
        on_chunk: Callable[[str], None] | None = None,
        conversation_id: int | None = None,
    ) -> str:
        if on_chunk and self.streamed_chunks:
            for chunk in self.streamed_chunks:
                on_chunk(chunk)
        return self.response_text
