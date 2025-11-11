"Chat service for conversational AI interactions with developer tools."

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import anthropic

from aura import config
from aura.event_bus import get_event_bus
from aura.events import (
    AgentEvent,
    ExecutionComplete,
    StatusUpdate,
    StreamingChunk,
    SystemErrorEvent,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from aura.models import ToolCallLog
from aura.utils.prompt_caching import build_cached_system_and_tools
from aura.tools.local_agent_tools import generate_commit_message
from aura.tools.tool_manager import ToolManager
from aura.tools.anthropic_tool_builder import build_anthropic_tool_schema

LOGGER = logging.getLogger(__name__)
_CHAT_SOURCE = "chat"

CHAT_SYSTEM_PROMPT = """
You are Aura's single-agent fallback. Work like a senior engineer sitting at the user's workstation.

- **Investigate first.** Use the provided tools to list files, read code, and understand context before editing.
- **Cite evidence.** Reference concrete file paths and line numbers when explaining behavior or decisions.
- **Edit directly.** Use create/modify/replace/delete file tools to apply fully working code. Never describe changes without making them.
- **Verify results.** Run linters or tests via the available tools when appropriate and summarize outcomes.
- **Be concise.** Respond with clear reasoning, the actions you took, and guidance for next steps.
""".strip()


ToolHandler = Callable[..., Any]


@dataclass
class ChatService:
    """Manages conversational interactions with developer tool access."""

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _event_bus: Any = field(init=False, repr=False)
    _tool_handlers: Mapping[str, ToolHandler] = field(init=False, repr=False)
    _active_conversation_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Anthropic client."""
        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._event_bus = get_event_bus()
        self._tool_handlers = {
            "list_project_files": self.tool_manager.list_project_files,
            "search_in_files": self.tool_manager.search_in_files,
            "read_project_file": self.tool_manager.read_project_file,
            "read_multiple_files": self.tool_manager.read_multiple_files,
            "get_function_definitions": self.tool_manager.get_function_definitions,
            "run_tests": self.tool_manager.run_tests,
            "lint_code": self.tool_manager.lint_code,
            "format_code": self.tool_manager.format_code,
            "install_package": self.tool_manager.install_package,
            "get_git_status": self.tool_manager.get_git_status,
            "git_commit": self.tool_manager.git_commit,
            "git_push": self.tool_manager.git_push,
            "git_diff": self.tool_manager.git_diff,
            "generate_commit_message": generate_commit_message,
            "find_definition": self.tool_manager.find_definition,
            "find_usages": self.tool_manager.find_usages,
            "get_imports": self.tool_manager.get_imports,
            "create_file": self.tool_manager.create_file,
            "modify_file": self.tool_manager.modify_file,
            "replace_file_lines": self.tool_manager.replace_file_lines,
            "delete_file": self.tool_manager.delete_file,
        }

    def send_message(
        self,
        user_message: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        conversation_id: int | None = None,
    ) -> str:
        """Send a message and stream the response with automatic tool usage."""
        started = time.perf_counter()
        self._active_conversation_id = conversation_id
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}],
            }
        ]
        tools = self._build_tool_definitions()

        LOGGER.info(
            "Chat request started | model=%s | prompt_chars=%d",
            self.model_name,
            len(user_message or ""),
        )
        self._emit_status("Chat agent: thinking...", "chat.start")

        try:
            max_tool_calls = 15
            tool_calls_count = 0
            while True:
                if tool_calls_count >= max_tool_calls:
                    error_message = "Error: Chat agent exceeded the maximum number of tool calls."
                    self._emit_status("Chat agent: failed", "chat.error")
                    self._emit_completion(error_message, success=False)
                    return error_message

                # Enable prompt caching for system and tools to reduce token costs
                cached_system, cached_tools = build_cached_system_and_tools(
                    system_prompt=CHAT_SYSTEM_PROMPT,
                    tools=tools,
                )

                response = self._client.messages.create(
                    model=self.model_name,
                    system=cached_system,
                    temperature=0,
                    max_tokens=4096,
                    tools=cached_tools,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    tool_calls_count += 1
                    tool_results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        tool_name = block.name or "tool"
                        tool_input = dict(block.input or {})
                        tool_id = block.id
                        result_payload = self._dispatch_tool_call(tool_name, tool_input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result_payload,
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})
                    continue

                final_text = self._collect_text(response.content)
                if final_text:
                    self._emit_streaming_chunk(
                        final_text,
                        on_chunk=on_chunk,
                        is_final=True,
                    )
                break

            duration = time.perf_counter() - started
            LOGGER.info(
                "Chat request completed | duration=%.2fs",
                duration,
            )
            self._emit_status("Chat agent: complete", "chat.complete")
            self._emit_completion(final_text, success=True)
            return final_text

        except anthropic.APIError as exc:
            duration = time.perf_counter() - started
            LOGGER.exception("Chat request failed | duration=%.2fs", duration)
            error_message = (
                "Error: Unable to contact Claude. Verify ANTHROPIC_API_KEY and network access."
            )
            self._emit_status("Chat agent: failed", "chat.error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="chat.api_error",
                    details={"message": str(exc)},
                    source=_CHAT_SOURCE,
                )
            )
            self._emit_completion(error_message, success=False)
            return error_message
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception("Chat request failed unexpectedly | duration=%.2fs", duration)
            error_message = f"Error: Chat failed: {exc}"
            self._emit_status("Chat agent: failed", "chat.error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="chat.unexpected_error",
                    details={"message": str(exc)},
                    source=_CHAT_SOURCE,
                )
            )
            self._emit_completion(error_message, success=False)
            return error_message

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Claude-compatible tool schemas."""
        tools = []
        for tool_name, handler in self._tool_handlers.items():
            tools.append(build_anthropic_tool_schema(handler, name=tool_name))
        return tools

    def _dispatch_tool_call(self, tool_name: str, tool_input: Mapping[str, Any]) -> str:
        """Execute a tool handler with full logging and auditing."""
        handler = self._tool_handlers.get(tool_name)
        params = self._sanitize_parameters((), dict(tool_input))
        self._event_bus.emit(
            ToolCallStarted(tool_name=tool_name, parameters=params, source=_CHAT_SOURCE)
        )
        started = time.perf_counter()
        success = False
        result_obj: Any = {"error": f"Unknown tool: {tool_name}"}
        try:
            if not handler:
                raise ValueError(f"Tool '{tool_name}' is not registered.")
            result_obj = handler(**tool_input)
            success = True
            return self._serialize_tool_result(result_obj)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool %s execution failed", tool_name)
            result_obj = {"error": str(exc)}
            serialized = self._serialize_tool_result(result_obj)
            self._record_tool_failure(tool_name, params, serialized, started)
            return serialized
        finally:
            duration = time.perf_counter() - started
            ToolCallLog.record(
                conversation_id=self._active_conversation_id,
                agent_role=_CHAT_SOURCE,
                tool_name=tool_name,
                tool_input=json.dumps(tool_input, ensure_ascii=False, default=str),
                tool_output=self._serialize_tool_result(result_obj),
                success=success,
                error_message=None if success else str(result_obj),
                execution_time_ms=round(duration * 1000, 2),
            )
            if success:
                self._event_bus.emit(
                    ToolCallCompleted(
                        tool_name=tool_name,
                        result=self._safe_value(result_obj, limit=320),
                        duration=duration,
                        source=_CHAT_SOURCE,
                    )
                )

    def _record_tool_failure(
        self,
        tool_name: str,
        params: Mapping[str, Any],
        result: str,
        started: float,
    ) -> None:
        duration = time.perf_counter() - started
        self._event_bus.emit(
            ToolCallFailed(
                tool_name=tool_name,
                error=result,
                duration=duration,
                source=_CHAT_SOURCE,
                parameters=params,
            )
        )
        self._event_bus.emit(
            SystemErrorEvent(
                error="chat.tool_failure",
                details={"tool": tool_name, "result": result},
                source=_CHAT_SOURCE,
            )
        )

    def _serialize_tool_result(self, payload: Any) -> str:
        """Convert tool outputs to JSON strings for tool_result blocks."""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return str(payload)

    def _collect_text(self, content: Sequence[Any]) -> str:
        """Concatenate text blocks from an Anthropic response."""
        parts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text or "")
        return "".join(parts)

    def _sanitize_parameters(
        self,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert args/kwargs into a JSON-friendly payload."""
        return {
            "args": [self._safe_value(value) for value in args],
            "kwargs": {key: self._safe_value(value) for key, value in kwargs.items()},
        }

    def _safe_value(self, value: Any, limit: int = 160) -> Any:
        """Return a compact, serializable value for event payloads."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= limit else f"{value[:limit]}..."
        text = str(value)
        return text if len(text) <= limit else f"{text[:limit]}..."

    def _emit_streaming_chunk(
        self,
        text: str,
        *,
        on_chunk: Callable[[str], None] | None,
        is_final: bool = False,
    ) -> None:
        """Emit a typed streaming event and forward to legacy callbacks."""
        if not text and not is_final:
            return
        payload = text or ""
        self._event_bus.emit(
            StreamingChunk(text=payload, source=_CHAT_SOURCE, is_final=is_final)
        )
        if not on_chunk:
            return
        callback_payload = (
            f"{config.STREAM_PREFIX}\n"
            if is_final and not payload
            else f"{config.STREAM_PREFIX}{payload}"
        )
        try:
            on_chunk(callback_payload)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Streaming callback failed (chat)", exc_info=True)

    def _emit_status(self, message: str, phase: str) -> None:
        """Emit a status update tied to this service."""
        self._event_bus.emit(
            StatusUpdate(message=message, phase=phase, source=_CHAT_SOURCE)
        )

    def _emit_completion(self, summary: str, success: bool) -> None:
        """Emit an execution completion summary."""
        self._event_bus.emit(
            ExecutionComplete(
                summary=summary or "",
                source=_CHAT_SOURCE,
                success=success,
            )
        )

    def clear_history(self) -> None:
        """Clear the conversation history (no-op with stateless client)."""
        pass
