"""Executor agent service for applying ExecutionPlans with write-only tools."""

from __future__ import annotations

import difflib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic

from aura.event_bus import get_event_bus
from aura.events import (
    AgentEvent,
    ExecutionComplete,
    FileOperation,
    PhaseTransition,
    StatusUpdate,
    StreamingChunk,
    SystemErrorEvent,
    TaskProgressEvent,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from aura.models import ExecutionPlan, ToolCallLog
from aura.prompts import EXECUTOR_PROMPT
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_EXECUTOR_SOURCE = "executor"


@dataclass
class ExecutorAgentService:
    """Apply ExecutionPlan operations via Claude Sonnet 4.5 and file tools."""

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _event_bus: Any = field(init=False, repr=False)
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _active_conversation_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._event_bus = get_event_bus()
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def execute_plan(
        self,
        execution_plan: ExecutionPlan,
        on_chunk: Optional[Callable[[str], None]] = None,
        conversation_id: int | None = None,
    ) -> str:
        """Execute a validated ExecutionPlan using the configured executor model."""
        started = time.perf_counter()
        self._active_conversation_id = conversation_id
        plan_json = execution_plan.to_json(indent=2)
        LOGGER.info(
            "Executor execution started | model=%s | operations=%d",
            self.model_name,
            len(execution_plan.operations),
        )
        self._event_bus.emit(
            PhaseTransition(from_phase="idle", to_phase="executor", source=_EXECUTOR_SOURCE)
        )
        self._emit_status("Executor agent: applying execution plan...", "executor.run")
        self._event_bus.emit(
            AgentEvent(
                name="executor.started",
                payload={"operations": len(execution_plan.operations)},
                source=_EXECUTOR_SOURCE,
            )
        )
        self._event_bus.emit(
            TaskProgressEvent(
                message="Executor applying plan operations",
                percent=0.6,
                source=_EXECUTOR_SOURCE,
            )
        )

        plan_message = (
            "The JSON ExecutionPlan below has been fully validated by the system. "
            "Execute each operation sequentially, using the provided content verbatim. "
            "Do not invent additional work or reorder operations.\n\n"
            f"{plan_json}"
        )
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": plan_message}],
            }
        ]
        tools = self._build_anthropic_tools()

        try:
            max_tool_calls = 15
            tool_calls_count = 0
            while True:
                if tool_calls_count >= max_tool_calls:
                    error_message = "Error: Executor exceeded the maximum number of tool calls."
                    self._emit_status("Executor agent: failed", "executor.error")
                    self._emit_completion(error_message, success=False)
                    return error_message

                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=8096,
                    system=EXECUTOR_PROMPT,
                    messages=messages,
                    tools=tools,
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
                        result = self._invoke_workspace_tool(tool_name, tool_input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": result,
                            }
                        )
                    messages.append({"role": "user", "content": tool_results})
                    continue

                if response.stop_reason in ("end_turn", "stop_sequence", None):
                    final_text = self._collect_text(response.content)
                    duration = time.perf_counter() - started
                    LOGGER.info(
                        "Executor execution completed | duration=%.2fs | response_chars=%d",
                        duration,
                        len(final_text),
                    )
                    if final_text:
                        self._emit_streaming_chunk(final_text, on_chunk=on_chunk)
                    self._emit_streaming_chunk(
                        "Execution complete",
                        on_chunk=on_chunk,
                        is_final=True,
                    )
                    self._emit_status("Executor agent: execution complete", "executor.complete")
                    self._event_bus.emit(
                        PhaseTransition(from_phase="executor", to_phase="idle", source=_EXECUTOR_SOURCE)
                    )
                    self._event_bus.emit(
                        AgentEvent(
                            name="executor.complete",
                            payload={"operations": len(execution_plan.operations)},
                            source=_EXECUTOR_SOURCE,
                        )
                    )
                    self._event_bus.emit(
                        TaskProgressEvent(
                            message="Executor finished applying plan",
                            percent=0.98,
                            source=_EXECUTOR_SOURCE,
                        )
                    )
                    summary = final_text or "Execution complete."
                    self._emit_completion(summary, success=True)
                    return summary

                LOGGER.warning("Executor received unexpected stop_reason: %s", response.stop_reason)
                break

        except anthropic.APIError as exc:
            duration = time.perf_counter() - started
            LOGGER.exception("Executor execution failed | duration=%.2fs | error=%s", duration, exc)
            error_message = (
                "Error: Unable to contact Claude for execution. Verify ANTHROPIC_API_KEY and network access."
            )
            self._emit_status("Executor agent: failed", "executor.error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="executor.api_error",
                    details={"message": str(exc)},
                    source=_EXECUTOR_SOURCE,
                )
            )
            self._event_bus.emit(
                PhaseTransition(from_phase="executor", to_phase="executor.error", source=_EXECUTOR_SOURCE)
            )
            self._emit_completion(error_message, success=False)
            return error_message
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception("Executor execution failed unexpectedly | duration=%.2fs", duration)
            error_message = f"Error: Execution failed: {exc}"
            self._emit_status("Executor agent: failed", "executor.error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="executor.unexpected_error",
                    details={"message": str(exc)},
                    source=_EXECUTOR_SOURCE,
                )
            )
            self._event_bus.emit(
                PhaseTransition(from_phase="executor", to_phase="executor.error", source=_EXECUTOR_SOURCE)
            )
            self._emit_completion(error_message, success=False)
            return error_message

    # ------------------------------------------------------------------ #
    # Tool execution helpers
    # ------------------------------------------------------------------ #
    def _build_anthropic_tools(self) -> list[dict[str, Any]]:
        """Return Claude tool schema for allowed write operations."""
        from aura.tools.anthropic_tool_builder import build_anthropic_tool_schema

        tools = [
            build_anthropic_tool_schema(self.tool_manager.create_file, name_override="create_file"),
            build_anthropic_tool_schema(self.tool_manager.modify_file, name_override="modify_file"),
            build_anthropic_tool_schema(self.tool_manager.delete_file, name_override="delete_file"),
        ]
        return tools

    def _invoke_workspace_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a workspace tool with auditing, logging, and DB persistence."""
        params = self._sanitize_tool_args(tool_input)
        self._event_bus.emit(
            ToolCallStarted(tool_name=tool_name, parameters=params, source=_EXECUTOR_SOURCE)
        )
        started = time.perf_counter()
        result_text = ""
        success = False
        try:
            result_text = self._execute_tool(tool_name, tool_input)
            success = True
            self._maybe_emit_file_operation(tool_name, tool_input)
            return result_text
        except Exception as exc:
            error_text = str(exc)
            result_text = error_text
            self._event_bus.emit(
                ToolCallFailed(
                    tool_name=tool_name,
                    error=error_text,
                    duration=time.perf_counter() - started,
                    source=_EXECUTOR_SOURCE,
                    parameters=params,
                )
            )
            self._event_bus.emit(
                SystemErrorEvent(
                    error="executor.tool_failure",
                    details={"tool": tool_name, "message": error_text},
                    source=_EXECUTOR_SOURCE,
                )
            )
            raise
        finally:
            duration = time.perf_counter() - started
            ToolCallLog.record(
                conversation_id=self._active_conversation_id,
                agent_role=_EXECUTOR_SOURCE,
                tool_name=tool_name,
                tool_input=json.dumps(tool_input, ensure_ascii=False, default=str),
                tool_output=result_text,
                success=success,
                error_message=None if success else result_text,
                execution_time_ms=round(duration * 1000, 2),
            )
            if success:
                self._event_bus.emit(
                    ToolCallCompleted(
                        tool_name=tool_name,
                        result=self._safe_value(result_text, limit=320),
                        duration=duration,
                        source=_EXECUTOR_SOURCE,
                    )
                )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Invoke the appropriate ToolManager method."""
        if tool_name == "create_file":
            path = tool_input.get("path", "")
            content = tool_input.get("content", "")
            result = self.tool_manager.create_file(path, content)
            return f"+ Created {path}\n{result}"

        if tool_name == "modify_file":
            path = tool_input.get("path", "")
            old_content = tool_input.get("old_content", "")
            new_content = tool_input.get("new_content", "")
            diff_block = self._build_diff_block(path, old_content, new_content)
            result = self.tool_manager.modify_file(path, old_content, new_content)
            parts = [f"~ Modified {path}", result]
            if diff_block:
                parts.append(diff_block)
            return "\n".join(parts)

        if tool_name == "delete_file":
            path = tool_input.get("path", "")
            result = self.tool_manager.delete_file(path)
            return f"- Deleted {path}\n{result}"

        raise ValueError(f"Unknown tool '{tool_name}'")

    def _build_diff_block(self, path: str, old_content: str, new_content: str) -> str:
        """Return a fenced unified diff block for modify_file operations."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{path}" if path else "a/file",
                tofile=f"b/{path}" if path else "b/file",
                lineterm="",
            )
        )
        if not diff_lines:
            return ""
        diff_text = "\n".join(diff_lines)
        return f"```diff\n{diff_text}\n```"

    def _collect_text(self, content: list[Any]) -> str:
        """Concatenate text blocks from an Anthropic response."""
        parts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text or "")
        return "".join(parts)

    def _sanitize_tool_args(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        """Return a sanitized copy of the provided tool arguments."""
        if not payload:
            return {}
        return {key: self._safe_value(value) for key, value in payload.items()}

    def _safe_value(self, value: Any, limit: int = 160) -> Any:
        """Return a compact representation suitable for event payloads."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= limit else f"{value[:limit]}..."
        text = str(value)
        return text if len(text) <= limit else f"{text[:limit]}..."

    def _maybe_emit_file_operation(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None,
    ) -> None:
        """Emit FileOperation and agent events for write tools."""
        if tool_name not in {"create_file", "modify_file", "delete_file"}:
            return
        if not tool_input:
            return
        path = tool_input.get("path")
        if not isinstance(path, str) or not path:
            return
        details: dict[str, Any] | None = None
        if tool_name == "create_file":
            content = tool_input.get("content", "")
            if isinstance(content, str):
                details = {"bytes": len(content.encode("utf-8"))}
        elif tool_name == "modify_file":
            old_len = len(tool_input.get("old_content", "") or "")
            new_len = len(tool_input.get("new_content", "") or "")
            details = {"old_length": old_len, "new_length": new_len}
        self._event_bus.emit(
            FileOperation(
                operation=tool_name,
                filepath=path,
                details=details,
                source=_EXECUTOR_SOURCE,
            )
        )
        self._event_bus.emit(
            AgentEvent(
                name="executor.file_operation",
                payload={"operation": tool_name, "file": path},
                source=_EXECUTOR_SOURCE,
            )
        )

    def _emit_streaming_chunk(
        self,
        text: str,
        *,
        on_chunk: Optional[Callable[[str], None]],
        is_final: bool = False,
    ) -> None:
        """Emit streaming chunks for executor output."""
        if not text and not is_final:
            return
        payload = text or ""
        self._event_bus.emit(
            StreamingChunk(text=payload, source=_EXECUTOR_SOURCE, is_final=is_final)
        )
        if not on_chunk:
            return
        message = f"{payload}\n" if payload else "\n"
        try:
            on_chunk(message)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Executor streaming callback failed", exc_info=True)

    def _emit_status(self, message: str, phase: str) -> None:
        """Emit a StatusUpdate for the executor."""
        self._event_bus.emit(
            StatusUpdate(message=message, phase=phase, source=_EXECUTOR_SOURCE)
        )

    def _emit_completion(self, summary: str, success: bool) -> None:
        """Emit an ExecutionComplete event."""
        self._event_bus.emit(
            ExecutionComplete(
                summary=summary or "",
                source=_EXECUTOR_SOURCE,
                success=success,
            )
        )


__all__ = ["ExecutorAgentService"]
