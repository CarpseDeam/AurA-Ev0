"""Analyst agent powered by Claude Sonnet 4.5 that produces execution plans."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import anthropic
from pydantic import ValidationError

from aura import config
from aura.event_bus import get_event_bus
from aura.events import (
    AgentEvent,
    ExecutionComplete,
    PhaseTransition,
    StatusUpdate,
    StreamingChunk,
    SystemErrorEvent,
    TaskProgressEvent,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from aura.models import ExecutionPlan, Message, MessageRole, ToolCallLog
from aura.prompts import ANALYST_PROMPT
from aura.tools import godot_tools
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_ANALYST_SOURCE = "analyst"


ToolHandler = Callable[..., Any]


def unwrap_tool_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Unwrap payload-wrapped tool inputs from Claude API.

    Claude sometimes wraps tool inputs in {"payload": "JSON_STRING"} format.
    This function extracts and parses that JSON string if present.
    """
    if "payload" in kwargs and len(kwargs) == 1:
        try:
            payload_value = kwargs["payload"]
            if isinstance(payload_value, str):
                return json.loads(payload_value)
        except (json.JSONDecodeError, TypeError) as exc:
            LOGGER.warning("Failed to unwrap tool input payload: %s", exc)
    return kwargs


@dataclass
class AnalystAgentService:
    """Runs the analyst loop with Claude and Aura's read-only tools."""

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _event_bus: Any = field(init=False, repr=False)
    _tool_handlers: Mapping[str, ToolHandler] = field(init=False, repr=False)
    _latest_plan: ExecutionPlan | None = field(default=None, init=False, repr=False)
    _active_conversation_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._event_bus = get_event_bus()
        self._tool_handlers = {
            "list_project_files": self.tool_manager.list_project_files,
            "read_project_file": self.tool_manager.read_project_file,
            "get_imports": self.tool_manager.get_imports,
            "get_project_structure": self.tool_manager.get_project_structure,
            "search_in_files": self.tool_manager.search_in_files,
            "get_git_status": self.tool_manager.get_git_status,
            "get_cyclomatic_complexity": self.tool_manager.get_cyclomatic_complexity,
            "detect_duplicate_code": self.tool_manager.detect_duplicate_code,
            "check_naming_conventions": self.tool_manager.check_naming_conventions,
            "analyze_type_hints": self.tool_manager.analyze_type_hints,
            "inspect_docstrings": self.tool_manager.inspect_docstrings,
            "get_function_signatures": self.tool_manager.get_function_signatures,
            "find_unused_imports": self.tool_manager.find_unused_imports,
            "get_class_hierarchy": self.tool_manager.get_class_hierarchy,
            "get_dependency_graph": self.tool_manager.get_dependency_graph,
            "get_code_metrics": self.tool_manager.get_code_metrics,
            "submit_execution_plan": self._handle_submit_execution_plan,
            "read_godot_scene_tree": godot_tools.read_godot_scene_tree,
            "get_project_godot_config": godot_tools.get_project_godot_config,
        }

    def analyze_and_plan(
        self,
        user_request: str,
        *,
        on_chunk: Callable[[str], None] | None = None,
        conversation_id: int | None = None,
    ) -> ExecutionPlan | str:
        """Gather context with Claude tools and return an execution plan."""
        started = time.perf_counter()
        self._latest_plan = None
        self._active_conversation_id = conversation_id
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_request}],
            }
        ]
        tools = self._build_tool_definitions()

        LOGGER.info(
            "Analyst analysis started | model=%s | prompt_chars=%d",
            self.model_name,
            len(user_request or ""),
        )
        self._event_bus.emit(
            PhaseTransition(from_phase="idle", to_phase="analyst", source=_ANALYST_SOURCE)
        )
        self._emit_status("Analyst agent: gathering context...", "analyst.start")
        self._event_bus.emit(
            AgentEvent(
                name="analyst.started",
                payload={"model": self.model_name},
                source=_ANALYST_SOURCE,
            )
        )
        self._event_bus.emit(
            TaskProgressEvent(
                message="Analyst collecting repository signals",
                percent=0.1,
                source=_ANALYST_SOURCE,
            )
        )

        try:
            max_tool_calls = 15
            tool_calls_count = 0
            while True:
                if tool_calls_count >= max_tool_calls:
                    error_message = "Error: Analyst exceeded the maximum number of tool calls."
                    self._emit_status("Analyst agent: failed", "analyst.error")
                    self._emit_completion(error_message, success=False)
                    return error_message

                response = self._client.messages.create(
                    model=self.model_name,
                    system=ANALYST_PROMPT,
                    temperature=0,
                    max_tokens=6000,
                    tools=tools,
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
                        source=_ANALYST_SOURCE,
                        on_chunk=on_chunk,
                        is_final=True,
                    )
                break

            if not self._latest_plan:
                error_message = (
                    "Error: Analyst did not provide a submit_execution_plan tool call."
                )
                self._emit_status("Analyst agent: failed", "analyst.error")
                self._event_bus.emit(
                    PhaseTransition(
                        from_phase="analyst",
                        to_phase="analyst.error",
                        source=_ANALYST_SOURCE,
                    )
                )
                self._emit_completion(error_message, success=False)
                return error_message

            duration = time.perf_counter() - started
            LOGGER.info(
                "Analyst plan completed | duration=%.2fs | operations=%d",
                duration,
                len(self._latest_plan.operations),
            )
            self._emit_status("Analyst agent: execution plan ready", "analyst.complete")
            self._event_bus.emit(
                PhaseTransition(
                    from_phase="analyst",
                    to_phase="idle",
                    source=_ANALYST_SOURCE,
                )
            )
            self._event_bus.emit(
                TaskProgressEvent(
                    message="Analyst execution plan ready",
                    percent=0.45,
                    source=_ANALYST_SOURCE,
                )
            )
            summary = (
                f"Analyst completed execution plan with {len(self._latest_plan.operations)} operations."
            )
            self._emit_completion(summary, success=True)
            return self._latest_plan

        except anthropic.APIError as exc:
            duration = time.perf_counter() - started
            LOGGER.exception("Analyst request failed | duration=%.2fs", duration)
            error_message = (
                "Error: Unable to contact Claude. Verify ANTHROPIC_API_KEY and network access."
            )
            self._emit_status("Analyst agent: failed", "analyst.error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.api_error",
                    details={"message": str(exc)},
                    source=_ANALYST_SOURCE,
                )
            )
            self._event_bus.emit(
                PhaseTransition(
                    from_phase="analyst",
                    to_phase="analyst.error",
                    source=_ANALYST_SOURCE,
                )
            )
            self._emit_completion(error_message, success=False)
            return error_message
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception("Analyst request failed unexpectedly | duration=%.2fs", duration)
            error_message = f"Error: Analyst failed: {exc}"
            self._emit_status("Analyst agent: failed", "analyst.error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.unexpected_error",
                    details={"message": str(exc)},
                    source=_ANALYST_SOURCE,
                )
            )
            self._event_bus.emit(
                PhaseTransition(
                    from_phase="analyst",
                    to_phase="analyst.error",
                    source=_ANALYST_SOURCE,
                )
            )
            self._emit_completion(error_message, success=False)
            return error_message

    def _build_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Claude-compatible tool schemas."""
        from aura.tools.anthropic_tool_builder import build_anthropic_tool_schema

        tools = []
        for tool_name, handler in self._tool_handlers.items():
            tools.append(build_anthropic_tool_schema(handler, name=tool_name))
        return tools

    def _dispatch_tool_call(self, tool_name: str, tool_input: Mapping[str, Any]) -> str:
        """Execute a tool handler with full logging and auditing."""
        handler = self._tool_handlers.get(tool_name)
        params = self._sanitize_parameters((), dict(tool_input))
        self._event_bus.emit(
            ToolCallStarted(tool_name=tool_name, parameters=params, source=_ANALYST_SOURCE)
        )
        self._event_bus.emit(
            AgentEvent(
                name="analyst.tool_call",
                payload={"tool": tool_name, "parameters": params},
                source=_ANALYST_SOURCE,
            )
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
        except ValidationError as exc:
            LOGGER.warning("Tool %s validation failed: %s", tool_name, exc)
            result_obj = {"error": "Validation failed", "details": exc.errors()}
            serialized = self._serialize_tool_result(result_obj)
            self._record_tool_failure(tool_name, params, serialized, started)
            return serialized
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
                agent_role=_ANALYST_SOURCE,
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
                        source=_ANALYST_SOURCE,
                    )
                )
                self._event_bus.emit(
                    AgentEvent(
                        name="analyst.tool_result",
                        payload={"tool": tool_name},
                        source=_ANALYST_SOURCE,
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
                source=_ANALYST_SOURCE,
                parameters=params,
            )
        )
        self._event_bus.emit(
            AgentEvent(
                name="analyst.tool_result",
                payload={"tool": tool_name, "error": True},
                source=_ANALYST_SOURCE,
            )
        )
        self._event_bus.emit(
            SystemErrorEvent(
                error="analyst.tool_failure",
                details={"tool": tool_name, "result": result},
                source=_ANALYST_SOURCE,
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

    def _handle_submit_execution_plan(self, **payload: Any) -> dict[str, Any]:
        """Validate and persist the submitted execution plan.

        Handles both direct JSON objects and payload-wrapped inputs from Claude.
        """
        unwrapped = unwrap_tool_input(payload)

        LOGGER.debug("Received execution plan input: %s", json.dumps(unwrapped, default=str)[:200])

        try:
            plan = ExecutionPlan.model_validate(unwrapped)
        except ValidationError as exc:
            LOGGER.warning("Execution plan validation failed: %s", exc)
            return {"success": False, "errors": exc.errors()}

        self._latest_plan = plan
        plan_json = plan.to_json(indent=2)
        if self._active_conversation_id is not None:
            Message.create(
                conversation_id=self._active_conversation_id,
                role=MessageRole.TOOL_RESULT,
                content=plan_json,
            )
        self._event_bus.emit(
            AgentEvent(
                name="analyst.blueprint_ready",
                payload={"operations": len(plan.operations)},
                source=_ANALYST_SOURCE,
            )
        )
        return {
            "success": True,
            "operations": len(plan.operations),
            "estimated_files": plan.estimated_files,
        }

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
        source: str,
        on_chunk: Callable[[str], None] | None,
        is_final: bool = False,
    ) -> None:
        """Emit a typed streaming event and forward to legacy callbacks."""
        if not text and not is_final:
            return
        payload = text or ""
        self._event_bus.emit(
            StreamingChunk(text=payload, source=source, is_final=is_final)
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
            LOGGER.debug("Streaming callback failed (analyst)", exc_info=True)

    def _emit_status(self, message: str, phase: str) -> None:
        """Emit a status update tied to this service."""
        self._event_bus.emit(
            StatusUpdate(message=message, phase=phase, source=_ANALYST_SOURCE)
        )

    def _emit_completion(self, summary: str, success: bool) -> None:
        """Emit an execution completion summary."""
        self._event_bus.emit(
            ExecutionComplete(
                summary=summary or "",
                source=_ANALYST_SOURCE,
                success=success,
            )
        )


__all__ = ["AnalystAgentService"]