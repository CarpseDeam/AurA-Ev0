"""Single-agent service that streams Claude responses and runs workspace tools."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import anthropic

from aura.event_bus import get_event_bus
from aura.events import (
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

LOGGER = logging.getLogger(__name__)
_SOURCE = "single_agent"

SONNET_4_5_INPUT_COST = 3.0 / 1_000_000  # $3.00 per 1M input tokens
SONNET_4_5_OUTPUT_COST = 15.0 / 1_000_000  # $15.00 per 1M output tokens

SYSTEM_PROMPT = """
You are Aura's single autonomous engineer working inside the user's repository.

Follow this workflow on every task:
1. INVESTIGATE — Use read-only tools first to understand the codebase.
2. PLAN — Think through the necessary changes internally; do not emit separate plans.
3. EXECUTE — Use write-capable tools to apply fully working code.

Quality bar:
- No placeholders, TODOs, or speculative text.
- Reference exact files/lines you touched.
- Verify work with the available tools whenever possible.
- Provide a concise final summary of what changed and any follow-up steps.

**CRITICAL - NO DOCUMENTATION FILES**
- NEVER create .md, .txt, or other documentation/summary files
- Summaries should ONLY be provided as text output in the UI
- Only create/modify files explicitly requested by the user
- Documentation files clutter the workspace and are not needed
""".strip()

ToolHandler = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class AgentTool:
    """Container describing a callable workspace tool."""

    name: str
    handler: ToolHandler
    schema: dict[str, Any]


class SingleAgentService:
    """Lightweight Anthropic wrapper that streams a single agent conversation."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        model_name: str,
        *,
        max_tokens: int = 8000,
        system_prompt: str = SYSTEM_PROMPT,
        max_tool_iterations: int = 24,
        temperature: float = 0.0,
        enable_cost_tracking: bool = True,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._system_prompt = system_prompt.strip()
        self._max_tokens = max_tokens
        self._max_tool_iterations = max_tool_iterations
        self._temperature = temperature
        self._enable_cost_tracking = enable_cost_tracking
        self._event_bus = get_event_bus()
        self._active_conversation_id: int | None = None

    @property
    def active_conversation_id(self) -> int | None:
        """Return the conversation ID associated with the current run."""
        return self._active_conversation_id

    @active_conversation_id.setter
    def active_conversation_id(self, identifier: int | None) -> None:
        """Assign the conversation ID for downstream logging."""
        self._active_conversation_id = identifier

    def execute_task(
        self,
        goal: str,
        tools: Sequence[AgentTool],
    ) -> Generator[str, None, str]:
        """Stream Claude output while executing tool calls inline."""
        if not goal:
            raise ValueError("Goal must be provided for execution.")

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": goal}]}
        ]
        tool_registry = {tool.name: tool.handler for tool in tools}
        tool_schemas = [tool.schema for tool in tools]
        total_input_tokens = 0
        total_output_tokens = 0
        iterations = 0
        self._emit_status("Single agent: investigating workspace...", "single_agent.start")

        while True:
            iterations += 1
            if iterations > self._max_tool_iterations:
                message = "Single agent exceeded the maximum number of tool iterations."
                self._handle_failure(message)
                raise RuntimeError(message)

            cached_system, cached_tools = build_cached_system_and_tools(
                system_prompt=self._system_prompt,
                tools=tool_schemas,
            )

            LOGGER.debug(
                "Dispatching Claude request | model=%s | tokens=%s",
                self._model_name,
                self._max_tokens,
            )

            try:
                streamed_any = False
                with self._client.messages.stream(
                    model=self._model_name,
                    system=cached_system,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=messages,
                    tools=cached_tools if cached_tools else None,
                ) as stream:
                    for delta in stream.text_stream:
                        if not delta:
                            continue
                        streamed_any = True
                        self._emit_stream(delta)
                        yield delta
                    response = stream.get_final_message()
            except anthropic.APIError as exc:
                message = "Claude API request failed. Check your connection and API key."
                LOGGER.exception("Anthropic API error: %s", exc)
                self._handle_failure(message, details={"error": str(exc)})
                raise

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            LOGGER.info(
                "Claude response | stop_reason=%s | input_tokens=%s | output_tokens=%s",
                response.stop_reason,
                input_tokens,
                output_tokens,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = self._collect_tool_results(response.content, tool_registry)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "end_turn":
                final_text = self._collect_text(response.content)
                if final_text and not streamed_any:
                    self._emit_stream(final_text)
                    yield final_text
                self._emit_status("Single agent: task complete", "single_agent.complete")
                self._emit_stream("", is_final=True)
                self._emit_completion(final_text, success=True)
                if self._enable_cost_tracking:
                    self._log_cost(goal, total_input_tokens, total_output_tokens)
                return final_text

            message = f"Claude stopped unexpectedly (reason={response.stop_reason})."
            self._handle_failure(message)
            raise RuntimeError(message)

    def _collect_text(self, content: Sequence[Any]) -> str:
        """Concatenate any text blocks from Claude."""
        chunks: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", "") or ""
                chunks.append(text)
        return "".join(chunks).strip()

    def _collect_tool_results(
        self,
        content: Sequence[Any],
        tool_registry: Mapping[str, ToolHandler],
    ) -> list[dict[str, Any]]:
        """Execute tool calls requested by Claude."""
        results: list[dict[str, Any]] = []
        for block in content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_name = getattr(block, "name", None) or "tool"
            handler = tool_registry.get(tool_name)
            payload = self._unwrap_tool_input(dict(getattr(block, "input", {}) or {}))
            tool_id = getattr(block, "id", "")

            LOGGER.debug("Executing tool %s with payload keys=%s", tool_name, list(payload.keys()))
            started = time.perf_counter()
            serialized_payload = json.dumps(payload, ensure_ascii=False, default=str)
            params = self._sanitize_parameters((), payload)
            self._event_bus.emit(
                ToolCallStarted(tool_name=tool_name, parameters=params, source=_SOURCE)
            )
            success = False
            result_payload: Any = {"error": f"Unknown tool: {tool_name}"}
            try:
                if handler is None:
                    raise ValueError(f"Tool '{tool_name}' is not registered.")
                result_payload = handler(**payload)
                success = True
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Tool %s failed", tool_name)
                result_payload = {"error": str(exc)}
                self._event_bus.emit(
                    ToolCallFailed(
                        tool_name=tool_name,
                        error=str(exc),
                        duration=time.perf_counter() - started,
                        source=_SOURCE,
                        parameters=params,
                    )
                )
            else:
                self._event_bus.emit(
                    ToolCallCompleted(
                        tool_name=tool_name,
                        result=self._safe_value(result_payload),
                        duration=time.perf_counter() - started,
                        source=_SOURCE,
                    )
                )
            finally:
                ToolCallLog.record(
                    conversation_id=self._active_conversation_id,
                    agent_role=_SOURCE,
                    tool_name=tool_name,
                    tool_input=serialized_payload,
                    tool_output=self._serialize_tool_result(result_payload),
                    success=success,
                    error_message=None if success else str(result_payload),
                    execution_time_ms=round((time.perf_counter() - started) * 1000, 2),
                )

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": self._serialize_tool_result(result_payload),
                }
            )
        return results

    def _unwrap_tool_input(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Handle Claude's occasional payload-wrapped tool inputs."""
        if "payload" in payload and len(payload) == 1:
            nested = payload.get("payload")
            if isinstance(nested, str):
                try:
                    return json.loads(nested)
                except json.JSONDecodeError:
                    LOGGER.warning("Failed to decode tool payload JSON")
        return dict(payload)

    def _serialize_tool_result(self, payload: Any) -> str:
        """Convert tool outputs to strings suitable for Claude tool_results."""
        if isinstance(payload, str):
            return payload
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except TypeError:
            return str(payload)

    def _sanitize_parameters(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a JSON-serializable view of tool parameters."""
        return {
            "args": [self._safe_value(arg) for arg in args],
            "kwargs": {key: self._safe_value(value) for key, value in kwargs.items()},
        }

    def _safe_value(self, value: Any, limit: int = 200) -> Any:
        """Clamp long values for log/event payloads."""
        if value is None or isinstance(value, (bool, int, float)):
            return value
        text = value if isinstance(value, str) else str(value)
        return text if len(text) <= limit else f"{text[:limit]}..."

    def _emit_stream(self, text: str, *, is_final: bool = False) -> None:
        """Emit streaming chunks to the event bus."""
        if not text and not is_final:
            return
        self._event_bus.emit(StreamingChunk(text=text, source=_SOURCE, is_final=is_final))

    def _emit_status(self, message: str, phase: str) -> None:
        """Publish a status update."""
        self._event_bus.emit(StatusUpdate(message=message, phase=phase, source=_SOURCE))

    def _emit_completion(self, summary: str, *, success: bool) -> None:
        """Publish a completion event."""
        self._event_bus.emit(
            ExecutionComplete(summary=summary or "", source=_SOURCE, success=success)
        )

    def _handle_failure(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Emit failure events and completion updates."""
        LOGGER.error("Single agent failure: %s", message)
        self._event_bus.emit(
            SystemErrorEvent(error=message, details=details, source=_SOURCE)
        )
        self._emit_completion(message, success=False)

    def _log_cost(self, goal: str, input_tokens: int, output_tokens: int) -> None:
        """Estimate and log USD cost for a completed task."""
        input_cost = input_tokens * SONNET_4_5_INPUT_COST
        output_cost = output_tokens * SONNET_4_5_OUTPUT_COST
        total_cost = input_cost + output_cost
        LOGGER.info(
            "Single agent task complete | goal_chars=%d | input_tokens=%s | output_tokens=%s | "
            "input_cost=$%.4f | output_cost=$%.4f | total_cost=$%.4f",
            len(goal),
            input_tokens,
            output_tokens,
            input_cost,
            output_cost,
            total_cost,
        )


__all__ = ["AgentTool", "SingleAgentService"]
