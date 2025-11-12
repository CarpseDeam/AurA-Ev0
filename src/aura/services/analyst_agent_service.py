"""Analyst agent powered by Claude Sonnet 4.5 that produces execution plans."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
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
from aura.utils.prompt_caching import build_cached_system_and_tools
from aura.prompts import ANALYST_PROMPT
from aura.tools import godot_tools
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_ANALYST_SOURCE = "analyst"

# History filtering constants
MAX_HISTORY_MESSAGES = 6  # Last 3 user/assistant pairs
INVESTIGATION_MAX_TOKENS = 8192

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


def filter_conversation_history(
    history: Sequence[dict[str, Any]],
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict[str, Any]]:
    """Keep only the last N user/assistant messages, filtering out tool_result messages."""
    filtered = [msg for msg in history if msg.get("role") != "tool_result"]
    if len(filtered) > max_messages:
        filtered = filtered[-max_messages:]
    return filtered


@dataclass
class AnalystAgentService:
    """Runs the analyst loop with Claude and Aura's read-only tools."""

    api_key: str
    tool_manager: ToolManager
    investigation_model: str
    planning_model: str
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _event_bus: Any = field(init=False, repr=False)
    _tool_handlers: Mapping[str, ToolHandler] = field(init=False, repr=False)
    _latest_plan: ExecutionPlan | None = field(default=None, init=False, repr=False)
    _active_conversation_id: int | None = field(default=None, init=False, repr=False)
    _current_user_request: str | None = field(default=None, init=False, repr=False)

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
            "verify_asset_paths": self.tool_manager.verify_asset_paths,
            "list_project_assets": self.tool_manager.list_project_assets,
            "search_assets_by_pattern": self.tool_manager.search_assets_by_pattern,
            "search_project_assets": self.tool_manager.search_project_assets,
            "list_scenes": self.tool_manager.list_scenes,
            "get_asset_metadata": self.tool_manager.get_asset_metadata,
            "respect_gitignore": self.tool_manager.respect_gitignore,
            "submit_execution_plan": self._handle_submit_execution_plan,
            # Godot read-only inspection tools
            "read_godot_scene": godot_tools.read_godot_scene,
            "read_godot_scene_tree": godot_tools.read_godot_scene_tree,
            "validate_godot_scene": godot_tools.validate_godot_scene,
            "get_project_godot_config": godot_tools.get_project_godot_config,
            # NOTE: add_godot_node and modify_godot_node_property are NOT included here
            # because they are investigation tools, not file operations. The Analyst must
            # use MODIFY operations with full .tscn file content instead.
        }

    def analyze_and_plan(
        self,
        user_request: str,
        *,
        on_chunk: Callable[[str], None] | None = None,
        conversation_id: int | None = None,
        conversation_history: Sequence[dict[str, Any]] | None = None,
    ) -> ExecutionPlan | str:
        """Gather context with Claude tools and return an execution plan.

        Args:
            user_request: The current user request/goal
            on_chunk: Optional callback for streaming response chunks
            conversation_id: Optional conversation ID for message persistence
            conversation_history: Optional list of previous conversation messages

        Returns:
            ExecutionPlan object or error string
        """
        started = time.perf_counter()
        self._latest_plan = None
        self._active_conversation_id = conversation_id
        self._current_user_request = user_request

        # Apply sliding window history filter
        filtered_history: list[dict[str, Any]] = []
        if conversation_history:
            filtered_history = filter_conversation_history(
                conversation_history,
                max_messages=MAX_HISTORY_MESSAGES,
            )
            LOGGER.info(
                "Analyst using conversation history | messages=%d | original_messages=%d",
                len(filtered_history),
                len(conversation_history),
            )

        # Build messages list with history + current request
        investigation_messages = list(filtered_history)
        investigation_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_request}],
        })

        LOGGER.info(
            "Analyst analysis started | investigation_model=%s | planning_model=%s | prompt_chars=%d",
            self.investigation_model,
            self.planning_model,
            len(user_request or ""),
        )
        self._event_bus.emit(
            PhaseTransition(from_phase="idle", to_phase="analyst", source=_ANALYST_SOURCE)
        )
        self._emit_status("Analyst agent: gathering context...", "analyst.start")
        self._event_bus.emit(
            AgentEvent(
                name="analyst.started",
                payload={
                    "investigation_model": self.investigation_model,
                    "planning_model": self.planning_model,
                },
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
            max_tool_calls = 30
            tool_calls = 0
            final_response_text = ""
            enforcement_retries = 0
            max_enforcement_retries = 3
            force_plan_submission = False

            # Main loop for investigation and planning
            while True:
                if tool_calls >= max_tool_calls:
                    error_message = (
                        "Error: Analyst exceeded the maximum number of tool calls."
                    )
                    self._emit_status("Analyst agent: failed", "analyst.error")
                    self._event_bus.emit(
                        SystemErrorEvent(
                            error="analyst.tool_limit",
                            details={"max_tool_calls": max_tool_calls},
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

                # Enable prompt caching for system and tools to reduce token costs
                cached_system, cached_tools = build_cached_system_and_tools(
                    system_prompt=ANALYST_PROMPT,
                    tools=self._build_tool_definitions(),
                )

                request_payload: dict[str, Any] = {
                    "model": self.investigation_model,
                    "system": cached_system,
                    "temperature": 0,
                    "max_tokens": INVESTIGATION_MAX_TOKENS,
                    "tools": cached_tools,
                    "messages": investigation_messages,
                }
                if force_plan_submission:
                    request_payload["tool_choice"] = {
                        "type": "tool",
                        "name": "submit_execution_plan",
                    }

                response = self._client.messages.create(**request_payload)
                investigation_messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    tool_calls += 1
                    if tool_calls % 5 == 0:
                        LOGGER.info("Analyst investigation in progress | tool_calls=%d/%d", tool_calls, max_tool_calls)
                    tool_results = self._collect_tool_results(response.content)
                    if tool_results:
                        investigation_messages.append({"role": "user", "content": tool_results})

                    # Check for plan immediately after collecting tool results
                    if self._latest_plan:
                        break
                    continue

                # If we get here, stop_reason is NOT "tool_use" (likely "end_turn")
                # BUT the response might still contain tool_use blocks that need results collected
                # Check if Analyst output narrative text instead of calling submit_execution_plan
                if self._latest_plan is None:
                    # First, collect any tool results from this response before injecting enforcement
                    tool_results = self._collect_tool_results(response.content)
                    if tool_results:
                        investigation_messages.append({"role": "user", "content": tool_results})

                    # Check if plan was submitted after collecting tool results
                    if self._latest_plan:
                        break

                    # Check enforcement retry limit
                    if enforcement_retries >= max_enforcement_retries:
                        duration = time.perf_counter() - started
                        LOGGER.error(
                            "Analyst failed to submit execution plan after %d enforcement attempts | duration=%.2fs",
                            enforcement_retries,
                            duration,
                        )
                        error_message = (
                            "Error: Unable to generate execution plan. The analyst could not formulate a valid plan "
                            "after multiple attempts. Please try:\n"
                            "1. Rephrasing your request with more specific details\n"
                            "2. Breaking down your request into smaller, focused tasks\n"
                            "3. Providing more context about what you want to accomplish"
                        )
                        self._emit_status("Analyst agent: failed to create plan", "analyst.error")
                        self._event_bus.emit(
                            SystemErrorEvent(
                                error="analyst.plan_generation_failed",
                                details={"enforcement_attempts": enforcement_retries},
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

                    # Force the Analyst to call the tool instead of ending with text
                    enforcement_retries += 1
                    force_plan_submission = True
                    LOGGER.warning(
                        "Analyst output narrative text instead of calling submit_execution_plan. "
                        "Injecting enforcement message (attempt %d/%d).",
                        enforcement_retries,
                        max_enforcement_retries,
                    )
                    investigation_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "Call submit_execution_plan now. No text output."}],
                    })
                    # Continue the loop to force another API call with the enforcement message
                    continue

                final_response_text = (self._collect_text(response.content) or "").strip()
                break

            if self._latest_plan:
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

            if final_response_text:
                duration = time.perf_counter() - started
                LOGGER.error(
                    "Analyst failed to submit ExecutionPlan | duration=%.2fs",
                    duration,
                )
                summary = self._safe_value(final_response_text, limit=200)
                error_message = (
                    f"Error: Analyst provided narrative text instead of ExecutionPlan. "
                    f"Response: {summary}"
                )
                self._emit_status("Analyst agent: failed to submit ExecutionPlan", "analyst.error")
                self._event_bus.emit(
                    PhaseTransition(
                        from_phase="analyst",
                        to_phase="analyst.error",
                        source=_ANALYST_SOURCE,
                    )
                )
                self._event_bus.emit(
                    SystemErrorEvent(
                        error="analyst.narrative_instead_of_plan",
                        details={"narrative_preview": summary},
                        source=_ANALYST_SOURCE,
                    )
                )
                self._emit_completion(error_message, success=False)
                return error_message

            error_message = "Error: Analyst did not provide a submit_execution_plan tool call."
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

    def _build_tool_definitions(self, allowed_tools: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return Claude-compatible tool schemas."""
        from aura.tools.anthropic_tool_builder import build_anthropic_tool_schema, build_pydantic_tool_schema

        tools: list[dict[str, Any]] = []
        if allowed_tools is None:
            tool_items = self._tool_handlers.items()
        else:
            tool_items = [
                (tool_name, self._tool_handlers[tool_name])
                for tool_name in allowed_tools
                if tool_name in self._tool_handlers
            ]

        for tool_name, handler in tool_items:
            # Use explicit Pydantic schema for submit_execution_plan
            if tool_name == "submit_execution_plan":
                tools.append(build_pydantic_tool_schema(
                    model=ExecutionPlan,
                    name="submit_execution_plan",
                    description=(
                        "Submit the final execution plan as a structured JSON object. "
                        "This plan must include all file operations (CREATE/MODIFY/DELETE), "
                        "task summary, project context, quality checklist, and estimated file count. "
                        "Call this tool ONLY when investigation is complete and you have gathered "
                        "all necessary context about the codebase.\n\n"
                        "CRITICAL - MODIFY operations require:\n"
                        "1. old_str: The exact text to replace (for validation)\n"
                        "2. new_str: The replacement text (for validation)\n"
                        "3. content: The COMPLETE file content AFTER the modification is applied\n\n"
                        "The 'content' field is mandatory and must contain the entire post-modification file, "
                        "not just the changed section. Read the full file, mentally apply your changes, "
                        "and include the complete result in 'content'."
                    ),
                    additional_required=["operations", "quality_checklist"],
                ))
            else:
                tools.append(build_anthropic_tool_schema(handler, name=tool_name))
        return tools

    def _dispatch_tool_call(self, tool_name: str, tool_input: Mapping[str, Any]) -> str:
        """Execute a tool handler with full logging, auditing, and compression."""
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
        serialized_result: str | None = None
        try:
            if not handler:
                raise ValueError(f"Tool '{tool_name}' is not registered.")
            result_obj = handler(**tool_input)
            success = True
            serialized_result = self._serialize_tool_result(result_obj)
        except ValidationError as exc:
            LOGGER.warning("Tool %s validation failed: %s", tool_name, exc)
            result_obj = {"error": "Validation failed", "details": exc.errors()}
            serialized_result = self._serialize_tool_result(result_obj)
            self._record_tool_failure(tool_name, params, serialized_result, started)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Tool %s execution failed", tool_name)
            result_obj = {"error": str(exc)}
            serialized_result = self._serialize_tool_result(result_obj)
            self._record_tool_failure(tool_name, params, serialized_result, started)
        finally:
            serialized_payload = serialized_result or self._serialize_tool_result(result_obj)
            duration = time.perf_counter() - started
            ToolCallLog.record(
                conversation_id=self._active_conversation_id,
                agent_role=_ANALYST_SOURCE,
                tool_name=tool_name,
                tool_input=json.dumps(tool_input, ensure_ascii=False, default=str),
                tool_output=serialized_payload,
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
        return serialized_payload

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

    def _render_plan_validation_error(self, exc: ValidationError) -> str:
        """Return a compact description of every ExecutionPlan schema failure."""

        def format_loc(loc: Sequence[Any] | tuple[Any, ...]) -> str:
            if not loc:
                return "payload"
            parts: list[str] = []
            for token in loc:
                if isinstance(token, int):
                    if parts:
                        parts[-1] = f"{parts[-1]}[{token}]"
                    else:
                        parts.append(f"[{token}]")
                    continue
                parts.append(str(token))
            return ".".join(parts)

        def format_reason(error: dict[str, Any]) -> str:
            message = str(error.get("msg", "Invalid value")).strip() or "Invalid value"
            error_type = error.get("type", "")
            received = error.get("input")

            if error_type == "missing" or message.lower() == "field required":
                return "field required"

            if error_type.startswith("type_error."):
                expected = error_type.split(".", 1)[1]
                received_label = "None" if received is None else type(received).__name__
                return f"expected {expected}, got {received_label}"

            if error_type.endswith("_type"):
                expected = error_type.removesuffix("_type")
                received_label = "None" if received is None else type(received).__name__
                return f"expected {expected}, got {received_label}"

            return message.rstrip(".")

        issues: list[str] = []
        for error in exc.errors():
            location = format_loc(error.get("loc") or ())
            reason = format_reason(error)
            issues.append(f"[{location}] {reason}")

        if not issues:
            issues.append("[payload] Invalid ExecutionPlan payload")

        return "Validation errors: " + ", ".join(issues)

    def _handle_submit_execution_plan(self, **payload: Any) -> dict[str, Any] | str:
        """Validate and persist the submitted execution plan.

        Handles both direct JSON objects and payload-wrapped inputs from Claude.
        """
        unwrapped = unwrap_tool_input(payload)
        # Check for empty operations array before Pydantic validation
        operations = unwrapped.get("operations")
        if operations is not None and isinstance(operations, list) and len(operations) == 0:
            error_msg = (
                "ExecutionPlan must include at least one file operation (CREATE/MODIFY/DELETE). "
                "You provided task_summary and project_context but forgot to include the operations array with actual file changes. "
                "Review what files need to be modified and create appropriate operations."
            )
            LOGGER.error("Empty operations array in execution plan submission")
            return {
                "success": False,
                "error": error_msg,
                "hint": "Add operations array with at least one CREATE, MODIFY, or DELETE operation"
            }

        LOGGER.debug("Received execution plan input: %s", json.dumps(unwrapped, default=str)[:200])

        try:
            plan = ExecutionPlan.model_validate(unwrapped)
        except ValidationError as exc:
            try:
                payload_json = json.dumps(unwrapped, default=str)
            except TypeError:
                payload_json = str(unwrapped)
            payload_preview = payload_json[:1000]
            LOGGER.error(
                "Execution plan validation failed: %s | payload preview (first 1000 chars): %s",
                exc,
                payload_preview,
            )
            return {
                "success": False,
                "error": "ExecutionPlan validation failed",
                "details": self._render_plan_validation_error(exc)
            }

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
        self._emit_status(
            f"Plan ready: {len(plan.operations)} operations", "analyst.plan_ready"
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

    def _collect_tool_results(self, response_content: Sequence[Any]) -> list[dict[str, Any]]:
        """Build tool_result payloads for Anthropic follow-up messages."""
        tool_results: list[dict[str, Any]] = []
        for block in response_content:
            if getattr(block, "type", None) != "tool_use":
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
        return tool_results

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
