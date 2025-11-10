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
from aura.prompts import ANALYST_PROMPT, ANALYST_PLANNING_PROMPT
from aura.tools import godot_tools
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_ANALYST_SOURCE = "analyst"

# History filtering constants
MAX_HISTORY_MESSAGES = 6  # Last 3 user/assistant pairs
MAX_HISTORY_CHARS = 12000  # Character limit for history
INVESTIGATION_MAX_TOKENS = 8192
PLANNING_MAX_TOKENS = 5000

ToolHandler = Callable[..., Any]


def count_message_chars(msg: dict[str, Any]) -> int:
    """Count characters in a message's content.

    Args:
        msg: Message dictionary with 'content' key

    Returns:
        Total character count of text content
    """
    total = 0
    content = msg.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                total += len(block.get("text", ""))
    elif isinstance(content, str):
        total += len(content)
    return total


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
    max_chars: int = MAX_HISTORY_CHARS,
) -> list[dict[str, Any]]:
    """Filter conversation history with sliding window and character limits.

    Args:
        history: List of message dictionaries with 'role' and 'content' keys
        max_messages: Maximum number of messages to keep (must be even to preserve pairs)
        max_chars: Maximum total character count for all message content

    Returns:
        Filtered list of messages that fits within constraints

    Strategy:
        1. Filter out tool_result messages - Analyst only needs user/assistant exchanges
        2. Keep only last N messages (sliding window)
        3. If exceeds character limit, remove oldest messages while keeping pairs intact
    """
    # Step 1: Filter out tool_result messages
    filtered = [msg for msg in history if msg.get("role") != "tool_result"]

    # Step 2: Apply sliding window - keep last max_messages
    if len(filtered) > max_messages:
        filtered = filtered[-max_messages:]

    # Step 3: Apply character limit, removing oldest pairs
    while filtered:
        total_chars = sum(count_message_chars(msg) for msg in filtered)
        if total_chars <= max_chars:
            break

        # Remove oldest pair (user + assistant) to keep context coherent
        # Remove at least 2 messages if possible to preserve pairs
        if len(filtered) >= 2:
            filtered = filtered[2:]
        elif filtered:
            # Last resort: remove single message if only one remains
            filtered = filtered[1:]

    return filtered


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
            "verify_asset_paths": self.tool_manager.verify_asset_paths,
            "list_project_assets": self.tool_manager.list_project_assets,
            "search_assets_by_pattern": self.tool_manager.search_assets_by_pattern,
            "get_asset_metadata": self.tool_manager.get_asset_metadata,
            "respect_gitignore": self.tool_manager.respect_gitignore,
            "submit_execution_plan": self._handle_submit_execution_plan,
            "read_godot_scene_tree": godot_tools.read_godot_scene_tree,
            "get_project_godot_config": godot_tools.get_project_godot_config,
        }

    def _has_execution_plan_submission(self, response_content: Sequence[Any]) -> bool:
        """Check if the response contains a submit_execution_plan tool call."""
        for block in response_content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "submit_execution_plan":
                return True
        return False

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

        # Apply sliding window history filter
        filtered_history: list[dict[str, Any]] = []
        history_chars = 0
        if conversation_history:
            filtered_history = filter_conversation_history(
                conversation_history,
                max_messages=MAX_HISTORY_MESSAGES,
                max_chars=MAX_HISTORY_CHARS,
            )
            history_chars = sum(count_message_chars(msg) for msg in filtered_history)
            LOGGER.info(
                "Analyst using conversation history | messages=%d | total_chars=%d | original_messages=%d",
                len(filtered_history),
                history_chars,
                len(conversation_history),
            )

        # Build messages list with history + current request
        investigation_messages = list(filtered_history)
        investigation_messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_request}],
        })

        investigation_tool_names = [
            name for name in self._tool_handlers.keys() if name != "submit_execution_plan"
        ]
        investigation_tools = self._build_tool_definitions(allowed_tools=investigation_tool_names)
        planning_tools = self._build_tool_definitions(allowed_tools=["submit_execution_plan"])

        LOGGER.info(
            "Analyst analysis started | model=%s | prompt_chars=%d | history_chars=%d",
            self.model_name,
            len(user_request or ""),
            history_chars,
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
            max_investigation_tool_calls = 15
            max_plan_tool_calls = 5
            max_narrative_retries = 2
            investigation_tool_calls = 0
            plan_tool_calls_count = 0
            narrative_retry_count = 0
            final_response_text = ""
            investigation_summary = ""

            # Phase 1: Investigation loop
            while True:
                if investigation_tool_calls >= max_investigation_tool_calls:
                    error_message = (
                        "Error: Analyst exceeded the maximum number of tool calls during investigation."
                    )
                    self._emit_status("Analyst agent: failed", "analyst.error")
                    self._event_bus.emit(
                        SystemErrorEvent(
                            error="analyst.investigation_tool_limit",
                            details={"max_tool_calls": max_investigation_tool_calls},
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

                response = self._client.messages.create(
                    model=self.model_name,
                    system=ANALYST_PROMPT,
                    temperature=0,
                    max_tokens=INVESTIGATION_MAX_TOKENS,
                    tools=investigation_tools,
                    messages=investigation_messages,
                )
                investigation_messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    investigation_tool_calls += 1
                    tool_results = self._collect_tool_results(response.content)
                    if tool_results:
                        investigation_messages.append({"role": "user", "content": tool_results})
                    continue

                investigation_summary = (self._collect_text(response.content) or "").strip()
                if not investigation_summary:
                    error_message = "Error: Analyst investigation did not return a structured summary."
                    self._emit_status("Analyst agent: failed", "analyst.error")
                    self._event_bus.emit(
                        SystemErrorEvent(
                            error="analyst.investigation_no_summary",
                            details={"tool_calls": investigation_tool_calls},
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

                self._emit_streaming_chunk(
                    investigation_summary,
                    source=_ANALYST_SOURCE,
                    on_chunk=on_chunk,
                    is_final=False,
                )
                investigation_duration = time.perf_counter() - started
                LOGGER.info(
                    "Analyst investigation summary ready | duration=%.2fs | tool_calls=%d",
                    investigation_duration,
                    investigation_tool_calls,
                )
                self._emit_status("Analyst agent: investigation summary ready", "analyst.investigation_complete")
                self._event_bus.emit(
                    TaskProgressEvent(
                        message="Analyst investigation summary ready",
                        percent=0.25,
                        source=_ANALYST_SOURCE,
                    )
                )
                break

            # Phase 2: Planning loop with fresh prompt
            planning_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_planning_user_message(
                                user_request=user_request,
                                investigation_summary=investigation_summary,
                            ),
                        }
                    ],
                }
            ]

            self._emit_status("Analyst agent: synthesizing execution plan...", "analyst.plan_start")
            self._event_bus.emit(
                TaskProgressEvent(
                    message="Analyst synthesizing execution plan",
                    percent=0.35,
                    source=_ANALYST_SOURCE,
                )
            )

            while True:
                if plan_tool_calls_count >= max_plan_tool_calls:
                    error_message = (
                        "Error: Analyst exceeded the maximum number of plan-generation tool calls."
                    )
                    self._emit_status("Analyst agent: failed", "analyst.error")
                    self._event_bus.emit(
                        SystemErrorEvent(
                            error="analyst.plan_tool_limit",
                            details={"max_tool_calls": max_plan_tool_calls},
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

                response = self._client.messages.create(
                    model=self.model_name,
                    system=ANALYST_PLANNING_PROMPT,
                    temperature=0,
                    max_tokens=PLANNING_MAX_TOKENS,
                    tools=planning_tools,
                    tool_choice={"type": "tool", "name": "submit_execution_plan"},
                    messages=planning_messages,
                )
                planning_messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    plan_tool_calls_count += 1
                    tool_results = self._collect_tool_results(response.content)
                    if tool_results:
                        planning_messages.append({"role": "user", "content": tool_results})
                    continue

                final_text = self._collect_text(response.content)
                if final_text:
                    final_response_text = final_text

                    has_plan_submission = self._has_execution_plan_submission(response.content)

                    if (
                        not has_plan_submission
                        and not self._latest_plan
                        and narrative_retry_count < max_narrative_retries
                    ):
                        narrative_retry_count += 1
                        LOGGER.warning(
                            "Analyst provided narrative without ExecutionPlan submission (attempt %d/%d). Retrying with enforcement prompt.",
                            narrative_retry_count,
                            max_narrative_retries,
                        )
                        self._emit_status(
                            f"Analyst agent: enforcing ExecutionPlan submission (retry {narrative_retry_count}/{max_narrative_retries})",
                            "analyst.retry",
                        )

                        enforcement_prompt = (
                            "Investigation is complete. Immediately generate the full ExecutionPlan JSON and call submit_execution_plan. "
                            "Do not provide narrative text—respond only by calling submit_execution_plan with the finalized plan."
                        )

                        planning_messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": enforcement_prompt}]
                        })
                        continue

                    self._emit_streaming_chunk(
                        final_text,
                        source=_ANALYST_SOURCE,
                        on_chunk=on_chunk,
                        is_final=True,
                    )
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

            if final_response_text and narrative_retry_count >= max_narrative_retries:
                duration = time.perf_counter() - started
                LOGGER.error(
                    "Analyst failed to submit ExecutionPlan after %d retries | duration=%.2fs",
                    narrative_retry_count,
                    duration,
                )
                summary = self._safe_value(final_response_text, limit=200)
                error_message = (
                    f"Error: Analyst provided narrative text instead of ExecutionPlan after {narrative_retry_count} retry attempts. "
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
                        details={"narrative_preview": summary, "retry_count": narrative_retry_count},
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
                        "all necessary context about the codebase."
                    ),
                    additional_required=["operations", "quality_checklist"],
                ))
            else:
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

    def _build_planning_user_message(
        self,
        *,
        user_request: str,
        investigation_summary: str,
    ) -> str:
        """Create the concise planning prompt for Phase 2."""
        request_text = (user_request or "").strip() or "(no explicit user request provided)"
        summary_text = (investigation_summary or "").strip()
        return (
            "Investigation is complete. Using only the information below, generate the ExecutionPlan and "
            "submit it via the submit_execution_plan tool.\n\n"
            f"User request:\n{request_text}\n\n"
            "Investigation summary (JSON):\n"
            f"{summary_text}"
        )

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
