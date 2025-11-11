"""Analyst agent powered by Claude Sonnet 4.5 that produces execution plans."""

from __future__ import annotations

import json
import logging
import re
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
TOOL_RESULT_COMPRESSION_THRESHOLD = 2000
SHORT_FILE_LINE_LIMIT = 100
MEDIUM_FILE_LINE_LIMIT = 300
FILE_HEAD_TAIL_SLICE = 20
LIST_PREVIEW_THRESHOLD = 30
LIST_PREVIEW_COUNT = 25
ANALYSIS_TOOL_NAMES = {
    "get_cyclomatic_complexity",
    "detect_duplicate_code",
    "check_naming_conventions",
    "analyze_type_hints",
    "inspect_docstrings",
    "get_code_metrics",
    "get_dependency_graph",
    "get_class_hierarchy",
}
STRUCTURE_TOOLS = {"get_project_structure", "list_project_files"}
FILE_READING_TOOLS = {"read_project_file", "read_multiple_files"}
SIGNATURE_TOOLS = {"get_function_signatures"}
DEPENDENCY_TOOLS = {"get_dependency_graph", "get_imports", "verify_asset_paths"}
SUMMARY_SECTION_LIMIT = 8

ToolHandler = Callable[..., Any]

# Common schema mistakes we can point out when validation fails.
_PLAN_FIELD_NAME_HINTS = {
    "summary": "task_summary",
    "context": "project_context",
    "estimated_file_count": "estimated_files",
}
_OPERATION_FIELD_NAME_HINTS = {
    "type": "operation_type",
    "action": "operation_type",
    "operation": "operation_type",
    "path": "file_path",
    "filepath": "file_path",
    "file": "file_path",
    "target": "file_path",
    "old": "old_str",
    "new": "new_str",
    "replacement": "new_str",
}
_FIELD_SUGGESTIONS = {
    "operation_type": "Allowed values: CREATE, MODIFY, DELETE.",
    "file_path": "Use a repo-relative path such as 'src/app.py'.",
    "old_str": "Provide the exact text to replace for MODIFY operations.",
    "new_str": "Provide the new text for MODIFY operations.",
    "content": "CREATE operations must include the full file content.",
    "rationale": "Explain why the change is needed.",
    "operations": "Include at least one file operation.",
    "estimated_files": "Send an integer count of affected files.",
    "task_summary": "Provide a single-sentence summary of the requested change.",
    "project_context": "Describe the repository context and constraints.",
}


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
            "get_asset_metadata": self.tool_manager.get_asset_metadata,
            "respect_gitignore": self.tool_manager.respect_gitignore,
            "submit_execution_plan": self._handle_submit_execution_plan,
            "read_godot_scene": godot_tools.read_godot_scene,
            "add_godot_node": godot_tools.add_godot_node,
            "modify_godot_node_property": godot_tools.modify_godot_node_property,
            "validate_godot_scene": godot_tools.validate_godot_scene,
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
        self._current_user_request = user_request

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
            "Analyst analysis started | investigation_model=%s | planning_model=%s | prompt_chars=%d | history_chars=%d",
            self.investigation_model,
            self.planning_model,
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
            max_investigation_tool_calls = 15
            max_plan_tool_calls = 5
            max_narrative_retries = 2
            investigation_tool_calls = 0
            plan_tool_calls_count = 0
            narrative_retry_count = 0
            final_response_text = ""
            investigation_summary = ""
            condensed_investigation_summary = ""

            LOGGER.info("Analyst Phase 1 (investigation) using model=%s", self.investigation_model)
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
                    model=self.investigation_model,
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
                try:
                    condensed_investigation_summary = self._synthesize_investigation_summary(investigation_messages)
                    LOGGER.info(
                        "Investigation summary synthesized | original_chars=%d | condensed_chars=%d",
                        len(investigation_summary),
                        len(condensed_investigation_summary),
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Failed to synthesize investigation summary; falling back to raw narrative")
                    condensed_investigation_summary = investigation_summary
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

            LOGGER.info("Analyst Phase 2 (planning) using model=%s", self.planning_model)
            # Phase 2: Planning loop with fresh prompt
            planning_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self._build_planning_user_message(
                                user_request=user_request,
                                investigation_summary=condensed_investigation_summary or investigation_summary,
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
                    model=self.planning_model,
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

                    # Check if plan was successfully validated - if so, exit loop immediately
                    if self._latest_plan is not None:
                        LOGGER.info(
                            "Analyst plan validated successfully on first attempt | operations=%d",
                            len(self._latest_plan.operations),
                        )
                        break

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
        compressed_payload = ""
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
            compressed_payload = self._compress_tool_result(tool_name, serialized_payload)
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
        return compressed_payload or serialized_payload

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

    def _render_plan_validation_error(
        self,
        raw_payload: Mapping[str, Any],
        exc: ValidationError,
    ) -> str:
        """Turn a ValidationError into actionable guidance for Claude."""

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
                else:
                    parts.append(str(token))
            return ".".join(parts)

        payload_mapping: Mapping[str, Any] = raw_payload if isinstance(raw_payload, Mapping) else {}
        issues: list[str] = []

        for error in exc.errors():
            loc = error.get("loc") or ()
            location = format_loc(loc)
            message = error.get("msg", "Invalid value")
            field_name = next((part for part in reversed(loc) if isinstance(part, str)), None)
            suggestion_bits: list[str] = []
            if field_name:
                if error.get("type") == "missing":
                    suggestion_bits.append(f"Include '{field_name}' exactly as spelled.")
                hint = _FIELD_SUGGESTIONS.get(field_name)
                if hint:
                    suggestion_bits.append(hint)
            elif location == "operations":
                suggestion_bits.append("Provide at least one operation in the list.")
            suggestion = " ".join(suggestion_bits) if suggestion_bits else ""
            issue = f"{location}: {message}"
            if suggestion:
                issue = f"{issue} — {suggestion}"
            issues.append(issue)

        if not issues:
            issues.append("payload: Unable to parse the schema error. Please compare your payload with the expected ExecutionPlan model.")

        hint_lines = self._collect_field_name_hints(payload_mapping)
        example_operation = {
            "operation_type": "MODIFY",
            "file_path": "src/example.py",
            "old_str": "prior code snippet",
            "new_str": "updated code snippet",
            "rationale": "Explain why this change is necessary.",
            "dependencies": [],
        }
        example_json = json.dumps(example_operation, indent=2)
        example_block = "\n".join(f"  {line}" for line in example_json.splitlines())

        lines = [
            "ExecutionPlan validation failed. Update your submit_execution_plan payload before retrying.",
            "",
            "Issues detected:",
            *[f"- {issue}" for issue in issues],
            "",
            "Field-name reminders:",
            *[f"- {hint}" for hint in hint_lines],
            "",
            "Example operation payload:",
            example_block,
        ]
        return "\n".join(lines)

    def _collect_field_name_hints(self, payload: Mapping[str, Any] | None) -> list[str]:
        """Surface concrete field-name corrections plus common reminders."""

        def add_hint(context: str, wrong: str, correct: str) -> None:
            hints.append(f"{context}: you used '{wrong}' but must use '{correct}'.")

        hints: list[str] = []
        payload = payload or {}

        if isinstance(payload, Mapping):
            for wrong, correct in _PLAN_FIELD_NAME_HINTS.items():
                if wrong in payload:
                    add_hint("payload", wrong, correct)

            operations = payload.get("operations")
            if isinstance(operations, Sequence) and not isinstance(operations, (str, bytes, bytearray)):
                for idx, operation in enumerate(operations):
                    if not isinstance(operation, Mapping):
                        continue
                    for wrong, correct in _OPERATION_FIELD_NAME_HINTS.items():
                        if wrong in operation:
                            add_hint(f"operations[{idx}]", wrong, correct)

        default_hints = [
            "Use 'operation_type' (CREATE/MODIFY/DELETE) instead of 'type' or 'action'.",
            "Provide 'file_path' for every operation (repo-relative, no leading './').",
            "CREATE operations must include 'content'. MODIFY operations must include both 'old_str' and 'new_str'.",
            "Always send 'estimated_files' as an integer count.",
        ]

        if hints:
            for note in default_hints:
                if note not in hints:
                    hints.append(note)
        else:
            hints = default_hints

        return hints

    def _handle_submit_execution_plan(self, **payload: Any) -> dict[str, Any] | str:
        """Validate and persist the submitted execution plan.

        Handles both direct JSON objects and payload-wrapped inputs from Claude.
        """
        unwrapped = unwrap_tool_input(payload)

        LOGGER.debug("Received execution plan input: %s", json.dumps(unwrapped, default=str)[:200])

        try:
            plan = ExecutionPlan.model_validate(unwrapped)
        except ValidationError as exc:
            LOGGER.warning("Execution plan validation failed: %s", exc)
            return self._render_plan_validation_error(unwrapped, exc)

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

    def _compress_tool_result(self, tool_name: str, result: str) -> str:
        """Apply heuristic compression rules to large tool outputs."""
        if not result:
            return result
        original_length = len(result)
        if original_length <= TOOL_RESULT_COMPRESSION_THRESHOLD:
            return result

        compressed_text = result
        parsed: Any = None
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            parsed = None

        if isinstance(parsed, dict):
            compressed_payload = self._compress_mapping_payload(tool_name, parsed)
            compressed_text = json.dumps(compressed_payload, ensure_ascii=False)
        elif isinstance(parsed, list):
            compressed_list = self._compress_list_payload(parsed)
            compressed_text = json.dumps(compressed_list, ensure_ascii=False)
        else:
            compressed_text = self._compress_text_block(result)

        if not compressed_text or len(compressed_text) >= original_length:
            return result

        savings = 1 - (len(compressed_text) / original_length)
        LOGGER.info(
            "Compressed %s tool payload | original=%d chars | compressed=%d chars | savings=%.1f%%",
            tool_name,
            original_length,
            len(compressed_text),
            savings * 100,
        )
        return compressed_text

    def _compress_mapping_payload(self, tool_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Recursively compress mapping structures."""
        if tool_name in ANALYSIS_TOOL_NAMES:
            return self._summarize_analysis_payload(payload)

        compact: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                compact[key] = (
                    self._compress_text_block(value) if len(value) > TOOL_RESULT_COMPRESSION_THRESHOLD else value
                )
                continue
            if isinstance(value, list):
                compact[key] = self._compress_list_payload(value)
                continue
            if isinstance(value, Mapping):
                compact[key] = self._compress_mapping_payload(tool_name, value)
                continue
            compact[key] = value
        return compact

    def _summarize_analysis_payload(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Keep only summary statistics for heavy analysis responses."""
        summary_keys = [
            key for key in payload.keys() if any(token in str(key).lower() for token in ("summary", "total", "aggregate", "stats"))
        ]
        condensed: dict[str, Any] = {}

        if summary_keys:
            for key in summary_keys:
                condensed[key] = payload[key]

        for key, value in payload.items():
            if key in condensed:
                continue
            if isinstance(value, (bool, int, float)) or value is None:
                condensed[key] = value
                continue
            if isinstance(value, str):
                condensed[key] = value if len(value) <= 400 else f"{value[:400]}... (+{len(value) - 400} chars)"
                continue
            if isinstance(value, Mapping):
                condensed[key] = {
                    inner_key: inner_val
                    for inner_key, inner_val in value.items()
                    if isinstance(inner_val, (bool, int, float, str)) or inner_val is None
                }
                continue
            if isinstance(value, list):
                condensed[key] = {
                    "total_items": len(value),
                    "sample": value[:3],
                }

        omitted = sorted(set(payload.keys()) - set(condensed.keys()))
        if omitted:
            condensed["_detail_fields_omitted"] = omitted
        return condensed

    def _compress_list_payload(self, items: Sequence[Any]) -> Sequence[Any]:
        """Keep short lists intact and summarize long ones."""
        length = len(items)
        if length <= LIST_PREVIEW_THRESHOLD:
            return items
        preview = list(items[:LIST_PREVIEW_COUNT])
        preview.append(
            {
                "_summary": True,
                "total_items": length,
                "omitted_count": max(length - LIST_PREVIEW_COUNT, 0),
            }
        )
        return preview

    def _compress_text_block(self, text: str) -> str:
        """Build textual previews for long file blobs."""
        lines = text.splitlines()
        total_lines = len(lines)
        if total_lines <= SHORT_FILE_LINE_LIMIT:
            return text

        imports = [line for line in lines if line.strip().startswith(("import ", "from "))]
        signatures = [
            line.strip()
            for line in lines
            if line.strip().startswith(("def ", "class ", "async def "))
        ]

        header = f"[compressed file preview | original_lines={total_lines}]"

        if total_lines <= MEDIUM_FILE_LINE_LIMIT:
            body_sections = [header]
            if imports:
                body_sections.append("Imports:\n" + "\n".join(imports[:20]))
            if signatures:
                body_sections.append("Signatures:\n" + "\n".join(signatures[:40]))
            head = "\n".join(lines[:FILE_HEAD_TAIL_SLICE])
            tail = "\n".join(lines[-FILE_HEAD_TAIL_SLICE:])
            body_sections.append("First lines:\n" + head)
            body_sections.append("Last lines:\n" + tail)
            body_sections.append(
                f"... {max(total_lines - FILE_HEAD_TAIL_SLICE * 2, 0)} middle lines omitted ..."
            )
            preview = "\n\n".join(section for section in body_sections if section.strip())
            return preview if preview else text

        docstrings = self._extract_docstrings(lines)
        body_sections = [header]
        if imports:
            body_sections.append("Imports:\n" + "\n".join(imports[:20]))
        if signatures:
            body_sections.append("Signatures:\n" + "\n".join(signatures[:40]))
        if docstrings:
            doc_excerpt = "\n---\n".join(docstrings[:5])
            body_sections.append("Docstrings:\n" + doc_excerpt)
        body_sections.append(f"(structure only – {total_lines} lines summarized)")
        preview = "\n\n".join(section for section in body_sections if section.strip())
        return preview if preview else text

    def _extract_docstrings(self, lines: Sequence[str]) -> list[str]:
        """Return up to five docstring excerpts from a file."""
        docstrings: list[str] = []
        capturing = False
        delimiter = ""
        buffer: list[str] = []
        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                continue
            if not capturing and stripped.startswith(('"""', "'''")):
                delimiter = stripped[:3]
                capturing = True
                buffer = [stripped]
                if stripped.count(delimiter) >= 2 and len(stripped) > 3:
                    docstrings.append(stripped)
                    capturing = False
                    buffer = []
                    delimiter = ""
                continue
            if capturing:
                buffer.append(stripped)
                if stripped.endswith(delimiter):
                    excerpt = "\n".join(buffer)
                    docstrings.append(excerpt if len(excerpt) <= 400 else f"{excerpt[:400]}...")
                    capturing = False
                    buffer = []
                    delimiter = ""
            if len(docstrings) >= 5:
                break
        return docstrings

    def _synthesize_investigation_summary(self, messages: Sequence[dict[str, Any]]) -> str:
        """Create a condensed JSON summary of Phase 1 findings."""
        user_request = (self._current_user_request or "").strip()
        latest_summary_text = self._extract_latest_assistant_summary(messages)
        sentences = self._split_sentences(latest_summary_text)
        tool_trace = self._collect_tool_trace(messages)

        project_structure = self._summarize_project_structure(tool_trace, sentences)
        relevant_files = self._summarize_relevant_files(tool_trace)
        code_patterns = self._select_sentences(
            sentences,
            keywords=("pattern", "convention", "style", "architecture", "naming", "idiom"),
        )
        dependencies = self._summarize_dependencies(tool_trace, sentences)
        constraints = self._select_sentences(
            sentences,
            keywords=("constraint", "must", "should", "avoid", "limitation", "restriction", "compatibility"),
        )
        key_signatures = self._gather_signatures(tool_trace, latest_summary_text)

        summary_payload = {
            "user_request": user_request,
            "project_structure": project_structure,
            "relevant_files": relevant_files,
            "code_patterns": code_patterns,
            "dependencies": dependencies,
            "constraints": constraints,
            "key_signatures": key_signatures,
        }
        return json.dumps(summary_payload, ensure_ascii=False, indent=2)

    def _extract_latest_assistant_summary(self, messages: Sequence[dict[str, Any]]) -> str:
        """Return the most recent assistant text block."""
        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            text = self._collect_text(content) if isinstance(content, list) else str(content or "")
            if text.strip():
                return text.strip()
        return ""

    def _split_sentences(self, text: str) -> list[str]:
        """Split summary text into normalized sentences."""
        if not text:
            return []
        sentences: list[str] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = re.split(r"(?<=[.!?])\s+", stripped)
            for part in parts:
                cleaned = part.strip("•*- \t")
                if cleaned:
                    sentences.append(cleaned)
        return sentences

    def _collect_tool_trace(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pair tool_use blocks with tool_results for downstream summarization."""
        tool_requests: dict[str, dict[str, Any]] = {}
        trace: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_requests[block.get("id")] = {
                        "tool": block.get("name"),
                        "input": dict(block.get("input") or {}),
                    }
                elif block_type == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    request = tool_requests.get(tool_use_id, {})
                    block_content = block.get("content")
                    if isinstance(block_content, list):
                        text_value = self._collect_text(block_content)
                    else:
                        text_value = block_content or ""
                    trace.append(
                        {
                            "tool": request.get("tool"),
                            "input": request.get("input") or {},
                            "output": text_value if isinstance(text_value, str) else str(text_value),
                        }
                    )
        return trace

    def _select_sentences(
        self,
        sentences: Sequence[str],
        *,
        keywords: Sequence[str],
        limit: int = SUMMARY_SECTION_LIMIT,
    ) -> list[str]:
        """Return up to `limit` sentences containing any of the keywords."""
        selected: list[str] = []
        lowered_keywords = [token.lower() for token in keywords]
        for sentence in sentences:
            lowered = sentence.lower()
            if any(token in lowered for token in lowered_keywords):
                selected.append(sentence)
            if len(selected) >= limit:
                break
        return selected

    def _summarize_project_structure(
        self,
        tool_trace: Sequence[Mapping[str, Any]],
        sentences: Sequence[str],
    ) -> list[str]:
        """Combine tool outputs and textual hints into a structure summary."""
        notes: list[str] = []
        for entry in tool_trace:
            tool = entry.get("tool")
            if tool == "get_project_structure":
                data = self._try_parse_json(entry.get("output"))
                if isinstance(data, Mapping):
                    directories = data.get("directories") or []
                    files = data.get("files") or []
                    root = data.get("root") or entry.get("input", {}).get("directory") or "."
                    dir_preview = ", ".join(directories[:SUMMARY_SECTION_LIMIT]) or "no subdirectories"
                    notes.append(
                        f"{root}: {len(directories)} dirs / {len(files)} files | sample dirs: {dir_preview}"
                    )
            elif tool == "list_project_files":
                data = self._try_parse_json(entry.get("output"))
                if isinstance(data, Mapping):
                    ext = data.get("extension") or entry.get("input", {}).get("extension") or "*"
                    count = data.get("count")
                    files = data.get("files") or []
                    sample = ", ".join(files[:3]) if files else "no matches"
                    notes.append(f"{count} files matching {ext} (sample: {sample})")
            if len(notes) >= SUMMARY_SECTION_LIMIT:
                break

        if notes:
            return notes[:SUMMARY_SECTION_LIMIT]
        return self._select_sentences(
            sentences,
            keywords=("structure", "directory", "folder", "module", "layout"),
        )

    def _summarize_relevant_files(self, tool_trace: Sequence[Mapping[str, Any]]) -> list[str]:
        """Extract per-file insights from tool outputs."""
        summaries: list[str] = []
        seen: set[str] = set()
        for entry in tool_trace:
            tool = entry.get("tool")
            if tool not in FILE_READING_TOOLS:
                continue
            raw_input = entry.get("input") or {}
            path = (
                raw_input.get("path")
                or raw_input.get("file")
                or raw_input.get("file_path")
                or raw_input.get("target")
            )
            if not path or path in seen:
                continue
            seen.add(path)
            description = self._describe_file_content(entry.get("output") or "")
            summaries.append(f"{path}: {description}")
            if len(summaries) >= SUMMARY_SECTION_LIMIT:
                break
        return summaries

    def _describe_file_content(self, text: str) -> str:
        """Infer the purpose of a file from truncated content."""
        if not text:
            return "empty or unavailable"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("#"):
                return line.lstrip("# ").strip()[:200]
            if line.startswith(('"""', "'''")):
                stripped = line.strip("\"' ")
                if stripped:
                    return stripped[:200]
        for line in lines:
            if line.startswith(("class ", "def ", "async def ")):
                return line.split(":")[0][:200]
        return f"{len(lines)} significant lines"

    def _summarize_dependencies(
        self,
        tool_trace: Sequence[Mapping[str, Any]],
        sentences: Sequence[str],
    ) -> list[str]:
        """Summarize dependency signals from tools or narrative sentences."""
        deps: list[str] = []
        for entry in tool_trace:
            tool = entry.get("tool")
            if tool not in DEPENDENCY_TOOLS:
                continue
            data = self._try_parse_json(entry.get("output"))
            if not isinstance(data, Mapping):
                continue
            collected = self._collect_dependency_values(data)
            if collected:
                deps.append(f"{tool}: {', '.join(collected[:SUMMARY_SECTION_LIMIT])}")
        if deps:
            return deps[:SUMMARY_SECTION_LIMIT]
        return self._select_sentences(
            sentences,
            keywords=("dependency", "import", "library", "package", "requirements"),
        )

    def _collect_dependency_values(self, payload: Mapping[str, Any]) -> list[str]:
        """Flatten dependency-like fields into a list."""
        collected: list[str] = []
        for key, value in payload.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("dependency", "import", "library", "package", "module", "existing", "missing")):
                collected.extend(self._flatten_dependency_value(value))
        return self._dedupe_entries(collected)

    def _flatten_dependency_value(self, value: Any) -> list[str]:
        """Flatten mixed dependency values into string tokens."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, (int, float)):
            return [str(value)]
        if isinstance(value, Mapping):
            names: list[str] = []
            for key in ("name", "module", "package", "import", "path"):
                candidate = value.get(key)
                if isinstance(candidate, str):
                    names.append(candidate)
            return names
        if isinstance(value, list):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_dependency_value(item))
            return flattened
        return []

    def _gather_signatures(
        self,
        tool_trace: Sequence[Mapping[str, Any]],
        narrative: str,
    ) -> list[str]:
        """Collect key function/class signatures without bodies."""
        signatures: list[str] = []
        seen: set[str] = set()

        for entry in tool_trace:
            tool = entry.get("tool")
            if tool == "get_function_signatures":
                data = self._try_parse_json(entry.get("output"))
                if isinstance(data, Mapping):
                    for fn in data.get("functions", []):
                        if not isinstance(fn, Mapping):
                            continue
                        name = fn.get("name")
                        params = fn.get("params") or []
                        location = entry.get("input", {}).get("file_path") or entry.get("input", {}).get("path")
                        signature = f"{name}({', '.join(params)})"
                        if location:
                            signature = f"{signature} [{location}]"
                        if name and signature not in seen:
                            seen.add(signature)
                            signatures.append(signature)
            elif tool in FILE_READING_TOOLS:
                for line in (entry.get("output") or "").splitlines():
                    stripped = line.strip()
                    if stripped.startswith(("def ", "class ", "async def ")):
                        signature = stripped.split(":")[0]
                        if signature and signature not in seen:
                            seen.add(signature)
                            signatures.append(signature)
            if len(signatures) >= SUMMARY_SECTION_LIMIT:
                break

        if len(signatures) < SUMMARY_SECTION_LIMIT and narrative:
            for line in narrative.splitlines():
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "async def ")) and stripped not in seen:
                    seen.add(stripped)
                    signatures.append(stripped.split(":")[0])
                if len(signatures) >= SUMMARY_SECTION_LIMIT:
                    break
        return signatures

    def _dedupe_entries(self, items: Sequence[str]) -> list[str]:
        """Preserve order while removing duplicates/empty entries."""
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _try_parse_json(self, text: Any) -> Any:
        """Attempt to parse JSON; return None on failure."""
        if not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return None

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
