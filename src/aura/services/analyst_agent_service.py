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
MAX_HISTORY_CHARS = 12000  # Character limit for history
INVESTIGATION_MAX_TOKENS = 8192
PLANNING_MAX_TOKENS = 5000

# Tool categorization for emergency plan construction
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
    "content": "CREATE/MODIFY operations must include the complete post-change file content.",
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
            # Godot read-only inspection tools
            "read_godot_scene": godot_tools.read_godot_scene,
            "read_godot_scene_tree": godot_tools.read_godot_scene_tree,
            "validate_godot_scene": godot_tools.validate_godot_scene,
            "get_project_godot_config": godot_tools.get_project_godot_config,
            # NOTE: add_godot_node and modify_godot_node_property are NOT included here
            # because they are investigation tools, not file operations. The Analyst must
            # use MODIFY operations with full .tscn file content instead.
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
            max_tool_calls = 30
            tool_calls = 0
            final_response_text = ""
            enforcement_retries = 0
            max_enforcement_retries = 1

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

                response = self._client.messages.create(
                    model=self.investigation_model,
                    system=cached_system,
                    temperature=0,
                    max_tokens=INVESTIGATION_MAX_TOKENS,
                    tools=cached_tools,
                    messages=investigation_messages,
                )
                investigation_messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    tool_calls += 1
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
                        LOGGER.warning(
                            "Analyst failed to submit plan after %d enforcement attempts. "
                            "Creating emergency fallback plan from investigation context.",
                            enforcement_retries,
                        )
                        self._latest_plan = self._construct_emergency_plan(investigation_messages)
                        break

                    # Force the Analyst to call the tool instead of ending with text
                    enforcement_retries += 1
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

    def _construct_emergency_plan(
        self,
        investigation_messages: list[dict[str, Any]],
    ) -> ExecutionPlan:
        """Build a minimal valid ExecutionPlan from investigation context as emergency fallback.

        This method is called when the analyst fails to submit a proper execution plan after
        enforcement attempts. It extracts file paths from tool calls and creates conservative
        MODIFY operations marked for manual review.

        Args:
            investigation_messages: The full conversation history with tool calls

        Returns:
            ExecutionPlan with conservative operations extracted from tool call history
        """
        from aura.models.execution_plan import FileOperation, OperationType

        # Extract file paths from tool call history
        file_paths: set[str] = set()
        user_request = self._current_user_request or "User request"

        for msg in investigation_messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if getattr(block, "type", None) == "tool_use":
                        tool_name = getattr(block, "name", "")
                        tool_input = dict(getattr(block, "input", {}) or {})

                        # Extract file paths from file reading/searching tools
                        if tool_name in FILE_READING_TOOLS:
                            path = tool_input.get("path") or tool_input.get("file_path")
                            if path and isinstance(path, str):
                                file_paths.add(path)
                            paths = tool_input.get("paths")
                            if isinstance(paths, list):
                                for p in paths:
                                    if isinstance(p, str):
                                        file_paths.add(p)

        # Create conservative MODIFY operations for discovered files
        operations: list[FileOperation] = []
        for file_path in sorted(file_paths)[:10]:  # Limit to first 10 files to prevent bloat
            operations.append(
                FileOperation(
                    operation_type=OperationType.MODIFY,
                    file_path=file_path,
                    content="# EMERGENCY FALLBACK: Content placeholder\n# This operation requires manual review",
                    old_str="placeholder_old",
                    new_str="placeholder_new",
                    rationale=(
                        "Emergency fallback operation created because analyst failed to submit proper plan. "
                        "This file was accessed during investigation and may need modification. "
                        "REQUIRES MANUAL REVIEW AND PROPER CONTENT."
                    ),
                    dependencies=[],
                )
            )

        # If no files were discovered, create a single placeholder operation
        if not operations:
            operations.append(
                FileOperation(
                    operation_type=OperationType.MODIFY,
                    file_path="REVIEW_REQUIRED.txt",
                    content="# Emergency fallback plan created\n# Analyst failed to submit proper execution plan\n# Manual review required",
                    old_str="placeholder",
                    new_str="placeholder",
                    rationale=(
                        "Emergency fallback operation. No file operations could be extracted from investigation. "
                        "Manual review and proper plan creation required."
                    ),
                    dependencies=[],
                )
            )

        # Build the emergency ExecutionPlan
        plan = ExecutionPlan(
            task_summary=f"Emergency fallback plan for: {user_request[:100]}",
            project_context=(
                "This is an emergency fallback plan created because the analyst agent failed to submit "
                "a proper execution plan after enforcement attempts. All operations require manual review "
                "and proper content before execution."
            ),
            operations=operations,
            quality_checklist=[
                "⚠️  EMERGENCY FALLBACK - Manual review required",
                "⚠️  Verify all file operations are correct",
                "⚠️  Replace placeholder content with actual changes",
                "⚠️  Confirm operations match user request",
            ],
            estimated_files=len(operations),
        )

        LOGGER.info(
            "Emergency fallback plan created | operations=%d | files_discovered=%d",
            len(operations),
            len(file_paths),
        )

        return plan


__all__ = ["AnalystAgentService"]
