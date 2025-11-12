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
import instructor
from openai import OpenAI
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
PLANNING_MAX_TOKENS = 8192

# Planning agent system prompt
PLANNING_SYSTEM_PROMPT = """You are a planning specialist. Your ONLY job is to convert investigation findings into a valid ExecutionPlan JSON.

You must call submit_execution_plan with ALL required fields:
- task_summary: Brief description of what will be done
- project_context: Relevant context from the investigation
- quality_checklist: List of quality criteria to verify
- estimated_files: Number of files that will be modified
- operations: Array of file operations (CREATE/MODIFY/DELETE)

The operations array is MANDATORY and must contain at least one CREATE, MODIFY, or DELETE operation with all required fields:
- For CREATE: path, content, rationale
- For MODIFY: path, content, old_str, new_str, rationale
- For DELETE: path, rationale

You cannot investigate or output narrative text - you can ONLY call submit_execution_plan."""

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
    use_local_investigation: bool = False
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _instructor_client: instructor.Instructor = field(init=False, repr=False)
    _local_client: Any = field(init=False, repr=False, default=None)
    _event_bus: Any = field(init=False, repr=False)
    _tool_handlers: Mapping[str, ToolHandler] = field(init=False, repr=False)
    _latest_plan: ExecutionPlan | None = field(default=None, init=False, repr=False)
    _active_conversation_id: int | None = field(default=None, init=False, repr=False)
    _current_user_request: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._instructor_client = instructor.from_anthropic(anthropic.Anthropic(api_key=self.api_key))

        # Initialize local client for investigation if enabled
        if self.use_local_investigation:
            self._local_client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            LOGGER.info("Local investigation enabled | model=deepseek-coder-v2:16b")

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

        This method implements a two-phase architecture:
        1. Investigation Phase: Uses read-only tools to gather context
        2. Planning Phase: Converts investigation findings into ExecutionPlan

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
            "Analyst two-phase architecture started | investigation_model=%s | planning_model=%s | prompt_chars=%d",
            self.investigation_model,
            self.planning_model,
            len(user_request or ""),
        )
        self._event_bus.emit(
            PhaseTransition(from_phase="idle", to_phase="analyst", source=_ANALYST_SOURCE)
        )
        self._emit_status("Analyst agent: starting two-phase analysis...", "analyst.start")
        self._event_bus.emit(
            AgentEvent(
                name="analyst.started",
                payload={
                    "investigation_model": self.investigation_model,
                    "planning_model": self.planning_model,
                    "architecture": "two-phase",
                },
                source=_ANALYST_SOURCE,
            )
        )
        self._event_bus.emit(
            TaskProgressEvent(
                message="Analyst Phase 1: Investigation",
                percent=0.1,
                source=_ANALYST_SOURCE,
            )
        )

        try:
            # Phase 1: Investigation
            investigation_result = self._run_investigation_phase(investigation_messages)

            if isinstance(investigation_result, str):
                # Error occurred during investigation
                error_message = f"Investigation phase failed: {investigation_result}"
                LOGGER.error(error_message)
                self._emit_status("Analyst agent: investigation failed", "analyst.error")
                self._event_bus.emit(
                    PhaseTransition(
                        from_phase="analyst",
                        to_phase="analyst.error",
                        source=_ANALYST_SOURCE,
                    )
                )
                self._emit_completion(error_message, success=False)
                return error_message

            investigation_summary, updated_messages = investigation_result

            # Update progress
            self._event_bus.emit(
                TaskProgressEvent(
                    message="Analyst Phase 2: Planning",
                    percent=0.3,
                    source=_ANALYST_SOURCE,
                )
            )

            # Phase 2: Planning
            planning_result = self._run_planning_phase(user_request, investigation_summary)

            if isinstance(planning_result, str):
                # Error occurred during planning
                error_message = f"Planning phase failed: {planning_result}"
                LOGGER.error(error_message)
                self._emit_status("Analyst agent: planning failed", "analyst.error")
                self._event_bus.emit(
                    PhaseTransition(
                        from_phase="analyst",
                        to_phase="analyst.error",
                        source=_ANALYST_SOURCE,
                    )
                )
                self._emit_completion(error_message, success=False)
                return error_message

            # Success - we have an execution plan
            plan = planning_result
            duration = time.perf_counter() - started

            LOGGER.info(
                "Analyst two-phase architecture completed | duration=%.2fs | operations=%d",
                duration,
                len(plan.operations),
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
                f"Analyst completed execution plan with {len(plan.operations)} operations."
            )
            self._emit_completion(summary, success=True)
            return plan

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

    def _run_investigation_phase(
        self,
        investigation_messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]] | str:
        """Run Phase 1: Investigation Agent with read-only tools.

        Returns:
            Tuple of (investigation_summary, updated_messages) on success
            Error string on failure
        """
        # Select client and model based on local investigation setting
        investigation_client = self._local_client if self.use_local_investigation else self._client
        investigation_model = "deepseek-coder-v2:16b" if self.use_local_investigation else self.investigation_model

        LOGGER.info(
            "Phase 1: Investigation Agent started | local=%s | model=%s",
            self.use_local_investigation,
            investigation_model,
        )
        self._emit_status("Investigation: gathering context...", "analyst.investigation")
        self._event_bus.emit(
            AgentEvent(
                name="analyst.investigation_started",
                payload={
                    "model": investigation_model,
                    "local": self.use_local_investigation,
                },
                source=_ANALYST_SOURCE,
            )
        )

        max_tool_calls = 30
        tool_calls = 0

        # Build read-only tool definitions (exclude submit_execution_plan)
        read_only_tools = [
            name for name in self._tool_handlers.keys()
            if name != "submit_execution_plan"
        ]

        try:
            while tool_calls < max_tool_calls:
                # Enable prompt caching (only for Anthropic client)
                if self.use_local_investigation:
                    # For local client, use plain system prompt and tools
                    system_prompt = ANALYST_PROMPT
                    tools = self._build_tool_definitions(allowed_tools=read_only_tools)
                else:
                    # For Anthropic, enable prompt caching
                    system_prompt, tools = build_cached_system_and_tools(
                        system_prompt=ANALYST_PROMPT,
                        tools=self._build_tool_definitions(allowed_tools=read_only_tools),
                    )

                response = investigation_client.messages.create(
                    model=investigation_model,
                    system=system_prompt,
                    temperature=0,
                    max_tokens=INVESTIGATION_MAX_TOKENS,
                    tools=tools,
                    messages=investigation_messages,
                )

                investigation_messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "tool_use":
                    tool_calls += 1
                    if tool_calls % 5 == 0:
                        LOGGER.info(
                            "Investigation in progress | tool_calls=%d/%d",
                            tool_calls,
                            max_tool_calls,
                        )

                    tool_results = self._collect_tool_results(response.content)
                    if tool_results:
                        investigation_messages.append({"role": "user", "content": tool_results})
                    continue

                # stop_reason is "end_turn" - investigation complete
                investigation_summary = (self._collect_text(response.content) or "").strip()

                if not investigation_summary:
                    LOGGER.warning("Investigation ended with no text summary")
                    investigation_summary = "Investigation completed with no specific findings."

                LOGGER.info(
                    "Phase 1 complete | tool_calls=%d | summary_length=%d",
                    tool_calls,
                    len(investigation_summary),
                )
                self._emit_status("Investigation: context gathered", "analyst.investigation_complete")
                self._event_bus.emit(
                    AgentEvent(
                        name="analyst.investigation_complete",
                        payload={
                            "tool_calls": tool_calls,
                            "summary_length": len(investigation_summary),
                        },
                        source=_ANALYST_SOURCE,
                    )
                )

                return investigation_summary, investigation_messages

            # Exceeded max tool calls
            error_msg = f"Investigation exceeded maximum tool calls ({max_tool_calls})"
            LOGGER.error(error_msg)
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.investigation_tool_limit",
                    details={"max_tool_calls": max_tool_calls},
                    source=_ANALYST_SOURCE,
                )
            )
            return error_msg

        except anthropic.APIError as exc:
            LOGGER.exception("Investigation API error (Anthropic)")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.investigation_api_error",
                    details={"message": str(exc)},
                    source=_ANALYST_SOURCE,
                )
            )
            return f"Investigation failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            # Catch all other errors including OpenAI/local API errors
            error_type = "local" if self.use_local_investigation else "unexpected"
            LOGGER.exception("Investigation %s error", error_type)
            self._event_bus.emit(
                SystemErrorEvent(
                    error=f"analyst.investigation_{error_type}_error",
                    details={"message": str(exc), "local": self.use_local_investigation},
                    source=_ANALYST_SOURCE,
                )
            )
            return f"Investigation failed: {exc}"

    def _run_planning_phase(
        self,
        user_request: str,
        investigation_summary: str,
    ) -> ExecutionPlan | str:
        """Run Phase 2: Planning Agent with instructor for guaranteed structured outputs.

        Args:
            user_request: Original user request
            investigation_summary: Text summary from investigation phase

        Returns:
            ExecutionPlan object on success
            Error string on failure
        """
        LOGGER.info("Phase 2: Planning Agent started")
        self._emit_status("Planning: generating execution plan...", "analyst.planning")
        self._event_bus.emit(
            AgentEvent(
                name="analyst.planning_started",
                payload={"model": self.planning_model},
                source=_ANALYST_SOURCE,
            )
        )

        # Build planning messages with user request and investigation findings
        planning_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_request}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": investigation_summary}],
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Now create the execution plan. Include all required fields: task_summary, project_context, quality_checklist, estimated_files, and operations array with at least one CREATE/MODIFY/DELETE operation.",
                }],
            },
        ]

        try:
            # Use instructor for guaranteed structured output
            plan = self._instructor_client.messages.create(
                model=self.planning_model,
                max_tokens=PLANNING_MAX_TOKENS,
                temperature=0,
                messages=planning_messages,
                response_model=ExecutionPlan,
            )

            LOGGER.info(
                "Phase 2 complete | operations=%d",
                len(plan.operations),
            )
            self._emit_status("Planning: execution plan ready", "analyst.planning_complete")
            self._event_bus.emit(
                AgentEvent(
                    name="analyst.planning_complete",
                    payload={"operations": len(plan.operations)},
                    source=_ANALYST_SOURCE,
                )
            )
            return plan

        except anthropic.APIError as exc:
            LOGGER.exception("Planning API error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.planning_api_error",
                    details={"message": str(exc)},
                    source=_ANALYST_SOURCE,
                )
            )
            return f"Planning failed: {exc}"
        except ValidationError as exc:
            LOGGER.exception("Planning validation error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.planning_validation_error",
                    details={"message": str(exc)},
                    source=_ANALYST_SOURCE,
                )
            )
            return f"Planning failed: {self._render_plan_validation_error(exc)}"
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Planning unexpected error")
            self._event_bus.emit(
                SystemErrorEvent(
                    error="analyst.planning_unexpected_error",
                    details={"message": str(exc)},
                    source=_ANALYST_SOURCE,
                )
            )
            return f"Planning failed: {exc}"

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
