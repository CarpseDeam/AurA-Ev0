"""Executor agent service for applying ExecutionPlans with write-only tools."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
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
from aura.utils.prompt_caching import build_cached_system_and_tools
from aura.exceptions import AuraExecutionError, FileVerificationError
from aura.models import (
    ExecutionPlan,
    FileOperation as PlanOperation,
    FileVerificationLog,
    OperationType,
    ToolCallLog,
)
from aura.prompts import EXECUTOR_PROMPT
from aura.tools import godot_tools
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_EXECUTOR_SOURCE = "executor"
_TOOL_TO_OPERATION: dict[str, OperationType] = {
    "create_file": OperationType.CREATE,
    "modify_file": OperationType.MODIFY,
    "delete_file": OperationType.DELETE,
}


class ExecutorRetryStrategy:
    """Intelligent retry strategy that analyzes failure types and prevents wasted API calls."""

    def __init__(self, max_retries_per_operation: int = 2):
        """Initialize the retry strategy.

        Args:
            max_retries_per_operation: Maximum retry attempts per unique operation
        """
        self.max_retries_per_operation = max_retries_per_operation
        self._retry_counts: dict[tuple[str, str], int] = {}  # (tool_name, file_path) -> retry_count

    def should_retry(
        self,
        tool_name: str,
        file_path: str,
        error_message: str,
        is_verification_error: bool,
    ) -> tuple[bool, str | None]:
        """Determine if an operation should be retried and provide guidance.

        Args:
            tool_name: Name of the tool that failed
            file_path: File path being operated on
            error_message: Error message from the failure
            is_verification_error: Whether this is a FileVerificationError

        Returns:
            Tuple of (should_retry, guidance_message)
            - should_retry: True if retry is viable, False to fail fast
            - guidance_message: Optional guidance for the model on how to fix the issue
        """
        operation_key = (tool_name, file_path)
        current_retries = self._retry_counts.get(operation_key, 0)

        # Check retry limit first
        if current_retries >= self.max_retries_per_operation:
            return (
                False,
                f"Maximum retry limit ({self.max_retries_per_operation}) reached for {file_path}. "
                "Operation cannot be retried further.",
            )

        # Analyze error type - FileVerificationError means bad plan data, fail fast
        if is_verification_error:
            LOGGER.warning(
                "FileVerificationError detected for %s on %s - failing fast (bad plan data)",
                tool_name,
                file_path,
            )
            return (
                False,
                f"FileVerificationError indicates invalid plan data for {file_path}. "
                "This operation cannot succeed with the current execution plan. "
                "The plan may be missing required content or have malformed data.",
            )

        # Analyze error message patterns
        error_lower = error_message.lower()

        # "not found" errors - likely dependency issue or bad path
        if any(pattern in error_lower for pattern in ["not found", "no such file", "does not exist"]):
            self._retry_counts[operation_key] = current_retries + 1
            return (
                True,
                f"File or dependency not found for {file_path}. "
                "Verify that all dependencies exist and the file path is correct. "
                "If this is a MODIFY operation, the file may not exist yet. "
                "Check if a CREATE operation should run first.",
            )

        # "old_content not found" - content mismatch, need to re-read
        if any(pattern in error_lower for pattern in ["old_content", "content mismatch", "does not match"]):
            self._retry_counts[operation_key] = current_retries + 1
            return (
                True,
                f"Content mismatch detected for {file_path}. "
                "The file contents have changed or don't match the expected old_content. "
                "Re-read the file to get the current contents before attempting modification.",
            )

        # Permission errors - fail fast, not fixable by retry
        if any(pattern in error_lower for pattern in ["permission denied", "access denied", "readonly"]):
            return (
                False,
                f"Permission denied for {file_path}. "
                "This is a file system permission issue that cannot be resolved by retrying.",
            )

        # Generic errors - allow one retry with general guidance
        self._retry_counts[operation_key] = current_retries + 1
        return (
            True,
            f"Operation failed for {file_path} with error: {error_message[:200]}. "
            "Review the error and adjust the operation accordingly.",
        )

    def reset_operation(self, tool_name: str, file_path: str) -> None:
        """Reset retry count for a successful operation.

        Args:
            tool_name: Name of the tool
            file_path: File path being operated on
        """
        operation_key = (tool_name, file_path)
        self._retry_counts.pop(operation_key, None)


@dataclass
class ExecutorAgentService:
    """Apply ExecutionPlan operations via Claude Sonnet 4.5 and file tools."""

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _event_bus: Any = field(init=False, repr=False)
    _client: anthropic.Anthropic = field(init=False, repr=False)
    _active_conversation_id: int | None = field(default=None, init=False, repr=False)
    _plan_operation_queues: dict[tuple[OperationType, str], deque[PlanOperation]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _retry_strategy: ExecutorRetryStrategy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._event_bus = get_event_bus()
        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._retry_strategy = ExecutorRetryStrategy(max_retries_per_operation=2)

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
        self._prepare_operation_queues(execution_plan)
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

                # Enable prompt caching for system and tools to reduce token costs
                cached_system, cached_tools = build_cached_system_and_tools(
                    system_prompt=EXECUTOR_PROMPT,
                    tools=tools,
                )

                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=8096,
                    system=cached_system,
                    messages=messages,
                    tools=cached_tools,
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
                    try:
                        self._run_final_verification(execution_plan)
                    except FileVerificationError as exc:
                        return self._handle_final_verification_failure(exc)
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
            build_anthropic_tool_schema(self.tool_manager.create_file, name="create_file"),
            build_anthropic_tool_schema(self.tool_manager.modify_file, name="modify_file"),
            build_anthropic_tool_schema(self.tool_manager.delete_file, name="delete_file"),
            build_anthropic_tool_schema(godot_tools.create_godot_script, name="create_godot_script"),
        ]
        return tools

    def _invoke_workspace_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a workspace tool with auditing, logging, and DB persistence."""
        prepared_input = self._prepare_tool_input(tool_name, dict(tool_input or {}))
        params = self._sanitize_tool_args(prepared_input)
        file_path = prepared_input.get("path", "unknown")

        self._event_bus.emit(
            ToolCallStarted(tool_name=tool_name, parameters=params, source=_EXECUTOR_SOURCE)
        )
        started = time.perf_counter()
        result_text = ""
        success = False
        try:
            result_text = self._execute_tool(tool_name, prepared_input)
            success = True
            # Reset retry count on successful operation
            self._retry_strategy.reset_operation(tool_name, file_path)
            self._maybe_emit_file_operation(tool_name, prepared_input)
            return result_text
        except FileVerificationError as exc:
            error_text = str(exc)
            result_text = error_text
            self._emit_verification_failure_event(tool_name, prepared_input, error_text)

            # Check retry strategy for verification errors
            should_retry, guidance = self._retry_strategy.should_retry(
                tool_name=tool_name,
                file_path=file_path,
                error_message=error_text,
                is_verification_error=True,
            )

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
                    details={
                        "tool": tool_name,
                        "message": error_text,
                        "retry_allowed": should_retry,
                    },
                    source=_EXECUTOR_SOURCE,
                )
            )

            # Fail fast for verification errors (bad plan data)
            LOGGER.error(
                "FileVerificationError - failing fast | tool=%s | file=%s | guidance=%s",
                tool_name,
                file_path,
                guidance,
            )
            raise

        except Exception as exc:
            error_text = str(exc)
            result_text = error_text

            # Check retry strategy for other errors
            should_retry, guidance = self._retry_strategy.should_retry(
                tool_name=tool_name,
                file_path=file_path,
                error_message=error_text,
                is_verification_error=False,
            )

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
                    details={
                        "tool": tool_name,
                        "message": error_text,
                        "retry_allowed": should_retry,
                    },
                    source=_EXECUTOR_SOURCE,
                )
            )

            # If retry is not allowed, fail fast
            if not should_retry:
                LOGGER.error(
                    "Retry denied - failing fast | tool=%s | file=%s | guidance=%s",
                    tool_name,
                    file_path,
                    guidance,
                )
                raise

            # If retry is allowed, return guidance as tool result instead of raising
            # This allows the model to learn from the error and adjust
            LOGGER.warning(
                "Retry allowed - returning guidance | tool=%s | file=%s | guidance=%s",
                tool_name,
                file_path,
                guidance,
            )
            result_text = f"ERROR: {guidance or error_text}"
            # Mark as success=False for logging, but don't raise
            return result_text

        finally:
            duration = time.perf_counter() - started
            ToolCallLog.record(
                conversation_id=self._active_conversation_id,
                agent_role=_EXECUTOR_SOURCE,
                tool_name=tool_name,
                tool_input=json.dumps(prepared_input, ensure_ascii=False, default=str),
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

    def _prepare_tool_input(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Augment tool input using the authoritative ExecutionPlan payload."""
        operation_type = _TOOL_TO_OPERATION.get(tool_name)
        if not operation_type or not self._plan_operation_queues:
            return tool_input
        path_value = tool_input.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise AuraExecutionError(
                "Workspace tools require a path parameter.",
                {"tool": tool_name},
            )
        plan_operation = self._pop_plan_operation(operation_type, path_value)
        if operation_type is OperationType.CREATE:
            if not plan_operation.content:
                raise FileVerificationError(
                    "ExecutionPlan missing file content for CREATE operation.",
                    {"file": plan_operation.file_path},
                )
            tool_input["content"] = plan_operation.content
        elif operation_type is OperationType.MODIFY:
            if not plan_operation.content:
                raise FileVerificationError(
                    "ExecutionPlan missing file content for MODIFY operation.",
                    {"file": plan_operation.file_path},
                )
            tool_input["new_content"] = plan_operation.content
            tool_input["old_content"] = self._read_current_file_contents(path_value)
        return tool_input

    def _prepare_operation_queues(self, execution_plan: ExecutionPlan) -> None:
        """Build FIFO queues for each planned file operation."""
        self._plan_operation_queues.clear()
        for operation in execution_plan.operations:
            key = (operation.operation_type, self._normalize_plan_path(operation.file_path))
            self._plan_operation_queues.setdefault(key, deque()).append(operation)

    def _pop_plan_operation(self, op_type: OperationType, file_path: str) -> PlanOperation:
        """Return and remove the next planned operation for the given path."""
        normalized_path = self._normalize_plan_path(file_path)
        queue = self._plan_operation_queues.get((op_type, normalized_path))
        if not queue:
            raise AuraExecutionError(
                f"No remaining {op_type.value} operation scheduled for '{file_path}'.",
                {"operation_type": op_type.value, "file": normalized_path},
            )
        return queue.popleft()

    @staticmethod
    def _normalize_plan_path(file_path: str) -> str:
        """Normalize file paths for consistent queue lookups."""
        normalized = (file_path or "").replace("\\", "/")
        return normalized.lstrip("./")

    def _read_current_file_contents(self, path: str) -> str:
        """Read the current workspace file contents prior to modification."""
        try:
            return self.tool_manager.read_project_file(path)
        except Exception as exc:  # noqa: BLE001
            raise FileVerificationError(
                f"Unable to read '{path}' before applying MODIFY operation.",
                {"file": path, "error": str(exc)},
            ) from exc

    def _run_final_verification(self, execution_plan: ExecutionPlan) -> None:
        """Confirm that on-disk files match the submitted execution plan."""
        workspace_root = Path(self.tool_manager.workspace_dir)
        failures: list[str] = []
        for operation in execution_plan.operations:
            normalized_path = self._normalize_plan_path(operation.file_path)
            target = workspace_root / normalized_path
            if operation.operation_type is OperationType.DELETE:
                exists = target.exists()
                actual_bytes = target.read_bytes() if exists else None
                FileVerificationLog.record(
                    phase="final",
                    operation=operation.operation_type.value,
                    file_path=normalized_path,
                    expected_digest=None,
                    actual_digest=self._digest_content(actual_bytes),
                    success=not exists,
                    details=None if not exists else "File still exists after DELETE.",
                    conversation_id=self._active_conversation_id,
                )
                if exists:
                    failures.append(f"{operation.file_path} should be deleted")
                continue

            expected_content = operation.content or ""
            actual_content: str | None = None
            details: str | None = None
            try:
                actual_content = target.read_text(encoding="utf-8")
            except FileNotFoundError:
                details = "File missing after write."
            except UnicodeDecodeError as exc:
                details = f"Unable to decode file contents: {exc}"
            except OSError as exc:  # noqa: BLE001
                details = f"Unable to read file: {exc}"

            success = actual_content == expected_content if actual_content is not None else False
            FileVerificationLog.record(
                phase="final",
                operation=operation.operation_type.value,
                file_path=normalized_path,
                expected_digest=self._digest_content(expected_content),
                actual_digest=self._digest_content(actual_content),
                success=success,
                details=details,
                conversation_id=self._active_conversation_id,
            )
            if not success:
                failures.append(details or f"{operation.file_path} contents differ from plan")

        if failures:
            self._event_bus.emit(
                AgentEvent(
                    name="executor.final_verification_failed",
                    payload={"failures": failures},
                    source=_EXECUTOR_SOURCE,
                )
            )
            raise FileVerificationError(
                "Final verification detected discrepancies.",
                {"failures": "; ".join(failures)},
            )

    @staticmethod
    def _digest_content(payload: str | bytes | None) -> str | None:
        """Return a SHA-256 digest for text or bytes payloads."""
        if payload is None:
            return None
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _emit_verification_failure_event(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        message: str,
    ) -> None:
        """Emit a dedicated event when file verification fails."""
        self._event_bus.emit(
            AgentEvent(
                name="executor.file_verification_failed",
                payload={
                    "tool": tool_name,
                    "file": tool_input.get("path"),
                    "message": message,
                },
                source=_EXECUTOR_SOURCE,
            )
        )

    def _handle_final_verification_failure(self, exc: FileVerificationError) -> str:
        """Convert a final verification failure into user-facing errors."""
        error_message = f"Error: Final verification failed: {exc}"
        self._emit_status("Executor agent: failed", "executor.error")
        self._event_bus.emit(
            SystemErrorEvent(
                error="executor.final_verification_failed",
                details={"message": str(exc)},
                source=_EXECUTOR_SOURCE,
            )
        )
        self._event_bus.emit(
            PhaseTransition(from_phase="executor", to_phase="executor.error", source=_EXECUTOR_SOURCE)
        )
        self._emit_completion(error_message, success=False)
        return error_message

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

        if tool_name == "create_godot_script":
            path = tool_input.get("path", "")
            class_name = tool_input.get("class_name", "")
            extends = tool_input.get("extends", "Node")
            template = tool_input.get("template", "basic")
            result = godot_tools.create_godot_script(path, class_name, extends, template)
            return json.dumps(result, ensure_ascii=False)

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
