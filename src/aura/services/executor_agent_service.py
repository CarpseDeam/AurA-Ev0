"""Executor agent service for executing prompts and creating files."""

from __future__ import annotations

import difflib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic

from aura.event_bus import get_event_bus
from aura.events import (
    ExecutionComplete,
    FileOperation,
    PhaseTransition,
    StatusUpdate,
    StreamingChunk,
    ToolCallCompleted,
    ToolCallStarted,
)
from aura.prompts import EXECUTOR_PROMPT
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_EXECUTOR_SOURCE = "executor"


@dataclass
class ExecutorAgentService:
    """Executes prompts using the configured executor model with write-only tools.

    This service receives comprehensive prompts from the analyst and
    executes them reliably using file creation and modification tools.
    Formats output to look like CLI tools for visual satisfaction.
    """

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _event_bus: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._event_bus = get_event_bus()

    def execute_prompt(
        self,
        engineered_prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute an engineered prompt with the executor provider.

        Args:
            engineered_prompt: Comprehensive prompt from analyst
            on_chunk: Optional callback for streaming CLI-style output

        Returns:
            Summary of execution results
        """
        started = time.perf_counter()
        prompt_length = len(engineered_prompt or "")
        LOGGER.info(
            "Executor execution started | model=%s | prompt_chars=%d | streaming=%s",
            self.model_name,
            prompt_length,
            bool(on_chunk),
        )
        self._event_bus.emit(
            PhaseTransition(from_phase="idle", to_phase="executor", source=_EXECUTOR_SOURCE)
        )
        self._emit_status("Executor agent: dispatching micro-actions...", "executor.run")

        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            tools = self._build_anthropic_tools()

            # Initialize conversation with user's prompt
            messages = [{"role": "user", "content": engineered_prompt}]

            # Tool execution loop - continues until the executor stops requesting tools
            while True:
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=8096,
                    system=EXECUTOR_PROMPT,
                    messages=messages,
                    tools=tools,
                )

                # Add assistant's response to conversation
                messages.append({"role": "assistant", "content": response.content})

                # Check stop reason
                if response.stop_reason == "tool_use":
                    # Executor agent wants to use tools - extract and execute them
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            tool_name = block.name or "tool"
                            tool_input = dict(block.input or {})
                            tool_id = block.id

                            params = self._sanitize_tool_args(tool_input)
                            self._event_bus.emit(
                                ToolCallStarted(
                                    tool_name=tool_name,
                                    parameters=params,
                                    source=_EXECUTOR_SOURCE,
                                )
                            )
                            tool_started = time.perf_counter()
                            try:
                                result = self._execute_tool(tool_name, tool_input)
                            except Exception as exc:
                                duration = time.perf_counter() - tool_started
                                self._event_bus.emit(
                                    ToolCallCompleted(
                                        tool_name=tool_name,
                                        result=f"error: {exc}",
                                        duration=duration,
                                        source=_EXECUTOR_SOURCE,
                                    )
                                )
                                self._maybe_emit_file_operation(tool_name, tool_input)
                                raise

                            duration = time.perf_counter() - tool_started
                            self._event_bus.emit(
                                ToolCallCompleted(
                                    tool_name=tool_name,
                                    result=self._safe_value(result, limit=320),
                                    duration=duration,
                                    source=_EXECUTOR_SOURCE,
                                )
                            )
                            self._maybe_emit_file_operation(tool_name, tool_input)

                            # Build tool result for the executor provider
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result,
                                }
                            )

                    # Add tool results to conversation
                    messages.append({"role": "user", "content": tool_results})

                    # Continue loop - the executor will see results and respond

                elif response.stop_reason in ("end_turn", "stop_sequence", None):
                    # Executor finished - extract final text
                    final_text = ""
                    for block in response.content:
                        if block.type == "text":
                            final_text += block.text

                    duration = time.perf_counter() - started
                    LOGGER.info(
                        "Executor execution completed | duration=%.2fs | response_chars=%d",
                        duration,
                        len(final_text),
                    )

                    if final_text:
                        self._emit_streaming_chunk(final_text, on_chunk=on_chunk)
                    self._emit_streaming_chunk(
                        "✓ Execution complete",
                        on_chunk=on_chunk,
                        is_final=True,
                    )
                    self._emit_status("Executor agent: execution complete", "executor.complete")
                    self._event_bus.emit(
                        PhaseTransition(from_phase="executor", to_phase="idle", source=_EXECUTOR_SOURCE)
                    )
                    summary = final_text or "Execution complete."
                    self._emit_completion(summary, success=True)
                    return summary

                else:
                    # Unexpected stop reason
                    LOGGER.warning(
                        "Unexpected stop_reason: %s",
                        response.stop_reason,
                    )
                    break

        except anthropic.APIError as exc:
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Executor execution failed | duration=%.2fs | error=%s",
                duration,
                str(exc),
            )
            error_message = (
                f"Error: Unable to reach the executor provider. Please verify ANTHROPIC_API_KEY "
                f"and network connectivity. Detail: {exc}"
            )
            self._emit_status("Executor agent: failed", "executor.error")
            self._event_bus.emit(
                PhaseTransition(from_phase="executor", to_phase="executor.error", source=_EXECUTOR_SOURCE)
            )
            self._emit_completion(error_message, success=False)
            return error_message
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Executor execution failed unexpectedly | duration=%.2fs",
                duration,
            )
            error_message = f"Error: Execution failed: {exc}"
            self._emit_status("Executor agent: failed", "executor.error")
            self._event_bus.emit(
                PhaseTransition(from_phase="executor", to_phase="executor.error", source=_EXECUTOR_SOURCE)
            )
            self._emit_completion(error_message, success=False)
            return error_message

    def _build_anthropic_tools(self) -> list[dict]:
        """Build Anthropic tool definitions for write operations."""
        return [
            {
                "name": "create_file",
                "description": (
                    "Create a new file with the specified content. The file will be "
                    "created in the workspace directory. Parent directories will be "
                    "created automatically if they don't exist."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "Relative path for the new file (e.g., 'src/utils/helper.py')"
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete file content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "modify_file",
                "description": (
                    "Modify an existing file by replacing old content with new content. "
                    "The old_content must match exactly for the replacement to succeed."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to modify",
                        },
                        "old_content": {
                            "type": "string",
                            "description": (
                                "Exact content to replace (must match exactly)"
                            ),
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New content to insert",
                        },
                    },
                    "required": ["path", "old_content", "new_content"],
                },
            },
            {
                "name": "delete_file",
                "description": (
                    "Delete a file from the workspace. Use with caution."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative path to the file to delete",
                        },
                    },
                    "required": ["path"],
                },
            },
        ]

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict,
    ) -> str:
        """Execute a tool and return formatted result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input parameters

        Returns:
            Formatted execution result
        """
        if tool_name == "create_file":
            path = tool_input.get("path", "")
            content = tool_input.get("content", "")
            result = self.tool_manager.create_file(path, content)
            return f"+ Created {path}\n{result}"

        elif tool_name == "modify_file":
            path = tool_input.get("path", "")
            old_content = tool_input.get("old_content", "")
            new_content = tool_input.get("new_content", "")
            diff_block = self._build_diff_block(path, old_content, new_content)
            result = self.tool_manager.modify_file(path, old_content, new_content)
            parts = [f"~ Modified {path}", result]
            if diff_block:
                parts.append(diff_block)
            return "\n".join(parts)

        elif tool_name == "delete_file":
            path = tool_input.get("path", "")
            result = self.tool_manager.delete_file(path)
            return f"- Deleted {path}\n{result}"

        else:
            return f"Error: Unknown tool '{tool_name}'"

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
        """Emit FileOperation events for write tools."""
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
