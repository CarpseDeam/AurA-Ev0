"""Claude executor service for executing prompts and creating files."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import anthropic

from aura.prompts import CLAUDE_EXECUTOR_PROMPT
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)


@dataclass
class ClaudeExecutorService:
    """Executes prompts using Claude Sonnet 4 with write-only tools.

    This service receives comprehensive prompts from the analyst and
    executes them reliably using file creation and modification tools.
    Formats output to look like CLI tools for visual satisfaction.
    """

    api_key: str
    tool_manager: ToolManager
    model_name: str = "claude-sonnet-4-20250514"

    def execute_prompt(
        self,
        engineered_prompt: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Execute an engineered prompt with Claude.

        Args:
            engineered_prompt: Comprehensive prompt from analyst
            on_chunk: Optional callback for streaming CLI-style output

        Returns:
            Summary of execution results
        """
        started = time.perf_counter()
        prompt_length = len(engineered_prompt or "")
        LOGGER.info(
            "Claude execution started | model=%s | prompt_chars=%d | streaming=%s",
            self.model_name,
            prompt_length,
            bool(on_chunk),
        )

        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            tools = self._build_anthropic_tools()

            full_response_text = []

            with client.messages.stream(
                model=self.model_name,
                max_tokens=8096,
                system=CLAUDE_EXECUTOR_PROMPT,
                messages=[{"role": "user", "content": engineered_prompt}],
                tools=tools,
            ) as stream:
                for event in stream:
                    formatted = self._format_stream_event(event)
                    if formatted and on_chunk:
                        on_chunk(formatted)
                        if not formatted.startswith(
                            ("+ ", "~ ", "- ", "⋯ ", "✓ ", "TOOL_CALL::")
                        ):
                            full_response_text.append(formatted)

            final_message = stream.get_final_message()
            result_text = "".join(full_response_text).strip()

            if not result_text and final_message.content:
                for block in final_message.content:
                    if hasattr(block, "text"):
                        result_text += block.text

            duration = time.perf_counter() - started
            LOGGER.info(
                "Claude execution completed | duration=%.2fs | response_chars=%d",
                duration,
                len(result_text),
            )

            if on_chunk:
                on_chunk("\n✓ Execution complete\n")

            return result_text or "Execution complete."

        except anthropic.APIError as exc:
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Claude execution failed | duration=%.2fs | error=%s",
                duration,
                str(exc),
            )
            return (
                f"Error: Unable to reach Claude API. Please verify ANTHROPIC_API_KEY "
                f"and network connectivity. Detail: {exc}"
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Claude execution failed unexpectedly | duration=%.2fs",
                duration,
            )
            return f"Error: Execution failed: {exc}"

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

    def _format_stream_event(
        self,
        event: anthropic.types.MessageStreamEvent,
    ) -> str:
        """Format streaming events into CLI-style output.

        Args:
            event: Anthropic stream event

        Returns:
            Formatted CLI-style string or empty string
        """
        event_type = event.type

        if event_type == "content_block_delta":
            if hasattr(event, "delta") and hasattr(event.delta, "text"):
                return event.delta.text

        elif event_type == "content_block_start":
            if hasattr(event, "content_block"):
                block = event.content_block
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_name = getattr(block, "name", "")
                    tool_id = getattr(block, "id", "")
                    return f"TOOL_CALL::{tool_name}::{tool_id}"

        elif event_type == "message_delta":
            if hasattr(event, "delta"):
                delta = event.delta
                if hasattr(delta, "stop_reason") and delta.stop_reason == "tool_use":
                    return ""

        return ""

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
            result = self.tool_manager.modify_file(path, old_content, new_content)
            return f"~ Modified {path}\n{result}"

        elif tool_name == "delete_file":
            path = tool_input.get("path", "")
            result = self.tool_manager.delete_file(path)
            return f"- Deleted {path}\n{result}"

        else:
            return f"Error: Unknown tool '{tool_name}'"
