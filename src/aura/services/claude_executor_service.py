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
    model_name: str

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

            # Initialize conversation with user's prompt
            messages = [{"role": "user", "content": engineered_prompt}]

            # Tool execution loop - continues until Claude stops requesting tools
            while True:
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=8096,
                    system=CLAUDE_EXECUTOR_PROMPT,
                    messages=messages,
                    tools=tools,
                )

                # Add assistant's response to conversation
                messages.append({"role": "assistant", "content": response.content})

                # Check stop reason
                if response.stop_reason == "tool_use":
                    # Claude wants to use tools - extract and execute them
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_input = block.input
                            tool_id = block.id

                            # Stream tool call notification
                            if on_chunk:
                                on_chunk(f"TOOL_CALL::{tool_name}::{tool_input}\n")

                            # Execute the tool
                            result = self._execute_tool(tool_name, tool_input)

                            # Stream the result
                            if on_chunk:
                                on_chunk(result + "\n")

                            # Build tool result for Claude
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result,
                                }
                            )

                    # Add tool results to conversation
                    messages.append({"role": "user", "content": tool_results})

                    # Continue loop - Claude will see results and respond

                elif response.stop_reason in ("end_turn", "stop_sequence", None):
                    # Claude finished - extract final text
                    final_text = ""
                    for block in response.content:
                        if block.type == "text":
                            final_text += block.text

                    duration = time.perf_counter() - started
                    LOGGER.info(
                        "Claude execution completed | duration=%.2fs | response_chars=%d",
                        duration,
                        len(final_text),
                    )

                    if on_chunk:
                        if final_text:
                            on_chunk(final_text + "\n")
                        on_chunk("✓ Execution complete\n")

                    return final_text or "Execution complete."

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
