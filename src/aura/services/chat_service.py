"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Set

from google import genai
from google.genai import types

from aura import config
from aura.prompt import AURA_SYSTEM_PROMPT
from aura.tools.local_agent_tools import generate_commit_message
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)


@dataclass
class ChatService:
    """Manages conversational interactions with developer tool access."""

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _client: genai.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Gemini client."""
        self._client = genai.Client(api_key=self.api_key)


    def send_message(
        self,
        user_message: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send a message and stream the response with automatic tool usage."""
        started = time.perf_counter()
        prompt_length = len(user_message or "")
        LOGGER.info(
            "Gemini request started | model=%s | prompt_chars=%d | streaming=%s",
            self.model_name,
            prompt_length,
            bool(on_chunk),
        )
        try:
            # Create config with Python functions as tools
            request_config = types.GenerateContentConfig(
                tools=[
                    self.tool_manager.list_project_files,
                    self.tool_manager.search_in_files,
                    self.tool_manager.read_project_file,
                    self.tool_manager.read_multiple_files,
                    self.tool_manager.get_function_definitions,
                    self.tool_manager.run_tests,
                    self.tool_manager.lint_code,
                    self.tool_manager.format_code,
                    self.tool_manager.install_package,
                    self.tool_manager.get_git_status,
                    self.tool_manager.git_commit,
                    self.tool_manager.git_push,
                    self.tool_manager.git_diff,
                    generate_commit_message,
                    self.tool_manager.find_definition,
                    self.tool_manager.find_usages,
                    self.tool_manager.get_imports,
                    self.tool_manager.create_file,
                    self.tool_manager.modify_file,
                    self.tool_manager.replace_file_lines,
                    self.tool_manager.delete_file,
                ],
                system_instruction=AURA_SYSTEM_PROMPT,
            )

            # The SDK automatically:
            # - Detects function calls
            # - Executes the functions
            # - Sends results back to model
            # - Repeats until model returns text
            stream = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=user_message,
                config=request_config,
            )

            try:
                iterator = iter(stream)
            except TypeError:
                LOGGER.debug("Streaming unavailable; falling back to blocking response")
                fallback_response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=request_config,
                )
                final_text = fallback_response.text or ""
                if final_text and on_chunk:
                    on_chunk(f"{config.STREAM_PREFIX}{final_text}")
                    on_chunk(f"{config.STREAM_PREFIX}\n")
                duration = time.perf_counter() - started
                LOGGER.info(
                    "Gemini request completed (fallback) | duration=%.2fs | streamed=%s",
                    duration,
                    bool(final_text),
                )
                return final_text

            aggregated_text = ""
            collected_parts: list[str] = []
            seen_calls: Set[str] = set()
            streamed_any = False

            for chunk in iterator:
                streamed_any = (
                    self._emit_function_calls(chunk, on_chunk, seen_calls) or streamed_any
                )
                addition, aggregated_text = self._extract_stream_addition(chunk, aggregated_text)
                if addition:
                    collected_parts.append(addition)
                    if on_chunk:
                        on_chunk(f"{config.STREAM_PREFIX}{addition}")
                    streamed_any = True

            final_text = "".join(collected_parts) or aggregated_text
            if not final_text and hasattr(stream, "text"):
                final_text = getattr(stream, "text") or ""

            if streamed_any and on_chunk:
                on_chunk(f"{config.STREAM_PREFIX}\n")

            duration = time.perf_counter() - started
            LOGGER.info(
                "Gemini request completed | duration=%.2fs | streamed=%s",
                duration,
                streamed_any,
            )
            return final_text

        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Gemini request failed | duration=%.2fs | prompt_preview=%s",
                duration,
                (user_message or "")[:80],
            )
            return "Error: Unable to reach Gemini. Please verify GEMINI_API_KEY and network connectivity."

    @staticmethod
    def _extract_stream_addition(
        chunk: types.GenerateContentResponse,
        aggregated_text: str,
    ) -> tuple[str, str]:
        """Return the new text produced by this chunk."""
        text = getattr(chunk, "text", "") or ""
        if not text:
            return "", aggregated_text

        if aggregated_text and aggregated_text.startswith(text):
            # Received a shorter chunk that is already included in the aggregate.
            return "", aggregated_text

        if text.startswith(aggregated_text):
            addition = text[len(aggregated_text) :]
            return addition, text

        # Fallback: assume chunk.text itself is the new addition
        return text, aggregated_text + text

    def _emit_function_calls(
        self,
        chunk: types.GenerateContentResponse,
        on_chunk: Optional[Callable[[str], None]],
        seen_calls: Set[str],
    ) -> bool:
        """Emit tool call notifications for newly observed function calls."""
        function_calls = getattr(chunk, "function_calls", None)
        if not function_calls:
            return False

        emitted = False
        for call in function_calls:
            name = getattr(call, "name", "")
            args_summary = self._summarize_args(getattr(call, "args", None))
            signature = f"{name}:{args_summary}"
            if signature in seen_calls:
                continue
            seen_calls.add(signature)
            LOGGER.debug(
                "Gemini tool call | name=%s | args=%s",
                name,
                args_summary,
            )
            if on_chunk:
                on_chunk(f"TOOL_CALL::{name}::{args_summary}")
            emitted = True
        return emitted

    @staticmethod
    def _summarize_args(args: object) -> str:
        """Return a compact JSON summary of function call arguments."""
        if not args:
            return "{}"
        try:
            summary = json.dumps(args, ensure_ascii=False)
        except TypeError:
            summary = str(args)
        if len(summary) > 160:
            summary = f"{summary[:157]}..."
        return summary

    def clear_history(self) -> None:
        """Clear the conversation history (no-op with stateless client)."""
        pass
