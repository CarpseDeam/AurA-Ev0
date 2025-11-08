"""Analyst agent service for analyzing requests and building prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Set

from google import genai
from google.genai import types

from aura.prompts import ANALYST_PROMPT
from aura.tools.git_tools import get_git_status, git_diff
from aura.tools.local_agent_tools import generate_commit_message
from aura.tools.python_tools import (
    format_code,
    get_function_definitions,
    lint_code,
    run_tests,
)
from aura.tools.symbol_tools import find_definition, find_usages, get_imports
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalystAgentService:
    """Analyzes requests and builds comprehensive prompts for the executor.

    This service uses the configured analyst provider with read-only tools to
    gather context, understand patterns, and engineer detailed prompts for the
    executor to carry out.
    """

    api_key: str
    tool_manager: ToolManager
    model_name: str
    _client: genai.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the provider client."""
        self._client = genai.Client(api_key=self.api_key)

    def analyze_and_plan(
        self,
        user_request: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Analyze user request and build comprehensive prompt for executor.

        Args:
            user_request: The user's request to analyze
            on_chunk: Optional callback for streaming output

        Returns:
            Comprehensive engineered prompt for the executor agent
        """
        started = time.perf_counter()
        prompt_length = len(user_request or "")
        LOGGER.info(
            "Analyst analysis started | model=%s | prompt_chars=%d | streaming=%s",
            self.model_name,
            prompt_length,
            bool(on_chunk),
        )

        try:
            request_config = types.GenerateContentConfig(
                tools=[
                    self.tool_manager.list_project_files,
                    self.tool_manager.read_project_file,
                    self.tool_manager.read_multiple_files,
                    self.tool_manager.search_in_files,
                    get_function_definitions,
                    find_definition,
                    find_usages,
                    get_imports,
                    get_git_status,
                    git_diff,
                    run_tests,
                    lint_code,
                    format_code,
                    generate_commit_message,
                ],
                system_instruction=ANALYST_PROMPT,
            )

            stream = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=user_request,
                config=request_config,
            )

            try:
                iterator = iter(stream)
            except TypeError:
                LOGGER.debug(
                    "Streaming unavailable; falling back to blocking response"
                )
                fallback_response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_request,
                    config=request_config,
                )
                # Use robust text extraction
                final_text = getattr(fallback_response, "text", "")
                if not final_text:
                    final_text = self._extract_text_from_parts(fallback_response)

                streamed_any = self._emit_function_calls(
                    fallback_response,
                    on_chunk,
                    set(),
                )
                duration = time.perf_counter() - started
                LOGGER.info(
                    "Analyst analysis completed (fallback) | duration=%.2fs | streamed=%s | text_length=%d",
                    duration,
                    streamed_any,
                    len(final_text),
                )
                return final_text

            aggregated_text = ""
            collected_parts: list[str] = []
            seen_calls: Set[str] = set()
            streamed_any = False

            for chunk in iterator:
                streamed_any = (
                    self._emit_function_calls(chunk, on_chunk, seen_calls)
                    or streamed_any
                )
                addition, aggregated_text = self._extract_stream_addition(
                    chunk, aggregated_text
                )
                if addition:
                    collected_parts.append(addition)

            final_text = "".join(collected_parts) or aggregated_text

            # Last resort: try to get text from the stream object itself
            if not final_text and hasattr(stream, "text"):
                final_text = getattr(stream, "text") or ""

            duration = time.perf_counter() - started
            LOGGER.info(
                "Analyst analysis completed | duration=%.2fs | streamed=%s | text_length=%d",
                duration,
                streamed_any,
                len(final_text),
            )

            if not final_text:
                LOGGER.warning(
                    "Analyst returned empty text despite successful execution. "
                    "This may indicate a response parsing issue."
                )

            return final_text

        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Analyst analysis failed | duration=%.2fs | prompt_preview=%s",
                duration,
                (user_request or "")[:80],
            )
            return (
                "Error: Unable to reach the analyst provider. Please verify "
                "GEMINI_API_KEY and network connectivity."
            )

    @staticmethod
    def _extract_stream_addition(
        chunk: types.GenerateContentResponse,
        aggregated_text: str,
    ) -> tuple[str, str]:
        """Return the new text produced by this chunk.

        This method robustly extracts text from API responses, even when they
        contain complex multi-part content (e.g., after tool calls).
        """
        # Try the simple .text property first (fast path)
        text = getattr(chunk, "text", "")

        # If that didn't work, manually extract from all text parts
        if not text:
            text = AnalystAgentService._extract_text_from_parts(chunk)

        # Still no text? Return early
        if not text:
            return "", aggregated_text

        # Handle incremental streaming logic
        if aggregated_text and aggregated_text.startswith(text):
            return "", aggregated_text

        if text.startswith(aggregated_text):
            addition = text[len(aggregated_text) :]
            return addition, text

        return text, aggregated_text + text

    @staticmethod
    def _extract_text_from_parts(chunk: types.GenerateContentResponse) -> str:
        """Extract text from all text parts in a complex response.

        When the API returns multi-part content (common after tool calls),
        the .text property may be empty. This method manually walks through
        all parts to extract text content.
        """
        text_parts = []

        try:
            # Access candidates array
            candidates = getattr(chunk, "candidates", None)
            if not candidates:
                return ""

            # Get the first candidate
            candidate = candidates[0] if len(candidates) > 0 else None
            if not candidate:
                return ""

            # Access the content within the candidate
            content = getattr(candidate, "content", None)
            if not content:
                return ""

            # Extract text from all parts
            parts = getattr(content, "parts", None)
            if not parts:
                return ""

            for part in parts:
                # Each part might have a 'text' field
                part_text = getattr(part, "text", None)
                if part_text:
                    text_parts.append(part_text)

        except (AttributeError, IndexError, TypeError) as exc:
            LOGGER.debug("Could not extract text from response parts: %s", exc)
            return ""

        return "".join(text_parts)

    def _emit_function_calls(
        self,
        chunk: types.GenerateContentResponse,
        on_chunk: Optional[Callable[[str], None]],
        seen_calls: Set[str],
    ) -> bool:
        """Emit tool call notifications for function calls."""
        function_calls = getattr(chunk, "function_calls", None)
        if not function_calls:
            return False

        emitted = False
        for call in function_calls:
            name = getattr(call, "name", "")
            args_json = self._serialize_args(getattr(call, "args", None))
            signature = hashlib.md5(f"{name}:{args_json}".encode("utf-8")).hexdigest()
            if signature in seen_calls:
                continue
            seen_calls.add(signature)
            log_preview = args_json if len(args_json) <= 240 else f"{args_json[:237]}..."
            LOGGER.debug(
                "Analyst tool call | name=%s | args=%s",
                name,
                log_preview,
            )
            if on_chunk:
                on_chunk(f"TOOL_CALL::{name}::{args_json}")
            emitted = True
        return emitted

    @staticmethod
    def _serialize_args(args: object) -> str:
        """Return a JSON payload describing function call arguments."""
        if args is None:
            return "{}"
        try:
            return json.dumps(args, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return json.dumps(str(args), ensure_ascii=False)
