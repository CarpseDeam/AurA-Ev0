"""Gemini analyst service for analyzing requests and building prompts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Set

from google import genai
from google.genai import types

from aura import config
from aura.prompts import GEMINI_ANALYST_PROMPT
from aura.tools.git_tools import get_git_status, git_diff
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
class GeminiAnalystService:
    """Analyzes requests and builds comprehensive prompts for the executor.

    This service uses Gemini 2.5 Pro with read-only tools to gather context,
    understand patterns, and engineer detailed prompts for Claude to execute.
    """

    api_key: str
    tool_manager: ToolManager
    model_name: str = "gemini-2.5-pro"
    _client: genai.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Gemini client."""
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
            Comprehensive engineered prompt for Claude executor
        """
        started = time.perf_counter()
        prompt_length = len(user_request or "")
        LOGGER.info(
            "Gemini analysis started | model=%s | prompt_chars=%d | streaming=%s",
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
                ],
                system_instruction=GEMINI_ANALYST_PROMPT,
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
                final_text = fallback_response.text or ""
                if final_text and on_chunk:
                    on_chunk(f"{config.STREAM_PREFIX}{final_text}")
                    on_chunk(f"{config.STREAM_PREFIX}\n")
                duration = time.perf_counter() - started
                LOGGER.info(
                    "Gemini analysis completed (fallback) | duration=%.2fs | streamed=%s",
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
                    self._emit_function_calls(chunk, on_chunk, seen_calls)
                    or streamed_any
                )
                addition, aggregated_text = self._extract_stream_addition(
                    chunk, aggregated_text
                )
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
                "Gemini analysis completed | duration=%.2fs | streamed=%s",
                duration,
                streamed_any,
            )
            return final_text

        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Gemini analysis failed | duration=%.2fs | prompt_preview=%s",
                duration,
                (user_request or "")[:80],
            )
            return (
                "Error: Unable to reach Gemini. Please verify GEMINI_API_KEY "
                "and network connectivity."
            )

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
            return "", aggregated_text

        if text.startswith(aggregated_text):
            addition = text[len(aggregated_text) :]
            return addition, text

        return text, aggregated_text + text

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
