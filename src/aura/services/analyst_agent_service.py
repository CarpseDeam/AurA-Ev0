"""Analyst agent service for analyzing requests and building prompts."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional

from google import genai
from google.genai import types

from aura import config
from aura.event_bus import get_event_bus
from aura.events import (
    ExecutionComplete,
    FileOperation,
    PhaseTransition,
    StatusUpdate,
    StreamingChunk,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
)
from aura.prompts import ANALYST_PROMPT
from aura.tools.local_agent_tools import generate_commit_message
from aura.tools.tool_manager import ToolManager

LOGGER = logging.getLogger(__name__)
_ANALYST_SOURCE = "analyst"
_FILE_TOOLS = {
    "create_file",
    "delete_file",
    "modify_file",
    "replace_file_lines",
    "read_multiple_files",
    "read_project_file",
}


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
    _event_bus: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the provider client."""
        self._client = genai.Client(api_key=self.api_key)
        self._event_bus = get_event_bus()

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
        workspace = getattr(self.tool_manager, "workspace_dir", None)
        if workspace:
            LOGGER.info("Analyst tool workspace: %s", workspace)
        self._event_bus.emit(
            PhaseTransition(from_phase="idle", to_phase="analyst", source=_ANALYST_SOURCE)
        )
        self._emit_status("Analyst agent: analyzing request...", "analyst.analyze")

        try:
            tool_catalog = [
                self.tool_manager.list_project_files,
                self.tool_manager.read_project_file,
                self.tool_manager.read_multiple_files,
                self.tool_manager.search_in_files,
                self.tool_manager.get_function_definitions,
                self.tool_manager.find_definition,
                self.tool_manager.find_usages,
                self.tool_manager.get_imports,
                self.tool_manager.get_git_status,
                self.tool_manager.git_diff,
                self.tool_manager.run_tests,
                self.tool_manager.lint_code,
                self.tool_manager.format_code,
                generate_commit_message,
            ]
            request_config = types.GenerateContentConfig(
                tools=self._instrument_tools(tool_catalog, source=_ANALYST_SOURCE),
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
                final_text = self._coalesce_response_text(fallback_response)
                duration = time.perf_counter() - started
                if final_text:
                    self._emit_streaming_chunk(
                        final_text,
                        source=_ANALYST_SOURCE,
                        on_chunk=on_chunk,
                    )
                    self._emit_streaming_chunk(
                        "",
                        source=_ANALYST_SOURCE,
                        on_chunk=on_chunk,
                        is_final=True,
                    )
                self._emit_status("Analyst agent: plan ready", "analyst.complete")
                self._event_bus.emit(
                    PhaseTransition(from_phase="analyst", to_phase="idle", source=_ANALYST_SOURCE)
                )
                self._emit_completion(final_text, success=True)
                LOGGER.info(
                    "Analyst analysis completed (fallback) | duration=%.2fs | text_length=%d",
                    duration,
                    len(final_text),
                )
                return final_text

            aggregated_text = ""
            collected_parts: list[str] = []
            streamed_text = False

            for chunk in iterator:
                addition, aggregated_text = self._extract_stream_addition(
                    chunk, aggregated_text
                )
                if addition:
                    collected_parts.append(addition)
                    self._emit_streaming_chunk(
                        addition,
                        source=_ANALYST_SOURCE,
                        on_chunk=on_chunk,
                    )
                    streamed_text = True

            final_text = "".join(collected_parts) or aggregated_text

            # Last resort: try to get text from the stream object itself
            if not final_text and hasattr(stream, "text"):
                final_text = getattr(stream, "text") or ""

            if streamed_text:
                self._emit_streaming_chunk(
                    "",
                    source=_ANALYST_SOURCE,
                    on_chunk=on_chunk,
                    is_final=True,
                )

            duration = time.perf_counter() - started
            self._emit_status("Analyst agent: plan ready", "analyst.complete")
            self._event_bus.emit(
                PhaseTransition(from_phase="analyst", to_phase="idle", source=_ANALYST_SOURCE)
            )
            self._emit_completion(final_text, success=True)
            LOGGER.info(
                "Analyst analysis completed | duration=%.2fs | streamed=%s | text_length=%d",
                duration,
                streamed_text,
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
            error_message = (
                "Error: Unable to reach the analyst provider. Please verify "
                "GEMINI_API_KEY and network connectivity."
            )
            self._emit_status("Analyst agent: failed", "analyst.error")
            self._event_bus.emit(
                PhaseTransition(from_phase="analyst", to_phase="analyst.error", source=_ANALYST_SOURCE)
            )
            self._emit_completion(error_message, success=False)
            return error_message

    @staticmethod
    def _extract_stream_addition(
        chunk: types.GenerateContentResponse,
        aggregated_text: str,
    ) -> tuple[str, str]:
        """Return the new text produced by this chunk.

        This method robustly extracts text from API responses, even when they
        contain complex multi-part content (e.g., after tool calls).
        """
        text_snapshot = AnalystAgentService._coalesce_response_text(chunk)
        if not text_snapshot:
            return "", aggregated_text

        if aggregated_text and aggregated_text.startswith(text_snapshot):
            return "", aggregated_text

        if text_snapshot.startswith(aggregated_text):
            addition = text_snapshot[len(aggregated_text) :]
            return addition, text_snapshot

        return text_snapshot, aggregated_text + text_snapshot

    @staticmethod
    def _coalesce_response_text(response: Any) -> str:
        """Prefer text gathered from structured parts, fallback to plain text."""
        if response is None:
            return ""

        text_from_parts = AnalystAgentService._extract_text_from_parts(response)
        if text_from_parts:
            return text_from_parts

        fallback_text = AnalystAgentService._get_structured_field(response, "text")
        return fallback_text if isinstance(fallback_text, str) else ""

    @staticmethod
    def _extract_text_from_parts(chunk: types.GenerateContentResponse) -> str:
        """Extract text from all text parts in a complex response.

        When the API returns multi-part content (common after tool calls),
        the .text property may be empty. This method manually walks through
        all parts to extract text content.
        """
        text_parts: list[str] = []
        non_text_detected = False

        try:
            for part in AnalystAgentService._iter_response_parts(chunk):
                extracted = AnalystAgentService._extract_text_from_part(part)
                if extracted:
                    text_parts.append(extracted)
                    continue
                if AnalystAgentService._part_contains_non_text_payload(part):
                    non_text_detected = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Could not extract text from response parts: %s", exc)
            return ""

        if non_text_detected and not text_parts:
            LOGGER.debug(
                "Analyst stream chunk contained only non-text payload; ignoring."
            )

        return "".join(text_parts)

    @staticmethod
    def _iter_response_parts(
        chunk: types.GenerateContentResponse,
    ) -> Iterator[Any]:
        """Yield every content part in the response."""
        candidates = AnalystAgentService._get_structured_field(chunk, "candidates")
        for candidate in AnalystAgentService._iterate_structured_sequence(candidates):
            yield from AnalystAgentService._parts_from_candidate(candidate)

        content = AnalystAgentService._get_structured_field(chunk, "content")
        yield from AnalystAgentService._parts_from_content(content)

    @staticmethod
    def _parts_from_candidate(candidate: Any) -> Iterator[Any]:
        """Yield all parts associated with a candidate."""
        if not candidate:
            return
        content = AnalystAgentService._get_structured_field(candidate, "content")
        yield from AnalystAgentService._parts_from_content(content)

    @staticmethod
    def _parts_from_content(content: Any) -> Iterator[Any]:
        """Yield all parts from a content object or a sequence of contents."""
        if not content:
            return

        if AnalystAgentService._is_non_string_sequence(content):
            for entry in content:
                yield from AnalystAgentService._parts_from_content(entry)
            return

        content_parts = AnalystAgentService._get_structured_field(content, "parts")
        if not content_parts:
            return
        for part in AnalystAgentService._iterate_structured_sequence(content_parts):
            yield part

    @staticmethod
    def _extract_text_from_part(part: Any) -> str:
        """Return textual content from a part if present."""
        if isinstance(part, str):
            return part
        if isinstance(part, Mapping):
            text_value = part.get("text")
            return text_value if isinstance(text_value, str) else ""
        text_value = getattr(part, "text", None)
        return text_value if isinstance(text_value, str) else ""

    @staticmethod
    def _part_contains_non_text_payload(part: Any) -> bool:
        """Return True if the part carries structured data instead of text."""
        if isinstance(part, Mapping):
            keys = set(part.keys())
            keys.discard("text")
            keys.discard("thought")
            return bool(keys)
        structured_fields = (
            "function_call",
            "function_response",
            "inline_data",
            "file_data",
            "code_execution_result",
            "executable_code",
            "video_metadata",
        )
        return any(getattr(part, field, None) is not None for field in structured_fields)

    @staticmethod
    def _get_structured_field(source: Any, field: str) -> Any:
        """Return an attribute/field regardless of whether the source is obj or dict."""
        if source is None:
            return None
        if isinstance(source, Mapping):
            return source.get(field)
        return getattr(source, field, None)

    @staticmethod
    def _is_non_string_sequence(value: Any) -> bool:
        """Return True if value behaves like a sequence we can iterate safely."""
        if value is None:
            return False
        return isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray, Mapping),
        )

    @staticmethod
    def _iterate_structured_sequence(value: Any) -> Iterator[Any]:
        """Yield items from a sequence-like value while handling scalars."""
        if value is None:
            return
        if AnalystAgentService._is_non_string_sequence(value):
            for item in value:
                yield item
            return
        yield value

    def _instrument_tools(
        self,
        tools: Sequence[Callable[..., Any]],
        source: str,
    ) -> list[Callable[..., Any]]:
        """Wrap tool callables so they emit events and telemetry."""
        wrapped: list[Callable[..., Any]] = []
        for tool in tools:
            if callable(tool):
                wrapped.append(self._wrap_tool(tool, source))
        return wrapped

    def _wrap_tool(
        self,
        tool: Callable[..., Any],
        source: str,
    ) -> Callable[..., Any]:
        """Return a callable that emits ToolCall events around execution."""
        tool_name = getattr(tool, "__name__", tool.__class__.__name__)
        if not tool_name:
            tool_name = tool.__class__.__name__

        @wraps(tool)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            params = self._sanitize_parameters(args, kwargs)
            self._event_bus.emit(
                ToolCallStarted(tool_name=tool_name, parameters=params, source=source)
            )
            started = time.perf_counter()
            try:
                result = tool(*args, **kwargs)
            except Exception as exc:
                duration = time.perf_counter() - started
                self._event_bus.emit(
                    ToolCallCompleted(
                        tool_name=tool_name,
                        result=f"error: {exc}",
                        duration=duration,
                        source=source,
                    )
                )
                LOGGER.exception(
                    "Tool %s failed after %.2fs | params=%s", tool_name, duration, params
                )
                self._event_bus.emit(
                    ToolCallFailed(
                        tool_name=tool_name,
                        error=str(exc),
                        duration=duration,
                        source=source,
                        parameters=params,
                    )
                )
                raise

            duration = time.perf_counter() - started
            self._event_bus.emit(
                ToolCallCompleted(
                    tool_name=tool_name,
                    result=self._summarize_result(result),
                    duration=duration,
                    source=source,
                )
            )
            self._maybe_emit_file_operation(tool_name, args, kwargs, source)
            return result

        return wrapper

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

    def _summarize_result(self, result: Any) -> Any:
        """Summarize tool results for event payloads."""
        return self._safe_value(result, limit=320)

    def _maybe_emit_file_operation(
        self,
        tool_name: str,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        source: str,
    ) -> None:
        """Emit a FileOperation event for relevant tools."""
        if tool_name not in _FILE_TOOLS:
            return

        if tool_name == "read_multiple_files":
            paths = kwargs.get("paths") or (args[0] if args else None)
            if paths is None:
                return
            if isinstance(paths, (str, bytes)):
                paths_iterable = [paths]
            elif isinstance(paths, Sequence):
                paths_iterable = list(paths)
            else:
                return
            paths_list = [str(path) for path in paths_iterable]
            if not paths_list:
                return
            details = {"count": len(paths_list)}
            filepath = paths_list[0]
        else:
            filepath = self._extract_path(args, kwargs)
            details = None

        if not filepath:
            return

        self._event_bus.emit(
            FileOperation(
                operation=tool_name,
                filepath=str(filepath),
                details=details,
                source=source,
            )
        )

    @staticmethod
    def _extract_path(
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> str | None:
        """Best-effort extraction of a file path argument."""
        candidate = kwargs.get("path")
        if isinstance(candidate, str):
            return candidate
        if args:
            first = args[0]
            if isinstance(first, str):
                return first
        return None

    def _emit_streaming_chunk(
        self,
        text: str,
        *,
        source: str,
        on_chunk: Optional[Callable[[str], None]],
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
