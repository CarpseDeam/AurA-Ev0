"""Typed feedback events shared across Aura services and UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


_Params = Mapping[str, Any] | Sequence[Any] | str | None


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    """Emitted before a tool callable is executed."""

    tool_name: str
    parameters: _Params = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallCompleted:
    """Emitted after a tool callable finishes."""

    tool_name: str
    result: Any = None
    duration: float | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class StatusUpdate:
    """Describes a high-level status message and associated phase."""

    message: str
    phase: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    """Signals that a service moved from one named phase to another."""

    from_phase: str
    to_phase: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class FileOperation:
    """Represents a workspace file operation (read/write/delete)."""

    operation: str
    filepath: str
    details: Mapping[str, Any] | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class StreamingChunk:
    """Encapsulates a streamed piece of text from a provider."""

    text: str
    source: str | None = None
    is_final: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionComplete:
    """Summary emitted when a service finishes executing."""

    summary: str
    source: str | None = None
    success: bool | None = None


FeedbackEvent = (
    ToolCallStarted
    | ToolCallCompleted
    | StatusUpdate
    | PhaseTransition
    | FileOperation
    | StreamingChunk
    | ExecutionComplete
)

__all__ = [
    "ExecutionComplete",
    "FeedbackEvent",
    "FileOperation",
    "PhaseTransition",
    "StatusUpdate",
    "StreamingChunk",
    "ToolCallCompleted",
    "ToolCallStarted",
]
