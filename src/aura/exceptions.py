"""Aura-specific exception hierarchy with structured context support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping


@dataclass(slots=True)
class AuraError(Exception):
    """Base class for all Aura exceptions with optional context metadata."""

    message: str
    context: MutableMapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize context mapping."""
        if not isinstance(self.context, Mapping):
            self.context = {"detail": str(self.context)}
        else:
            self.context = dict(self.context)
        Exception.__init__(self, self.__str__())

    def __str__(self) -> str:
        """Include context metadata in the string representation."""
        if self.context:
            context_parts = ", ".join(f"{key}={value}" for key, value in self.context.items())
            return f"{self.message} ({context_parts})"
        return self.message


class AuraConfigurationError(AuraError):
    """Raised when Aura configuration or environment is invalid."""


class AuraExecutionError(AuraError):
    """Raised for execution/runtime failures within Aura."""


class AuraValidationError(AuraError):
    """Raised when user or system inputs fail validation."""


class AuraToolError(AuraError):
    """Raised when a developer tool invocation fails."""


class FileVerificationError(AuraToolError):
    """Raised when filesystem verification fails or content is missing."""


__all__ = [
    "AuraError",
    "AuraConfigurationError",
    "AuraExecutionError",
    "AuraValidationError",
    "AuraToolError",
    "FileVerificationError",
]
