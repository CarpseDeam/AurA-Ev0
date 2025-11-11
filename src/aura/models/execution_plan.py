"""Structured execution plan contract shared between analyst and executor."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class OperationType(str, Enum):
    """Supported file operations that the executor can perform."""

    CREATE = "CREATE"
    MODIFY = "MODIFY"
    DELETE = "DELETE"


class FileOperation(BaseModel):
    """Single file operation with strict validation rules."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    operation_type: OperationType = Field(alias="operation_type")
    file_path: str = Field(alias="file_path")
    content: str | None = Field(
        default=None,
        description="Full file contents after this operation (required for CREATE/MODIFY).",
    )
    old_str: str | None = Field(default=None, alias="old_str")
    new_str: str | None = Field(default=None, alias="new_str")
    rationale: str = Field(
        description="Why this change is required and how it satisfies the plan.",
    )
    dependencies: list[str] = Field(default_factory=list)

    @field_validator("file_path")
    @classmethod
    def _validate_file_path(cls, value: str) -> str:
        """Ensure file paths are normalized and non-empty."""
        if not value:
            raise ValueError("file_path is required")
        normalized = value.replace("\\", "/")
        if normalized.startswith(("/", "./")):
            normalized = normalized.lstrip("./")
        if not normalized:
            raise ValueError("file_path cannot resolve to empty")
        if normalized.endswith("/"):
            raise ValueError("file_path must reference a file, not a directory")
        if "\n" in normalized:
            raise ValueError("file_path cannot contain newline characters")
        return normalized

    @field_validator("dependencies", mode="before")
    @classmethod
    def _ensure_dependency_list(cls, value: Any) -> list[str]:
        """Normalize dependency payloads to a simple list."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @field_validator("rationale")
    @classmethod
    def _require_rationale(cls, value: str) -> str:
        """Rationales are mandatory to keep plans auditable."""
        if not value:
            raise ValueError("rationale is required")
        return value.strip()

    @model_validator(mode="after")
    def _validate_payload(self) -> "FileOperation":
        """Enforce operation-specific payload requirements."""
        if self.operation_type is OperationType.CREATE:
            if not self.content:
                raise ValueError("CREATE operations must include file content.")
        elif self.operation_type is OperationType.MODIFY:
            if not self.content:
                raise ValueError("MODIFY operations must include full file content in 'content'.")
            if not self.old_str or not self.new_str:
                raise ValueError("MODIFY operations require old_str and new_str fields.")
        elif self.operation_type is OperationType.DELETE:
            if self.content or self.new_str or self.old_str:
                raise ValueError("DELETE operations should not specify content/old/new strings.")
        return self

    def describe(self) -> str:
        """Return a compact description for logging."""
        return f"{self.operation_type}:{self.file_path}"


class ExecutionPlan(BaseModel):
    """Structured plan emitted by the analyst and consumed by the executor."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    task_summary: str = Field(description="Single sentence describing the requested change.")
    project_context: str = Field(description="Concise repository context and constraints.")
    operations: list[FileOperation] = Field(default_factory=list)
    quality_checklist: list[str] = Field(default_factory=list)
    estimated_files: int = Field(ge=0, description="Rough number of impacted files.")
    is_emergency: bool = Field(default=False, description="Indicates this is an emergency fallback plan requiring manual review")

    @field_validator("task_summary", "project_context")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        """Ensure core narrative fields are populated."""
        value = (value or "").strip()
        if not value:
            raise ValueError("field cannot be empty")
        return value

    @field_validator("quality_checklist", mode="before")
    @classmethod
    def _normalize_checklist(cls, value: Any) -> list[str]:
        """Convert arbitrary checklists into a clean list of bullet items."""
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @model_validator(mode="after")
    def _ensure_operations_present(self) -> "ExecutionPlan":
        """Plans must include at least one actionable operation."""
        if not self.operations:
            raise ValueError("ExecutionPlan operations must contain at least one item.")
        return self

    def to_json(self, *, indent: int | None = None) -> str:
        """Return a JSON string representation of the plan."""
        return self.model_dump_json(by_alias=True, indent=indent)

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "ExecutionPlan":
        """Parse an ExecutionPlan from a JSON payload."""
        return cls.model_validate_json(payload)


__all__ = ["ExecutionPlan", "FileOperation", "OperationType", "ValidationError"]
