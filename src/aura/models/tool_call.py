"""Persistence helpers for tracking individual tool executions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from aura.database import get_connection


@dataclass(slots=True)
class ToolCallLog:
    """Represents a single tool execution for auditing."""

    id: Optional[int]
    conversation_id: Optional[int]
    agent_role: str
    tool_name: str
    tool_input: str | None
    tool_output: str | None
    success: bool
    error_message: str | None
    execution_time_ms: float | None
    created_at: Optional[datetime] = None

    @staticmethod
    def record(
        *,
        conversation_id: Optional[int],
        agent_role: str,
        tool_name: str,
        tool_input: str | None,
        tool_output: str | None,
        success: bool,
        error_message: str | None,
        execution_time_ms: float | None,
    ) -> "ToolCallLog":
        """Persist a tool call entry and return the stored record."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tool_calls (
                    conversation_id,
                    agent_role,
                    tool_name,
                    tool_input,
                    tool_output,
                    success,
                    error_message,
                    execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    agent_role,
                    tool_name,
                    tool_input,
                    tool_output,
                    1 if success else 0,
                    error_message,
                    execution_time_ms,
                ),
            )
            tool_call_id = cursor.lastrowid
            conn.commit()
        return ToolCallLog.get_by_id(tool_call_id)  # type: ignore[arg-type]

    @staticmethod
    def get_by_id(identifier: int) -> Optional["ToolCallLog"]:
        """Load a tool call entry by primary key."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    agent_role,
                    tool_name,
                    tool_input,
                    tool_output,
                    success,
                    error_message,
                    execution_time_ms,
                    created_at
                FROM tool_calls
                WHERE id = ?
                """,
                (identifier,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return ToolCallLog._from_row(row)

    @staticmethod
    def _from_row(row: Any) -> "ToolCallLog":
        """Convert a sqlite row into a ToolCallLog."""
        created = row["created_at"]
        created_at = datetime.fromisoformat(created) if created else None
        return ToolCallLog(
            id=row["id"],
            conversation_id=row["conversation_id"],
            agent_role=row["agent_role"],
            tool_name=row["tool_name"],
            tool_input=row["tool_input"],
            tool_output=row["tool_output"],
            success=bool(row["success"]),
            error_message=row["error_message"],
            execution_time_ms=row["execution_time_ms"],
            created_at=created_at,
        )


__all__ = ["ToolCallLog"]
