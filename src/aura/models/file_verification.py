"""Persistence helpers for file write verification events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from aura.database import get_connection


@dataclass(slots=True)
class FileVerificationLog:
    """Represents a single file verification attempt."""

    id: Optional[int]
    conversation_id: Optional[int]
    phase: str
    operation: str
    file_path: str
    expected_digest: str | None
    actual_digest: str | None
    success: bool
    details: str | None
    created_at: Optional[datetime] = None

    @staticmethod
    def record(
        *,
        phase: str,
        operation: str,
        file_path: str,
        expected_digest: str | None,
        actual_digest: str | None,
        success: bool,
        details: str | None = None,
        conversation_id: Optional[int] = None,
    ) -> "FileVerificationLog":
        """Persist a verification entry and return the stored record."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO file_write_verifications (
                    conversation_id,
                    phase,
                    operation,
                    file_path,
                    expected_digest,
                    actual_digest,
                    success,
                    details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    phase,
                    operation,
                    file_path,
                    expected_digest,
                    actual_digest,
                    1 if success else 0,
                    details,
                ),
            )
            verification_id = cursor.lastrowid
            conn.commit()
        return FileVerificationLog.get_by_id(verification_id)  # type: ignore[arg-type]

    @staticmethod
    def get_by_id(identifier: int) -> Optional["FileVerificationLog"]:
        """Load a verification entry by id."""
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    conversation_id,
                    phase,
                    operation,
                    file_path,
                    expected_digest,
                    actual_digest,
                    success,
                    details,
                    created_at
                FROM file_write_verifications
                WHERE id = ?
                """,
                (identifier,),
            )
            row = cursor.fetchone()
        if not row:
            return None
        return FileVerificationLog._from_row(row)

    @staticmethod
    def _from_row(row: Any) -> "FileVerificationLog":
        """Convert a sqlite row into a FileVerificationLog."""
        created = row["created_at"]
        created_at = datetime.fromisoformat(created) if created else None
        return FileVerificationLog(
            id=row["id"],
            conversation_id=row["conversation_id"],
            phase=row["phase"],
            operation=row["operation"],
            file_path=row["file_path"],
            expected_digest=row["expected_digest"],
            actual_digest=row["actual_digest"],
            success=bool(row["success"]),
            details=row["details"],
            created_at=created_at,
        )


__all__ = ["FileVerificationLog"]
