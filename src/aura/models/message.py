"""
Message model for managing individual messages and CRUD operations.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..database import get_connection

logger = logging.getLogger(__name__)


class MessageRole:
    """Constants for supported message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    _ALL = frozenset({USER, ASSISTANT, TOOL_RESULT})

    @classmethod
    def validate(cls, role: str) -> None:
        """Ensure the provided role is supported."""
        if role not in cls._ALL:
            allowed = ", ".join(sorted(cls._ALL))
            raise ValueError(f"Invalid role: {role}. Must be one of: {allowed}")


ALLOWED_MESSAGE_ORDER_COLUMNS = frozenset({
    "id",
    "conversation_id",
    "role",
    "content",
    "created_at",
})


@dataclass
class Message:
    """
    Represents a single message in a conversation.

    Messages have a role (user or assistant) and content.

    Attributes:
        id: Unique message identifier (None for new messages)
        conversation_id: ID of the parent conversation
        role: Message role ('user' or 'assistant')
        content: Message content/text
        created_at: Timestamp of creation
    """

    id: Optional[int]
    conversation_id: int
    role: str
    content: str
    created_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate role after initialization."""
        MessageRole.validate(self.role)

    @staticmethod
    def create(
        conversation_id: int,
        role: str,
        content: str
    ) -> 'Message':
        """
        Create a new message in the database.

        Args:
            conversation_id: Parent conversation ID
            role: Message role ('user' or 'assistant')
            content: Message content

        Returns:
            Message instance with assigned ID

        Raises:
            ValueError: If role is invalid
            sqlite3.Error: If database operation fails
        """
        MessageRole.validate(role)

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
            """, (conversation_id, role, content))

            message_id = cursor.lastrowid

            # Update conversation's updated_at timestamp
            cursor.execute("""
                UPDATE conversations
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))

            conn.commit()

        logger.info(f"Created {role} message ID {message_id} in conversation {conversation_id}")

        return Message.get_by_id(message_id)

    @staticmethod
    def get_by_id(message_id: int) -> Optional['Message']:
        """
        Retrieve a message by ID.

        Args:
            message_id: The message ID to retrieve

        Returns:
            Message instance if found, None otherwise
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE id = ?
            """, (message_id,))

            row = cursor.fetchone()

        if row:
            return Message._from_row(row)
        return None

    @staticmethod
    def get_by_conversation(
        conversation_id: int,
        order_by: str = "created_at",
        ascending: bool = True
    ) -> List['Message']:
        """
        Retrieve all messages for a conversation.

        Args:
            conversation_id: The conversation ID to filter by
            order_by: Column to sort by (default: "created_at")
            ascending: Sort order (default: True for chronological)

        Returns:
            List of Message instances
        """
        if order_by not in ALLOWED_MESSAGE_ORDER_COLUMNS:
            allowed = ", ".join(sorted(ALLOWED_MESSAGE_ORDER_COLUMNS))
            raise ValueError(f"Invalid order_by column: {order_by}. Must be one of: {allowed}")

        order = "ASC" if ascending else "DESC"
        query = f"""
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY {order_by} {order}
        """

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (conversation_id,))
            rows = cursor.fetchall()

        return [Message._from_row(row) for row in rows]

    def delete(self) -> None:
        """
        Delete this message from the database.

        Raises:
            ValueError: If message has no ID (not saved to database)
        """
        if self.id is None:
            raise ValueError("Cannot delete message without ID")

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE id = ?", (self.id,))
            conn.commit()

        logger.info(f"Deleted message ID {self.id}")

    @staticmethod
    def _from_row(row: Any) -> 'Message':
        """
        Create a Message instance from a database row.

        Args:
            row: sqlite3.Row object from query result

        Returns:
            Message instance
        """
        return Message(
            id=row['id'],
            conversation_id=row['conversation_id'],
            role=row['role'],
            content=row['content'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary representation.

        Returns:
            Dict with all message fields
        """
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
