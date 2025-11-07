"""
Conversation model for managing conversation threads and CRUD operations.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from ..database import get_connection

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    """
    Represents a conversation thread in Aura.

    A conversation groups related messages and belongs to a project.

    Attributes:
        id: Unique conversation identifier (None for new conversations)
        project_id: ID of the parent project (None for standalone conversations)
        title: Conversation title (auto-generated from first message)
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
    """

    id: Optional[int]
    project_id: Optional[int]
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @staticmethod
    def create(
        project_id: Optional[int] = None,
        title: Optional[str] = None
    ) -> 'Conversation':
        """
        Create a new conversation in the database.

        Args:
            project_id: Optional parent project ID
            title: Optional conversation title (can be set later)

        Returns:
            Conversation instance with assigned ID

        Raises:
            sqlite3.Error: If database operation fails
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (project_id, title)
                VALUES (?, ?)
            """, (project_id, title))

            conn.commit()
            conversation_id = cursor.lastrowid

        logger.info(f"Created conversation ID {conversation_id} in project {project_id}")

        return Conversation.get_by_id(conversation_id)

    @staticmethod
    def get_by_id(conversation_id: int) -> Optional['Conversation']:
        """
        Retrieve a conversation by ID.

        Args:
            conversation_id: The conversation ID to retrieve

        Returns:
            Conversation instance if found, None otherwise
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, title, created_at, updated_at
                FROM conversations
                WHERE id = ?
            """, (conversation_id,))

            row = cursor.fetchone()

        if row:
            return Conversation._from_row(row)
        return None

    @staticmethod
    def get_by_project(
        project_id: int,
        order_by: str = "updated_at",
        ascending: bool = False,
        limit: Optional[int] = None
    ) -> List['Conversation']:
        """
        Retrieve all conversations for a project.

        Args:
            project_id: The project ID to filter by
            order_by: Column to sort by (default: "updated_at")
            ascending: Sort order (default: False for descending)
            limit: Optional limit on number of results

        Returns:
            List of Conversation instances
        """
        order = "ASC" if ascending else "DESC"
        query = f"""
            SELECT id, project_id, title, created_at, updated_at
            FROM conversations
            WHERE project_id = ?
            ORDER BY {order_by} {order}
        """

        if limit:
            query += f" LIMIT {limit}"

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (project_id,))
            rows = cursor.fetchall()

        return [Conversation._from_row(row) for row in rows]

    @staticmethod
    def get_recent(limit: int = 20) -> List['Conversation']:
        """
        Get most recently updated conversations across all projects.

        Args:
            limit: Maximum number of conversations to return (default: 20)

        Returns:
            List of Conversation instances ordered by updated_at DESC
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, project_id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
            """, (limit,))

            rows = cursor.fetchall()

        return [Conversation._from_row(row) for row in rows]

    @staticmethod
    def get_most_recent() -> Optional['Conversation']:
        """
        Get the most recently updated conversation.

        Returns:
            Most recent Conversation instance, or None if no conversations exist
        """
        recent = Conversation.get_recent(limit=1)
        return recent[0] if recent else None

    def update(
        self,
        title: Optional[str] = None,
        project_id: Optional[int] = None
    ) -> None:
        """
        Update conversation fields.

        Only provided fields will be updated. Pass None to leave unchanged.

        Args:
            title: New conversation title
            project_id: New project ID (to move conversation)

        Raises:
            ValueError: If conversation has no ID (not saved to database)
        """
        if self.id is None:
            raise ValueError("Cannot update conversation without ID")

        updates = []
        values = []

        if title is not None:
            updates.append("title = ?")
            values.append(title)
            self.title = title

        if project_id is not None:
            updates.append("project_id = ?")
            values.append(project_id)
            self.project_id = project_id

        if not updates:
            return  # Nothing to update

        # Always update the updated_at timestamp
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(self.id)

        query = f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?"

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()

        logger.info(f"Updated conversation ID {self.id}")

        # Refresh from database to get updated timestamp
        updated = Conversation.get_by_id(self.id)
        if updated:
            self.updated_at = updated.updated_at

    def delete(self) -> None:
        """
        Delete this conversation from the database.

        This will CASCADE delete all associated messages.

        Raises:
            ValueError: If conversation has no ID (not saved to database)
        """
        if self.id is None:
            raise ValueError("Cannot delete conversation without ID")

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM conversations WHERE id = ?", (self.id,))
            conn.commit()

        logger.info(f"Deleted conversation ID {self.id}")

    def get_messages(self) -> List['Message']:
        """
        Get all messages in this conversation.

        Returns:
            List of Message instances ordered by created_at

        Raises:
            ValueError: If conversation has no ID
        """
        if self.id is None:
            raise ValueError("Cannot get messages for conversation without ID")

        from .message import Message
        return Message.get_by_conversation(self.id)

    def get_history(self) -> List[Tuple[str, str]]:
        """
        Get conversation history as (role, content) tuples.

        This format matches the orchestrator's in-memory history format.

        Returns:
            List of (role, content) tuples

        Raises:
            ValueError: If conversation has no ID
        """
        if self.id is None:
            raise ValueError("Cannot get history for conversation without ID")

        messages = self.get_messages()
        return [(msg.role, msg.content) for msg in messages]

    def get_message_count(self) -> int:
        """
        Get the number of messages in this conversation.

        Returns:
            Message count

        Raises:
            ValueError: If conversation has no ID
        """
        if self.id is None:
            raise ValueError("Cannot count messages for conversation without ID")

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM messages
                WHERE conversation_id = ?
            """, (self.id,))

            count = cursor.fetchone()[0]

        return count

    def generate_title_from_first_message(self) -> None:
        """
        Auto-generate a title from the first user message.

        Takes the first 50 characters of the first message.
        Updates the conversation in the database if a title is generated.

        Raises:
            ValueError: If conversation has no ID
        """
        if self.id is None:
            raise ValueError("Cannot generate title for conversation without ID")

        if self.title:
            return  # Already has a title

        messages = self.get_messages()
        if not messages:
            return  # No messages yet

        # Find first user message
        first_user_message = next((msg for msg in messages if msg.role == 'user'), None)
        if not first_user_message:
            return

        # Generate title from first 50 chars
        title = first_user_message.content[:50]
        if len(first_user_message.content) > 50:
            title += "..."

        self.update(title=title)

    @staticmethod
    def _from_row(row: Any) -> 'Conversation':
        """
        Create a Conversation instance from a database row.

        Args:
            row: sqlite3.Row object from query result

        Returns:
            Conversation instance
        """
        return Conversation(
            id=row['id'],
            project_id=row['project_id'],
            title=row['title'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary representation.

        Returns:
            Dict with all conversation fields
        """
        return {
            'id': self.id,
            'project_id': self.project_id,
            'title': self.title,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
