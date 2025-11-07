"""
Project model for managing project metadata and CRUD operations.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..database import get_connection

logger = logging.getLogger(__name__)

ALLOWED_PROJECT_ORDER_COLUMNS = frozenset({
    "id",
    "name",
    "description",
    "working_directory",
    "created_at",
    "updated_at",
    "custom_instructions",
    "settings",
})


@dataclass
class Project:
    """
    Represents a project in Aura.

    A project groups related conversations and maintains shared context like
    working directory, custom instructions, and model settings.

    Attributes:
        id: Unique project identifier (None for new projects)
        name: Project name (e.g., "Aura Development")
        description: Optional project description
        working_directory: Path to project's working directory
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
        custom_instructions: Optional additional system prompt text
        settings: Dict of project-specific settings (model selections, etc.)
    """

    id: Optional[int]
    name: str
    description: Optional[str] = None
    working_directory: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    custom_instructions: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

    @staticmethod
    def create(
        name: str,
        description: Optional[str] = None,
        working_directory: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> 'Project':
        """
        Create a new project in the database.

        Args:
            name: Project name (required)
            description: Optional project description
            working_directory: Optional working directory path
            custom_instructions: Optional custom system prompt additions
            settings: Optional dict of project-specific settings

        Returns:
            Project instance with assigned ID

        Raises:
            sqlite3.Error: If database operation fails
        """
        settings_json = json.dumps(settings) if settings else None

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO projects (name, description, working_directory, custom_instructions, settings)
                VALUES (?, ?, ?, ?, ?)
            """, (name, description, working_directory, custom_instructions, settings_json))

            conn.commit()
            project_id = cursor.lastrowid

        logger.info(f"Created project: {name} (ID: {project_id})")

        return Project.get_by_id(project_id)

    @staticmethod
    def get_by_id(project_id: int) -> Optional['Project']:
        """
        Retrieve a project by ID.

        Args:
            project_id: The project ID to retrieve

        Returns:
            Project instance if found, None otherwise
        """
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, name, description, working_directory,
                       created_at, updated_at, custom_instructions, settings
                FROM projects
                WHERE id = ?
            """, (project_id,))

            row = cursor.fetchone()

        if row:
            return Project._from_row(row)
        return None

    @staticmethod
    def get_all(order_by: str = "updated_at", ascending: bool = False) -> List['Project']:
        """
        Retrieve all projects.

        Args:
            order_by: Column to sort by (default: "updated_at")
            ascending: Sort order (default: False for descending)

        Returns:
            List of Project instances
        """
        if order_by not in ALLOWED_PROJECT_ORDER_COLUMNS:
            allowed = ", ".join(sorted(ALLOWED_PROJECT_ORDER_COLUMNS))
            raise ValueError(f"Invalid order_by column: {order_by}. Must be one of: {allowed}")

        order = "ASC" if ascending else "DESC"
        query = f"""
            SELECT id, name, description, working_directory,
                   created_at, updated_at, custom_instructions, settings
            FROM projects
            ORDER BY {order_by} {order}
        """

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

        return [Project._from_row(row) for row in rows]

    def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        working_directory: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update project fields.

        Only provided fields will be updated. Pass None to leave unchanged.

        Args:
            name: New project name
            description: New description
            working_directory: New working directory
            custom_instructions: New custom instructions
            settings: New settings dict

        Raises:
            ValueError: If project has no ID (not saved to database)
        """
        if self.id is None:
            raise ValueError("Cannot update project without ID")

        # Build dynamic UPDATE query for changed fields
        updates = []
        values = []

        if name is not None:
            updates.append("name = ?")
            values.append(name)
            self.name = name

        if description is not None:
            updates.append("description = ?")
            values.append(description)
            self.description = description

        if working_directory is not None:
            updates.append("working_directory = ?")
            values.append(working_directory)
            self.working_directory = working_directory

        if custom_instructions is not None:
            updates.append("custom_instructions = ?")
            values.append(custom_instructions)
            self.custom_instructions = custom_instructions

        if settings is not None:
            updates.append("settings = ?")
            values.append(json.dumps(settings))
            self.settings = settings

        if not updates:
            return  # Nothing to update

        # Always update the updated_at timestamp
        updates.append("updated_at = CURRENT_TIMESTAMP")
        values.append(self.id)

        query = f"UPDATE projects SET {', '.join(updates)} WHERE id = ?"

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()

        logger.info(f"Updated project ID {self.id}")

        # Refresh from database to get updated timestamp
        updated = Project.get_by_id(self.id)
        if updated:
            self.updated_at = updated.updated_at

    def delete(self) -> None:
        """
        Delete this project from the database.

        This will CASCADE delete all associated conversations and messages.

        Raises:
            ValueError: If project has no ID (not saved to database)
        """
        if self.id is None:
            raise ValueError("Cannot delete project without ID")

        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE id = ?", (self.id,))
            conn.commit()

        logger.info(f"Deleted project ID {self.id}")

    def get_conversations(self) -> List['Conversation']:
        """
        Get all conversations for this project.

        Returns:
            List of Conversation instances

        Raises:
            ValueError: If project has no ID
        """
        if self.id is None:
            raise ValueError("Cannot get conversations for project without ID")

        from .conversation import Conversation
        return Conversation.get_by_project(self.id)

    @staticmethod
    def _from_row(row: Any) -> 'Project':
        """
        Create a Project instance from a database row.

        Args:
            row: sqlite3.Row object from query result

        Returns:
            Project instance
        """
        settings = json.loads(row['settings']) if row['settings'] else None

        return Project(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            working_directory=row['working_directory'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
            custom_instructions=row['custom_instructions'],
            settings=settings
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert project to dictionary representation.

        Returns:
            Dict with all project fields
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'working_directory': self.working_directory,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'custom_instructions': self.custom_instructions,
            'settings': self.settings
        }
