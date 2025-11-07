"""
Database module for managing projects, conversations, and messages.

This module provides SQLite database initialization, connection management,
and schema creation for Aura's project organization system.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def get_database_path() -> Path:
    """
    Get the path to the Aura database file.

    Returns:
        Path object pointing to ~/.aura/conversations.db
    """
    aura_dir = Path.home() / ".aura"
    aura_dir.mkdir(exist_ok=True)
    return aura_dir / "conversations.db"


@contextmanager
def get_connection():
    """
    Context manager for database connections.

    Yields:
        sqlite3.Connection: Database connection with row factory enabled

    Example:
        >>> with get_connection() as conn:
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM projects")
    """
    db_path = get_database_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Enable accessing columns by name
    conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
    try:
        yield conn
    finally:
        conn.close()


def initialize_database() -> None:
    """
    Initialize the database schema.

    Creates all required tables if they don't exist:
    - projects: Project metadata and settings
    - conversations: Conversation threads within projects
    - messages: Individual messages in conversations
    - project_files: Files attached to projects (for future use)

    This function is idempotent and safe to call multiple times.
    """
    db_path = get_database_path()

    logger.info(f"Initializing database at {db_path}")

    with get_connection() as conn:
        cursor = conn.cursor()

        # Create projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                working_directory TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                custom_instructions TEXT,
                settings TEXT
            )
        """)

        # Create conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        # Create messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)

        # Create project_files table (for future use)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_content TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_project_id
            ON conversations(project_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
            ON conversations(updated_at DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
            ON messages(conversation_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_created_at
            ON messages(created_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_project_files_project_id
            ON project_files(project_id)
        """)

        conn.commit()
        logger.info("Database schema initialized successfully")


def reset_database() -> None:
    """
    Drop all tables and reinitialize the database.

    WARNING: This will delete all data! Use only for testing or fresh starts.
    """
    logger.warning("Resetting database - all data will be lost!")

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS project_files")
        cursor.execute("DROP TABLE IF EXISTS messages")
        cursor.execute("DROP TABLE IF EXISTS conversations")
        cursor.execute("DROP TABLE IF EXISTS projects")

        conn.commit()

    initialize_database()
    logger.info("Database reset complete")


def check_database_health() -> bool:
    """
    Check if the database is accessible and has the expected schema.

    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            # Check that all required tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name IN ('projects', 'conversations', 'messages', 'project_files')
            """)

            tables = [row[0] for row in cursor.fetchall()]
            expected_tables = {'projects', 'conversations', 'messages', 'project_files'}

            if set(tables) == expected_tables:
                logger.info("Database health check passed")
                return True
            else:
                missing = expected_tables - set(tables)
                logger.error(f"Database health check failed. Missing tables: {missing}")
                return False

    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


if __name__ == "__main__":
    # Allow running this module directly to initialize database
    logging.basicConfig(level=logging.INFO)
    initialize_database()
    check_database_health()
