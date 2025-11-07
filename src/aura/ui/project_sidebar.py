"""
Project sidebar UI component for displaying projects and recent conversations.
"""

import logging
from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QScrollArea
)
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QFont

from ..models import Project, Conversation
from ..config import COLORS

logger = logging.getLogger(__name__)


class ProjectSidebar(QWidget):
    """
    Left sidebar showing projects and recent conversations.

    Signals:
        project_selected: Emitted when a project is clicked (project_id)
        conversation_selected: Emitted when a conversation is clicked (conversation_id)
        new_project_clicked: Emitted when "New Project" button is clicked
        new_conversation_clicked: Emitted when "New Chat" button is clicked
    """

    project_selected = Signal(int)  # project_id
    conversation_selected = Signal(int)  # conversation_id
    new_project_clicked = Signal()
    new_conversation_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_project_id: Optional[int] = None
        self._current_conversation_id: Optional[int] = None
        self._setup_ui()
        self._apply_styling()

    def _setup_ui(self) -> None:
        """Create and layout UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Projects section
        projects_section = self._create_projects_section()
        layout.addWidget(projects_section)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Recent conversations section
        recents_section = self._create_recents_section()
        layout.addWidget(recents_section)

    def _create_header(self) -> QWidget:
        """Create the sidebar header with title."""
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("AURA")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(title)

        return header

    def _create_projects_section(self) -> QWidget:
        """Create the projects section with list and new button."""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Section header with "New Project" button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        section_label = QLabel("Projects")
        section_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        header_layout.addWidget(section_label)

        header_layout.addStretch()

        new_project_btn = QPushButton("+ New")
        new_project_btn.setMaximumWidth(60)
        new_project_btn.clicked.connect(self.new_project_clicked.emit)
        header_layout.addWidget(new_project_btn)

        layout.addLayout(header_layout)

        # Projects list
        self._projects_list = QListWidget()
        self._projects_list.setMaximumHeight(200)
        self._projects_list.itemClicked.connect(self._on_project_clicked)
        layout.addWidget(self._projects_list)

        return section

    def _create_recents_section(self) -> QWidget:
        """Create the recent conversations section with list and new button."""
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Section header with "New Chat" button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        section_label = QLabel("Recent Conversations")
        section_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        header_layout.addWidget(section_label)

        header_layout.addStretch()

        new_chat_btn = QPushButton("+ New")
        new_chat_btn.setMaximumWidth(60)
        new_chat_btn.clicked.connect(self.new_conversation_clicked.emit)
        header_layout.addWidget(new_chat_btn)

        layout.addLayout(header_layout)

        # Recent conversations list
        self._recents_list = QListWidget()
        self._recents_list.itemClicked.connect(self._on_conversation_clicked)
        layout.addWidget(self._recents_list, 1)  # Give it stretch factor

        return section

    def _on_project_clicked(self, item: QListWidgetItem) -> None:
        """Handle project list item click."""
        project_id = item.data(Qt.UserRole)
        if project_id:
            self._current_project_id = project_id
            self.project_selected.emit(project_id)

    def _on_conversation_clicked(self, item: QListWidgetItem) -> None:
        """Handle conversation list item click."""
        conversation_id = item.data(Qt.UserRole)
        if conversation_id:
            self._current_conversation_id = conversation_id
            self.conversation_selected.emit(conversation_id)

    def refresh_projects(self) -> None:
        """Reload and display all projects."""
        self._projects_list.clear()

        try:
            projects = Project.get_all(order_by="updated_at", ascending=False)

            for project in projects:
                item = QListWidgetItem(project.name)
                item.setData(Qt.UserRole, project.id)

                # Highlight current project
                if project.id == self._current_project_id:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self._projects_list.addItem(item)

            logger.info(f"Refreshed projects list: {len(projects)} projects")

        except Exception as e:
            logger.error(f"Failed to refresh projects: {e}")

    def refresh_recent_conversations(self, limit: int = 20) -> None:
        """Reload and display recent conversations."""
        self._recents_list.clear()

        try:
            conversations = Conversation.get_recent(limit=limit)

            for conv in conversations:
                # Display title or "(Untitled)"
                title = conv.title if conv.title else "(Untitled)"

                # Add project name if available
                if conv.project_id:
                    try:
                        project = Project.get_by_id(conv.project_id)
                        if project:
                            title = f"{project.name} • {title}"
                    except Exception:
                        pass

                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, conv.id)

                # Highlight current conversation
                if conv.id == self._current_conversation_id:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)

                self._recents_list.addItem(item)

            logger.info(f"Refreshed recent conversations: {len(conversations)} conversations")

        except Exception as e:
            logger.error(f"Failed to refresh recent conversations: {e}")

    def refresh_all(self) -> None:
        """Refresh both projects and recent conversations."""
        self.refresh_projects()
        self.refresh_recent_conversations()

    def set_current_project(self, project_id: Optional[int]) -> None:
        """
        Set the currently selected project.

        Args:
            project_id: Project ID to mark as current
        """
        self._current_project_id = project_id
        self.refresh_projects()

    def set_current_conversation(self, conversation_id: Optional[int]) -> None:
        """
        Set the currently selected conversation.

        Args:
            conversation_id: Conversation ID to mark as current
        """
        self._current_conversation_id = conversation_id
        self.refresh_recent_conversations()

    def _apply_styling(self) -> None:
        """Apply CSS styling to the sidebar."""
        # Sidebar background slightly lighter than main background
        sidebar_bg = "#161b22"
        hover_bg = "#21262d"

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {sidebar_bg};
                color: {COLORS.text};
            }}

            QLabel {{
                color: {COLORS.text};
            }}

            QPushButton {{
                background-color: {COLORS.accent};
                color: {COLORS.background};
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }}

            QPushButton:hover {{
                background-color: {COLORS.prompt};
            }}

            QPushButton:pressed {{
                background-color: {COLORS.accent};
            }}

            QListWidget {{
                background-color: {COLORS.background};
                border: 1px solid {COLORS.border};
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }}

            QListWidget::item {{
                padding: 8px;
                border-radius: 4px;
                color: {COLORS.text};
            }}

            QListWidget::item:hover {{
                background-color: {hover_bg};
            }}

            QListWidget::item:selected {{
                background-color: {COLORS.accent};
                color: {COLORS.background};
            }}

            QFrame {{
                color: {COLORS.border};
            }}
        """)

    def sizeHint(self) -> QSize:
        """Provide a reasonable default size for the sidebar."""
        return QSize(280, 600)
