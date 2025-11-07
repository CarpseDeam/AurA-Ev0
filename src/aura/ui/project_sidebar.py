"""
Project sidebar UI component for displaying projects and recent conversations.
"""

import logging
from typing import Optional, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFrame, QScrollArea
)
from PySide6.QtCore import Signal, Qt, QSize, QPropertyAnimation, QEasingCurve
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
        state_changed: Emitted when the sidebar is collapsed or expanded (bool)
    """

    project_selected = Signal(int)  # project_id
    conversation_selected = Signal(int)  # conversation_id
    new_project_clicked = Signal()
    new_conversation_clicked = Signal()
    state_changed = Signal(bool)  # collapsed

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_project_id: Optional[int] = None
        self._current_conversation_id: Optional[int] = None
        self._collapsed = False
        self._collapsed_width = 40
        self._expanded_width = 280

        # Animation for smooth collapse/expand
        self._animation = QPropertyAnimation(self, b"minimumWidth")
        self._animation.setDuration(250)
        self._animation.setEasingCurve(QEasingCurve.InOutQuad)

        self._setup_ui()
        self._apply_styling()
        self.setMinimumWidth(self._expanded_width)

    def _setup_ui(self) -> None:
        """Create and layout UI components."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Header
        header = self._create_header()
        self.main_layout.addWidget(header)

        # Main content area
        self._content_widget = QWidget()
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Projects section
        projects_section = self._create_projects_section()
        content_layout.addWidget(projects_section)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        content_layout.addWidget(separator)

        # Recent conversations section
        recents_section = self._create_recents_section()
        content_layout.addWidget(recents_section)

        self.main_layout.addWidget(self._content_widget)

    def _create_header(self) -> QWidget:
        """Create the sidebar header with title and collapse button."""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)

        self.title_label = QLabel("AURA")
        self.title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(self.title_label)

        header_layout.addStretch()

        self.toggle_button = QPushButton("◀")
        self.toggle_button.setFixedSize(24, 24)
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.setToolTip("Collapse Sidebar")
        self.toggle_button.clicked.connect(self._toggle_collapse)
        header_layout.addWidget(self.toggle_button)

        return header

    def _toggle_collapse(self) -> None:
        """Toggle the collapsed state of the sidebar."""
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed: bool) -> None:
        """Set the collapsed state of the sidebar and animate the change."""
        if self._collapsed == collapsed:
            return

        self._collapsed = collapsed
        self.state_changed.emit(self._collapsed)

        start_width = self.width()
        end_width = self._collapsed_width if self._collapsed else self._expanded_width

        if self._collapsed:
            self._content_widget.hide()
            self.toggle_button.setText("▶")
            self.toggle_button.setToolTip("Expand Sidebar")
            self.title_label.hide()
        else:
            self.toggle_button.setText("◀")
            self.toggle_button.setToolTip("Collapse Sidebar")
            self.title_label.show()
            self._content_widget.show()

        self._animation.setStartValue(start_width)
        self._animation.setEndValue(end_width)
        self._animation.start()

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

            #toggleButton {{
                background-color: transparent;
                border: 1px solid {COLORS.border};
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            }}

            #toggleButton:hover {{
                background-color: {hover_bg};
                border-color: {COLORS.accent};
            }}

            #toggleButton:pressed {{
                background-color: {sidebar_bg};
            }}
        """)

    def sizeHint(self) -> QSize:
        """Provide a reasonable default size for the sidebar."""
        return QSize(280, 600)
