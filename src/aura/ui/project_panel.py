"""
Project panel UI component for displaying project details and settings.
"""

import logging
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QScrollArea, QLineEdit
)
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QFont

from ..models import Project
from ..config import COLORS

logger = logging.getLogger(__name__)


class ProjectPanel(QWidget):
    """
    Right sidebar showing project details and settings.

    Displayed when a project is active. Shows:
    - Project name and description
    - Working directory
    - Custom instructions
    - Files list (placeholder for future)

    Signals:
        edit_project_clicked: Emitted when "Edit Project" button is clicked
        close_clicked: Emitted when close/hide button is clicked
    """

    edit_project_clicked = Signal()
    close_clicked = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_project: Optional[Project] = None
        self._setup_ui()
        self._apply_styling()

    def _setup_ui(self) -> None:
        """Create and layout UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header with close button
        header = self._create_header()
        layout.addWidget(header)

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        self._content_layout = QVBoxLayout(scroll_content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(16)

        # Project name
        self._name_section = self._create_section("Project Name")
        self._name_label = QLabel()
        self._name_label.setWordWrap(True)
        self._name_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self._name_section.layout().addWidget(self._name_label)
        self._content_layout.addWidget(self._name_section)

        # Description
        self._desc_section = self._create_section("Description")
        self._desc_label = QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_section.layout().addWidget(self._desc_label)
        self._content_layout.addWidget(self._desc_section)

        # Working directory
        self._workdir_section = self._create_section("Working Directory")
        self._workdir_label = QLabel()
        self._workdir_label.setWordWrap(True)
        self._workdir_label.setFont(QFont("Cascadia Code, Consolas, monospace", 10))
        self._workdir_section.layout().addWidget(self._workdir_label)
        self._content_layout.addWidget(self._workdir_section)

        # Custom instructions
        self._instructions_section = self._create_section("Custom Instructions")
        self._instructions_text = QTextEdit()
        self._instructions_text.setReadOnly(True)
        self._instructions_text.setMaximumHeight(150)
        self._instructions_section.layout().addWidget(self._instructions_text)
        self._content_layout.addWidget(self._instructions_section)

        # Files section (placeholder)
        self._files_section = self._create_section("Attached Files")
        self._files_label = QLabel("(Coming soon)")
        self._files_label.setStyleSheet(f"color: {COLORS.secondary};")
        self._files_section.layout().addWidget(self._files_label)
        self._content_layout.addWidget(self._files_section)

        self._content_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # Edit button at bottom
        edit_btn = QPushButton("Edit Project")
        edit_btn.clicked.connect(self.edit_project_clicked.emit)
        layout.addWidget(edit_btn)

    def _create_header(self) -> QWidget:
        """Create the panel header with title and close button."""
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Project Details")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        close_btn = QPushButton("×")
        close_btn.setMaximumWidth(30)
        close_btn.setMaximumHeight(30)
        close_btn.setFont(QFont("Segoe UI", 16))
        close_btn.clicked.connect(self.close_clicked.emit)
        header_layout.addWidget(close_btn)

        return header

    def _create_section(self, title: str) -> QWidget:
        """
        Create a labeled section container.

        Args:
            title: Section title

        Returns:
            QWidget with vertical layout
        """
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel(title)
        label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        label.setStyleSheet(f"color: {COLORS.secondary};")
        layout.addWidget(label)

        return section

    def set_project(self, project: Optional[Project]) -> None:
        """
        Display the given project's information.

        Args:
            project: Project to display, or None to clear
        """
        self._current_project = project

        if project is None:
            self._clear_display()
            return

        # Update all fields
        self._name_label.setText(project.name or "(Unnamed)")

        if project.description:
            self._desc_label.setText(project.description)
            self._desc_section.show()
        else:
            self._desc_section.hide()

        if project.working_directory:
            self._workdir_label.setText(project.working_directory)
            self._workdir_section.show()
        else:
            self._workdir_section.hide()

        if project.custom_instructions:
            self._instructions_text.setPlainText(project.custom_instructions)
            self._instructions_section.show()
        else:
            self._instructions_section.hide()

        logger.info(f"Displaying project: {project.name} (ID: {project.id})")

    def _clear_display(self) -> None:
        """Clear all displayed information."""
        self._name_label.setText("")
        self._desc_label.setText("")
        self._workdir_label.setText("")
        self._instructions_text.clear()

    def get_current_project(self) -> Optional[Project]:
        """
        Get the currently displayed project.

        Returns:
            Current Project instance, or None
        """
        return self._current_project

    def _apply_styling(self) -> None:
        """Apply CSS styling to the panel."""
        # Panel background slightly lighter than main background
        panel_bg = "#161b22"

        self.setStyleSheet(f"""
            QWidget {{
                background-color: {panel_bg};
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
                padding: 8px 16px;
                font-size: 12px;
            }}

            QPushButton:hover {{
                background-color: {COLORS.prompt};
            }}

            QPushButton:pressed {{
                background-color: {COLORS.accent};
            }}

            QTextEdit {{
                background-color: {COLORS.background};
                border: 1px solid {COLORS.border};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS.text};
            }}

            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
        """)

    def sizeHint(self) -> QSize:
        """Provide a reasonable default size for the panel."""
        return QSize(320, 600)
