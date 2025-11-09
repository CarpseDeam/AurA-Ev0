"""
Project dialog for creating and editing projects.
"""

import logging
from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextEdit, QFileDialog, QFormLayout, QDialogButtonBox, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ..models import Project
from ..config import COLORS

logger = logging.getLogger(__name__)


class ProjectDialog(QDialog):
    """
    Dialog for creating or editing a project.

    Allows user to set:
    - Project name (required)
    - Description (optional)
    - Working directory (optional, with file picker)
    - Custom instructions (optional)
    """

    def __init__(self, project: Optional[Project] = None, parent: Optional[QDialog] = None):
        super().__init__(parent)

        self._project = project
        self._is_edit_mode = project is not None

        self.setWindowTitle("Edit Project" if self._is_edit_mode else "New Project")
        self.setModal(True)
        self.setMinimumWidth(500)

        self._setup_ui()
        self._apply_styling()

        if self._is_edit_mode:
            self._populate_fields()

    def _setup_ui(self) -> None:
        """Create and layout UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("Edit Project" if self._is_edit_mode else "Create New Project")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        # Project name (required)
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g., Aura Development")
        form.addRow("Project Name*:", self._name_input)

        # Description (optional)
        self._desc_input = QTextEdit()
        self._desc_input.setPlaceholderText("Optional description of this project")
        self._desc_input.setMaximumHeight(80)
        form.addRow("Description:", self._desc_input)

        # Working directory (optional, with file picker)
        workdir_layout = QHBoxLayout()
        self._workdir_input = QLineEdit()
        self._workdir_input.setPlaceholderText("Select working directory...")
        workdir_layout.addWidget(self._workdir_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        browse_btn.clicked.connect(self._browse_directory)
        workdir_layout.addWidget(browse_btn)

        form.addRow("Working Directory:", workdir_layout)

        # Custom instructions (optional)
        self._instructions_input = QTextEdit()
        self._instructions_input.setPlaceholderText(
            "Optional additional system prompt instructions for this project"
        )
        self._instructions_input.setMaximumHeight(120)
        form.addRow("Custom Instructions:", self._instructions_input)

        layout.addLayout(form)

        # Button box
        button_box = QDialogButtonBox()

        if self._is_edit_mode:
            delete_btn = QPushButton("Delete Project")
            delete_btn.clicked.connect(self._on_delete_clicked)
            button_box.addButton(delete_btn, QDialogButtonBox.DestructiveRole)

        save_btn = button_box.addButton(QDialogButtonBox.Save)
        cancel_btn = button_box.addButton(QDialogButtonBox.Cancel)

        save_btn.clicked.connect(self._on_save_clicked)
        cancel_btn.clicked.connect(self.reject)

        layout.addWidget(button_box)

    def _browse_directory(self) -> None:
        """Open file dialog to select working directory."""
        current_dir = self._workdir_input.text() or str(Path.home())

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            current_dir,
            QFileDialog.ShowDirsOnly
        )

        if directory:
            self._workdir_input.setText(directory)

    def _populate_fields(self) -> None:
        """Populate form fields with existing project data."""
        if not self._project:
            return

        self._name_input.setText(self._project.name or "")
        self._desc_input.setPlainText(self._project.description or "")
        self._workdir_input.setText(self._project.working_directory or "")
        self._instructions_input.setPlainText(self._project.custom_instructions or "")

    def _on_save_clicked(self) -> None:
        """Handle save button click."""
        name = self._name_input.text().strip()

        if not name:
            # TODO: Show error message
            logger.warning("Project name is required")
            return

        description = self._desc_input.toPlainText().strip() or None
        working_dir_input = self._workdir_input.text().strip()
        if not working_dir_input:
            QMessageBox.warning(
                self,
                "Working Directory Required",
                "Every project must reference a working directory. Please select one.",
            )
            return
        try:
            working_directory_path = Path(working_dir_input).expanduser().resolve()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Invalid Directory",
                f"Unable to use '{working_dir_input}': {exc}",
            )
            return
        if not working_directory_path.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Directory",
                f"The selected directory does not exist: {working_directory_path}",
            )
            return
        working_directory = str(working_directory_path)
        custom_instructions = self._instructions_input.toPlainText().strip() or None

        try:
            if self._is_edit_mode:
                # Update existing project
                self._project.update(
                    name=name,
                    description=description,
                    working_directory=working_directory,
                    custom_instructions=custom_instructions
                )
                logger.info(f"Updated project: {name} (ID: {self._project.id})")
            else:
                # Create new project
                self._project = Project.create(
                    name=name,
                    description=description,
                    working_directory=working_directory,
                    custom_instructions=custom_instructions
                )
                logger.info(f"Created project: {name} (ID: {self._project.id})")

            self.accept()

        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            # TODO: Show error dialog

    def _on_delete_clicked(self) -> None:
        """Handle delete button click with confirmation."""
        if not self._is_edit_mode or not self._project:
            return

        reply = QMessageBox.warning(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this project? This will also delete all of its "
            "conversations and messages permanently. This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        try:
            project_name = self._project.name
            project_id = self._project.id
            self._project.delete()
            logger.info(f"Deleted project: {project_name} (ID: {project_id})")

            # Set result to -1 to indicate deletion
            self.done(-1)

        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            QMessageBox.critical(self, "Error", f"Failed to delete project: {e}")

    def get_project(self) -> Optional[Project]:
        """
        Get the created or updated project.

        Returns:
            Project instance if saved, None otherwise
        """
        return self._project

    def _apply_styling(self) -> None:
        """Apply CSS styling to the dialog."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS.background};
                color: {COLORS.text};
            }}

            QLabel {{
                color: {COLORS.text};
            }}

            QLineEdit {{
                background-color: {COLORS.code_block_bg};
                border: 1px solid {COLORS.border};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS.text};
                selection-background-color: {COLORS.accent};
            }}

            QLineEdit:focus {{
                border-color: {COLORS.accent};
            }}

            QTextEdit {{
                background-color: {COLORS.code_block_bg};
                border: 1px solid {COLORS.border};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS.text};
                selection-background-color: {COLORS.accent};
            }}

            QTextEdit:focus {{
                border-color: {COLORS.accent};
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

            QPushButton[text="Delete Project"] {{
                background-color: {COLORS.error};
            }}

            QPushButton[text="Delete Project"]:hover {{
                background-color: #c93a31;
            }}
        """)
