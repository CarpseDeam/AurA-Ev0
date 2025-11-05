"""Main application window for Aura."""

from __future__ import annotations

import html
import os
import subprocess
from datetime import datetime

from PySide6.QtGui import QAction, QFont, QTextCursor, QTextOption
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aura import config
from aura.services import AgentRunner
from aura.ui.agent_settings_dialog import AgentSettingsDialog
from aura.utils import scan_directory
from aura.utils.agent_finder import find_cli_agents


class MainWindow(QMainWindow):
    """Displays the primary Aura workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self._working_directory = os.getcwd()
        self._selected_agent: str = config.DEFAULT_AGENT
        self._agent_path: str = ""
        self.output_view = QTextEdit(self)
        self.input_field = QLineEdit(self)
        self.clear_button = QPushButton("Clear", self)
        self.status_bar = QStatusBar(self)
        self.status_label = QLabel(self)
        self.directory_label = QLabel(self)
        self.toolbar = self.addToolBar("Project")
        self.current_runner: AgentRunner | None = None
        self._configure_window()
        self._build_layout()
        self._apply_styles()
        self._build_toolbar()
        self._setup_status_bar()
        self._connect_signals()
        self._detect_default_agent()
        self._set_ready_state()

    def _configure_window(self) -> None:
        """Configure top-level window properties."""
        self.setWindowTitle("Aura")
        self.resize(*config.WINDOW_DIMENSIONS)
        self.output_view.setReadOnly(True)
        self.output_view.setAcceptRichText(False)
        self.output_view.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.input_field.setPlaceholderText("Enter a request")
        self.input_field.setClearButtonEnabled(True)
        font = QFont(config.FONT_FAMILY)
        self.output_view.setFont(font)
        self.input_field.setFont(font)
        self.input_field.setFocus()

    def _build_layout(self) -> None:
        """Create and assign the central layout."""
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self.output_view)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_field)
        self.clear_button.setFixedWidth(72)
        input_row.addWidget(self.clear_button)
        layout.addLayout(input_row)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        self.setCentralWidget(container)

    def _apply_styles(self) -> None:
        """Apply the dark theme styling."""
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
            }}
            QTextEdit {{
                background-color: #1b1b1b;
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.agent_output};
                border-radius: 6px;
                padding: 10px;
            }}
            QLineEdit {{
                background-color: #232323;
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.accent};
                border-radius: 6px;
                padding: 8px;
            }}
            QLineEdit:focus {{
                border-color: {config.COLORS.success};
            }}
            """
        )
        self.clear_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #2d2d2d;
                color: {config.COLORS.text};
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                border-color: {config.COLORS.accent};
            }}
            """
        )

    def _setup_status_bar(self) -> None:
        """Initialize the status bar widgets."""
        self.status_label.setText("Ready")
        self.directory_label.setText(f"Dir: {self._working_directory}")
        self.directory_label.setStyleSheet("color: #9e9e9e;")
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.directory_label, 0)
        self.setStatusBar(self.status_bar)

    def _build_toolbar(self) -> None:
        """Create the application toolbar."""
        self.toolbar.setMovable(False)
        dir_action = QAction("Set Working Directory", self)
        dir_action.triggered.connect(self._select_working_directory)
        self.toolbar.addAction(dir_action)
        agent_action = QAction("Agent Settings...", self)
        agent_action.triggered.connect(self._open_agent_settings)
        self.toolbar.addAction(agent_action)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self.input_field.returnPressed.connect(self._handle_submit)
        self.clear_button.clicked.connect(self.clear_output)

    def _handle_submit(self) -> None:
        """Handle the submission of a prompt."""
        prompt = self.input_field.text().strip()
        if not prompt:
            return
        if self.current_runner is not None and self.current_runner.isRunning():
            self.display_output("An agent run is already in progress.", "#FF6B6B")
            return
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.display_output(f"> {prompt}", config.COLORS.accent)
        self.execute_command(prompt)

    def execute_command(self, prompt: str) -> None:
        """Execute an agent command for the given prompt."""
        if not prompt:
            self.input_field.setEnabled(True)
            self.input_field.setFocus()
            return
        if not self._validate_environment():
            self.input_field.setEnabled(True)
            self.input_field.setFocus()
            return
        command_prompt = self._build_command_prompt(prompt)
        agent_executable = self._agent_path or self._selected_agent
        command = [agent_executable, "-p", command_prompt, "--yolo"]
        try:
            runner = AgentRunner(
                command=command,
                working_directory=self._working_directory,
                parent=self,
            )
        except ValueError as exc:
            self.display_output(f"Unable to start agent: {exc}", "#FF6B6B")
            self.input_field.setEnabled(True)
            self.input_field.setFocus()
            return
        runner.output_line.connect(self.display_output)
        runner.process_finished.connect(self.handle_process_finished)
        runner.process_error.connect(self.handle_process_error)
        self.current_runner = runner
        self._set_running_state()
        runner.start()

    def display_output(self, text: str, color: str | None = None) -> None:
        """Render output text in the transcript."""
        chosen_color = color or self._resolve_line_color(text)
        escaped_text = html.escape(text)
        timestamp = datetime.now().strftime("%H:%M:%S")
        payload = (
            f'<span style="color: #888888;">[{timestamp}]</span> '
            f'<span style="color: {chosen_color}; white-space: pre-wrap;">{escaped_text}</span><br>'
        )
        cursor = self.output_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_view.setTextCursor(cursor)
        self.output_view.insertHtml(payload)
        self.output_view.ensureCursorVisible()

    def handle_process_finished(self, exit_code: int) -> None:
        """Handle completion of the agent process."""
        if exit_code == 0:
            self.display_output("Agent run completed successfully.", config.COLORS.success)
            self._set_completed_state()
        else:
            self.display_output(f"Agent exited with code {exit_code}", "#FF6B6B")
            self._set_error_state()
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        self.current_runner = None

    def handle_process_error(self, error: str) -> None:
        """Present an error emitted by the agent process."""
        self.display_output(error, "#FF6B6B")
        self._set_error_state()
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    def clear_output(self) -> None:
        """Clear the output transcript."""
        self.output_view.clear()

    def set_working_directory(self, path: str) -> None:
        """Update the working directory for agent runs."""
        if not path:
            raise ValueError("Working directory must be provided.")
        resolved = os.path.abspath(path)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Directory does not exist: {resolved}")
        self._working_directory = resolved
        self.directory_label.setText(f"Dir: {self._working_directory}")
        self.display_output(f"Working directory set to {self._working_directory}", config.COLORS.accent)

    def _resolve_line_color(self, text: str) -> str:
        """Choose a color based on output content."""
        stripped = text.strip()
        lowered = text.lower()
        if stripped.startswith(("✓", "✅")):
            return config.COLORS.success
        if "error" in lowered or "failed" in lowered:
            return "#FF6B6B"
        if stripped.startswith(("Creating", "Modifying")):
            return config.COLORS.accent
        return config.COLORS.agent_output

    def _set_ready_state(self) -> None:
        """Display the ready state."""
        self._update_status("Ready", config.COLORS.text)

    def _set_running_state(self) -> None:
        """Display the running state."""
        self._update_status("⚡ Running...", config.COLORS.accent)

    def _set_completed_state(self) -> None:
        """Display the completed state."""
        self._update_status("Completed", config.COLORS.success)

    def _set_error_state(self) -> None:
        """Display the error state."""
        self._update_status("Error", "#FF6B6B")

    def _update_status(self, message: str, color: str) -> None:
        """Apply text and color to the status indicator."""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 500;")

    def _select_working_directory(self) -> None:
        """Prompt the user to choose a working directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Working Directory", self._working_directory)
        if not path:
            return
        try:
            self.set_working_directory(path)
        except (ValueError, FileNotFoundError) as exc:
            self.display_output(str(exc), "#FF6B6B")

    def _validate_environment(self) -> bool:
        """Ensure prerequisites are met before starting the agent."""
        if not os.path.isdir(self._working_directory):
            self.display_output("Working directory does not exist.", "#FF6B6B")
            self._set_error_state()
            return False
        if not self._agent_path:
            agent_display = config.AGENT_DISPLAY_NAMES.get(self._selected_agent, self._selected_agent)
            self.display_output(
                f"{agent_display} not found. Use 'Agent Settings...' to configure.", "#FF6B6B"
            )
            self._set_error_state()
            return False
        return True

    def _detect_default_agent(self) -> None:
        """Detect and set the default available agent."""
        agents = find_cli_agents()
        for agent in agents:
            if agent.is_available and agent.name == self._selected_agent:
                self._agent_path = agent.executable_path
                self.display_output(
                    f"Using {agent.display_name} at {agent.executable_path}", config.COLORS.success
                )
                return
        for agent in agents:
            if agent.is_available:
                self._selected_agent = agent.name
                self._agent_path = agent.executable_path
                self.display_output(
                    f"Using {agent.display_name} at {agent.executable_path}", config.COLORS.success
                )
                return
        self.display_output(
            "No CLI agents found. Use 'Agent Settings...' to configure.", "#FF6B6B"
        )

    def _open_agent_settings(self) -> None:
        """Open the agent settings dialog."""
        dialog = AgentSettingsDialog(self)
        if dialog.exec():
            self._detect_default_agent()

    def _build_command_prompt(self, prompt: str) -> str:
        """Compose the prompt with project context."""
        context = self.get_project_context()
        return f"{context}\n\nTask: {prompt}"

    def get_project_context(self) -> str:
        """Return a concise description of the workspace."""
        try:
            snapshot = scan_directory(self._working_directory, max_depth=2)
        except (ValueError, FileNotFoundError) as exc:
            return f"Working in: {self._working_directory}\nFiles: unavailable ({exc})"
        python_files = [item for item in snapshot["files"] if item.endswith(".py")]
        directory_lines = "\n".join(f"- {item}" for item in snapshot["directories"]) or "- None"
        file_lines = "\n".join(f"- {item}" for item in python_files) or "- None"
        return (
            f"Working in: {self._working_directory}\n"
            f"Directories:\n{directory_lines}\n"
            f"Python files:\n{file_lines}"
        )
