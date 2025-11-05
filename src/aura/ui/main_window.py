"""Main application window for Aura."""

from __future__ import annotations

import html
import os

from PySide6.QtGui import QFont, QTextCursor, QTextOption
from PySide6.QtWidgets import QLineEdit, QMainWindow, QTextEdit, QVBoxLayout, QWidget

from aura import config
from aura.services import AgentRunner


class MainWindow(QMainWindow):
    """Displays the primary Aura workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self.output_view = QTextEdit(self)
        self.input_field = QLineEdit(self)
        self.current_runner: AgentRunner | None = None
        self._configure_window()
        self._build_layout()
        self._apply_styles()
        self._connect_signals()

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
        layout.addWidget(self.input_field)
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

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self.input_field.returnPressed.connect(self._handle_submit)

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
        """Execute a Gemini command for the given prompt."""
        if not prompt:
            self.input_field.setEnabled(True)
            self.input_field.setFocus()
            return
        command = ["gemini", "-p", prompt, "--yolo"]
        try:
            runner = AgentRunner(
                command=command,
                working_directory=os.getcwd(),
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
        runner.start()

    def display_output(self, text: str, color: str | None = None) -> None:
        """Render output text in the transcript."""
        chosen_color = color or config.COLORS.agent_output
        escaped_text = html.escape(text)
        payload = f'<span style="color: {chosen_color};">{escaped_text}</span><br>'
        cursor = self.output_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_view.setTextCursor(cursor)
        self.output_view.insertHtml(payload)
        self.output_view.ensureCursorVisible()

    def handle_process_finished(self, exit_code: int) -> None:
        """Handle completion of the agent process."""
        if exit_code == 0:
            self.display_output("Agent run completed successfully.", config.COLORS.success)
        else:
            self.display_output(f"Agent exited with code {exit_code}", "#FF6B6B")
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
        self.current_runner = None

    def handle_process_error(self, error: str) -> None:
        """Present an error emitted by the agent process."""
        self.display_output(error, "#FF6B6B")
        self.input_field.setEnabled(True)
        self.input_field.setFocus()
