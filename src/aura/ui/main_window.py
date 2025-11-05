"""Main application window for Aura."""

from __future__ import annotations

from PySide6.QtGui import QFont, QTextOption
from PySide6.QtWidgets import QLineEdit, QMainWindow, QTextEdit, QVBoxLayout, QWidget

from aura import config


class MainWindow(QMainWindow):
    """Displays the primary Aura workspace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the main window."""
        super().__init__(parent)
        self.output_view = QTextEdit(self)
        self.input_field = QLineEdit(self)
        self._configure_window()
        self._build_layout()
        self._apply_styles()

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
