"""Reusable output panel widget for rendering orchestration logs."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Optional

from PySide6.QtGui import QFont, QTextCursor, QTextOption
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget

from src.aura import config


class OutputPanel(QWidget):
    """Encapsulates the formatted transcript view for orchestration output."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._text_edit = QTextEdit(self)
        self._configure_widget()

    def _configure_widget(self) -> None:
        """Initialize widget layout and styling."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text_edit)

        self._text_edit.setReadOnly(True)
        self._text_edit.setAcceptRichText(True)
        self._text_edit.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self._text_edit.setFont(QFont(config.FONT_FAMILY))

    @property
    def text_edit(self) -> QTextEdit:
        """Expose the internal text edit widget for styling or testing."""
        return self._text_edit

    def display_output(self, text: str, color: Optional[str] = None) -> None:
        """Append timestamped output to the transcript."""
        chosen_color = color or self._resolve_line_color(text)
        escaped_text = html.escape(text)
        timestamp = datetime.now().strftime("%H:%M:%S")
        payload = (
            f'<span style="color: #888888;">[{timestamp}]</span> '
            f'<span style="color: {chosen_color}; white-space: pre-wrap;">{escaped_text}</span><br>'
        )
        self._append_html(payload)

    def append_to_log(self, text: str, color: Optional[str] = None) -> None:
        """Append streaming output without timestamp metadata."""
        if not text:
            return
        chosen_color = color or config.COLORS.agent_output
        escaped_text = html.escape(text)
        payload = f'<span style="color: {chosen_color}; white-space: pre-wrap;">{escaped_text}</span><br>'
        self._append_html(payload)

    def display_startup_header(self) -> None:
        """Render the startup ASCII art header."""
        header_html = """
        <div style="font-family: 'Courier New', 'Consolas', monospace; font-size: 14px; line-height: 1.2; margin: 20px 0; white-space: pre;">
<span style="color: #5294E2;">    ___    ____  ____  ___    </span>
<span style="color: #7B68EE;">   /   |  / __ \/ __ \/   |   </span>
<span style="color: #9370DB;">  / /| | / /_/ / /_/ / /| |   </span>
<span style="color: #BA55D3;"> / ___ |/ _, _/ _, _/ ___ |   </span>
<span style="color: #DA70D6;">/_/  |_/_/ |_/_/ |_/_/  |_|   </span>
<span style="color: #EE82EE;">                               </span>
        </div>
        """
        self._append_html(header_html)

    def clear(self) -> None:
        """Remove all text from the panel."""
        self._text_edit.clear()

    def _append_html(self, html_content: str) -> None:
        """Append HTML content to the underlying text edit."""
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.insertHtml(html_content)
        self._text_edit.ensureCursorVisible()

    def _resolve_line_color(self, text: str) -> str:
        """Infer an output color when one is not provided."""
        stripped = text.strip()
        lowered = stripped.lower()
        if stripped.startswith("Session complete"):
            return config.COLORS.success
        if "error" in lowered or "failed" in lowered:
            return "#FF6B6B"
        if stripped.startswith(("Creating", "Modifying")):
            return config.COLORS.accent
        return config.COLORS.agent_output