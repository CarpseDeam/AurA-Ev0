"""Reusable output panel widget for rendering orchestration logs."""

from __future__ import annotations

import html
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
        """Append output to the transcript."""
        chosen_color = color or self._resolve_line_color(text)
        escaped_text = html.escape(text)
        payload = f'<span style="color: {chosen_color};">{escaped_text}</span><br>'
        self._append_html(payload)

    def display_thinking(self, text: str) -> None:
        """Render thinking/reasoning steps with a purple ellipsis."""
        self.display_output(f"⋯ {text}", config.COLORS.thinking)

    def display_tool_call(self, tool_name: str, args_summary: str) -> None:
        """Render tool calls with a gold gear symbol."""
        self.display_output(f"⚙ {tool_name}({args_summary})", config.COLORS.tool_call)

    def display_file_operation(self, action: str, path: str) -> None:
        """Render file operations with appropriate symbols and colors."""
        symbol = "+" if action.lower() == "creating" else "~"
        self.display_output(f"{symbol} {action} {path}", config.COLORS.accent)

    def display_success(self, text: str) -> None:
        """Render success messages with a green checkmark."""
        self.display_output(f"✓ {text}", config.COLORS.success)

    def display_error(self, text: str) -> None:
        """Render error messages with a red X mark."""
        self.display_output(f"✗ {text}", config.COLORS.error)

    def append_to_log(self, text: str, color: Optional[str] = None) -> None:
        """Append streaming output without timestamp metadata."""
        if not text:
            return
        chosen_color = color or config.COLORS.agent_output
        escaped_text = html.escape(text)
        payload = f'<span style="color: {chosen_color};">{escaped_text}</span><br>'
        self._append_html(payload)

    def display_startup_header(self) -> None:
        """Render the startup ASCII art header."""
        header_html = """
<pre style="font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 14px; line-height: 1.2; margin: 20px 0; display: block;">
<span style="color: #00CED1;">   █████╗ ██╗   ██╗██████╗  █████╗ </span>
<span style="color: #40E0D0;">  ██╔══██╗██║   ██║██╔══██╗██╔══██╗</span>
<span style="color: #42A5F5;">  ███████║██║   ██║██████╔╝███████║</span>
<span style="color: #64B5F6;">  ██╔══██║██║   ██║██╔══██╗██╔══██║</span>
<span style="color: #7E85E8;">  ██║  ██║╚██████╔╝██║  ██║██║  ██║</span>
<span style="color: #9370DB;">  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝</span>
<span style="color: #B19CD9;">  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
<span style="color: #DA70D6;">   AI-Powered Development Assistant</span>
<span style="color: #FF69B4;">  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
</pre><br>
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
            return config.COLORS.error
        if stripped.startswith(("Creating", "Modifying")):
            return config.COLORS.accent
        return config.COLORS.agent_output