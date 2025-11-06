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

        # Set font with configured size for better readability
        font = QFont(config.FONT_FAMILY)
        font.setPointSize(config.FONT_SIZE_OUTPUT)
        self._text_edit.setFont(font)

    @property
    def text_edit(self) -> QTextEdit:
        """Expose the internal text edit widget for styling or testing."""
        return self._text_edit

    def display_output(
        self,
        text: str,
        color: Optional[str] = None,
        font_size: Optional[int] = None,
    ) -> None:
        """Append output to the transcript with an automatic newline."""
        chosen_color = color or self._resolve_line_color(text)
        normalized = text if text.endswith("\n") else f"{text}\n"
        self._append_text(normalized, chosen_color, font_size)

    def display_stream_chunk(self, text: str, color: Optional[str] = None) -> None:
        """Append a streaming chunk without forcing a newline."""
        if not text:
            return
        chosen_color = color or config.COLORS.agent_output
        self._append_text(text, chosen_color, None)

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
        normalized = text if text.endswith("\n") else f"{text}\n"
        self._append_text(normalized, chosen_color, None)

    def display_startup_header(self) -> None:
        """Render the startup ASCII art header."""
        header_html = """
<pre style="font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 16px; line-height: 0.95; margin: 20px 0;">
<span style="color: #00CED1;">  █████╗ ██╗   ██╗██████╗  █████╗ </span>
<span style="color: #20D5E0;"> ██╔══██╗██║   ██║██╔══██╗██╔══██╗</span>
<span style="color: #40B5F5;"> ███████║██║   ██║██████╔╝███████║</span>
<span style="color: #60A0F0;"> ██╔══██║██║   ██║██╔══██╗██╔══██║</span>
<span style="color: #8080E8;"> ██║  ██║╚██████╔╝██║  ██║██║  ██║</span>
<span style="color: #A060DD;"> ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝</span>
</pre>
<div style="color: #C090D0; font-size: 13px; margin: 10px 0 10px 16px; letter-spacing: 1px;">
AI-Powered Development Assistant
</div>
<div style="color: #555555; margin: 0 0 20px 16px;">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</div>
<br>
        """
        self._append_html(header_html)

    def clear(self) -> None:
        """Remove all text from the panel."""
        self._text_edit.clear()

    def _append_text(
        self,
        text: str,
        color: str,
        font_size: Optional[int] = None,
    ) -> None:
        """Append raw text to the panel, preserving streaming continuity."""
        if not text:
            return
        escaped_text = html.escape(text).replace("\n", "<br>")
        style = f"color: {color};"
        if font_size is not None:
            style += f" font-size: {font_size}px; font-weight: 500;"
        payload = f'<span style="{style}">{escaped_text}</span>'
        self._append_html(payload)

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
