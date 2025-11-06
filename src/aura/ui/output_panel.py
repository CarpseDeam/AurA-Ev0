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

        font = QFont(config.FONT_FAMILY, config.FONT_SIZE_OUTPUT)
        try:
            font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
        except AttributeError:
            pass
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)

        self._text_edit.setFont(font)
        self._text_edit.document().setDefaultFont(font)
        self._text_edit.setStyleSheet(
            "QTextEdit {"
            f" background-color: {config.COLORS.background};"
            f" color: {config.COLORS.text};"
            " border: none;"
            "}"
        )
        self._text_edit.document().setDocumentMargin(12.0)

    @property
    def text_edit(self) -> QTextEdit:
        """Expose the internal text edit widget for styling or testing."""
        return self._text_edit

    def display_output(
        self, text: str, color: Optional[str] = None, *, font_size: Optional[int] = None
    ) -> None:
        """Append output to the transcript."""
        chosen_color = color or self._resolve_line_color(text)
        payload = self._build_html_block(
            text=text,
            color=chosen_color,
            font_size=font_size or config.FONT_SIZE_OUTPUT,
        )
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
        chosen_color = color or config.COLORS.text
        payload = self._build_html_block(
            text=text,
            color=chosen_color,
            font_size=config.FONT_SIZE_OUTPUT,
        )
        self._append_html(payload)

    def display_startup_header(self) -> None:
        """Render the startup ASCII art header."""
        header_html = """
<pre style="font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 18px; line-height: 1.5; margin: 24px 0; display: block;">
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
        if self._is_timestamp_prefix(stripped):
            return config.COLORS.secondary
        return config.COLORS.text

    def _is_timestamp_prefix(self, text: str) -> bool:
        """Return True when the line begins with a timestamp-like token."""
        if not text.startswith("["):
            return False
        closing_index = text.find("]")
        if closing_index <= 1:
            return False
        token = text[1:closing_index]
        has_digit = any(char.isdigit() for char in token)
        has_time_separator = any(char in (":", "-", "/") for char in token)
        return has_digit and has_time_separator

    def _build_html_block(self, text: str, color: str, font_size: int) -> str:
        """Return a styled HTML block for consistent typography."""
        font_family = config.FONT_FAMILY.replace('"', '\\"')
        content = html.escape(text) if text else "&nbsp;"
        return (
            '<div style="'
            f'font-family: \\"{font_family}\\";'
            f" font-size: {font_size}px;"
            " line-height: 1.5;"
            " margin: 0 0 8px 0;"
            " white-space: pre-wrap;"
            f" color: {color};"
            '">'
            f"{content}"
            "</div>"
        )
