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
        <div style="font-family: 'Courier New', 'Consolas', monospace; font-size: 18px; line-height: 1.1; margin: 20px 0; white-space: pre;">
<span style="color: #5294E2;">�������������������������������������������ͻ</span>
<span style="color: #6B7FEE;">�   </span><span style="color: #7B68EE;">����ۻ �ۻ   �ۻ�����ۻ  ����ۻ</span><span style="color: #6B7FEE;">   �</span>
<span style="color: #8875E8;">�  </span><span style="color: #9370DB;">������ۻ�ۺ   �ۺ������ۻ������ۻ</span><span style="color: #8875E8;">  �</span>
<span style="color: #A565DD;">�  </span><span style="color: #BA55D3;">������ۺ�ۺ   �ۺ������ɼ������ۺ</span><span style="color: #A565DD;">  �</span>
<span style="color: #C25DD8;">�  </span><span style="color: #DA70D6;">������ۺ��ۻ ��ɼ������ۻ������ۺ</span><span style="color: #C25DD8;">  �</span>
<span style="color: #DA6FD7;">�  </span><span style="color: #EE82EE;">�ۺ  �ۺ �����ɼ �ۺ  �ۺ�ۺ  �ۺ</span><span style="color: #DA6FD7;">  �</span>
<span style="color: #EE7CC9;">�  </span><span style="color: #FF69B4;">�ͼ  �ͼ  ���ͼ  �ͼ  �ͼ�ͼ  �ͼ</span><span style="color: #EE7CC9;">  �</span>
<span style="color: #FF4EA3;">�������������������������������������������ͼ</span>
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
