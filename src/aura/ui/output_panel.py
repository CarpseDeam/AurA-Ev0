"""Reusable output panel widget for rendering orchestration logs."""

from __future__ import annotations

import html
import io
import keyword
import re
import tokenize
from typing import Optional

from PySide6.QtGui import QFont, QTextCursor, QTextOption
from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget

from aura import config


class OutputPanel(QWidget):
    """Encapsulates the formatted transcript view for orchestration output."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._text_edit = QTextEdit(self)
        self._stream_buffer: str = ""
        self._needs_leading_break = False
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

        # Custom scrollbar styling
        self._text_edit.setStyleSheet(
            f"""
            QTextEdit {{
                background: {config.COLORS.background};
                color: {config.COLORS.text};
                border: none;
                padding: 16px;
                selection-background-color: {config.COLORS.accent};
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {config.COLORS.border};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #2a2a2a;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            """
        )

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
        """Append formatted output to the transcript."""
        if text is None:
            return
        self._flush_stream_buffer()
        self._ensure_leading_break()
        chosen_color = color or self._resolve_line_color(text)
        self._render_formatted_block(
            text,
            chosen_color,
            font_size,
            add_trailing_break=True,
        )

    def display_stream_chunk(self, text: str, color: Optional[str] = None) -> None:
        """Append a streaming chunk while buffering for structural parsing."""
        if not text:
            return
        chosen_color = color or config.COLORS.agent_output
        normalized = self._normalize_content(text)
        self._stream_buffer += normalized
        for segment in self._consume_stream_segments():
            if segment.strip():
                self._render_formatted_block(segment, chosen_color, None)

    def display_thinking(self, text: str) -> None:
        """Render thinking/reasoning steps with a purple ellipsis."""
        self.display_output(f"⋯ {text}", config.COLORS.thinking)

    def display_progress(self, message: str) -> None:
        """Display progress message with animated indicator."""
        # Use the thinking color with animation
        self.display_output(f"⋯ {message}", config.COLORS.thinking)

    def display_tool_call(self, tool_name: str, args_summary: str) -> None:
        """Render tool calls with a structured summary block."""
        truncated = args_summary if len(args_summary) <= 200 else f"{args_summary[:197]}..."
        safe_name = html.escape(tool_name)
        safe_args = html.escape(truncated)
        payload = (
            f'<div style="margin:{config.OUTPUT_BLOCK_SPACING_PX}px 0;'
            f' color:{config.COLORS.tool_call};">'
            f'<div style="font-weight:600;">⚙ {safe_name}</div>'
            f'<div style="color:{config.COLORS.secondary}; font-size:{config.FONT_SIZE_OUTPUT - 1}px;'
            f' margin-left:14px; white-space:pre-wrap;">{safe_args}</div>'
            "</div>"
        )
        self._append_html(payload)

    def display_file_operation(self, action: str, path: str) -> None:
        """Render file operations with appropriate symbols and colors."""
        lower = action.lower()
        icon = "+"
        if "delete" in lower:
            icon = "−"
        elif "modify" in lower or "update" in lower:
            icon = "~"
        self._render_file_operation_block(
            title=action.upper(),
            icon=icon,
            path=path,
            content=None,
            footer=None,
        )

    def display_file_creation(self, path: str, content: str) -> None:
        """Render a newly created file with formatted code content."""
        self._render_file_operation_block(
            title="CREATE FILE",
            icon="+",
            path=path,
            content=content,
            footer=self._format_file_stats(content),
        )

    def display_file_modification(self, path: str, content: str) -> None:
        """Render a modified file preview."""
        self._render_file_operation_block(
            title="MODIFY FILE",
            icon="~",
            path=path,
            content=content,
            footer=self._format_file_stats(content),
        )

    def display_file_deletion(self, path: str) -> None:
        """Render a deletion summary without file content."""
        self._render_file_operation_block(
            title="DELETE FILE",
            icon="−",
            path=path,
            content=None,
            footer="File removed from workspace.",
        )

    def display_diff_block(self, diff_text: str) -> None:
        """Render a syntax-highlighted diff preview."""
        if not diff_text:
            return
        self._flush_stream_buffer()
        self._ensure_leading_break()
        normalized = self._normalize_content(diff_text)
        lines = normalized.splitlines() or [normalized]
        rows = "".join(self._build_diff_line_html(line) for line in lines)
        container = (
            f'<div style="margin:{config.OUTPUT_BLOCK_SPACING_PX}px 0;">'
            f'<div style="color:{config.COLORS.secondary};'
            f' font-size:{config.FONT_SIZE_OUTPUT - 1}px; margin-bottom:4px;">'
            "Diff preview</div>"
            f'<div style="border:1px solid {config.COLORS.code_block_border};'
            f' background:{config.COLORS.code_block_bg}; border-radius:8px;'
            ' overflow:hidden;">'
            f"{rows}"
            "</div>"
            "</div>"
        )
        self._append_html(container)

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
        self._ensure_leading_break()
        self._render_formatted_block(text, chosen_color, None, add_trailing_break=True)

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
        self._stream_buffer = ""
        self._needs_leading_break = False

    def _render_formatted_block(
        self,
        text: str,
        color: str,
        font_size: Optional[int],
        add_trailing_break: bool = False,
    ) -> None:
        """Detect structure within text and render the appropriate HTML."""
        normalized = self._normalize_content(text)
        if not normalized.strip():
            self._append_html("<br>", mark_content=False)
            return

        if self._looks_like_code(normalized):
            self._append_html(self._build_code_block_html(normalized))
            return

        paragraphs = [chunk.strip() for chunk in normalized.split("\n\n") if chunk.strip()]
        if not paragraphs:
            self._append_html("<br>", mark_content=False)
            return

        parts = [self._format_paragraph(paragraph, color, font_size) for paragraph in paragraphs]
        self._append_html("".join(parts))
        if add_trailing_break:
            self._append_html("<br>", mark_content=False)

    def _format_paragraph(self, paragraph: str, color: str, font_size: Optional[int]) -> str:
        """Return HTML for a paragraph, list, or header block."""
        if self._is_header_line(paragraph):
            return self._build_header_html(paragraph)
        if self._is_list_block(paragraph):
            return self._build_list_html(paragraph)
        return self._build_plain_paragraph(paragraph, color, font_size)

    def _build_header_html(self, text: str) -> str:
        """Return styled HTML for headers such as === SECTION ===."""
        safe = html.escape(text.strip("=" + " ").strip())
        return (
            f'<div style="margin:{config.OUTPUT_SECTION_SPACING_PX}px 0 '
            f'{config.OUTPUT_BLOCK_SPACING_PX}px 0; color:{config.COLORS.header};'
            f' font-weight:600; font-size:{config.OUTPUT_HEADER_FONT_SIZE}px; letter-spacing:0.5px;">'
            f"{safe}</div>"
        )

    def _build_plain_paragraph(self, text: str, color: str, font_size: Optional[int]) -> str:
        """Return HTML for standard paragraphs."""
        escaped = html.escape(text).replace("\n", "<br>")
        style = [
            f"color: {color}",
            f"margin: {config.OUTPUT_BLOCK_SPACING_PX}px 0",
            f"line-height: {config.LINE_HEIGHT}",
        ]
        if font_size is not None:
            style.append(f"font-size: {font_size}px")
        return f'<div style="{"; ".join(style)}">{escaped}</div>'

    def _build_list_html(self, paragraph: str) -> str:
        """Return HTML for bullet or ordered lists."""
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            return ""
        ordered = all(re.match(r"^\d+[.)]\s+", line) for line in lines)
        cleaned_items: list[str] = []
        for line in lines:
            cleaned = re.sub(r"^(\d+[.)]|[-*•])\s+", "", line)
            cleaned_items.append(f"<li>{html.escape(cleaned)}</li>")
        tag = "ol" if ordered else "ul"
        return (
            f'<{tag} style="margin:{config.OUTPUT_BLOCK_SPACING_PX}px 0 '
            f'{config.OUTPUT_BLOCK_SPACING_PX}px 22px; color:{config.COLORS.text};'
            ' padding-left:12px;">'
            f'{"".join(cleaned_items)}</{tag}>'
        )

    def _is_header_line(self, text: str) -> bool:
        """Determine whether a paragraph looks like a section header."""
        stripped = text.strip()
        if not stripped:
            return False
        if stripped.startswith("===") and stripped.endswith("==="):
            return True
        if stripped.startswith("---") and stripped.endswith("---"):
            return True
        if re.fullmatch(r"[A-Z0-9 /&'_-]+:?$", stripped) and len(stripped) <= 60:
            return True
        return False

    def _is_list_block(self, paragraph: str) -> bool:
        """Check whether most lines within the paragraph are list items."""
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            return False
        bullet_lines = sum(
            1 for line in lines if re.match(r"^([-*•]|[\d]+[.)])\s+", line)
        )
        return bullet_lines >= max(1, len(lines) // 2)

    def _looks_like_code(self, text: str) -> bool:
        """Heuristically determine if the text resembles Python code."""
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        patterns = (
            r"^\s*(def|class|async\s+def)\s+",
            r"^\s*(from|import)\s+",
            r"^\s*@\w+",
            r"^\s*(if|for|while|try|except|with)\b.*:\s*$",
        )
        matches = sum(1 for line in lines if any(re.match(pattern, line) for pattern in patterns))
        return matches >= 2 or (matches >= 1 and len(lines) <= 4 and ":" in lines[-1])

    def _build_code_block_html(self, code: str) -> str:
        """Return syntax-highlighted HTML for code blocks."""
        highlighted = self._syntax_highlight(code)
        return (
            f'<div style="background:{config.COLORS.code_block_bg};'
            f' border-left:4px solid {config.COLORS.code_block_border};'
            f' border-radius:6px; padding:{config.CODE_BLOCK_PADDING_PX}px;'
            " font-family: 'JetBrains Mono','Consolas',monospace; margin:"
            f'{config.OUTPUT_BLOCK_SPACING_PX}px 0; white-space:normal;">'
            f'<pre style="margin:0; white-space:pre-wrap; color:{config.COLORS.text};">'
            f"{highlighted}</pre></div>"
        )

    def _syntax_highlight(self, code: str) -> str:
        """Perform lightweight syntax highlighting for Python code."""
        buffer = io.StringIO(code)
        parts: list[str] = []
        try:
            for token in tokenize.generate_tokens(buffer.readline):
                tok_type, tok_string = token.type, token.string
                if tok_type in (tokenize.NL, tokenize.NEWLINE):
                    parts.append("\n")
                    continue
                if tok_type == tokenize.INDENT:
                    parts.append(tok_string)
                    continue
                if tok_type == tokenize.DEDENT:
                    continue
                escaped = html.escape(tok_string)
                color = None
                if tok_type == tokenize.NAME and tok_string in keyword.kwlist:
                    color = config.COLORS.code_keyword
                elif tok_type == tokenize.STRING:
                    color = config.COLORS.code_string
                elif tok_type == tokenize.COMMENT:
                    color = config.COLORS.code_comment
                if color:
                    parts.append(f'<span style="color:{color};">{escaped}</span>')
                else:
                    parts.append(escaped)
        except tokenize.TokenError:
            return html.escape(code)
        return "".join(parts)

    def _format_file_stats(self, content: str) -> str:
        """Return a concise human-readable size summary."""
        byte_count = len(content.encode("utf-8"))
        if byte_count >= 1024:
            size = f"{byte_count / 1024:.1f} KB"
        else:
            size = f"{byte_count} B"
        lines = max(1, len(content.splitlines()))
        return f"{size} · {lines} lines"

    def _build_diff_line_html(self, line: str) -> str:
        """Return styled HTML for a single diff line."""
        background, color, weight = self._diff_line_style(line)
        safe_line = html.escape(line or "")
        return (
            "<div style="font-family:'JetBrains Mono','Consolas',monospace;""
            f' white-space:pre; padding:4px 12px; background:{background};'
            f' color:{color}; font-weight:{weight};">'
            f"{safe_line}</div>"
        )

    def _diff_line_style(self, line: str) -> tuple[str, str, str]:
        """Return diff styling metadata."""
        addition_bg = 'rgba(63, 185, 80, 0.18)'
        addition_color = '#c2f4cf'
        deletion_bg = 'rgba(248, 81, 73, 0.18)'
        deletion_color = '#f8c0ba'
        context_bg = '#1f2a37'
        neutral_bg = 'transparent'
        neutral_color = config.COLORS.text
        secondary_color = config.COLORS.secondary

        if line.startswith('@@'):
            return context_bg, config.COLORS.accent, '600'
        if line.startswith('+') and not line.startswith('+++'):
            return addition_bg, addition_color, '500'
        if line.startswith('-') and not line.startswith('---'):
            return deletion_bg, deletion_color, '500'
        if line.startswith(('---', '+++')):
            return neutral_bg, secondary_color, '500'
        return neutral_bg, neutral_color, '400'

    def _normalize_content(self, text: str) -> str:
        """Normalize newlines and unescape common sequences for readability."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\\r\\n", "\n")
        normalized = normalized.replace("\\n", "\n")
        normalized = normalized.replace("\\t", "\t")
        return normalized

    def _consume_stream_segments(self) -> list[str]:
        """Flush completed segments from the stream buffer."""
        segments: list[str] = []
        while True:
            double_newline = self._stream_buffer.find("\n\n")
            if double_newline != -1:
                segments.append(self._stream_buffer[:double_newline])
                self._stream_buffer = self._stream_buffer[double_newline + 2 :]
                continue
            if self._stream_buffer.endswith("\n"):
                trimmed = self._stream_buffer.rstrip("\n")
                if trimmed:
                    segments.append(trimmed)
                self._stream_buffer = ""
                break
            if len(self._stream_buffer) > config.STREAM_CHUNK_FLUSH_THRESHOLD:
                segments.append(self._stream_buffer)
                self._stream_buffer = ""
                break
            break
        return segments

    def _flush_stream_buffer(self) -> None:
        """Flush any partial stream buffer to the panel."""
        if not self._stream_buffer.strip():
            self._stream_buffer = ""
            return
        self._render_formatted_block(self._stream_buffer, config.COLORS.agent_output, None)
        self._stream_buffer = ""

    def _render_file_operation_block(
        self,
        title: str,
        icon: str,
        path: str,
        content: Optional[str],
        footer: Optional[str],
    ) -> None:
        """Render a structured file operation summary."""
        safe_title = html.escape(title)
        safe_path = html.escape(path)
        header = (
            f'<div style="margin:{config.OUTPUT_SECTION_SPACING_PX}px 0 '
            f'{config.OUTPUT_BLOCK_SPACING_PX}px 0;">'
            f'<div style="color:{config.COLORS.header}; font-size:{config.OUTPUT_HEADER_FONT_SIZE}px;'
            ' font-weight:600;">'
            f"{icon} {safe_title}</div>"
            f'<div style="color:{config.COLORS.secondary}; font-size:{config.FONT_SIZE_OUTPUT - 1}px;'
            ' margin-top:2px;">'
            f"{safe_path}</div>"
            "</div>"
        )
        parts = [header]
        if content is not None:
            normalized = self._normalize_content(content)
            parts.append(self._build_code_block_html(normalized))
            parts.append(
                f'<div style="border-bottom:1px solid {config.COLORS.code_block_border};'
                f' margin:{config.OUTPUT_BLOCK_SPACING_PX}px 0;"></div>'
            )
        if footer:
            parts.append(
                f'<div style="color:{config.COLORS.secondary}; font-size:{config.FONT_SIZE_OUTPUT - 2}px;'
                f' margin-bottom:{config.OUTPUT_BLOCK_SPACING_PX}px;">{html.escape(footer)}</div>'
            )
        self._append_html("".join(parts))

    def _ensure_leading_break(self) -> None:
        """Insert a break if the previous append did not finish with one."""
        if self._needs_leading_break:
            self._append_html("<br>", mark_content=False)

    def _append_html(self, html_content: str, *, mark_content: bool = True) -> None:
        """Append HTML content to the underlying text edit with smooth scrolling."""
        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
        self._text_edit.insertHtml(html_content)

        # Smooth scroll to bottom
        scrollbar = self._text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        self._text_edit.ensureCursorVisible()
        self._needs_leading_break = mark_content

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
