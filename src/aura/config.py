"""Configuration settings for the Aura UI."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ColorPalette:
    """Defines the Aura color palette."""

    background: str
    text: str
    accent: str
    success: str
    error: str
    agent_output: str
    thinking: str
    tool_call: str
    secondary: str
    border: str
    prompt: str
    header: str
    code_block_bg: str
    code_block_border: str
    code_keyword: str
    code_string: str
    code_comment: str


@dataclass(frozen=True)
class Icons:
    """Defines the iconography used throughout the UI."""

    SUCCESS: str = "✓"
    THINKING: str = "✨"
    READ_FILE: str = "❖"
    EDIT: str = "✓"
    WRITE_FILE: str = "✓"
    TOOL: str = "⚙"


COLORS = ColorPalette(
    background="#0d0d0d",
    text="#e6edf3",
    accent="#58a6ff",
    success="#3fb950",
    error="#f85149",
    agent_output="#e6edf3",
    thinking="#8b949e",
    tool_call="#ffa657",
    secondary="#8b949e",
    border="#21262d",
    prompt="#58a6ff",
    header="#9cdcfe",
    code_block_bg="#0a0a0a",
    code_block_border="#1f2630",
    code_keyword="#9cdcfe",
    code_string="#a5d6ff",
    code_comment="#6a9955",
)

ICONS = Icons()

FONT_FAMILY: str = "Cascadia Code, JetBrains Mono, Consolas, monospace"
FONT_SIZE_OUTPUT: int = 13  # Default output panel font size
FONT_SIZE_INPUT: int = 14  # Input field font size
FONT_SIZE_HEADER: int = 16
OUTPUT_HEADER_FONT_SIZE: int = 15
FONT_SIZE_STATUS: int = 11  # Status bar font size
LINE_HEIGHT: float = 1.6
LETTER_SPACING: str = "0.5px"
WINDOW_DIMENSIONS: tuple[int, int] = (900, 700)

DEFAULT_AGENT: str = "claude"
STREAM_PREFIX: str = "STREAM::"

AGENT_SEARCH_PATHS: list[str] = [
    os.path.join(os.getenv("APPDATA", ""), "npm"),
    os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs"),
    "/usr/local/bin",
    os.path.expanduser("~/.local/bin"),
]

AGENT_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude Code",
    "codex": "Codex",
}

AUTO_COMMIT_SESSIONS: bool = False
AUTO_PUSH_ON_COMPLETE: bool = False

OUTPUT_SECTION_SEPARATOR: str = "=" * 70
OUTPUT_SUBSECTION_SEPARATOR: str = "-" * 50
OUTPUT_SECTION_SPACING_PX: int = 14
OUTPUT_BLOCK_SPACING_PX: int = 10
CODE_BLOCK_PADDING_PX: int = 12
STREAM_CHUNK_FLUSH_THRESHOLD: int = 400

__all__ = [
    "COLORS",
    "ICONS",
    "FONT_FAMILY",
    "FONT_SIZE_OUTPUT",
    "FONT_SIZE_INPUT",
    "FONT_SIZE_HEADER",
    "OUTPUT_HEADER_FONT_SIZE",
    "FONT_SIZE_STATUS",
    "LINE_HEIGHT",
    "LETTER_SPACING",
    "WINDOW_DIMENSIONS",
    "ColorPalette",
    "Icons",
    "DEFAULT_AGENT",
    "STREAM_PREFIX",
    "AGENT_SEARCH_PATHS",
    "AGENT_DISPLAY_NAMES",
    "AUTO_COMMIT_SESSIONS",
    "AUTO_PUSH_ON_COMPLETE",
    "OUTPUT_SECTION_SEPARATOR",
    "OUTPUT_SUBSECTION_SEPARATOR",
    "OUTPUT_SECTION_SPACING_PX",
    "OUTPUT_BLOCK_SPACING_PX",
    "CODE_BLOCK_PADDING_PX",
    "STREAM_CHUNK_FLUSH_THRESHOLD",
]
