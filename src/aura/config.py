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


COLORS = ColorPalette(
    background="#000000",
    text="#e8e8e8",
    accent="#64B5F6",
    success="#66BB6A",
    error="#FF6B6B",
    agent_output="#FFD27F",
    thinking="#9370DB",
    tool_call="#FFD27F",
    secondary="#b0b0b0",
)

FONT_FAMILY: str = "JetBrains Mono"
FONT_SIZE_OUTPUT: int = 14  # Increased from 13 for better readability
FONT_SIZE_INPUT: int = 14  # Input field font size
FONT_SIZE_HEADER: int = 16
FONT_SIZE_STATUS: int = 11  # Status bar font size
WINDOW_DIMENSIONS: tuple[int, int] = (900, 700)

DEFAULT_AGENT: str = "gemini"
STREAM_PREFIX: str = "STREAM::"

AGENT_SEARCH_PATHS: list[str] = [
    os.path.join(os.getenv("APPDATA", ""), "npm"),
    os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs"),
    "/usr/local/bin",
    os.path.expanduser("~/.local/bin"),
]

AGENT_DISPLAY_NAMES: dict[str, str] = {
    "gemini": "Gemini 2.5 Pro",
    "claude": "Claude Code",
    "codex": "Codex",
}

# PythonCoderAgent Configuration
# Set to True to use the native PythonCoderAgent (recommended for better performance)
# Set to False to use the CLI wrapper (fallback for compatibility)
USE_NATIVE_PYTHON_AGENT: bool = True
AUTO_COMMIT_SESSIONS: bool = False
AUTO_PUSH_ON_COMPLETE: bool = False

__all__ = [
    "COLORS",
    "FONT_FAMILY",
    "FONT_SIZE_OUTPUT",
    "FONT_SIZE_INPUT",
    "FONT_SIZE_HEADER",
    "FONT_SIZE_STATUS",
    "WINDOW_DIMENSIONS",
    "ColorPalette",
    "DEFAULT_AGENT",
    "STREAM_PREFIX",
    "AGENT_SEARCH_PATHS",
    "AGENT_DISPLAY_NAMES",
    "USE_NATIVE_PYTHON_AGENT",
    "AUTO_COMMIT_SESSIONS",
    "AUTO_PUSH_ON_COMPLETE",
]
