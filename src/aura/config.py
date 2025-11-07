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


COLORS = ColorPalette(
    background="#0a0e14",
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
)

FONT_FAMILY: str = "Cascadia Code, JetBrains Mono, Consolas, monospace"
FONT_SIZE_OUTPUT: int = 14  # Increased from 13 for better readability
FONT_SIZE_INPUT: int = 14  # Input field font size
FONT_SIZE_HEADER: int = 16
FONT_SIZE_STATUS: int = 11  # Status bar font size
LINE_HEIGHT: float = 1.6
LETTER_SPACING: str = "0.5px"
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

AUTO_COMMIT_SESSIONS: bool = False
AUTO_PUSH_ON_COMPLETE: bool = False

__all__ = [
    "COLORS",
    "FONT_FAMILY",
    "FONT_SIZE_OUTPUT",
    "FONT_SIZE_INPUT",
    "FONT_SIZE_HEADER",
    "FONT_SIZE_STATUS",
    "LINE_HEIGHT",
    "LETTER_SPACING",
    "WINDOW_DIMENSIONS",
    "ColorPalette",
    "DEFAULT_AGENT",
    "STREAM_PREFIX",
    "AGENT_SEARCH_PATHS",
    "AGENT_DISPLAY_NAMES",
    "AUTO_COMMIT_SESSIONS",
    "AUTO_PUSH_ON_COMPLETE",
]
