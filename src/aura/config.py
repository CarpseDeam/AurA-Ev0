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
    background="#1e1e1e",
    text="#e0e0e0",
    accent="#64B5F6",
    success="#66BB6A",
    error="#FF6B6B",
    agent_output="#FFD27F",
    thinking="#9370DB",
    tool_call="#FFD27F",
    secondary="#888888",
)

FONT_FAMILY: str = "JetBrains Mono"
WINDOW_DIMENSIONS: tuple[int, int] = (900, 700)

DEFAULT_AGENT: str = "gemini"

AGENT_SEARCH_PATHS: list[str] = [
    os.path.join(os.getenv("APPDATA", ""), "npm"),
    os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs"),
    "/usr/local/bin",
    os.path.expanduser("~/.local/bin"),
]

AGENT_DISPLAY_NAMES: dict[str, str] = {
    "gemini": "Gemini CLI",
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
    "WINDOW_DIMENSIONS",
    "ColorPalette",
    "DEFAULT_AGENT",
    "AGENT_SEARCH_PATHS",
    "AGENT_DISPLAY_NAMES",
    "USE_NATIVE_PYTHON_AGENT",
    "AUTO_COMMIT_SESSIONS",
    "AUTO_PUSH_ON_COMPLETE",
]
