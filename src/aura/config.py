"""Configuration settings for the Aura UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColorPalette:
    """Defines the Aura color palette."""

    background: str
    text: str
    accent: str
    success: str
    agent_output: str


COLORS = ColorPalette(
    background="#1e1e1e",
    text="#e0e0e0",
    accent="#64B5F6",
    success="#66BB6A",
    agent_output="#FFD27F",
)

FONT_FAMILY: str = "JetBrains Mono"
WINDOW_DIMENSIONS: tuple[int, int] = (900, 700)

__all__ = ["COLORS", "FONT_FAMILY", "WINDOW_DIMENSIONS", "ColorPalette"]
