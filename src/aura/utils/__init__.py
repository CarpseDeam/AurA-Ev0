"""Utility modules for the Aura application."""

from .file_filter import is_file_protected, load_gitignore_patterns
from .model_discovery import (
    ModelInfo,
    discover_claude_models,
    discover_gemini_models,
    get_available_models,
)
from .project_scanner import scan_directory
from .safety import is_safe_working_directory
from .settings import load_settings, save_settings

__all__ = [
    "ModelInfo",
    "discover_claude_models",
    "discover_gemini_models",
    "get_available_models",
    "is_file_protected",
    "is_safe_working_directory",
    "load_gitignore_patterns",
    "load_settings",
    "save_settings",
    "scan_directory",
]
