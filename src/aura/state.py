"""Application state management for Aura.

This module centralizes all application-level state and provides
Qt signals for state change notifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


class AppState(QObject):
    """Centralized application state with signal-based notifications.

    This class decouples the UI from business logic by managing all
    application-level state and emitting signals when state changes.
    """

    # Signals for state changes
    working_directory_changed = Signal(str)
    selected_agent_changed = Signal(str)
    agent_path_changed = Signal(str)
    status_changed = Signal(str, str)  # message, color
    gemini_model_changed = Signal(str)
    claude_model_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize application state with default values."""
        super().__init__(parent)
        self._working_directory: str = ""
        self._selected_agent: str = ""
        self._agent_path: str = ""
        self._status_message: str = "Ready"
        self._status_color: str = "#ffffff"
        self._selected_gemini_model: str = "gemini-1.5-pro-latest"
        self._selected_claude_model: str = "claude-3-sonnet-20240229"

    @property
    def working_directory(self) -> str:
        """Get the current working directory."""
        return self._working_directory

    def set_working_directory(self, path: str) -> None:
        """Set the working directory and emit change signal.

        Args:
            path: New working directory path

        Raises:
            ValueError: If path is empty
            FileNotFoundError: If path does not exist
        """
        if not path:
            raise ValueError("Working directory must be provided.")
        resolved = str(Path(path).resolve())
        if not Path(resolved).is_dir():
            raise FileNotFoundError(f"Directory does not exist: {resolved}")

        if self._working_directory != resolved:
            self._working_directory = resolved
            self.working_directory_changed.emit(resolved)

    @property
    def selected_agent(self) -> str:
        """Get the currently selected agent."""
        return self._selected_agent

    def set_selected_agent(self, agent: str) -> None:
        """Set the selected agent and emit change signal.

        Args:
            agent: Agent name to select
        """
        if self._selected_agent != agent:
            self._selected_agent = agent
            self.selected_agent_changed.emit(agent)

    @property
    def agent_path(self) -> str:
        """Get the current agent executable path."""
        return self._agent_path

    def set_agent_path(self, path: str) -> None:
        """Set the agent path and emit change signal.

        Args:
            path: Path to agent executable
        """
        if self._agent_path != path:
            self._agent_path = path
            self.agent_path_changed.emit(path)

    @property
    def status_message(self) -> str:
        """Get the current status message."""
        return self._status_message

    @property
    def status_color(self) -> str:
        """Get the current status color."""
        return self._status_color

    def set_status(self, message: str, color: str) -> None:
        """Set the status message and color, emit change signal.

        Args:
            message: Status message to display
            color: CSS color for the status
        """
        if self._status_message != message or self._status_color != color:
            self._status_message = message
            self._status_color = color
            self.status_changed.emit(message, color)

    @property
    def selected_gemini_model(self) -> str:
        """Get the selected Gemini model."""
        return self._selected_gemini_model

    def set_gemini_model(self, model_id: str) -> None:
        """Set the selected Gemini model and emit change signal."""
        if self._selected_gemini_model != model_id:
            self._selected_gemini_model = model_id
            self.gemini_model_changed.emit(model_id)

    @property
    def selected_claude_model(self) -> str:
        """Get the selected Claude model."""
        return self._selected_claude_model

    def set_claude_model(self, model_id: str) -> None:
        """Set the selected Claude model and emit change signal."""
        if self._selected_claude_model != model_id:
            self._selected_claude_model = model_id
            self.claude_model_changed.emit(model_id)
