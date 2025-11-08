"""Application state management for Aura.

This module centralizes all application-level state and provides
Qt signals for state change notifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

_APP_STATE: "AppState | None" = None


def get_app_state() -> "AppState | None":
    """Return the global AppState instance if one has been created."""
    return _APP_STATE


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
    analyst_model_changed = Signal(str)
    executor_model_changed = Signal(str)
    specialist_model_changed = Signal(str)
    local_model_endpoint_changed = Signal(str)
    current_project_changed = Signal(object)  # project_id (int or None)
    current_conversation_changed = Signal(object)  # conversation_id (int or None)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize application state with default values."""
        super().__init__(parent)
        self._working_directory: str = ""
        self._selected_agent: str = ""
        self._agent_path: str = ""
        self._status_message: str = "Ready"
        self._status_color: str = "#ffffff"
        self._analyst_model: str = "gemini-1.5-pro-latest"
        self._executor_model: str = "claude-3-sonnet-20240229"
        self._specialist_model: str = "phi-3-mini"
        self._local_model_endpoint: str = ""
        self._current_project_id: Optional[int] = None
        self._current_conversation_id: Optional[int] = None
        global _APP_STATE
        _APP_STATE = self

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
    def analyst_model(self) -> str:
        """Get the selected analyst model."""
        return self._analyst_model

    def set_analyst_model(self, model_id: str) -> None:
        """Set the analyst model and emit change signal."""
        if self._analyst_model != model_id:
            self._analyst_model = model_id
            self.analyst_model_changed.emit(model_id)

    @property
    def executor_model(self) -> str:
        """Get the selected executor model."""
        return self._executor_model

    def set_executor_model(self, model_id: str) -> None:
        """Set the executor model and emit change signal."""
        if self._executor_model != model_id:
            self._executor_model = model_id
            self.executor_model_changed.emit(model_id)

    @property
    def specialist_model(self) -> str:
        """Get the specialist model used by local tools."""
        return self._specialist_model

    def set_specialist_model(self, model_name: str) -> None:
        """Set the specialist model and emit change signal."""
        if self._specialist_model != model_name:
            self._specialist_model = model_name
            self.specialist_model_changed.emit(model_name)

    @property
    def local_model_endpoint(self) -> str:
        """Get the local model endpoint."""
        return self._local_model_endpoint

    def set_local_model_endpoint(self, endpoint: str) -> None:
        """Set the local model endpoint and emit change signal."""
        if self._local_model_endpoint != endpoint:
            self._local_model_endpoint = endpoint
            self.local_model_endpoint_changed.emit(endpoint)

    @property
    def current_project_id(self) -> Optional[int]:
        """Get the current project ID."""
        return self._current_project_id

    def set_current_project(self, project_id: Optional[int]) -> None:
        """Set the current project and emit change signal.

        Args:
            project_id: Project ID to set as current, or None to clear
        """
        if self._current_project_id != project_id:
            self._current_project_id = project_id
            self.current_project_changed.emit(project_id)

    @property
    def current_conversation_id(self) -> Optional[int]:
        """Get the current conversation ID."""
        return self._current_conversation_id

    def set_current_conversation(self, conversation_id: Optional[int]) -> None:
        """Set the current conversation and emit change signal.

        Args:
            conversation_id: Conversation ID to set as current, or None to clear
        """
        if self._current_conversation_id != conversation_id:
            self._current_conversation_id = conversation_id
            self.current_conversation_changed.emit(conversation_id)
