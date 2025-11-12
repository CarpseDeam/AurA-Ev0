"""Application state management for Aura.

This module centralizes all application-level state and provides
Qt signals for state change notifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal
from aura.utils.settings import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_ANTHROPIC_API_KEY,
    DEFAULT_COST_TRACKING,
    DEFAULT_MAX_TOKENS_BUDGET,
    DEFAULT_SPECIALIST_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOL_CALL_LIMIT,
)

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
    status_changed = Signal(str, str)  # message, color
    agent_model_changed = Signal(str)
    anthropic_api_key_changed = Signal(str)
    max_tokens_budget_changed = Signal(int)
    tool_call_limit_changed = Signal(int)
    temperature_changed = Signal(float)
    cost_tracking_changed = Signal(bool)
    specialist_model_changed = Signal(str)
    local_model_endpoint_changed = Signal(str)
    use_local_investigation_changed = Signal(bool)
    current_project_changed = Signal(object)  # project_id (int or None)
    current_conversation_changed = Signal(object)  # conversation_id (int or None)

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize application state with default values."""
        super().__init__(parent)
        self._working_directory: str = ""
        self._status_message: str = "Ready"
        self._status_color: str = "#ffffff"
        self._agent_model: str = DEFAULT_AGENT_MODEL
        self._anthropic_api_key: str = DEFAULT_ANTHROPIC_API_KEY
        self._max_tokens_budget: int = DEFAULT_MAX_TOKENS_BUDGET
        self._tool_call_limit: int = DEFAULT_TOOL_CALL_LIMIT
        self._temperature: float = DEFAULT_TEMPERATURE
        self._cost_tracking_enabled: bool = DEFAULT_COST_TRACKING
        self._specialist_model: str = DEFAULT_SPECIALIST_MODEL
        self._local_model_endpoint: str = ""
        self._use_local_investigation: bool = False
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
    def agent_model(self) -> str:
        """Get the configured single-agent model identifier."""
        return self._agent_model

    def set_agent_model(self, model_id: str) -> None:
        """Persist the selected agent model and emit change signal."""
        sanitized = (model_id or "").strip()
        if sanitized and self._agent_model != sanitized:
            self._agent_model = sanitized
            self.agent_model_changed.emit(sanitized)

    @property
    def anthropic_api_key(self) -> str:
        """Return the stored Anthropic API key."""
        return self._anthropic_api_key

    def set_anthropic_api_key(self, api_key: str) -> None:
        """Update the Anthropic API key."""
        sanitized = (api_key or "").strip()
        if self._anthropic_api_key != sanitized:
            self._anthropic_api_key = sanitized
            self.anthropic_api_key_changed.emit(sanitized)

    @property
    def max_tokens_budget(self) -> int:
        """Return the maximum tokens available per request."""
        return self._max_tokens_budget

    def set_max_tokens_budget(self, tokens: int) -> None:
        """Set the maximum Claude token budget."""
        try:
            value = int(tokens)
        except (TypeError, ValueError):
            value = DEFAULT_MAX_TOKENS_BUDGET
        value = max(1_000, min(value, 400_000))
        if self._max_tokens_budget != value:
            self._max_tokens_budget = value
            self.max_tokens_budget_changed.emit(value)

    @property
    def tool_call_limit(self) -> int:
        """Return the maximum allowed tool iterations."""
        return self._tool_call_limit

    def set_tool_call_limit(self, limit: int) -> None:
        """Set the total allowed tool calls for a run."""
        try:
            value = int(limit)
        except (TypeError, ValueError):
            value = DEFAULT_TOOL_CALL_LIMIT
        value = max(1, min(value, 100))
        if self._tool_call_limit != value:
            self._tool_call_limit = value
            self.tool_call_limit_changed.emit(value)

    @property
    def temperature(self) -> float:
        """Return the sampling temperature."""
        return self._temperature

    def set_temperature(self, temperature: float) -> None:
        """Set the Claude sampling temperature."""
        try:
            value = float(temperature)
        except (TypeError, ValueError):
            value = DEFAULT_TEMPERATURE
        value = max(0.0, min(value, 1.0))
        if self._temperature != value:
            self._temperature = value
            self.temperature_changed.emit(value)

    @property
    def cost_tracking_enabled(self) -> bool:
        """Get whether to log estimated Anthropic costs."""
        return self._cost_tracking_enabled

    def set_cost_tracking_enabled(self, enabled: bool) -> None:
        """Enable or disable cost tracking logs."""
        normalized = bool(enabled)
        if self._cost_tracking_enabled != normalized:
            self._cost_tracking_enabled = normalized
            self.cost_tracking_changed.emit(normalized)

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
    def use_local_investigation(self) -> bool:
        """Get whether to use local model for investigation phase."""
        return self._use_local_investigation

    def set_use_local_investigation(self, enabled: bool) -> None:
        """Set whether to use local model for investigation and emit change signal."""
        if self._use_local_investigation != enabled:
            self._use_local_investigation = enabled
            self.use_local_investigation_changed.emit(enabled)

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
