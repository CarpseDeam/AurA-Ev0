"""Status bar composition helper for Aura's main window."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QLabel, QStatusBar

from src.aura import config
from src.aura.state import AppState


class StatusBarManager(QObject):
    """Owns the status bar content and reacts to application state changes."""

    def __init__(
        self,
        status_bar: QStatusBar,
        app_state: AppState,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._status_bar = status_bar
        self._app_state = app_state

        self._status_label = QLabel(self._status_bar)
        self._separator = QLabel("|", self._status_bar)
        self._directory_label = QLabel(self._status_bar)

        self._setup_status_bar()
        self._connect_signals()

    @property
    def status_bar(self) -> QStatusBar:
        """Return the managed status bar instance."""
        return self._status_bar

    def update_status(self, message: str, color: str, *, persist: bool = False) -> None:
        """Update the status text and optionally persist to AppState."""
        self._apply_status(message, color)
        if persist:
            self._app_state.set_status(message, color)

    def update_directory(self, path: str, *, persist: bool = False) -> None:
        """Update the visible working directory path."""
        formatted = self._format_directory(path)
        self._directory_label.setText(formatted)
        if persist:
            self._app_state.set_working_directory(path)

    def _setup_status_bar(self) -> None:
        """Configure the status bar labels and layout."""
        self._status_label.setText("Ready")
        self._status_label.setStyleSheet(
            f"color: {config.COLORS.secondary}; font-weight: 400; padding: 2px 8px;"
        )

        self._separator.setStyleSheet("color: #333333; padding: 0 8px;")

        initial_directory = self._format_directory(self._app_state.working_directory)
        self._directory_label.setText(initial_directory)
        self._directory_label.setStyleSheet(
            f"color: {config.COLORS.secondary}; padding: 2px 8px; font-size: 11px;"
        )

        # Order matches the previous implementation to maintain layout.
        self._status_bar.addWidget(self._status_label, 0)
        self._status_bar.addWidget(self._separator, 0)
        self._status_bar.addPermanentWidget(self._directory_label, 1)

    def _connect_signals(self) -> None:
        """Listen for AppState changes that affect the status bar."""
        self._app_state.status_changed.connect(self._on_status_changed)
        self._app_state.working_directory_changed.connect(self._on_working_directory_changed)

    def _apply_status(self, message: str, color: str) -> None:
        """Apply the status text and color styling."""
        self._status_label.setText(message)
        self._status_label.setStyleSheet(f"color: {color}; font-weight: 400; padding: 2px 8px;")

    def _on_status_changed(self, message: str, color: str) -> None:
        """Synchronize with AppState status updates."""
        self._apply_status(message, color)

    def _on_working_directory_changed(self, path: str) -> None:
        """Update the directory label from AppState notifications."""
        self._directory_label.setText(self._format_directory(path))

    @staticmethod
    def _format_directory(path: str) -> str:
        """Truncate overly long directory strings."""
        if len(path) <= 50:
            return path
        return "..." + path[-47:]
