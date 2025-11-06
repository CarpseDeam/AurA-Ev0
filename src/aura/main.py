"""Application entry point for Aura."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from aura import config
from aura.orchestrator import Orchestrator
from aura.services import ChatService
from aura.state import AppState
from aura.ui.main_window import MainWindow


# Suppress SDK spam
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
for sdk_logger in [
    "google.genai",
    "google_genai",
    "google.generativeai",
    "google.ai.generativelanguage",
]:
    logging.getLogger(sdk_logger).setLevel(logging.WARNING)


class ApplicationController:
    """Coordinates application components and wires them together.

    This controller implements the dependency injection pattern,
    creating all major components and connecting their signals/slots.
    """

    def __init__(self) -> None:
        """Initialize the application controller."""
        self.app_state: AppState | None = None
        self.chat_service: ChatService | None = None
        self.orchestrator: Orchestrator | None = None
        self.main_window: MainWindow | None = None

    def setup(self) -> MainWindow:
        """Create and wire all application components.

        Returns:
            Configured MainWindow ready to display
        """
        # Initialize application state
        self.app_state = AppState()

        # Set initial working directory
        app_source_path = Path(__file__).resolve().parent.parent
        workspace_path = app_source_path.parent / "aura-workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        self.app_state.set_working_directory(str(workspace_path))
        self.app_state.set_selected_agent(config.DEFAULT_AGENT)

        # Initialize services if API key is available
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            self.chat_service = ChatService(api_key=api_key)
            self.orchestrator = Orchestrator(
                self.chat_service,
                self.app_state.working_directory,
                self.app_state.agent_path or "",
            )

        # Create main window with dependencies
        self.main_window = MainWindow(
            app_state=self.app_state,
            orchestrator=self.orchestrator,
        )

        # Wire signals and slots
        self._connect_signals()

        return self.main_window

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        if not self.main_window or not self.app_state:
            return

        if self.orchestrator:
            self.app_state.working_directory_changed.connect(
                self._on_working_directory_changed,
                Qt.ConnectionType.UniqueConnection,
            )

            self.main_window.execution_requested.connect(
                self.orchestrator.execute_goal,
                Qt.ConnectionType.UniqueConnection,
            )

            self.orchestrator.progress_update.connect(
                self.main_window._on_progress_update,
                Qt.ConnectionType.UniqueConnection,
            )

    def _on_working_directory_changed(self, path: str) -> None:
        """Handle working directory changes.

        Args:
            path: New working directory path
        """
        if not self.orchestrator:
            return

        # Update orchestrator workspace without recreating it
        try:
            self.orchestrator.update_working_directory(path)
        except FileNotFoundError as exc:
            logging.getLogger("aura").error("Failed to update working directory: %s", exc)
            self.main_window._on_progress_update("Invalid working directory")


def _create_application() -> QApplication:
    """Create the Qt application instance."""
    existing = QApplication.instance()
    if existing is not None:
        return existing
    app = QApplication(sys.argv)
    app.setApplicationName("Aura")
    return app


def run() -> int:
    """Run the Aura UI event loop."""
    app = _create_application()

    # Create and setup application controller
    controller = ApplicationController()
    window = controller.setup()

    window.show()
    return app.exec()


def main() -> None:
    """Launch the Aura application."""
    try:
        exit_code = run()
    except Exception:  # noqa: BLE001
        logging.getLogger("aura").exception("Aura terminated unexpectedly")
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
