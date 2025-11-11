"""Application entry point for Aura."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from aura import config
from aura.exceptions import AuraConfigurationError
from aura.orchestrator import Orchestrator
from aura.services import ChatService
from aura.state import AppState
from aura.ui.main_window import MainWindow
from aura.ui.cli_heartbeat_display import CliHeartbeatDisplay
from aura.utils import load_settings
from aura.utils.settings import (
    DEFAULT_ANALYST_INVESTIGATION_MODEL,
    DEFAULT_ANALYST_PLANNING_MODEL,
    DEFAULT_EXECUTOR_MODEL,
    DEFAULT_SPECIALIST_MODEL,
)
from aura.database import initialize_database
from aura.models import Conversation


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure application-wide structured logging outputs."""
    logs_root = Path(__file__).resolve().parents[2] / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    log_path = logs_root / "aura.log"

    root_logger = logging.getLogger()
    if getattr(configure_logging, "_configured", False):
        return

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)



    configure_logging._configured = True  # type: ignore[attr-defined]


configure_logging()


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
        self.cli_heartbeat_display: CliHeartbeatDisplay | None = None

    def setup(self) -> MainWindow:
        """Create and wire all application components.

        Returns:
            Configured MainWindow ready to display
        """
        # Initialize application state
        self.app_state = AppState()

        # Initialize database
        try:
            initialize_database()
            LOGGER.info("Database initialized successfully")
        except Exception as e:
            LOGGER.error(f"Failed to initialize database: {e}")

        # Set initial working directory
        app_source_path = Path(__file__).resolve().parent.parent
        workspace_path = app_source_path.parent / "aura-workspace"
        workspace_path.mkdir(parents=True, exist_ok=True)
        self.app_state.set_working_directory(str(workspace_path))

        # Load settings and apply them to the AppState
        settings = load_settings()
        planning_model = settings.get(
            "analyst_planning_model",
            settings.get("analyst_model", DEFAULT_ANALYST_PLANNING_MODEL),
        )
        investigation_model = settings.get(
            "analyst_investigation_model",
            DEFAULT_ANALYST_INVESTIGATION_MODEL,
        )
        self.app_state.set_analyst_planning_model(planning_model)
        self.app_state.set_analyst_investigation_model(investigation_model)
        self.app_state.set_executor_model(settings.get("executor_model", DEFAULT_EXECUTOR_MODEL))
        self.app_state.set_specialist_model(settings.get("specialist_model", DEFAULT_SPECIALIST_MODEL))
        
        orchestrator_warning: str | None = None

        try:
            analyst_key = self._require_analyst_api_key()
            executor_key = self._get_executor_api_key(fallback=analyst_key)

            self.orchestrator = Orchestrator(
                app_state=self.app_state,
                analyst_api_key=analyst_key,
                executor_api_key=executor_key,
            )
            self.chat_service = self.orchestrator.chat_service
        except AuraConfigurationError as exc:
            orchestrator_warning = str(exc)
            LOGGER.error("Failed to initialize orchestrator: %s", exc)

        # Create main window with dependencies
        self.main_window = MainWindow(
            app_state=self.app_state,
            orchestrator=self.orchestrator,
        )

        # Create CLI heartbeat display
        self.cli_heartbeat_display = CliHeartbeatDisplay()

        # Load most recent conversation
        self._load_last_conversation()

        # Wire signals and slots
        self._connect_signals()

        if orchestrator_warning:
            self.main_window.output_panel.display_error(orchestrator_warning)

        return self.main_window

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        if not self.main_window or not self.app_state or not self.cli_heartbeat_display:
            return

        self.cli_heartbeat_display.new_message.connect(
            self.main_window.output_panel.display_output,
            Qt.ConnectionType.UniqueConnection,
        )

        if self.orchestrator:
            self.app_state.working_directory_changed.connect(
                self._on_working_directory_changed,
                Qt.ConnectionType.UniqueConnection,
            )
            self.app_state.working_directory_changed.connect(
                self.orchestrator.update_working_directory
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

    def _require_analyst_api_key(self) -> str:
        """Return a validated analyst API key.

        Raises:
            AuraConfigurationError: If the Anthropic API key is not set

        Returns:
            The analyst API key
        """
        api_key = (
            os.getenv("ANTHROPIC_ANALYST_API_KEY", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
        )
        if not api_key:
            raise AuraConfigurationError(
                "Analyst API key required for Aura analysis. "
                "Set ANTHROPIC_API_KEY (or ANTHROPIC_ANALYST_API_KEY) and restart Aura.",
                context={"env_var": "ANTHROPIC_API_KEY"},
            )
        return api_key

    def _get_executor_api_key(self, fallback: str | None = None) -> str | None:
        """Return executor API key if available.

        Returns:
            The executor API key or None if not set. When None, Aura falls
            back to single-agent mode using only the analyst agent.
        """
        api_key = (
            os.getenv("ANTHROPIC_EXECUTOR_API_KEY", "").strip()
            or os.getenv("ANTHROPIC_API_KEY", "").strip()
            or (fallback or "")
        )
        if not api_key:
            LOGGER.warning(
                "ANTHROPIC_EXECUTOR_API_KEY not set. Running in single-agent mode (analyst only). "
                "Set ANTHROPIC_EXECUTOR_API_KEY to enable executor write access."
            )
        return api_key or None

    def _load_last_conversation(self) -> None:
        """Load the most recent conversation on startup."""
        if not self.main_window or not self.app_state or not self.orchestrator:
            return

        try:
            # Refresh sidebars first
            self.main_window.refresh_sidebars()

            # Load most recent conversation
            conv = Conversation.get_most_recent()
            if conv:
                self.app_state.set_current_conversation(conv.id)

                # Set project if conversation has one
                if conv.project_id:
                    self.app_state.set_current_project(conv.project_id)
                    self.main_window.project_sidebar.set_current_project(conv.project_id)

                # Load conversation history
                self.orchestrator.load_conversation_history(conv.id)

                # Display conversation in output panel
                self.main_window.output_panel.display_output(
                    f"Loaded conversation: {conv.title or '(Untitled)'}",
                    config.COLORS.accent
                )

                # Display conversation history
                messages = conv.get_messages()
                for msg in messages:
                    if msg.role == "user":
                        self.main_window.output_panel.display_output(
                            f"> {msg.content}",
                            config.COLORS.prompt,
                        )
                    else:
                        self.main_window.output_panel.display_output(
                            msg.content,
                            config.COLORS.agent_output,
                        )

                # Update sidebar highlighting
                self.main_window.project_sidebar.set_current_conversation(conv.id)

                LOGGER.info(f"Loaded last conversation: {conv.title} (ID: {conv.id})")
            else:
                LOGGER.info("No previous conversations found")

        except Exception as e:
            LOGGER.error(f"Failed to load last conversation: {e}")


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

