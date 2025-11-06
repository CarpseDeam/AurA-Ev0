"""Main application window for Aura."""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from aura import config
from aura.events import Event, EventType, get_event_bus
from aura.orchestrator import Orchestrator
from aura.services import AgentRunner
from aura.state import AppState
from aura.ui import AgentSettingsDialog
from aura.ui.agent_execution_manager import AgentExecutionManager
from aura.ui.orchestration_handler import OrchestrationHandler
from aura.ui.output_panel import OutputPanel
from aura.ui.status_bar_manager import StatusBarManager

LOGGER = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Displays the primary Aura workspace."""

    _event_received = Signal(object)
    execution_requested = Signal(str)  # Emitted when user requests execution

    def __init__(
        self,
        app_state: AppState,
        orchestrator: Optional[Orchestrator] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.app_state = app_state
        self.orchestrator = orchestrator
        self.output_panel = OutputPanel(self)
        self.input_field = QLineEdit(self); self.clear_button = QPushButton("Clear", self)
        self.toolbar = self.addToolBar("Project")
        self.current_runner: Optional[AgentRunner] = None; self._event_bus = get_event_bus()

        status_bar = QStatusBar(self); self.status_bar_manager = StatusBarManager(status_bar, self.app_state, self)
        self.setStatusBar(self.status_bar_manager.status_bar)

        self.orchestration_handler = OrchestrationHandler(
            output_panel=self.output_panel,
            status_manager=self.status_bar_manager,
            app_state=self.app_state,
            parent=self,
        )
        self.agent_manager = AgentExecutionManager(
            app_state=self.app_state,
            output_panel=self.output_panel,
            status_manager=self.status_bar_manager,
            orchestrator=self.orchestrator,
        )

        self._configure_window(); self._build_layout(); self._apply_styles(); self._build_toolbar()
        self._connect_signals(); self._connect_handler_signals(); self._subscribe_to_events()
        self._event_received.connect(self.orchestration_handler.handle_background_event)

        self.status_bar_manager.update_status("Ready", config.COLORS.text, persist=True)

        if self.orchestrator:
            self._connect_orchestrator_signals()
            self.output_panel.display_startup_header()
            self.output_panel.display_output("Aura orchestration ready", config.COLORS.success)

        self.agent_manager.detect_default_agent()

    def _configure_window(self) -> None:
        self.setWindowTitle("Aura"); self.resize(*config.WINDOW_DIMENSIONS)

        # Configure input field with larger font for better readability
        input_font = QFont(config.FONT_FAMILY)
        input_font.setPointSize(config.FONT_SIZE_INPUT)
        self.input_field.setPlaceholderText("Enter a request")
        self.input_field.setClearButtonEnabled(True)
        self.input_field.setFont(input_font)
        self.input_field.setFocus()

    def _build_layout(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(16, 16, 16, 16); layout.setSpacing(12); layout.addWidget(self.output_panel)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_field); self.clear_button.setFixedWidth(72); input_row.addWidget(self.clear_button)
        layout.addLayout(input_row); layout.setStretch(0, 1); layout.setStretch(1, 0); self.setCentralWidget(container)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            f"QMainWindow {{background: {config.COLORS.background}; color: {config.COLORS.text};}}"
            f"QTextEdit {{background: {config.COLORS.background}; "
            f"color: {config.COLORS.text}; border: none; padding: 16px; "
            f"selection-background-color: {config.COLORS.accent};}}"
            f"QLineEdit {{background: {config.COLORS.background}; "
            f"color: {config.COLORS.text}; border: none; border-bottom: 1px solid #333333; padding: 10px 14px; "
            f"font-size: {config.FONT_SIZE_INPUT}px;}}"
            f"QLineEdit:focus {{border-bottom: 1px solid #333333;}}"
            f"QStatusBar {{background: {config.COLORS.background}; "
            f"color: {config.COLORS.text}; border: none; padding: 6px; "
            f"font-size: {config.FONT_SIZE_STATUS}px;}}"
            f"QToolBar {{background: {config.COLORS.background}; "
            "border: none; spacing: 8px; padding: 4px;}}"
        )
        self.clear_button.setStyleSheet(
            f"QPushButton {{background: {config.COLORS.background}; "
            f"color: {config.COLORS.text}; border: 1px solid #333333; padding: 8px 12px; "
            f"font-size: {config.FONT_SIZE_STATUS}px; font-weight: 500;}}"
            f"QPushButton:hover {{border-color: {config.COLORS.accent};}}"
            f"QPushButton:pressed {{background: #111111;}}"
        )

    def _build_toolbar(self) -> None:
        self.toolbar.setMovable(False)
        dir_action = QAction("Set Working Directory", self); dir_action.triggered.connect(self._select_working_directory); self.toolbar.addAction(dir_action)
        agent_action = QAction("Agent Settings...", self); agent_action.triggered.connect(self._open_agent_settings); self.toolbar.addAction(agent_action)

    def _connect_signals(self) -> None:
        self.input_field.returnPressed.connect(self._handle_submit); self.clear_button.clicked.connect(self.clear_output)

    def _connect_handler_signals(self) -> None:
        self.orchestration_handler.request_input_enabled.connect(self._set_input_enabled); self.orchestration_handler.request_input_focus.connect(self.input_field.setFocus)

    def _subscribe_to_events(self) -> None:
        self._event_bus.subscribe(EventType.ERROR, self._emit_event_signal)

    def _emit_event_signal(self, event: Event) -> None:
        self._event_received.emit(event)

    def _connect_orchestrator_signals(self) -> None:
        from PySide6.QtCore import Qt

        assert self.orchestrator is not None
        # Use UniqueConnection to prevent duplicate connections if called multiple times
        self.orchestrator.planning_started.connect(
            self.orchestration_handler.handle_planning_started,
            Qt.ConnectionType.UniqueConnection
        )
        self.orchestrator.plan_ready.connect(
            self.orchestration_handler.handle_plan_ready,
            Qt.ConnectionType.UniqueConnection
        )
        self.orchestrator.session_started.connect(
            self.orchestration_handler.handle_session_started,
            Qt.ConnectionType.UniqueConnection
        )
        self.orchestrator.session_output.connect(
            self.orchestration_handler.handle_session_output,
            Qt.ConnectionType.UniqueConnection
        )
        self.orchestrator.session_complete.connect(
            self.orchestration_handler.handle_session_complete,
            Qt.ConnectionType.UniqueConnection
        )
        self.orchestrator.all_sessions_complete.connect(
            self.orchestration_handler.handle_all_complete,
            Qt.ConnectionType.UniqueConnection
        )
        self.orchestrator.error_occurred.connect(
            self.orchestration_handler.handle_error,
            Qt.ConnectionType.UniqueConnection
        )

    def _handle_submit(self) -> None:
        prompt = self.input_field.text().strip()
        if not prompt:
            return
        if self.current_runner and self.current_runner.isRunning():
            self.output_panel.display_output("An agent run is already in progress.", "#FF6B6B")
            return
        self.input_field.clear(); self.output_panel.display_output(f"> {prompt}", config.COLORS.accent)
        self._set_input_enabled(False)
        if not self.orchestrator:
            self.execute_command(prompt); return
        normalized = prompt.lower()
        approval_keywords = {"start", "yes", "go", "build it", "lets do it", "let's do it"}
        if self.app_state.current_plan and normalized in approval_keywords:
            return
        if self._should_orchestrate(prompt):
            import threading
            LOGGER.info("EXECUTION_REQUESTED: Emitting execution_requested signal (thread: %s, goal: %s)",
                       threading.current_thread().name, prompt[:50])
            self.execution_requested.emit(prompt)
        else:
            self.execute_command(prompt)

    def execute_command(self, prompt: str) -> None:
        if not prompt or not self.agent_manager.validate_environment():
            self._set_input_enabled(True); self.input_field.setFocus(); return
        command_prompt = self.agent_manager.compose_prompt(prompt)
        agent_executable = self.app_state.agent_path or self.app_state.selected_agent
        command = [agent_executable, "-p", command_prompt, "--yolo"]
        try:
            runner = AgentRunner(command=command, working_directory=self.app_state.working_directory, parent=self)
        except ValueError as exc:
            self.output_panel.display_output(f"Unable to start agent: {exc}", "#FF6B6B")
            self._set_input_enabled(True); self.input_field.setFocus(); return
        runner.output_line.connect(self.output_panel.display_output)
        runner.process_finished.connect(self.handle_process_finished); runner.process_error.connect(self.handle_process_error)
        self.current_runner = runner
        self.status_bar_manager.update_status("Running...", config.COLORS.accent, persist=True)
        runner.start()

    def _should_orchestrate(self, prompt: str) -> bool:
        lower = prompt.lower()
        if any(word in lower for word in ["build", "create app", "add feature", "implement"]):
            return len(prompt) > 30
        return len(prompt) > 50

    def _set_input_enabled(self, enabled: bool) -> None:
        self.input_field.setEnabled(enabled)

    def handle_process_finished(self, exit_code: int) -> None:
        if exit_code == 0:
            self.output_panel.display_output("Agent run completed successfully.", config.COLORS.success)
            self.status_bar_manager.update_status("Completed", config.COLORS.success, persist=True)
        else:
            self.output_panel.display_output(f"Agent exited with code {exit_code}", "#FF6B6B")
            self.status_bar_manager.update_status("Error", "#FF6B6B", persist=True)

        self._set_input_enabled(True); self.input_field.setFocus()
        self.current_runner = None

    def handle_process_error(self, error: str) -> None:
        self.output_panel.display_output(error, "#FF6B6B")
        self.status_bar_manager.update_status("Error", "#FF6B6B", persist=True)
        self._set_input_enabled(True); self.input_field.setFocus()

    def clear_output(self) -> None:
        self.output_panel.clear()

    def _select_working_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Working Directory", self.app_state.working_directory)
        if not path:
            return
        try:
            self.agent_manager.set_working_directory(path)
        except (ValueError, FileNotFoundError) as exc:
            self.output_panel.display_output(str(exc), "#FF6B6B")

    def _open_agent_settings(self) -> None:
        dialog = AgentSettingsDialog(self)
        if dialog.exec():
            self.agent_manager.detect_default_agent()

    def _on_progress_update(self, message: str) -> None:
        if not message:
            return
        LOGGER.debug("Progress update: %s", message); self.status_bar_manager.update_status(message, config.COLORS.accent, persist=True)
