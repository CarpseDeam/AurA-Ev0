"""Main application window for Aura."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, Qt, QEasingCurve, QVariantAnimation, QAbstractAnimation
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut, QIcon
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from aura import config
from aura.event_bus import get_event_bus
from aura.events import (
    ExecutionComplete,
    FileOperation,
    StatusUpdate,
    StreamingChunk,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    PhaseTransition,
)
from aura.exceptions import (
    AuraConfigurationError,
    AuraError,
    AuraExecutionError,
    AuraValidationError,
)
from aura.orchestrator import Orchestrator
from aura.state import AppState
from aura.ui import AgentSettingsDialog
from aura.ui.orchestration_handler import OrchestrationHandler, VerbosityLevel
from aura.ui.output_panel import OutputPanel
from aura.ui.status_bar_manager import StatusBarManager
from aura.ui.project_sidebar import ProjectSidebar
from aura.ui.project_panel import ProjectPanel
from aura.ui.project_dialog import ProjectDialog
from aura.models import Project, Conversation
from aura.utils.settings import load_settings, save_settings

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
        self.verbosity_selector: QComboBox | None = None
        self._workspace_blocked_project_id: Optional[int] = None

        # Model status badges
        self.analyst_badge = QLabel(self)
        self.executor_badge = QLabel(self)
        self._event_bus = get_event_bus()
        self._prompting_for_directory = False

        # Initialize project sidebars
        self.project_sidebar = ProjectSidebar(self)
        self.project_panel = ProjectPanel(self)
        self.project_panel.hide()  # Hidden by default

        status_bar = QStatusBar(self); self.status_bar_manager = StatusBarManager(status_bar, self.app_state, self)
        self.setStatusBar(self.status_bar_manager.status_bar)

        self.orchestration_handler = OrchestrationHandler(
            output_panel=self.output_panel,
            status_manager=self.status_bar_manager,
            app_state=self.app_state,
            parent=self,
        )

        self._configure_window(); self._build_layout(); self._configure_sidebar_animation(); self._apply_styles(); self._build_toolbar()
        self._connect_signals(); self._connect_sidebar_signals(); self._connect_handler_signals(); self._subscribe_to_events()
        self._event_received.connect(self.orchestration_handler.handle_feedback_event)

        self._load_and_apply_settings()

        self.status_bar_manager.update_status("Ready", config.COLORS.text, persist=True)

        if self.orchestrator:
            self._connect_orchestrator_signals()
            self.output_panel.display_startup_header()
            self.output_panel.display_output("Aura orchestration ready", config.COLORS.success)

    def _configure_window(self) -> None:
        self.setWindowTitle("Aura"); self.resize(*config.WINDOW_DIMENSIONS)

        # Configure input field with larger font for better readability
        input_font = QFont(config.FONT_FAMILY)
        input_font.setPointSize(config.FONT_SIZE_INPUT)
        self.input_field.setPlaceholderText("Enter a request")
        self.input_field.setClearButtonEnabled(True)
        self.input_field.setFont(input_font)
        self.input_field.setFocus()

        # Set window icon
        icon_path = Path(__file__).resolve().parent.parent / "assets" / "logo.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            LOGGER.warning("Logo icon not found at %s", icon_path)

    def _build_layout(self) -> None:
        """Build the main window layout with a splitter for resizable sidebars."""
        container = QWidget(self)
        main_layout = QHBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #2c313a; }")
        self.splitter.setCollapsible(0, False)

        # Left sidebar (project sidebar)
        self.splitter.addWidget(self.project_sidebar)

        # Center panel (output, input, etc.)
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(16, 16, 16, 16)
        center_layout.setSpacing(12)
        center_layout.addWidget(self.output_panel)

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_field)
        self.clear_button.setFixedWidth(72)
        input_row.addWidget(self.clear_button)
        center_layout.addLayout(input_row)
        center_layout.setStretch(0, 1)
        center_layout.setStretch(1, 0)
        self.splitter.addWidget(center_widget)

        # Configure splitter sizes
        self.splitter.setStretchFactor(0, 0)  # Sidebar doesn't stretch
        self.splitter.setStretchFactor(1, 1)  # Main content stretches

        main_layout.addWidget(self.splitter)

        # Right sidebar (project panel) - outside the splitter for now
        main_layout.addWidget(self.project_panel)

        self.setCentralWidget(container)

    def _configure_sidebar_animation(self) -> None:
        """Prepare the animation used when collapsing/expanding the project sidebar."""
        self._sidebar_animation = QVariantAnimation(self)
        self._sidebar_animation.setDuration(250)
        self._sidebar_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self._sidebar_animation.valueChanged.connect(self._on_sidebar_animation_value_changed)
        self._sidebar_animation.finished.connect(self._on_sidebar_animation_finished)
        self._sidebar_animation_total = 0

    def _animate_sidebar_width(self, target_width: int) -> None:
        """Animate the splitter so the project sidebar reaches the requested width."""
        if not self.splitter:
            return

        target_width = max(int(target_width), 0)
        sizes = self.splitter.sizes()
        current_width = sizes[0] if sizes else self.project_sidebar.width()

        if current_width == target_width:
            self._apply_sidebar_width(target_width)
            return

        total = sum(sizes) if sizes else self.splitter.size().width()
        if total <= 0:
            total = max(self.width(), target_width + 1)

        self._sidebar_animation_total = max(total, target_width + 1)

        if self._sidebar_animation.state() == QAbstractAnimation.State.Running:
            self._sidebar_animation.stop()

        self._sidebar_animation.setStartValue(current_width)
        self._sidebar_animation.setEndValue(target_width)
        self._sidebar_animation.start()

    def _apply_sidebar_width(self, width: int) -> None:
        """Apply a splitter width without animation."""
        if not self.splitter:
            return

        width = max(int(width), 0)
        total = self.splitter.size().width()
        if total <= 0:
            total = sum(self.splitter.sizes())
        if total <= 0:
            total = max(self.width(), width + 1)

        main_width = max(total - width, 1)
        self.splitter.setSizes([width, main_width])

    def _on_sidebar_animation_value_changed(self, value: float) -> None:
        """Adjust splitter sizes on each animation tick."""
        width = max(int(value), 0)
        total = self._sidebar_animation_total or self.splitter.size().width()
        if total <= 0:
            total = max(self.width(), width + 1)
        main_width = max(total - width, 1)
        self.splitter.setSizes([width, main_width])

    def _on_sidebar_animation_finished(self) -> None:
        """Ensure the final animation frame snaps exactly to the requested width."""
        if self._sidebar_animation.state() != QAbstractAnimation.State.Stopped:
            return
        final_width = int(self._sidebar_animation.endValue() or 0)
        self._apply_sidebar_width(final_width)
        self._sidebar_animation_total = 0

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow {{
                background: {config.COLORS.background};
                color: {config.COLORS.text};
            }}
            QTextEdit {{
                background: {config.COLORS.background};
                color: {config.COLORS.text};
                border: none;
                padding: 16px;
                line-height: {config.LINE_HEIGHT};
                letter-spacing: {config.LETTER_SPACING};
                selection-background-color: {config.COLORS.accent};
            }}
            QLineEdit {{
                background: transparent;
                color: {config.COLORS.text};
                border: none;
                border-bottom: 1px solid {config.COLORS.border};
                padding: 12px 16px;
                font-size: {config.FONT_SIZE_INPUT}px;
            }}
            QLineEdit:focus {{
                background: #1a1a1a;
                border-bottom: 2px solid {config.COLORS.accent};
                padding-bottom: 11px;
            }}
            QStatusBar {{
                background: {config.COLORS.background};
                color: {config.COLORS.text};
                border: none;
                border-top: 1px solid {config.COLORS.border};
                padding: 6px;
                font-size: {config.FONT_SIZE_STATUS}px;
            }}
            QToolBar {{
                background: {config.COLORS.background};
                border: none;
                spacing: 8px;
                padding: 4px;
            }}
        """)
        self.clear_button.setStyleSheet(
            f"QPushButton {{background: {config.COLORS.background}; "
            f"color: {config.COLORS.text}; border: 1px solid {config.COLORS.border}; "
            f"border-radius: 6px; padding: 8px 12px; "
            f"font-size: {config.FONT_SIZE_STATUS}px; font-weight: 500;}}"
            f"QPushButton:hover {{background: #2a2a2a; border-color: {config.COLORS.accent};}}"
            f"QPushButton:pressed {{background: #1a1a1a;}}"
        )

    def _build_toolbar(self) -> None:
        self.toolbar.setMovable(False)
        dir_action = QAction("Set Working Directory", self); dir_action.triggered.connect(self._select_working_directory); self.toolbar.addAction(dir_action)
        agent_action = QAction("Agent Settings...", self); agent_action.triggered.connect(self._open_agent_settings); self.toolbar.addAction(agent_action)

        # Verbosity controls
        verbosity_label = QLabel("Verbosity:", self)
        verbosity_label.setStyleSheet(
            f"color: {config.COLORS.secondary}; font-size: 11px; font-weight: 500; padding-left: 6px;"
        )
        self.toolbar.addWidget(verbosity_label)
        self.verbosity_selector = QComboBox(self)
        self._configure_verbosity_selector()
        self.toolbar.addWidget(self.verbosity_selector)

        # Add spacer to push badges to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        # Add model status badges
        self._update_analyst_badge(self.app_state.analyst_model)
        self._update_executor_badge(self.app_state.executor_model)
        self._style_model_badges()
        self.toolbar.addWidget(self.analyst_badge)
        self.toolbar.addWidget(self.executor_badge)

    def _configure_verbosity_selector(self) -> None:
        """Initialize the verbosity drop-down."""
        if not self.verbosity_selector:
            return
        self.verbosity_selector.clear()
        for level in VerbosityLevel:
            self.verbosity_selector.addItem(level.name.title(), level.value)
        self.verbosity_selector.setMinimumWidth(130)
        self.verbosity_selector.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.verbosity_selector.setStyleSheet(
            f"""
            QComboBox {{
                background: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.border};
                border-radius: 6px;
                padding: 4px 10px;
                font-size: 11px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QListView {{
                background: {config.COLORS.background};
                color: {config.COLORS.text};
                selection-background-color: {config.COLORS.accent};
            }}
            """
        )
        self.verbosity_selector.setToolTip("Control how much streaming output is shown.")
        self.verbosity_selector.currentIndexChanged.connect(self._on_verbosity_changed)
        self._select_verbosity_in_ui(self.orchestration_handler.verbosity)

    def _select_verbosity_in_ui(self, level: VerbosityLevel) -> None:
        """Update the combo box selection without retriggering persistence."""
        if not self.verbosity_selector:
            return
        target_index = self.verbosity_selector.findData(level.value)
        if target_index == -1:
            return
        block_state = self.verbosity_selector.blockSignals(True)
        self.verbosity_selector.setCurrentIndex(target_index)
        self.verbosity_selector.blockSignals(block_state)

    def _apply_verbosity_level(self, level: VerbosityLevel, *, persist: bool) -> None:
        """Apply verbosity to handler, update UI, and optionally persist."""
        self.orchestration_handler.set_verbosity(level)
        self._select_verbosity_in_ui(level)
        if persist:
            settings = load_settings()
            settings["verbosity"] = level.value
            save_settings(settings)

    def _on_verbosity_changed(self, index: int) -> None:
        """Handle live changes to verbosity from the toolbar."""
        if not self.verbosity_selector or index < 0:
            return
        data = self.verbosity_selector.itemData(index)
        level = VerbosityLevel.from_value(str(data) if data is not None else None)
        self._apply_verbosity_level(level, persist=True)

    def _connect_signals(self) -> None:
        self.input_field.returnPressed.connect(self._handle_submit); self.clear_button.clicked.connect(self.clear_output)
        self.splitter.splitterMoved.connect(self._on_splitter_moved)

        # Connect model change signals to update badges
        self.app_state.analyst_model_changed.connect(self._update_analyst_badge)
        self.app_state.executor_model_changed.connect(self._update_executor_badge)

        # Keyboard shortcut for toggling sidebar
        shortcut = QShortcut(QKeySequence("Ctrl+B"), self)
        shortcut.activated.connect(self.project_sidebar._toggle_collapse)

    def _update_analyst_badge(self, model_name: str) -> None:
        """Update the analyst model badge display."""
        self.analyst_badge.setText(f"Analyst: {model_name}")

    def _update_executor_badge(self, model_name: str) -> None:
        """Update the executor model badge display."""
        self.executor_badge.setText(f"Executor: {model_name}")

    def _style_model_badges(self) -> None:
        """Apply styling to the model status badges."""
        badge_style = f"""
            QLabel {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.border};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 500;
                margin-left: 6px;
            }}
        """
        self.analyst_badge.setStyleSheet(badge_style)
        self.executor_badge.setStyleSheet(badge_style)

    def _connect_handler_signals(self) -> None:
        self.orchestration_handler.request_input_enabled.connect(self._set_input_enabled); self.orchestration_handler.request_input_focus.connect(self.input_field.setFocus)

    def _subscribe_to_events(self) -> None:
        """Subscribe UI to backend feedback events."""
        event_types = (
            StreamingChunk,
            ToolCallStarted,
            ToolCallCompleted,
            ToolCallFailed,
            FileOperation,
            PhaseTransition,
            StatusUpdate,
            ExecutionComplete,
        )
        for event_type in event_types:
            self._event_bus.subscribe(event_type, self._emit_event_signal)

    def _emit_event_signal(self, event: object) -> None:
        self._event_received.emit(event)

    def _load_and_apply_settings(self) -> None:
        """Load sidebar state from settings and apply it."""
        settings = load_settings()
        saved_verbosity = VerbosityLevel.from_value(settings.get("verbosity"))
        self._apply_verbosity_level(saved_verbosity, persist=False)
        sidebar_width = int(settings.get("sidebar_width", self.project_sidebar.sizeHint().width()))
        self.project_sidebar.expanded_width = sidebar_width
        sidebar_width = self.project_sidebar.expanded_width

        is_collapsed = bool(settings.get("sidebar_collapsed", False))
        self.project_sidebar.set_collapsed(is_collapsed, emit_signal=False)

        initial_width = self.project_sidebar.collapsed_width if is_collapsed else sidebar_width
        self._apply_sidebar_width(initial_width)

    def _on_sidebar_state_changed(self, is_collapsed: bool) -> None:
        """Save the sidebar's collapsed state and adjust the splitter."""
        settings = load_settings()
        settings["sidebar_collapsed"] = is_collapsed
        expanded_width = int(settings.get("sidebar_width", self.project_sidebar.expanded_width))
        self.project_sidebar.expanded_width = expanded_width
        expanded_width = self.project_sidebar.expanded_width
        save_settings(settings)
        LOGGER.info(f"Sidebar collapsed state saved: {is_collapsed}")

        target_width = self.project_sidebar.collapsed_width if is_collapsed else expanded_width
        self._animate_sidebar_width(target_width)

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        """Save the sidebar's width when the splitter is moved."""
        if self.project_sidebar.is_collapsed():
            return
        if self._sidebar_animation.state() == QAbstractAnimation.State.Running:
            return

        sizes = self.splitter.sizes()
        if not sizes:
            return

        sidebar_width = sizes[0]
        if sidebar_width <= self.project_sidebar.collapsed_width + 4:
            return

        settings = load_settings()
        settings["sidebar_width"] = sidebar_width
        save_settings(settings)
        self.project_sidebar.expanded_width = sidebar_width
        LOGGER.debug(f"Sidebar width saved: {sidebar_width}")

    def _connect_orchestrator_signals(self) -> None:
        from PySide6.QtCore import Qt

        assert self.orchestrator is not None
        # Use UniqueConnection to prevent duplicate connections if called multiple times
        self.orchestrator.planning_started.connect(
            self.orchestration_handler.handle_planning_started,
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
        self.orchestrator.progress_update.connect(
            self._on_progress_update,
            Qt.ConnectionType.UniqueConnection
        )

    def _handle_submit(self) -> None:
        prompt = self.input_field.text().strip()
        if not prompt:
            return
        if self._workspace_blocked_project_id:
            self.output_panel.display_error(
                "Select a working directory for the current project before running an agent."
            )
            return

        # IMMEDIATE FEEDBACK - happens instantly before any processing
        self.input_field.clear()
        self.output_panel.display_output(f"> {prompt}", config.COLORS.prompt)
        has_orchestrator = self.orchestrator is not None
        status_message = "⋯ Analyst/Executor pipeline running..." if has_orchestrator else "⋯ Orchestrator unavailable"
        self.status_bar_manager.update_status(status_message, config.COLORS.thinking, persist=True)
        self._set_input_enabled(False)

        try:
            if not has_orchestrator:
                self._display_user_error(
                    "Cloud orchestration is unavailable. Set ANTHROPIC_API_KEY and restart Aura."
                )
                self._recover_input_after_error()
                return

            LOGGER.info("EXECUTION_REQUESTED: goal=%s", prompt[:80])
            self.execution_requested.emit(prompt)
        except AuraConfigurationError as exc:
            LOGGER.warning("Configuration issue while submitting prompt: %s", exc)
            self._handle_configuration_issue(exc)
            self._recover_input_after_error()
        except (AuraValidationError, AuraExecutionError) as exc:
            LOGGER.warning("Validation/execution error while submitting prompt: %s", exc)
            self._display_user_error(str(exc))
            self._recover_input_after_error()
        except AuraError as exc:
            LOGGER.warning("Aura error during submission: %s", exc)
            self._display_user_error(str(exc))
            self._recover_input_after_error()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Unexpected error handling submit")
            self._display_user_error("Something went wrong while starting your request. Please try again.")
            self._recover_input_after_error()

    def _set_input_enabled(self, enabled: bool) -> None:
        self.input_field.setEnabled(enabled)

    def clear_output(self) -> None:
        self.output_panel.clear()

    def _display_user_error(self, message: str) -> None:
        """Show a user-facing error and update status bar."""
        if not message:
            message = "Something went wrong. Check the logs for more details."
        self.output_panel.display_error(message)
        self.status_bar_manager.update_status("Error", config.COLORS.error, persist=True)

    def _recover_input_after_error(self) -> None:
        """Re-enable input controls after a failure."""
        self._set_input_enabled(True)
        self.input_field.setFocus()

    def _handle_configuration_issue(self, error: AuraConfigurationError) -> None:
        """Display configuration issues and guide the user to resolve them."""
        self._display_user_error(str(error))
        issue = (error.context or {}).get("issue") if isinstance(error.context, dict) else None
        if issue in {"working_directory_missing", "working_directory_invalid"}:
            self._prompt_for_working_directory()

    def _prompt_for_working_directory(self) -> None:
        """Prompt the user to choose a new working directory."""
        if self._prompting_for_directory:
            return
        self._prompting_for_directory = True
        try:
            self.output_panel.display_output(
                "Please select a valid working directory to continue.",
                config.COLORS.accent,
            )
            self._select_working_directory()
        finally:
            self._prompting_for_directory = False

    def _select_working_directory(self) -> None:
        """Allow the user to override the working directory manually."""
        seed = self.app_state.working_directory or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Working Directory", seed)
        if not path:
            return
        project_name = None
        project = None
        current_project_id = self.app_state.current_project_id
        if current_project_id:
            project = Project.get_by_id(current_project_id)
            if project:
                project_name = project.name
        if self._apply_project_working_directory(path, project_name=project_name):
            if current_project_id and project:
                project.update(working_directory=str(Path(path).resolve()))
                self.project_sidebar.refresh_projects()
                self.project_panel.set_project(project)
            self._workspace_blocked_project_id = None
            self._set_input_enabled(True)

    def _apply_project_working_directory(self, raw_path: str, *, project_name: Optional[str] = None) -> bool:
        """Resolve and propagate a project-provided working directory."""
        if not raw_path:
            return False
        try:
            resolved_path = Path(raw_path).expanduser().resolve()
            resolved = str(resolved_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Invalid project working directory '%s': %s", raw_path, exc)
            self._display_user_error(f"Project working directory is invalid: {raw_path}")
            return False

        current = self.app_state.working_directory
        if current:
            try:
                if Path(current).resolve() == resolved_path:
                    LOGGER.debug("Workspace already set to %s; skipping update", resolved)
                    return True
            except Exception:  # noqa: BLE001
                LOGGER.debug("Unable to compare workspace paths; continuing", exc_info=True)

        LOGGER.info(
            "Setting ToolManager workspace to: %s | project=%s",
            resolved,
            project_name or "unknown",
        )
        try:
            self.app_state.set_working_directory(resolved)
        except (ValueError, FileNotFoundError) as exc:
            LOGGER.warning("Failed to set working directory %s: %s", resolved, exc)
            self._display_user_error("Please select a valid working directory that exists on disk.")
            return False
        workspace = self.app_state.working_directory
        if not workspace:
            LOGGER.error("Workspace application returned empty path for %s", resolved)
            self._display_user_error("Workspace not applied. Please retry selecting the directory.")
            return False

        if self.orchestrator:
            try:
                self.orchestrator.update_working_directory(workspace)
            except AuraConfigurationError as exc:
                LOGGER.warning("Failed to sync orchestrator workspace %s: %s", resolved, exc)
                self._display_user_error(str(exc))
                return False

        try:
            if Path(workspace).resolve() != resolved_path:
                LOGGER.error(
                    "Workspace mismatch detected | expected=%s | current=%s",
                    resolved,
                    workspace,
                )
                self._display_user_error(
                    "Working directory mismatch detected. Please re-select the project directory."
                )
                return False
        except Exception:  # noqa: BLE001
            LOGGER.debug("Unable to verify workspace equality; continuing", exc_info=True)

        if not self._verify_tool_manager_workspace(resolved):
            return False

        if project_name:
            self.status_bar_manager.update_status(f"Project: {project_name}", config.COLORS.accent, persist=True)
        else:
            self.status_bar_manager.update_status("Workspace updated", config.COLORS.accent, persist=True)
        LOGGER.info("Working directory set to %s", resolved)
        return True

    def _open_agent_settings(self) -> None:
        dialog = AgentSettingsDialog(self.app_state, self)
        dialog.exec()

    def _on_progress_update(self, message: str) -> None:
        """Update status bar with immediate visual feedback for progress."""
        if not message:
            return
        LOGGER.debug("Progress update: %s", message)
        # Update status bar with immediate visual feedback
        color = config.COLORS.thinking if "⋯" in message else config.COLORS.accent
        self.status_bar_manager.update_status(message, color, persist=True)


    def _connect_sidebar_signals(self) -> None:
        """Connect signals from project sidebars."""
        # Project sidebar signals
        self.project_sidebar.project_selected.connect(self._on_project_selected)
        self.project_sidebar.conversation_selected.connect(self._on_conversation_selected)
        self.project_sidebar.new_project_clicked.connect(self._on_new_project)
        self.project_sidebar.new_conversation_clicked.connect(self._on_new_conversation)
        self.project_sidebar.state_changed.connect(self._on_sidebar_state_changed)
        self.project_sidebar.conversation_deleted.connect(self._on_conversation_deleted)

        # Project panel signals
        self.project_panel.edit_project_clicked.connect(self._on_edit_project)
        self.project_panel.close_clicked.connect(self._on_close_project_panel)

    def _on_project_selected(self, project_id: int) -> None:
        """Handle project selection from sidebar."""
        try:
            project = Project.get_by_id(project_id)
            if project:
                # Clear any existing conversation context as we switch projects
                self.app_state.set_current_conversation(None)
                if self.orchestrator:
                    self.orchestrator.reset_history()
                LOGGER.info("Loading project: %s (ID: %s)", project.name, project_id)
                LOGGER.info("Project working directory: %s", project.working_directory or "<unset>")

                self.output_panel.clear()

                # Update app state
                self.app_state.set_current_project(project_id)

                workspace_ready = self._ensure_project_workspace(project)
                if workspace_ready:
                    self._workspace_blocked_project_id = None
                    current_workspace = self.app_state.working_directory
                    self._set_input_enabled(True)
                    self.output_panel.display_output(
                        f"Loaded project: {project.name} | Working directory: {current_workspace}",
                        config.COLORS.success,
                    )
                else:
                    self._block_workspace_for_project(project)
                    self.output_panel.display_output(f"Loaded project: {project.name}", config.COLORS.success)

                # Show project panel
                self.project_panel.set_project(project)
                self.project_panel.show()

                # Refresh sidebar to highlight selection
                self.project_sidebar.set_current_project(project_id)

                LOGGER.info(f"Selected project: {project.name} (ID: {project_id})")

        except Exception as e:
            LOGGER.error(f"Failed to select project: {e}")

    def _ensure_project_workspace(self, project: Project) -> bool:
        """Ensure the selected project has an active working directory."""
        raw_path = (project.working_directory or "").strip()
        if raw_path:
            return self._apply_project_working_directory(raw_path, project_name=project.name)
        return self._prompt_for_project_working_directory(project)

    def _prompt_for_project_working_directory(self, project: Project) -> bool:
        """Request the user to provide a working directory when missing."""
        message = (
            f"Project '{project.name}' has no working directory. "
            "Select one now to enable file tools."
        )
        choice = QMessageBox.question(
            self,
            "Set Project Working Directory",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            seed = project.working_directory or self.app_state.working_directory or str(Path.home())
            selected = QFileDialog.getExistingDirectory(self, "Select Project Directory", seed)
            if selected:
                resolved = str(Path(selected).expanduser().resolve())
                LOGGER.info(
                    "User selected working directory %s for project %s",
                    resolved,
                    project.id,
                )
                project.update(working_directory=resolved)
                self.project_sidebar.refresh_projects()
                self.project_panel.set_project(project)
                return self._apply_project_working_directory(resolved, project_name=project.name)
        LOGGER.warning(
            "Project %s lacks a working directory; agent operations disabled until one is set.",
            project.id,
        )
        self._display_user_error(
            f"Project '{project.name}' has no working directory. Use 'Set Working Directory' to continue."
        )
        return False

    def _block_workspace_for_project(self, project: Project) -> None:
        """Disable agent operations until the workspace is configured."""
        self._workspace_blocked_project_id = project.id
        self._set_input_enabled(False)
        self.status_bar_manager.update_status(
            f"Workspace required for {project.name}",
            config.COLORS.error,
            persist=True,
        )

    def _verify_tool_manager_workspace(self, expected: str) -> bool:
        """Compare the orchestrator/tool manager workspace to the requested path."""
        if not expected:
            return False
        expected_path = None
        try:
            expected_path = Path(expected).resolve()
        except Exception:  # noqa: BLE001
            LOGGER.debug("Unable to resolve expected workspace %s", expected, exc_info=True)
        if not self.orchestrator:
            LOGGER.info("ToolManager now using: %s (no orchestrator available for verification)", expected)
            return True

        tm_path = ""
        tm = getattr(self.orchestrator, "_tool_manager", None)
        if tm is not None:
            tm_path = str(getattr(tm, "workspace_dir", "") or "")
        orch_path = ""
        if hasattr(self.orchestrator, "_working_dir"):
            try:
                orch_path = str(self.orchestrator._working_dir)
            except Exception:  # noqa: BLE001
                LOGGER.debug("Unable to read orchestrator working dir", exc_info=True)

        LOGGER.info(
            "ToolManager now using: %s | orchestrator _working_dir=%s",
            tm_path or "<unknown>",
            orch_path or "<unknown>",
        )

        mismatch = False
        if expected_path:
            if tm_path:
                try:
                    if Path(tm_path).resolve() != expected_path:
                        mismatch = True
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Unable to resolve ToolManager path %s", tm_path, exc_info=True)
            if orch_path:
                try:
                    if Path(orch_path).resolve() != expected_path:
                        mismatch = True
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Unable to resolve orchestrator path %s", orch_path, exc_info=True)

        if mismatch:
            LOGGER.error(
                "ToolManager workspace mismatch detected | expected=%s | tool_manager=%s | orchestrator=%s",
                expected,
                tm_path or "<unknown>",
                orch_path or "<unknown>",
            )
            self._display_user_error(
                "ToolManager is still using the previous workspace. Please re-select the project directory."
            )
            return False
        return True

    def _on_conversation_selected(self, conversation_id: int) -> None:
        """Handle conversation selection from sidebar."""
        try:
            conv = Conversation.get_by_id(conversation_id)
            if conv:
                # Update app state
                self.app_state.set_current_conversation(conversation_id)

                # Load conversation history into orchestrator
                if self.orchestrator:
                    self.orchestrator.load_conversation_history(conversation_id)

                # Clear output and display conversation
                self.output_panel.clear()
                self.output_panel.display_output(f"Loaded conversation: {conv.title or '(Untitled)'}", config.COLORS.accent)

                # Display conversation history
                messages = conv.get_messages()
                for msg in messages:
                    if msg.role == 'user':
                        self.output_panel.display_output(f"> {msg.content}", config.COLORS.prompt)
                    else:
                        self.output_panel.display_output(msg.content, config.COLORS.agent_output)

                # Refresh sidebar to highlight selection
                self.project_sidebar.set_current_conversation(conversation_id)

                LOGGER.info(f"Selected conversation ID: {conversation_id}")

        except Exception as e:
            LOGGER.error(f"Failed to select conversation: {e}")

    def _on_conversation_deleted(self, conversation_id: int) -> None:
        """Handle cleanup after a conversation is deleted."""
        LOGGER.info(f"Conversation {conversation_id} was deleted.")
        if self.app_state.current_conversation_id == conversation_id:
            self.app_state.set_current_conversation(None)
            self.output_panel.clear()
            if self.orchestrator:
                self.orchestrator.reset_history()
            LOGGER.info("Active conversation was deleted. Cleared panel and reset state.")

    def _on_new_project(self) -> None:
        """Handle new project button click."""
        dialog = ProjectDialog(parent=self)
        result = dialog.exec()

        if result == QDialog.Accepted:
            project = dialog.get_project()
            if project:
                # Refresh sidebar to show new project
                self.project_sidebar.refresh_projects()

                # Select the new project
                if project.id:
                    self._on_project_selected(project.id)

                LOGGER.info(f"Created new project: {project.name}")

    def _on_new_conversation(self) -> None:
        """Handle new conversation button click."""
        # Clear current conversation
        self.app_state.set_current_conversation(None)

        # Reset orchestrator history
        if self.orchestrator:
            self.orchestrator.reset_history()

        # Clear output
        self.output_panel.clear()
        self.output_panel.display_output("New conversation started", config.COLORS.success)

        # Refresh sidebar
        self.project_sidebar.set_current_conversation(None)

        LOGGER.info("Started new conversation")

    def _on_edit_project(self) -> None:
        """Handle edit project button click."""
        project = self.project_panel.get_current_project()
        if not project:
            return

        dialog = ProjectDialog(project=project, parent=self)
        result = dialog.exec()

        if result == dialog.Accepted:
            # Refresh sidebars
            self.project_sidebar.refresh_projects()

            # Reload project in panel
            updated_project = Project.get_by_id(project.id)
            if updated_project:
                self.project_panel.set_project(updated_project)

            LOGGER.info(f"Updated project: {project.name}")

        elif result == -1:  # Deletion
            # Hide panel
            self.project_panel.hide()

            # Clear state and refresh sidebars
            self.app_state.set_current_project(None)
            self.project_sidebar.refresh_all()

            LOGGER.info(f"Deleted project: {project.name}")

    def _on_close_project_panel(self) -> None:
        """Handle project panel close button click."""
        self.project_panel.hide()
        self.app_state.set_current_project(None)
        self.project_sidebar.set_current_project(None)
        LOGGER.info("Closed project panel and cleared current project.")

    def refresh_sidebars(self) -> None:
        """Refresh both sidebars (call after database changes)."""
        self.project_sidebar.refresh_all()
