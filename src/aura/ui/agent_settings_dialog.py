"""Dialog for configuring CLI agents."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import os

from aura import config
from aura.state import AppState
from aura.utils.agent_finder import AgentInfo, find_cli_agents, validate_agent
from aura.utils.model_discovery import discover_claude_models, discover_gemini_models
from aura.utils.settings import save_settings


class AgentSettingsDialog(QDialog):
    """Dialog to view and configure available CLI agents."""

    def __init__(self, app_state: AppState, parent: QWidget | None = None) -> None:
        """Initialize the agent settings dialog."""
        super().__init__(parent)
        self.app_state = app_state
        self.agents: list[AgentInfo] = []
        self.table = QTableWidget(self)
        self.refresh_button = QPushButton("Refresh", self)
        self.test_button = QPushButton("Test", self)
        self.set_default_button = QPushButton("Set as Default", self)
        self.custom_path_button = QPushButton("Add Custom Path...", self)
        self.save_settings_button = QPushButton("Save Settings", self)
        self.save_settings_button.setObjectName("save_settings_button")

        # Model selection widgets
        self.gemini_model_combo = QComboBox(self)
        self.gemini_refresh_button = QPushButton("Refresh Models", self)
        self.claude_model_combo = QComboBox(self)
        self.claude_refresh_button = QPushButton("Refresh Models", self)

        self.close_button = QPushButton("Close", self)
        self._configure_dialog()
        self._build_layout()
        self._apply_styles()
        self._connect_signals()
        self.refresh_agents()
        self._refresh_gemini_models()
        self._refresh_claude_models()

    def _configure_dialog(self) -> None:
        """Configure dialog properties."""
        self.setWindowTitle("Agent Settings")
        self.resize(700, 400)
        self.setModal(True)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Path", "Version", "Status"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

    def _build_layout(self) -> None:
        """Create dialog layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.addWidget(self.table)
        button_row = QHBoxLayout()
        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.test_button)
        button_row.addWidget(self.set_default_button)
        button_row.addWidget(self.custom_path_button)
        button_row.addWidget(self.save_settings_button)
        button_row.addStretch()
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        # Model selection section
        model_layout = QFormLayout()
        model_layout.setContentsMargins(0, 20, 0, 0)
        model_layout.addRow(QLabel("API MODELS"))

        gemini_layout = QHBoxLayout()
        gemini_layout.addWidget(self.gemini_model_combo)
        gemini_layout.addWidget(self.gemini_refresh_button)
        model_layout.addRow("Gemini (Analyst) Model:", gemini_layout)

        claude_layout = QHBoxLayout()
        claude_layout.addWidget(self.claude_model_combo)
        claude_layout.addWidget(self.claude_refresh_button)
        model_layout.addRow("Claude (Executor) Model:", claude_layout)

        layout.addLayout(model_layout)

    def _apply_styles(self) -> None:
        """Apply dark theme styling."""
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
            }}
            QTableWidget {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.border};
                gridline-color: {config.COLORS.border};
            }}
            QHeaderView::section {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.border};
                padding: 6px;
                font-weight: normal;
            }}
            QPushButton {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.border};
                border-radius: 6px;
                padding: 6px 12px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background: #2a2a2a;
                border-color: {config.COLORS.accent};
            }}
            QPushButton:pressed {{
                background: #1a1a1a;
            }}
            QPushButton#save_settings_button {{
                background-color: {config.COLORS.accent};
                color: #ffffff;
                border-color: {config.COLORS.accent};
            }}
            QComboBox {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid {config.COLORS.border};
                border-radius: 6px;
                padding: 6px;
            }}
            QComboBox:hover {{
                border-color: {config.COLORS.accent};
            }}
            QLabel {{
                font-weight: bold;
                padding-top: 10px;
            }}
            """
        )

    def _connect_signals(self) -> None:
        """Connect button signals."""
        self.refresh_button.clicked.connect(self.refresh_agents)
        self.test_button.clicked.connect(self.test_selected_agent)
        self.set_default_button.clicked.connect(self._set_as_default)
        self.custom_path_button.clicked.connect(self._add_custom_path)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.close_button.clicked.connect(self.accept)

        # Model selection signals
        self.gemini_refresh_button.clicked.connect(self._refresh_gemini_models)
        self.claude_refresh_button.clicked.connect(self._refresh_claude_models)
        self.gemini_model_combo.currentTextChanged.connect(self._on_gemini_model_selected)
        self.claude_model_combo.currentTextChanged.connect(self._on_claude_model_selected)

    def refresh_agents(self) -> None:
        """Scan system for available agents."""
        self.agents = find_cli_agents()
        self._populate_table()

    def _populate_table(self) -> None:
        """Update table with agent information."""
        self.table.setRowCount(len(self.agents))
        for row, agent in enumerate(self.agents):
            name_item = QTableWidgetItem(agent.display_name)
            path_item = QTableWidgetItem(agent.executable_path or "(not found)")
            version_item = QTableWidgetItem(agent.version)
            status_item = QTableWidgetItem("✓" if agent.is_available else "✗")
            if agent.is_available:
                status_item.setForeground(Qt.GlobalColor.green)
            else:
                status_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, path_item)
            self.table.setItem(row, 2, version_item)
            self.table.setItem(row, 3, status_item)

    def test_selected_agent(self) -> None:
        """Validate the selected agent works."""
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.agents):
            QMessageBox.warning(self, "No Selection", "Please select an agent to test.")
            return
        agent = self.agents[current_row]
        if not agent.executable_path:
            QMessageBox.critical(self, "Test Failed", f"{agent.display_name} executable not found.")
            return
        is_valid, version = validate_agent(agent.executable_path, agent.name)
        if is_valid:
            QMessageBox.information(
                self, "Test Successful", f"{agent.display_name} is working.\nVersion: {version}"
            )
        else:
            QMessageBox.critical(
                self, "Test Failed", f"{agent.display_name} failed validation."
            )

    def _set_as_default(self) -> None:
        """Set selected agent as default."""
        current_row = self.table.currentRow()
        if current_row < 0 or current_row >= len(self.agents):
            QMessageBox.warning(self, "No Selection", "Please select an agent to set as default.")
            return
        agent = self.agents[current_row]
        if not agent.is_available:
            QMessageBox.warning(
                self, "Agent Unavailable", f"{agent.display_name} is not available."
            )
            return
        
        self.app_state.set_selected_agent(agent.name)
        if agent.executable_path:
            self.app_state.set_agent_executable(agent.name, agent.executable_path)

        QMessageBox.information(
            self,
            "Default Set",
            f"{agent.display_name} set as default agent. Click 'Save Settings' to persist this change.",
        )

    def _save_settings(self) -> None:
        """Save current settings to disk."""
        settings = {
            "gemini_model": self.app_state.gemini_model,
            "claude_model": self.app_state.claude_model,
            "selected_agent": self.app_state.selected_agent,
            "agent_executable": self.app_state.agent_executable.get(self.app_state.selected_agent)
        }
        save_settings(settings)
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully.")

    def _add_custom_path(self) -> None:
        """Allow user to manually specify agent path."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Agent Executable", "", "Executables (*.exe *.cmd);;All Files (*)"
        )
        if not file_path:
            return
        agent_name = self._prompt_agent_name()
        if not agent_name:
            return
        is_valid, version = validate_agent(file_path, agent_name)
        if is_valid:
            QMessageBox.information(
                self, "Agent Added", f"Custom agent validated.\nVersion: {version}"
            )
        else:
            QMessageBox.warning(self, "Validation Failed", "Unable to validate custom agent.")

    def _prompt_agent_name(self) -> str:
        """Prompt user to select agent type for custom path."""
        from PySide6.QtWidgets import QInputDialog

        items = list(config.AGENT_DISPLAY_NAMES.keys())
        item, ok = QInputDialog.getItem(
            self, "Select Agent Type", "Choose the agent type:", items, 0, False
        )
        return item if ok else ""

    def _refresh_gemini_models(self) -> None:
        """Fetch and display available Gemini models."""
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            self._update_model_combo(self.gemini_model_combo, [], "Set GEMINI_API_KEY to discover models")
            return

        models = discover_gemini_models(api_key)
        self._update_model_combo(self.gemini_model_combo, models, "Unable to fetch models (check network/key)")

    def _refresh_claude_models(self) -> None:
        """Fetch and display available Claude models."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self._update_model_combo(self.claude_model_combo, [], "Set ANTHROPIC_API_KEY to discover models")
            return

        models = discover_claude_models(api_key)
        self._update_model_combo(self.claude_model_combo, models, "Unable to fetch models (check network/key)")

    def _update_model_combo(self, combo: QComboBox, models: list, error_message: str) -> None:
        """Helper to populate a model combo box."""
        current_selection = combo.currentData()
        combo.clear()

        if not models:
            combo.addItem(error_message)
            combo.setEnabled(False)
            return

        combo.setEnabled(True)
        for model in models:
            combo.addItem(model.display_name, model.model_id)

        if current_selection:
            index = combo.findData(current_selection)
            if index != -1:
                combo.setCurrentIndex(index)
        else:
            if combo.count() > 0:
                combo.setCurrentIndex(0)

    def _on_gemini_model_selected(self, text: str) -> None:
        """Handle selection of a Gemini model."""
        model_id = self.gemini_model_combo.currentData()
        if model_id:
            self.app_state.set_gemini_model(model_id)

    def _on_claude_model_selected(self, text: str) -> None:
        """Handle selection of a Claude model."""
        model_id = self.claude_model_combo.currentData()
        if model_id:
            self.app_state.set_claude_model(model_id)


__all__ = ["AgentSettingsDialog"]
