"""Dialog for configuring CLI agents."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aura import config
from aura.utils.agent_finder import AgentInfo, find_cli_agents, validate_agent


class AgentSettingsDialog(QDialog):
    """Dialog to view and configure available CLI agents."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the agent settings dialog."""
        super().__init__(parent)
        self.agents: list[AgentInfo] = []
        self.table = QTableWidget(self)
        self.refresh_button = QPushButton("Refresh", self)
        self.test_button = QPushButton("Test", self)
        self.set_default_button = QPushButton("Set as Default", self)
        self.custom_path_button = QPushButton("Add Custom Path...", self)
        self.close_button = QPushButton("Close", self)
        self._configure_dialog()
        self._build_layout()
        self._apply_styles()
        self._connect_signals()
        self.refresh_agents()

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
        button_row.addStretch()
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

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
                border: 1px solid #333333;
                gridline-color: #222222;
            }}
            QHeaderView::section {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid #333333;
                padding: 6px;
                font-weight: normal;
            }}
            QPushButton {{
                background-color: {config.COLORS.background};
                color: {config.COLORS.text};
                border: 1px solid #333333;
                padding: 6px 12px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                border-color: {config.COLORS.accent};
            }}
            """
        )

    def _connect_signals(self) -> None:
        """Connect button signals."""
        self.refresh_button.clicked.connect(self.refresh_agents)
        self.test_button.clicked.connect(self.test_selected_agent)
        self.set_default_button.clicked.connect(self._set_as_default)
        self.custom_path_button.clicked.connect(self._add_custom_path)
        self.close_button.clicked.connect(self.accept)

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
        QMessageBox.information(
            self,
            "Default Set",
            f"{agent.display_name} set as default agent.\n(Configuration persistence not yet implemented)",
        )

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


__all__ = ["AgentSettingsDialog"]
