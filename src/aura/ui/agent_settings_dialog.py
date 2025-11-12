"""Dialog for configuring Aura's single-agent runtime."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aura.state import AppState
from aura.utils.settings import AGENT_MODEL_OPTIONS, load_settings, save_settings

MODEL_LABELS = {
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5 (Sep 29, 2025)",
    "claude-opus-4-20250514": "Claude Opus 4 (May 14, 2025)",
}


class AgentSettingsDialog(QDialog):
    """Lightweight dialog for configuring the single agent runtime."""

    def __init__(self, app_state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state

        self.model_combo = QComboBox(self)
        self.api_key_input = QLineEdit(self)
        self.max_tokens_spin = QSpinBox(self)
        self.tool_call_spin = QSpinBox(self)
        self.temperature_spin = QDoubleSpinBox(self)
        self.cost_tracking_checkbox = QCheckBox("Enable cost tracking", self)

        self.save_settings_button = QPushButton("Save", self)
        self.save_settings_button.setObjectName("save_settings_button")
        self.close_button = QPushButton("Close", self)

        self._configure_dialog()
        self._build_layout()
        self._apply_styles()
        self._connect_signals()
        self._populate_initial_values()
        self._update_save_button_state()

    def _configure_dialog(self) -> None:
        self.setWindowTitle("Agent Settings")
        self.resize(520, 360)
        self.setModal(True)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QLabel(
            "Configure the Anthropic model, credentials, and behavior limits for Aura's "
            "single agent."
        )
        header.setWordWrap(True)
        header.setObjectName("header_label")
        layout.addWidget(header)

        layout.addWidget(self._build_model_section())
        layout.addWidget(self._build_generation_section())
        layout.addWidget(self._build_behavior_section())

        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.setSpacing(12)
        button_row.addWidget(self.save_settings_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

    def _build_model_section(self) -> QGroupBox:
        self.model_combo.setEditable(False)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.addItems([])
        for model_id in AGENT_MODEL_OPTIONS:
            label = MODEL_LABELS.get(model_id, model_id)
            self.model_combo.addItem(label, model_id)

        self.api_key_input.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
        self.api_key_input.setPlaceholderText("Enter Anthropic API key")

        group = QGroupBox("1. Model Selection", self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)
        form.addRow("Model", self.model_combo)
        form.addRow("Anthropic API Key", self.api_key_input)
        group.setLayout(form)
        return group

    def _build_generation_section(self) -> QGroupBox:
        self.max_tokens_spin.setRange(1_000, 400_000)
        self.max_tokens_spin.setSingleStep(5_000)
        self.max_tokens_spin.setSuffix(" tokens")

        self.tool_call_spin.setRange(1, 100)
        self.tool_call_spin.setSingleStep(1)

        group = QGroupBox("2. Generation Limits", self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)
        form.addRow("Max token budget", self.max_tokens_spin)
        form.addRow("Tool call limit", self.tool_call_spin)
        group.setLayout(form)
        return group

    def _build_behavior_section(self) -> QGroupBox:
        self.temperature_spin.setRange(0.0, 1.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setDecimals(2)

        self.cost_tracking_checkbox.setChecked(True)

        group = QGroupBox("3. Behavior", self)
        form = QFormLayout()
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)
        form.addRow("Temperature", self.temperature_spin)
        form.addRow("Cost tracking", self.cost_tracking_checkbox)
        group.setLayout(form)
        return group

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QLabel#header_label {
                font-size: 14px;
                font-weight: 600;
                color: #e6edf3;
            }
            QGroupBox {
                border: 1px solid #2a2f3a;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: 600;
                color: #e6edf3;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 4px;
            }
            QPushButton#save_settings_button:disabled {
                background: #333;
                color: #777;
            }
        """
        )

    def _connect_signals(self) -> None:
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.api_key_input.textChanged.connect(self._on_api_key_changed)
        self.max_tokens_spin.valueChanged.connect(self._on_max_tokens_changed)
        self.tool_call_spin.valueChanged.connect(self._on_tool_call_changed)
        self.temperature_spin.valueChanged.connect(self._on_temperature_changed)
        self.cost_tracking_checkbox.toggled.connect(self._on_cost_tracking_toggled)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.close_button.clicked.connect(self.close)

    def _populate_initial_values(self) -> None:
        self._set_combo_value(self.model_combo, self.app_state.agent_model)
        self.api_key_input.setText(self.app_state.anthropic_api_key)
        self.max_tokens_spin.setValue(self.app_state.max_tokens_budget)
        self.tool_call_spin.setValue(self.app_state.tool_call_limit)
        self.temperature_spin.setValue(self.app_state.temperature)
        self.cost_tracking_checkbox.setChecked(self.app_state.cost_tracking_enabled)

    def _set_combo_value(self, combo: QComboBox, target_value: str) -> None:
        combo.blockSignals(True)
        try:
            index = combo.findData(target_value) if target_value else -1
            if index < 0 and combo.count() > 0:
                index = 0
            combo.setCurrentIndex(index)
        finally:
            combo.blockSignals(False)
        if combo is self.model_combo and combo.currentData() and combo.currentData() != target_value:
            self.app_state.set_agent_model(str(combo.currentData()))

    def _on_model_changed(self) -> None:
        model_id = self.model_combo.currentData()
        if model_id:
            self.app_state.set_agent_model(str(model_id))
        self._update_save_button_state()

    def _on_api_key_changed(self, text: str) -> None:
        self.app_state.set_anthropic_api_key(text.strip())
        self._update_save_button_state()

    def _on_max_tokens_changed(self, value: int) -> None:
        self.app_state.set_max_tokens_budget(value)

    def _on_tool_call_changed(self, value: int) -> None:
        self.app_state.set_tool_call_limit(value)

    def _on_temperature_changed(self, value: float) -> None:
        self.app_state.set_temperature(value)

    def _on_cost_tracking_toggled(self, checked: bool) -> None:
        self.app_state.set_cost_tracking_enabled(checked)

    def _update_save_button_state(self) -> None:
        model_ok = bool(self.model_combo.currentData())
        api_key_ok = bool(self.api_key_input.text().strip())
        self.save_settings_button.setEnabled(model_ok and api_key_ok)

    def _validate_required_fields(self) -> bool:
        if not self.model_combo.currentData():
            return False
        if not self.api_key_input.text().strip():
            return False
        return True

    def _save_settings(self) -> None:
        if not self._validate_required_fields():
            QMessageBox.warning(self, "Missing Information", "Model and API key are required.")
            return

        settings = load_settings()
        settings.update(
            {
                "agent_model": self.app_state.agent_model,
                "anthropic_api_key": self.app_state.anthropic_api_key,
                "max_tokens_budget": self.app_state.max_tokens_budget,
                "tool_call_limit": self.app_state.tool_call_limit,
                "temperature": self.app_state.temperature,
                "enable_cost_tracking": self.app_state.cost_tracking_enabled,
            }
        )
        save_settings(settings)
        QMessageBox.information(self, "Settings Saved", "Agent configuration saved successfully.")


__all__ = ["AgentSettingsDialog"]
