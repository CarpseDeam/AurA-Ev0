"""Dialog for configuring Aura's API model selections."""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aura.state import AppState
from aura.utils.model_discovery import (
    ModelInfo,
    discover_claude_models,
    discover_gemini_models,
    discover_ollama_models,
)
from aura.utils.settings import save_settings


class AgentSettingsDialog(QDialog):
    """Lightweight dialog for selecting analyst/executor/specialist models."""

    def __init__(self, app_state: AppState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state

        self.analyst_investigation_combo = QComboBox(self)
        self.analyst_model_combo = QComboBox(self)
        self.analyst_refresh_button = QPushButton("Refresh", self)

        self.executor_model_combo = QComboBox(self)
        self.executor_refresh_button = QPushButton("Refresh", self)

        self.specialist_model_combo = QComboBox(self)
        self.specialist_model_combo.setEditable(False)
        self.specialist_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.specialist_refresh_button = QPushButton("Refresh", self)

        self.use_local_investigation_checkbox = QCheckBox("Use local model for investigation (deepseek-coder-v2:16b)", self)
        self.use_local_investigation_checkbox.setChecked(self.app_state.use_local_investigation)

        self.save_settings_button = QPushButton("Save", self)
        self.save_settings_button.setObjectName("save_settings_button")
        self.close_button = QPushButton("Close", self)

        self._configure_dialog()
        self._build_layout()
        self._apply_styles()
        self._connect_signals()

        self._refresh_cloud_models()
        self._refresh_ollama_models()

    def _configure_dialog(self) -> None:
        self.setWindowTitle("Model Settings")
        self.resize(520, 280)
        self.setModal(True)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        header = QLabel("Configure which models power Aura's Analyst, Executor, and Specialist roles.")
        header.setObjectName("header_label")
        header.setWordWrap(True)
        layout.addWidget(header)

        form = QFormLayout()
        form.setContentsMargins(0, 8, 0, 0)
        form.setSpacing(12)

        analyst_column = QVBoxLayout()
        analyst_column.setSpacing(6)
        investigation_label = QLabel("Phase 1 – investigation (tool calls)", self)
        investigation_label.setObjectName("analyst_phase_label")
        analyst_column.addWidget(investigation_label)
        analyst_column.addWidget(self.analyst_investigation_combo)
        planning_label = QLabel("Phase 2 – planning (ExecutionPlan)", self)
        planning_label.setObjectName("analyst_phase_label")
        analyst_column.addWidget(planning_label)
        analyst_column.addWidget(self.analyst_model_combo)

        analyst_row = QHBoxLayout()
        analyst_row.addLayout(analyst_column, stretch=1)
        analyst_row.addWidget(self.analyst_refresh_button)
        form.addRow("Analyst Models:", analyst_row)

        executor_row = QHBoxLayout()
        executor_row.addWidget(self.executor_model_combo, stretch=1)
        executor_row.addWidget(self.executor_refresh_button)
        form.addRow("Executor Model (write-enabled):", executor_row)

        specialist_row = QHBoxLayout()
        specialist_row.addWidget(self.specialist_model_combo, stretch=1)
        specialist_row.addWidget(self.specialist_refresh_button)
        form.addRow("Specialist Model (local/Ollama):", specialist_row)

        # Add local investigation checkbox
        form.addRow("", self.use_local_investigation_checkbox)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.save_settings_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QLabel#header_label {
                font-size: 14px;
                font-weight: 600;
                color: #e6edf3;
            }
            QPushButton#save_settings_button {
                font-weight: 600;
                padding: 6px 16px;
            }
            QLabel#analyst_phase_label {
                font-size: 12px;
                color: #8b949e;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.analyst_refresh_button.clicked.connect(self._refresh_cloud_models)
        self.executor_refresh_button.clicked.connect(self._refresh_cloud_models)
        self.specialist_refresh_button.clicked.connect(self._refresh_ollama_models)
        self.analyst_investigation_combo.currentTextChanged.connect(
            self._on_analyst_investigation_model_selected
        )
        self.analyst_model_combo.currentTextChanged.connect(self._on_analyst_model_selected)
        self.executor_model_combo.currentTextChanged.connect(self._on_executor_model_selected)
        self.specialist_model_combo.currentTextChanged.connect(self._on_specialist_model_selected)
        self.use_local_investigation_checkbox.stateChanged.connect(self._on_use_local_investigation_changed)
        self.save_settings_button.clicked.connect(self._save_settings)
        self.close_button.clicked.connect(self.accept)

    def _refresh_cloud_models(self) -> None:
        """Fetch and populate hosted Claude/Gemini models."""
        models: list[ModelInfo | dict | str] = []

        gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if gemini_api_key:
            models.extend(discover_gemini_models(gemini_api_key))

        claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if claude_api_key:
            models.extend(discover_claude_models(claude_api_key))

        has_any_api_key = bool(gemini_api_key or claude_api_key)
        error_message = (
            "Set GEMINI_API_KEY or ANTHROPIC_API_KEY to discover models"
            if not has_any_api_key
            else "Unable to fetch models (check network/key)"
        )

        self._update_model_combo(self.analyst_investigation_combo, models, error_message)
        self._update_model_combo(self.analyst_model_combo, models, error_message)
        self._update_model_combo(self.executor_model_combo, models, error_message)

    def _refresh_ollama_models(self) -> None:
        """Fetch local Ollama models for the specialist role."""
        models = discover_ollama_models()
        message = "No local models found (is Ollama running?)"
        self._update_model_combo(self.specialist_model_combo, models, message)

        # Re-select stored specialist model when available.
        current_model_text = self.app_state.specialist_model
        index = self.specialist_model_combo.findText(current_model_text)
        if index != -1:
            self.specialist_model_combo.setCurrentIndex(index)
        elif self.specialist_model_combo.count() > 0:
            self.specialist_model_combo.setCurrentIndex(0)

    def _update_model_combo(
        self,
        combo: QComboBox,
        models: list[ModelInfo | dict | str],
        error_message: str,
    ) -> None:
        """Populate the supplied combo box with the discovered models."""
        current_selection = combo.currentData()
        if not current_selection:
            stripped_text = combo.currentText().strip()
            current_selection = stripped_text or None

        if not current_selection:
            if combo is self.analyst_model_combo:
                current_selection = self.app_state.analyst_planning_model
            elif combo is self.analyst_investigation_combo:
                current_selection = self.app_state.analyst_investigation_model
            elif combo is self.executor_model_combo:
                current_selection = self.app_state.executor_model
            elif combo is self.specialist_model_combo:
                current_selection = self.app_state.specialist_model

        combo.blockSignals(True)
        try:
            combo.clear()

            if not models:
                combo.addItem(error_message, None)
                combo.setItemData(0, error_message, Qt.ItemDataRole.ToolTipRole)
                combo.setEnabled(False)
                combo.setCurrentIndex(0)
                return

            combo.setEnabled(True)
            for model in models:
                model_id = ""
                display_text = ""
                tooltip_text: str | None = None

                if isinstance(model, ModelInfo):
                    model_id = model.model_id
                    display_text = model.display_name or model.model_id
                    tooltip_text = model.description
                elif isinstance(model, dict):
                    model_id = str(
                        model.get("model_id")
                        or model.get("id")
                        or model.get("name")
                        or model.get("value")
                        or ""
                    ).strip()
                    display_text = (
                        model.get("display_name")
                        or model.get("name")
                        or model.get("title")
                        or model_id
                    )
                    tooltip_text = model.get("description")
                else:
                    model_id = str(model).strip()
                    display_text = model_id

                if not model_id:
                    continue

                item_index = combo.count()
                combo.addItem(display_text, model_id)
                if tooltip_text:
                    combo.setItemData(item_index, tooltip_text, Qt.ItemDataRole.ToolTipRole)

            if combo.count() == 0:
                combo.addItem(error_message, None)
                combo.setItemData(0, error_message, Qt.ItemDataRole.ToolTipRole)
                combo.setEnabled(False)
                combo.setCurrentIndex(0)
                return

            target_index = -1
            if current_selection:
                target_index = combo.findData(current_selection)
                if target_index == -1 and isinstance(current_selection, str):
                    target_index = combo.findText(current_selection)

            combo.setCurrentIndex(target_index if target_index != -1 else 0)
        finally:
            combo.blockSignals(False)

    def _on_analyst_model_selected(self, text: str) -> None:
        model_id = self.analyst_model_combo.currentData() or text.strip()
        if model_id:
            self.app_state.set_analyst_model(model_id)

    def _on_analyst_investigation_model_selected(self, text: str) -> None:
        model_id = self.analyst_investigation_combo.currentData() or text.strip()
        if model_id:
            self.app_state.set_analyst_investigation_model(model_id)

    def _on_executor_model_selected(self, text: str) -> None:
        model_id = self.executor_model_combo.currentData() or text.strip()
        if model_id:
            self.app_state.set_executor_model(model_id)

    def _on_specialist_model_selected(self, text: str) -> None:
        model_id = self.specialist_model_combo.currentData() or text.strip()
        if model_id:
            self.app_state.set_specialist_model(model_id)

    def _on_use_local_investigation_changed(self, state: int) -> None:
        """Handle local investigation checkbox state changes."""
        enabled = state == Qt.CheckState.Checked.value
        self.app_state.set_use_local_investigation(enabled)

    def _save_settings(self) -> None:
        settings = {
            "analyst_model": self.app_state.analyst_planning_model,
            "analyst_planning_model": self.app_state.analyst_planning_model,
            "analyst_investigation_model": self.app_state.analyst_investigation_model,
            "executor_model": self.app_state.executor_model,
            "specialist_model": self.app_state.specialist_model,
            "use_local_investigation": self.app_state.use_local_investigation,
        }
        save_settings(settings)
        QMessageBox.information(self, "Settings Saved", "Model selections have been saved successfully.")


__all__ = ["AgentSettingsDialog"]
