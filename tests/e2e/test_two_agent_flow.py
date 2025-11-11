"""End-to-end test covering the full analyst/executor orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from aura import config
from aura.models.execution_plan import ExecutionPlan, FileOperation, OperationType
from aura.orchestrator import Orchestrator
from aura.ui.orchestration_handler import OrchestrationHandler
from tests.helpers import RecordingOutputPanel, RecordingStatusManager, SimpleChatService


class FakeAnalystAgent:
    """Deterministic analyst that always returns a simple plan."""

    def __init__(self, api_key: str, tool_manager, model_name: str) -> None:  # noqa: ANN001
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.calls: List[str] = []

    def analyze_and_plan(self, user_request: str, on_chunk=None, conversation_id=None, conversation_history=None) -> ExecutionPlan:
        self.calls.append(user_request)
        if on_chunk:
            on_chunk("<thinking>Validating workspace context...</thinking>")

        operation = FileOperation(
            operation_type=OperationType.CREATE,
            file_path="hello.txt",
            content="world",
            rationale="Create the requested file",
            dependencies=[],
        )
        return ExecutionPlan(
            task_summary="Create hello.txt with requested contents.",
            project_context="Workspace inspected.",
            operations=[operation],
            quality_checklist=["Confirm hello.txt exists with correct content"],
            estimated_files=1,
        )


class FakeExecutorAgent:
    """Executor that applies the provided plan via ToolManager."""

    def __init__(self, api_key: str, tool_manager, model_name: str) -> None:  # noqa: ANN001
        self.tool_manager = tool_manager
        self.model_name = model_name
        self.executed_plans: list[ExecutionPlan] = []

    def execute_plan(self, execution_plan: ExecutionPlan, on_chunk=None, conversation_id=None) -> str:
        self.executed_plans.append(execution_plan)
        for operation in execution_plan.operations:
            if operation.operation_type is OperationType.CREATE and operation.content is not None:
                self.tool_manager.create_file(operation.file_path, operation.content)
        if on_chunk:
            on_chunk("Executor finished applying the plan.")
        return "Execution complete."


def test_two_agent_flow_creates_file_and_updates_ui(
    monkeypatch: pytest.MonkeyPatch,
    app_state,
    workspace_dir: Path,
) -> None:
    monkeypatch.setattr("aura.orchestrator.AnalystAgentService", FakeAnalystAgent)
    monkeypatch.setattr("aura.orchestrator.ExecutorAgentService", FakeExecutorAgent)

    chat_service = SimpleChatService(response_text="unused", tool_manager=None)
    orchestrator = Orchestrator(
        app_state=app_state,
        chat_service=chat_service,
        analyst_api_key="key",
        executor_api_key="key",
        use_background_thread=False,
    )

    output_panel = RecordingOutputPanel()
    status_manager = RecordingStatusManager()
    handler = OrchestrationHandler(
        output_panel=output_panel,
        status_manager=status_manager,
        app_state=app_state,
    )
    handler.request_input_enabled.connect(lambda _: None)
    handler.request_input_focus.connect(lambda: None)

    orchestrator.planning_started.connect(handler.handle_planning_started)
    orchestrator.session_started.connect(handler.handle_session_started)
    orchestrator.session_output.connect(handler.handle_session_output)
    orchestrator.session_complete.connect(handler.handle_session_complete)
    orchestrator.all_sessions_complete.connect(handler.handle_all_complete)
    orchestrator.error_occurred.connect(handler.handle_error)

    def progress_listener(message: str) -> None:
        color = config.COLORS.thinking if "⋯" in message else config.COLORS.accent
        status_manager.update_status(message, color, True)

    orchestrator.progress_update.connect(progress_listener)

    orchestrator.execute_goal("Create hello.txt with the word world.")

    created_file = workspace_dir / "hello.txt"
    assert created_file.read_text(encoding="utf-8") == "world"

    success_messages = [entry for entry in output_panel.messages if entry[0] == "success"]
    assert any("Response complete" in message[1] for message in success_messages)
    assert any("Conversation finished" in message[1] for message in success_messages)

    status_messages = [entry[0] for entry in status_manager.updates]
    assert "Completed" in status_messages
    assert status_messages[-1].startswith("✅ Conversation complete")
