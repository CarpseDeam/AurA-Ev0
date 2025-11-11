"""Integration tests for orchestration components."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aura.models import Conversation, MessageRole
from aura.orchestrator import Orchestrator
from aura.services.analyst_agent_service import AnalystAgentService
from aura.tools.tool_manager import ToolManager
from tests.helpers import SimpleChatService


def test_orchestrator_updates_tool_manager_workspace(app_state, workspace_dir: Path) -> None:
    chat_service = SimpleChatService(response_text="ok", tool_manager=None)
    orchestrator = Orchestrator(
        app_state=app_state,
        chat_service=chat_service,
        use_background_thread=False,
    )

    new_workspace = workspace_dir / "nested"
    new_workspace.mkdir()
    app_state.set_working_directory(str(new_workspace))

    orchestrator.update_working_directory(str(new_workspace))
    assert Path(orchestrator._tool_manager.workspace_dir) == new_workspace.resolve()


def test_analyst_service_routes_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aura.services.analyst_agent_service.anthropic.Anthropic",
        lambda api_key: object(),
    )
    fake_tool_manager = MagicMock(spec=ToolManager)

    service = AnalystAgentService(
        api_key="token",
        tool_manager=fake_tool_manager,
        model_name="claude-sonnet",
    )

    handler = service._tool_handlers["list_project_files"]
    handler()

    fake_tool_manager.list_project_files.assert_called_once()


def test_orchestrator_persists_conversation_history(app_state, isolated_db, workspace_dir: Path) -> None:
    chat_service = SimpleChatService(
        response_text="Completed successfully.",
        tool_manager=None,
        streamed_chunks=["chunk-1", "chunk-2"],
    )
    orchestrator = Orchestrator(
        app_state=app_state,
        chat_service=chat_service,
        use_background_thread=False,
    )

    orchestrator.execute_goal("Say hello to the user.")

    conversation = Conversation.get_most_recent()
    assert conversation is not None
    assert app_state.current_conversation_id == conversation.id

    messages = conversation.get_messages()
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert "Say hello" in messages[0].content
    assert messages[1].role == MessageRole.ASSISTANT
    assert "Completed successfully." in messages[1].content
