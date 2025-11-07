from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from aura import config
from aura.exceptions import AuraConfigurationError
from aura.orchestrator import Orchestrator


def _build_orchestrator(mock_chat_service, workspace: Path, *, background: bool = False) -> Orchestrator:
    return Orchestrator(
        working_dir=str(workspace),
        agent_path="",
        chat_service=mock_chat_service,
        use_background_thread=background,
    )


def test_execute_goal_starts_conversation_successfully(qt_app, mock_chat_service, temp_workspace):
    mock_chat_service.send_message.return_value = "Conversation complete"
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace, background=False)

    outputs: list[str] = []
    results = []
    orchestrator.session_output.connect(outputs.append)
    orchestrator.session_complete.connect(lambda _idx, result: results.append(result))

    orchestrator.execute_goal("Ship it")

    assert mock_chat_service.send_message.called, "ChatService should be invoked"
    assert outputs == [
        f"{config.STREAM_PREFIX}Conversation complete",
        f"{config.STREAM_PREFIX}\n",
    ]
    assert results and results[0].success is True


def test_streaming_output_is_emitted_via_signals(qt_app, mock_chat_service, temp_workspace):
    def fake_send(prompt, on_chunk=None):
        if on_chunk:
            on_chunk("chunk-1")
            on_chunk("chunk-2")
        return "final text"

    mock_chat_service.send_message.side_effect = fake_send
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace, background=False)

    outputs: list[str] = []
    orchestrator.session_output.connect(outputs.append)

    orchestrator.execute_goal("Need chunking")

    assert outputs[:2] == ["chunk-1", "chunk-2"], "Streamed chunks should pass through the signal"


def test_error_handling_when_chat_service_fails(qt_app, mock_chat_service, temp_workspace):
    mock_chat_service.send_message.side_effect = RuntimeError("network down")
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace, background=False)

    errors: list[str] = []
    orchestrator.error_occurred.connect(errors.append)

    orchestrator.execute_goal("Handle failure")

    assert errors, "Errors should be emitted on failure"
    assert "Unable to contact Gemini" in errors[0]


def test_history_tracking_stores_conversation_turns(qt_app, mock_chat_service, temp_workspace):
    mock_chat_service.send_message.side_effect = ["Response one", "Response two"]
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace, background=False)

    orchestrator.execute_goal("Goal 1")
    orchestrator.execute_goal("Goal 2")

    history = orchestrator.history
    assert history == (
        ("user", "Goal 1"),
        ("assistant", "Response one"),
        ("user", "Goal 2"),
        ("assistant", "Response two"),
    )


def test_update_working_directory_validates_path_exists(qt_app, mock_chat_service, temp_workspace):
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace)
    with pytest.raises(AuraConfigurationError):
        orchestrator.update_working_directory(str(temp_workspace / "missing"))


def test_update_agent_path_stores_path_correctly(qt_app, mock_chat_service, temp_workspace):
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace)
    orchestrator.update_agent_path("/tmp/agent")
    assert orchestrator._agent_path == "/tmp/agent"


def test_update_working_directory_refreshes_tool_manager(qt_app, mock_chat_service, temp_workspace):
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace)
    new_workspace = temp_workspace / "nested"
    new_workspace.mkdir()

    orchestrator.update_working_directory(str(new_workspace))

    assert orchestrator._tool_manager.workspace_dir == new_workspace.resolve()
    assert mock_chat_service.tool_manager is orchestrator._tool_manager


def test_execute_goal_runs_inline_when_background_disabled(qt_app, mock_chat_service, temp_workspace):
    mock_chat_service.send_message.return_value = "done"
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace, background=False)

    with mock.patch.object(
        orchestrator, "_run_conversation", wraps=orchestrator._run_conversation
    ) as run_spy, mock.patch.object(
        orchestrator, "_start_background_conversation"
    ) as bg_mock:
        orchestrator.execute_goal("Sync please")

    assert run_spy.call_count == 1
    bg_mock.assert_not_called()


def test_execute_goal_uses_background_thread_when_enabled(
    qt_app, mock_chat_service, temp_workspace
):
    mock_chat_service.send_message.return_value = "done"
    orchestrator = _build_orchestrator(mock_chat_service, temp_workspace, background=True)

    with mock.patch.object(Orchestrator, "_start_background_conversation") as bg_called, mock.patch.object(
        Orchestrator, "_run_conversation"
    ) as sync_called:
        orchestrator.execute_goal("Async mode")

    bg_called.assert_called_once()
    sync_called.assert_not_called()
