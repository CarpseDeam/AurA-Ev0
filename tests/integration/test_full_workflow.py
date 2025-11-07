from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from aura.orchestrator import Orchestrator
from aura.services.chat_service import ChatService


class IntegrationStream:
    """Yield predetermined chunks to simulate Gemini streaming."""

    def __init__(self, generator_factory):
        self._generator = generator_factory()
        self.text = ""

    def __iter__(self):
        return self._generator


@pytest.fixture
def integration_config(monkeypatch):
    created = []

    class DummyConfig:
        def __init__(self, **kwargs):
            created.append(SimpleNamespace(**kwargs))

    monkeypatch.setattr("aura.services.chat_service.types.GenerateContentConfig", DummyConfig)
    return created


def test_full_workflow_success(
    qt_app,
    temp_workspace,
    mock_api_key,
    fake_genai_client,
    integration_config,
):
    with mock.patch("aura.orchestrator.find_cli_agents") as mock_find_agents, \
         mock.patch("aura.orchestrator.run_agent_command_sync", return_value=(0, "agent ok")) as mock_run_sync:

        # Mock the find_cli_agents to return a dummy agent
        dummy_agent = SimpleNamespace(name="gemini", is_available=True, executable_path="dummy/path/gemini")
        mock_find_agents.return_value = [dummy_agent]

        def stream_factory():
            def generator():
                yield SimpleNamespace(text="<CLI_PROMPT>plan work</CLI_PROMPT>", function_calls=None)

            return iter(generator())

        fake_genai_client.models.generate_content_stream.return_value = IntegrationStream(stream_factory)

        service = ChatService(api_key=mock_api_key)
        orchestrator = Orchestrator(
            chat_service=service,
            working_dir=str(temp_workspace),
            agent_path="",
            use_background_thread=False,
        )

        events: list[str] = []
        orchestrator.planning_started.connect(lambda: events.append("planning"))
        orchestrator.session_started.connect(lambda *_: events.append("session_started"))
        orchestrator.session_output.connect(lambda text: events.append(f"output:{text}"))
        orchestrator.session_complete.connect(lambda *_: events.append("session_complete"))
        orchestrator.all_sessions_complete.connect(lambda: events.append("all_complete"))

        orchestrator.execute_goal("Ship feature")

    assert events[0] == "planning"
    assert events[1] == "session_started"
    assert "output:STREAM::<CLI_PROMPT>plan work</CLI_PROMPT>" in events
    assert "output:TOOL_CALL::gemini_cli::Executing agent..." in events
    assert "output:STREAM::agent ok" in events
    assert "session_complete" in events
    assert "all_complete" in events
    mock_run_sync.assert_called_once()
    assert integration_config


def test_full_workflow_error_path(
    qt_app,
    temp_workspace,
    mock_api_key,
    fake_genai_client,
    integration_config,
):
    def stream_factory():
        def generator():
            yield SimpleNamespace(text="Error: CLI failed", function_calls=None)

        return iter(generator())

    fake_genai_client.models.generate_content_stream.return_value = IntegrationStream(stream_factory)

    service = ChatService(api_key=mock_api_key)
    orchestrator = Orchestrator(
        chat_service=service,
        working_dir=str(temp_workspace),
        agent_path="",
        use_background_thread=False,
    )

    errors: list[str] = []
    results = []
    orchestrator.error_occurred.connect(errors.append)
    orchestrator.session_complete.connect(lambda _idx, result: results.append(result))

    orchestrator.execute_goal("This fails")

    assert errors, "Error signal should be emitted for failed workflow"
    assert results and results[0].success is False
    assert integration_config
