from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from aura.orchestrator import Orchestrator
from aura.services.chat_service import ChatService, execute_cli_agent


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
    with mock.patch("aura.services.chat_service.AgentRunner") as runner_cls, mock.patch(
        "aura.services.chat_service.run_agent_command_sync", return_value=(0, "agent ok")
    ):

        def stream_factory():
            def generator():
                tool_call = SimpleNamespace(
                    name="execute_cli_agent",
                    args={"prompt": "plan work", "working_directory": str(temp_workspace)},
                )
                yield SimpleNamespace(text="", function_calls=[tool_call])
                execute_cli_agent("plan work", working_directory=str(temp_workspace))
                yield SimpleNamespace(text="All done", function_calls=None)

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
    assert events[2].startswith("output:TOOL_CALL::execute_cli_agent::")
    assert events[3].startswith("output:STREAM::All done")
    assert "session_complete" in events
    assert "all_complete" in events
    runner_cls.assert_called()
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
