from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from aura import config
from aura.services.chat_service import (
    AURA_SYSTEM_PROMPT,
    ChatService,
    execute_cli_agent,
)


class FakeStream:
    """Simple iterable mimicking the Gemini streaming interface."""

    def __init__(self, chunks, *, final_text: str = "") -> None:
        self._chunks = list(chunks)
        self.text = final_text

    def __iter__(self):
        return iter(self._chunks)


@pytest.fixture
def capture_config(monkeypatch):
    created = []

    class DummyConfig:
        def __init__(self, **kwargs):
            self.tools = kwargs.get("tools", [])
            self.system_instruction = kwargs.get("system_instruction", "")
            created.append(self)

    monkeypatch.setattr("aura.services.chat_service.types.GenerateContentConfig", DummyConfig)
    return created


def test_send_message_with_mocked_gemini_api(fake_genai_client, mock_api_key, capture_config):
    stream = FakeStream(
        [
            SimpleNamespace(text="Hello", function_calls=None),
            SimpleNamespace(text="Hello world", function_calls=None),
        ]
    )
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    result = service.send_message("Hi there")

    assert result == "Hello world"
    assert capture_config[0].tools, "Tools should be included in the config"


def test_streaming_chunks_are_emitted_via_callback(fake_genai_client, mock_api_key, capture_config):
    stream = FakeStream(
        [
            SimpleNamespace(text="Alpha", function_calls=None),
            SimpleNamespace(text="AlphaBeta", function_calls=None),
        ]
    )
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    chunks: list[str] = []
    service.send_message("Stream please", on_chunk=chunks.append)

    assert chunks == [
        f"{config.STREAM_PREFIX}Alpha",
        f"{config.STREAM_PREFIX}Beta",
        f"{config.STREAM_PREFIX}\n",
    ]


def test_tool_execution_emits_tool_call_chunks(fake_genai_client, mock_api_key, capture_config):
    call = SimpleNamespace(name="execute_cli_agent", args={"prompt": "build"})
    stream = FakeStream(
        [
            SimpleNamespace(text="", function_calls=[call]),
            SimpleNamespace(text="", function_calls=[call]),
            SimpleNamespace(text="Agent complete", function_calls=None),
        ]
    )
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    tool_events: list[str] = []
    result = service.send_message("Need tool", on_chunk=tool_events.append)

    assert tool_events[0].startswith("TOOL_CALL::execute_cli_agent::")
    assert tool_events[1] == f"{config.STREAM_PREFIX}Agent complete"
    assert tool_events[2] == f"{config.STREAM_PREFIX}\n"
    assert result.endswith("Agent complete")


def test_send_message_error_handling(fake_genai_client, mock_api_key, capture_config):
    fake_genai_client.models.generate_content_stream.side_effect = RuntimeError("boom")
    service = ChatService(api_key=mock_api_key)

    result = service.send_message("Fail please")

    assert result.startswith("Error:")


def test_execute_cli_agent_runs_successfully(temp_workspace):
    with mock.patch("aura.services.chat_service.AgentRunner") as runner_cls, mock.patch(
        "aura.services.chat_service.run_agent_command_sync", return_value=(0, "ok")
    ):
        result = execute_cli_agent("Do work", working_directory=str(temp_workspace))

    runner_cls.assert_called_once()
    assert result["success"] is True
    assert result["exit_code"] == 0


def test_execute_cli_agent_handles_missing_workspace(temp_workspace):
    missing = temp_workspace / "gone"
    result = execute_cli_agent("noop", working_directory=str(missing))

    assert result["success"] is False
    assert "workspace" in result["output"].lower()


def test_all_tools_are_registered_in_config(fake_genai_client, mock_api_key, capture_config):
    stream = FakeStream([])
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    service.send_message("Check config")

    tools = capture_config[0].tools
    assert len(tools) == 17
    assert execute_cli_agent in tools


def test_system_prompt_is_applied(fake_genai_client, mock_api_key, capture_config):
    stream = FakeStream([])
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    service.send_message("Prompt check")

    assert capture_config[0].system_instruction == AURA_SYSTEM_PROMPT


def test_conversation_without_streaming_falls_back_to_final_text(
    fake_genai_client, mock_api_key, capture_config
):
    stream = FakeStream([], final_text="Fallback response")
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    result = service.send_message("No streaming")

    assert result == "Fallback response"
