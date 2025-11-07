from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from aura import config
from aura.services.chat_service import (
    AURA_SYSTEM_PROMPT,
    ChatService,
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


def test_cli_prompt_tag_is_emitted(fake_genai_client, mock_api_key, capture_config):
    stream = FakeStream(
        [
            SimpleNamespace(text="<CLI_PROMPT>build</CLI_PROMPT>", function_calls=None),
        ]
    )
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    events: list[str] = []
    result = service.send_message("Need tool", on_chunk=events.append)

    assert "<CLI_PROMPT>build</CLI_PROMPT>" in result


def test_send_message_error_handling(fake_genai_client, mock_api_key, capture_config):
    fake_genai_client.models.generate_content_stream.side_effect = RuntimeError("boom")
    service = ChatService(api_key=mock_api_key)

    result = service.send_message("Fail please")

    assert result == "Error: Unable to reach Gemini. Please verify GEMINI_API_KEY and network connectivity."





def test_all_tools_are_registered_in_config(fake_genai_client, mock_api_key, capture_config):
    stream = FakeStream([])
    fake_genai_client.models.generate_content_stream.return_value = stream
    service = ChatService(api_key=mock_api_key)

    service.send_message("Check config")

    tools = capture_config[0].tools
    assert len(tools) == 16


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
