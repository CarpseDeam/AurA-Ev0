"""Tests for the simplified Aura orchestrator conversation flow."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest
from PySide6.QtCore import QCoreApplication

from aura import config
from aura.orchestrator import Orchestrator, SessionResult


class _StubChatService:
    """Simple chat service stub that records prompts and emits canned replies."""

    def __init__(self, response: str, *, stream: bool = True) -> None:
        self.response = response
        self.stream = stream
        self.prompts: List[str] = []

    def send_message(self, prompt: str, on_chunk=None) -> str:  # noqa: D401 - signature mirrors real service
        self.prompts.append(prompt)
        if self.stream and on_chunk:
            on_chunk(f"{config.STREAM_PREFIX}stub-response")
            on_chunk(f"{config.STREAM_PREFIX}\n")
        return self.response


@pytest.fixture(scope="module")
def qt_app() -> QCoreApplication:
    """Ensure a Qt application instance exists for signal delivery."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def test_execute_goal_streams_response(tmp_path: Path, qt_app: QCoreApplication) -> None:
    """Orchestrator should emit streaming output and completion signals."""
    chat_service = _StubChatService("Stub reply")
    orchestrator = Orchestrator(
        chat_service=chat_service,
        working_dir=str(tmp_path),
        agent_path="",
        use_background_thread=False,
    )

    streamed_chunks: List[str] = []
    completed: List[SessionResult] = []

    orchestrator.session_output.connect(streamed_chunks.append)
    orchestrator.session_complete.connect(lambda _idx, result: completed.append(result))

    orchestrator.execute_goal("Build a demo tool")

    assert chat_service.prompts == ["Build a demo tool"]
    assert streamed_chunks == [
        f"{config.STREAM_PREFIX}stub-response",
        f"{config.STREAM_PREFIX}\n",
    ]
    assert completed, "Expected session_complete to fire once"
    assert completed[0].success is True
    assert orchestrator.history[-1][1] == "Stub reply"


def test_execute_goal_reports_error(tmp_path: Path, qt_app: QCoreApplication) -> None:
    """Errors from ChatService should surface via error signal and failed result."""
    chat_service = _StubChatService("Error: missing credentials", stream=False)
    orchestrator = Orchestrator(
        chat_service=chat_service,
        working_dir=str(tmp_path),
        agent_path="",
        use_background_thread=False,
    )

    errors: List[str] = []
    results: List[SessionResult] = []

    orchestrator.error_occurred.connect(errors.append)
    orchestrator.session_complete.connect(lambda _idx, result: results.append(result))

    orchestrator.execute_goal("Build a demo tool")

    assert chat_service.prompts == ["Build a demo tool"]
    assert errors == ["Error: missing credentials"]
    assert results and results[0].success is False
