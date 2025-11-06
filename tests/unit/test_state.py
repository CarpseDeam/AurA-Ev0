from __future__ import annotations

import pytest

from aura.state import AppState


def test_working_directory_setter_validates_and_emits(qt_app, tmp_path):
    state = AppState()
    events: list[str] = []
    state.working_directory_changed.connect(events.append)

    state.set_working_directory(str(tmp_path))

    assert events == [str(tmp_path.resolve())]
    assert state.working_directory == str(tmp_path.resolve())


def test_working_directory_setter_rejects_empty_path(qt_app):
    state = AppState()
    with pytest.raises(ValueError):
        state.set_working_directory("")


def test_working_directory_setter_rejects_missing_path(qt_app, tmp_path):
    state = AppState()
    with pytest.raises(FileNotFoundError):
        state.set_working_directory(str(tmp_path / "missing"))


def test_selected_agent_emits_signal(qt_app):
    state = AppState()
    events: list[str] = []
    state.selected_agent_changed.connect(events.append)

    state.set_selected_agent("gemini")

    assert events == ["gemini"]
    assert state.selected_agent == "gemini"


def test_status_setter_updates_message_and_color(qt_app):
    state = AppState()
    events = []
    state.status_changed.connect(lambda message, color: events.append((message, color)))

    state.set_status("Running", "#00ff00")

    assert events == [("Running", "#00ff00")]
    assert state.status_message == "Running"
    assert state.status_color == "#00ff00"
