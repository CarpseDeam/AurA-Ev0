"""Shared pytest fixtures for the Aura test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from aura import database
from aura.state import AppState
from aura.tools.tool_manager import ToolManager


@pytest.fixture()
def workspace_dir(tmp_path: Path) -> Path:
    """Provide an isolated workspace directory for filesystem-heavy tests."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture()
def tool_manager(workspace_dir: Path) -> ToolManager:
    """Instantiate a ToolManager scoped to the temporary workspace."""
    return ToolManager(str(workspace_dir))


@pytest.fixture()
def app_state(workspace_dir: Path) -> AppState:
    """Return an AppState that already tracks the temporary workspace."""
    state = AppState()
    state.set_working_directory(str(workspace_dir))
    return state


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Redirect the SQLite database to a temporary file for each test."""
    db_path = tmp_path / "aura.db"
    monkeypatch.setattr(database, "get_database_path", lambda: db_path)
    database.initialize_database()
    yield db_path


@pytest.fixture(scope="session", autouse=True)
def qt_app() -> QApplication:
    """Ensure a QApplication exists for Qt-based components used in tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
