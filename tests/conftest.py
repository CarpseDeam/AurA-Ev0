"""Shared pytest configuration and fixtures for the Aura test suite."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from PySide6.QtWidgets import QApplication

from aura.services.chat_service import ChatService


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI flags for selective test execution."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that exercise the chat/orchestration stack.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers, enable logging, and configure coverage defaults."""
    logging.basicConfig(level=os.environ.get("PYTEST_LOG_LEVEL", "INFO"))
    for marker in ("integration", "slow", "fast"):
        config.addinivalue_line("markers", f"{marker}: custom marker from conftest")

    if config.pluginmanager.hasplugin("cov"):
        if not getattr(config.option, "cov_source", None):
            config.option.cov_source = ["src/aura"]
        if not getattr(config.option, "cov_report", None):
            config.option.cov_report = ["term-missing"]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless explicitly requested."""
    if config.getoption("--run-integration"):
        return
    skip_marker = pytest.mark.skip(reason="use --run-integration to run integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """Provide a QApplication instance for all Qt-dependent tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_api_key() -> str:
    """Return a deterministic mock API key for ChatService tests."""
    return "test-api-key"


@pytest.fixture
def temp_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a temporary workspace populated with representative files."""
    workspace = tmp_path_factory.mktemp("workspace")
    (workspace / "src").mkdir()
    (workspace / "src" / "module.py").write_text("print('hello world')\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Sample Project\n", encoding="utf-8")
    return workspace


@pytest.fixture
def sample_python_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a sample Python module used for AST-driven tool tests."""
    root = tmp_path_factory.mktemp("python_sample")
    file_path = root / "sample_module.py"
    file_path.write_text(
        (
            "class Sample:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n\n"
            "def helper(alpha, beta=1):\n"
            '    \"\"\"Example helper.\"\"\"\n'
            "    return alpha + beta\n"
        ),
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def mock_chat_service() -> ChatService:
    """Provide a ChatService mock with a controllable send_message method."""
    service = mock.create_autospec(ChatService, instance=True)
    service.send_message.return_value = "ok"
    return service


@pytest.fixture
def mock_subprocess() -> SimpleNamespace:
    """Patch subprocess.Popen used by the agent runner to avoid real execution."""
    patcher = mock.patch("aura.services.agent_runner.subprocess.Popen")
    popen_patch = patcher.start()

    def _make_stream(lines: list[str]) -> io.StringIO:
        data = "\n".join(lines)
        if data:
            data += "\n"
        return io.StringIO(data)

    def factory(
        stdout: list[str] | None = None,
        stderr: list[str] | None = None,
        returncode: int = 0,
    ):
        process = mock.Mock()
        process.stdout = _make_stream(stdout or [])
        process.stderr = _make_stream(stderr or [])
        process.wait.return_value = returncode
        popen_patch.return_value = process
        return process

    yield SimpleNamespace(configure=factory, popen=popen_patch)
    patcher.stop()


@pytest.fixture
def fake_genai_client():
    """Patch genai.Client so ChatService can be instantiated safely."""
    client = mock.Mock()
    client.models = mock.Mock()
    client.models.generate_content_stream.return_value = []
    with mock.patch("aura.services.chat_service.genai.Client", return_value=client):
        yield client
