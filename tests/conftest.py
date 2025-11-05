"""Shared pytest fixtures for Aura test suite."""

import copy
import os
from typing import Generator
from unittest.mock import Mock

import pytest

from aura.services.chat_service import ChatService
from aura.services.planning_service import PlanningService
from tests.fixtures.mock_responses import get_mock_plan


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require API key)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line(
        "markers",
        "fast: marks tests as fast (mocked, no API)",
    )


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get GEMINI_API_KEY from environment."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY environment variable not set")
    return key


@pytest.fixture(scope="session")
def chat_service(api_key: str) -> ChatService:
    """Create a ChatService instance for testing."""
    return ChatService(api_key=api_key)


@pytest.fixture(scope="function")
def planning_service(chat_service: ChatService) -> Generator[PlanningService, None, None]:
    """Create a fresh PlanningService instance for each test."""
    service = PlanningService(chat_service)
    yield service
    # Cleanup: clear chat history for next test
    chat_service.clear_history()


@pytest.fixture(scope="function")
def mock_chat_service() -> Mock:
    """Provide a mocked chat service that never hits the real API."""
    mock = Mock(spec=ChatService)
    mock.clear_history = Mock()
    mock.send_message.side_effect = RuntimeError("Mock chat service does not support send_message.")
    return mock


@pytest.fixture(scope="function")
def mock_planning_service(mock_chat_service: Mock) -> PlanningService:
    """Create a mocked PlanningService for fast tests."""
    service = PlanningService(mock_chat_service)

    simple_keywords = ("calculator", "hello", "simple", "organizer", "converter", "cli")
    medium_keywords = ("todo", "url", "api", "csv", "queue", "processor", "shortener", "pipeline")

    def mock_plan_sessions(goal: str, context: str):
        goal_lower = goal.lower()
        if any(word in goal_lower for word in simple_keywords):
            complexity = "simple"
        elif any(word in goal_lower for word in medium_keywords):
            complexity = "medium"
        else:
            complexity = "complex"

        payload = copy.deepcopy(get_mock_plan(complexity))

        context_lower = context.lower()
        if any(keyword in context_lower for keyword in ("existing", "main.py", "src/")):
            for session in payload["sessions"]:
                session["prompt"] += " Use existing project files when they are available."
            payload["reasoning"] += " Leverages existing project structure when it is provided."
        else:
            payload["reasoning"] += " Starts from a blank slate when no existing code is supplied."

        return service._build_plan(payload)

    service.plan_sessions = mock_plan_sessions  # type: ignore[assignment]
    return service


@pytest.fixture(scope="session")
def sample_project_context() -> str:
    """Provide a sample project context for testing."""
    return """Working directory: /home/user/project
Directories:
- src
- tests
- docs
Python files:
- src/__init__.py
- src/main.py
- tests/test_main.py
"""


@pytest.fixture(scope="session")
def empty_project_context() -> str:
    """Provide an empty project context for testing."""
    return """Working directory: /home/user/newproject
Directories:
- None
Python files:
- None
"""
