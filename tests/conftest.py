"""Shared pytest fixtures for Aura test suite."""

import os
from typing import Generator

import pytest

from aura.services.chat_service import ChatService
from aura.services.planning_service import PlanningService


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
