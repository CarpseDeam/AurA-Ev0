"""Tests for session executors."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.aura.agents import AgentResult, SessionContext
from src.aura.execution.executors import NativeAgentExecutor


@pytest.fixture
def native_agent_executor() -> NativeAgentExecutor:
    """Fixture for a NativeAgentExecutor with a mock API key."""
    return NativeAgentExecutor(api_key="test-key")


def test_validate_context_gathering_with_signatures(
    native_agent_executor: NativeAgentExecutor, caplog
) -> None:
    """Test that validation passes when function signatures are present."""
    session_context = SessionContext(
        working_dir=".",
        session_prompt="test",
        previous_work=["some work"],
        project_files=[],
        function_signatures={"file1.py": [{"name": "func1"}]},
    )
    agent_result = AgentResult(
        success=True, summary="summary", files_created=(), files_modified=(),
        commands_run=(), output_lines=(), errors=(), duration_seconds=0.0
    )

    with caplog.at_level(logging.INFO):
        native_agent_executor._validate_context_gathering(session_context, agent_result)

    assert "Validation passed: 1 function signatures were auto-injected" in caplog.text


def test_validate_context_gathering_no_signatures(
    native_agent_executor: NativeAgentExecutor, caplog
) -> None:
    """Test that a warning is logged when no function signatures are present."""
    session_context = SessionContext(
        working_dir=".",
        session_prompt="test",
        previous_work=["some work"],
        project_files=[],
        function_signatures={},
    )
    agent_result = AgentResult(
        success=True, summary="summary", files_created=(), files_modified=(),
        commands_run=(), output_lines=(), errors=(), duration_seconds=0.0
    )

    with caplog.at_level(logging.WARNING):
        native_agent_executor._validate_context_gathering(session_context, agent_result)

    assert "Validation warning: No function signatures were found" in caplog.text


def test_validate_context_gathering_no_previous_work(
    native_agent_executor: NativeAgentExecutor, caplog
) -> None:
    """Test that validation is skipped when there is no previous work."""
    session_context = SessionContext(
        working_dir=".",
        session_prompt="test",
        previous_work=[],
        project_files=[],
    )
    agent_result = AgentResult(
        success=True, summary="summary", files_created=(), files_modified=(),
        commands_run=(), output_lines=(), errors=(), duration_seconds=0.0
    )

    with caplog.at_level(logging.INFO):
        native_agent_executor._validate_context_gathering(session_context, agent_result)

    assert "Validation skipped: No previous work for this session." in caplog.text
