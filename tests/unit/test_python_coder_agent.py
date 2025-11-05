"""Tests for PythonCoderAgent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from aura.agents.python_coder import (
    AgentResult,
    PlanParseError,
    PythonCoderAgent,
    SessionContext,
)


# ============================================================================
# Test SessionContext
# ============================================================================


def test_session_context_creation() -> None:
    """Test that SessionContext can be created with required fields."""
    context = SessionContext(
        working_dir=Path("/tmp/test"),
        session_prompt="Create a hello world script",
        previous_work=["Session 1: Created main.py"],
        project_files=["main.py", "utils.py"],
    )

    assert context.working_dir == Path("/tmp/test").resolve()
    assert context.session_prompt == "Create a hello world script"
    assert context.previous_work == ("Session 1: Created main.py",)
    assert context.project_files == ("main.py", "utils.py")


def test_session_context_immutable() -> None:
    """Test that SessionContext is immutable (frozen dataclass)."""
    context = SessionContext(
        working_dir=Path("/tmp/test"),
        session_prompt="Test prompt",
        previous_work=[],
        project_files=[],
    )

    with pytest.raises(AttributeError):
        context.session_prompt = "Modified prompt"  # type: ignore


def test_session_context_converts_to_tuples() -> None:
    """Test that SessionContext converts sequences to tuples."""
    context = SessionContext(
        working_dir=Path("/tmp/test"),
        session_prompt="Test",
        previous_work=["item1", "item2"],
        project_files=["file1.py"],
    )

    assert isinstance(context.previous_work, tuple)
    assert isinstance(context.project_files, tuple)


# ============================================================================
# Test AgentResult
# ============================================================================


def test_agent_result_has_expected_fields() -> None:
    """Test that AgentResult has all expected fields."""
    result = AgentResult(
        success=True,
        summary="Created hello.py",
        files_created=("hello.py",),
        files_modified=(),
        commands_run=("python hello.py",),
        output_lines=("Hello World",),
        errors=(),
        duration_seconds=1.5,
    )

    assert result.success is True
    assert result.summary == "Created hello.py"
    assert result.files_created == ("hello.py",)
    assert result.files_modified == ()
    assert result.commands_run == ("python hello.py",)
    assert result.output_lines == ("Hello World",)
    assert result.errors == ()
    assert result.duration_seconds == 1.5


def test_agent_result_immutable() -> None:
    """Test that AgentResult is immutable (frozen dataclass)."""
    result = AgentResult(
        success=True,
        summary="Test",
        files_created=(),
        files_modified=(),
        commands_run=(),
        output_lines=(),
        errors=(),
        duration_seconds=1.0,
    )

    with pytest.raises(AttributeError):
        result.success = False  # type: ignore


# ============================================================================
# Test PythonCoderAgent Initialization
# ============================================================================


def test_agent_requires_api_key() -> None:
    """Test that PythonCoderAgent requires an API key."""
    with pytest.raises(ValueError, match="Gemini API key is required"):
        PythonCoderAgent(api_key="")


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_agent_initialization_configures_genai(
    mock_model_class: Mock,
    mock_configure: Mock,
) -> None:
    """Test that agent initialization configures genai with API key."""
    agent = PythonCoderAgent(api_key="test-key-12345")

    mock_configure.assert_called_once_with(api_key="test-key-12345")
    mock_model_class.assert_called_once()


# ============================================================================
# Test execute_session with Mocked Gemini
# ============================================================================


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_execute_session_with_simple_plan(
    mock_model_class: Mock,
    mock_configure: Mock,
    tmp_path: Path,
) -> None:
    """Test execute_session with a simple mocked plan."""
    # Mock Gemini response
    mock_response = Mock()
    mock_response.text = """{
        "summary": "Creating hello.py",
        "files": [
            {
                "path": "hello.py",
                "action": "create",
                "content": "print('Hello World')"
            }
        ],
        "commands": ["python hello.py"]
    }"""

    mock_model_instance = Mock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Create agent and context
    agent = PythonCoderAgent(api_key="test-key")
    context = SessionContext(
        working_dir=tmp_path,
        session_prompt="Create a hello world script",
        previous_work=[],
        project_files=[],
    )

    # Execute session
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = Mock(returncode=0, stdout="Hello World\n", stderr="")
        result = agent.execute_session(context)

    # Verify result
    assert result.success is True
    assert result.summary == "Creating hello.py"
    assert "hello.py" in result.files_created
    assert result.commands_run == ("python hello.py",)
    assert result.duration_seconds > 0

    # Verify file was created
    hello_file = tmp_path / "hello.py"
    assert hello_file.exists()
    assert hello_file.read_text() == "print('Hello World')"


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_execute_session_handles_invalid_json(
    mock_model_class: Mock,
    mock_configure: Mock,
    tmp_path: Path,
) -> None:
    """Test that execute_session handles invalid JSON gracefully."""
    # Mock Gemini response with invalid JSON
    mock_response = Mock()
    mock_response.text = "This is not valid JSON"

    mock_model_instance = Mock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Create agent and context
    agent = PythonCoderAgent(api_key="test-key")
    context = SessionContext(
        working_dir=tmp_path,
        session_prompt="Test prompt",
        previous_work=[],
        project_files=[],
    )

    # Execute session
    result = agent.execute_session(context)

    # Verify error handling
    assert result.success is False
    assert len(result.errors) > 0
    assert "parse" in result.errors[0].lower() or "json" in result.errors[0].lower()


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_execute_session_emits_progress_updates(
    mock_model_class: Mock,
    mock_configure: Mock,
    tmp_path: Path,
) -> None:
    """Test that execute_session emits progress_update signals."""
    # Mock Gemini response
    mock_response = Mock()
    mock_response.text = """{
        "summary": "Creating test.py",
        "files": [{"path": "test.py", "action": "create", "content": "pass"}],
        "commands": []
    }"""

    mock_model_instance = Mock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Create agent and context
    agent = PythonCoderAgent(api_key="test-key")
    context = SessionContext(
        working_dir=tmp_path,
        session_prompt="Test",
        previous_work=[],
        project_files=[],
    )

    # Connect signal spy
    emitted_signals = []
    agent.progress_update.connect(lambda msg: emitted_signals.append(msg))

    # Execute session
    agent.execute_session(context)

    # Verify signals were emitted
    assert len(emitted_signals) > 0
    assert any("test.py" in msg for msg in emitted_signals)


# ============================================================================
# Test Command Execution Security
# ============================================================================


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_command_execution_rejects_dangerous_commands(
    mock_model_class: Mock,
    mock_configure: Mock,
    tmp_path: Path,
) -> None:
    """Test that dangerous commands are rejected."""
    # Mock Gemini response with dangerous command
    mock_response = Mock()
    mock_response.text = """{
        "summary": "Dangerous operation",
        "files": [],
        "commands": ["rm -rf /"]
    }"""

    mock_model_instance = Mock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Create agent and context
    agent = PythonCoderAgent(api_key="test-key")
    context = SessionContext(
        working_dir=tmp_path,
        session_prompt="Test",
        previous_work=[],
        project_files=[],
    )

    # Execute session
    result = agent.execute_session(context)

    # Verify command was rejected
    assert len(result.errors) > 0
    assert "rejected" in result.errors[0].lower()
    assert len(result.commands_run) == 0


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_command_execution_allows_safe_commands(
    mock_model_class: Mock,
    mock_configure: Mock,
    tmp_path: Path,
) -> None:
    """Test that safe commands are allowed."""
    # Mock Gemini response with safe commands
    mock_response = Mock()
    mock_response.text = """{
        "summary": "Safe operations",
        "files": [],
        "commands": ["python --version", "pip list", "pytest --help"]
    }"""

    mock_model_instance = Mock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Create agent and context
    agent = PythonCoderAgent(api_key="test-key")
    context = SessionContext(
        working_dir=tmp_path,
        session_prompt="Test",
        previous_work=[],
        project_files=[],
    )

    # Execute session with mocked subprocess
    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = Mock(returncode=0, stdout="", stderr="")
        result = agent.execute_session(context)

    # Verify commands were executed
    assert len(result.commands_run) == 3
    assert result.commands_run[0] == "python --version"
    assert result.commands_run[1] == "pip list"
    assert result.commands_run[2] == "pytest --help"


# ============================================================================
# Test Edge Cases
# ============================================================================


@patch("aura.agents.python_coder.genai.configure")
@patch("aura.agents.python_coder.genai.GenerativeModel")
def test_execute_session_handles_empty_response(
    mock_model_class: Mock,
    mock_configure: Mock,
    tmp_path: Path,
) -> None:
    """Test that empty Gemini response is handled."""
    # Mock empty response
    mock_response = Mock()
    mock_response.text = ""

    mock_model_instance = Mock()
    mock_model_instance.generate_content.return_value = mock_response
    mock_model_class.return_value = mock_model_instance

    # Create agent and context
    agent = PythonCoderAgent(api_key="test-key")
    context = SessionContext(
        working_dir=tmp_path,
        session_prompt="Test",
        previous_work=[],
        project_files=[],
    )

    # Execute session
    result = agent.execute_session(context)

    # Verify error handling
    assert result.success is False
    assert len(result.errors) > 0
