from __future__ import annotations

from unittest import mock

import pytest

from aura.services.agent_runner import AgentRunner, run_agent_command_sync


def test_initialization_validates_command_list(tmp_path):
    with pytest.raises(ValueError):
        AgentRunner(command=[], working_directory=str(tmp_path))


def test_initialization_validates_working_directory():
    with pytest.raises(ValueError):
        AgentRunner(command=["gemini"], working_directory="")


def test_process_output_emitted_line_by_line(qt_app, mock_subprocess, tmp_path):
    mock_subprocess.configure(stdout=["line 1", "line 2"], returncode=0)
    runner = AgentRunner(command=["gemini", "-p", "goal"], working_directory=str(tmp_path))
    outputs: list[str] = []
    finished: list[int] = []
    runner.output_line.connect(outputs.append)
    runner.process_finished.connect(finished.append)

    runner.run()

    assert outputs[:2] == ["line 1", "line 2"]
    assert finished == [0]


def test_exit_code_and_error_signal_on_failure(qt_app, mock_subprocess, tmp_path):
    mock_subprocess.configure(stdout=["bad"], returncode=2)
    runner = AgentRunner(command=["gemini", "-p", "goal"], working_directory=str(tmp_path))
    errors: list[str] = []
    finished: list[int] = []
    runner.process_error.connect(errors.append)
    runner.process_finished.connect(finished.append)

    runner.run()

    assert any("code 2" in err for err in errors)
    assert finished == [2]


def test_spawn_failure_emits_process_error(qt_app, mock_subprocess, tmp_path):
    mock_subprocess.popen.side_effect = OSError("spawn failed")
    runner = AgentRunner(command=["gemini", "-p", "goal"], working_directory=str(tmp_path))
    errors: list[str] = []
    runner.process_error.connect(errors.append)
    runner.process_finished.connect(lambda exit_code: errors.append(f"exit:{exit_code}"))

    runner.run()

    assert any("spawn failed" in err for err in errors)


def test_file_protection_warnings_are_detected(qt_app, mock_subprocess, tmp_path):
    mock_subprocess.configure(stdout=["Creating file src/aura/secret.py"], returncode=0)
    runner = AgentRunner(command=["gemini", "-p", "goal"], working_directory=str(tmp_path))
    outputs: list[str] = []
    runner.output_line.connect(outputs.append)

    with mock.patch("aura.services.agent_runner.is_file_protected", return_value=True):
        runner.run()

    assert any("protected file" in line for line in outputs)


def test_run_agent_command_sync_collects_output(mock_subprocess, tmp_path):
    runner = AgentRunner(command=["gemini", "-p", "goal"], working_directory=str(tmp_path))
    mock_subprocess.configure(stdout=["stdout line"], stderr=["stderr line"], returncode=1)

    exit_code, output = run_agent_command_sync(runner)

    assert exit_code == 1
    assert "stdout line" in output and "[stderr] stderr line" in output
