from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from aura.tools import git_tools


def test_get_git_status_parses_output():
    with mock.patch(
        "aura.tools.git_tools.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout=" M app.py\n", stderr=""),
    ):
        status = git_tools.get_git_status()

    assert status == "M app.py"


def test_get_git_status_handles_error():
    with mock.patch(
        "aura.tools.git_tools.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="fatal"),
    ):
        status = git_tools.get_git_status()

    assert status.startswith("Error:")


def test_git_commit_validates_command_format():
    with mock.patch(
        "aura.tools.git_tools.subprocess.run",
        side_effect=[
            SimpleNamespace(returncode=0, stdout="", stderr=""),  # git add
            SimpleNamespace(returncode=0, stdout="1 file changed", stderr=""),  # commit
        ],
    ) as mock_run:
        result = git_tools.git_commit("Initial commit")

    assert "Committed successfully" in result
    assert mock_run.call_args_list[0].args[0] == ["git", "add", "."]
    assert mock_run.call_args_list[1].args[0][:3] == ["git", "commit", "-m"]


def test_git_commit_handles_failure():
    with mock.patch(
        "aura.tools.git_tools.subprocess.run",
        side_effect=[
            SimpleNamespace(returncode=0, stdout="", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="rejected"),
        ],
    ):
        message = git_tools.git_commit("broken")

    assert message.startswith("Error committing")


def test_git_push_handles_errors_gracefully():
    with mock.patch(
        "aura.tools.git_tools.subprocess.run",
        return_value=SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    ):
        error = git_tools.git_push(remote="origin", branch="main")

    assert error.startswith("Error pushing")


def test_git_diff_returns_diff_output():
    with mock.patch(
        "aura.tools.git_tools.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="diff --git", stderr=""),
    ):
        diff = git_tools.git_diff(file_path="app.py")

    assert diff.startswith("diff --git")
