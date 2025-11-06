from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from aura.tools import python_tools


def test_run_tests_parses_results():
    with mock.patch(
        "aura.tools.python_tools.subprocess.run",
        return_value=SimpleNamespace(
            returncode=0,
            stdout="collected 2 items\n2 passed in 0.34s\n",
            stderr="",
        ),
    ):
        result = python_tools.run_tests("tests/")

    assert result["passed"] == 2
    assert result["duration"] == 0.34


def test_lint_code_parses_errors_warnings_and_score():
    mock_output = (
        "module.py:10: error: badness\n"
        "module.py:20: warning: caution\n"
        "Your code has been rated at 7.50/10\n"
    )
    with mock.patch(
        "aura.tools.python_tools.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout=mock_output, stderr=""),
    ):
        result = python_tools.lint_code(file_paths=["module.py"])

    assert "error" in result["errors"][0].lower()
    assert "warning" in result["warnings"][0].lower()
    assert result["score"] == 7.50


def test_format_code_counts_reformatted_files():
    with mock.patch(
        "aura.tools.python_tools.subprocess.run",
        return_value=SimpleNamespace(
            returncode=0,
            stdout="1 file reformatted, 1 file left unchanged\n",
            stderr="",
        ),
    ):
        result = python_tools.format_code(file_paths=["module.py"])

    assert result["formatted"] == 1
    assert result["errors"] == []


def test_install_package_validates_spec():
    with mock.patch(
        "aura.tools.python_tools.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout="done", stderr=""),
    ) as run_mock:
        result = python_tools.install_package("requests", ">=2.0")

    assert "Successfully installed requests>=2.0" in result
    assert "requests>=2.0" in run_mock.call_args[0][0]


def test_get_function_definitions_parses_file(sample_python_file):
    functions = python_tools.get_function_definitions(str(sample_python_file))

    assert any(func["name"] == "helper" for func in functions)


def test_get_function_definitions_handles_syntax_error(tmp_path):
    bad_file = tmp_path / "broken.py"
    bad_file.write_text("def bad(:\n    pass\n", encoding="utf-8")

    assert python_tools.get_function_definitions(str(bad_file)) == []
