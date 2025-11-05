"""Tool functions for Aura AI agent.

This package organizes tool functions by category:
- file_system_tools: File reading, listing, and searching
- git_tools: Version control operations
- python_tools: Testing, linting, formatting, and code analysis
"""

from __future__ import annotations

from src.aura.tools.file_system_tools import (
    list_project_files,
    read_multiple_files,
    read_project_file,
    search_in_files,
)
from src.aura.tools.git_tools import git_commit, git_diff, git_push, get_git_status
from src.aura.tools.python_tools import (
    format_code,
    get_function_definitions,
    install_package,
    lint_code,
    run_tests,
)

__all__ = [
    # File system tools
    "read_project_file",
    "list_project_files",
    "search_in_files",
    "read_multiple_files",
    # Git tools
    "get_git_status",
    "git_commit",
    "git_push",
    "git_diff",
    # Python tools
    "run_tests",
    "lint_code",
    "install_package",
    "format_code",
    "get_function_definitions",
]
