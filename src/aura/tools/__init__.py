"""Tool utilities for Aura."""

from __future__ import annotations

from .git_helper import GitHelper
from .git_tools import (
    create_new_branch,
    git_blame,
    git_commit,
    git_diff,
    git_push,
    get_git_status,
)
from .project_tools import (
    get_database_schema,
    get_project_dependencies,
    update_documentation,
)
from .python_tools import (
    format_code,
    generate_test_file,
    get_cyclomatic_complexity,
    get_function_definitions,
    install_package,
    lint_code,
    run_tests,
)
from .symbol_tools import (
    find_definition,
    find_usages,
    get_class_hierarchy,
    get_dependency_graph,
    get_imports,
    safe_rename_symbol,
)
from .tool_manager import ToolManager

__all__ = [
    "ToolManager",
    # Git tools
    "get_git_status",
    "git_commit",
    "git_push",
    "git_diff",
    "git_blame",
    "create_new_branch",
    "GitHelper",
    # Python tools
    "run_tests",
    "lint_code",
    "install_package",
    "format_code",
    "get_function_definitions",
    "get_cyclomatic_complexity",
    "generate_test_file",
    # Symbol tools
    "find_definition",
    "find_usages",
    "get_imports",
    "get_dependency_graph",
    "get_class_hierarchy",
    "safe_rename_symbol",
    # Project tools
    "get_project_dependencies",
    "update_documentation",
    "get_database_schema",
]
