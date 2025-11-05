"""File filtering for safe agent operations."""

from __future__ import annotations

from pathlib import Path
from typing import List

# Patterns that should NEVER be modified by agents
PROTECTED_PATTERNS = [
    "src/aura/**/*.py",  # Aura's source code
    "tests/**/*.py",  # Test files
    ".git/**",  # Git internals
    "*.pyc",  # Python cache
    "__pycache__/**",  # Python cache dirs
    ".venv/**",  # Virtual environment
    "venv/**",  # Virtual environment
]


def is_file_protected(file_path: str, base_dir: str) -> bool:
    """Check if file should be protected from agent modifications."""
    try:
        relative = Path(file_path).relative_to(base_dir)
    except ValueError:
        # File outside base directory
        return False

    # Use PurePath.match() for proper ** glob support
    for pattern in PROTECTED_PATTERNS:
        if relative.match(pattern):
            return True

    return False


def load_gitignore_patterns(base_dir: str) -> List[str]:
    """Load patterns from .gitignore if it exists."""
    gitignore_path = Path(base_dir) / ".gitignore"
    if not gitignore_path.exists():
        return []

    patterns = []
    with open(gitignore_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)

    return patterns
