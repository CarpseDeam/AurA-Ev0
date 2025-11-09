"""Real-time CLI-style task list display system for Aura."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a single task in the task list."""

    task_id: str
    text: str
    status: TaskStatus = TaskStatus.PENDING
    indent_level: int = 0
    parent_id: Optional[str] = None
    tool_name: Optional[str] = None

    def __hash__(self) -> int:
        """Make Task hashable for set operations."""
        return hash(self.task_id)


@dataclass
class TaskGroup:
    """Represents a group of related tasks with a header."""

    group_id: str
    header: str
    icon: str = "❄"
    tasks: list[Task] = field(default_factory=list)

    def __hash__(self) -> int:
        """Make TaskGroup hashable for set operations."""
        return hash(self.group_id)


class TaskListDisplay:
    """Parses agent responses and manages task list state."""

    # Task detection patterns
    ACTION_VERB_PATTERN = re.compile(
        r"^(Creating|Create|Updating|Update|Reading|Read|Writing|Write|"
        r"Deleting|Delete|Building|Build|Installing|Install|Running|Run|"
        r"Testing|Test|Fixing|Fix|Adding|Add|Removing|Remove|Modifying|Modify|"
        r"Enhancing|Enhance|Implementing|Implement|Refactoring|Refactor|"
        r"Analyzing|Analyze|Planning|Plan|Configuring|Configure)\s+(.+)",
        re.IGNORECASE,
    )

    NUMBERED_LIST_PATTERN = re.compile(r"^\s*(\d+)[.)]\s+(.+)")
    BULLET_LIST_PATTERN = re.compile(r"^\s*[-*•]\s+(.+)")

    # Tool name to task verb mapping
    TOOL_TO_VERB = {
        "read_file": "Read",
        "read_project_file": "Read",
        "read_multiple_files": "Read",
        "create_file": "Create",
        "modify_file": "Modify",
        "replace_file_lines": "Modify",
        "delete_file": "Delete",
        "list_project_files": "List",
    }

    def __init__(self) -> None:
        """Initialize the task list display."""
        self._groups: dict[str, TaskGroup] = {}
        self._tasks_by_id: dict[str, Task] = {}
        self._tasks_by_tool: dict[str, str] = {}  # tool_name -> task_id
        self._current_group_id: Optional[str] = None

    def parse_text_for_tasks(self, text: str, source: Optional[str] = None) -> list[TaskGroup]:
        """Parse text and extract task structures.

        Args:
            text: The text to parse
            source: The source of the text (e.g., "analyst", "executor")

        Returns:
            List of newly created task groups
        """
        if not text:
            return []

        new_groups: list[TaskGroup] = []

        # Check for section headers (e.g., "❄ Planning scene system...")
        header_match = re.match(r"^([❄⚡🔥💫✨🌟⭐]+)\s*(.+?)(\.\.\.)?$", text.strip())
        if header_match:
            icon = header_match.group(1)
            header = header_match.group(2).strip()
            group = self._create_task_group(header, icon)
            new_groups.append(group)
            return new_groups

        # Check for action verb patterns
        action_match = self.ACTION_VERB_PATTERN.match(text.strip())
        if action_match:
            verb = action_match.group(1)
            target = action_match.group(2).strip()
            task_text = f"{verb.capitalize()} {target}"
            self._add_task_to_current_group(task_text)
            return []

        # Check for numbered lists
        lines = text.strip().split("\n")
        if len(lines) >= 2 and all(self.NUMBERED_LIST_PATTERN.match(line.strip()) for line in lines if line.strip()):
            self._parse_numbered_list(lines)
            return []

        # Check for bullet lists
        if len(lines) >= 2 and sum(1 for line in lines if self.BULLET_LIST_PATTERN.match(line.strip())) >= len(lines) // 2:
            self._parse_bullet_list(lines)
            return []

        return new_groups

    def create_task_from_tool_call(self, tool_name: str, parameters: dict) -> Optional[str]:
        """Create a task from a tool call event.

        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters

        Returns:
            Task ID if a task was created, None otherwise
        """
        verb = self.TOOL_TO_VERB.get(tool_name)
        if not verb:
            # Generic tool handling
            verb = "Run"

        # Extract target from parameters
        target = self._extract_target_from_params(tool_name, parameters)
        task_text = f"{verb} {target}" if target else verb

        task = self._add_task_to_current_group(task_text, tool_name=tool_name)
        if task:
            # Map tool name to task ID for later updates
            self._tasks_by_tool[tool_name] = task.task_id
            return task.task_id
        return None

    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update the status of a specific task.

        Args:
            task_id: The task ID to update
            status: The new status

        Returns:
            True if the task was found and updated, False otherwise
        """
        task = self._tasks_by_id.get(task_id)
        if task:
            task.status = status
            return True
        return False

    def update_task_status_by_tool(self, tool_name: str, status: TaskStatus) -> bool:
        """Update task status by tool name.

        Args:
            tool_name: The tool name
            status: The new status

        Returns:
            True if the task was found and updated, False otherwise
        """
        task_id = self._tasks_by_tool.get(tool_name)
        if task_id:
            return self.update_task_status(task_id, status)
        return False

    def get_all_groups(self) -> list[TaskGroup]:
        """Get all task groups in order.

        Returns:
            List of all task groups
        """
        return list(self._groups.values())

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID.

        Args:
            task_id: The task ID

        Returns:
            The task if found, None otherwise
        """
        return self._tasks_by_id.get(task_id)

    def clear(self) -> None:
        """Clear all tasks and groups."""
        self._groups.clear()
        self._tasks_by_id.clear()
        self._tasks_by_tool.clear()
        self._current_group_id = None

    def _create_task_group(self, header: str, icon: str = "❄") -> TaskGroup:
        """Create a new task group.

        Args:
            header: The group header text
            icon: The icon for the group

        Returns:
            The created task group
        """
        group_id = str(uuid.uuid4())
        group = TaskGroup(group_id=group_id, header=header, icon=icon)
        self._groups[group_id] = group
        self._current_group_id = group_id
        return group

    def _add_task_to_current_group(
        self,
        task_text: str,
        indent_level: int = 0,
        tool_name: Optional[str] = None,
    ) -> Optional[Task]:
        """Add a task to the current group.

        Args:
            task_text: The task text
            indent_level: Indentation level for hierarchy
            tool_name: Associated tool name if any

        Returns:
            The created task if successful, None otherwise
        """
        if not self._current_group_id:
            # Create a default group if none exists
            self._create_task_group("Tasks", "⚙️")

        if not self._current_group_id:
            return None

        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            text=task_text,
            indent_level=indent_level,
            tool_name=tool_name,
        )

        group = self._groups[self._current_group_id]
        group.tasks.append(task)
        self._tasks_by_id[task_id] = task

        return task

    def _parse_numbered_list(self, lines: list[str]) -> None:
        """Parse a numbered list into tasks.

        Args:
            lines: Lines containing the numbered list
        """
        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = self.NUMBERED_LIST_PATTERN.match(line)
            if match:
                task_text = match.group(2).strip()
                self._add_task_to_current_group(task_text)

    def _parse_bullet_list(self, lines: list[str]) -> None:
        """Parse a bullet list into tasks with hierarchy.

        Args:
            lines: Lines containing the bullet list
        """
        for line in lines:
            if not line.strip():
                continue

            # Count leading whitespace to determine indent level
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            indent_level = indent // 2  # Assume 2 spaces per indent

            match = self.BULLET_LIST_PATTERN.match(stripped)
            if match:
                task_text = match.group(1).strip()
                self._add_task_to_current_group(task_text, indent_level=indent_level)

    def _extract_target_from_params(self, tool_name: str, parameters: dict) -> str:
        """Extract a target description from tool parameters.

        Args:
            tool_name: The tool name
            parameters: Tool parameters

        Returns:
            Target description string
        """
        if not isinstance(parameters, dict):
            return tool_name

        # Try common parameter names
        for key in ("path", "file_path", "file", "target", "directory", "filepath"):
            if key in parameters:
                value = parameters[key]
                if value:
                    # Shorten long paths
                    value_str = str(value)
                    if len(value_str) > 50:
                        return f"...{value_str[-47:]}"
                    return value_str

        return tool_name
