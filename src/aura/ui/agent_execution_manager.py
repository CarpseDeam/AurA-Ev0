"""Helper for agent-related operations initiated from the main window."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

from pathlib import Path

from aura import config
from aura.exceptions import AuraConfigurationError, AuraValidationError
from aura.state import AppState
from aura.ui.output_panel import OutputPanel
from aura.ui.status_bar_manager import StatusBarManager
from aura.utils import find_cli_agents, scan_directory

if TYPE_CHECKING:
    from aura.orchestrator import Orchestrator

LOGGER = logging.getLogger(__name__)


class AgentExecutionManager:
    """Coordinates agent environment setup, selection, and context gathering."""

    def __init__(
        self,
        app_state: AppState,
        output_panel: OutputPanel,
        status_manager: StatusBarManager,
        orchestrator: Optional["Orchestrator"] = None,
    ) -> None:
        self._app_state = app_state
        self._output_panel = output_panel
        self._status_manager = status_manager
        self._orchestrator = orchestrator

    def validate_environment(self) -> None:
        """Ensure prerequisites are met before starting the agent."""
        working_dir = self._app_state.working_directory
        if not working_dir or not os.path.isdir(working_dir):
            raise AuraConfigurationError(
                "Workspace is unavailable. Please choose a valid working directory.",
                context={"issue": "working_directory_missing", "path": working_dir},
            )

        agent_path = self._app_state.agent_path
        if not agent_path:
            agent_display = config.AGENT_DISPLAY_NAMES.get(
                self._app_state.selected_agent,
                self._app_state.selected_agent,
            )
            raise AuraConfigurationError(
                f"{agent_display} is not configured. Open Agent Settings to select an agent.",
                context={"issue": "agent_missing", "agent": agent_display},
            )

        executable = Path(agent_path)
        if not executable.exists():
            raise AuraConfigurationError(
                "The configured agent executable was not found. Please update Agent Settings.",
                context={"issue": "agent_missing", "path": agent_path},
            )

        if not os.access(executable, os.X_OK):
            raise AuraConfigurationError(
                "Aura cannot run the configured agent. Ensure the executable has run permissions.",
                context={"issue": "agent_not_executable", "path": agent_path},
            )

    def detect_default_agent(self) -> None:
        """Identify an available agent and update state accordingly."""
        agents = find_cli_agents()
        for agent in agents:
            if agent.is_available and agent.name == self._app_state.selected_agent:
                self._app_state.set_agent_path(agent.executable_path)
                if self._orchestrator:
                    self._orchestrator.update_agent_path(agent.executable_path)
                return

        for agent in agents:
            if agent.is_available:
                self._app_state.set_selected_agent(agent.name)
                self._app_state.set_agent_path(agent.executable_path)
                if self._orchestrator:
                    self._orchestrator.update_agent_path(agent.executable_path)
                return

        self._output_panel.display_output(
            "No CLI agents found. Use 'Agent Settings...' to configure.",
            "#FFB74D",
        )

    def set_working_directory(self, path: str) -> str:
        """Update the working directory in AppState and synchronize dependents."""
        try:
            resolved = str(Path(path).expanduser().resolve())
        except Exception as exc:  # noqa: BLE001
            raise AuraConfigurationError(
                "Please select a valid working directory that exists on disk.",
                context={"issue": "working_directory_invalid", "path": path},
            ) from exc
        try:
            self._app_state.set_working_directory(resolved)
        except (ValueError, FileNotFoundError) as exc:
            raise AuraConfigurationError(
                "Please select a valid working directory that exists on disk.",
                context={"issue": "working_directory_invalid", "path": resolved},
            ) from exc

        workspace = self._app_state.working_directory
        LOGGER.info("Working directory changed: %s", workspace)

        if self._orchestrator:
            try:
                self._orchestrator.update_working_directory(workspace)
            except AuraConfigurationError as exc:
                LOGGER.error("Failed to sync orchestrator workspace: %s", exc)
                raise

        self._output_panel.display_output(
            f"Working directory set to {workspace}",
            config.COLORS.accent,
        )
        return workspace

    def compose_prompt(self, prompt: str) -> str:
        """Combine workspace context with the user's prompt."""
        normalized = (prompt or "").strip()
        if not normalized:
            raise AuraValidationError(
                "Please enter a request before running the agent.",
                context={"issue": "empty_prompt"},
            )
        return f"{self.describe_workspace()}\n\nTask: {normalized}"

    def describe_workspace(self) -> str:
        """Return a concise description of the workspace."""
        try:
            snapshot = scan_directory(self._app_state.working_directory, max_depth=2)
        except (ValueError, FileNotFoundError) as exc:
            return f"Working in: {self._app_state.working_directory}\nFiles: unavailable ({exc})"

        python_files = [item for item in snapshot["files"] if item.endswith(".py")]
        directory_lines = "\n".join(f"- {item}" for item in snapshot["directories"]) or "- None"
        file_lines = "\n".join(f"- {item}" for item in python_files) or "- None"
        return (
            f"Working in: {self._app_state.working_directory}\n"
            f"Directories:\n{directory_lines}\n"
            f"Python files:\n{file_lines}"
        )
