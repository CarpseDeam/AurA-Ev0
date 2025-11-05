"""Helper for agent-related operations initiated from the main window."""

from __future__ import annotations

import logging
import os
from typing import Optional

from src.aura import config
from src.aura.orchestrator import Orchestrator
from src.aura.services.chat_service import get_session_context_manager
from src.aura.state import AppState
from src.aura.ui.output_panel import OutputPanel
from src.aura.ui.status_bar_manager import StatusBarManager
from src.aura.utils import find_cli_agents, scan_directory

LOGGER = logging.getLogger(__name__)


class AgentExecutionManager:
    """Coordinates agent environment setup, selection, and context gathering."""

    def __init__(
        self,
        app_state: AppState,
        output_panel: OutputPanel,
        status_manager: StatusBarManager,
        orchestrator: Optional[Orchestrator] = None,
    ) -> None:
        self._app_state = app_state
        self._output_panel = output_panel
        self._status_manager = status_manager
        self._orchestrator = orchestrator

    def validate_environment(self) -> bool:
        """Ensure prerequisites are met before starting the agent."""
        if not os.path.isdir(self._app_state.working_directory):
            self._output_panel.display_output("Working directory does not exist.", "#FF6B6B")
            self._status_manager.update_status("Error", "#FF6B6B", persist=True)
            return False

        if not self._app_state.agent_path:
            agent_display = config.AGENT_DISPLAY_NAMES.get(
                self._app_state.selected_agent,
                self._app_state.selected_agent,
            )
            self._output_panel.display_output(
                f"{agent_display} not found. Use 'Agent Settings...' to configure.",
                "#FF6B6B",
            )
            self._status_manager.update_status("Error", "#FF6B6B", persist=True)
            return False

        return True

    def detect_default_agent(self) -> None:
        """Identify an available agent and update state accordingly."""
        agents = find_cli_agents()
        for agent in agents:
            if agent.is_available and agent.name == self._app_state.selected_agent:
                self._app_state.set_agent_path(agent.executable_path)
                self._output_panel.display_output(
                    f"Using {agent.display_name} at {agent.executable_path}",
                    config.COLORS.success,
                )
                if self._orchestrator:
                    self._orchestrator.update_agent_path(agent.executable_path)
                return

        for agent in agents:
            if agent.is_available:
                self._app_state.set_selected_agent(agent.name)
                self._app_state.set_agent_path(agent.executable_path)
                self._output_panel.display_output(
                    f"Using {agent.display_name} at {agent.executable_path}",
                    config.COLORS.success,
                )
                if self._orchestrator:
                    self._orchestrator.update_agent_path(agent.executable_path)
                return

        self._output_panel.display_output(
            "No CLI agents found. Use 'Agent Settings...' to configure.",
            "#FFB74D",
        )

    def set_working_directory(self, path: str) -> None:
        """Update the working directory in AppState and clear session context."""
        self._app_state.set_working_directory(path)
        get_session_context_manager().clear()
        LOGGER.info("Cleared session context due to working directory change: %s", path)
        self._output_panel.display_output(f"Working directory set to {path}", config.COLORS.accent)

    def compose_prompt(self, prompt: str) -> str:
        """Combine workspace context with the user's prompt."""
        return f"{self.describe_workspace()}\n\nTask: {prompt}"

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
