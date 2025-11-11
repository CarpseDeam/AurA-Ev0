
import time
import logging
from PySide6.QtCore import QObject, Signal

from aura.event_bus import get_event_bus
from aura.events import (
    PhaseTransition,
    ToolCallStarted,
    ToolCallCompleted,
    StatusUpdate,
    ExecutionComplete,
)

LOGGER = logging.getLogger(__name__)

class CliHeartbeatDisplay(QObject):
    """
    A QObject that subscribes to the event bus and translates events into
    CLI-style output for display in the UI.
    """
    new_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._event_bus = get_event_bus()
        self._phase_start_time = None
        self._current_phase = None
        self._subscribe_to_events()
        LOGGER.info("CLI Heartbeat Display initialized.")

    def _subscribe_to_events(self):
        self._event_bus.subscribe(PhaseTransition, self._on_phase_transition)
        self._event_bus.subscribe(ToolCallStarted, self._on_tool_call_started)
        self._event_bus.subscribe(StatusUpdate, self._on_status_update)
        self._event_bus.subscribe(ExecutionComplete, self._on_execution_complete)
        LOGGER.info("Subscribed to events.")

    def _on_phase_transition(self, event: PhaseTransition):
        if event.to_phase == "analyst":
            self._current_phase = "analyst"
            self._phase_start_time = time.perf_counter()
            self.new_message.emit("⚡ Analyst: Investigating codebase...")
        elif event.to_phase == "executor":
            if self._current_phase == "analyst" and self._phase_start_time:
                duration = time.perf_counter() - self._phase_start_time
                self.new_message.emit(f"✓ Analyst complete ({duration:.1f}s)")
            self._current_phase = "executor"
            self._phase_start_time = time.perf_counter()
            self.new_message.emit("⚙️ Executor: Applying changes...")

    def _on_tool_call_started(self, event: ToolCallStarted):
        if event.source == self._current_phase:
            action = self._format_tool_name(event.tool_name, event.parameters.get('kwargs', {}))
            self.new_message.emit(f"  ├─ {action}")

    def _on_status_update(self, event: StatusUpdate):
        if event.phase == "analyst.plan_ready":
            self.new_message.emit(f"📋 {event.message}")

    def _on_execution_complete(self, event: ExecutionComplete):
        if self._phase_start_time:
            duration = time.perf_counter() - self._phase_start_time
            if event.success:
                self.new_message.emit(f"✓ {self._current_phase.capitalize()} complete ({duration:.1f}s)")
            else:
                self.new_message.emit(f"❌ {self._current_phase.capitalize()} failed ({duration:.1f}s)")
        self._current_phase = None
        self._phase_start_time = None


    def _format_tool_name(self, tool_name: str, params: dict) -> str:
        """Translate technical tool names into friendly action descriptions."""
        
        file_path = params.get("file_path") or params.get("path")
        if file_path and isinstance(file_path, str):
            # show only the filename
            file_path = file_path.split('/')[-1].split('\\')[-1]

        if tool_name == "read_project_file":
            return f"Reading {file_path}"
        if tool_name == "list_project_files":
            return "Scanning files"
        if tool_name == "create_file":
            return f"Creating {file_path}"
        if tool_name == "add_godot_node":
            node_type = params.get("node_type", "Node")
            return f"Adding {node_type} to scene"
        if tool_name == "submit_execution_plan":
            return "Generating execution plan..."
        if tool_name == "search_in_files":
            pattern = params.get('pattern', '...')
            return f"Searching for '{pattern}'"
        if tool_name == "get_project_structure":
            return "Analyzing project structure"
        
        # Default fallback
        return tool_name.replace("_", " ").capitalize()
