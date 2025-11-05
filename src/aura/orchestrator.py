"""Coordinates planning and execution of Aura sessions."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import QObject, QEventLoop, QThread, Signal

from aura import config
from aura.events import EventType, get_event_bus
from aura.execution import CliAgentExecutor, NativeAgentExecutor, SessionExecutor
from aura.services import PlanningService
from aura.services.planning_service import Session
from aura.tools import GitHelper
from aura.utils import scan_directory

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionResult:
    """Outcome details for a single session execution."""

    session_name: str
    exit_code: int
    duration_seconds: float
    files_created: List[str]
    success: bool


class Orchestrator(QObject):
    """Runs planned sessions sequentially with context passing."""

    planning_started = Signal()
    plan_ready = Signal(object)
    session_started = Signal(int, object)
    session_output = Signal(str)
    session_complete = Signal(int, object)
    all_sessions_complete = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        planning_service: PlanningService,
        working_dir: str,
        agent_path: str,
        api_key: str | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Store dependencies and validate the working directory."""
        super().__init__(parent)
        if not planning_service:
            raise ValueError("Planning service is required.")
        resolved = Path(working_dir).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Working directory does not exist: {resolved}")
        self._planning_service = planning_service
        self._working_dir = resolved
        self._agent_path = agent_path
        self._api_key = api_key
        self._thread: QThread | None = None
        self._worker: _ExecutionWorker | None = None
        self._event_bus = get_event_bus()

    def execute_goal(self, goal: str) -> None:
        """Plan and execute the provided goal on a background thread."""
        if not goal or not goal.strip():
            self.error_occurred.emit("Goal must be provided.")
            return
        if self._thread is not None and self._thread.isRunning():
            self.error_occurred.emit("An orchestration run is already in progress.")
            return
        self._thread = QThread(self)
        self._worker = _ExecutionWorker(
            self._planning_service,
            self._working_dir,
            self._agent_path,
            goal.strip(),
            self._api_key,
        )
        self._move_worker_to_thread()
        self._thread.start()

    def update_agent_path(self, agent_path: str) -> None:
        """Update the agent executable path for subsequent runs."""
        if not agent_path:
            raise ValueError("Agent path must be provided.")
        self._agent_path = agent_path

    def _move_worker_to_thread(self) -> None:
        """Wire worker signals and move it to the execution thread."""
        assert self._worker is not None and self._thread is not None
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._cleanup_worker)
        self._thread.finished.connect(self._thread.deleteLater)
        self._worker.planning_started.connect(self.planning_started)
        self._worker.plan_ready.connect(self.plan_ready)
        self._worker.session_started.connect(self.session_started)
        self._worker.session_output.connect(self.session_output)
        self._worker.session_complete.connect(self.session_complete)
        self._worker.all_sessions_complete.connect(self.all_sessions_complete)
        self._worker.error_occurred.connect(self.error_occurred)

    def _cleanup_worker(self) -> None:
        """Release worker resources after execution."""
        if self._worker is not None:
            self._worker.deleteLater()
        self._worker = None
        self._thread = None


class _ExecutionWorker(QObject):
    """Performs planning and session execution off the UI thread."""

    finished = Signal()
    planning_started = Signal()
    plan_ready = Signal(object)
    session_started = Signal(int, object)
    session_output = Signal(str)
    session_complete = Signal(int, object)
    all_sessions_complete = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        planning_service: PlanningService,
        working_dir: Path,
        agent_path: str,
        goal: str,
        api_key: str | None = None,
    ) -> None:
        """Initialize execution state."""
        super().__init__()
        self._planning_service = planning_service
        self._working_dir = working_dir
        self._agent_path = agent_path
        self._goal = goal
        self._api_key = api_key
        self._context_notes: List[str] = []
        self._event_bus = get_event_bus()
        self._git = GitHelper(str(working_dir))

    def run(self) -> None:
        """Entry point when the worker thread starts."""
        try:
            self._execute()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Orchestration failed: %s", exc)
            message = str(exc)
            self._event_bus.publish(EventType.ERROR, error=message)
            self.error_occurred.emit(message)
        finally:
            self.finished.emit()

    def _execute(self) -> None:
        """Perform planning then execute all sessions sequentially."""
        self.planning_started.emit()
        self._event_bus.publish(EventType.PLANNING_STARTED)
        self.session_output.emit("Analyzing request...")
        project_context = self._build_project_context()
        plan = self._planning_service.plan_sessions(self._goal, project_context)
        if not plan or not plan.sessions:
            raise ValueError("Planning produced no sessions.")
        self.plan_ready.emit(plan)
        self._event_bus.publish(EventType.PLAN_READY, plan=plan)
        session_count = len(plan.sessions)
        estimated_minutes = getattr(plan, "total_estimated_minutes", 0)
        self.session_output.emit(f"  ├─ Generated {session_count} sessions")
        self.session_output.emit(f"  └─ Estimated {estimated_minutes} minutes")
        all_results: List[SessionResult] = []
        for index, session in enumerate(plan.sessions):
            self.session_output.emit("")
            self.session_output.emit(f"Executing Session {index + 1}/{session_count}: {session.name}")
            self.session_started.emit(index, session)
            self._event_bus.publish(EventType.SESSION_STARTED, index=index, session=session)
            result = self._run_session(index, session)
            self.session_complete.emit(index, result)
            self._event_bus.publish(EventType.SESSION_COMPLETE, index=index, result=result)
            self._update_context(index, session, result)
            all_results.append(result)
            if config.AUTO_COMMIT_SESSIONS:
                if result.success and result.files_created:
                    commit_msg = f"Session {index + 1}: {session.name}"
                    self.session_output.emit("Committing changes...")
                    self.session_output.emit(f"  └─ {commit_msg}")
                    if self._git.commit(commit_msg, result.files_created):
                        self._event_bus.publish(
                            EventType.SESSION_OUTPUT,
                            text=f"  └─ {commit_msg}",
                        )
                    else:
                        self._event_bus.publish(
                            EventType.ERROR,
                            error=f"Failed to commit changes for {commit_msg}",
                        )
            if result.success:
                self.session_output.emit(f"  └─ ✓ Complete in {result.duration_seconds:.1f}s")
            else:
                self.session_output.emit("  └─ ✗ Failed")
            if not result.success:
                error_message = f"Session '{session.name}' failed with exit code {result.exit_code}."
                self._event_bus.publish(EventType.ERROR, error=error_message)
                self.error_occurred.emit(error_message)
                return
        self.all_sessions_complete.emit()
        self._event_bus.publish(EventType.ALL_COMPLETE)
        self.session_output.emit("")
        self.session_output.emit("All sessions complete")
        total_files = sum(len(result.files_created) for result in all_results)
        total_duration = sum(result.duration_seconds for result in all_results)
        self.session_output.emit(f"  ├─ Created {total_files} files")
        if config.AUTO_PUSH_ON_COMPLETE:
            self.session_output.emit("Pushing to GitHub...")
            self._event_bus.publish(EventType.SESSION_OUTPUT, text="Pushing to GitHub...")
            if self._git.push():
                self.session_output.emit("  ├─ ✓ Pushed to GitHub")
                self._event_bus.publish(EventType.SESSION_OUTPUT, text="  ├─ ✓ Pushed to GitHub")
            else:
                self.session_output.emit("  ├─ ✗ Push failed")
                self._event_bus.publish(EventType.ERROR, error="Failed to push to GitHub")
        self.session_output.emit(f"  └─ Total time: {total_duration:.1f}s")

    def _build_project_context(self) -> str:
        """Summarize the working directory for planning."""
        snapshot = scan_directory(str(self._working_dir), max_depth=2)
        directories = "\n".join(f"- {entry}" for entry in snapshot["directories"]) or "- None"
        python_files = [
            entry for entry in snapshot["files"] if entry.endswith(".py")
        ]
        files = "\n".join(f"- {entry}" for entry in python_files) or "- None"
        return (
            f"Working directory: {self._working_dir}\n"
            f"Directories:\n{directories}\n"
            f"Python files:\n{files}"
        )

    def _run_session(self, index: int, session: Session) -> SessionResult:
        """Execute a single session using appropriate executor strategy."""
        # Select executor strategy based on configuration
        executor = self._select_executor()

        # Prepare execution context
        context = {
            "working_dir": self._working_dir,
            "context_notes": self._context_notes,
        }

        # Try using native agent if enabled and API key available
        if config.USE_NATIVE_PYTHON_AGENT and self._api_key:
            try:
                result = executor.execute(session, context)
                LOGGER.info("Session '%s' executed using native PythonCoderAgent", session.name)
                return result
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Native agent failed, falling back to CLI: %s", exc)
                self.session_output.emit(f"⚠️ Native agent failed, using CLI fallback: {exc}")
                # Create CLI executor for fallback
                executor = CliAgentExecutor(
                    self._agent_path,
                    output_signal=self.session_output,
                    error_signal=self.error_occurred,
                    parent=self,
                )

        # Execute using selected executor
        if not config.USE_NATIVE_PYTHON_AGENT:
            LOGGER.info("Native agent disabled via config, using CLI agent")
        return executor.execute(session, context)

    def _select_executor(self) -> SessionExecutor:
        """Select the appropriate session executor based on configuration.

        Returns:
            SessionExecutor instance (Native or CLI)
        """
        if config.USE_NATIVE_PYTHON_AGENT and self._api_key:
            return NativeAgentExecutor(
                api_key=self._api_key,
                output_signal=self.session_output,
            )
        return CliAgentExecutor(
            self._agent_path,
            output_signal=self.session_output,
            error_signal=self.error_occurred,
            parent=self,
        )

    def _update_context(self, index: int, session: Session, result: SessionResult) -> None:
        """Record the session outcome to inform subsequent work."""
        if result.files_created:
            files = ", ".join(result.files_created)
            summary = f"Session {index + 1} ({session.name}) created: {files}"
        else:
            summary = f"Session {index + 1} ({session.name}) completed with no new files."
        self._context_notes.append(summary)

    def _format_context(self) -> str:
        """Render accumulated context notes."""
        return "\n".join(self._context_notes)
