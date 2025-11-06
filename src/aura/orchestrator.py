"""Coordinates planning and execution of Aura sessions."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import QObject, QEventLoop, QThread, Signal

from src.aura import config
from src.aura.events import EventType, get_event_bus
from src.aura.execution import CliAgentExecutor, NativeAgentExecutor, SessionExecutor
from src.aura.services import PlanningService
from src.aura.services.chat_service import ChatService
from src.aura.services.planning_service import Session
from src.aura.tools import GitHelper
from src.aura.utils import scan_directory

LOGGER = logging.getLogger(__name__)


def _infer_project_name(goal: str) -> str:
    """Infer a project name from the user's goal.

    Args:
        goal: The user's request/goal

    Returns:
        A kebab-case project name (e.g., "blog-api", "password-gen")
    """
    # Extract key phrases that indicate project type
    goal_lower = goal.lower()

    # Look for common project patterns
    patterns = [
        r"(?:build|create|make)\s+(?:a|an)?\s*([a-z0-9\s-]+?)(?:\s+(?:app|application|api|system|service|tool|generator|website|platform))",
        r"([a-z0-9\s-]+?)\s+(?:app|application|api|system|service|tool|generator|website|platform)",
        r"(?:build|create|make)\s+(?:a|an)?\s*([a-z0-9\s-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, goal_lower)
        if match:
            name = match.group(1).strip()
            # Convert to kebab-case
            name = re.sub(r'\s+', '-', name)
            name = re.sub(r'[^a-z0-9-]', '', name)
            # Limit to 3-4 words max
            parts = name.split('-')[:4]
            name = '-'.join(parts)
            if name and len(name) > 2:
                return name

    # Fallback: use first few meaningful words
    words = re.findall(r'\b[a-z]{3,}\b', goal_lower)
    if words:
        # Take first 2-3 meaningful words
        project_words = words[:3]
        name = '-'.join(project_words)
        if len(name) > 2:
            return name

    # Ultimate fallback
    return "project"


def _ensure_unique_project_dir(base_dir: Path, project_name: str) -> Path:
    """Ensure the project directory doesn't already exist, append number if needed.

    Args:
        base_dir: The base workspace directory
        project_name: The inferred project name

    Returns:
        A unique project directory path
    """
    project_dir = base_dir / project_name
    if not project_dir.exists():
        return project_dir

    # If exists, try appending numbers
    counter = 2
    while counter < 100:
        candidate = base_dir / f"{project_name}-{counter}"
        if not candidate.exists():
            return candidate
        counter += 1

    # If we still can't find a unique name, use timestamp
    timestamp = int(time.time())
    return base_dir / f"{project_name}-{timestamp}"


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
    progress_update = Signal(str)

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

    def is_running(self) -> bool:
        """Check if orchestration is currently running."""
        return self._thread is not None and self._thread.isRunning()

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
        self._worker.progress_update.connect(self.progress_update)

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
    progress_update = Signal(str)

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
        # Infer project name and create project directory
        project_name = _infer_project_name(self._goal)
        project_dir = _ensure_unique_project_dir(self._working_dir, project_name)

        LOGGER.info("Inferred project name: %s", project_name)
        LOGGER.info("Creating project directory: %s", project_dir)

        # Create the project directory
        try:
            project_dir.mkdir(parents=True, exist_ok=False)
            self.session_output.emit(f"📁 Creating project: {project_dir.name}")
            self.session_output.emit(f"   └─ Location: {project_dir}")
            self.session_output.emit("")
        except FileExistsError:
            LOGGER.warning("Project directory already exists: %s", project_dir)
            self.session_output.emit(f"📁 Using existing project: {project_dir.name}")
            self.session_output.emit("")

        # Update working directory to the project directory
        self._working_dir = project_dir

        self.progress_update.emit("Generating session plan...")
        self.planning_started.emit()
        self._event_bus.publish(EventType.PLANNING_STARTED)
        self.session_output.emit("Analyzing request...")

        # NEW: Intelligent discovery phase using ChatService with tools
        LOGGER.info("Starting intelligent project discovery phase")
        self.session_output.emit("  ├─ Phase 1: Discovering project context...")
        project_context = self._discover_project_context(self._goal)

        LOGGER.info("Requesting session plan from planning service")
        self.session_output.emit("  └─ Phase 2: Generating session plan...")
        plan = self._planning_service.plan_sessions(self._goal, project_context)

        if not plan or not plan.sessions:
            raise ValueError("Planning produced no sessions.")

        self.progress_update.emit("Session plan ready")
        self.plan_ready.emit(plan)
        self._event_bus.publish(EventType.PLAN_READY, plan=plan)

        session_count = len(plan.sessions)
        estimated_minutes = getattr(plan, "total_estimated_minutes", 0)
        self.session_output.emit(f"  ├─ Generated {session_count} sessions")
        self.session_output.emit(f"  └─ Estimated {estimated_minutes} minutes")

        all_results: List[SessionResult] = []
        for index, session in enumerate(plan.sessions):
            self.progress_update.emit(f"Session {index + 1}/{session_count}: {session.name}")
            self.session_output.emit("")
            self.session_output.emit(f"Executing Session {index + 1}/{session_count}: {session.name}")
            self.session_started.emit(index, session)
            self._event_bus.publish(EventType.SESSION_STARTED, index=index, session=session)

            LOGGER.info("Executing session %d/%d: %s", index + 1, session_count, session.name)
            result = self._run_session(index, session)
            self.session_complete.emit(index, result)
            self._event_bus.publish(EventType.SESSION_COMPLETE, index=index, result=result)
            self._update_context(index, session, result)
            all_results.append(result)
            if config.AUTO_COMMIT_SESSIONS:
                if result.success and result.files_created:
                    commit_msg = f"Session {index + 1}: {session.name}"
                    self.progress_update.emit("Committing changes...")
                    self.session_output.emit("Committing changes...")
                    self.session_output.emit(f"  └─ {commit_msg}")
                    LOGGER.info("Committing changes: %s", commit_msg)
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

        self._install_dependencies()
        self.progress_update.emit("All sessions complete")
        self.all_sessions_complete.emit()
        self._event_bus.publish(EventType.ALL_COMPLETE)
        self.session_output.emit("")
        self.session_output.emit("All sessions complete")

        total_files = sum(len(result.files_created) for result in all_results)
        total_duration = sum(result.duration_seconds for result in all_results)
        self.session_output.emit(f"  ├─ Created {total_files} files")

        if config.AUTO_PUSH_ON_COMPLETE:
            self.progress_update.emit("Pushing to GitHub...")
            self.session_output.emit("Pushing to GitHub...")
            self._event_bus.publish(EventType.SESSION_OUTPUT, text="Pushing to GitHub...")
            LOGGER.info("Pushing changes to GitHub")
            if self._git.push():
                self.session_output.emit("  ├─ ✓ Pushed to GitHub")
                self._event_bus.publish(EventType.SESSION_OUTPUT, text="  ├─ ✓ Pushed to GitHub")
            else:
                self.session_output.emit("  ├─ ✗ Push failed")
                self._event_bus.publish(EventType.ERROR, error="Failed to push to GitHub")

        self.session_output.emit(f"  └─ Total time: {total_duration:.1f}s")
        LOGGER.info("Orchestration complete: %d sessions, %.1fs total", session_count, total_duration)

    def _install_dependencies(self) -> None:
        """Install dependencies from requirements.txt once sessions complete."""
        requirements_path = self._working_dir / "requirements.txt"
        if not requirements_path.exists():
            LOGGER.info("No requirements.txt detected; skipping dependency installation")
            return
        self.progress_update.emit("Installing dependencies...")
        self.session_output.emit(" Installing dependencies...")
        result = self._run_dependency_install()
        if result is None:
            self.session_output.emit(" ⚠️ Dependency installation failed to start. See logs for details.")
            return
        summary = self._summarize_pip_output(result.stdout, result.stderr, result.returncode)
        if summary:
            self.session_output.emit(f"  {summary}")
        if result.returncode == 0:
            self.session_output.emit(" ✅ Dependencies installed successfully")
        else:
            LOGGER.warning("Dependency installation exited with %d", result.returncode)
            self.session_output.emit(" ⚠️ Dependency installation encountered issues; please review the logs.")

    def _run_dependency_install(self) -> subprocess.CompletedProcess[str] | None:
        """Run pip install to apply dependency changes once."""
        try:
            return subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=self._working_dir,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to launch dependency installation: %s", exc)
            return None

    @staticmethod
    def _summarize_pip_output(stdout: str | None, stderr: str | None, returncode: int) -> str:
        """Return a concise summary line from pip output."""
        success_lines = [
            line.strip()
            for line in (stdout or "").splitlines()
            if "Successfully installed" in line
        ]
        if success_lines:
            return success_lines[-1]
        satisfied_lines = [
            line.strip()
            for line in (stdout or "").splitlines()
            if "Requirement already satisfied" in line
        ]
        if satisfied_lines:
            return satisfied_lines[0]
        error_lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
        if error_lines and returncode != 0:
            return error_lines[-1]
        return ""

    def _discover_project_context(self, goal: str) -> str:
        """Use ChatService with tools to discover project context intelligently.

        This method triggers the mandatory tool usage workflow defined in AURA_SYSTEM_PROMPT,
        causing ChatService to analyze the project using its 8 developer tools before planning.
        """
        self.session_output.emit("  └─ Discovering project context with AI tools...")
        LOGGER.info("Starting intelligent project discovery with ChatService")

        api_key = self._api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            LOGGER.warning("No API key available for discovery phase, falling back to basic context")
            return self._build_project_context()

        try:
            chat = ChatService(api_key=api_key)

            # Discovery-focused prompt that triggers mandatory tool usage
            discovery_prompt = (
                f"Analyze this project for the following task: {goal}\n\n"
                "Use your developer tools to understand:\n"
                "1. What files and directories exist (list_project_files)\n"
                "2. What relevant code patterns are already implemented (search_in_files)\n"
                "3. Function signatures in key files (get_function_definitions)\n"
                "4. Implementation details of relevant modules (read_project_file)\n\n"
                "Gather comprehensive context about the codebase that will help plan "
                "focused coding sessions. Be thorough but concise in your analysis."
            )

            # Get discovery response with automatic tool calling
            # The SDK will automatically execute all tool calls behind the scenes
            self.session_output.emit("    └─ Running discovery phase with automatic tool calling...")
            combined_discovery = chat.send_message(discovery_prompt)

            # Note: With automatic function calling, we don't get visibility into individual tool calls
            # The SDK handles the entire function calling loop internally
            LOGGER.info("Discovery phase completed with automatic function calling")

            # Build rich context combining basic info with AI discovery
            basic_context = self._build_project_context()

            return (
                f"{basic_context}\n\n"
                f"AI Discovery Analysis:\n"
                f"{combined_discovery}"
            )

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Discovery phase failed, falling back to basic context: %s", exc)
            self.session_output.emit(f"  └─ Discovery failed: {exc}, using basic context")
            return self._build_project_context()

    def _build_project_context(self) -> str:
        """Summarize the working directory for planning (basic fallback)."""
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
