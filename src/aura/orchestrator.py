"""Coordinates planning and execution of Aura sessions."""

from __future__ import annotations

import hashlib
import inspect
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


def _log_emit_trace(message: str, prefix: str = "EMIT_TRACE") -> None:
    """Log message emission with unique ID and location for debugging duplicates.

    Args:
        message: The message being emitted
        prefix: Log prefix (default: EMIT_TRACE)
    """
    # Create unique ID from message content
    msg_id = hashlib.md5(message.encode()).hexdigest()[:8]

    # Get caller's frame information
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        location = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
        func_name = caller_frame.f_code.co_name
    else:
        location = "unknown"
        func_name = "unknown"

    # Truncate message for logging
    msg_preview = message[:50].replace('\n', '\\n')

    LOGGER.info(f"{prefix} [ID:{msg_id}] [{func_name}@{location}]: {msg_preview}")


PROJECT_INDICATOR_DIRS = {"app", "src", "tests"}
PROJECT_INDICATOR_FILES = {
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "manage.py",
    "pipfile",
    "pipfile.lock",
}
PROJECT_IGNORE_DIRS = {".git", ".idea", ".venv", "__pycache__", ".pytest_cache"}


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

        # Log worker thread creation for debugging
        import threading
        LOGGER.info("THREAD_CREATE: Creating new worker thread (current thread: %s)", threading.current_thread().name)

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

        # Log thread start
        LOGGER.info("THREAD_START: Worker thread started (thread ID: %s)", id(self._thread))

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
        # Resolve which project directory to use (existing vs new)
        project_dir = self._resolve_project_directory()

        # Update working directory to the project directory and refresh git helper
        self._working_dir = project_dir
        self._git = GitHelper(str(self._working_dir))

        self.progress_update.emit("Generating session plan...")
        self.planning_started.emit()
        _log_emit_trace("Analyzing request...")
        self.session_output.emit("Analyzing request...")

        # NEW: Intelligent discovery phase using ChatService with tools
        LOGGER.info("Starting intelligent project discovery phase")
        _log_emit_trace("  ├─ Phase 1: Discovering project context...")
        self.session_output.emit("  ├─ Phase 1: Discovering project context...")
        project_context = self._discover_project_context(self._goal)

        LOGGER.info("Requesting session plan from planning service")
        _log_emit_trace("  └─ Phase 2: Generating session plan...")
        self.session_output.emit("  └─ Phase 2: Generating session plan...")
        plan = self._planning_service.plan_sessions(self._goal, project_context)

        if not plan or not plan.sessions:
            raise ValueError("Planning produced no sessions.")

        self.progress_update.emit("Session plan ready")
        self.plan_ready.emit(plan)

        session_count = len(plan.sessions)
        estimated_minutes = getattr(plan, "total_estimated_minutes", 0)
        _log_emit_trace(f"  ├─ Generated {session_count} sessions")
        self.session_output.emit(f"  ├─ Generated {session_count} sessions")
        _log_emit_trace(f"  └─ Estimated {estimated_minutes} minutes")
        self.session_output.emit(f"  └─ Estimated {estimated_minutes} minutes")

        all_results: List[SessionResult] = []
        for index, session in enumerate(plan.sessions):
            self.progress_update.emit(f"Session {index + 1}/{session_count}: {session.name}")
            _log_emit_trace("")
            self.session_output.emit("")
            _log_emit_trace(f"Executing Session {index + 1}/{session_count}: {session.name}")
            self.session_output.emit(f"Executing Session {index + 1}/{session_count}: {session.name}")
            self.session_started.emit(index, session)

            LOGGER.info("Executing session %d/%d: %s", index + 1, session_count, session.name)
            result = self._run_session(index, session)
            self.session_complete.emit(index, result)
            self._update_context(index, session, result)
            all_results.append(result)
            if config.AUTO_COMMIT_SESSIONS:
                if result.success and result.files_created:
                    commit_msg = f"Session {index + 1}: {session.name}"
                    self.progress_update.emit("Committing changes...")
                    _log_emit_trace("Committing changes...")
                    self.session_output.emit("Committing changes...")
                    _log_emit_trace(f"  └─ {commit_msg}")
                    self.session_output.emit(f"  └─ {commit_msg}")
                    LOGGER.info("Committing changes: %s", commit_msg)
                    if not self._git.commit(commit_msg, result.files_created):
                        self._event_bus.publish(
                            EventType.ERROR,
                            error=f"Failed to commit changes for {commit_msg}",
                        )
            if result.success:
                _log_emit_trace(f"  └─ ✓ Complete in {result.duration_seconds:.1f}s")
                self.session_output.emit(f"  └─ ✓ Complete in {result.duration_seconds:.1f}s")
            else:
                _log_emit_trace("  └─ ✗ Failed")
                self.session_output.emit("  └─ ✗ Failed")
            if not result.success:
                error_message = f"Session '{session.name}' failed with exit code {result.exit_code}."
                self._event_bus.publish(EventType.ERROR, error=error_message)
                self.error_occurred.emit(error_message)
                return

        self._install_dependencies()
        self.progress_update.emit("All sessions complete")
        self.all_sessions_complete.emit()
        _log_emit_trace("")
        self.session_output.emit("")
        _log_emit_trace("All sessions complete")
        self.session_output.emit("All sessions complete")

        total_files = sum(len(result.files_created) for result in all_results)
        total_duration = sum(result.duration_seconds for result in all_results)
        _log_emit_trace(f"  ├─ Created {total_files} files")
        self.session_output.emit(f"  ├─ Created {total_files} files")

        if config.AUTO_PUSH_ON_COMPLETE:
            self.progress_update.emit("Pushing to GitHub...")
            _log_emit_trace("Pushing to GitHub...")
            self.session_output.emit("Pushing to GitHub...")
            LOGGER.info("Pushing changes to GitHub")
            if self._git.push():
                _log_emit_trace("  ├─ ✓ Pushed to GitHub")
                self.session_output.emit("  ├─ ✓ Pushed to GitHub")
            else:
                _log_emit_trace("  ├─ ✗ Push failed")
                self.session_output.emit("  ├─ ✗ Push failed")
                self._event_bus.publish(EventType.ERROR, error="Failed to push to GitHub")

        _log_emit_trace(f"  └─ Total time: {total_duration:.1f}s")
        self.session_output.emit(f"  └─ Total time: {total_duration:.1f}s")
        LOGGER.info("Orchestration complete: %d sessions, %.1fs total", session_count, total_duration)

    def _resolve_project_directory(self) -> Path:
        """Determine the project directory for the current run."""
        base_dir = self._working_dir
        has_existing_project, indicators = self._detect_project_indicators(base_dir)

        if has_existing_project:
            indicator_text = ", ".join(indicators[:3])
            if indicator_text:
                LOGGER.info(
                    "Existing project detected at %s (indicators: %s)",
                    base_dir,
                    indicator_text,
                )
            else:
                LOGGER.info("Existing project detected at %s", base_dir)
            _log_emit_trace(f"📁 Using existing project: {base_dir.name}")
            self.session_output.emit(f"📁 Using existing project: {base_dir.name}")
            _log_emit_trace(f"   ├─ Location: {base_dir}")
            self.session_output.emit(f"   ├─ Location: {base_dir}")
            if indicator_text:
                _log_emit_trace(f"   └─ Indicators: {indicator_text}")
                self.session_output.emit(f"   └─ Indicators: {indicator_text}")
            else:
                _log_emit_trace("   └─ Indicators: project files detected")
                self.session_output.emit("   └─ Indicators: project files detected")
            _log_emit_trace("")
            self.session_output.emit("")
            return base_dir

        project_name = _infer_project_name(self._goal)
        project_dir = _ensure_unique_project_dir(base_dir, project_name)

        LOGGER.info("Inferred project name: %s", project_name)
        LOGGER.info("Preparing project directory: %s", project_dir)

        try:
            project_dir.mkdir(parents=True, exist_ok=False)
            _log_emit_trace(f"📁 Creating project: {project_dir.name}")
            self.session_output.emit(f"📁 Creating project: {project_dir.name}")
        except FileExistsError:
            LOGGER.warning("Project directory already exists: %s", project_dir)
            _log_emit_trace(f"📁 Using existing project: {project_dir.name}")
            self.session_output.emit(f"📁 Using existing project: {project_dir.name}")
        _log_emit_trace(f"   └─ Location: {project_dir}")
        self.session_output.emit(f"   └─ Location: {project_dir}")
        _log_emit_trace("")
        self.session_output.emit("")
        return project_dir

    def _detect_project_indicators(self, directory: Path) -> tuple[bool, list[str]]:
        """Return whether the directory looks like an existing project."""
        try:
            snapshot = scan_directory(str(directory), max_depth=2)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Unable to scan %s for project indicators: %s",
                directory,
                exc,
            )
            return False, []

        files = snapshot.get("files", [])
        directories = snapshot.get("directories", [])

        has_python_files = False
        key_files_found: set[str] = set()
        key_dirs_found: set[str] = set()

        for file_rel in files:
            relative = Path(file_rel)
            parts = relative.parts
            if parts and parts[0] in PROJECT_IGNORE_DIRS:
                continue
            if relative.suffix.lower() == ".py":
                has_python_files = True
            name_lower = relative.name.lower()
            if name_lower in PROJECT_INDICATOR_FILES:
                key_files_found.add(name_lower)

        for dir_rel in directories:
            relative = Path(dir_rel)
            parts = relative.parts
            if not parts or parts[0] in PROJECT_IGNORE_DIRS:
                continue
            for part in parts:
                if part in PROJECT_INDICATOR_DIRS:
                    key_dirs_found.add(part)

        indicators: list[str] = []
        if has_python_files:
            indicators.append("python files")
        indicators.extend(sorted(key_files_found))
        indicators.extend(f"{name} directory" for name in sorted(key_dirs_found))

        return bool(indicators), indicators

    def _emit_stream_chunk(self, chunk: str) -> None:
        """Relay streaming chunks from agents to the UI."""
        if not chunk:
            return
        _log_emit_trace(chunk)
        self.session_output.emit(chunk)

    def _install_dependencies(self) -> None:
        """Install dependencies from requirements.txt once sessions complete."""
        requirements_path = self._working_dir / "requirements.txt"
        if not requirements_path.exists():
            LOGGER.info("No requirements.txt detected; skipping dependency installation")
            return
        self.progress_update.emit("Installing dependencies...")
        _log_emit_trace(" Installing dependencies...")
        self.session_output.emit(" Installing dependencies...")
        result = self._run_dependency_install()
        if result is None:
            _log_emit_trace(" ⚠️ Dependency installation failed to start. See logs for details.")
            self.session_output.emit(" ⚠️ Dependency installation failed to start. See logs for details.")
            return
        summary = self._summarize_pip_output(result.stdout, result.stderr, result.returncode)
        if summary:
            _log_emit_trace(f"  {summary}")
            self.session_output.emit(f"  {summary}")
        if result.returncode == 0:
            _log_emit_trace(" ✅ Dependencies installed successfully")
            self.session_output.emit(" ✅ Dependencies installed successfully")
        else:
            LOGGER.warning("Dependency installation exited with %d", result.returncode)
            _log_emit_trace(" ⚠️ Dependency installation encountered issues; please review the logs.")
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

    @staticmethod
    def _is_modification_request(goal: str) -> bool:
        """Detect if the goal is modifying existing code vs creating from scratch.

        Args:
            goal: The user's request/goal

        Returns:
            True if this is a modification request, False if pure creation
        """
        goal_lower = goal.lower()

        # Modification keywords
        modification_keywords = {
            "add", "update", "modify", "change", "edit", "fix", "include",
            "extend", "enhance", "improve", "refactor", "adjust", "alter",
            "append", "insert", "remove", "delete", "replace"
        }

        # Modification phrases
        modification_phrases = [
            "add field", "add method", "add function", "add endpoint",
            "add route", "add email", "add role", "add parameter",
            "update to", "change the", "modify the", "fix the"
        ]

        # Pure creation keywords (override modification detection)
        creation_keywords = {
            "create", "build", "make", "generate", "new", "from scratch",
            "start a", "start an", "initialize"
        }

        # Creation phrases
        creation_phrases = [
            "build a", "create a", "make a", "new project",
            "from scratch", "start from", "generate a"
        ]

        # Check for pure creation first (takes priority)
        if any(keyword in goal_lower for keyword in creation_keywords):
            # Check if it's unambiguously creation
            if any(phrase in goal_lower for phrase in creation_phrases):
                return False

        # Check for modification keywords
        has_modification_keyword = any(keyword in goal_lower for keyword in modification_keywords)

        # Check for modification phrases
        has_modification_phrase = any(phrase in goal_lower for phrase in modification_phrases)

        return has_modification_keyword or has_modification_phrase

    def _discover_project_context(self, goal: str) -> str:
        """Use ChatService with tools to discover project context intelligently.

        This method triggers the mandatory tool usage workflow defined in AURA_SYSTEM_PROMPT,
        causing ChatService to analyze the project using its 8 developer tools before planning.
        """
        _log_emit_trace("  └─ Discovering project context with AI tools...")
        self.session_output.emit("  └─ Discovering project context with AI tools...")
        LOGGER.info("Starting intelligent project discovery with ChatService")

        # Detect if this is a modification request
        is_modification = self._is_modification_request(goal)

        if is_modification:
            LOGGER.info("Detected MODIFICATION request - will enhance discovery with symbol resolution")
            _log_emit_trace("    ├─ Detected modification request - using enhanced context gathering...")
            self.session_output.emit("    ├─ Detected modification request - using enhanced context gathering...")
        else:
            LOGGER.info("Detected CREATION request - will use standard discovery")
            _log_emit_trace("    ├─ Detected creation request - using standard discovery...")
            self.session_output.emit("    ├─ Detected creation request - using standard discovery...")

        api_key = self._api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            LOGGER.warning("No API key available for discovery phase, falling back to basic context")
            return self._build_project_context()

        try:
            chat = ChatService(api_key=api_key)

            # Build discovery prompt based on request type
            if is_modification:
                # Enhanced prompt for modification requests - MANDATES symbol resolution
                discovery_prompt = (
                    f"⚠️ MODIFICATION REQUEST DETECTED ⚠️\n\n"
                    f"Task: {goal}\n\n"
                    "This is a MODIFICATION request (not pure creation). You MUST follow the enhanced discovery protocol:\n\n"
                    "MANDATORY STEPS - YOU CANNOT SKIP THESE:\n\n"
                    "1. Call list_project_files() to locate existing files\n"
                    "2. Call read_multiple_files() to read ALL relevant existing code files\n"
                    "3. Call find_definition() on EVERY class/function being modified\n"
                    "   - Example: If adding field to User, call find_definition('User') to see current __init__\n"
                    "4. Call find_usages() to understand where modified entities are referenced\n"
                    "   - This shows you how existing code uses these symbols\n"
                    "5. Call get_imports() on files being modified to verify available dependencies\n"
                    "6. Call get_function_definitions() to see all function signatures in relevant files\n\n"
                    "CRITICAL: You CANNOT plan modifications without reading existing code and resolving symbols.\n"
                    "If you skip find_definition() or read_project_file(), you WILL create bugs.\n\n"
                    "Extract:\n"
                    "- Exact current signatures and fields\n"
                    "- Usage patterns and dependencies\n"
                    "- Existing implementation details\n"
                    "- Where changes will have impact\n\n"
                    "Be thorough - gather ALL context needed to modify code without breaking existing functionality."
                )
            else:
                # Standard prompt for creation requests
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
            _log_emit_trace("    └─ Running discovery phase with automatic tool calling...")
            self.session_output.emit("    └─ Running discovery phase with automatic tool calling...")
            combined_discovery = chat.send_message(
                discovery_prompt,
                on_chunk=self._emit_stream_chunk,
            )

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
            _log_emit_trace(f"  └─ Discovery failed: {exc}, using basic context")
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
                _log_emit_trace(f"⚠️ Native agent failed, using CLI fallback: {exc}")
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
