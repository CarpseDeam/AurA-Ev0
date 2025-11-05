"""Session execution strategies using the Strategy pattern.

This module defines execution strategies for different agent types,
following the Strategy design pattern to improve modularity.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QEventLoop, QObject, Signal

from src.aura.agents import PythonCoderAgent, SessionContext
from src.aura.orchestrator import SessionResult
from src.aura.services import AgentRunner
from src.aura.utils import scan_directory

if TYPE_CHECKING:
    from src.aura.services.planning_service import Session

LOGGER = logging.getLogger(__name__)


class SessionExecutor(ABC):
    """Abstract base class for session execution strategies.

    Concrete implementations define how to execute a session
    using different agent types (native Python, CLI, etc.).
    """

    @abstractmethod
    def execute(
        self,
        session: Session,
        context: dict[str, object],
    ) -> SessionResult:
        """Execute a single session.

        Args:
            session: Session to execute
            context: Execution context with required parameters

        Returns:
            SessionResult with execution details
        """
        pass


class NativeAgentExecutor(SessionExecutor):
    """Executes sessions using the native PythonCoderAgent.

    This strategy runs the agent in-process without subprocess overhead.
    """

    def __init__(
        self,
        api_key: str,
        output_signal: Signal | None = None,
    ) -> None:
        """Initialize the native agent executor.

        Args:
            api_key: Gemini API key for agent
            output_signal: Optional signal to emit output lines
        """
        self._api_key = api_key
        self._output_signal = output_signal

    def execute(
        self,
        session: Session,
        context: dict[str, object],
    ) -> SessionResult:
        """Execute session using PythonCoderAgent.

        Args:
            session: Session to execute
            context: Must contain 'working_dir' (Path) and 'context_notes' (list)

        Returns:
            SessionResult with execution details

        Raises:
            KeyError: If required context parameters are missing
        """
        working_dir = context["working_dir"]
        context_notes = context.get("context_notes", [])

        # Get project files from scan
        snapshot = scan_directory(str(working_dir), max_depth=2)
        python_files = [f for f in snapshot["files"] if f.endswith(".py")]

        # Build session context
        session_context = SessionContext(
            working_dir=working_dir,
            session_prompt=session.prompt,
            previous_work=context_notes,
            project_files=python_files,
        )

        # Create and execute agent
        agent = PythonCoderAgent(api_key=self._api_key)
        if self._output_signal:
            agent.progress_update.connect(self._output_signal)

        agent_result = agent.execute_session(session_context)

        # Emit output lines if signal provided
        if self._output_signal:
            for line in agent_result.output_lines:
                self._output_signal.emit(line)

        # Convert AgentResult to SessionResult
        all_files = list(agent_result.files_created) + list(agent_result.files_modified)
        return SessionResult(
            session_name=session.name,
            exit_code=0 if agent_result.success else 1,
            duration_seconds=agent_result.duration_seconds,
            files_created=all_files,
            success=agent_result.success,
        )


class CliAgentExecutor(SessionExecutor):
    """Executes sessions using a CLI agent subprocess.

    This strategy runs external agent executables via subprocess,
    providing fallback compatibility with various CLI tools.
    """

    def __init__(
        self,
        agent_path: str,
        output_signal: Signal | None = None,
        error_signal: Signal | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the CLI agent executor.

        Args:
            agent_path: Path to CLI agent executable
            output_signal: Optional signal to emit output lines
            error_signal: Optional signal to emit error messages
            parent: Optional parent QObject
        """
        self._agent_path = agent_path
        self._output_signal = output_signal
        self._error_signal = error_signal
        self._parent = parent

    def execute(
        self,
        session: Session,
        context: dict[str, object],
    ) -> SessionResult:
        """Execute session using CLI agent subprocess.

        Args:
            session: Session to execute
            context: Must contain 'working_dir' (Path), 'context_notes' (list),
                    'before_snapshot' (dict), and 'after_snapshot' (dict)

        Returns:
            SessionResult with execution details

        Raises:
            KeyError: If required context parameters are missing
        """
        working_dir = context["working_dir"]
        context_notes = context.get("context_notes", [])

        # Take before snapshot
        before = self._snapshot_directory(working_dir)

        # Prepare prompt with context
        prompt = self._prepare_prompt(session.prompt, context_notes)

        # Create and run agent
        runner = AgentRunner(
            command=[self._agent_path, "-p", prompt, "--yolo"],
            working_directory=str(working_dir),
            parent=self._parent,
        )

        if self._output_signal:
            runner.output_line.connect(self._output_signal)
        if self._error_signal:
            runner.process_error.connect(self._error_signal)

        exit_code, duration = self._await_runner(runner)

        # Take after snapshot and detect changes
        after = self._snapshot_directory(working_dir)
        files_created = self._detect_file_changes(before, after)

        success = exit_code == 0
        return SessionResult(
            session_name=session.name,
            exit_code=exit_code,
            duration_seconds=duration,
            files_created=files_created,
            success=success,
        )

    def _await_runner(self, runner: AgentRunner) -> tuple[int, float]:
        """Start the runner and block until it finishes.

        Args:
            runner: AgentRunner instance to execute

        Returns:
            Tuple of (exit_code, duration_seconds)
        """
        loop = QEventLoop()
        result = {"code": 1}

        def _on_finished(code: int) -> None:
            result["code"] = code
            loop.quit()

        runner.process_finished.connect(_on_finished)
        start = time.monotonic()
        runner.start()
        loop.exec()
        runner.wait()
        elapsed = time.monotonic() - start
        runner.deleteLater()
        return result["code"], elapsed

    def _prepare_prompt(self, original_prompt: str, context_notes: list[str]) -> str:
        """Combine accumulated context with the session's prompt.

        Args:
            original_prompt: Original session prompt
            context_notes: List of context notes from previous sessions

        Returns:
            Combined prompt with context
        """
        if not context_notes:
            return original_prompt
        context = "\n".join(context_notes)
        return f"Previous work:\n{context}\n\n{original_prompt}"

    def _snapshot_directory(self, working_dir: Path) -> dict[str, float]:
        """Capture a timestamp snapshot of the working directory.

        Args:
            working_dir: Directory to snapshot

        Returns:
            Dictionary mapping relative paths to modification times
        """
        snapshot: dict[str, float] = {}
        for root, _, files in working_dir.walk():
            for filename in files:
                path = Path(root) / filename
                relative = str(path.relative_to(working_dir))
                snapshot[relative] = path.stat().st_mtime
        return snapshot

    def _detect_file_changes(
        self, before: dict[str, float], after: dict[str, float]
    ) -> list[str]:
        """Identify new or modified files between snapshots.

        Args:
            before: Snapshot before execution
            after: Snapshot after execution

        Returns:
            List of file paths that were created or modified
        """
        created = [path for path in after if path not in before]
        updated = [
            path for path in after if path in before and after[path] != before[path]
        ]
        annotated = created + [f"{path} (updated)" for path in updated]
        return sorted(annotated)
