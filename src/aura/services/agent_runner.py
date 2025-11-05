"""Agent runner thread for executing Gemini CLI commands."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Mapping

from PySide6.QtCore import QObject, QThread, Signal

LOGGER = logging.getLogger(__name__)


class AgentRunner(QThread):
    """Runs a Gemini CLI command on a background thread; example: runner = AgentRunner(command=['gemini', '-p', 'Create hello.py'], working_directory='/path'); runner.output_line.connect(self.handle_output); runner.start()."""

    output_line = Signal(str)
    process_finished = Signal(int)
    process_error = Signal(str)

    def __init__(
        self,
        command: list[str],
        working_directory: str,
        environment: Mapping[str, str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the agent runner configuration."""
        super().__init__(parent)
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("Command must be a non-empty list of strings.")
        if not working_directory:
            raise ValueError("Working directory must be a non-empty string.")
        if environment is not None and not all(
            isinstance(key, str) and isinstance(val, str) for key, val in environment.items()
        ):
            raise ValueError("Environment must map strings to strings.")
        if any(item.startswith("--output-format") for item in command):
            raise ValueError("Gemini command must not specify --output-format.")
        sanitized_command = list(command)
        if "--yolo" not in sanitized_command:
            sanitized_command.append("--yolo")
        self._command = sanitized_command
        self._cwd = working_directory
        self._environment = dict(environment) if environment is not None else None

    def run(self) -> None:
        """Execute the configured Gemini command."""
        env = self._build_environment()
        process = self._launch_process(env)
        if process is None:
            return
        exit_code = self._monitor_process(process)
        self.process_finished.emit(exit_code)

    def _build_environment(self) -> dict[str, str]:
        """Compose the environment for the subprocess."""
        env = os.environ.copy()
        if self._environment is not None:
            env.update(self._environment)
        return env

    def _launch_process(self, env: Mapping[str, str]) -> subprocess.Popen[str] | None:
        """Start the Gemini CLI subprocess."""
        try:
            return subprocess.Popen(
                self._command,
                cwd=self._cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"Failed to start process: {exc}"
            LOGGER.exception(message)
            self.process_error.emit(message)
            self.process_finished.emit(1)
            return None

    def _monitor_process(self, process: subprocess.Popen[str]) -> int:
        """Stream process output and return its exit code."""
        assert process.stdout is not None
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                self.output_line.emit(line.rstrip("\r\n"))
        except Exception as exc:  # noqa: BLE001
            message = f"Error while reading process output: {exc}"
            LOGGER.exception(message)
            self.process_error.emit(message)
        finally:
            process.stdout.close()
        exit_code = process.wait()
        if exit_code != 0:
            error_message = f"Process exited with code {exit_code}"
            LOGGER.error(error_message)
            self.process_error.emit(error_message)
        return exit_code
