"""Agent runner thread for executing Gemini CLI commands."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from threading import Thread
from typing import IO, Mapping

from PySide6.QtCore import QObject, QThread, Signal

from aura.utils.file_filter import is_file_protected

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
        LOGGER.info(
            "AgentRunner starting | cwd=%s | command=%s",
            self._cwd,
            self._summarize_command(),
        )
        env = self._build_environment()
        process = self._launch_process(env)
        if process is None:
            return
        exit_code = self._monitor_process(process)
        LOGGER.info("AgentRunner finished | exit_code=%s", exit_code)
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
                stderr=subprocess.PIPE,
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
        assert process.stdout is not None and process.stderr is not None

        stderr_thread = Thread(
            target=self._drain_stderr,
            args=(process.stderr,),
            daemon=True,
        )
        stderr_thread.start()
        try:
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                stripped_line = line.rstrip("\r\n")
                self._check_file_protection(stripped_line)
                self.output_line.emit(stripped_line)
        except Exception as exc:  # noqa: BLE001
            message = f"Error while reading process output: {exc}"
            LOGGER.exception(message)
            self.process_error.emit(message)
        finally:
            process.stdout.close()
        stderr_thread.join(timeout=1)
        process.stderr.close()
        exit_code = process.wait()
        if exit_code != 0:
            error_message = f"Process exited with code {exit_code}"
            LOGGER.error(error_message)
            self.process_error.emit(error_message)
        return exit_code

    def _check_file_protection(self, line: str) -> None:
        """Check if line indicates modification of protected files."""
        patterns = [
            r"(?:Creating|Modifying|Writing|Editing|Deleting|Updated?)\s+(?:file\s+)?['\"]?([^'\":\n]+)['\"]?",
            r"(?:Write|Edit|Delete):\s+([^\n]+)",
            r"File:\s+([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if not match:
                continue
            file_path = match.group(1).strip()
            abs_path = Path(self._cwd) / file_path
            if is_file_protected(str(abs_path), self._cwd):
                warning = f"⚠️  Agent attempting to modify protected file: {file_path}"
                LOGGER.warning(warning)
                self.output_line.emit(warning)
            break

    def _drain_stderr(self, pipe: IO[str]) -> None:
        """Stream stderr output to logs and the UI."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                stripped_line = line.rstrip("\r\n")
                LOGGER.error("Agent stderr | %s", stripped_line)
                self.output_line.emit(f"[stderr] {stripped_line}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read stderr: %s", exc)

    def _summarize_command(self) -> str:
        """Return a redacted preview of the command to avoid logging prompts."""
        if not self._command:
            return ""
        preview = self._command[0]
        if len(self._command) > 1:
            preview += " ..."
        return preview


def run_agent_command_sync(runner: AgentRunner) -> tuple[int, str]:
    """Execute a Gemini CLI command synchronously and return exit code and output."""
    env = runner._build_environment()
    LOGGER.info(
        "AgentRunner(sync) starting | cwd=%s | command=%s",
        runner._cwd,
        runner._summarize_command(),
    )

    try:
        process = subprocess.Popen(
            runner._command,
            cwd=runner._cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to start process: {exc}"
        LOGGER.exception(message)
        return 1, message

    assert process.stdout is not None and process.stderr is not None
    output_lines: list[str] = []
    stderr_lines: list[str] = []

    def _drain_sync_stderr() -> None:
        try:
            for line in iter(process.stderr.readline, ""):
                if not line:
                    break
                stripped = line.rstrip("\r\n")
                LOGGER.error("Agent stderr (sync) | %s", stripped)
                stderr_lines.append(f"[stderr] {stripped}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to read stderr (sync): %s", exc)

    stderr_thread = Thread(target=_drain_sync_stderr, daemon=True)
    stderr_thread.start()
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            stripped_line = line.rstrip("\r\n")
            runner._check_file_protection(stripped_line)
            output_lines.append(stripped_line)
    except Exception as exc:  # noqa: BLE001
        message = f"Error while reading process output: {exc}"
        LOGGER.exception(message)
        output_lines.append(message)
    finally:
        process.stdout.close()
    stderr_thread.join(timeout=1)
    process.stderr.close()

    exit_code = process.wait()
    if stderr_lines:
        output_lines.extend(stderr_lines)
    LOGGER.info("AgentRunner(sync) finished | exit_code=%s", exit_code)
    if exit_code != 0:
        message = f"Process exited with code {exit_code}"
        LOGGER.error(message)
        output_lines.append(message)

    return exit_code, "\n".join(output_lines)
