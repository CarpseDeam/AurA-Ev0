"""Agent for running Gemini-powered Python coding sessions."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Mapping, Sequence, Tuple

import google.generativeai as genai
from PySide6.QtCore import QObject, Signal

LOGGER = logging.getLogger(__name__)


class PlanParseError(RuntimeError):
    """Raised when the Gemini response cannot be parsed."""


@dataclass(frozen=True)
class SessionContext:
    """Immutable context for a coding session."""

    working_dir: Path
    session_prompt: str
    previous_work: Sequence[str]
    project_files: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "working_dir", Path(self.working_dir).resolve())
        object.__setattr__(self, "previous_work", tuple(self.previous_work))
        object.__setattr__(self, "project_files", tuple(self.project_files))


@dataclass(frozen=True)
class AgentResult:
    """Structured result from a coding session."""

    success: bool
    summary: str
    files_created: Sequence[str]
    files_modified: Sequence[str]
    commands_run: Sequence[str]
    output_lines: Sequence[str]
    errors: Sequence[str]
    duration_seconds: float


@dataclass(frozen=True)
class _FileOperation:
    path: Path
    action: str
    content: str


class PythonCoderAgent(QObject):
    """Gemini-backed agent that writes code and executes commands."""

    progress_update = Signal(str)
    command_executed = Signal(str, int)

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro") -> None:
        super().__init__()
        if not api_key:
            raise ValueError("Gemini API key is required.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model,
            system_instruction=(
                "You are a Python coding agent. You MUST respond with valid JSON containing these keys:\n"
                "- 'summary': A brief description of what you're doing\n"
                "- 'files': An array of file operations, where each item has:\n"
                "  - 'path': The file path (relative or absolute)\n"
                "  - 'action': Either 'create' or 'modify'\n"
                "  - 'content': The complete file content as a string\n"
                "- 'commands': An array of shell command strings to execute\n\n"
                "Example response:\n"
                '{"summary": "Creating hello.py", "files": [{"path": "hello.py", '
                '"action": "create", "content": "print(\'Hello World\')"}], "commands": ["python hello.py"]}'
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )

    def execute_session(self, context: SessionContext) -> AgentResult:
        """Run a coding session and return structured results."""
        LOGGER.info(
            "Starting coding session: working_dir=%s, prompt_length=%d, previous_work_count=%d",
            context.working_dir,
            len(context.session_prompt),
            len(context.previous_work),
        )
        start = perf_counter()
        created: List[str] = []
        modified: List[str] = []
        commands: List[str] = []
        outputs: List[str] = []
        errors: List[str] = []
        summary = ""
        try:
            prompt = self._build_prompt(context)
            LOGGER.debug("Built prompt with %d characters", len(prompt))

            plan = self._request_plan(prompt)
            summary = plan.get("summary", "")
            LOGGER.info("Received plan: %s", summary)

            file_ops = self._parse_file_operations(plan.get("files", []), context.working_dir)
            LOGGER.debug("Parsed %d file operations", len(file_ops))

            created, modified = self._apply_files(file_ops, context.working_dir)
            LOGGER.info("Applied files: %d created, %d modified", len(created), len(modified))

            commands, cmd_outputs, cmd_errors = self._execute_commands(
                plan.get("commands", []),
                context.working_dir,
            )
            LOGGER.info("Executed %d commands", len(commands))

            outputs.extend(cmd_outputs)
            errors.extend(cmd_errors)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Coding session failed: %s", exc)
            errors.append(str(exc))

        duration = perf_counter() - start
        success = not errors

        LOGGER.info(
            "Session completed: success=%s, duration=%.2fs, files=%d, commands=%d",
            success,
            duration,
            len(created) + len(modified),
            len(commands),
        )

        return AgentResult(
            success=success,
            summary=summary,
            files_created=tuple(created),
            files_modified=tuple(modified),
            commands_run=tuple(commands),
            output_lines=tuple(outputs),
            errors=tuple(errors),
            duration_seconds=duration,
        )

    def _build_prompt(self, context: SessionContext) -> str:
        sections = [context.session_prompt.strip()]
        if context.previous_work:
            sections.append("Previous work:\n" + "\n".join(context.previous_work))
        if context.project_files:
            sections.append("Relevant project files:\n" + "\n".join(context.project_files))
        sections.append(
            "Respond with JSON as described in the system instruction. Only emit valid JSON."
        )
        return "\n\n".join(section for section in sections if section)

    def _request_plan(self, prompt: str) -> Mapping[str, object]:
        LOGGER.debug("Requesting plan from Gemini API")
        response = self._model.generate_content(prompt)
        payload = (response.text or "").strip()

        if not payload:
            LOGGER.error("Gemini returned empty response")
            raise PlanParseError("Empty response from Gemini.")

        LOGGER.debug("Received response from Gemini: %d characters", len(payload))

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse JSON response: %s", payload[:200])
            raise PlanParseError(f"Failed to parse Gemini response: {payload}") from exc

        if not isinstance(data, dict):
            LOGGER.error("Response is not a JSON object: %s", type(data))
            raise PlanParseError("Gemini response must be a JSON object.")

        return data

    def _parse_file_operations(
        self,
        items: Iterable[Mapping[str, object]],
        working_dir: Path,
    ) -> List[_FileOperation]:
        if not items:
            LOGGER.debug("No file operations in plan")
            return []

        operations: List[_FileOperation] = []
        for item in items:
            if not isinstance(item, Mapping):
                LOGGER.error("Invalid file entry type: %s", type(item))
                raise PlanParseError("File entries must be JSON objects.")

            path = Path(str(item.get("path", ""))).expanduser()
            if not path.is_absolute():
                path = working_dir / path

            action = str(item.get("action", "modify")).lower()
            content = str(item.get("content", ""))

            if action not in {"create", "modify"}:
                LOGGER.error("Unsupported action '%s' for %s", action, path)
                raise PlanParseError(f"Unsupported action '{action}' for {path}.")

            operations.append(_FileOperation(path=path, action=action, content=content))

        return operations

    def _apply_files(
        self,
        operations: Sequence[_FileOperation],
        working_dir: Path,
    ) -> Tuple[List[str], List[str]]:
        created: List[str] = []
        modified: List[str] = []
        for op in operations:
            op.path.parent.mkdir(parents=True, exist_ok=True)
            existed = op.path.exists()
            op.path.write_text(op.content, encoding="utf-8")
            rel_path = self._to_relative(working_dir, op.path)
            self.progress_update.emit(f"wrote {rel_path}")
            if existed:
                modified.append(rel_path)
            else:
                created.append(rel_path)
        return created, modified

    def _execute_commands(
        self,
        commands: Iterable[str],
        working_dir: Path,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Execute shell commands generated by the LLM.

        WARNING: This executes commands from LLM output using shell=True. Commands are
        validated against an allowlist of safe command prefixes. This is intended for
        local development environments with trusted LLM outputs only.

        Security considerations:
        - Only allows commands starting with known-safe prefixes
        - Rejects commands containing dangerous patterns (rm -rf, sudo, etc.)
        - Uses shell=True for convenience but validates input first
        - Suitable for local trusted use, not production or untrusted environments
        """
        executed: List[str] = []
        outputs: List[str] = []
        errors: List[str] = []

        # Allowlist of safe command prefixes for local development
        safe_prefixes = ("python", "pip", "pytest", "git", "npm", "node", "poetry")
        dangerous_patterns = ("rm -rf", "sudo", "chmod", "chown", ">", ">>", "|", "&", ";")

        for command in commands or []:
            cmd = command.strip()
            if not cmd:
                continue

            # Validate command safety
            cmd_lower = cmd.lower()
            if not any(cmd_lower.startswith(prefix) for prefix in safe_prefixes):
                LOGGER.warning("Skipping command with disallowed prefix: %s", cmd)
                errors.append(f"Command '{cmd}' rejected: must start with allowed prefix")
                continue

            if any(pattern in cmd for pattern in dangerous_patterns):
                LOGGER.warning("Skipping command with dangerous pattern: %s", cmd)
                errors.append(f"Command '{cmd}' rejected: contains dangerous pattern")
                continue

            executed.append(cmd)
            LOGGER.debug("Executing command: %s", cmd)

            result = subprocess.run(
                cmd,
                cwd=working_dir,
                shell=True,  # Necessary for pip/python with args, but validated above
                check=False,
                text=True,
                capture_output=True,
            )

            self.command_executed.emit(cmd, result.returncode)
            outputs.extend(self._split_output(result.stdout))
            outputs.extend(self._split_output(result.stderr))

            if result.returncode != 0:
                LOGGER.warning("Command failed with exit code %d: %s", result.returncode, cmd)
                errors.append(f"Command '{cmd}' failed with exit code {result.returncode}.")
            else:
                LOGGER.debug("Command succeeded: %s", cmd)

        return executed, outputs, errors

    @staticmethod
    def _split_output(stream: str | None) -> List[str]:
        return [line for line in (stream or "").splitlines() if line]

    @staticmethod
    def _to_relative(base: Path, target: Path) -> str:
        try:
            return str(target.relative_to(base))
        except ValueError:
            return str(target)
