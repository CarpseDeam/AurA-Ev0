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
                "You are a Python coding agent. Respond with JSON containing keys "
                "'summary', 'files', and 'commands'. Each file item must include "
                "'path', 'action' (create|modify), and 'content'. Commands should be "
                "an array of shell command strings."
            ),
        )

    def execute_session(self, context: SessionContext) -> AgentResult:
        """Run a coding session and return structured results."""
        start = perf_counter()
        created: List[str] = []
        modified: List[str] = []
        commands: List[str] = []
        outputs: List[str] = []
        errors: List[str] = []
        summary = ""
        try:
            prompt = self._build_prompt(context)
            plan = self._request_plan(prompt)
            summary = plan.get("summary", "")
            file_ops = self._parse_file_operations(plan.get("files", []), context.working_dir)
            created, modified = self._apply_files(file_ops, context.working_dir)
            commands, cmd_outputs, cmd_errors = self._execute_commands(
                plan.get("commands", []),
                context.working_dir,
            )
            outputs.extend(cmd_outputs)
            errors.extend(cmd_errors)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Coding session failed: %s", exc)
            errors.append(str(exc))
        duration = perf_counter() - start
        return AgentResult(
            success=not errors,
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
        response = self._model.generate_content(prompt)
        payload = (response.text or "").strip()
        if not payload:
            raise PlanParseError("Empty response from Gemini.")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"Failed to parse Gemini response: {payload}") from exc
        if not isinstance(data, dict):
            raise PlanParseError("Gemini response must be a JSON object.")
        return data

    def _parse_file_operations(
        self,
        items: Iterable[Mapping[str, object]],
        working_dir: Path,
    ) -> List[_FileOperation]:
        if not items:
            return []
        operations: List[_FileOperation] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise PlanParseError("File entries must be JSON objects.")
            path = Path(str(item.get("path", ""))).expanduser()
            if not path.is_absolute():
                path = working_dir / path
            action = str(item.get("action", "modify")).lower()
            content = str(item.get("content", ""))
            if action not in {"create", "modify"}:
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
        executed: List[str] = []
        outputs: List[str] = []
        errors: List[str] = []
        for command in commands or []:
            cmd = command.strip()
            if not cmd:
                continue
            executed.append(cmd)
            result = subprocess.run(
                cmd,
                cwd=working_dir,
                shell=True,
                check=False,
                text=True,
                capture_output=True,
            )
            self.command_executed.emit(cmd, result.returncode)
            outputs.extend(self._split_output(result.stdout))
            outputs.extend(self._split_output(result.stderr))
            if result.returncode != 0:
                errors.append(f"Command '{cmd}' failed with exit code {result.returncode}.")
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
