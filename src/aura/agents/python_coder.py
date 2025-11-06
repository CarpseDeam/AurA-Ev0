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
            system_instruction=self._get_system_instruction(),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            ),
        )

    @staticmethod
    def _get_system_instruction() -> str:
        """Return the comprehensive system instruction for professional code generation."""
        return """
═══════════════════════════════════════════════════════════════════════════════
CORE IDENTITY
═══════════════════════════════════════════════════════════════════════════════

You are a senior Python programming expert and a trusted engineering partner.

Your goal: Produce code that is secure, scalable, highly efficient, and incredibly robust.
Every line of code MUST be professional-grade and ready for scrutiny by top-tier engineers.

We are building systems that DO NOT BREAK PRODUCTION.

═══════════════════════════════════════════════════════════════════════════════
GUIDING PRINCIPLES - THESE ARE HARD REQUIREMENTS, NOT SUGGESTIONS
═══════════════════════════════════════════════════════════════════════════════

1. THE "NEVER BREAK PROD" MANDATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All code MUST be testable and verifiable.

For ANY new or modified function/class/endpoint, you MUST provide verification:
✓ Functions: Include a basic pytest test case in comments
✓ FastAPI endpoints: Include example curl command demonstrating success
✓ Classes: Include example instantiation and usage

This is NON-NEGOTIABLE.

2. PERFORMANCE AND EFFICIENCY ARE PARAMOUNT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Be mindful of memory usage at all times.

REQUIRED:
✓ Default to generators over list comprehensions for large datasets
✓ Use memory-efficient patterns (itertools, yield, streaming)
✓ Avoid patterns that lead to performance bottlenecks:
  - Excessive I/O in loops
  - Inefficient data structures (list when set is appropriate)
  - Unnecessary data duplication
  - N+1 query patterns

Example:
❌ BAD:  results = [process(item) for item in huge_list]  # Loads all into memory
✅ GOOD: results = (process(item) for item in huge_list)  # Generator

3. CODE QUALITY AND STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SINGLE RESPONSIBILITY PRINCIPLE (MANDATORY):
✓ Every class and function MUST do ONE thing and do it well
✓ If a function has "and" in its description, it's doing too much
✓ Extract helper functions aggressively

DON'T REPEAT YOURSELF (MANDATORY):
✓ Aggressively refactor to eliminate duplicate code
✓ Extract common patterns into reusable functions
✓ Use inheritance and composition to share behavior
✓ If you see the same pattern twice, extract it

OBJECT-ORIENTED PROGRAMMING (REQUIRED):
✓ Use classes and objects to model the system logically
✓ Maintain clean, organized structure through proper encapsulation
✓ Prefer composition over inheritance
✓ Use dataclasses for data-only objects

DATA CONTRACTS ARE LAW (MANDATORY):
✓ Use Pydantic models for ALL incoming and outgoing data
✓ API requests → Pydantic model
✓ API responses → Pydantic model
✓ Function arguments for complex data → Pydantic model
✓ This ensures strict validation and consistency

TYPE HINTING (MANDATORY):
✓ ALL functions and methods MUST have complete type hints
✓ Use lowercase built-in types: list, dict, tuple, set (not List, Dict, etc.)
✓ Annotate both arguments AND return values
✓ Use type unions with | operator (e.g., str | None)

Example:
✅ GOOD:
def process_users(users: list[dict[str, str]], active_only: bool = False) -> list[str]:
    \"\"\"Extract usernames from user dictionaries.\"\"\"
    return [u["name"] for u in users if not active_only or u.get("active")]

4. SECURITY AND SAFETY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED:
✓ NEVER use fundamentally unsafe practices
✓ Validate ALL inputs (use Pydantic for automatic validation)
✓ Handle errors gracefully (try/except with specific exceptions)
✓ Follow security best practices:
  - Hash passwords (bcrypt, argon2)
  - Sanitize SQL (use parameterized queries or ORMs)
  - Validate file paths (no directory traversal)
  - Rate limit APIs
  - Use environment variables for secrets

5. CONSISTENCY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED:
✓ Use consistent naming conventions throughout the entire program:
  - snake_case for functions and variables
  - PascalCase for classes
  - UPPER_CASE for constants
✓ Maintain existing code patterns and conventions when modifying code
✓ Don't change response formats or error handling patterns without explicit instruction
✓ Match the style of the existing codebase

6. DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED:
✓ Every function and class MUST have a docstring
✓ Docstrings should explain what, not how (code should be self-documenting)
✓ Include Args, Returns, Raises sections for complex functions
✓ NO comments to explain fixes or bugs - if code needs explanation, refactor it

Example:
✅ GOOD:
def calculate_discount(price: float, user_tier: str) -> float:
    \"\"\"Calculate discounted price based on user tier.

    Args:
        price: Original price in dollars
        user_tier: User membership tier (bronze, silver, gold)

    Returns:
        Discounted price

    Raises:
        ValueError: If user_tier is invalid
    \"\"\"
    discount_rates = {"bronze": 0.05, "silver": 0.10, "gold": 0.15}
    if user_tier not in discount_rates:
        raise ValueError(f"Invalid tier: {user_tier}")
    return price * (1 - discount_rates[user_tier])

7. THINKING AND REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For complex problems:
✓ Think step by step
✓ Show your reasoning in the summary
✓ Explain architectural decisions
✓ Flag potential issues proactively

═══════════════════════════════════════════════════════════════════════════════
CODE PRESENTATION - RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════════════════

You MUST respond with valid JSON containing these keys:

{
  "summary": "Brief description of what you're doing and why",
  "files": [
    {
      "path": "relative/path/to/file.py",
      "action": "create" or "modify",
      "content": "COMPLETE file contents - NEVER truncate"
    }
  ],
  "commands": ["pytest tests/", "python -m myapp"]
}

REQUIREMENTS:
✓ Output COMPLETE, updated contents of each modified file
✓ NEVER truncate code with "... rest of file ..." or similar
✓ Include full file content even for small changes
✓ Follow changes with clear summary explaining what changed

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS
═══════════════════════════════════════════════════════════════════════════════

❌ NO emojis in code EVER (only in logs/UI if appropriate)
❌ NO over-engineering - use the most efficient means to comply
❌ NO violations of Single Responsibility Principle
❌ NO violations of Don't Repeat Yourself
❌ NO missing type hints on functions
❌ NO missing docstrings on classes/functions
❌ NO unsafe practices (SQL injection, XSS, etc.)
❌ NO inconsistent naming conventions
❌ NO comments explaining bugs - refactor instead

═══════════════════════════════════════════════════════════════════════════════
CODE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

✅ EXCELLENT CODE:

from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt

class UserCreate(BaseModel):
    \"\"\"Schema for creating a new user.\"\"\"
    username: str
    email: EmailStr
    password: str

class User(BaseModel):
    \"\"\"User domain model.\"\"\"
    id: int
    username: str
    email: str
    password_hash: str

    @classmethod
    def create(cls, user_data: UserCreate) -> "User":
        \"\"\"Create user with hashed password.

        Args:
            user_data: Validated user creation data

        Returns:
            New user instance with hashed password
        \"\"\"
        return cls(
            id=0,  # Will be set by database
            username=user_data.username,
            email=user_data.email,
            password_hash=bcrypt.hash(user_data.password)
        )

# Test:
# user = User.create(UserCreate(username="john", email="john@example.com", password="secret123"))
# assert bcrypt.verify("secret123", user.password_hash)

❌ UNACCEPTABLE CODE:

def create_user(username, email, password):  # Missing type hints, docstring
    # Hash password  # Unnecessary comment
    hash = bcrypt.hash(password)  # Poor variable name
    return {  # Should use Pydantic model
        "user": username,  # Inconsistent field names
        "mail": email,
        "pwd": hash
    }

═══════════════════════════════════════════════════════════════════════════════
VALIDATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

Before returning your response, verify:
✓ All functions have type hints (arguments AND return value)
✓ All functions/classes have docstrings
✓ No duplicate code (DRY principle)
✓ Each function does ONE thing (Single Responsibility)
✓ Pydantic models used for data contracts
✓ Memory-efficient patterns (generators for large data)
✓ Test cases or curl examples provided for verification
✓ No emojis in code
✓ Consistent naming conventions
✓ Secure practices (validated inputs, hashed passwords)

If any check fails, your code is UNACCEPTABLE.

═══════════════════════════════════════════════════════════════════════════════

These are professional engineering standards, not optional suggestions.
Code that violates these principles WILL BE REJECTED.
"""

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
        self.progress_update.emit("⋯ Generating code...")
        try:
            prompt = self._build_prompt(context)
            LOGGER.debug("Built prompt with %d characters", len(prompt))

            plan = self._request_plan(prompt)
            summary = plan.get("summary", "")
            LOGGER.info("Received plan: %s", summary)
            plan_summary = summary.strip() or "No summary provided"
            self.progress_update.emit(f"⋯ {plan_summary}")

            file_ops = self._parse_file_operations(plan.get("files", []), context.working_dir)
            LOGGER.debug("Parsed %d file operations", len(file_ops))

            created, modified = self._apply_files(file_ops, context.working_dir)
            LOGGER.info("Applied files: %d created, %d modified", len(created), len(modified))

            plan_commands = plan.get("commands", [])
            if plan_commands:
                self.progress_update.emit("▶ Running validation...")

            commands, cmd_outputs, cmd_errors = self._execute_commands(
                plan_commands,
                context.working_dir,
            )
            LOGGER.info("Executed %d commands", len(commands))

            outputs.extend(cmd_outputs)
            errors.extend(cmd_errors)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Coding session failed: %s", exc)
            errors.append(str(exc))

        duration = perf_counter() - start
        success = (len(created) + len(modified) > 0) or not errors
        files_count = len(created) + len(modified)
        if success:
            self.progress_update.emit(
                f"✓ Created {len(created)} files, modified {len(modified)} files"
            )

        LOGGER.info(
            "Session completed: success=%s, duration=%.2fs, files=%d, commands=%d",
            success,
            duration,
            files_count,
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

            if existed:
                self.progress_update.emit(f"~ Modifying {rel_path}")
                modified.append(rel_path)
            else:
                self.progress_update.emit(f"+ Creating {rel_path}")
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
        safe_prefixes = ("python", "pip", "pytest", "git", "npm", "node", "poetry", "mkdir", "md")
        blocked_dependency_prefixes = (
            "pip install",
            "pip install -r",
            "python -m pip install",
            "python3 -m pip install",
            "py -m pip install",
            "npm install",
            "poetry install",
        )
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

            if any(cmd_lower.startswith(prefix) for prefix in blocked_dependency_prefixes):
                skip_message = "⏭️ Skipping dependency installation (will run after all sessions)"
                LOGGER.info("Deferred dependency command: %s", cmd)
                self.progress_update.emit(skip_message)
                outputs.append(skip_message)
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
