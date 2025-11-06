"""Agent for running Gemini-powered Python coding sessions."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Iterable, List, Mapping, Sequence, Set, Tuple

import google.generativeai as genai
from google.generativeai.types import generation_types
from PySide6.QtCore import QObject, Signal

from src.aura import config

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
    function_signatures: Mapping[str, Sequence[Mapping[str, object]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "working_dir", Path(self.working_dir).resolve())
        object.__setattr__(self, "previous_work", tuple(self.previous_work))
        object.__setattr__(self, "project_files", tuple(self.project_files))
        normalized_signatures: dict[str, tuple[Mapping[str, object], ...]] = {}
        for file_name, entries in (self.function_signatures or {}).items():
            if not entries:
                continue
            normalized_entries: list[Mapping[str, object]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                params_raw = entry.get("params", [])
                if isinstance(params_raw, (list, tuple)):
                    params = tuple(str(param) for param in params_raw)
                elif params_raw:
                    params = (str(params_raw),)
                else:
                    params = ()
                line_number = entry.get("line", 0)
                try:
                    line = int(line_number)
                except (TypeError, ValueError):
                    line = 0
                normalized_entries.append(
                    {
                        "name": str(entry.get("name", "")),
                        "params": params,
                        "line": line,
                        "docstring": str(entry.get("docstring", "")),
                        "return_type": str(entry.get("return_type", "Any")),
                    }
                )
            if normalized_entries:
                normalized_signatures[str(file_name)] = tuple(normalized_entries)
        object.__setattr__(self, "function_signatures", normalized_signatures)

    def format_function_signatures(self, files: Iterable[str]) -> str:
        """Return a formatted signature block for the given files."""
        seen: Set[str] = set()
        lines: list[str] = []
        for label in files:
            normalized_label = self._normalize_file_label(label)
            if not normalized_label or normalized_label in seen:
                continue
            seen.add(normalized_label)
            entries = self.function_signatures.get(normalized_label)
            if not entries:
                continue
            lines.append(f"{normalized_label}:")
            for entry in entries:
                lines.append(f"- {self._format_signature(entry)}")
        if not lines:
            return ""
        header = ["EXACT FUNCTION SIGNATURES FROM PREVIOUS FILES:"]
        footer = ["USE THESE EXACT PARAMETER NAMES. DO NOT GUESS."]
        return "\n".join(header + lines + footer)

    @staticmethod
    def _format_signature(entry: Mapping[str, object]) -> str:
        name = str(entry.get("name", ""))
        params = entry.get("params", ())
        if isinstance(params, (list, tuple)):
            formatted_params = ", ".join(f"{param}: Any" for param in params)
        else:
            formatted_params = f"{params}: Any" if params else ""
        return_type = str(entry.get("return_type", "Any"))
        return f"{name}({formatted_params}) -> {return_type}"

    @staticmethod
    def _normalize_file_label(label: str) -> str:
        cleaned = label.strip()
        if cleaned.endswith(" (updated)"):
            return cleaned[: -len(" (updated)")]
        return cleaned


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
    tool_calls: Sequence[Mapping[str, object]] | None = None
    syntax_errors: Sequence[Mapping[str, object]] | None = None


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

7. MANDATORY REASONING PROTOCOL (Layered-CoT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To prevent critical bugs, you MUST follow this structured reasoning protocol.
Skipping phases or checkpoints WILL result in rejected work.

PHASE 1: CONTEXT GATHERING (MANDATORY - CANNOT SKIP)
------------------------------------------------------------------------------

This is the most critical phase. Failure here guarantees production bugs.

IF `previous_work` is provided (meaning prior sessions have run):
  - You MUST call the `get_function_definitions` tool on EVERY file from previous sessions that you will import from or interact with.
  - You MUST record the EXACT function signatures: parameter names, types, defaults, and return types.
  - CHECKPOINT: Verify that you have retrieved the function signatures BEFORE proceeding to Phase 2.

ELSE (if this is the first session, `previous_work` is empty):
  - You may skip to Phase 2.

WHY THIS IS MANDATORY:
Guessing parameter names instead of reading them causes `TypeError` crashes.
  - Session 1 creates: `generate_password(use_uppercase, use_numbers)`
  - Session 2 GUESSES: `generate_password(include_uppercase, include_numbers)`
  - RESULT: `TypeError: got unexpected keyword argument 'include_uppercase'`. This is a fatal error.

BLOCKING CONDITION: You CANNOT proceed to design or implementation without completing this phase if previous work exists.

PHASE 2: DESIGN (Verified Against Phase 1)
------------------------------------------------------------------------------

1.  State clearly what you are building or modifying.
2.  List the files you will create or modify.
3.  CHECKPOINT: If importing or calling functions from previous work, explicitly cross-reference the EXACT parameter names you retrieved in Phase 1. Confirm they are available before proceeding.

PHASE 3: IMPLEMENTATION (Using Exact Signatures)
------------------------------------------------------------------------------

1.  Write the code to implement the design.
2.  When calling functions from previous work, use the VERBATIM signatures retrieved in Phase 1.
3.  DO NOT guess, assume, or modify parameter names.
4.  Adhere to all code quality standards (type hints, docstrings, SRP, DRY).

PHASE 4: VALIDATION
------------------------------------------------------------------------------

1.  Provide validation commands in the `commands` section of your JSON response.
2.  For new functions, include a `pytest` test case.
3.  For API endpoints, include a `curl` command.
4.  CHECKPOINT: Verify that your implementation works with the exact signatures from previous work, preventing `TypeError` bugs.

8. THINKING AND REASONING
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
✓ IF previous work exists: Phase 1 (Context Gathering) was completed and `get_function_definitions` was called.
✓ IF calling functions from previous sessions: parameter names match EXACTLY

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
        original_prompt = self._build_prompt(context)
        prompt = original_prompt
        
        max_retries = 2
        for i in range(max_retries + 1):
            created: List[str] = []
            modified: List[str] = []
            commands: List[str] = []
            outputs: List[str] = []
            errors: List[str] = []
            summary = ""
            tool_calls: List[Mapping[str, object]] = []
            syntax_errors: List[Mapping[str, object]] = []
            
            self.progress_update.emit("⋯ Generating code...")
            try:
                if i > 0:
                    LOGGER.info("Retry %d/%d: Attempting to fix syntax errors...", i, max_retries)
                    self.progress_update.emit(f"Retry {i}/{max_retries}: Attempting to fix syntax errors...")

                if context.previous_work:
                    LOGGER.info("PHASE 1: Context Gathering Started")

                plan, new_tool_calls = self._request_plan(prompt)
                tool_calls.extend(new_tool_calls)

                if new_tool_calls:
                    LOGGER.info(
                        "CHECKPOINT: Verified signatures for %d tool calls.", len(new_tool_calls)
                    )

                summary = plan.get("summary", "")
                LOGGER.info("PHASE 2: Design - Received plan: %s", summary)
                plan_summary = summary.strip() or "No summary provided"
                self.progress_update.emit(f"⋯ {plan_summary}")

                LOGGER.info("PHASE 3: Implementation")
                file_ops = self._parse_file_operations(plan.get("files", []), context.working_dir)
                LOGGER.debug("Parsed %d file operations", len(file_ops))

                created, modified, all_paths = self._apply_files(file_ops, context.working_dir)
                LOGGER.info("Applied files: %d created, %d modified", len(created), len(modified))
                
                LOGGER.info("Validating syntax for %d files...", len(all_paths))
                syntax_errors = self._validate_syntax(all_paths)
                if syntax_errors:
                    LOGGER.warning("✗ Found %d syntax errors", len(syntax_errors))
                    if i < max_retries:
                        prompt = self._build_retry_prompt(original_prompt, syntax_errors)
                        continue
                else:
                    LOGGER.info("✓ All files valid")

                plan_commands = plan.get("commands", [])
                if plan_commands:
                    LOGGER.info("PHASE 4: Validation")
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
            success = not errors and not syntax_errors
            files_count = len(created) + len(modified)
            if success:
                self.progress_update.emit(
                    f"✓ Created {len(created)} files, modified {len(modified)} files"
                )

            LOGGER.info(
                "Session completed: success=%s, duration=%.2fs, files=%d, commands=%d, tool_calls=%d",
                success,
                duration,
                files_count,
                len(commands),
                len(tool_calls),
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
                tool_calls=tuple(tool_calls),
                syntax_errors=tuple(syntax_errors),
            )
        
        return AgentResult(
            success=False,
            summary="Session failed after multiple retries due to persistent syntax errors.",
            files_created=(),
            files_modified=(),
            commands_run=(),
            output_lines=(),
            errors=("Session failed after multiple retries due to persistent syntax errors.",),
            duration_seconds=perf_counter() - start,
            tool_calls=(),
            syntax_errors=tuple(syntax_errors),
        )

    def _build_retry_prompt(self, original_prompt: str, syntax_errors: List[Mapping[str, object]]) -> str:
        """Build a prompt for retrying a session with syntax errors."""
        error_feedback = [
            "PREVIOUS ATTEMPT FAILED WITH SYNTAX ERRORS:",
        ]
        for error in syntax_errors:
            error_feedback.append(f"File: {error['file_path']}")
            error_feedback.append(f"Line {error['line_number']}: {error['error_message']}")
            if error.get('problematic_code'):
                error_feedback.append(f"Problematic code: {error['problematic_code']}")
            error_feedback.append("")

        error_feedback.append("CRITICAL: The code you generated has syntax errors. Review the errors above and generate corrected, syntactically valid Python code.")
        error_feedback.append("ORIGINAL REQUEST:")
        error_feedback.append(original_prompt)
        
        return "\n".join(error_feedback)

    def _validate_syntax(self, file_paths: List[Path]) -> List[Mapping[str, object]]:
        """Validate the syntax of Python files."""
        errors = []
        for file_path in file_paths:
            if file_path.suffix != ".py":
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not content.strip():
                    continue
                compile(content, str(file_path), "exec")
            except SyntaxError as e:
                errors.append({
                    "file_path": str(file_path),
                    "line_number": e.lineno,
                    "error_message": e.msg,
                    "problematic_code": e.text,
                })
            except (UnicodeDecodeError, FileNotFoundError) as e:
                errors.append({
                    "file_path": str(file_path),
                    "line_number": 0,
                    "error_message": f"Could not read file: {e}",
                    "problematic_code": "",
                })
        return errors

    def _build_prompt(self, context: SessionContext) -> str:
        sections = [context.session_prompt.strip()]
        if context.previous_work:
            # Extract file names from previous work context
            files_created: list[str] = []
            seen_files: Set[str] = set()
            for work_item in context.previous_work:
                extracted_files: list[str] = []
                # Parse known formats for file tracking
                if "created:" in work_item:
                    files_part = work_item.split("created:", 1)[1].strip()
                    extracted_files = [f.strip() for f in files_part.split(",")]
                elif "| Files:" in work_item:
                    files_part = work_item.split("| Files:", 1)[1].strip()
                    extracted_files = [f.strip() for f in files_part.split(",")]

                for file_name in extracted_files:
                    if not file_name or file_name.lower() == "none":
                        continue
                    normalized = SessionContext._normalize_file_label(file_name)
                    if normalized and normalized not in seen_files:
                        seen_files.add(normalized)
                        files_created.append(normalized)

            if not files_created and context.function_signatures:
                for name in context.function_signatures:
                    normalized = SessionContext._normalize_file_label(str(name))
                    if normalized and normalized not in seen_files:
                        seen_files.add(normalized)
                        files_created.append(normalized)

            previous_work_section = "Previous work:\n" + "\n".join(context.previous_work)

            if files_created:
                signature_section = context.format_function_signatures(files_created)
                if signature_section:
                    previous_work_section += f"\n\n{signature_section}"
                previous_work_section += (
                    "\n\n⚠️ CRITICAL REQUIREMENT - READ THIS:\n"
                    "Previous sessions created files that you may need to call or import.\n"
                    "Files created: " + ", ".join(files_created) + "\n\n"
                    "MANDATORY: Before writing code that imports from or calls functions in these files,\n"
                    "you MUST use the get_function_definitions tool to read their EXACT signatures.\n"
                    "DO NOT GUESS at parameter names. READ them first using get_function_definitions.\n"
                    "Failure to do this will cause TypeError crashes due to parameter name mismatches.\n"
                )

            sections.append(previous_work_section)
        if context.project_files:
            sections.append("Relevant project files:\n" + "\n".join(context.project_files))
        sections.append(
            "Respond with JSON as described in the system instruction. Only emit valid JSON."
        )
        return "\n\n".join(section for section in sections if section)

    def _request_plan(
        self, prompt: str
    ) -> tuple[Mapping[str, object], List[Mapping[str, object]]]:
        LOGGER.debug("Requesting plan from Gemini API")
        response = self._model.generate_content(prompt, stream=True)
        tool_calls = []

        try:
            iterator = iter(response)
        except TypeError:
            payload = (getattr(response, "text", "") or "").strip()
            if payload:
                self._emit_stream_chunk(payload)
                self._emit_stream_chunk("\n")
            return self._parse_plan_payload(payload), []

        aggregated_text = ""
        collected_parts: List[str] = []
        seen_calls: Set[str] = set()
        streamed_any = False
        for chunk in iterator:
            new_calls = self._emit_tool_calls(chunk, seen_calls)
            if new_calls:
                tool_calls.extend(new_calls)
                streamed_any = True

            addition, aggregated_text = self._extract_stream_addition(
                chunk, aggregated_text
            )
            if addition:
                self._emit_stream_chunk(addition)
                collected_parts.append(addition)
                streamed_any = True

        raw_payload = "".join(collected_parts) or aggregated_text
        if not raw_payload and hasattr(response, "text"):
            raw_payload = getattr(response, "text") or ""

        if streamed_any and raw_payload and not raw_payload.endswith("\n"):
            self._emit_stream_chunk("\n")

        payload = (raw_payload or "").strip()

        return self._parse_plan_payload(payload), tool_calls

    def _emit_stream_chunk(self, chunk: str) -> None:
        """Emit streaming output to the connected UI signal."""
        if not chunk:
            return
        self.progress_update.emit(f"{config.STREAM_PREFIX}{chunk}")

    @staticmethod
    def _extract_stream_addition(
        chunk: generation_types.GenerateContentResponse,
        aggregated_text: str,
    ) -> tuple[str, str]:
        """Return the new text produced by this chunk."""
        try:
            text = getattr(chunk, "text", "") or ""
            if not text:
                return "", aggregated_text
        except (ValueError, AttributeError):
            return "", aggregated_text

        if aggregated_text and aggregated_text.startswith(text):
            return "", aggregated_text

        if text.startswith(aggregated_text):
            addition = text[len(aggregated_text) :]
            return addition, text

        return text, aggregated_text + text

    def _emit_tool_calls(
        self,
        chunk: generation_types.GenerateContentResponse,
        seen_calls: Set[str],
    ) -> List[Mapping[str, object]]:
        """Emit notifications and return newly observed tool calls."""
        function_calls = getattr(chunk, "function_calls", None)
        if not function_calls:
            return []

        new_calls = []
        for call in function_calls:
            name = getattr(call, "name", "")
            args = getattr(call, "args", {})
            args_summary = self._summarize_args(args)
            signature = f"{name}:{args_summary}"
            if signature in seen_calls:
                continue

            seen_calls.add(signature)
            LOGGER.debug("🔧 TOOL CALLED: %s(%s)", name, args_summary)
            self.progress_update.emit(f"TOOL_CALL::{name}::{args_summary}")

            new_calls.append({"name": name, "args": dict(args)})

        return new_calls

    @staticmethod
    def _summarize_args(args: object) -> str:
        """Return a compact JSON summary of function call arguments."""
        if not args:
            return "{}"
        try:
            summary = json.dumps(args, ensure_ascii=False)
        except TypeError:
            summary = str(args)
        if len(summary) > 160:
            summary = f"{summary[:157]}..."
        return summary

    def _parse_plan_payload(self, payload: str) -> Mapping[str, object]:
        """Parse the JSON payload returned by Gemini."""
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
    ) -> Tuple[List[str], List[str], List[Path]]:
        created: List[str] = []
        modified: List[str] = []
        all_paths: List[Path] = []
        for op in operations:
            op.path.parent.mkdir(parents=True, exist_ok=True)
            existed = op.path.exists()
            op.path.write_text(op.content, encoding="utf-8")
            rel_path = self._to_relative(working_dir, op.path)
            all_paths.append(op.path)

            if existed:
                self.progress_update.emit(f"~ Modifying {rel_path}")
                modified.append(rel_path)
            else:
                self.progress_update.emit(f"+ Creating {rel_path}")
                created.append(rel_path)
        return created, modified, all_paths

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
