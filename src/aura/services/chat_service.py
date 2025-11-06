"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, Optional, Set

from google import genai
from google.genai import types

from src.aura import config
from src.aura.agents import PythonCoderAgent, SessionContext
from src.aura.tools.file_system_tools import (
    list_project_files,
    read_multiple_files,
    read_project_file,
    search_in_files,
)
from src.aura.tools.git_tools import git_commit, git_diff, git_push, get_git_status
from src.aura.tools.python_tools import (
    format_code,
    get_function_definitions,
    install_package,
    lint_code,
    run_tests,
)
# Symbol resolution tools for understanding code structure
from src.aura.tools.symbol_tools import find_definition, find_usages, get_imports

LOGGER = logging.getLogger(__name__)


class SessionContextManager:
    """Manage shared session context between tool calls."""

    def __init__(self) -> None:
        self._context: list[str] = []
        self._function_signatures: dict[str, list[dict[str, object]]] = {}
        self._lock = Lock()

    def get_context(self) -> tuple[str, ...]:
        """Return stored context entries."""
        with self._lock:
            return tuple(self._context)

    def add_entry(
        self,
        entry: str,
        *,
        files: Iterable[str] | None = None,
        working_dir: Path | None = None,
    ) -> None:
        """Append a context entry and store any discovered function signatures."""
        sanitized = entry.strip()
        if not sanitized:
            return

        if files:
            for file_label in files:
                normalized_label = self._normalize_file_label(file_label)
                if not normalized_label.endswith(".py"):
                    continue
                file_path = Path(normalized_label)
                if not file_path.is_absolute():
                    if working_dir is not None:
                        file_path = working_dir / file_path
                    else:
                        file_path = Path.cwd() / file_path
                self.parse_and_store_file(file_path, display_path=normalized_label)

        with self._lock:
            self._context.append(sanitized)

    def parse_and_store_file(
        self,
        file_path: str | Path,
        *,
        display_path: str | None = None,
    ) -> None:
        """Parse a Python file and cache its function signatures."""
        path = Path(file_path)
        signatures = get_function_definitions(str(path))
        key = display_path or self._normalize_storage_key(path)
        normalized = self._normalize_definitions(signatures)
        with self._lock:
            self._function_signatures[key] = normalized

    def get_function_signatures(self) -> dict[str, list[dict[str, object]]]:
        """Return a deep copy of stored function signatures."""
        with self._lock:
            return {
                name: [
                    {
                        "name": entry.get("name", ""),
                        "params": list(entry.get("params", [])),
                        "line": entry.get("line", 0),
                        "docstring": entry.get("docstring", ""),
                        "return_type": entry.get("return_type", "Any"),
                    }
                    for entry in entries
                ]
                for name, entries in self._function_signatures.items()
            }

    def get_function_signatures_for_files(self, file_list: Iterable[str]) -> str:
        """Return a formatted string of cached signatures for the given files."""
        lines: list[str] = []
        with self._lock:
            for file_name in file_list:
                normalized_name = self._normalize_file_label(file_name)
                entries = self._function_signatures.get(normalized_name)
                if not entries:
                    continue
                lines.append(f"{normalized_name}:")
                for entry in entries:
                    lines.append(f"- {self._format_signature(entry)}")
        if not lines:
            return ""

        header = ["EXACT FUNCTION SIGNATURES FROM PREVIOUS FILES:"]
        footer = ["USE THESE EXACT PARAMETER NAMES. DO NOT GUESS."]
        return "\n".join(header + lines + footer)

    def clear(self) -> None:
        """Remove all stored context entries."""
        with self._lock:
            self._context.clear()
            self._function_signatures.clear()

    @staticmethod
    def _normalize_file_label(label: str) -> str:
        """Strip status annotations like ' (updated)' from file labels."""
        cleaned = label.strip()
        if cleaned.endswith(" (updated)"):
            return cleaned[: -len(" (updated)")]
        return cleaned

    @staticmethod
    def _normalize_storage_key(path: Path) -> str:
        """Return a storage key for a given file path."""
        try:
            return path.as_posix()
        except Exception:  # noqa: BLE001
            return str(path)

    @staticmethod
    def _normalize_definitions(
        signatures: Iterable[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Normalize signature dictionaries for safe storage."""
        normalized: list[dict[str, object]] = []
        if signatures is None:
            return normalized
        for item in signatures:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            params_raw = item.get("params", [])
            if isinstance(params_raw, (list, tuple)):
                params = [str(param) for param in params_raw]
            else:
                params = [str(params_raw)]
            docstring = str(item.get("docstring", ""))
            line_number = item.get("line", 0)
            try:
                line = int(line_number)
            except (TypeError, ValueError):
                line = 0
            return_type = item.get("return_type")
            normalized.append(
                {
                    "name": name,
                    "params": params,
                    "line": line,
                    "docstring": docstring,
                    "return_type": str(return_type) if return_type else "Any",
                }
            )
        return normalized

    @staticmethod
    def _format_signature(entry: dict[str, object]) -> str:
        """Format a signature dictionary for display."""
        name = str(entry.get("name", ""))
        params = entry.get("params", [])
        if isinstance(params, (list, tuple)):
            formatted_params = ", ".join(f"{param}: Any" for param in params)
        else:
            formatted_params = f"{params}: Any" if params else ""
        return_type = str(entry.get("return_type", "Any"))
        return f"{name}({formatted_params}) -> {return_type}"


_SESSION_CONTEXT_MANAGER: SessionContextManager | None = None


def get_session_context_manager() -> SessionContextManager:
    """Return the singleton session context manager."""
    global _SESSION_CONTEXT_MANAGER
    if _SESSION_CONTEXT_MANAGER is None:
        _SESSION_CONTEXT_MANAGER = SessionContextManager()
    return _SESSION_CONTEXT_MANAGER


def execute_python_session(session_prompt: str, working_directory: str) -> dict[str, object]:
    """Execute a Python coding session using PythonCoderAgent."""
    LOGGER.info("🔧 TOOL CALLED: execute_python_session(%s)", session_prompt)

    context_manager = get_session_context_manager()

    try:
        agent = PythonCoderAgent(api_key=os.getenv("GEMINI_API_KEY", ""))
    except ValueError as exc:
        LOGGER.error("Failed to create PythonCoderAgent: %s", exc)
        return {
            "success": False,
            "summary": "",
            "files_created": [],
            "files_modified": [],
            "errors": [str(exc)],
        }

    try:
        working_dir = Path(working_directory) if working_directory else Path.cwd()
        project_files = list_project_files(str(working_dir))

        context = SessionContext(
            working_dir=working_dir,
            session_prompt=session_prompt,
            previous_work=list(context_manager.get_context()),
            project_files=project_files,
            function_signatures=context_manager.get_function_signatures(),
        )

        result = agent.execute_session(context)

        if result.success:
            files = list(result.files_created) + list(result.files_modified)
            ordered_files = list(dict.fromkeys(files))
            files_section = ", ".join(ordered_files) if ordered_files else "none"
            summary_text = (result.summary or "").strip() or "No summary provided"
            context_manager.add_entry(
                f"Session: {summary_text} | Files: {files_section}",
                files=ordered_files,
                working_dir=working_dir,
            )

        return {
            "success": result.success,
            "summary": result.summary,
            "files_created": list(result.files_created),
            "files_modified": list(result.files_modified),
            "commands_run": list(result.commands_run),
            "output_lines": list(result.output_lines),
            "errors": list(result.errors),
            "duration_seconds": result.duration_seconds,
        }
    except Exception as exc:
        LOGGER.exception("Python coding session failed: %s", exc)
        return {
            "success": False,
            "summary": "",
            "files_created": [],
            "files_modified": [],
            "commands_run": [],
            "output_lines": [],
            "errors": [f"Session execution failed: {exc}"],
        }


def clear_session_context() -> dict[str, str]:
    """Clear all session context to start fresh."""
    LOGGER.info("🔧 TOOL CALLED: clear_session_context")
    context_manager = get_session_context_manager()
    context_manager.clear()
    return {"status": "Session context cleared successfully"}


AURA_SYSTEM_PROMPT = """
You are Aura, an AI orchestration system that coordinates coding agents to build software projects.

═══════════════════════════════════════════════════════════════════════════════
MANDATORY 4-STEP PROTOCOL - TOOL USAGE IS REQUIRED, NOT OPTIONAL
═══════════════════════════════════════════════════════════════════════════════

ABSOLUTE RULE: You MUST use tools to gather context BEFORE planning or generating code.
Skipping tool usage is a CRITICAL FAILURE. This is not negotiable.

STEP 1: DISCOVERY (ALWAYS DO THIS FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED ACTIONS:
✓ Call list_project_files() to understand directory structure
✓ Identify key files: models, routes, schemas, tests, configs, utilities
✓ Call search_in_files() to find existing patterns relevant to the request

PURPOSE: You cannot plan intelligently without knowing what already exists.

VALIDATION: If your response doesn't include list_project_files() as your FIRST tool call,
you have FAILED this step.

STEP 2: READ EXISTING CODE (MANDATORY FOR MODIFICATIONS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IF modifying existing code (keywords: add, update, modify, change, edit, fix, include, extend):
✓ Call read_multiple_files() on ALL relevant Python files
✓ Call read_project_file() on individual files if needed
✓ You CANNOT plan modifications without reading what exists first

PURPOSE: Blind modifications create bugs. You must understand the current implementation.

VALIDATION: If you are modifying code but haven't called read_multiple_files() or
read_project_file(), you WILL create bugs. This is guaranteed.

STEP 3: SYMBOL RESOLUTION (CRITICAL FOR MODIFICATIONS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHEN MODIFYING EXISTING ENTITIES (classes, functions, models, routes):
✓ Call find_definition(symbol_name) to see exact signatures and fields
✓ Call find_usages(symbol_name) to understand where the symbol is referenced
✓ Call get_imports(file_path) to verify what modules/dependencies are available
✓ Call get_function_definitions(file_path) to see all function signatures in a file

EXAMPLES OF REQUIRED BEHAVIOR:

Request: "Add email field to User model"
REQUIRED WORKFLOW:
1. list_project_files() → find models
2. read_project_file("models/user.py") → read current User model
3. find_definition("User") → see exact __init__ signature and existing fields
4. find_usages("User") → see where User is instantiated (to understand impact)
5. get_imports("models/user.py") → see available dependencies
6. NOW you can plan to add email field with correct syntax

Request: "Modify login route to return tokens"
REQUIRED WORKFLOW:
1. list_project_files() → find route files
2. search_in_files(pattern="login") → locate login implementation
3. read_project_file("routes/auth.py") → read current route
4. find_definition("login") → see exact signature
5. find_usages("login") → see who calls it
6. get_function_definitions("routes/auth.py") → see helper functions
7. NOW you can modify the route properly

PURPOSE: You MUST know the exact signatures, fields, and usage patterns before modifying.
Guessing leads to:
- Wrong parameter names
- Missing required fields
- Broken imports
- Incompatible changes

VALIDATION: If you are modifying an entity (User, login, etc.) but haven't called
find_definition() on it, you are doing it WRONG.

STEP 4: PLANNING & EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ONLY AFTER completing Steps 1-3:
✓ Plan your sessions based on ACTUAL discovered code (not assumptions)
✓ Reference the exact files, functions, and symbols you found with tools
✓ Call execute_python_session() with prompts that use discovered context
✓ Use the EXACT signatures and fields you verified in Step 3

═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS
═══════════════════════════════════════════════════════════════════════════════

❌ NEVER execute_python_session() without first calling discovery tools
❌ NEVER modify code without calling find_definition() on modified entities
❌ NEVER assume file locations - verify with list_project_files()
❌ NEVER guess at function signatures - verify with find_definition()
❌ NEVER skip read_project_file() when modifying existing files
❌ NEVER ignore find_usages() when changing existing functions/classes

VIOLATION CONSEQUENCES: If you skip mandatory tool calls, you WILL create:
- Field mismatch bugs (wrong __init__ parameters)
- Import errors (wrong module paths)
- Breaking changes (incompatible signatures)
- Missing dependencies (unverified imports)

═══════════════════════════════════════════════════════════════════════════════
WORKFLOW EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

✅ CORRECT: "Build a REST API for todos"
1. list_project_files() → see project structure
2. search_in_files(pattern="model") → find existing models to maintain consistency
3. search_in_files(pattern="route") → find existing routes to match patterns
4. get_function_definitions("models/base.py") → understand base model
5. Plan sessions based on discovered patterns
6. execute_python_session() with context-aware prompts

✅ CORRECT: "Add role field to User model"
1. list_project_files() → locate User model
2. read_project_file("models/user.py") → read current implementation
3. find_definition("User") → see exact __init__(username, email, password_hash)
4. find_usages("User") → find User("john", "john@example.com", hash) calls
5. get_imports("models/user.py") → see available types/modules
6. Plan to add role parameter with default value for backward compatibility
7. execute_python_session() with prompt: "Add optional role field with default='user'"

❌ WRONG: "Add role to User"
Immediately calls execute_python_session() without discovering:
- Where User is defined
- What fields User currently has
- How User is instantiated throughout the codebase
Result: Creates User(username, email, password_hash, role) breaking all existing calls

═══════════════════════════════════════════════════════════════════════════════

Your role:
- Understand user requirements completely through intelligent tool usage
- Break work into 3-7 focused sessions (each ~15 min)
- Each session creates/modifies specific files with context awareness
- Coordinate sessions to build coherent, maintainable projects
- Maintain context between sessions

Available tools:
- list_project_files: List files in directory (ALWAYS CALL FIRST)
- search_in_files: Search for patterns in codebase
- read_project_file: Read single file contents
- read_multiple_files: Read multiple files efficiently (USE THIS for modifications)
- find_definition: Find where symbols are defined (MANDATORY for modifications)
- find_usages: Find all references to a symbol (MANDATORY for modifications)
- get_imports: Extract imports from a Python file (verify dependencies)
- get_function_definitions: Extract function signatures from files
- execute_python_session: Generate/modify code (ONLY after gathering context)
- run_tests: Execute test suite
- lint_code: Check code quality
- format_code: Auto-format code
- install_package: Install Python packages
- get_git_status: Check git status
- git_commit: Commit changes
- git_push: Push to remote
- git_diff: Show git diff
- clear_session_context: Clear session history

Session design principles:
- Each session has a clear, single purpose
- Sessions build on each other logically
- Early sessions create foundations (models, configs)
- Middle sessions add features
- Final sessions add tests and polish
- Each session should complete in ~15 minutes

Communication style:
- Direct and confident
- Technical but accessible
- Explain your reasoning briefly
- Flag potential issues proactively

═══════════════════════════════════════════════════════════════════════════════
FINAL VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Before responding, ask yourself:
1. Did I call list_project_files() first? (Required: YES)
2. Am I modifying code? If YES → Did I read the files? (Required: YES)
3. Am I modifying a symbol? If YES → Did I call find_definition()? (Required: YES)
4. Did I verify with tools before planning? (Required: YES)

If any answer is NO, you have FAILED the mandatory protocol.

Tool usage is not optional. It is REQUIRED. Every. Single. Time.
""".strip()


@dataclass
class ChatService:
    """Manages conversational interactions with developer tool access."""

    api_key: str
    model_name: str = "gemini-2.5-pro"
    _client: genai.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Gemini client."""
        self._client = genai.Client(api_key=self.api_key)


    def send_message(
        self,
        user_message: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Send a message and stream the response with automatic tool usage."""
        try:
            # Create config with Python functions as tools
            request_config = types.GenerateContentConfig(
                tools=[
                    list_project_files,
                    search_in_files,
                    read_project_file,
                    read_multiple_files,
                    get_function_definitions,
                    run_tests,
                    lint_code,
                    format_code,
                    install_package,
                    get_git_status,
                    git_commit,
                    git_push,
                    git_diff,
                    find_definition,
                    find_usages,
                    get_imports,
                    clear_session_context,
                    execute_python_session,
                ],
                system_instruction=AURA_SYSTEM_PROMPT,
            )

            # The SDK automatically:
            # - Detects function calls
            # - Executes the functions
            # - Sends results back to model
            # - Repeats until model returns text
            LOGGER.info("🤖 Sending message to Gemini with 18 tools available")
            stream = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=user_message,
                config=request_config,
            )

            try:
                iterator = iter(stream)
            except TypeError:
                LOGGER.debug("Streaming unavailable; falling back to blocking response")
                fallback_response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=request_config,
                )
                final_text = fallback_response.text or ""
                if final_text and on_chunk:
                    on_chunk(f"{config.STREAM_PREFIX}{final_text}")
                    on_chunk(f"{config.STREAM_PREFIX}\n")
                LOGGER.info("✅ Received response from Gemini")
                return final_text

            aggregated_text = ""
            collected_parts: list[str] = []
            seen_calls: Set[str] = set()
            streamed_any = False

            for chunk in iterator:
                streamed_any = (
                    self._emit_function_calls(chunk, on_chunk, seen_calls) or streamed_any
                )
                addition, aggregated_text = self._extract_stream_addition(chunk, aggregated_text)
                if addition:
                    collected_parts.append(addition)
                    if on_chunk:
                        on_chunk(f"{config.STREAM_PREFIX}{addition}")
                    streamed_any = True

            final_text = "".join(collected_parts) or aggregated_text
            if not final_text and hasattr(stream, "text"):
                final_text = getattr(stream, "text") or ""

            if streamed_any and on_chunk:
                on_chunk(f"{config.STREAM_PREFIX}\n")

            LOGGER.info("✅ Received response from Gemini")

            return final_text

        except Exception as e:
            LOGGER.exception("Failed to generate content with automatic function calling")
            return f"Error: {str(e)}"

    @staticmethod
    def _extract_stream_addition(
        chunk: types.GenerateContentResponse,
        aggregated_text: str,
    ) -> tuple[str, str]:
        """Return the new text produced by this chunk."""
        text = getattr(chunk, "text", "") or ""
        if not text:
            return "", aggregated_text

        if aggregated_text and aggregated_text.startswith(text):
            # Received a shorter chunk that is already included in the aggregate.
            return "", aggregated_text

        if text.startswith(aggregated_text):
            addition = text[len(aggregated_text) :]
            return addition, text

        # Fallback: assume chunk.text itself is the new addition
        return text, aggregated_text + text

    def _emit_function_calls(
        self,
        chunk: types.GenerateContentResponse,
        on_chunk: Optional[Callable[[str], None]],
        seen_calls: Set[str],
    ) -> bool:
        """Emit tool call notifications for newly observed function calls."""
        function_calls = getattr(chunk, "function_calls", None)
        if not function_calls:
            return False

        emitted = False
        for call in function_calls:
            name = getattr(call, "name", "")
            args_summary = self._summarize_args(getattr(call, "args", None))
            signature = f"{name}:{args_summary}"
            if signature in seen_calls:
                continue
            seen_calls.add(signature)
            if on_chunk:
                on_chunk(f"TOOL_CALL::{name}::{args_summary}")
            emitted = True
        return emitted

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

    def clear_history(self) -> None:
        """Clear the conversation history (no-op with stateless client)."""
        pass
