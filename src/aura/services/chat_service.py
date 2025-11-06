"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from google import genai
from google.genai import types

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
        self._lock = Lock()

    def get_context(self) -> tuple[str, ...]:
        """Return stored context entries."""
        with self._lock:
            return tuple(self._context)

    def add_entry(self, entry: str) -> None:
        """Append a context entry if it is non-empty."""
        sanitized = entry.strip()
        if not sanitized:
            return
        with self._lock:
            self._context.append(sanitized)

    def clear(self) -> None:
        """Remove all stored context entries."""
        with self._lock:
            self._context.clear()


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
        )

        result = agent.execute_session(context)

        if result.success:
            files = list(result.files_created) + list(result.files_modified)
            ordered_files = list(dict.fromkeys(files))
            files_section = ", ".join(ordered_files) if ordered_files else "none"
            summary_text = (result.summary or "").strip() or "No summary provided"
            context_manager.add_entry(f"Session: {summary_text} | Files: {files_section}")

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

CRITICAL MANDATORY REQUIREMENTS - TOOL USAGE IS NOT OPTIONAL:

YOU MUST USE TOOLS BEFORE GENERATING ANY CODE. This is not a suggestion.

REQUIRED WORKFLOW - Follow this EXACT order every single time:

1. DISCOVERY PHASE (MANDATORY - Do this FIRST):
   - ALWAYS call list_project_files() to see what files exist
   - ALWAYS call search_in_files() to find relevant code patterns
   - If you skip this step, you are FAILING

2. UNDERSTANDING PHASE (MANDATORY - Do this SECOND):
   - ALWAYS call find_definition() for ANY class or function you will use or extend
   - ALWAYS call get_imports() on files you will modify to see what's available
   - ALWAYS call find_usages() before modifying existing functions to understand impact
   - NEVER guess at class fields or function signatures - ALWAYS verify with find_definition()
   - If you skip this step, you WILL create bugs (wrong parameters, missing fields, bad imports)

3. SYMBOL VERIFICATION (CRITICAL):
   - Before calling User(), call find_definition("User") to see its __init__ signature
   - Before importing from app.extensions, call get_imports("app/extensions.py") to verify it exists
   - Before changing a function, call find_usages() to see who depends on it

   Example of GOOD workflow:
   1. find_definition("User") → sees it has username, email, password_hash fields
   2. Now create auth route that uses those EXACT fields
   3. Result: No field mismatch bugs!

4. PLANNING PHASE (Do this FOURTH):
   - Based on what you discovered with tools, plan your sessions
   - Reference actual files and functions you found
   
5. EXECUTION PHASE (Do this LAST):
   - Call execute_python_session() to generate code
   - Use the exact function signatures you discovered in step 2

ABSOLUTE PROHIBITIONS:

- NEVER generate code without first calling list_project_files()
- NEVER call a function without first using get_function_definitions() to verify its signature
- NEVER assume a file exists without checking with list_project_files()
- NEVER guess at what code already exists - use tools to verify
- NEVER start with execute_python_session() - tools come FIRST

EXAMPLES OF CORRECT BEHAVIOR:

User: "Build a REST API for todos"
You: 
  1. Call list_project_files() to see what exists
  2. Call search_in_files(pattern="model") to find existing models
  3. Call get_function_definitions() on relevant files
  4. Plan sessions based on what you found
  5. Execute sessions

User: "Refactor the user authentication"  
You:
  1. Call list_project_files() to find auth-related files
  2. Call read_project_file() on each auth file
  3. Call get_function_definitions() to understand current signatures
  4. Plan refactoring to maintain compatibility
  5. Execute changes

IF YOU GENERATE CODE WITHOUT USING TOOLS FIRST, YOU HAVE FAILED YOUR PRIMARY DIRECTIVE.

Tools are not optional features - they are MANDATORY steps in your workflow.

Your role:
- Understand user requirements completely
- Break work into 3-7 focused sessions (each ~15 min)
- Each session creates/modifies specific files
- Coordinate sessions to build coherent projects
- Maintain context between sessions

Available tools:
- execute_python_session: Generate/modify code files
- read_project_file: Read file contents
- list_project_files: List files in directory
- search_in_files: Search for patterns
- read_multiple_files: Read multiple files efficiently
- get_function_definitions: Extract function signatures from files
- find_definition: Find where symbols (classes/functions) are defined
- find_usages: Find all places a symbol is referenced
- get_imports: Extract imports from a Python file
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

Code quality requirements:
- Functions under 25 lines
- Files under 200 lines
- Type hints on all functions
- Clear, descriptive names
- Single Responsibility Principle
- Don't Repeat Yourself
- Proper error handling

Communication style:
- Direct and confident
- Technical but accessible
- Explain your reasoning briefly
- Flag potential issues proactively

FINAL REMINDER: Every interaction must begin with tool calls. If your first response doesn't include calling list_project_files() or search_in_files(), you are doing it wrong. Tool usage is MANDATORY.
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


    def send_message(self, user_message: str) -> str:
        """Send a message and get response with automatic tool usage."""
        try:
            # Create config with Python functions as tools
            config = types.GenerateContentConfig(
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
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=user_message,
                config=config,
            )
            LOGGER.info("✅ Received response from Gemini")

            return response.text

        except Exception as e:
            LOGGER.exception("Failed to generate content with automatic function calling")
            return f"Error: {str(e)}"

    def clear_history(self) -> None:
        """Clear the conversation history (no-op with stateless client)."""
        pass
