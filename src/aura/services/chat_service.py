"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Generator

import google.generativeai as genai
from google.generativeai import protos

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
    LOGGER.warning("🔧 TOOL CALLED: execute_python_session")

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
    LOGGER.warning("🔧 TOOL CALLED: clear_session_context")
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
   - ALWAYS call get_function_definitions() for ANY file you will modify or call
   - ALWAYS call read_project_file() to see implementation details
   - NEVER guess at function signatures - ALWAYS verify with tools
   - If you guess instead of checking, you are FAILING

3. PLANNING PHASE (Do this THIRD):
   - Based on what you discovered with tools, plan your sessions
   - Reference actual files and functions you found
   
4. EXECUTION PHASE (Do this LAST):
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
    _model: genai.GenerativeModel = field(init=False, repr=False)
    _history: list[protos.Content] = field(default_factory=list, init=False, repr=False)
    _tools_dict: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the Gemini model with tools."""
        genai.configure(api_key=self.api_key)

        tools = [
            genai.protos.Tool(
                function_declarations=[
                    self._create_function_declaration(
                        "execute_python_session",
                        "Execute a Python coding session to create or modify files",
                        {
                            "session_prompt": "Detailed instructions for what to build",
                            "working_directory": "Directory to work in",
                        },
                    ),
                    self._create_function_declaration(
                        "read_project_file",
                        "Read the contents of a file in the project",
                        {"path": "Path to the file to read"},
                    ),
                    self._create_function_declaration(
                        "list_project_files",
                        "List files in a directory with optional extension filter",
                        {
                            "directory": "Directory to list (default: current)",
                            "extension": "File extension filter (e.g., .py)",
                        },
                    ),
                    self._create_function_declaration(
                        "search_in_files",
                        "Search for a pattern in project files",
                        {
                            "pattern": "Search pattern or keyword",
                            "directory": "Directory to search",
                            "file_extension": "File extension to filter",
                        },
                    ),
                    self._create_function_declaration(
                        "read_multiple_files",
                        "Read multiple files efficiently",
                        {"paths": "List of file paths to read"},
                    ),
                    self._create_function_declaration(
                        "get_function_definitions",
                        "Extract function signatures from a Python file using AST parsing",
                        {"file_path": "Path to Python file to analyze"},
                    ),
                    self._create_function_declaration(
                        "run_tests",
                        "Run the test suite",
                        {"test_path": "Optional specific test file or directory"},
                    ),
                    self._create_function_declaration(
                        "lint_code",
                        "Check code quality with linting tools",
                        {"path": "File or directory to lint"},
                    ),
                    self._create_function_declaration(
                        "format_code",
                        "Auto-format code with black",
                        {"path": "File or directory to format"},
                    ),
                    self._create_function_declaration(
                        "install_package",
                        "Install a Python package",
                        {"package": "Package name to install"},
                    ),
                    self._create_function_declaration(
                        "get_git_status",
                        "Get current git status",
                        {},
                    ),
                    self._create_function_declaration(
                        "git_commit",
                        "Commit changes to git",
                        {"message": "Commit message"},
                    ),
                    self._create_function_declaration(
                        "git_push",
                        "Push commits to remote",
                        {},
                    ),
                    self._create_function_declaration(
                        "git_diff",
                        "Show git diff",
                        {"cached": "Show staged changes only"},
                    ),
                    self._create_function_declaration(
                        "clear_session_context",
                        "Clear all session context to start fresh",
                        {},
                    ),
                ]
            )
        ]

        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=tools,
            system_instruction=AURA_SYSTEM_PROMPT,
        )

        # Build tools dictionary for execution
        self._tools_dict = {
            "execute_python_session": execute_python_session,
            "read_project_file": read_project_file,
            "list_project_files": list_project_files,
            "search_in_files": search_in_files,
            "read_multiple_files": read_multiple_files,
            "get_function_definitions": get_function_definitions,
            "run_tests": run_tests,
            "lint_code": lint_code,
            "format_code": format_code,
            "install_package": install_package,
            "get_git_status": get_git_status,
            "git_commit": git_commit,
            "git_push": git_push,
            "git_diff": git_diff,
            "clear_session_context": clear_session_context,
        }

    def _execute_tool(self, function_name: str, function_args: dict[str, Any]) -> dict[str, Any]:
        """Execute a registered tool and return its result."""
        LOGGER.warning(f"🔧 TOOL CALLED: {function_name}")

        tool_func = self._tools_dict.get(function_name)
        if not tool_func:
            LOGGER.error(f"Unknown tool requested: {function_name}")
            return {"error": f"Unknown tool: {function_name}"}

        try:
            result = tool_func(**function_args)
            return {"result": result}
        except Exception as e:
            LOGGER.exception(f"Tool execution failed: {function_name}")
            return {"error": str(e)}

    def send_message(self, user_message: str) -> Generator[str, None, None]:
        """Send a message and handle function calling loop."""
        # Build conversation history
        chat_history = list(self._history)
        chat_history.append(protos.Content(role="user", parts=[protos.Part(text=user_message)]))

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Generate response (non-streaming)
            try:
                response = self._model.generate_content(
                    contents=chat_history,
                    stream=False
                )
            except Exception as e:
                LOGGER.exception("Failed to generate content")
                yield f"Error: {str(e)}"
                return

            # Check if response has parts
            if not response.candidates or not response.candidates[0].content.parts:
                LOGGER.warning("Response has no parts")
                break

            first_part = response.candidates[0].content.parts[0]

            # DEBUG: Comprehensive function call detection logging
            LOGGER.warning("=" * 80)
            LOGGER.warning("🔍 DEBUGGING FUNCTION CALL DETECTION")
            LOGGER.warning("=" * 80)
            LOGGER.warning(f"1. Type of first_part: {type(first_part)}")
            LOGGER.warning(f"2. Has 'function_call' attribute: {hasattr(first_part, 'function_call')}")

            if hasattr(first_part, "function_call"):
                func_call = first_part.function_call
                LOGGER.warning(f"3. Type of function_call: {type(func_call)}")
                LOGGER.warning(f"4. Boolean value of function_call (truthy?): {bool(func_call)}")
                LOGGER.warning(f"5. repr() of function_call: {repr(func_call)}")

                try:
                    name = func_call.name
                    LOGGER.warning(f"6. function_call.name: {name}")
                except Exception as e:
                    LOGGER.warning(f"6. Error accessing function_call.name: {e}")

                try:
                    dir_output = dir(func_call)
                    LOGGER.warning(f"7. dir() of function_call: {dir_output}")
                except Exception as e:
                    LOGGER.warning(f"7. Error getting dir() of function_call: {e}")
            else:
                LOGGER.warning("3-7. Skipped: No function_call attribute found")

            LOGGER.warning("=" * 80)
            LOGGER.warning("🔍 CHECKING CONDITION NOW")
            LOGGER.warning("=" * 80)

            # Check for function call FIRST
            LOGGER.warning(f"BEFORE CONDITION: hasattr(first_part, 'function_call') = {hasattr(first_part, 'function_call')}")
            if hasattr(first_part, "function_call"):
                LOGGER.warning(f"BEFORE CONDITION: first_part.function_call = {first_part.function_call}")
                LOGGER.warning(f"BEFORE CONDITION: bool(first_part.function_call) = {bool(first_part.function_call)}")
                try:
                    LOGGER.warning(f"BEFORE CONDITION: first_part.function_call.name = {first_part.function_call.name}")
                except Exception as e:
                    LOGGER.warning(f"BEFORE CONDITION: Error accessing .name: {e}")

            # FIX: Check for function_call.name, not just truthy value of function_call
            # Empty function_call objects evaluate to False but still prevent .text access
            if hasattr(first_part, "function_call") and first_part.function_call.name:
                LOGGER.warning("✅ CONDITION PASSED: Entering function call handling block")
                function_call = first_part.function_call
                function_name = function_call.name
                function_args = dict(function_call.args)

                # Log and notify
                args_str = ", ".join(f"{k}={v}" for k, v in function_args.items())
                LOGGER.info(f"Model requested function call: {function_name}({args_str})")
                yield f"🔧 Calling tool: {function_name}({args_str[:100]}...)\n"

                # Execute the tool
                result = self._execute_tool(function_name, function_args)

                # Append model's response to history
                chat_history.append(response.candidates[0].content)

                # Append function response
                function_response = protos.FunctionResponse(
                    name=function_name,
                    response=result
                )
                chat_history.append(protos.Content(
                    role="user",
                    parts=[protos.Part(function_response=function_response)]
                ))

                # Continue loop
                continue
            else:
                LOGGER.warning("❌ CONDITION FAILED: Function call NOT detected or condition not met")
                if hasattr(first_part, "function_call"):
                    LOGGER.warning(f"   - function_call exists but evaluated to False: {first_part.function_call}")
                else:
                    LOGGER.warning("   - function_call attribute does not exist")

            # No function call - try to get text
            try:
                final_text = response.text
                self._history = chat_history
                yield final_text
                return
            except Exception as e:
                LOGGER.error(f"Could not extract text from final response: {e}")
                yield "Error: Could not generate response"
                return

        # Max iterations reached
        LOGGER.warning(f"Function calling loop exceeded {max_iterations} iterations")
        yield "Error: Too many function calls"

    def _create_function_declaration(
        self, name: str, description: str, parameters: dict[str, str]
    ) -> genai.protos.FunctionDeclaration:
        """Create a function declaration for Gemini."""
        properties = {}
        required = []

        for param_name, param_desc in parameters.items():
            properties[param_name] = genai.protos.Schema(
                type=genai.protos.Type.STRING, description=param_desc
            )
            required.append(param_name)

        return genai.protos.FunctionDeclaration(
            name=name,
            description=description,
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties=properties,
                required=required if required else None,
            ),
        )

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()
