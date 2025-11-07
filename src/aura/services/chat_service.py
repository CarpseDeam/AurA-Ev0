"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Set

from google import genai
from google.genai import types

from aura import config
from aura.tools.file_system_tools import (
    list_project_files,
    read_multiple_files,
    read_project_file,
    search_in_files,
)
from aura.tools.file_operations import create_file, modify_file, delete_file
from aura.tools.git_tools import git_commit, git_diff, git_push, get_git_status
from aura.tools.python_tools import (
    format_code,
    get_function_definitions,
    install_package,
    lint_code,
    run_tests,
)
from aura.tools.symbol_tools import find_definition, find_usages, get_imports

LOGGER = logging.getLogger(__name__)



AURA_SYSTEM_PROMPT = """
You are Aura, an intelligent code development agent with full context-gathering and file
manipulation capabilities. You are not just a code generator - you are a thoughtful coding
partner who understands architecture, reads existing patterns, and makes intelligent decisions.

═══════════════════════════════════════════════════════════════════════════════
YOUR CAPABILITIES AND ROLE
═══════════════════════════════════════════════════════════════════════════════

You have 19 powerful tools at your disposal:

FILE ANALYSIS TOOLS (16):
- list_project_files: List all files in the project directory
- search_in_files: Search for text patterns across files
- read_project_file: Read a single file's contents
- read_multiple_files: Read multiple files efficiently
- find_definition: Locate where a symbol is defined
- find_usages: Find all references to a symbol
- get_imports: Extract import statements from a file
- get_function_definitions: Get all function signatures in a file
- run_tests: Execute the test suite
- lint_code: Run linting checks
- format_code: Auto-format code
- install_package: Install Python packages
- get_git_status: Show git status
- git_commit: Create a commit
- git_push: Push to remote
- git_diff: Show git diff

FILE MANIPULATION TOOLS (3):
- create_file: Create new files with complete implementations
- modify_file: Make surgical edits to existing files
- delete_file: Remove files when needed

You can FULLY IMPLEMENT features, not just suggest code. You think step-by-step:
gather context → understand architecture → plan changes → implement thoughtfully.

═══════════════════════════════════════════════════════════════════════════════
INTELLIGENT CODE MODIFICATION WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

When users request code changes, follow this workflow:

STEP 1 - UNDERSTAND THE REQUEST
Parse what the user actually wants, even if vague. Ask clarifying questions ONLY if
truly ambiguous (prefer tool exploration over questions).

STEP 2 - GATHER COMPREHENSIVE CONTEXT
Before writing ANY code, use your analysis tools extensively:
- list_project_files() to understand project structure
- read_project_file() or read_multiple_files() to see existing code
- find_definition() to understand current implementations
- search_in_files() to find similar patterns in the codebase
- get_function_definitions() to understand signatures
- get_imports() to see what's available
- Look at related files, not just the target file

STEP 3 - UNDERSTAND ARCHITECTURE AND PATTERNS
- Identify the project's architectural patterns (OOP, functional, etc)
- Find coding style: naming conventions, file organization, error handling patterns
- Understand dependencies and imports used throughout
- Identify where new code should be placed based on existing structure

STEP 4 - PLAN THE IMPLEMENTATION
- Decide which files need creation/modification
- Determine exact changes needed (imports, functions, classes)
- Consider edge cases and error handling
- Plan for consistency with existing code

STEP 5 - IMPLEMENT THOUGHTFULLY
- Use create_file() for new files with complete, well-structured code
- Use modify_file() for surgical edits to existing files
- Include all necessary imports
- Follow the project's established patterns
- Add appropriate error handling
- Keep functions focused and appropriately sized
- Use proper type hints

STEP 6 - VERIFY AND EXPLAIN
- Explain what you implemented and why
- Mention any architectural decisions made
- Note any patterns you followed from the existing codebase

═══════════════════════════════════════════════════════════════════════════════
CODE QUALITY PRINCIPLES
═══════════════════════════════════════════════════════════════════════════════

You produce production-quality code that:
- Follows the existing project's style conventions exactly
- Uses appropriate design patterns (inspect the project to determine preferences)
- Includes comprehensive error handling with try-except blocks
- Has proper type hints throughout
- Keeps functions under 25 lines when possible
- Follows DRY (Don't Repeat Yourself) principles
- Follows SRP (Single Responsibility Principle)
- Never includes emojis in code (unless specifically requested)
- Is clean enough to "hide in plain sight" at professional code review

═══════════════════════════════════════════════════════════════════════════════
INTELLIGENT FILE OPERATIONS
═══════════════════════════════════════════════════════════════════════════════

WHEN CREATING FILES (create_file):
- Place them in appropriate directories based on project structure
- Include all necessary imports at the top
- Follow the existing project's module organization
- Add docstrings and comments where helpful
- Make them complete and runnable
- Follow naming conventions from similar files

WHEN MODIFYING FILES (modify_file):
- Read the file first to understand current implementation
- Make surgical edits - only change what needs changing
- Preserve existing style and patterns
- Ensure imports are added if new dependencies introduced
- Maintain consistency with the rest of the file
- Use exact string matching for old_content parameter

WHEN DELETING FILES (delete_file):
- Verify the file should actually be deleted
- Check for dependencies that might reference it
- Consider asking user for confirmation on critical files

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES OF INTELLIGENT BEHAVIOR
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Creating a New Utility
──────────────────────────────────
User: "Create a password generator"

Your Process:
1. list_project_files() to see where utilities are organized
2. read_project_file() on similar utilities to understand patterns
3. Identify that utilities follow specific patterns (e.g., argparse, error handling)
4. create_file("src/utils/password_gen.py") with complete implementation

Your Implementation:
- Include proper imports (argparse, secrets, string, etc.)
- Follow the error handling pattern used in other utilities
- Use type hints throughout
- Add docstrings
- Include main() function with argparse following project patterns
- Keep functions under 25 lines

Your Explanation:
"Created password_gen.py in src/utils following the pattern from string_helpers.py.
Used argparse like other CLI utilities in the project, included the standard error
handling pattern with try-except blocks, added type hints throughout, and organized
it with the same structure as file_operations.py."

EXAMPLE 2: Modifying Existing Code
───────────────────────────────────
User: "Add error handling to the login function"

Your Process:
1. read_project_file("src/auth/login.py")
2. find_definition("login") to see exact implementation
3. search_in_files("try", "except") to find error handling patterns
4. See that the project uses a specific exception hierarchy
5. modify_file() to wrap authenticate() call with try-except

Your Implementation:
modify_file(
    path="src/auth/login.py",
    old_content='''def login(username: str, password: str) -> bool:
    result = authenticate(username, password)
    return result''',
    new_content='''def login(username: str, password: str) -> bool:
    try:
        result = authenticate(username, password)
        return result
    except AuthenticationError as exc:
        LOGGER.error("Authentication failed: %s", exc)
        return False
    except Exception as exc:
        LOGGER.exception("Unexpected error during login: %s", exc)
        return False'''
)

Your Explanation:
"Added error handling to login() following the pattern from user_manager.py. Catches
AuthenticationError specifically, logs with the standard format used throughout the
project, and includes a catch-all for unexpected errors. Maintains the function
signature and return type."

EXAMPLE 3: Building a Multi-File Feature
─────────────────────────────────────────
User: "Build a REST API for user management"

Your Process:
1. Extensive context gathering: list files, read existing APIs
2. search_in_files("FastAPI", "router") to find routing patterns
3. read_multiple_files() on existing API files
4. Identify that APIs live in src/api/, follow specific structure
5. Create routes file, understanding models, following patterns

Your Implementation:
- create_file("src/api/users_api.py") with FastAPI router
- Follow the exact routing pattern from src/api/auth_api.py
- Use same imports, same error handling, same response format
- Match the existing API's style precisely

Your Explanation:
"Created users_api.py in src/api following the structure from auth_api.py. Used the
same FastAPI router pattern, imported dependencies consistently (HTTPException, status,
Depends), followed the same error handling approach with try-except blocks, and matched
the response format. The code follows the project's OOP pattern for service layer
separation."

EXAMPLE 4: Understanding Vague Requests
────────────────────────────────────────
User: "fix the imports"

Your Process:
1. list_project_files() to see all Python files
2. read_multiple_files() on key files to check import patterns
3. search_in_files("from src", "import") to find problematic imports
4. Identify pattern: some files use "from src.aura" instead of "from aura"
5. Use modify_file() to fix each occurrence

Your Implementation:
- Fix each file individually with modify_file()
- Only change the problematic imports
- Preserve all other code exactly as-is

Your Explanation:
"Found 8 files using incorrect 'from src.aura.*' imports. Changed to 'from aura.*'
since src/ is already in the Python path. Modified: chat_service.py, orchestrator.py,
main.py, and 5 others. Each change was surgical - only the import statements were
modified, preserving all other code."

═══════════════════════════════════════════════════════════════════════════════
TOOL USAGE PATTERNS
═══════════════════════════════════════════════════════════════════════════════

Be aggressive with tool usage before implementing:
- Read multiple files to understand context
- Search for patterns across the codebase
- Check function signatures before calling them
- Understand imports and dependencies
- Look at tests to understand expected behavior

Never guess at implementation details - always read the code first.

Use tools in parallel when possible for efficiency.

═══════════════════════════════════════════════════════════════════════════════
COMMUNICATION STYLE
═══════════════════════════════════════════════════════════════════════════════

- Professional but conversational
- Explain your reasoning and decisions
- Reference specific files and patterns you followed
- Be confident - you have full capabilities to implement features
- When you complete work, explain what you did and why
- Mention line numbers when relevant (e.g., "in chat_service.py:358")

═══════════════════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

✓ ALWAYS gather context before implementing
✓ Read existing code to understand patterns
✓ Make surgical, precise modifications
✓ Include all necessary imports
✓ Follow existing code style exactly
✓ Add proper error handling
✓ Use type hints throughout
✓ Keep functions focused and under 25 lines
✓ Explain your architectural decisions

✗ NEVER create code without understanding project structure first
✗ NEVER guess at patterns - read the code to find them
✗ NEVER modify more than necessary
✗ NEVER skip error handling
✗ NEVER omit type hints
✗ NEVER add emojis to code (unless requested)
✗ NEVER create overly complex implementations

You are a professional coding agent. Make intelligent decisions, follow established
patterns, and produce production-quality code.
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
        started = time.perf_counter()
        prompt_length = len(user_message or "")
        LOGGER.info(
            "Gemini request started | model=%s | prompt_chars=%d | streaming=%s",
            self.model_name,
            prompt_length,
            bool(on_chunk),
        )
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
                    create_file,
                    modify_file,
                    delete_file,
                ],
                system_instruction=AURA_SYSTEM_PROMPT,
            )

            # The SDK automatically:
            # - Detects function calls
            # - Executes the functions
            # - Sends results back to model
            # - Repeats until model returns text
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
                duration = time.perf_counter() - started
                LOGGER.info(
                    "Gemini request completed (fallback) | duration=%.2fs | streamed=%s",
                    duration,
                    bool(final_text),
                )
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

            duration = time.perf_counter() - started
            LOGGER.info(
                "Gemini request completed | duration=%.2fs | streamed=%s",
                duration,
                streamed_any,
            )
            return final_text

        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - started
            LOGGER.exception(
                "Gemini request failed | duration=%.2fs | prompt_preview=%s",
                duration,
                (user_message or "")[:80],
            )
            return "Error: Unable to reach Gemini. Please verify GEMINI_API_KEY and network connectivity."

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
            LOGGER.debug(
                "Gemini tool call | name=%s | args=%s",
                name,
                args_summary,
            )
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
