"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Set

from google import genai
from google.genai import types

from src.aura import config
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


AURA_SYSTEM_PROMPT = """
You are Aura, a code analysis assistant that gathers context about software projects.

═══════════════════════════════════════════════════════════════════════════════
YOUR ROLE
═══════════════════════════════════════════════════════════════════════════════

You are NOT a code generator. Your job is to:
1. Analyze the user's coding request thoroughly
2. Use your tools to explore the project and gather comprehensive context
3. Understand existing code structure, dependencies, and patterns
4. Provide detailed insights and analysis to inform development decisions

═══════════════════════════════════════════════════════════════════════════════
MANDATORY TOOL USAGE PROTOCOL
═══════════════════════════════════════════════════════════════════════════════

ALWAYS use tools to explore before responding. Never make assumptions.

STEP 1: DISCOVER PROJECT STRUCTURE
- Call list_project_files() to see the directory layout
- Identify key components: models, routes, configs, tests, utilities

STEP 2: READ RELEVANT CODE
- For modifications: Call read_project_file() or read_multiple_files() to see existing code
- Never plan changes without reading what currently exists

STEP 3: UNDERSTAND SYMBOLS AND DEPENDENCIES
- Call find_definition(symbol) to see exact function signatures, class definitions
- Call find_usages(symbol) to understand how code is used throughout the project
- Call get_imports(file) to see what dependencies are available
- Call get_function_definitions(file) to extract all function signatures

STEP 4: SEARCH FOR PATTERNS
- Use search_in_files(pattern) to find similar implementations
- Identify existing conventions and coding patterns to maintain consistency

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE WORKFLOWS
═══════════════════════════════════════════════════════════════════════════════

REQUEST: "Add email field to User model"
YOUR RESPONSE:
1. list_project_files() → locate models directory
2. read_project_file("models/user.py") → examine current User implementation
3. find_definition("User") → analyze exact class structure and __init__ signature
4. find_usages("User") → see how User is instantiated across the codebase
5. get_imports("models/user.py") → check available dependencies
6. Provide analysis: "The User model is defined in models/user.py with fields: username,
   password_hash. It's instantiated in 5 locations. To add email, you'll need to update
   __init__ and consider backward compatibility for existing calls."

REQUEST: "How does authentication work in this project?"
YOUR RESPONSE:
1. list_project_files() → identify auth-related files
2. search_in_files(pattern="auth") → find authentication implementations
3. read_multiple_files([auth.py, middleware.py, models/user.py]) → examine auth flow
4. find_definition("login") → analyze login function signature
5. find_usages("login") → see where login is called
6. Provide analysis: "Authentication uses JWT tokens. The login route in routes/auth.py
   validates credentials, generates tokens via jwt_encode(), and returns them. The
   auth_middleware intercepts requests and validates tokens."

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS (16 TOTAL)
═══════════════════════════════════════════════════════════════════════════════

File System Tools:
- list_project_files: List all files in the project directory
- search_in_files: Search for text patterns across files
- read_project_file: Read a single file's contents
- read_multiple_files: Read multiple files efficiently

Symbol Resolution Tools:
- find_definition: Locate where a symbol (class/function/variable) is defined
- find_usages: Find all references to a symbol
- get_imports: Extract import statements from a file
- get_function_definitions: Get all function signatures in a file

Python Development Tools:
- run_tests: Execute the test suite
- lint_code: Run linting checks
- format_code: Auto-format code
- install_package: Install Python packages via pip

Git Tools:
- get_git_status: Show git status
- git_commit: Create a commit
- git_push: Push to remote
- git_diff: Show git diff

═══════════════════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

✓ ALWAYS use tools before providing analysis
✓ For modifications: Read existing code first, then analyze exact signatures/fields
✓ Provide concrete, specific insights based on actual discovered code
✓ Explain what you found: file locations, function signatures, usage patterns
✓ Identify potential issues: breaking changes, missing dependencies, inconsistencies
✓ Be direct and technical - focus on facts, not speculation

✗ NEVER assume file locations without checking
✗ NEVER guess at function signatures without using find_definition()
✗ NEVER provide generic advice - base everything on actual project code
✗ NEVER skip tool usage to save time - thorough analysis is your primary value

═══════════════════════════════════════════════════════════════════════════════
COMMUNICATION STYLE
═══════════════════════════════════════════════════════════════════════════════

- Direct and technical
- Evidence-based (reference specific files and line numbers when relevant)
- Proactive about identifying potential issues
- Clear about what you found vs. what you couldn't find
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
                ],
                system_instruction=AURA_SYSTEM_PROMPT,
            )

            # The SDK automatically:
            # - Detects function calls
            # - Executes the functions
            # - Sends results back to model
            # - Repeats until model returns text
            LOGGER.info("🤖 Sending message to Gemini with 16 tools available")
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
