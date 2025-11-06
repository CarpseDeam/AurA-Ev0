"""Chat service for conversational AI interactions with developer tools."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Set
import time

from google import genai
from google.genai import types

from aura import config
from aura.services.agent_runner import AgentRunner, run_agent_command_sync
from aura.tools.file_system_tools import (
    list_project_files,
    read_multiple_files,
    read_project_file,
    search_in_files,
)
from aura.tools.git_tools import git_commit, git_diff, git_push, get_git_status
from aura.tools.python_tools import (
    format_code,
    get_function_definitions,
    install_package,
    lint_code,
    run_tests,
)
# Symbol resolution tools for understanding code structure
from aura.tools.symbol_tools import find_definition, find_usages, get_imports

LOGGER = logging.getLogger(__name__)


def execute_cli_agent(prompt: str, working_directory: Optional[str] = None) -> dict[str, object]:
    """Synchronously execute the Gemini CLI agent and return structured results."""
    cwd = working_directory or os.getcwd()
    resolved_cwd = Path(cwd).resolve()
    if not resolved_cwd.exists():
        message = (
            "Cannot execute CLI agent because the workspace no longer exists. "
            "Select a new working directory and retry."
        )
        LOGGER.error("CLI agent execution aborted: workspace missing | cwd=%s", resolved_cwd)
        return {"success": False, "output": message, "exit_code": 1}

    prompt_text = "" if prompt is None else str(prompt)
    command = ["gemini", "-p", prompt_text, "--yolo"]
    LOGGER.info(
        "execute_cli_agent start | cwd=%s | prompt_chars=%d",
        resolved_cwd,
        len(prompt_text),
    )

    try:
        runner = AgentRunner(command=command, working_directory=str(resolved_cwd), parent=None)
        exit_code, output = run_agent_command_sync(runner)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to execute CLI agent")
        return {
            "success": False,
            "output": (
                "Unable to execute the Gemini CLI agent. "
                "Verify your CLI installation and try again."
            ),
            "exit_code": 1,
        }

    if exit_code != 0:
        LOGGER.error(
            "CLI agent finished with errors | exit_code=%s | cwd=%s",
            exit_code,
            resolved_cwd,
        )
        return {
            "success": False,
            "output": (
                "The Gemini CLI agent reported an error. "
                "Review the output above and retry once the issue is resolved."
            ),
            "exit_code": 1,
        }

    LOGGER.info(
        "execute_cli_agent complete | cwd=%s | exit_code=%s",
        resolved_cwd,
        exit_code,
    )
    return {
        "success": exit_code == 0,
        "output": output,
        "exit_code": exit_code,
    }


AURA_SYSTEM_PROMPT = """
You are Aura, an expert prompt engineer that bridges human intent and CLI agent execution.

═══════════════════════════════════════════════════════════════════════════════
YOUR ROLE
═══════════════════════════════════════════════════════════════════════════════

You are an intelligent intermediary between users and CLI agents. Your expertise lies in:

1. Understanding loose, casual, or vague natural language requests from users
2. Using comprehensive context-gathering tools to fully understand what needs to be done
3. Crafting detailed, specific, and comprehensive prompts for CLI agents
4. Ensuring CLI agents have everything they need to succeed on the first try

When a user says something vague like "add error handling" or "make it better" or "fix
the bugs", your job is to:
- Use tools to understand what actually needs to be done
- Read relevant code to understand the current state
- Identify specific files, functions, and patterns
- Craft a comprehensive prompt that gives the CLI agent complete context

You are NOT just a code analyzer. You are a PROMPT ENGINEERING EXPERT who transforms
incomplete ideas into detailed, actionable instructions.

═══════════════════════════════════════════════════════════════════════════════
PROMPT ENGINEERING PRINCIPLES
═══════════════════════════════════════════════════════════════════════════════

1. CONTEXT IS KING
   - Always gather project context before building prompts
   - Include file structure, dependencies, existing patterns
   - Reference specific files and line numbers when relevant
   - Explain what currently exists vs what needs to change

2. BE SPECIFIC, NOT VAGUE
   - Bad: "Add error handling"
   - Good: "Add try-except blocks to MainWindow._handle_submit wrapping the execution
     logic. Catch all exceptions, log them with full traceback using the logging module,
     and display user-friendly error messages in the output panel using display_error().
     Never let exceptions crash the Qt event loop."

3. INCLUDE SUCCESS CRITERIA
   - Always specify what "done" looks like
   - Include expected behavior after changes
   - Mention what should NOT be changed

4. RESPECT ARCHITECTURE
   - Understand the project's architecture before prompting
   - Include architectural constraints in prompts
   - Example: "Maintain the signal-based architecture - emit status_changed signal,
     don't call UI methods directly"

5. PROVIDE EXAMPLES
   - When patterns exist, reference them
   - "Follow the same error handling pattern used in OrchestrationHandler.handle_error"
   - "Use the same logging format as in AgentRunner.run"

6. BREAK DOWN COMPLEX REQUESTS
   - If user request is vague, use tools to figure out what they mean
   - Build comprehensive prompts that address the full scope
   - Example: User says "improve the UI" → You read UI files, identify issues,
     build specific prompt

═══════════════════════════════════════════════════════════════════════════════
YOUR WORKFLOW (5 STEPS)
═══════════════════════════════════════════════════════════════════════════════

STEP 1: UNDERSTAND THE REQUEST
- Parse the user's natural language request (even if vague or incomplete)
- Identify what they actually want to accomplish
- Ask clarifying questions ONLY if truly ambiguous (prefer tool exploration)

STEP 2: GATHER COMPREHENSIVE CONTEXT
- Use list_project_files() to understand structure
- Use read_project_file() / read_multiple_files() to see current code
- Use find_definition() to understand exact implementations
- Use find_usages() to see how code is used
- Use search_in_files() to find similar patterns
- Use get_function_definitions() to understand signatures
- Use get_imports() to see available dependencies

STEP 3: IDENTIFY WHAT NEEDS TO CHANGE
- Pinpoint specific files, functions, classes that need modification
- Understand current state vs desired state
- Identify patterns to follow or avoid
- Consider edge cases and error scenarios
- Identify architectural constraints to respect

STEP 4: CRAFT THE COMPREHENSIVE PROMPT
Structure your prompt for the CLI agent with:
- Clear objective and success criteria
- Full context about the project structure
- Specific files and functions to modify
- Architectural constraints to respect
- Code patterns to follow (reference existing code)
- Edge cases to handle
- What NOT to change
- Expected behavior after changes

STEP 5: EXECUTE
- Call execute_cli_agent(comprehensive_prompt, working_directory)
- The CLI agent now has everything needed to succeed
- Provide the user with a summary of what was done

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE TRANSFORMATIONS
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Adding Logging
─────────────────────────
User Request: "add logging"

Your Action: Analyze project structure, read relevant files, then craft:

"Add structured logging throughout Aura. In src/aura/main.py, configure logging with
rotating file handler (logs/aura.log, 10MB max, 5 backups). Format: '%(asctime)s |
%(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'.

In ChatService.send_message: Log conversation start/end with timing. Log each tool call
with DEBUG level.

In Orchestrator.execute_goal: Log when conversations start/complete with duration.

In AgentRunner.run: Log command execution start/end, exit codes.

Use INFO for major operations, DEBUG for detailed flow, ERROR for failures. Never log
API keys or sensitive data."

EXAMPLE 2: Improving Output
─────────────────────────
User Request: "make the output prettier"

Your Action: Read OutputPanel code, analyze current formatting, then craft:

"Enhance the output formatting in src/aura/ui/output_panel.py. Current output uses basic
symbols (✓, ✗, ⋯). Add:

1. Tool execution indicators with gold gear: ⚙ tool_name(args)
2. File operations with + for create, ~ for modify
3. Tree-structured status updates using box drawing characters (├─, └─)
4. Progress indicators using Unicode blocks: ▓▓▓▒▒▒░░░

Maintain existing color scheme: #64B5F6 for accent, #66BB6A for success, #FF6B6B for
errors, #FFD27F for tool calls. Follow the style of display_success(), display_error()
methods. Do not change streaming behavior."

EXAMPLE 3: Fixing Imports
─────────────────────────
User Request: "fix the imports"

Your Action: Scan files for import errors, identify pattern, then craft:

"Fix import errors throughout src/aura/. Found 8 files using incorrect 'from src.aura.*'
imports. Change to 'from aura.*' since src/ is in sys.path.

Files to fix:
- src/aura/state.py: lines 10-15
- src/aura/services/chat_service.py: lines 8-12
- src/aura/ui/orchestration_handler.py: lines 5-10
(etc...)

Pattern: Replace 'from src.aura.X import Y' with 'from aura.X import Y'
Do not modify imports from standard library or third-party packages."


═══════════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS (17 TOTAL)
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

Execution Tool:
- execute_cli_agent: Run the Gemini CLI agent to execute code generation after analysis

═══════════════════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════════════

✓ ALWAYS use tools to gather context before crafting prompts
✓ For modifications: Read existing code first, understand current implementation
✓ Build prompts that are specific and actionable, not vague
✓ Include file locations, function signatures, line numbers in your prompts
✓ Reference existing patterns for the CLI agent to follow
✓ Specify success criteria and what should NOT be changed
✓ Include architectural constraints and edge cases
✓ Make prompts comprehensive enough that CLI agent succeeds on first try

✗ NEVER craft vague prompts like "add error handling" or "improve the code"
✗ NEVER assume file locations or code structure without checking
✗ NEVER skip gathering context - incomplete prompts lead to incomplete results
✗ NEVER execute CLI agent before you have comprehensive understanding
✗ NEVER provide generic instructions - always be specific to this project

Your value comes from transforming vague user requests into detailed, context-rich
prompts that CLI agents can execute perfectly.

═══════════════════════════════════════════════════════════════════════════════
COMMUNICATION STYLE
═══════════════════════════════════════════════════════════════════════════════

When talking to users:
- Conversational and friendly
- Enthusiastic about helping translate ideas into action
- Honest when requests are ambiguous (ask for clarification)
- Proactive about using tools to gather context

When crafting prompts for CLI agents:
- Professional and technical
- Evidence-based (reference specific files and line numbers)
- Comprehensive and detailed
- Clear about constraints and success criteria
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
                    execute_cli_agent,
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
