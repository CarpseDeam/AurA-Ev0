"""Local specialist tools powered by configurable Ollama models."""

from __future__ import annotations

import logging
from textwrap import dedent

import ollama

from aura.state import get_app_state

LOGGER = logging.getLogger(__name__)
DEFAULT_SPECIALIST_MODEL = "phi-3-mini"


def generate_commit_message(diff_content: str) -> str:
    """Generate a conventional commit message from a git diff."""
    diff = (diff_content or "").strip()
    if not diff:
        return "Error: No diff content provided for commit message generation."

    app_state = get_app_state()
    model_name = DEFAULT_SPECIALIST_MODEL
    if app_state:
        model_name = (app_state.specialist_model or DEFAULT_SPECIALIST_MODEL).strip() or DEFAULT_SPECIALIST_MODEL

    prompt = dedent(
        f"""
        You are an elite release engineer. Write a single conventional commit message for the diff below.

        Requirements:
        - Use lowercase commit types (feat, fix, chore, docs, refactor, test, build).
        - Include a scope in parentheses when it clarifies the affected area.
        - Keep the summary under 72 characters.
        - Only add an optional body if the diff spans multiple concepts; otherwise return a single line.
        - Never mention the phrase "commit message"; just output the message itself.

        Git diff:
        {diff}
        """
    ).strip()

    try:
        response = ollama.generate(model=model_name, prompt=prompt)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to generate commit message with model %s", model_name)
        return f"Error: Unable to reach local model '{model_name}': {exc}"

    raw_message = (response.get("response") or "").strip()
    if not raw_message:
        return "Error: Local model returned an empty commit message."

    return _sanitize_commit_message(raw_message)


def _sanitize_commit_message(message: str) -> str:
    """Normalize whitespace and strip formatting artifacts."""
    cleaned = message.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("` \n")
    cleaned_lines = [line.rstrip() for line in cleaned.splitlines()]
    # Remove trailing empty lines but keep intentional spacing if body exists
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    return "\n".join(cleaned_lines).strip()
