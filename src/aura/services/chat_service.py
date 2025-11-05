"""Conversational interface for Aura with Gemini streaming."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Mapping

import google.generativeai as genai

LOGGER = logging.getLogger(__name__)

AURA_SYSTEM_PROMPT = (
    "You are Aura, an AI orchestrator with personality. You help developers build "
    "clean code by breaking requests into focused sessions.\n\n"
    "Traits:\n"
    "- Enthusiastic but not annoying\n"
    "- Speaks like a helpful senior dev\n"
    '- Casual language ("let\'s", "we\'re gonna", "looks good")\n'
    '- Celebrates wins ("Nice!", "Boom!")\n'
    "- Honest about challenges\n"
    "- Explains technical choices clearly\n"
    "- NO corporate speak, NO robot language\n\n"
    "You coordinate CLI coding agents but users talk to YOU. When discussing plans "
    "or results, speak naturally in first person.\n\n"
    "Example:\n"
    'Bad: "The system will now create the user model"\n'
    'Good: "Let me build that user model for you"\n'
)


@dataclass
class ChatMessage:
    """Represents a single chat message in the conversation history."""

    role: str
    content: str


@dataclass
class ChatService:
    """Streams conversational replies from Gemini with Aura's personality."""

    api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model: str = "gemini-2.5-pro"
    _history: List[ChatMessage] = field(default_factory=list, init=False)
    _client_configured: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate API key and configure client."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        genai.configure(api_key=self.api_key)
        self._client_configured = True
        self._append_system_prompt()

    def send_message(self, message: str) -> Iterator[str]:
        """Send a message and yield the streaming response."""
        if not message:
            raise ValueError("Message must be a non-empty string.")
        self._history.append(ChatMessage(role="user", content=message))
        LOGGER.debug("Sending chat message: %s", message)
        model = genai.GenerativeModel(self.model)
        chat_history = [{"role": msg.role, "parts": [msg.content]} for msg in self._history]
        response = model.generate_content(
            chat_history,
            stream=True,
        )
        collected = []
        try:
            for chunk in response:
                text = chunk.text or ""
                if text:
                    collected.append(text)
                    yield text
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Chat streaming failed: %s", exc)
            yield "[Aura] Uh-oh, something went sideways talking to Gemini."
            return
        combined = "".join(collected)
        if combined:
            self._history.append(ChatMessage(role="model", content=combined))

    def get_history(self) -> List[Mapping[str, str]]:
        """Return the chat history including system prompt."""
        return [{"role": entry.role, "content": entry.content} for entry in self._history]

    def clear_history(self) -> None:
        """Reset the conversation history."""
        self._history.clear()
        self._append_system_prompt()

    def _append_system_prompt(self) -> None:
        """Ensure the system prompt is first in the history."""
        if self._history and self._history[0].role == "system":
            self._history[0] = ChatMessage(role="system", content=AURA_SYSTEM_PROMPT)
        else:
            self._history.insert(0, ChatMessage(role="system", content=AURA_SYSTEM_PROMPT))
