"""Session planning service leveraging Aura's chat layer."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterator, List

from aura.services.chat_service import ChatService

LOGGER = logging.getLogger(__name__)

PLANNING_PROMPT_TEMPLATE = """Decompose this coding request into 3-7 focused sessions.

Rules:
1. Each session creates ONE module/feature
2. Sessions run sequentially (build on previous work)
3. Estimate 10-25 minutes per session
4. Max 2-4 files per session
5. No file over 200 lines
6. Clear, specific deliverables

User goal: {goal}

Project context:
{project_context}

Return JSON:
{{
  "sessions": [
    {{
      "name": "User Authentication Model",
      "prompt": "Create User model class with fields: id, email, password_hash. Include bcrypt hashing utility. File: models/user.py",
      "estimated_minutes": 15,
      "dependencies": []
    }},
    {{
      "name": "Auth Routes",
      "prompt": "Create login/logout endpoints using User model from models/user.py. Do NOT recreate User model. File: routes/auth.py",
      "estimated_minutes": 12,
      "dependencies": ["User Authentication Model"]
    }}
  ],
  "total_estimated_minutes": 27,
  "reasoning": "Split auth into model and routes for clean separation"
}}
"""


@dataclass(frozen=True)
class Session:
    """Represents a focused coding session."""

    name: str
    prompt: str
    estimated_minutes: int
    dependencies: List[str]


@dataclass(frozen=True)
class SessionPlan:
    """Holds the aggregate session plan."""

    sessions: List[Session]
    total_estimated_minutes: int
    reasoning: str


class PlanningService:
    """Produces structured plans using the chat service."""

    def __init__(self, chat_service: ChatService) -> None:
        """Store the chat service dependency."""
        self._chat = chat_service

    def plan_sessions(self, goal: str, project_context: str) -> SessionPlan:
        """Generate a session plan for the given goal."""
        prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal.strip(), project_context=project_context.strip())
        response_text = self._collect_response(self._chat.send_message(prompt))
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            LOGGER.exception("Failed to parse planning response JSON: %s", exc)
            raise ValueError("Planning service returned invalid JSON.") from exc
        return self._build_plan(payload)

    def _collect_response(self, stream: Iterator[str]) -> str:
        """Gather streaming chunks into a response string."""
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
        response = "".join(chunks)

        # Try to extract JSON from response (handles natural language + JSON)
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)
        else:
            # Strip markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]  # Remove ```json
            elif response.startswith("```"):
                response = response[3:]  # Remove ```

            if response.endswith("```"):
                response = response[:-3]  # Remove trailing ```

        return response.strip()

    def _build_plan(self, payload: dict) -> SessionPlan:
        """Validate and construct the session plan."""
        if "sessions" not in payload or not isinstance(payload["sessions"], list):
            raise ValueError("Planning response missing 'sessions' list.")
        sessions = [self._build_session(entry) for entry in payload["sessions"]]
        total_minutes = payload.get("total_estimated_minutes")
        reasoning = payload.get("reasoning")
        if not isinstance(total_minutes, int) or total_minutes <= 0:
            total_minutes = sum(session.estimated_minutes for session in sessions)
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = "Reasoning not provided."
        return SessionPlan(
            sessions=sessions,
            total_estimated_minutes=total_minutes,
            reasoning=reasoning.strip(),
        )

    def _build_session(self, data: dict) -> Session:
        """Create a Session from the parsed dictionary."""
        name = self._require_str(data, "name")
        prompt = self._require_str(data, "prompt")
        estimated_minutes = data.get("estimated_minutes")
        if not isinstance(estimated_minutes, int) or not 1 <= estimated_minutes <= 60:
            raise ValueError(f"Invalid estimated_minutes for session '{name}'.")
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
            raise ValueError(f"Invalid dependencies for session '{name}'.")
        return Session(
            name=name.strip(),
            prompt=prompt.strip(),
            estimated_minutes=estimated_minutes,
            dependencies=[dep.strip() for dep in dependencies],
        )

    def _require_str(self, data: dict, key: str) -> str:
        """Ensure the specified key exists as a non-empty string."""
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Planning response missing '{key}'.")
        return value
