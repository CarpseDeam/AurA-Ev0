"""Session planning service leveraging Gemini JSON responses."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List

import google.generativeai as genai

from src.aura.services.chat_service import AURA_SYSTEM_PROMPT


LOGGER = logging.getLogger(__name__)

PLANNING_PROMPT_TEMPLATE = """Decompose this coding request into 3-7 focused sessions.

Rules:
1. Each session creates ONE module/feature
2. Sessions run sequentially (build on previous work)
3. Estimate 10-25 minutes per session
4. Max 2-4 files per session
5. No file over 200 lines
6. Clear, specific deliverables

SESSION NAMING RULES - THIS IS CRITICAL:
⚠️  Names MUST be 2-5 words MAX
⚠️  MUST be SPECIFIC, NEVER generic
⚠️  Use CONCRETE nouns, not abstract concepts

FORBIDDEN WORDS (will cause rejection):
❌ implementation, setup, main, core, general, base, initial, add features, improvements

GOOD NAMES (use these patterns):
✅ "User Login Routes" - specific feature + component type
✅ "Password Hashing Utility" - specific purpose + component type
✅ "Database Schema Migrations" - specific task + component type
✅ "Comment Model Class" - specific entity + component type

BAD NAMES (DO NOT USE):
❌ "Core Implementation" - too vague, forbidden word
❌ "Main Setup" - too vague, forbidden word
❌ "Add Features" - too vague, forbidden word
❌ "Basic Structure" - too vague
❌ "Initial Configuration" - too vague

VALIDATION: Before finalizing, check each session name:
- Does it contain forbidden words? → REWRITE
- Could this apply to any project? → TOO VAGUE, BE MORE SPECIFIC
- Does it name a specific feature/component? → GOOD

User goal: {goal}

Project context:
{project_context}

CRITICAL: Review all session names. If any name could apply to multiple different projects, it's too vague. Make it MORE specific.

Return JSON:
{{
  "sessions": [
    {{
      "name": "User Model Class",
      "prompt": "Create User model class with fields: id, email, password_hash. Include bcrypt hashing utility. File: models/user.py",
      "estimated_minutes": 15,
      "dependencies": []
    }},
    {{
      "name": "Login Logout Routes",
      "prompt": "Create login/logout endpoints using User model from models/user.py. Do NOT recreate User model. File: routes/auth.py",
      "estimated_minutes": 12,
      "dependencies": ["User Model Class"]
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
    """Produces structured plans using a dedicated Gemini model instance."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-pro") -> None:
        """Configure a dedicated planning model that returns JSON."""
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set.")
        self._api_key = api_key
        self._model_name = model

        genai.configure(api_key=self._api_key)
        self._model = genai.GenerativeModel(
            self._model_name,
            system_instruction=AURA_SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"},
        )

    def plan_sessions(self, goal: str, project_context: str) -> SessionPlan:
        """Generate a session plan for the given goal."""
        prompt = PLANNING_PROMPT_TEMPLATE.format(goal=goal.strip(), project_context=project_context.strip())
        response_text = self._collect_response(prompt)
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError as exc:
            LOGGER.exception("Failed to parse planning response JSON: %s", exc)
            raise ValueError("Planning service returned invalid JSON.") from exc
        return self._build_plan(payload)

    def _collect_response(self, prompt: str) -> str:
        """Request a plan and normalize the JSON payload."""
        response = self._model.generate_content(prompt)

        text_chunks: list[str] = []
        direct_text = getattr(response, "text", None)
        if direct_text:
            text_chunks.append(direct_text)

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    text_chunks.append(part_text)

        combined = "".join(text_chunks)
        if not combined.strip():
            raise ValueError("Planning service returned an empty response.")

        # Try to extract JSON from response (handles natural language + JSON)
        json_match = re.search(r'\{[\s\S]*\}', combined)
        if json_match:
            response_text = json_match.group(0)
        else:
            # Strip markdown code blocks if present
            response_text = combined.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```

            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```

        return response_text.strip()

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
