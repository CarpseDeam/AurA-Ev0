
import json
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LATEST_CLAUDE_SONNET_MODEL = "claude-sonnet-4-5-20250929"
LATEST_CLAUDE_OPUS_MODEL = "claude-opus-4-20250514"
AGENT_MODEL_OPTIONS = [LATEST_CLAUDE_SONNET_MODEL, LATEST_CLAUDE_OPUS_MODEL]
DEFAULT_AGENT_MODEL = AGENT_MODEL_OPTIONS[0]
DEFAULT_MAX_TOKENS_BUDGET = 100_000
DEFAULT_TOOL_CALL_LIMIT = 25
DEFAULT_TEMPERATURE = 0.0
DEFAULT_COST_TRACKING = True
DEFAULT_ANTHROPIC_API_KEY = ""
DEFAULT_LOCAL_MODEL_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_SPECIALIST_MODEL = "phi-3-mini"
MODEL_MIGRATIONS: Dict[str, str] = {
    "claude-3-sonnet-20240229": LATEST_CLAUDE_SONNET_MODEL,
}
LEGACY_MODEL_KEYS = (
    "analyst_planning_model",
    "analyst_model",
    "analyst_investigation_model",
    "executor_model",
)


def _normalize_model_setting(settings: Dict[str, Any], key: str, default: str) -> bool:
    """
    Normalize deprecated, empty, or whitespace-only model identifiers.

    Returns:
        bool: True if the setting was changed.
    """
    value = settings.get(key)
    if isinstance(value, str):
        value = value.strip()

    if not value:
        settings[key] = default
        return True

    replacement = MODEL_MIGRATIONS.get(value)
    if replacement and replacement != value:
        logger.info("Upgrading %s model from %s to %s", key, value, replacement)
        settings[key] = replacement
        return True

    settings[key] = value
    return False


def _ensure_int_setting(
    settings: Dict[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> bool:
    """Ensure an integer configuration value stays within a safe range."""
    value = settings.get(key, default)
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default

    if value < minimum:
        value = minimum
    elif value > maximum:
        value = maximum

    if settings.get(key) != value:
        settings[key] = value
        return True
    return False


def _ensure_float_setting(
    settings: Dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> bool:
    """Ensure a float configuration value stays within a safe range."""
    value = settings.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default

    if value < minimum:
        value = minimum
    elif value > maximum:
        value = maximum

    if settings.get(key) != value:
        settings[key] = value
        return True
    return False


def _ensure_bool_setting(settings: Dict[str, Any], key: str, default: bool) -> bool:
    """Ensure a boolean configuration value."""
    value = settings.get(key, default)
    if isinstance(value, bool):
        normalized = value
    elif isinstance(value, str):
        normalized = value.strip().lower() in {"1", "true", "yes", "on"}
    else:
        normalized = bool(value)

    if settings.get(key) != normalized:
        settings[key] = normalized
        return True
    return False


def _apply_legacy_model_defaults(settings: Dict[str, Any]) -> bool:
    """Populate the single-agent model from any legacy fields if necessary."""
    if settings.get("agent_model"):
        return False

    for key in LEGACY_MODEL_KEYS:
        value = settings.get(key)
        if isinstance(value, str) and value.strip():
            settings["agent_model"] = value.strip()
            return True

    settings["agent_model"] = DEFAULT_AGENT_MODEL
    return True


def _default_settings() -> Dict[str, Any]:
    """Return a fresh copy of default settings."""
    return {
        "agent_model": DEFAULT_AGENT_MODEL,
        "anthropic_api_key": DEFAULT_ANTHROPIC_API_KEY,
        "max_tokens_budget": DEFAULT_MAX_TOKENS_BUDGET,
        "tool_call_limit": DEFAULT_TOOL_CALL_LIMIT,
        "temperature": DEFAULT_TEMPERATURE,
        "enable_cost_tracking": DEFAULT_COST_TRACKING,
        "specialist_model": DEFAULT_SPECIALIST_MODEL,
        "local_model_endpoint": DEFAULT_LOCAL_MODEL_ENDPOINT,
        "use_local_investigation": False,
        "sidebar_collapsed": False,
        "sidebar_width": 280,
        "verbosity": "normal",
    }


def get_settings_path() -> Path:
    """
    Determines the appropriate path for the settings file.

    Returns:
        Path: The path to the settings.json file.
    """
    return Path.home() / ".aura" / "settings.json"


def load_settings() -> Dict[str, Any]:
    """
    Loads settings from the settings file.

    If the file doesn't exist or is invalid, returns default settings.

    Returns:
        Dict[str, Any]: A dictionary containing the application settings.
    """
    settings_path = get_settings_path()
    if not settings_path.exists():
        logger.info("Settings file not found. Using default settings.")
        return _default_settings()

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.error("Failed to load or parse settings file: %s. Using defaults.", exc)
        return _default_settings()

    updated = False
    updated |= _apply_legacy_model_defaults(settings)
    updated |= _normalize_model_setting(settings, "agent_model", DEFAULT_AGENT_MODEL)
    updated |= _ensure_int_setting(
        settings,
        "max_tokens_budget",
        DEFAULT_MAX_TOKENS_BUDGET,
        minimum=1_000,
        maximum=400_000,
    )
    updated |= _ensure_int_setting(
        settings,
        "tool_call_limit",
        DEFAULT_TOOL_CALL_LIMIT,
        minimum=1,
        maximum=100,
    )
    updated |= _ensure_float_setting(
        settings,
        "temperature",
        DEFAULT_TEMPERATURE,
        minimum=0.0,
        maximum=1.0,
    )
    updated |= _ensure_bool_setting(settings, "enable_cost_tracking", DEFAULT_COST_TRACKING)
    settings.setdefault("anthropic_api_key", DEFAULT_ANTHROPIC_API_KEY)
    settings.setdefault("specialist_model", DEFAULT_SPECIALIST_MODEL)
    settings.setdefault("local_model_endpoint", DEFAULT_LOCAL_MODEL_ENDPOINT)
    settings.setdefault("use_local_investigation", False)
    settings.setdefault("sidebar_collapsed", False)
    settings.setdefault("sidebar_width", 280)
    settings.setdefault("verbosity", "normal")

    if updated:
        save_settings(settings)
    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """
    Saves the given settings to the settings file.

    Args:
        settings (Dict[str, Any]): A dictionary containing the settings to save.
    """
    settings_path = get_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
        logger.info("Settings successfully saved to %s", settings_path)
    except IOError as exc:
        logger.error("Failed to save settings to %s: %s", settings_path, exc)
