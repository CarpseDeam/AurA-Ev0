
import json
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LATEST_CLAUDE_SONNET_MODEL = "claude-sonnet-4-5-20250929"
LATEST_CLAUDE_HAIKU_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_ANALYST_PLANNING_MODEL = LATEST_CLAUDE_SONNET_MODEL
DEFAULT_ANALYST_INVESTIGATION_MODEL = LATEST_CLAUDE_HAIKU_MODEL
DEFAULT_ANALYST_MODEL = DEFAULT_ANALYST_PLANNING_MODEL
DEFAULT_EXECUTOR_MODEL = LATEST_CLAUDE_SONNET_MODEL
DEFAULT_LOCAL_MODEL_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_SPECIALIST_MODEL = "phi-3-mini"
MODEL_MIGRATIONS: Dict[str, str] = {
    "claude-3-sonnet-20240229": LATEST_CLAUDE_SONNET_MODEL,
}


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
        return {
            "analyst_model": DEFAULT_ANALYST_MODEL,
            "analyst_planning_model": DEFAULT_ANALYST_PLANNING_MODEL,
            "analyst_investigation_model": DEFAULT_ANALYST_INVESTIGATION_MODEL,
            "executor_model": DEFAULT_EXECUTOR_MODEL,
            "specialist_model": DEFAULT_SPECIALIST_MODEL,
            "local_model_endpoint": DEFAULT_LOCAL_MODEL_ENDPOINT,
            "use_local_investigation": False,
            "sidebar_collapsed": False,
            "sidebar_width": 280,
            "verbosity": "normal",
        }

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
            updated = False
            updated |= _normalize_model_setting(settings, "analyst_model", DEFAULT_ANALYST_MODEL)
            planning_default = settings.get("analyst_model", DEFAULT_ANALYST_PLANNING_MODEL)
            updated |= _normalize_model_setting(settings, "analyst_planning_model", planning_default)
            updated |= _normalize_model_setting(
                settings,
                "analyst_investigation_model",
                DEFAULT_ANALYST_INVESTIGATION_MODEL,
            )
            updated |= _normalize_model_setting(settings, "executor_model", DEFAULT_EXECUTOR_MODEL)
            settings.setdefault("specialist_model", DEFAULT_SPECIALIST_MODEL)
            settings.setdefault("local_model_endpoint", DEFAULT_LOCAL_MODEL_ENDPOINT)
            settings.setdefault("use_local_investigation", False)
            settings.setdefault("sidebar_collapsed", False)
            settings.setdefault("sidebar_width", 280)
            settings.setdefault("verbosity", "normal")
            settings.setdefault("analyst_planning_model", planning_default)
            settings.setdefault("analyst_investigation_model", DEFAULT_ANALYST_INVESTIGATION_MODEL)
            if updated:
                save_settings(settings)
            return settings
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse settings file: {e}. Using default settings.")
        return {
            "analyst_model": DEFAULT_ANALYST_MODEL,
            "analyst_planning_model": DEFAULT_ANALYST_PLANNING_MODEL,
            "analyst_investigation_model": DEFAULT_ANALYST_INVESTIGATION_MODEL,
            "executor_model": DEFAULT_EXECUTOR_MODEL,
            "specialist_model": DEFAULT_SPECIALIST_MODEL,
            "local_model_endpoint": DEFAULT_LOCAL_MODEL_ENDPOINT,
            "use_local_investigation": False,
            "sidebar_collapsed": False,
            "sidebar_width": 280,
            "verbosity": "normal",
        }

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
        logger.info(f"Settings successfully saved to {settings_path}")
    except IOError as e:
        logger.error(f"Failed to save settings to {settings_path}: {e}")
