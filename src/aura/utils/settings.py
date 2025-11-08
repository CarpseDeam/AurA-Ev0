
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_LOCAL_MODEL_ENDPOINT = "http://localhost:11434/api/generate"

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
            "gemini_model": DEFAULT_GEMINI_MODEL,
            "claude_model": DEFAULT_CLAUDE_MODEL,
            "local_model_endpoint": DEFAULT_LOCAL_MODEL_ENDPOINT,
            "selected_agent": None,
            "agent_executable": None,
            "sidebar_collapsed": False,
            "sidebar_width": 280,
        }

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
            # Ensure all keys are present, providing defaults for any missing ones
            settings.setdefault("gemini_model", DEFAULT_GEMINI_MODEL)
            settings.setdefault("claude_model", DEFAULT_CLAUDE_MODEL)
            settings.setdefault("local_model_endpoint", DEFAULT_LOCAL_MODEL_ENDPOINT)
            settings.setdefault("selected_agent", None)
            settings.setdefault("agent_executable", None)
            settings.setdefault("sidebar_collapsed", False)
            settings.setdefault("sidebar_width", 280)
            return settings
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse settings file: {e}. Using default settings.")
        return {
            "gemini_model": DEFAULT_GEMINI_MODEL,
            "claude_model": DEFAULT_CLAUDE_MODEL,
            "local_model_endpoint": DEFAULT_LOCAL_MODEL_ENDPOINT,
            "selected_agent": None,
            "agent_executable": None,
            "sidebar_collapsed": False,
            "sidebar_width": 280,
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

