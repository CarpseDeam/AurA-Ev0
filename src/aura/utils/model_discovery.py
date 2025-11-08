
import dataclasses
import subprocess
import time
import logging
from typing import List, Dict, Tuple

# Configure logging
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ModelInfo:
    """A dataclass to hold information about an available model."""
    model_id: str
    display_name: str
    description: str
    provider: str

# Module-level cache
_model_cache: Dict[str, Tuple[List[ModelInfo], float]] = {}
CACHE_DURATION = 300  # 5 minutes

def discover_gemini_models(api_key: str) -> List[ModelInfo]:
    """
    Discovers available Gemini models from the Google GenAI API.
    Caches results for 5 minutes.
    """
    if "gemini" in _model_cache:
        models, timestamp = _model_cache["gemini"]
        if time.time() - timestamp < CACHE_DURATION:
            return models

    fallback_models = [
        ModelInfo("gemini-1.5-pro-latest", "Gemini 1.5 Pro", "The latest Gemini 1.5 Pro model.", "google"),
        ModelInfo("gemini-1.5-flash-latest", "Gemini 1.5 Flash", "The latest Gemini 1.5 Flash model.", "google"),
    ]

    if not api_key:
        logger.warning("GEMINI_API_KEY not provided. Returning fallback models.")
        return fallback_models

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append(ModelInfo(
                    model_id=m.name,
                    display_name=m.display_name,
                    description=m.description,
                    provider="google"
                ))
        _model_cache["gemini"] = (models, time.time())
        return models
    except Exception as e:
        logger.error(f"Failed to discover Gemini models: {e}")
        return fallback_models

def discover_claude_models(api_key: str) -> List[ModelInfo]:
    """
    Discovers available Claude models from the Anthropic API.
    Caches results for 5 minutes.
    """
    if "claude" in _model_cache:
        models, timestamp = _model_cache["claude"]
        if time.time() - timestamp < CACHE_DURATION:
            logger.debug("Returning cached Claude models.")
            return models

    fallback_models = [
        ModelInfo("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "Claude Sonnet 4.5", "anthropic"),
        ModelInfo("claude-sonnet-4-20250514", "Claude Sonnet 4", "Claude Sonnet 4", "anthropic"),
        ModelInfo("claude-opus-4-1-20250805", "Claude Opus 4.1", "Claude Opus 4.1", "anthropic"),
        ModelInfo("claude-opus-4-20250514", "Claude Opus 4", "Claude Opus 4", "anthropic"),
        ModelInfo("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "Claude Haiku 4.5", "anthropic"),
    ]

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not provided. Returning fallback models.")
        return fallback_models

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.models.list()
        
        models = []
        for model in response.data:
            models.append(ModelInfo(
                model_id=model.id,
                display_name=model.display_name,
                description=f"Provider: Anthropic, Model: {model.display_name}",
                provider="anthropic"
            ))
        
        if not models:
            logger.warning("No Claude models found from API. Returning fallback models.")
            models = fallback_models

        _model_cache["claude"] = (models, time.time())
        logger.info(f"Discovered {len(models)} Claude models.")
        return models
    except Exception as e:
        logger.error(f"Failed to discover Claude models: {e}")
        return fallback_models


def discover_ollama_models() -> List[ModelInfo]:
    """
    Discovers available Ollama models by running `ollama list`.
    Caches results for 5 minutes.
    """
    if "ollama" in _model_cache:
        models, timestamp = _model_cache["ollama"]
        if time.time() - timestamp < CACHE_DURATION:
            logger.debug("Returning cached Ollama models.")
            return models

    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        if not lines or len(lines) < 2:
            _model_cache["ollama"] = ([], time.time())
            return []

        models = []
        # Skip header line
        for line in lines[1:]:
            parts = line.split()
            if parts:
                model_name = parts[0]
                models.append(ModelInfo(
                    model_id=model_name,
                    display_name=model_name,
                    description=f"Provider: Ollama",
                    provider="ollama"
                ))
        
        _model_cache["ollama"] = (models, time.time())
        logger.info(f"Discovered {len(models)} Ollama models.")
        return models
    except FileNotFoundError:
        logger.warning("`ollama` command not found. Please make sure Ollama is installed and in your PATH.")
        _model_cache["ollama"] = ([], time.time())
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute `ollama list`: {e.stderr}")
        _model_cache["ollama"] = ([], time.time())
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while discovering Ollama models: {e}")
        _model_cache["ollama"] = ([], time.time())
        return []


def get_available_models(
    gemini_api_key: str | None = None,
    claude_api_key: str | None = None,
) -> Dict[str, List[ModelInfo]]:
    """
    Convenience helper that returns available models from all providers.

    Args:
        gemini_api_key: Optional Gemini API key.
        claude_api_key: Optional Claude API key.

    Returns:
        Dict[str, List[ModelInfo]]: Mapping of provider name to model list.
    """
    return {
        "gemini": discover_gemini_models(gemini_api_key or ""),
        "claude": discover_claude_models(claude_api_key or ""),
        "ollama": discover_ollama_models(),
    }
