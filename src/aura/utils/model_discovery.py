
import dataclasses
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
    Discovers available Claude models.
    Since the Anthropic SDK doesn't have a model listing API, this returns a hardcoded list.
    Caches results for 5 minutes.
    """
    if "claude" in _model_cache:
        models, timestamp = _model_cache["claude"]
        if time.time() - timestamp < CACHE_DURATION:
            return models

    # Hardcoded list as Anthropic SDK doesn't support model listing
    models = [
        ModelInfo("claude-3-opus-20240229", "Claude 3 Opus", "Most powerful model for complex tasks.", "anthropic"),
        ModelInfo("claude-3-sonnet-20240229", "Claude 3 Sonnet", "Balanced model for performance and speed.", "anthropic"),
        ModelInfo("claude-3-haiku-20240307", "Claude 3 Haiku", "Fastest model for near-instant responses.", "anthropic"),
    ]
    
    _model_cache["claude"] = (models, time.time())
    return models
