"""
Utilities for Claude API prompt caching to reduce token costs by 60-80%.

Prompt caching allows you to cache frequently used content (system prompts, tool definitions)
that doesn't change between requests in a session. Cached content gets a 90% discount on
token costs after the initial cache write.

Cache behavior:
- Persists for 5 minutes of inactivity
- Refreshes on each use
- First call incurs small cache write premium (~25% more)
- Subsequent calls get 90% discount on cached tokens
- Minimum cacheable size: 1024 tokens

Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""
from typing import Any


def build_cached_system_prompt(prompt: str) -> list[dict[str, Any]]:
    """
    Convert a system prompt string into a cacheable format.

    Args:
        prompt: The system prompt text to cache.

    Returns:
        A list with a single text block marked for caching.

    Example:
        >>> system = build_cached_system_prompt("You are a helpful assistant...")
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-5-20250929",
        ...     system=system,  # This will be cached
        ...     messages=[...]
        ... )
    """
    return [
        {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"}
        }
    ]


def mark_tools_for_caching(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Add cache_control metadata to the last tool in a tool list.

    According to Anthropic's API, only the last item in a list can have cache_control.
    This marks the entire tools array for caching by adding the marker to the final tool.

    Args:
        tools: List of Anthropic tool schemas.

    Returns:
        The same tool list with cache_control added to the last tool.

    Example:
        >>> tools = [{"name": "tool1", ...}, {"name": "tool2", ...}]
        >>> cached_tools = mark_tools_for_caching(tools)
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-5-20250929",
        ...     system=[...],
        ...     tools=cached_tools,  # Tools will be cached
        ...     messages=[...]
        ... )
    """
    if not tools:
        return tools

    # Make a shallow copy to avoid mutating the original list
    tools_copy = tools.copy()

    # Add cache_control to the last tool
    # This caches the entire tools array per Anthropic's API design
    if len(tools_copy) > 0:
        # Make a copy of the last tool to avoid mutating shared references
        last_tool = tools_copy[-1].copy()
        last_tool["cache_control"] = {"type": "ephemeral"}
        tools_copy[-1] = last_tool

    return tools_copy


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for validation.

    Anthropic uses ~4 characters per token on average for English text.
    This is a conservative estimate - actual tokenization may vary.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated number of tokens.
    """
    return len(text) // 4


def is_cacheable(text: str, min_tokens: int = 1024) -> bool:
    """
    Check if a text block meets the minimum token requirement for caching.

    Args:
        text: The text to check.
        min_tokens: Minimum tokens required (default: 1024).

    Returns:
        True if the text is likely large enough to cache.
    """
    return estimate_token_count(text) >= min_tokens


def build_cached_system_and_tools(
    system_prompt: str,
    tools: list[dict[str, Any]],
    cache_system: bool = True,
    cache_tools: bool = True,
) -> tuple[list[dict[str, Any]] | str, list[dict[str, Any]]]:
    """
    Convenience function to prepare both system and tools with caching.

    Args:
        system_prompt: The system prompt text.
        tools: List of tool schemas.
        cache_system: Whether to enable caching for system prompt (default: True).
        cache_tools: Whether to enable caching for tools (default: True).

    Returns:
        Tuple of (system, tools) ready for API call with caching enabled.

    Example:
        >>> system, tools = build_cached_system_and_tools(
        ...     system_prompt=ANALYST_PROMPT,
        ...     tools=investigation_tools
        ... )
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-5-20250929",
        ...     system=system,
        ...     tools=tools,
        ...     messages=[...]
        ... )
    """
    # Prepare system prompt
    prepared_system = build_cached_system_prompt(system_prompt) if cache_system else system_prompt

    # Prepare tools
    prepared_tools = mark_tools_for_caching(tools) if cache_tools and tools else tools

    return prepared_system, prepared_tools
