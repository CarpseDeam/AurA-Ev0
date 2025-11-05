"""Standard test prompts for planning validation."""

from dataclasses import dataclass
from typing import List


@dataclass
class TestPrompt:
    """A test prompt with metadata."""

    name: str
    prompt: str
    complexity: str  # "simple", "medium", "complex"
    domain: str  # "cli", "web_api", "data", "full_app"
    expected_session_range: tuple[int, int]  # min, max expected sessions


# Simple Projects (3-4 sessions expected)
SIMPLE_PROMPTS: List[TestPrompt] = [
    TestPrompt(
        name="Calculator CLI",
        prompt="Build a command-line calculator that supports basic arithmetic operations",
        complexity="simple",
        domain="cli",
        expected_session_range=(3, 4),
    ),
    TestPrompt(
        name="File Organizer",
        prompt="Create a Python script that organizes files in a directory by extension",
        complexity="simple",
        domain="cli",
        expected_session_range=(3, 4),
    ),
    TestPrompt(
        name="Temperature Converter",
        prompt="Build a temperature converter CLI tool that converts between Celsius, Fahrenheit, and Kelvin",
        complexity="simple",
        domain="cli",
        expected_session_range=(3, 4),
    ),
]

# Medium Projects (4-6 sessions expected)
MEDIUM_PROMPTS: List[TestPrompt] = [
    TestPrompt(
        name="Todo API",
        prompt="Build a REST API for a todo application with user authentication, CRUD operations for todos, and proper error handling",
        complexity="medium",
        domain="web_api",
        expected_session_range=(4, 6),
    ),
    TestPrompt(
        name="URL Shortener",
        prompt="Create a URL shortener service with API endpoints for creating short URLs, retrieving original URLs, and tracking click statistics",
        complexity="medium",
        domain="web_api",
        expected_session_range=(4, 6),
    ),
    TestPrompt(
        name="CSV Data Processor",
        prompt="Build a data processing pipeline that reads CSV files, validates data, applies transformations, and exports to multiple formats with error reporting",
        complexity="medium",
        domain="data",
        expected_session_range=(4, 6),
    ),
    TestPrompt(
        name="Task Queue System",
        prompt="Implement a simple task queue system with worker processes, job scheduling, retry logic, and status tracking",
        complexity="medium",
        domain="full_app",
        expected_session_range=(4, 6),
    ),
]

# Complex Projects (5-7 sessions expected)
COMPLEX_PROMPTS: List[TestPrompt] = [
    TestPrompt(
        name="Blog Platform",
        prompt="Build a blog platform with user management, posts with rich text, comments, tags, search functionality, and an admin dashboard",
        complexity="complex",
        domain="full_app",
        expected_session_range=(5, 7),
    ),
    TestPrompt(
        name="E-commerce API",
        prompt="Create an e-commerce REST API with user authentication, product catalog, shopping cart, order processing, payment integration, and inventory management",
        complexity="complex",
        domain="web_api",
        expected_session_range=(5, 7),
    ),
    TestPrompt(
        name="Chat Application",
        prompt="Build a real-time chat application with user authentication, multiple chat rooms, direct messages, message history, and online status indicators",
        complexity="complex",
        domain="full_app",
        expected_session_range=(5, 7),
    ),
    TestPrompt(
        name="Data Analytics Dashboard",
        prompt="Create a data analytics dashboard that imports data from multiple sources, performs statistical analysis, generates visualizations, and exports reports",
        complexity="complex",
        domain="data",
        expected_session_range=(5, 7),
    ),
]

# Edge Cases
EDGE_CASE_PROMPTS: List[TestPrompt] = [
    TestPrompt(
        name="Minimal Task",
        prompt="Create a hello world function",
        complexity="simple",
        domain="cli",
        expected_session_range=(1, 3),
    ),
    TestPrompt(
        name="Vague Request",
        prompt="Build something cool for managing data",
        complexity="medium",
        domain="data",
        expected_session_range=(3, 6),
    ),
    TestPrompt(
        name="Very Complex",
        prompt="Build a complete social media platform with user profiles, posts, stories, direct messaging, notifications, friend system, content moderation, analytics, and admin tools",
        complexity="complex",
        domain="full_app",
        expected_session_range=(6, 8),  # Might exceed 7, which should trigger validation
    ),
]

# All prompts combined
ALL_PROMPTS: List[TestPrompt] = SIMPLE_PROMPTS + MEDIUM_PROMPTS + COMPLEX_PROMPTS
ALL_PROMPTS_WITH_EDGE_CASES: List[TestPrompt] = ALL_PROMPTS + EDGE_CASE_PROMPTS


def get_prompts_by_complexity(complexity: str) -> List[TestPrompt]:
    """Get all prompts matching a specific complexity level."""
    return [p for p in ALL_PROMPTS if p.complexity == complexity]


def get_prompts_by_domain(domain: str) -> List[TestPrompt]:
    """Get all prompts matching a specific domain."""
    return [p for p in ALL_PROMPTS if p.domain == domain]
