"""Mock API responses for fast testing."""

MOCK_SIMPLE_PLAN = {
    "sessions": [
        {
            "name": "Calculator Functions",
            "prompt": (
                "Create pure math helpers for addition, subtraction, multiplication, and division with docstrings and "
                "guard clauses. Validate numeric inputs, raise helpful errors, and keep each helper small for reuse. "
                "File: src/calculator.py. Keep the implementation focused and under one hundred lines."
            ),
            "estimated_minutes": 12,
            "dependencies": [],
        },
        {
            "name": "Calculator CLI Commands",
            "prompt": (
                "Build a Typer-based CLI that imports the calculator helpers and wires user friendly commands. Format "
                "output consistently, reuse existing utility functions, and keep the CLI logic minimal. File: src/main.py. "
                "Use existing modules where possible and keep every command small."
            ),
            "estimated_minutes": 10,
            "dependencies": ["Calculator Functions"],
        },
        {
            "name": "Unit Tests",
            "prompt": (
                "Write pytest cases for happy paths and edge cases covering each calculator helper. Use parametrization, "
                "include division by zero scenarios, and keep every test focused on a single behavior. File: tests/test_calculator.py. "
                "Keep the test suite concise and easy to iterate."
            ),
            "estimated_minutes": 14,
            "dependencies": ["Calculator Functions"],
        },
    ],
    "total_estimated_minutes": 36,
    "reasoning": (
        "Split the work into reusable math helpers, a focused CLI wrapper, and a lean pytest suite so each piece stays small, "
        "easy to iterate, and aligned with the single-responsibility goal."
    ),
}

MOCK_MEDIUM_PLAN = {
    "sessions": [
        {
            "name": "User Model Class",
            "prompt": (
                "Design a User dataclass with email, password hashing, and timestamp fields plus validation helpers. "
                "Keep the model focused, add bcrypt utilities, and document each step. File: src/models/user.py. "
                "Keep the module tight and under sixty lines."
            ),
            "estimated_minutes": 15,
            "dependencies": [],
        },
        {
            "name": "Auth Routes",
            "prompt": (
                "Implement login and logout routes that import the existing User model and reuse hashing utilities. "
                "Validate payloads, return JSON responses, and keep handler functions small. File: src/routes/auth.py. "
                "Use existing helpers so the routes stay minimal."
            ),
            "estimated_minutes": 12,
            "dependencies": ["User Model Class"],
        },
        {
            "name": "Todo CRUD Operations",
            "prompt": (
                "Create Todo model and CRUD routes that lean on the authentication flow and enforce ownership checks. "
                "Wire repository helpers, add error handling, and keep each controller focused. File: src/routes/todos.py. "
                "Reuse existing models and keep files compact."
            ),
            "estimated_minutes": 18,
            "dependencies": ["User Model Class"],
        },
        {
            "name": "Tests Suite",
            "prompt": (
                "Author pytest coverage for auth and todo flows using fixtures and helper factories. Cover happy paths, "
                "failure cases, and regression scenarios while keeping every test small. File: tests/test_auth_todos.py. "
                "Reuse existing factories to maintain concise tests."
            ),
            "estimated_minutes": 12,
            "dependencies": ["Auth Routes", "Todo CRUD Operations"],
        },
    ],
    "total_estimated_minutes": 57,
    "reasoning": (
        "Separate the domain foundation, route layers, and validation into distinct passes so we can split responsibilities, "
        "reuse prior work, and maintain compact modules throughout the stack."
    ),
}

MOCK_COMPLEX_PLAN = {
    "sessions": [
        {
            "name": "User Authentication Model",
            "prompt": (
                "Build the User model with JWT utilities, password hashing, and role fields while keeping helpers tightly "
                "scoped. Document each method and validate payloads. File: src/models/user.py. Keep the implementation small and "
                "extendable."
            ),
            "estimated_minutes": 16,
            "dependencies": [],
        },
        {
            "name": "Post Model Blog",
            "prompt": (
                "Create the Post model with relationships to the existing User model plus summary helpers and query managers. "
                "Ensure fields map to the database schema and keep data logic compact. File: src/models/post.py. "
                "Reuse shared utilities to stay concise."
            ),
            "estimated_minutes": 14,
            "dependencies": ["User Authentication Model"],
        },
        {
            "name": "Comment System",
            "prompt": (
                "Implement the Comment model and repository functions by importing the existing User and Post models. "
                "Add validation, soft delete hooks, and keep functions focused. File: src/models/comment.py. "
                "Reference previous work to avoid duplication."
            ),
            "estimated_minutes": 15,
            "dependencies": ["User Authentication Model", "Post Model Blog"],
        },
        {
            "name": "Auth Routes Endpoints",
            "prompt": (
                "Expose registration, login, refresh, and logout endpoints that use the existing authentication helpers. "
                "Wire response schemas, reuse validation, and keep handlers brief. File: src/routes/auth.py. "
                "Use previous modules rather than rewriting logic."
            ),
            "estimated_minutes": 12,
            "dependencies": ["User Authentication Model"],
        },
        {
            "name": "Blog CRUD Routes",
            "prompt": (
                "Add create, list, update, and delete routes for blog posts that import Post and Comment models. "
                "Ensure controllers reference existing services, enforce permissions, and remain small. File: src/routes/blog.py. "
                "Reuse established helpers to keep files tidy."
            ),
            "estimated_minutes": 18,
            "dependencies": ["Post Model Blog", "Auth Routes Endpoints"],
        },
        {
            "name": "Testing Suite Complete",
            "prompt": (
                "Compose an end-to-end pytest suite covering auth, posts, and comments with fixtures for seeded data. "
                "Reference existing factories, keep each test focused, and document regressions. File: tests/test_blog_app.py. "
                "Keep scenarios small and leverage previous work."
            ),
            "estimated_minutes": 12,
            "dependencies": ["Comment System", "Blog CRUD Routes"],
        },
    ],
    "total_estimated_minutes": 87,
    "reasoning": (
        "Layer the system by first solidifying authentication, then expanding to content models, before wiring routes and "
        "full coverage. This split keeps each session manageable and encourages reuse of earlier building blocks."
    ),
}


def get_mock_plan(complexity: str) -> dict:
    """Get mock plan by complexity level."""
    plans = {
        "simple": MOCK_SIMPLE_PLAN,
        "medium": MOCK_MEDIUM_PLAN,
        "complex": MOCK_COMPLEX_PLAN,
    }
    try:
        return plans[complexity]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unknown complexity: {complexity}") from exc
