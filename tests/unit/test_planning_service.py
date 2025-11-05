"""Comprehensive tests for PlanningService quality validation."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Set

import pytest

from aura.services.planning_service import PlanningService, SessionPlan
from tests.fixtures.test_prompts import (
    ALL_PROMPTS,
    COMPLEX_PROMPTS,
    MEDIUM_PROMPTS,
    SIMPLE_PROMPTS,
    TestPrompt,
)


# ============================================================================
# Test Utilities
# ============================================================================


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def extract_file_paths(text: str) -> List[str]:
    """Extract file path hints from session prompts."""
    # Look for patterns like "File: path/to/file.py" or "Create: path/to/file.py"
    patterns = [
        r"File:\s*([a-zA-Z0-9_./]+\.py)",
        r"Create:\s*([a-zA-Z0-9_./]+\.py)",
        r"Edit:\s*([a-zA-Z0-9_./]+\.py)",
        r"`([a-zA-Z0-9_./]+\.py)`",
    ]
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        paths.extend(matches)
    return paths


def detect_duplicate_concepts(prompts: List[str]) -> List[tuple[str, str]]:
    """Detect potential duplicate work across session prompts."""
    duplicates = []
    # Key phrases that indicate specific work
    key_phrases = [
        "user model",
        "authentication",
        "database",
        "routes",
        "api endpoints",
        "validation",
        "error handling",
        "tests",
    ]

    for phrase in key_phrases:
        occurrences = []
        for idx, prompt in enumerate(prompts):
            if phrase in prompt.lower():
                # Check if it's creation vs usage
                is_creation = any(
                    word in prompt.lower()
                    for word in ["create", "build", "implement", "add"]
                )
                if is_creation:
                    occurrences.append((idx, "create"))
                else:
                    occurrences.append((idx, "use"))

        # If same phrase is created multiple times, that's a duplicate
        creation_count = sum(1 for _, action in occurrences if action == "create")
        if creation_count > 1:
            duplicates.append(
                (phrase, f"Created in {creation_count} different sessions")
            )

    return duplicates


def has_small_file_emphasis(prompt: str) -> bool:
    """Check if prompt emphasizes small, focused implementations."""
    emphasis_phrases = [
        "only",
        "single responsibility",
        "focused",
        "small",
        "concise",
        "minimal",
        "keep it simple",
        "one purpose",
    ]
    prompt_lower = prompt.lower()
    return any(phrase in prompt_lower for phrase in emphasis_phrases)


def measure_reasoning_quality(reasoning: str) -> dict:
    """Analyze reasoning quality."""
    return {
        "length": len(reasoning),
        "word_count": count_words(reasoning),
        "has_because": "because" in reasoning.lower(),
        "has_strategy": any(
            word in reasoning.lower()
            for word in ["strategy", "approach", "structure", "organize"]
        ),
        "explains_decomposition": any(
            word in reasoning.lower()
            for word in ["split", "separate", "divide", "break", "session"]
        ),
    }


def calculate_plan_similarity(plan1: SessionPlan, plan2: SessionPlan) -> float:
    """Calculate similarity between two plans (0.0 to 1.0)."""
    # Compare session count
    session_count_diff = abs(len(plan1.sessions) - len(plan2.sessions))
    session_count_similarity = max(0, 1 - (session_count_diff * 0.2))

    # Compare session names
    names1 = {s.name.lower() for s in plan1.sessions}
    names2 = {s.name.lower() for s in plan2.sessions}
    name_overlap = len(names1 & names2) / max(len(names1), len(names2))

    # Average similarities
    return (session_count_similarity + name_overlap) / 2


# ============================================================================
# Test 1: Session Count Validation
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", ALL_PROMPTS, ids=lambda p: p.name)
def test_session_count_within_range(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify plans have appropriate session counts (3-7 for normal projects)."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    min_expected, max_expected = test_prompt.expected_session_range
    session_count = len(plan.sessions)

    assert (
        min_expected <= session_count <= max_expected
    ), f"Expected {min_expected}-{max_expected} sessions, got {session_count}"

    # Hard constraint: no more than 7 sessions for any project
    assert session_count <= 7, f"Too many sessions: {session_count}. Plans should be 3-7 sessions."


# ============================================================================
# Test 2: Session Focus Validation
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", SIMPLE_PROMPTS + MEDIUM_PROMPTS, ids=lambda p: p.name)
def test_session_names_are_focused(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify session names are specific and concise (2-5 words)."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    vague_keywords = ["implementation", "add features", "setup", "main", "core", "general"]

    for session in plan.sessions:
        word_count = count_words(session.name)

        # Session names should be 2-7 words (a bit more lenient)
        assert (
            2 <= word_count <= 7
        ), f"Session name '{session.name}' has {word_count} words, expected 2-7"

        # Session names should not be overly vague
        name_lower = session.name.lower()
        for vague_word in vague_keywords:
            assert (
                vague_word not in name_lower
            ), f"Session name '{session.name}' contains vague keyword '{vague_word}'"


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_session_prompts_are_detailed(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify session prompts are detailed and specific."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    for session in plan.sessions:
        # Session prompts should be substantial (at least 20 words)
        word_count = count_words(session.prompt)
        assert (
            word_count >= 20
        ), f"Session '{session.name}' prompt is too brief: {word_count} words"

        # Should not be just the session name
        assert (
            session.prompt.lower() != session.name.lower()
        ), f"Session prompt is just the name: '{session.prompt}'"


# ============================================================================
# Test 3: Time Estimation Validation
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", ALL_PROMPTS[:5], ids=lambda p: p.name)
def test_time_estimates_are_reasonable(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify time estimates are 10-25 minutes per session."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    for session in plan.sessions:
        assert (
            10 <= session.estimated_minutes <= 25
        ), f"Session '{session.name}' estimate {session.estimated_minutes}min is outside 10-25min range"

    # Total time should be reasonable
    total = plan.total_estimated_minutes
    session_count = len(plan.sessions)
    assert (
        total >= session_count * 10
    ), f"Total time {total}min is too low for {session_count} sessions"
    assert (
        total <= session_count * 25
    ), f"Total time {total}min is too high for {session_count} sessions"


# ============================================================================
# Test 4: Dependency Logic Validation
# ============================================================================


@pytest.mark.integration
def test_dependencies_are_logical(
    planning_service: PlanningService,
    empty_project_context: str,
):
    """Verify dependency chains make sense."""
    # Use Todo API as it naturally has dependencies
    prompt = "Build a REST API for a todo application with user authentication and CRUD operations"
    plan = planning_service.plan_sessions(prompt, empty_project_context)

    # Collect all session names
    session_names = {session.name for session in plan.sessions}

    # Check dependencies
    for session in plan.sessions:
        if session.dependencies:
            for dep in session.dependencies:
                # Each dependency should refer to an actual session name
                assert (
                    dep in session_names
                ), f"Session '{session.name}' depends on unknown session '{dep}'"


@pytest.mark.integration
def test_later_sessions_reference_earlier_work(
    planning_service: PlanningService,
    empty_project_context: str,
):
    """Verify later sessions reference earlier work properly."""
    prompt = "Build a blog API with user authentication, posts, and comments"
    plan = planning_service.plan_sessions(prompt, empty_project_context)

    # If we have multiple sessions, later ones should ideally reference earlier ones
    if len(plan.sessions) >= 3:
        later_sessions = plan.sessions[2:]  # Sessions after the first two

        reference_count = 0
        for session in later_sessions:
            # Check if prompt references existing work
            if any(
                word in session.prompt.lower()
                for word in ["existing", "use", "import", "from", "previous"]
            ):
                reference_count += 1

        # At least some later sessions should reference earlier work
        # This is a soft check - we expect at least 1 reference
        assert (
            reference_count >= 1
        ), "Later sessions should reference earlier work with words like 'existing', 'use', etc."


# ============================================================================
# Test 5: No Duplication Detection
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:3], ids=lambda p: p.name)
def test_no_duplicate_work(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify sessions don't duplicate work."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    prompts = [session.prompt for session in plan.sessions]
    duplicates = detect_duplicate_concepts(prompts)

    # Allow some duplicates for things like "tests" which might appear multiple times
    # But flag if too many
    assert (
        len(duplicates) <= 1
    ), f"Found duplicate work across sessions: {duplicates}"


# ============================================================================
# Test 6: File Organization Hints
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_file_path_hints_present(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify session prompts include file path hints."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    sessions_with_paths = 0
    for session in plan.sessions:
        paths = extract_file_paths(session.prompt)
        if paths:
            sessions_with_paths += 1

    # At least half of sessions should have file path hints
    expected_min = len(plan.sessions) // 2
    assert (
        sessions_with_paths >= expected_min
    ), f"Only {sessions_with_paths}/{len(plan.sessions)} sessions have file path hints"


# ============================================================================
# Test 7: Small File Emphasis
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_small_file_emphasis(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify prompts emphasize small, focused implementations."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    sessions_with_emphasis = 0
    for session in plan.sessions:
        if has_small_file_emphasis(session.prompt):
            sessions_with_emphasis += 1

    # At least some sessions should emphasize small files
    # This is a softer requirement - at least 30% of sessions
    expected_min = max(1, len(plan.sessions) // 3)
    assert (
        sessions_with_emphasis >= expected_min
    ), f"Only {sessions_with_emphasis}/{len(plan.sessions)} sessions emphasize small files"


# ============================================================================
# Test 8: Reasoning Quality
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", ALL_PROMPTS[:5], ids=lambda p: p.name)
def test_reasoning_quality(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
):
    """Verify plan reasoning is substantial and explains decomposition."""
    plan = planning_service.plan_sessions(test_prompt.prompt, empty_project_context)

    quality = measure_reasoning_quality(plan.reasoning)

    # Reasoning should be substantial
    assert (
        quality["length"] >= 50
    ), f"Reasoning too short: {quality['length']} characters"
    assert quality["word_count"] >= 10, f"Reasoning too brief: {quality['word_count']} words"

    # Should explain strategy
    assert (
        quality["explains_decomposition"]
    ), "Reasoning should explain decomposition strategy"


# ============================================================================
# Test 9: Consistency Across Runs
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_consistency_across_runs(
    planning_service: PlanningService,
    empty_project_context: str,
):
    """Verify plans are relatively consistent across multiple runs."""
    prompt = "Build a REST API for a todo application with authentication"

    # Generate 3 plans
    plans = []
    for _ in range(3):
        plan = planning_service.plan_sessions(prompt, empty_project_context)
        plans.append(plan)
        # Clear history between runs to ensure independence
        planning_service._chat_service.clear_history()

    # Check session counts are similar (within 1 of each other)
    session_counts = [len(p.sessions) for p in plans]
    count_range = max(session_counts) - min(session_counts)
    assert count_range <= 1, f"Session counts vary too much: {session_counts}"

    # Check plans are somewhat similar
    similarity_01 = calculate_plan_similarity(plans[0], plans[1])
    similarity_02 = calculate_plan_similarity(plans[0], plans[2])
    similarity_12 = calculate_plan_similarity(plans[1], plans[2])

    avg_similarity = (similarity_01 + similarity_02 + similarity_12) / 3

    # Plans should be at least 60% similar
    assert (
        avg_similarity >= 0.6
    ), f"Plans are too inconsistent. Average similarity: {avg_similarity:.2f}"


# ============================================================================
# Test 10: Project Context Integration
# ============================================================================


@pytest.mark.integration
def test_existing_project_context_adaptation(
    planning_service: PlanningService,
    sample_project_context: str,
):
    """Verify plans adapt to existing project context."""
    prompt = "Add user authentication to the application"
    plan = planning_service.plan_sessions(prompt, sample_project_context)

    # At least one session should reference existing code
    references_existing = False
    for session in plan.sessions:
        if any(
            word in session.prompt.lower()
            for word in ["existing", "current", "main.py", "src/"]
        ):
            references_existing = True
            break

    assert (
        references_existing
    ), "Plan should reference existing project structure"


@pytest.mark.integration
def test_empty_vs_existing_project_difference(
    planning_service: PlanningService,
    empty_project_context: str,
    sample_project_context: str,
):
    """Verify plans differ appropriately for empty vs existing projects."""
    prompt = "Add user authentication"

    plan_empty = planning_service.plan_sessions(prompt, empty_project_context)
    planning_service._chat_service.clear_history()
    plan_existing = planning_service.plan_sessions(prompt, sample_project_context)

    # Plans should be different
    similarity = calculate_plan_similarity(plan_empty, plan_existing)

    # They should not be identical (similarity < 0.9)
    assert similarity < 0.9, f"Plans are too similar for different contexts: {similarity:.2f}"


# ============================================================================
# Regression Test Infrastructure
# ============================================================================


REGRESSION_DIR = Path(__file__).parent.parent / "fixtures" / "regression_plans"


def save_plan_as_fixture(plan: SessionPlan, test_name: str) -> None:
    """Save a plan as a regression fixture."""
    REGRESSION_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = REGRESSION_DIR / f"{test_name}.json"

    plan_data = {
        "sessions": [
            {
                "name": s.name,
                "prompt": s.prompt,
                "estimated_minutes": s.estimated_minutes,
                "dependencies": s.dependencies,
            }
            for s in plan.sessions
        ],
        "total_estimated_minutes": plan.total_estimated_minutes,
        "reasoning": plan.reasoning,
    }

    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(plan_data, f, indent=2)


@pytest.mark.integration
@pytest.mark.slow
def test_save_successful_plans_as_fixtures(
    planning_service: PlanningService,
    empty_project_context: str,
):
    """Generate and save successful plans for regression testing."""
    test_cases = [
        ("todo_api", "Build a REST API for a todo application with authentication"),
        ("blog_platform", "Build a blog platform with posts, comments, and user management"),
    ]

    for test_name, prompt in test_cases:
        plan = planning_service.plan_sessions(prompt, empty_project_context)

        # Only save if it passes basic quality checks
        if 3 <= len(plan.sessions) <= 7:
            save_plan_as_fixture(plan, test_name)

        planning_service._chat_service.clear_history()


# ============================================================================
# Summary Test
# ============================================================================


@pytest.mark.integration
def test_comprehensive_quality_check(
    planning_service: PlanningService,
    empty_project_context: str,
):
    """Run a comprehensive quality check on a single plan."""
    prompt = "Build a REST API for a todo application with user authentication and CRUD operations"
    plan = planning_service.plan_sessions(prompt, empty_project_context)

    issues = []

    # Check 1: Session count
    if not (3 <= len(plan.sessions) <= 7):
        issues.append(f"Session count {len(plan.sessions)} outside 3-7 range")

    # Check 2: Session names
    for session in plan.sessions:
        word_count = count_words(session.name)
        if not (2 <= word_count <= 7):
            issues.append(f"Session name '{session.name}' has {word_count} words")

    # Check 3: Time estimates
    for session in plan.sessions:
        if not (10 <= session.estimated_minutes <= 25):
            issues.append(
                f"Session '{session.name}' estimate {session.estimated_minutes}min outside range"
            )

    # Check 4: Reasoning
    quality = measure_reasoning_quality(plan.reasoning)
    if quality["length"] < 50:
        issues.append(f"Reasoning too short: {quality['length']} characters")

    # Check 5: File paths
    sessions_with_paths = sum(
        1 for s in plan.sessions if extract_file_paths(s.prompt)
    )
    if sessions_with_paths < len(plan.sessions) // 2:
        issues.append(
            f"Only {sessions_with_paths}/{len(plan.sessions)} sessions have file paths"
        )

    # Print summary
    print("\n" + "=" * 70)
    print("COMPREHENSIVE QUALITY CHECK")
    print("=" * 70)
    print(f"Prompt: {prompt}")
    print(f"Sessions: {len(plan.sessions)}")
    print(f"Total time: {plan.total_estimated_minutes} minutes")
    print(f"Reasoning: {plan.reasoning}")
    print("\nSession breakdown:")
    for idx, session in enumerate(plan.sessions, 1):
        print(f"  {idx}. {session.name} ({session.estimated_minutes}min)")
        if session.dependencies:
            print(f"     Dependencies: {', '.join(session.dependencies)}")
    print("\n" + "=" * 70)

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  [FAIL] {issue}")
        print("=" * 70)
        pytest.fail(f"Quality check failed with {len(issues)} issues:\n" + "\n".join(issues))
    else:
        print("[PASS] All quality checks passed!")
        print("=" * 70)
