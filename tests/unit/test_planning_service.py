"""Comprehensive tests for PlanningService quality validation."""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Set

import pytest

from aura.services.planning_service import PlanningService, SessionPlan
from tests.fixtures.test_prompts import (
    ALL_PROMPTS,
    QUICK_TEST_PROMPTS,
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
                is_creation = any(
                    word in prompt.lower() for word in ["create", "build", "implement", "add"]
                )
                if is_creation:
                    occurrences.append((idx, "create"))
                else:
                    occurrences.append((idx, "use"))

        creation_count = sum(1 for _, action in occurrences if action == "create")
        if creation_count > 1:
            duplicates.append((phrase, f"Created in {creation_count} different sessions"))

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
            word in reasoning.lower() for word in ["split", "separate", "divide", "break", "session"]
        ),
    }


def calculate_plan_similarity(plan1: SessionPlan, plan2: SessionPlan) -> float:
    """Calculate similarity between two plans (0.0 to 1.0)."""
    session_count_diff = abs(len(plan1.sessions) - len(plan2.sessions))
    session_count_similarity = max(0, 1 - (session_count_diff * 0.2))

    names1 = {s.name.lower() for s in plan1.sessions}
    names2 = {s.name.lower() for s in plan2.sessions}
    name_overlap = len(names1 & names2) / max(len(names1), len(names2))

    return (session_count_similarity + name_overlap) / 2


# ============================================================================
# Shared Assertions
# ============================================================================


def _assert_session_count_within_range(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)

    min_expected, max_expected = test_prompt.expected_session_range
    session_count = len(plan.sessions)

    assert (
        min_expected <= session_count <= max_expected
    ), f"Expected {min_expected}-{max_expected} sessions, got {session_count}"
    assert session_count <= 7, f"Too many sessions: {session_count}. Plans should be 3-7 sessions."


def _assert_session_names_are_focused(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)
    vague_keywords = ["implementation", "add features", "setup", "main", "core", "general"]

    for session in plan.sessions:
        word_count = count_words(session.name)
        assert 2 <= word_count <= 7, f"Session name '{session.name}' has {word_count} words, expected 2-7"

        name_lower = session.name.lower()
        for vague_word in vague_keywords:
            assert vague_word not in name_lower, (
                f"Session name '{session.name}' contains vague keyword '{vague_word}'"
            )


def _assert_session_prompts_are_detailed(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)

    for session in plan.sessions:
        word_count = count_words(session.prompt)
        assert (
            word_count >= 20
        ), f"Session '{session.name}' prompt is too brief: {word_count} words"
        assert session.prompt.lower() != session.name.lower(), (
            f"Session prompt is just the name: '{session.prompt}'"
        )


def _assert_time_estimates_are_reasonable(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)

    for session in plan.sessions:
        assert 10 <= session.estimated_minutes <= 25, (
            f"Session '{session.name}' estimate {session.estimated_minutes}min is outside 10-25min range"
        )

    total = plan.total_estimated_minutes
    session_count = len(plan.sessions)
    assert total >= session_count * 10, f"Total time {total}min is too low for {session_count} sessions"
    assert total <= session_count * 25, f"Total time {total}min is too high for {session_count} sessions"


def _assert_dependencies_are_logical(
    service: PlanningService,
    project_context: str,
    prompt: str,
) -> None:
    plan = service.plan_sessions(prompt, project_context)
    session_names = {session.name for session in plan.sessions}

    for session in plan.sessions:
        for dep in session.dependencies:
            assert dep in session_names, (
                f"Session '{session.name}' depends on unknown session '{dep}'"
            )


def _assert_later_sessions_reference_earlier_work(
    service: PlanningService,
    project_context: str,
    prompt: str,
) -> None:
    plan = service.plan_sessions(prompt, project_context)

    if len(plan.sessions) >= 3:
        later_sessions = plan.sessions[2:]
        reference_count = 0
        for session in later_sessions:
            if any(
                word in session.prompt.lower() for word in ["existing", "use", "import", "from", "previous"]
            ):
                reference_count += 1
        assert reference_count >= 1, (
            "Later sessions should reference earlier work with words like 'existing', 'use', etc."
        )


def _assert_no_duplicate_work(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)
    prompts = [session.prompt for session in plan.sessions]
    duplicates = detect_duplicate_concepts(prompts)
    assert len(duplicates) <= 1, f"Found duplicate work across sessions: {duplicates}"


def _assert_file_path_hints_present(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)
    sessions_with_paths = sum(1 for session in plan.sessions if extract_file_paths(session.prompt))
    expected_min = len(plan.sessions) // 2
    assert sessions_with_paths >= expected_min, (
        f"Only {sessions_with_paths}/{len(plan.sessions)} sessions have file path hints"
    )


def _assert_small_file_emphasis(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)
    sessions_with_emphasis = sum(1 for session in plan.sessions if has_small_file_emphasis(session.prompt))
    expected_min = max(1, len(plan.sessions) // 3)
    assert sessions_with_emphasis >= expected_min, (
        f"Only {sessions_with_emphasis}/{len(plan.sessions)} sessions emphasize small files"
    )


def _assert_reasoning_quality(
    service: PlanningService,
    project_context: str,
    test_prompt: TestPrompt,
) -> None:
    plan = service.plan_sessions(test_prompt.prompt, project_context)
    quality = measure_reasoning_quality(plan.reasoning)

    assert quality["length"] >= 50, f"Reasoning too short: {quality['length']} characters"
    assert quality["word_count"] >= 10, f"Reasoning too brief: {quality['word_count']} words"
    assert quality["explains_decomposition"], "Reasoning should explain decomposition strategy"


def _run_consistency_check_real(
    service: PlanningService,
    project_context: str,
    prompt: str,
) -> None:
    plans = []
    for _ in range(3):
        plan = service.plan_sessions(prompt, project_context)
        plans.append(plan)
        service._chat.clear_history()

    session_counts = [len(plan.sessions) for plan in plans]
    count_range = max(session_counts) - min(session_counts)
    assert count_range <= 1, f"Session counts vary too much: {session_counts}"

    similarity_01 = calculate_plan_similarity(plans[0], plans[1])
    similarity_02 = calculate_plan_similarity(plans[0], plans[2])
    similarity_12 = calculate_plan_similarity(plans[1], plans[2])
    avg_similarity = (similarity_01 + similarity_02 + similarity_12) / 3

    assert avg_similarity >= 0.6, (
        f"Plans are too inconsistent. Average similarity: {avg_similarity:.2f}"
    )


def _run_consistency_check_mocked(
    service: PlanningService,
    project_context: str,
    prompt: str,
) -> None:
    plans = [service.plan_sessions(prompt, project_context) for _ in range(3)]
    assert plans[0] == plans[1] == plans[2], "Mock planning service should be deterministic"


def _assert_existing_project_context_adaptation(
    service: PlanningService,
    project_context: str,
    prompt: str,
) -> None:
    plan = service.plan_sessions(prompt, project_context)
    references_existing = any(
        any(keyword in session.prompt.lower() for keyword in ["existing", "current", "main.py", "src/"])
        for session in plan.sessions
    )
    assert references_existing, "Plan should reference existing project structure"


def _assert_empty_vs_existing_project_difference(
    service: PlanningService,
    empty_project_context: str,
    sample_project_context: str,
    prompt: str,
) -> None:
    plan_empty = service.plan_sessions(prompt, empty_project_context)
    if hasattr(service, "_chat"):
        service._chat.clear_history()
    plan_existing = service.plan_sessions(prompt, sample_project_context)
    similarity = calculate_plan_similarity(plan_empty, plan_existing)
    assert similarity < 0.9, f"Plans are too similar for different contexts: {similarity:.2f}"


def _assert_empty_vs_existing_project_difference_mocked(
    service: PlanningService,
    empty_project_context: str,
    sample_project_context: str,
    prompt: str,
) -> None:
    plan_empty = service.plan_sessions(prompt, empty_project_context)
    plan_existing = service.plan_sessions(prompt, sample_project_context)
    assert plan_empty.reasoning != plan_existing.reasoning, (
        "Mock plans should adjust reasoning when project context changes"
    )


def _run_comprehensive_quality_check(
    service: PlanningService,
    project_context: str,
    prompt: str,
) -> None:
    plan = service.plan_sessions(prompt, project_context)
    issues = []

    if not (3 <= len(plan.sessions) <= 7):
        issues.append(f"Session count {len(plan.sessions)} outside 3-7 range")

    for session in plan.sessions:
        word_count = count_words(session.name)
        if not (2 <= word_count <= 7):
            issues.append(f"Session name '{session.name}' has {word_count} words")

    for session in plan.sessions:
        if not (10 <= session.estimated_minutes <= 25):
            issues.append(
                f"Session '{session.name}' estimate {session.estimated_minutes}min outside range"
            )

    quality = measure_reasoning_quality(plan.reasoning)
    if quality["length"] < 50:
        issues.append(f"Reasoning too short: {quality['length']} characters")

    sessions_with_paths = sum(1 for session in plan.sessions if extract_file_paths(session.prompt))
    if sessions_with_paths < len(plan.sessions) // 2:
        issues.append(
            f"Only {sessions_with_paths}/{len(plan.sessions)} sessions have file paths"
        )

    if issues:
        pytest.fail(f"Quality check failed with {len(issues)} issues:\n" + "\n".join(issues))


# ============================================================================
# Test 1: Session Count Validation
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_session_count_within_range_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify session counts with mocked API."""
    _assert_session_count_within_range(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_session_count_within_range_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """Live API smoke: ensure session counts stay in the expected range."""
    _assert_session_count_within_range(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 2: Session Focus Validation
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_session_names_are_focused_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked session names are focused."""
    _assert_session_names_are_focused(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_session_names_are_focused_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify real session names are focused."""
    _assert_session_names_are_focused(planning_service, empty_project_context, test_prompt)


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_session_prompts_are_detailed_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked session prompts are detailed."""
    _assert_session_prompts_are_detailed(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_session_prompts_are_detailed_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify real session prompts are detailed."""
    _assert_session_prompts_are_detailed(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 3: Time Estimation Validation
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_time_estimates_are_reasonable_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked time estimates are within range."""
    _assert_time_estimates_are_reasonable(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_time_estimates_are_reasonable_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify real time estimates are within range."""
    _assert_time_estimates_are_reasonable(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 4: Dependency Logic Validation
# ============================================================================


@pytest.mark.fast
def test_dependencies_are_logical_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """FAST: Verify mocked dependency chains make sense."""
    prompt = "Build a REST API for a todo application with user authentication and CRUD operations"
    _assert_dependencies_are_logical(mock_planning_service, empty_project_context, prompt)


@pytest.mark.integration
@pytest.mark.slow
def test_dependencies_are_logical_real(
    planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """SLOW: Verify dependency chains make sense."""
    prompt = "Build a REST API for a todo application with user authentication and CRUD operations"
    _assert_dependencies_are_logical(planning_service, empty_project_context, prompt)


@pytest.mark.fast
def test_later_sessions_reference_earlier_work_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """FAST: Verify mocked later sessions reference earlier work."""
    prompt = "Build a blog API with user authentication, posts, and comments"
    _assert_later_sessions_reference_earlier_work(mock_planning_service, empty_project_context, prompt)


@pytest.mark.integration
@pytest.mark.slow
def test_later_sessions_reference_earlier_work_real(
    planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """SLOW: Verify later sessions reference earlier work."""
    prompt = "Build a blog API with user authentication, posts, and comments"
    _assert_later_sessions_reference_earlier_work(planning_service, empty_project_context, prompt)


# ============================================================================
# Test 5: No Duplication Detection
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:3], ids=lambda p: p.name)
def test_no_duplicate_work_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked sessions don't duplicate work."""
    _assert_no_duplicate_work(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:3], ids=lambda p: p.name)
def test_no_duplicate_work_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify sessions don't duplicate work."""
    _assert_no_duplicate_work(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 6: File Organization Hints
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_file_path_hints_present_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked prompts include file path hints."""
    _assert_file_path_hints_present(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_file_path_hints_present_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify prompts include file path hints."""
    _assert_file_path_hints_present(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 7: Small File Emphasis
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_small_file_emphasis_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked prompts emphasize small files."""
    _assert_small_file_emphasis(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", MEDIUM_PROMPTS[:2], ids=lambda p: p.name)
def test_small_file_emphasis_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify prompts emphasize small files."""
    _assert_small_file_emphasis(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 8: Reasoning Quality
# ============================================================================


@pytest.mark.fast
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_reasoning_quality_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """FAST: Verify mocked reasoning quality."""
    _assert_reasoning_quality(mock_planning_service, empty_project_context, test_prompt)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("test_prompt", QUICK_TEST_PROMPTS, ids=lambda p: p.name)
def test_reasoning_quality_real(
    planning_service: PlanningService,
    empty_project_context: str,
    test_prompt: TestPrompt,
) -> None:
    """SLOW: Verify reasoning quality."""
    _assert_reasoning_quality(planning_service, empty_project_context, test_prompt)


# ============================================================================
# Test 9: Consistency Across Runs
# ============================================================================


@pytest.mark.fast
def test_consistency_across_runs_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """FAST: Verify mocked plans are deterministic."""
    prompt = "Build a REST API for a todo application with authentication"
    _run_consistency_check_mocked(mock_planning_service, empty_project_context, prompt)


@pytest.mark.integration
@pytest.mark.slow
def test_consistency_across_runs_real(
    planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """SLOW: Verify plans are relatively consistent across runs."""
    prompt = "Build a REST API for a todo application with authentication"
    _run_consistency_check_real(planning_service, empty_project_context, prompt)


# ============================================================================
# Test 10: Project Context Integration
# ============================================================================


@pytest.mark.fast
def test_existing_project_context_adaptation_mocked(
    mock_planning_service: PlanningService,
    sample_project_context: str,
) -> None:
    """FAST: Verify mocked plans adapt to existing project context."""
    prompt = "Add user authentication to the application"
    _assert_existing_project_context_adaptation(mock_planning_service, sample_project_context, prompt)


@pytest.mark.integration
@pytest.mark.slow
def test_existing_project_context_adaptation_real(
    planning_service: PlanningService,
    sample_project_context: str,
) -> None:
    """SLOW: Verify plans adapt to existing project context."""
    prompt = "Add user authentication to the application"
    _assert_existing_project_context_adaptation(planning_service, sample_project_context, prompt)


@pytest.mark.fast
def test_empty_vs_existing_project_difference_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    sample_project_context: str,
) -> None:
    """FAST: Verify mocked plans differ for empty vs existing projects."""
    prompt = "Add user authentication"
    _assert_empty_vs_existing_project_difference_mocked(
        mock_planning_service,
        empty_project_context,
        sample_project_context,
        prompt,
    )


@pytest.mark.integration
@pytest.mark.slow
def test_empty_vs_existing_project_difference_real(
    planning_service: PlanningService,
    empty_project_context: str,
    sample_project_context: str,
) -> None:
    """SLOW: Verify plans differ for empty vs existing projects."""
    prompt = "Add user authentication"
    _assert_empty_vs_existing_project_difference(
        planning_service,
        empty_project_context,
        sample_project_context,
        prompt,
    )


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
                "name": session.name,
                "prompt": session.prompt,
                "estimated_minutes": session.estimated_minutes,
                "dependencies": session.dependencies,
            }
            for session in plan.sessions
        ],
        "total_estimated_minutes": plan.total_estimated_minutes,
        "reasoning": plan.reasoning,
    }

    with open(fixture_path, "w", encoding="utf-8") as handle:
        json.dump(plan_data, handle, indent=2)


@pytest.mark.fast
def test_save_successful_plans_as_fixtures_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAST: Verify mocked plans can be saved as fixtures."""
    monkeypatch.setattr(sys.modules[__name__], "REGRESSION_DIR", tmp_path)

    test_name = "mocked_todo_api"
    prompt = "Build a REST API for a todo application with authentication"
    plan = mock_planning_service.plan_sessions(prompt, empty_project_context)
    save_plan_as_fixture(plan, test_name)

    saved_path = tmp_path / f"{test_name}.json"
    assert saved_path.exists(), "Mock fixture file was not created"


@pytest.mark.integration
@pytest.mark.slow
def test_save_successful_plans_as_fixtures_real(
    planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """SLOW: Generate and save successful plans for regression testing."""
    test_cases = [
        ("todo_api", "Build a REST API for a todo application with authentication"),
        ("blog_platform", "Build a blog platform with posts, comments, and user management"),
    ]

    for test_name, prompt in test_cases:
        plan = planning_service.plan_sessions(prompt, empty_project_context)

        if 3 <= len(plan.sessions) <= 7:
            save_plan_as_fixture(plan, test_name)

        planning_service._chat.clear_history()


# ============================================================================
# Summary Test
# ============================================================================


@pytest.mark.fast
def test_comprehensive_quality_check_mocked(
    mock_planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """FAST: Comprehensive quality check using mocked service."""
    prompt = "Build a REST API for a todo application with user authentication and CRUD operations"
    _run_comprehensive_quality_check(mock_planning_service, empty_project_context, prompt)


@pytest.mark.integration
@pytest.mark.slow
def test_comprehensive_quality_check_real(
    planning_service: PlanningService,
    empty_project_context: str,
) -> None:
    """SLOW: Comprehensive quality check on real service."""
    prompt = "Build a REST API for a todo application with user authentication and CRUD operations"
    _run_comprehensive_quality_check(planning_service, empty_project_context, prompt)
