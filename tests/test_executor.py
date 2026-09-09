"""
Tests for the NEXUS Executor.
"""

from app.executor import Executor
from app.planner import PlanStep
from app.tools import (
    CalculatorTool,
    ToolRegistry,
)


def create_registry() -> ToolRegistry:
    """Create a registry containing the calculator."""

    registry = ToolRegistry()

    registry.register(
        CalculatorTool()
    )

    return registry


def test_executor_runs_single_step() -> None:
    """Executor should execute a single planned step."""

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Calculate 25 * 17",
            tool="calculator",
            arguments={
                "expression": "25 * 17"
            },
        )
    ]

    history = executor.execute_plan(
        plan
    )

    assert len(history) == 1

    assert history[0].step == 1

    assert (
        history[0].result
        == "425"
    )

    assert history[0].success is True


def test_executor_runs_multiple_steps_in_order() -> None:
    """
    Executor should execute multiple steps
    in the correct order.
    """

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Calculate 25 * 17",
            tool="calculator",
            arguments={
                "expression": "25 * 17"
            },
        ),
        PlanStep(
            description="Add 100 to the first result",
            tool="calculator",
            arguments={
                "expression": (
                    "{{step_1.result}} + 100"
                )
            },
        ),
    ]

    history = executor.execute_plan(
        plan
    )

    assert len(history) == 2

    assert history[0].result == "425"

    assert history[1].result == "525"

    assert history[0].step == 1

    assert history[1].step == 2


def test_executor_passes_previous_result_to_next_step() -> None:
    """
    A later step should receive the result
    produced by an earlier step.
    """

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Calculate 10 + 5",
            tool="calculator",
            arguments={
                "expression": "10 + 5"
            },
        ),
        PlanStep(
            description="Multiply the result by 2",
            tool="calculator",
            arguments={
                "expression": (
                    "{{step_1.result}} * 2"
                )
            },
        ),
    ]

    history = executor.execute_plan(
        plan
    )

    assert history[0].result == "15"

    assert (
        history[1].arguments["expression"]
        == "15 * 2"
    )

    assert history[1].result == "30"


def test_executor_stops_after_failed_step() -> None:
    """
    Executor should stop instead of continuing
    after a failed step.
    """

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Divide by zero",
            tool="calculator",
            arguments={
                "expression": "10 / 0"
            },
        ),
        PlanStep(
            description="This must not execute",
            tool="calculator",
            arguments={
                "expression": "5 + 5"
            },
        ),
    ]

    history = executor.execute_plan(
        plan
    )

    assert len(history) == 1

    assert history[0].success is False

    assert (
        history[0].result
        == "Error: division by zero."
    )


def test_executor_rejects_unknown_tool() -> None:
    """Executor should safely reject unknown tools."""

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Use unavailable tool",
            tool="web_search",
            arguments={},
        )
    ]

    history = executor.execute_plan(
        plan
    )

    assert len(history) == 1

    assert history[0].success is False

    assert (
        "unknown tool"
        in history[0].result
    )


def test_executor_rejects_unavailable_step_reference() -> None:
    """
    Executor should not use a result that doesn't exist.
    """

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Use missing result",
            tool="calculator",
            arguments={
                "expression": (
                    "{{step_5.result}} + 100"
                )
            },
        )
    ]

    history = executor.execute_plan(
        plan
    )

    assert len(history) == 1

    assert history[0].success is False

    assert (
        "step_5"
        in history[0].result
    )


def test_executor_handles_no_tool_step() -> None:
    """Executor should handle a step requiring no tool."""

    executor = Executor(
        create_registry()
    )

    plan = [
        PlanStep(
            description="Answer directly",
            tool=None,
            arguments={},
        )
    ]

    history = executor.execute_plan(
        plan
    )

    assert len(history) == 1

    assert history[0].success is True

    assert (
        history[0].tool
        is None
  )
