"""
Tests for the NEXUS Brain, Memory, Tools,
Planner, Executor, AgentState, and Evaluation system.
"""

from app.brain import AgentState, Brain
from app.memory import Memory
from app.tools import CalculatorTool, ToolRegistry


class FakeLLMClient:
    """Fake LLM client used to test NEXUS deterministically."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, message: str) -> str:
        """Return predictable responses for different NEXUS stages."""

        self.calls.append(message)

        # -------------------------------------------------
        # PLANNER
        # -------------------------------------------------
        if "You are the planning engine for NEXUS." in message:
            if "User request:\nWhat is 25 * 17?" in message:
                return (
                    '{"steps": ['
                    '{'
                    '"description": "Calculate 25 * 17", '
                    '"tool": "calculator", '
                    '"arguments": {"expression": "25 * 17"}'
                    '}'
                    ']}'
                )

            return (
                '{"steps": ['
                '{'
                '"description": "Answer the user\'s request directly.", '
                '"tool": null, '
                '"arguments": {}'
                '}'
                ']}'
            )

        # -------------------------------------------------
        # LEGACY DECISION ENGINE
        # -------------------------------------------------
        if "You are the decision engine for NEXUS." in message:
            if "User message:\nWhat is 25 * 17?" in message:
                return (
                    '{"tool": "calculator", '
                    '"arguments": {"expression": "25 * 17"}}'
                )

            return (
                '{"tool": null, '
                '"arguments": {}}'
            )

        # -------------------------------------------------
        # V5.4 EXECUTOR RESPONSE
        # -------------------------------------------------
        if "A plan was executed." in message:
            return "The answer is 425."

        # -------------------------------------------------
        # LEGACY TOOL RESPONSE
        # -------------------------------------------------
        if "A tool was used." in message:
            return "The answer is 425."

        # -------------------------------------------------
        # NORMAL RESPONSE
        # -------------------------------------------------
        return "Normal NEXUS response."


# =========================================================
# BASIC BRAIN TESTS
# =========================================================


def test_brain_returns_normal_response() -> None:
    """Brain should return a normal LLM response."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    response = brain.handle_message(
        "Hello NEXUS"
    )

    assert response == "Normal NEXUS response."


def test_brain_stores_conversation() -> None:
    """Brain should store user and assistant messages."""

    fake_client = FakeLLMClient()
    memory = Memory()

    brain = Brain(
        llm_client=fake_client,
        memory=memory,
    )

    brain.handle_message(
        "Hello NEXUS"
    )

    messages = memory.get_messages()

    assert len(messages) == 2

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello NEXUS"

    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == (
        "Normal NEXUS response."
    )


def test_brain_uses_previous_conversation() -> None:
    """Brain should include previous conversation."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    brain.handle_message(
        "My favorite language is Python."
    )

    brain.handle_message(
        "What is my favorite language?"
    )

    last_call = fake_client.calls[-1]

    assert (
        "My favorite language is Python."
        in last_call
    )

    assert (
        "What is my favorite language?"
        in last_call
    )


# =========================================================
# CALCULATOR TESTS
# =========================================================


def test_calculator_tool() -> None:
    """Calculator should correctly calculate expressions."""

    calculator = CalculatorTool()

    result = calculator.execute(
        expression="25 * 17"
    )

    assert result == "425"


def test_calculator_rejects_unsupported_characters() -> None:
    """Calculator should reject unsupported input."""

    calculator = CalculatorTool()

    result = calculator.execute(
        expression="25 * abc"
    )

    assert (
        result
        == "Error: expression contains "
        "unsupported characters."
    )


def test_calculator_handles_missing_expression() -> None:
    """Calculator should handle missing expressions."""

    calculator = CalculatorTool()

    result = calculator.execute()

    assert result == (
        "Error: expression is required."
    )


def test_calculator_handles_division_by_zero() -> None:
    """Calculator should safely handle division by zero."""

    calculator = CalculatorTool()

    result = calculator.execute(
        expression="10 / 0"
    )

    assert result == (
        "Error: division by zero."
    )


# =========================================================
# TOOL REGISTRY TESTS
# =========================================================


def test_tool_registry() -> None:
    """ToolRegistry should register and retrieve tools."""

    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(
        calculator
    )

    assert (
        registry.get("calculator")
        is calculator
    )

    assert len(
        registry.list_tools()
    ) == 1


# =========================================================
# BRAIN + CALCULATOR
# =========================================================


def test_brain_uses_calculator() -> None:
    """
    Brain should plan and execute the calculator
    when a calculation is required.
    """

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    response = brain.handle_message(
        "What is 25 * 17?"
    )

    assert response == (
        "The answer is 425."
    )

    assert any(
        "You are the planning engine for NEXUS."
        in call
        for call in fake_client.calls
    )

    assert any(
        "425" in call
        for call in fake_client.calls
    )


# =========================================================
# LEGACY DECISION COMPATIBILITY
# =========================================================


def test_unknown_tool_is_handled() -> None:
    """Brain should safely handle an unknown tool."""

    fake_client = FakeLLMClient()

    registry = ToolRegistry()

    brain = Brain(
        llm_client=fake_client,
        tool_registry=registry,
    )

    result = brain._execute_tool(
        "does_not_exist",
        {},
    )

    assert "does_not_exist" in result


def test_invalid_tool_decision_is_rejected() -> None:
    """Invalid tool decisions should become no-tool decisions."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    decision = brain._validate_decision(
        {
            "tool": "does_not_exist",
            "arguments": {},
        }
    )

    assert decision == {
        "tool": None,
        "arguments": {},
    }


def test_invalid_arguments_are_rejected() -> None:
    """Tool arguments must be a dictionary."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    decision = brain._validate_decision(
        {
            "tool": "calculator",
            "arguments": "25 * 17",
        }
    )

    assert decision == {
        "tool": None,
        "arguments": {},
    }


def test_invalid_decision_format_is_rejected() -> None:
    """Non-dictionary decisions should be rejected."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    decision = brain._validate_decision(
        "calculator"
    )

    assert decision == {
        "tool": None,
        "arguments": {},
    }


# =========================================================
# AGENT STATE TESTS
# =========================================================


def test_agent_state_defaults() -> None:
    """AgentState should initialize with safe defaults."""

    state = AgentState(
        user_message="Hello NEXUS"
    )

    assert state.user_message == (
        "Hello NEXUS"
    )

    assert state.decision == {}

    assert state.tool_name is None

    assert state.arguments == {}

    assert state.plan == []

    assert state.observation is None

    assert state.observations == []

    assert state.execution_history == []

    assert state.evaluation == ""

    assert state.final_response == ""


def test_agent_state_tracks_normal_response() -> None:
    """AgentState should track a normal response."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    response = brain.handle_message(
        "Hello NEXUS"
    )

    state = brain.last_state

    assert state is not None

    assert state.user_message == (
        "Hello NEXUS"
    )

    assert state.tool_name is None

    assert state.arguments == {}

    assert state.observation is None

    assert state.observations == []

    assert state.execution_history == []

    assert state.evaluation == (
        "No tool required."
    )

    assert state.final_response == response


# =========================================================
# V5.4 EXECUTOR INTEGRATION
# =========================================================


def test_brain_creates_executor() -> None:
    """Brain should own an Executor instance."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    assert brain.executor is not None


def test_brain_records_executor_history() -> None:
    """
    Brain should expose the execution history
    produced by Executor.
    """

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    response = brain.handle_message(
        "What is 25 * 17?"
    )

    assert response == (
        "The answer is 425."
    )

    state = brain.last_state

    assert state is not None

    assert len(
        state.execution_history
    ) == 1

    execution = (
        state.execution_history[0]
    )

    assert execution["step"] == 1

    assert execution["description"] == (
        "Calculate 25 * 17"
    )

    assert execution["tool"] == (
        "calculator"
    )

    assert execution["arguments"] == {
        "expression": "25 * 17"
    }

    assert execution["result"] == "425"

    assert execution["success"] is True


def test_brain_uses_executor_for_execution() -> None:
    """
    Brain should delegate actual plan execution
    to its Executor.
    """

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    original_executor = brain.executor

    calls = []

    original_execute_plan = (
        original_executor.execute_plan
    )

    def tracked_execute_plan(plan):
        calls.append(plan)

        return original_execute_plan(
            plan
        )

    original_executor.execute_plan = (
        tracked_execute_plan
    )

    response = brain.handle_message(
        "What is 25 * 17?"
    )

    assert response == (
        "The answer is 425."
    )

    assert len(calls) == 1

    assert len(calls[0]) == 1

    assert (
        calls[0][0].tool
        == "calculator"
    )

    assert (
        calls[0][0].arguments
        == {
            "expression": "25 * 17"
        }
    )


def test_brain_tracks_plan() -> None:
    """
    Brain should store the Planner's generated
    plan inside AgentState.
    """

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    brain.handle_message(
        "What is 25 * 17?"
    )

    state = brain.last_state

    assert state is not None

    assert len(state.plan) == 1

    assert state.plan[0]["tool"] == (
        "calculator"
    )

    assert state.plan[0]["arguments"] == {
        "expression": "25 * 17"
    }


def test_brain_tracks_execution_observation() -> None:
    """
    Brain should copy Executor observations
    into AgentState.
    """

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    brain.handle_message(
        "What is 25 * 17?"
    )

    state = brain.last_state

    assert state is not None

    assert state.observation == "425"

    assert state.observations == [
        "425"
    ]

    assert state.evaluation == (
        "SUCCESS: all executed steps completed."
    )


# =========================================================
# EVALUATION TESTS
# =========================================================


def test_evaluation_detects_success() -> None:
    """Evaluator should recognize a valid result."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    result = brain._evaluate_result(
        "425"
    )

    assert result == (
        "SUCCESS: usable tool result."
    )


def test_evaluation_detects_empty_result() -> None:
    """Evaluator should detect an empty result."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    result = brain._evaluate_result(
        None
    )

    assert result == (
        "FAILED: tool returned no result."
    )


def test_evaluation_detects_tool_error() -> None:
    """Evaluator should detect a tool error."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    result = brain._evaluate_result(
        "Error: division by zero."
    )

    assert result == (
        "FAILED: tool returned an error."
    )


def test_evaluation_detects_invalid_arguments() -> None:
    """Evaluator should detect invalid arguments."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    result = brain._evaluate_result(
        "The tool received invalid arguments: test"
    )

    assert result == (
        "FAILED: invalid tool arguments."
    )


def test_evaluation_detects_execution_failure() -> None:
    """Evaluator should detect execution failure."""

    fake_client = FakeLLMClient()

    brain = Brain(
        llm_client=fake_client
    )

    result = brain._evaluate_result(
        "Tool execution failed: test"
    )

    assert result == (
        "FAILED: tool execution failed."
    )
