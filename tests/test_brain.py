"""
Tests for the NEXUS Brain, Memory, Tools,
AgentState, and Evaluation system.
"""

from app.brain import AgentState, Brain
from app.memory import Memory
from app.tools import CalculatorTool, ToolRegistry


class FakeLLMClient:
    """Fake LLM client used for testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, message: str) -> str:
        """Return predictable responses for tests."""

        self.calls.append(message)

        if "Return ONLY valid JSON." in message:
            if "User message:\nWhat is 25 * 17?" in message:
                return (
                    '{"tool": "calculator", '
                    '"arguments": {"expression": "25 * 17"}}'
                )

            return (
                '{"tool": null, '
                '"arguments": {}}'
            )

        if "A tool was used." in message:
            return "The answer is 425."

        return "Normal NEXUS response."


def test_brain_returns_normal_response() -> None:
    """Brain should return a normal LLM response."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message("Hello NEXUS")

    assert response == "Normal NEXUS response."


def test_brain_stores_conversation() -> None:
    """Brain should store user and assistant messages."""

    fake_client = FakeLLMClient()
    memory = Memory()

    brain = Brain(
        llm_client=fake_client,
        memory=memory,
    )

    brain.handle_message("Hello NEXUS")

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello NEXUS"
    assert messages[1]["role"] == "assistant"


def test_brain_uses_previous_conversation() -> None:
    """Brain should include previous conversation."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    brain.handle_message(
        "My favorite language is Python."
    )

    brain.handle_message(
        "What is my favorite language?"
    )

    last_call = fake_client.calls[-1]

    assert "My favorite language is Python." in last_call
    assert "What is my favorite language?" in last_call


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


def test_tool_registry() -> None:
    """ToolRegistry should register and retrieve tools."""

    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    assert registry.get("calculator") is calculator
    assert len(registry.list_tools()) == 1


def test_brain_uses_calculator() -> None:
    """Brain should execute the calculator when required."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message(
        "What is 25 * 17?"
    )

    assert response == "The answer is 425."

    assert any(
        "Return ONLY valid JSON." in call
        for call in fake_client.calls
    )

    assert any(
        "425" in call
        for call in fake_client.calls
    )


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
    brain = Brain(llm_client=fake_client)

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
    brain = Brain(llm_client=fake_client)

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
    brain = Brain(llm_client=fake_client)

    decision = brain._validate_decision(
        "calculator"
    )

    assert decision == {
        "tool": None,
        "arguments": {},
    }


def test_calculator_handles_missing_expression() -> None:
    """Calculator should handle missing expressions."""

    calculator = CalculatorTool()

    result = calculator.execute()

    assert result == "Error: expression is required."


def test_calculator_handles_division_by_zero() -> None:
    """Calculator should safely handle division by zero."""

    calculator = CalculatorTool()

    result = calculator.execute(
        expression="10 / 0"
    )

    assert result == "Error: division by zero."


def test_agent_state_defaults() -> None:
    """AgentState should initialize with safe defaults."""

    state = AgentState(
        user_message="Hello NEXUS"
    )

    assert state.user_message == "Hello NEXUS"
    assert state.decision == {}
    assert state.tool_name is None
    assert state.arguments == {}
    assert state.observation is None
    assert state.evaluation == ""
    assert state.final_response == ""


def test_agent_state_tracks_normal_response() -> None:
    """AgentState should track a normal response."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message(
        "Hello NEXUS"
    )

    state = brain.last_state

    assert state is not None
    assert state.user_message == "Hello NEXUS"
    assert state.tool_name is None
    assert state.arguments == {}
    assert state.observation is None
    assert state.evaluation == "No tool required."
    assert state.final_response == response


def test_agent_state_tracks_tool_execution() -> None:
    """AgentState should track tool execution."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message(
        "What is 25 * 17?"
    )

    state = brain.last_state

    assert state is not None
    assert state.user_message == "What is 25 * 17?"
    assert state.tool_name == "calculator"

    assert state.arguments == {
        "expression": "25 * 17"
    }

    assert state.observation == "425"
    assert state.evaluation == (
        "SUCCESS: usable tool result."
    )
    assert state.final_response == response


def test_evaluation_detects_success() -> None:
    """Evaluator should recognize a valid result."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    result = brain._evaluate_result("425")

    assert result == (
        "SUCCESS: usable tool result."
    )


def test_evaluation_detects_empty_result() -> None:
    """Evaluator should detect an empty result."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    result = brain._evaluate_result(None)

    assert result == (
        "FAILED: tool returned no result."
    )


def test_evaluation_detects_tool_error() -> None:
    """Evaluator should detect a tool error."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    result = brain._evaluate_result(
        "Error: division by zero."
    )

    assert result == (
        "FAILED: tool returned an error."
    )


def test_evaluation_detects_invalid_arguments() -> None:
    """Evaluator should detect invalid arguments."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    result = brain._evaluate_result(
        "The tool received invalid arguments: test"
    )

    assert result == (
        "FAILED: invalid tool arguments."
    )


def test_evaluation_detects_execution_failure() -> None:
    """Evaluator should detect execution failure."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    result = brain._evaluate_result(
        "Tool execution failed: test"
    )

    assert result == (
        "FAILED: tool execution failed."
    )
