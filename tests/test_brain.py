"""
Tests for the NEXUS Brain, Memory, and Tool system.
"""

from app.brain import Brain
from app.memory import Memory
from app.tools import CalculatorTool, ToolRegistry


class FakeLLMClient:
    """
    Fake LLM client used for testing.

    It simulates:
    - normal LLM responses
    - tool decisions
    - final responses after tool execution
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, message: str) -> str:
        """Return a predictable response for tests."""

        self.calls.append(message)

        # Tool decision request
        if "Return ONLY valid JSON." in message:

            # Only trigger the calculator when the
            # actual user message is the math question.
            if "User message:\nWhat is 25 * 17?" in message:
                return (
                    '{"tool": "calculator", '
                    '"arguments": {"expression": "25 * 17"}}'
                )

            # No tool required.
            return (
                '{"tool": null, '
                '"arguments": {}}'
            )

        # Final response after tool execution.
        if "A tool was used." in message:
            return "The answer is 425."

        # Normal response.
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
        == "Error: expression contains unsupported characters."
    )


def test_tool_registry() -> None:
    """ToolRegistry should register and retrieve tools."""

    registry = ToolRegistry()
    calculator = CalculatorTool()

    registry.register(calculator)

    assert registry.get("calculator") is calculator
    assert len(registry.list_tools()) == 1


def test_brain_uses_calculator() -> None:
    """
    Brain should:

    1. Ask the LLM for a tool decision.
    2. Execute the calculator.
    3. Send the result back to the LLM.
    4. Return the final response.
    """

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message(
        "What is 25 * 17?"
    )

    assert response == "The answer is 425."

    # Verify that the decision engine was called.
    assert any(
        "Return ONLY valid JSON." in call
        for call in fake_client.calls
    )

    # Verify that the calculator result
    # reached the final LLM call.
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
