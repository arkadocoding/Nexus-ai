"""
Tests for the NEXUS Brain and Memory system.
"""

from app.brain import Brain
from app.memory import Memory


class FakeLLMClient:
    """
    Fake LLM client used for testing.

    It records the message sent to it so we can verify
    that Brain actually provides conversation context.
    """

    def __init__(self) -> None:
        self.last_message = ""

    def generate(self, message: str) -> str:
        self.last_message = message
        return f"Fake response to: {message}"


def test_brain_returns_llm_response() -> None:
    """Brain should return the LLM's response."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message("Hello NEXUS")

    assert response.startswith("Fake response to:")


def test_brain_stores_conversation() -> None:
    """Brain should store both user and assistant messages."""

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
    """Brain should include previous messages in the LLM context."""

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    brain.handle_message("My favorite language is Python.")
    brain.handle_message("What is my favorite language?")

    assert "My favorite language is Python." in fake_client.last_message
    assert "What is my favorite language?" in fake_client.last_message
