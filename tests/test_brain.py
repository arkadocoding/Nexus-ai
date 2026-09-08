"""
Tests for the NEXUS Brain.
"""

from app.brain import Brain


class FakeLLMClient:
    """
    A fake LLM client used for testing.

    This lets us test Brain without making a real API call.
    """

    def generate(self, message: str) -> str:
        return f"Fake response to: {message}"


def test_brain_returns_llm_response() -> None:
    """
    Brain should pass the user's message to the LLM
    and return the LLM's response.
    """

    fake_client = FakeLLMClient()
    brain = Brain(llm_client=fake_client)

    response = brain.handle_message("Hello NEXUS")

    assert response == "Fake response to: Hello NEXUS"
