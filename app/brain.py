"""
brain.py

The core orchestration logic for NEXUS.

V2 adds conversation memory:
- User messages are stored.
- NEXUS responses are stored.
- Previous conversation is included when generating a response.
"""

from typing import Any

from app.memory import Memory


class Brain:
    """
    The thinking/orchestration layer of NEXUS.

    Brain does not directly know how an LLM provider works.
    It receives an LLM client and uses Memory to maintain
    conversation context.
    """

    def __init__(
        self,
        llm_client: Any,
        memory: Memory | None = None,
    ) -> None:
        """
        Args:
            llm_client: Object with a generate(message) method.
            memory: Optional Memory instance.

        If no memory is provided, Brain creates one automatically.
        """
        self.llm_client = llm_client
        self.memory = memory if memory is not None else Memory()

    def handle_message(self, user_message: str) -> str:
        """
        Handle one user message.

        The flow is:

        1. Store the user's message.
        2. Build context from conversation history.
        3. Send that context to the LLM.
        4. Store NEXUS's response.
        5. Return the response.
        """

        # Store the user's message.
        self.memory.add("user", user_message)

        # Build the conversation context.
        context = self._build_context()

        # Ask the LLM for a response.
        try:
            response = self.llm_client.generate(context)
        except Exception as error:
            return f"Something went wrong while talking to the LLM: {error}"

        # Store NEXUS's response.
        self.memory.add("assistant", response)

        return response

    def _build_context(self) -> str:
        """
        Convert stored conversation history into text
        that can be sent to the LLM.
        """

        messages = self.memory.get_messages()

        if not messages:
            return ""

        lines = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"NEXUS: {content}")

        lines.append("NEXUS:")

        return "\n".join(lines)
