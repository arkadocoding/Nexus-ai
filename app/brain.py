"""
brain.py

The core orchestration logic for NEXUS.

V3 adds the Tool Registry:
- NEXUS keeps conversation memory.
- NEXUS has access to registered tools.
- Tool selection/execution will be added next.
"""

from typing import Any

from app.memory import Memory
from app.tools import ToolRegistry, create_default_registry


class Brain:
    """
    The thinking/orchestration layer of NEXUS.

    Brain manages:
    - conversation memory
    - LLM communication
    - available tools
    """

    def __init__(
        self,
        llm_client: Any,
        memory: Memory | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """
        Args:
            llm_client: Object with a generate(message) method.
            memory: Optional Memory instance.
            tool_registry: Optional ToolRegistry instance.

        If memory or a tool registry is not provided,
        Brain creates them automatically.
        """

        self.llm_client = llm_client

        self.memory = (
            memory
            if memory is not None
            else Memory()
        )

        self.tools = (
            tool_registry
            if tool_registry is not None
            else create_default_registry()
        )

    def handle_message(self, user_message: str) -> str:
        """
        Handle one user message.

        Current flow:

        1. Store the user's message.
        2. Build conversation context.
        3. Send context to the LLM.
        4. Store NEXUS's response.
        5. Return the response.

        Tool selection and execution will be
        added in the next step.
        """

        self.memory.add(
            "user",
            user_message,
        )

        context = self._build_context()

        try:
            response = self.llm_client.generate(
                context
            )
        except Exception as error:
            return (
                "Something went wrong while "
                f"talking to the LLM: {error}"
            )

        self.memory.add(
            "assistant",
            response,
        )

        return response

    def _build_context(self) -> str:
        """
        Convert conversation history into text
        that can be sent to the LLM.
        """

        messages = self.memory.get_messages()

        if not messages:
            return ""

        lines: list[str] = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                lines.append(
                    f"User: {content}"
                )

            elif role == "assistant":
                lines.append(
                    f"NEXUS: {content}"
                )

        lines.append("NEXUS:")

        return "\n".join(lines)
