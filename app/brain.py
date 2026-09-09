"""
brain.py

The core orchestration logic for NEXUS.

V3 adds basic tool decision-making:
- NEXUS remembers conversations.
- NEXUS knows which tools are available.
- NEXUS can decide whether to use a tool.
- Tool results are sent back to the LLM.
"""

import json
from typing import Any

from app.memory import Memory
from app.tools import ToolRegistry, create_default_registry


class Brain:
    """
    The thinking/orchestration layer of NEXUS.
    """

    def __init__(
        self,
        llm_client: Any,
        memory: Memory | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
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
        Handle a user message.

        Flow:

        User message
            ↓
        Memory
            ↓
        Decide
          ↙   ↘
       normal  tool
         ↓      ↓
        LLM   execute
                ↓
               LLM
                ↓
             response
        """

        self.memory.add("user", user_message)

        decision = self._decide(user_message)

        if decision["tool"] is not None:
            response = self._execute_tool(
                decision["tool"],
                decision["arguments"],
            )
        else:
            response = self._generate_response()

        self.memory.add("assistant", response)

        return response

    def _decide(self, user_message: str) -> dict[str, Any]:
        """
        Ask the LLM whether a tool is required.

        Expected format:

        {
            "tool": "calculator",
            "arguments": {
                "expression": "25 * 17"
            }
        }

        Or:

        {
            "tool": null,
            "arguments": {}
        }
        """

        tools = self.tools.list_tools()

        tool_descriptions = []

        for tool in tools:
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}"
            )

        prompt = f"""
You are the decision engine for NEXUS.

Available tools:
{chr(10).join(tool_descriptions)}

User message:
{user_message}

Decide whether NEXUS needs a tool.

Return ONLY valid JSON.

If a tool is needed:
{{
    "tool": "tool_name",
    "arguments": {{
        "argument_name": "value"
    }}
}}

If no tool is needed:
{{
    "tool": null,
    "arguments": {{}}
}}

For mathematical calculations, use the calculator tool.

Example:

User: What is 25 * 17?

Return:
{{
    "tool": "calculator",
    "arguments": {{
        "expression": "25 * 17"
    }}
}}
"""

        try:
            raw_decision = self.llm_client.generate(
                prompt
            )

            decision = json.loads(raw_decision)

            if not isinstance(decision, dict):
                raise ValueError("Invalid decision format.")

            return {
                "tool": decision.get("tool"),
                "arguments": decision.get(
                    "arguments",
                    {},
                ),
            }

        except Exception:
            # If the decision cannot be parsed,
            # safely fall back to a normal response.
            return {
                "tool": None,
                "arguments": {},
            }

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Execute the selected tool and use its result
        to generate the final NEXUS response.
        """

        tool = self.tools.get(tool_name)

        if tool is None:
            return (
                f"NEXUS does not have a tool named "
                f"'{tool_name}'."
            )

        try:
            result = tool.execute(**arguments)
        except Exception as error:
            return f"Tool execution failed: {error}"

        context = self._build_context()

        final_prompt = f"""
You are NEXUS.

Conversation:
{context}

A tool was used.

Tool:
{tool_name}

Tool result:
{result}

Answer the user's original question naturally.
Do not mention internal implementation details
unless the user asks.
"""

        try:
            return self.llm_client.generate(
                final_prompt
            )
        except Exception as error:
            return (
                "The tool worked, but NEXUS could not "
                f"generate the final response: {error}"
            )

    def _generate_response(self) -> str:
        """
        Generate a normal response without using a tool.
        """

        context = self._build_context()

        try:
            return self.llm_client.generate(context)
        except Exception as error:
            return (
                "Something went wrong while talking "
                f"to the LLM: {error}"
            )

    def _build_context(self) -> str:
        """
        Convert conversation history into text.
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
