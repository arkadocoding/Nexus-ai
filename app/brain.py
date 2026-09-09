"""
brain.py

The core orchestration logic for NEXUS.

V3 includes:
- conversation memory
- tool registry
- structured tool decisions
- decision validation
- safe tool execution
"""

import json
from typing import Any

from app.memory import Memory
from app.tools import ToolRegistry, create_default_registry


class Brain:
    """
    The thinking/orchestration layer of NEXUS.

    Brain manages:
    - conversation memory
    - LLM communication
    - tool selection
    - tool execution
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
        Handle one user message.

        Flow:

        User
          ↓
        Memory
          ↓
        Decide
        ↙     ↘
      normal  tool
        ↓      ↓
       LLM   execute
              ↓
             LLM
              ↓
           response
        """

        self.memory.add(
            "user",
            user_message,
        )

        decision = self._decide(user_message)

        if decision["tool"] is not None:
            response = self._execute_tool(
                decision["tool"],
                decision["arguments"],
            )
        else:
            response = self._generate_response()

        self.memory.add(
            "assistant",
            response,
        )

        return response

    def _decide(
        self,
        user_message: str,
    ) -> dict[str, Any]:
        """
        Ask the LLM whether a tool is required.

        The decision is validated before it can
        reach the tool execution layer.
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

        except (json.JSONDecodeError, TypeError):
            return self._empty_decision()

        return self._validate_decision(decision)

    def _validate_decision(
        self,
        decision: Any,
    ) -> dict[str, Any]:
        """
        Validate an LLM-generated tool decision.

        Invalid decisions safely become a normal
        response instead of reaching tool execution.
        """

        if not isinstance(decision, dict):
            return self._empty_decision()

        tool_name = decision.get("tool")
        arguments = decision.get("arguments", {})

        # No tool requested.
        if tool_name is None:
            return self._empty_decision()

        # Tool name must be a string.
        if not isinstance(tool_name, str):
            return self._empty_decision()

        # Arguments must be a dictionary.
        if not isinstance(arguments, dict):
            return self._empty_decision()

        # Tool must actually exist.
        tool = self.tools.get(tool_name)

        if tool is None:
            return self._empty_decision()

        return {
            "tool": tool.name,
            "arguments": arguments,
        }

    def _empty_decision(self) -> dict[str, Any]:
        """
        Return a safe decision meaning:
        no tool should be used.
        """

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
        Execute a validated tool and send its result
        back to the LLM.
        """

        tool = self.tools.get(tool_name)

        if tool is None:
            return (
                f"NEXUS does not have a tool named "
                f"'{tool_name}'."
            )

        try:
            result = tool.execute(**arguments)

        except TypeError as error:
            return (
                "The tool received invalid arguments: "
                f"{error}"
            )

        except Exception as error:
            return (
                f"Tool execution failed: {error}"
            )

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
                "generate the final response: "
                f"{error}"
            )

    def _generate_response(self) -> str:
        """
        Generate a normal response without a tool.
        """

        context = self._build_context()

        try:
            return self.llm_client.generate(
                context
            )

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
