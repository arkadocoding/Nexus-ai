"""
brain.py

The core orchestration logic for NEXUS.

V4.4 introduces evaluation.

Agent flow:

UNDERSTAND
    ↓
DECIDE
    ↓
ACT
    ↓
OBSERVE
    ↓
EVALUATE
    ↓
RESPOND
"""

import json
from dataclasses import dataclass, field
from typing import Any

from app.memory import Memory
from app.tools import ToolRegistry, create_default_registry


@dataclass
class AgentState:
    """
    Stores the state of one NEXUS agent execution.
    """

    user_message: str

    decision: dict[str, Any] = field(
        default_factory=dict
    )

    tool_name: str | None = None

    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    observation: Any = None

    evaluation: str = ""

    final_response: str = ""


class Brain:
    """
    The thinking and orchestration layer of NEXUS.
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

        self.last_state: AgentState | None = None

    def handle_message(
        self,
        user_message: str,
    ) -> str:
        """
        Run one NEXUS agent cycle.
        """

        state = AgentState(
            user_message=user_message
        )

        self.last_state = state

        self.memory.add(
            "user",
            user_message,
        )

        decision = self._decide(
            user_message
        )

        state.decision = decision

        state.tool_name = decision["tool"]

        state.arguments = decision[
            "arguments"
        ]

        if decision["tool"] is not None:
            response = self._execute_tool(
                decision["tool"],
                decision["arguments"],
            )
        else:
            state.evaluation = "No tool required."

            response = self._generate_response()

        state.final_response = response

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
        Decide whether NEXUS should use a tool.
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

            decision = json.loads(
                raw_decision
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return self._empty_decision()

        return self._validate_decision(
            decision
        )

    def _validate_decision(
        self,
        decision: Any,
    ) -> dict[str, Any]:
        """
        Validate an LLM-generated tool decision.
        """

        if not isinstance(
            decision,
            dict,
        ):
            return self._empty_decision()

        tool_name = decision.get(
            "tool"
        )

        arguments = decision.get(
            "arguments",
            {},
        )

        if tool_name is None:
            return self._empty_decision()

        if not isinstance(
            tool_name,
            str,
        ):
            return self._empty_decision()

        if not isinstance(
            arguments,
            dict,
        ):
            return self._empty_decision()

        tool = self.tools.get(
            tool_name
        )

        if tool is None:
            return self._empty_decision()

        return {
            "tool": tool.name,
            "arguments": arguments,
        }

    def _empty_decision(
        self,
    ) -> dict[str, Any]:
        """
        Return a safe no-tool decision.
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
        ACT:

        Execute the selected tool.

        OBSERVE:

        Store the tool result.

        EVALUATE:

        Check whether the tool execution
        produced a usable result.
        """

        tool = self.tools.get(
            tool_name
        )

        if tool is None:
            return (
                f"NEXUS does not have a tool named "
                f"'{tool_name}'."
            )

        try:
            result = tool.execute(
                **arguments
            )

        except TypeError as error:
            result = (
                "The tool received invalid arguments: "
                f"{error}"
            )

        except Exception as error:
            result = (
                f"Tool execution failed: {error}"
            )

        if self.last_state is not None:
            self.last_state.observation = result

            self.last_state.evaluation = (
                self._evaluate_result(result)
            )

        return self._respond_after_tool(
            tool_name,
            result,
        )

    def _evaluate_result(
        self,
        result: Any,
    ) -> str:
        """
        Evaluate a tool result locally.

        V4.4 intentionally uses simple deterministic
        evaluation. We will later replace this with
        an LLM-based evaluator.
        """

        if result is None:
            return "FAILED: tool returned no result."

        if isinstance(result, str):
            if result.startswith("Error:"):
                return "FAILED: tool returned an error."

            if result.startswith(
                "The tool received invalid arguments:"
            ):
                return "FAILED: invalid tool arguments."

            if result.startswith(
                "Tool execution failed:"
            ):
                return "FAILED: tool execution failed."

        return "SUCCESS: usable tool result."

    def _respond_after_tool(
        self,
        tool_name: str,
        tool_result: Any,
    ) -> str:
        """
        Convert the observed and evaluated tool
        result into a natural final response.
        """

        context = self._build_context()

        evaluation = ""

        if self.last_state is not None:
            evaluation = self.last_state.evaluation

        final_prompt = f"""
You are NEXUS.

Conversation:
{context}

A tool was used.

Tool:
{tool_name}

Tool result:
{tool_result}

Evaluation:
{evaluation}

Answer the user's original question naturally.

If the tool failed, clearly explain that the
operation could not be completed.

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

    def _generate_response(
        self,
    ) -> str:
        """
        Generate a normal response without tools.
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

    def _build_context(
        self,
    ) -> str:
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
