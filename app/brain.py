"""
brain.py

The core orchestration logic for NEXUS.

V5.2 introduces real Planner integration.

Flow:

UNDERSTAND
    ↓
PLAN
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
from app.planner import Planner
from app.tools import ToolRegistry, create_default_registry


@dataclass
class AgentState:
    """
    Stores the state of one NEXUS agent execution.

    V5.2 keeps the existing fields for compatibility
    and tracks the real execution plan and observations.
    """

    user_message: str

    decision: dict[str, Any] = field(
        default_factory=dict
    )

    tool_name: str | None = None

    arguments: dict[str, Any] = field(
        default_factory=dict
    )

    plan: list[dict[str, Any]] = field(
        default_factory=list
    )

    observation: Any = None

    observations: list[Any] = field(
        default_factory=list
    )

    evaluation: str = ""

    final_response: str = ""


class Brain:
    """
    The thinking and orchestration layer of NEXUS.

    Brain decides HOW to execute the plan.
    Planner decides WHAT needs to happen.
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

        self.planner = Planner(
            llm_client
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

        # -------------------------------------------------
        # PLAN
        # -------------------------------------------------

        plan_steps = self.planner.create_plan(
            user_message,
            self.tools.list_tools(),
        )

        state.plan = [
            {
                "step": index,
                "description": step.description,
                "tool": step.tool,
                "arguments": step.arguments,
            }
            for index, step in enumerate(
                plan_steps,
                start=1,
            )
        ]

        # -------------------------------------------------
        # COMPATIBILITY DECISION
        # -------------------------------------------------

        first_tool_step = next(
            (
                step
                for step in plan_steps
                if step.tool is not None
            ),
            None,
        )

        if first_tool_step is None:
            decision = self._empty_decision()

            state.decision = decision
            state.tool_name = None
            state.arguments = {}

            state.evaluation = (
                "No tool required."
            )

            response = self._generate_response()

        else:
            decision = {
                "tool": first_tool_step.tool,
                "arguments": first_tool_step.arguments,
            }

            state.decision = decision
            state.tool_name = first_tool_step.tool
            state.arguments = (
                first_tool_step.arguments
            )

            # -------------------------------------------------
            # ACT + OBSERVE + EVALUATE
            # -------------------------------------------------

            for step in plan_steps:
                if step.tool is None:
                    continue

                self._execute_tool_only(
                    step.tool,
                    step.arguments,
                )

            # Use the final observation to generate
            # the final natural-language response.

            final_tool = first_tool_step.tool

            final_result = (
                state.observations[-1]
                if state.observations
                else None
            )

            response = self._respond_after_tool(
                final_tool,
                final_result,
            )

        # -------------------------------------------------
        # FINAL RESPONSE
        # -------------------------------------------------

        state.final_response = response

        self.memory.add(
            "assistant",
            response,
        )

        return response

    def _execute_tool_only(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """
        Execute one planned tool step.

        This method performs:

        ACT
        ↓
        OBSERVE
        ↓
        EVALUATE
        """

        tool = self.tools.get(
            tool_name
        )

        if tool is None:
            result = (
                f"NEXUS does not have a tool named "
                f"'{tool_name}'."
            )

            self._record_observation(
                result
            )

            return result

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

        self._record_observation(
            result
        )

        return result

    def _decide(
        self,
        user_message: str,
    ) -> dict[str, Any]:
        """
        Compatibility decision method.

        The real planning path is now handled by Planner.
        This method remains available for existing tests
        and older code.
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
        Compatibility wrapper for the existing API.
        """

        result = self._execute_tool_only(
            tool_name,
            arguments,
        )

        return self._respond_after_tool(
            tool_name,
            result,
        )

    def _record_observation(
        self,
        result: Any,
    ) -> None:
        """
        Store an observation from tool execution.
        """

        if self.last_state is None:
            return

        self.last_state.observation = result

        self.last_state.observations.append(
            result
        )

        self.last_state.evaluation = (
            self._evaluate_result(result)
        )

    def _evaluate_result(
        self,
        result: Any,
    ) -> str:
        """
        Evaluate a tool result locally.
        """

        if result is None:
            return (
                "FAILED: tool returned no result."
            )

        if isinstance(result, str):
            if result.startswith("Error:"):
                return (
                    "FAILED: tool returned an error."
                )

            if result.startswith(
                "The tool received invalid arguments:"
            ):
                return (
                    "FAILED: invalid tool arguments."
                )

            if result.startswith(
                "Tool execution failed:"
            ):
                return (
                    "FAILED: tool execution failed."
                )

        return (
            "SUCCESS: usable tool result."
        )

    def _respond_after_tool(
        self,
        tool_name: str,
        tool_result: Any,
    ) -> str:
        """
        Convert observations into the final response.
        """

        context = self._build_context()

        evaluation = ""

        if self.last_state is not None:
            evaluation = (
                self.last_state.evaluation
            )

        observations = []

        if self.last_state is not None:
            observations = (
                self.last_state.observations
            )

        final_prompt = f"""
You are NEXUS.

Conversation:
{context}

A plan was executed.

Tool:
{tool_name}

Latest tool result:
{tool_result}

All observations:
{observations}

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
