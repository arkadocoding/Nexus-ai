"""
brain.py

The core orchestration logic for NEXUS.

V5.4 introduces the real execution architecture.

Flow:

UNDERSTAND
    ↓
PLAN
    ↓
EXECUTE
    ↓
OBSERVE
    ↓
EVALUATE
    ↓
RESPOND

Brain coordinates the system.

Planner decides WHAT should happen.

Executor carries out the plan.

Tools perform the actual actions.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from app.executor import ExecutionRecord, Executor
from app.memory import Memory
from app.planner import Planner
from app.tools import (
    ToolRegistry,
    create_default_registry,
)


@dataclass
class AgentState:
    """
    Stores the state of one NEXUS execution.

    V5.4 adds execution_history so the Brain can
    expose what the Executor actually did.
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

    execution_history: list[
        dict[str, Any]
    ] = field(
        default_factory=list
    )

    final_response: str = ""


class Brain:
    """
    The orchestration layer of NEXUS.

    Brain:
        Coordinates the complete agent cycle.

    Planner:
        Decides what should happen.

    Executor:
        Executes the plan.

    Tools:
        Perform individual capabilities.
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

        self.executor = Executor(
            self.tools
        )

        self.last_state: AgentState | None = None

    def handle_message(
        self,
        user_message: str,
    ) -> str:
        """
        Run one complete NEXUS agent cycle.

        V5.4:

        User
        ↓
        Planner
        ↓
        Executor
        ↓
        Observations
        ↓
        Response
        """

        state = AgentState(
            user_message=user_message
        )

        self.last_state = state

        self.memory.add(
            "user",
            user_message,
        )

        # =================================================
        # PLAN
        # =================================================

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

        # =================================================
        # DETERMINE COMPATIBILITY DECISION
        # =================================================

        first_tool_step = next(
            (
                step
                for step in plan_steps
                if step.tool is not None
            ),
            None,
        )

        if first_tool_step is None:
            # ---------------------------------------------
            # NO TOOL REQUIRED
            # ---------------------------------------------

            state.decision = (
                self._empty_decision()
            )

            state.tool_name = None
            state.arguments = {}

            state.evaluation = (
                "No tool required."
            )

            response = self._generate_response()

        else:
            # ---------------------------------------------
            # TOOL PLAN
            # ---------------------------------------------

            state.decision = {
                "tool": first_tool_step.tool,
                "arguments": (
                    first_tool_step.arguments
                ),
            }

            state.tool_name = (
                first_tool_step.tool
            )

            state.arguments = (
                first_tool_step.arguments
            )

            # =================================================
            # EXECUTOR
            # =================================================
            #
            # IMPORTANT:
            #
            # Brain no longer executes tools itself.
            #
            # Executor owns execution.
            #

            history = (
                self.executor.execute_plan(
                    plan_steps
                )
            )

            self._record_execution_history(
                history
            )

            response = (
                self._respond_after_execution(
                    history
                )
            )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        state.final_response = response

        self.memory.add(
            "assistant",
            response,
        )

        return response

    # =====================================================
    # EXECUTION STATE
    # =====================================================

    def _record_execution_history(
        self,
        history: list[ExecutionRecord],
    ) -> None:
        """
        Copy Executor results into AgentState.

        This gives Brain visibility into everything
        the Executor actually did.
        """

        if self.last_state is None:
            return

        state = self.last_state

        state.execution_history = [
            {
                "step": record.step,
                "description": record.description,
                "tool": record.tool,
                "arguments": record.arguments,
                "result": record.result,
                "success": record.success,
            }
            for record in history
        ]

        state.observations = [
            record.result
            for record in history
        ]

        if history:
            latest = history[-1]

            state.observation = (
                latest.result
            )

            state.evaluation = (
                self._evaluate_execution(
                    history
                )
            )

    def _evaluate_execution(
        self,
        history: list[ExecutionRecord],
    ) -> str:
        """
        Evaluate the overall execution.

        Execution is successful only when every
        executed step succeeds.
        """

        if not history:
            return (
                "FAILED: no execution occurred."
            )

        failed_step = next(
            (
                record
                for record in history
                if not record.success
            ),
            None,
        )

        if failed_step is not None:
            return (
                f"FAILED: step "
                f"{failed_step.step} failed."
            )

        return (
            "SUCCESS: all executed steps completed."
        )

    # =====================================================
    # RESPONSE
    # =====================================================

    def _respond_after_execution(
        self,
        history: list[ExecutionRecord],
    ) -> str:
        """
        Convert the complete execution history
        into a natural final response.
        """

        if not history:
            return (
                "NEXUS could not execute the plan."
            )

        latest = history[-1]

        context = self._build_context()

        execution_history = [
            {
                "step": record.step,
                "tool": record.tool,
                "arguments": record.arguments,
                "result": record.result,
                "success": record.success,
            }
            for record in history
        ]

        evaluation = ""

        if self.last_state is not None:
            evaluation = (
                self.last_state.evaluation
            )

        final_prompt = f"""
You are NEXUS.

Conversation:
{context}

A plan was executed.

Execution history:
{execution_history}

Latest result:
{latest.result}

Overall evaluation:
{evaluation}

Answer the user's original request naturally.

Use the execution results to produce the answer.

If execution failed, clearly explain what
could not be completed.

Do not mention internal implementation details
unless the user asks.
"""

        try:
            return self.llm_client.generate(
                final_prompt
            )

        except Exception as error:
            return (
                "NEXUS completed the execution, "
                "but could not generate the final response: "
                f"{error}"
            )

    # =====================================================
    # LEGACY DECISION COMPATIBILITY
    # =====================================================

    def _decide(
        self,
        user_message: str,
    ) -> dict[str, Any]:
        """
        Compatibility decision method.

        V5.4 runtime no longer depends on this method.

        Planner is now the real planning layer.
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
            raw_decision = (
                self.llm_client.generate(
                    prompt
                )
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

    # =====================================================
    # LEGACY EXECUTION COMPATIBILITY
    # =====================================================

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Compatibility method.

        New runtime execution belongs to Executor.

        This method is retained for older code/tests.
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

        return self._respond_after_tool(
            tool_name,
            result,
        )

    def _record_observation(
        self,
        result: Any,
    ) -> None:
        """
        Compatibility observation method.
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
        Evaluate one individual tool result.
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
        Compatibility response method.
        """

        context = self._build_context()

        evaluation = ""

        if self.last_state is not None:
            evaluation = (
                self.last_state.evaluation
            )

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

    # =====================================================
    # NORMAL RESPONSE
    # =====================================================

    def _generate_response(
        self,
    ) -> str:
        """
        Generate a response when no tool is needed.
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

    # =====================================================
    # MEMORY CONTEXT
    # =====================================================

    def _build_context(
        self,
    ) -> str:
        """
        Convert conversation history into text.
        """

        messages = (
            self.memory.get_messages()
        )

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
