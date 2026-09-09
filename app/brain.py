"""
brain.py

Core orchestration layer for NEXUS.

V5.1 architecture:

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

from dataclasses import dataclass, field
from typing import Any

from app.memory import Memory
from app.planner import PlanStep, Planner
from app.tools import (
    ToolRegistry,
    create_default_registry,
)


@dataclass
class AgentState:
    """State of one NEXUS execution."""

    user_message: str

    plan: list[PlanStep] = field(
        default_factory=list
    )

    observations: list[Any] = field(
        default_factory=list
    )

    evaluation: str = ""

    final_response: str = ""


class Brain:
    """Thinking and orchestration layer of NEXUS."""

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
        """Run one complete NEXUS execution."""

        state = AgentState(
            user_message=user_message
        )

        self.last_state = state

        self.memory.add(
            "user",
            user_message,
        )

        # PLAN
        state.plan = self.planner.create_plan(
            user_message,
            self.tools.list_tools(),
        )

        # ACT + OBSERVE
        for step in state.plan:

            if step.tool is None:
                continue

            result = self._execute_tool(
                step.tool,
                step.arguments,
            )

            state.observations.append(
                result
            )

        # EVALUATE
        state.evaluation = (
            self._evaluate_execution()
        )

        # RESPOND
        response = self._generate_final_response()

        state.final_response = response

        self.memory.add(
            "assistant",
            response,
        )

        return response

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute one planned tool."""

        tool = self.tools.get(
            tool_name
        )

        if tool is None:
            return (
                f"Error: unknown tool '{tool_name}'."
            )

        try:
            return tool.execute(
                **arguments
            )

        except TypeError as error:
            return (
                "Error: invalid tool arguments: "
                f"{error}"
            )

        except Exception as error:
            return (
                "Error: tool execution failed: "
                f"{error}"
            )

    def _evaluate_execution(self) -> str:
        """Evaluate the overall execution."""

        if self.last_state is None:
            return "FAILED: no execution state."

        observations = (
            self.last_state.observations
        )

        if not observations:
            return "SUCCESS: direct response required."

        for result in observations:

            if result is None:
                return (
                    "FAILED: tool returned no result."
                )

            if isinstance(result, str):

                if result.startswith(
                    "Error:"
                ):
                    return (
                        "FAILED: at least one "
                        "tool execution failed."
                    )

        return (
            "SUCCESS: all planned tool "
            "executions completed."
        )

    def _generate_final_response(self) -> str:
        """Generate the final response using plan + results."""

        if self.last_state is None:
            return (
                "NEXUS could not create an execution state."
            )

        state = self.last_state

        plan_text = "\n".join(
            f"{index + 1}. {step.description}"
            for index, step in enumerate(
                state.plan
            )
        )

        observations_text = "\n".join(
            f"{index + 1}. {result}"
            for index, result in enumerate(
                state.observations
            )
        )

        context = self._build_context()

        prompt = f"""
You are NEXUS.

User request:
{state.user_message}

Execution plan:
{plan_text}

Observed results:
{observations_text}

Evaluation:
{state.evaluation}

Conversation:
{context}

Now answer the user's original request.

Use the observed results when available.

Do not expose internal planning details
unless the user specifically asks about them.

Be natural and concise.
"""

        try:
            return self.llm_client.generate(
                prompt
            )

        except Exception as error:
            return (
                "NEXUS completed the operation, "
                "but could not generate the final response: "
                f"{error}"
            )

    def _build_context(self) -> str:
        """Build conversation context."""

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

        return "\n".join(lines)
