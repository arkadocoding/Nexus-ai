"""
planner.py

Planning layer for NEXUS.

The Planner decides WHAT needs to happen.
The Brain decides HOW to execute those steps.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """One step in a NEXUS execution plan."""

    description: str
    tool: str | None = None
    arguments: dict[str, Any] = field(
        default_factory=dict
    )


class Planner:
    """
    Creates structured execution plans.

    The planner does not execute tools.
    It only decides the sequence of actions.
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def create_plan(
        self,
        user_message: str,
        available_tools: list[Any],
    ) -> list[PlanStep]:
        """Create a plan for the user's request."""

        tool_descriptions = []

        for tool in available_tools:
            tool_descriptions.append(
                f"- {tool.name}: {tool.description}"
            )

        prompt = f"""
You are the planning engine for NEXUS.

Your job is to create an execution plan.

Available tools:
{chr(10).join(tool_descriptions)}

User request:
{user_message}

Create the smallest useful sequence of steps
needed to solve the user's request.

Each step must contain:

- description
- tool
- arguments

Use "tool": null when a step does not need
a tool.

Return ONLY valid JSON in this format:

{{
    "steps": [
        {{
            "description": "Calculate 25 * 17",
            "tool": "calculator",
            "arguments": {{
                "expression": "25 * 17"
            }}
        }}
    ]
}}

Rules:

1. Do not invent tools.
2. Use available tools when appropriate.
3. Keep the plan as short as possible.
4. A later step may depend on an earlier step's result.
5. Do not execute anything yourself.
"""

        try:
            raw_plan = self.llm_client.generate(
                prompt
            )

            data = json.loads(raw_plan)

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return self._fallback_plan(
                user_message
            )

        return self._parse_plan(
            data,
            available_tools,
            user_message,
        )

    def _parse_plan(
        self,
        data: Any,
        available_tools: list[Any],
        user_message: str,
    ) -> list[PlanStep]:
        """Validate and convert raw planner output."""

        if not isinstance(data, dict):
            return self._fallback_plan(
                user_message
            )

        raw_steps = data.get("steps")

        if not isinstance(raw_steps, list):
            return self._fallback_plan(
                user_message
            )

        available_names = {
            tool.name
            for tool in available_tools
        }

        steps: list[PlanStep] = []

        for raw_step in raw_steps:
            if not isinstance(
                raw_step,
                dict,
            ):
                continue

            description = raw_step.get(
                "description",
                "",
            )

            tool = raw_step.get("tool")

            arguments = raw_step.get(
                "arguments",
                {},
            )

            if not isinstance(
                description,
                str,
            ):
                continue

            if tool is not None:
                if not isinstance(tool, str):
                    continue

                if tool not in available_names:
                    continue

            if not isinstance(
                arguments,
                dict,
            ):
                continue

            steps.append(
                PlanStep(
                    description=description,
                    tool=tool,
                    arguments=arguments,
                )
            )

        if not steps:
            return self._fallback_plan(
                user_message
            )

        return steps

    def _fallback_plan(
        self,
        user_message: str,
    ) -> list[PlanStep]:
        """Create a safe no-tool fallback plan."""

        return [
            PlanStep(
                description=(
                    "Answer the user's request directly."
                ),
                tool=None,
                arguments={},
            )
        ]
