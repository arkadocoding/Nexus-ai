"""
planner.py

Planning system for NEXUS.

The planner converts a user request into a
structured sequence of steps before execution.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class PlanStep:
    """Represents one step in a NEXUS plan."""

    description: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None


class Planner:
    """
    Creates execution plans for NEXUS.

    V1 of the planner is intentionally simple.

    The LLM decides the high-level plan while
    the Brain remains responsible for execution.
    """

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client

    def create_plan(
        self,
        user_message: str,
    ) -> list[PlanStep]:
        """
        Ask the LLM to create a structured plan.
        """

        prompt = f"""
You are the planning system for NEXUS.

Your job is to break the user's request into
clear, actionable steps.

User request:
{user_message}

Create a short plan.

Rules:
- Use only the number of steps actually needed.
- Keep steps specific and actionable.
- Do not perform the task.
- Do not invent tools.
- If the request is simple, use one step.
- If no tool is required, tool should be null.

Return ONLY valid JSON in this format:

{{
    "steps": [
        {{
            "description": "step description",
            "tool": null,
            "arguments": {{}}
        }}
    ]
}}

Example:

User request:
What is 25 * 17?

Return:

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
"""

        try:
            raw_plan = self.llm_client.generate(
                prompt
            )

            return self._parse_plan(raw_plan)

        except Exception:
            return [
                PlanStep(
                    description=user_message
                )
            ]

    def _parse_plan(
        self,
        raw_plan: str,
    ) -> list[PlanStep]:
        """
        Parse the LLM-generated plan.

        Parsing is isolated from execution so the
        Brain never executes raw LLM output directly.
        """

        import json

        data = json.loads(raw_plan)

        if not isinstance(data, dict):
            raise ValueError(
                "Plan must be a JSON object."
            )

        steps = data.get("steps")

        if not isinstance(steps, list):
            raise ValueError(
                "Plan steps must be a list."
            )

        parsed_steps: list[PlanStep] = []

        for step in steps:
            if not isinstance(step, dict):
                continue

            description = step.get(
                "description",
                "",
            )

            tool = step.get(
                "tool"
            )

            arguments = step.get(
                "arguments",
                {},
            )

            if not isinstance(
                description,
                str,
            ):
                continue

            if not isinstance(
                arguments,
                dict,
            ):
                arguments = {}

            if tool is not None and not isinstance(
                tool,
                str,
            ):
                tool = None

            parsed_steps.append(
                PlanStep(
                    description=description,
                    tool=tool,
                    arguments=arguments,
                )
            )

        if not parsed_steps:
            raise ValueError(
                "Plan contains no valid steps."
            )

        return parsed_steps
