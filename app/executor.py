"""
executor.py

Execution engine for NEXUS.

The Executor is responsible for:
- executing planned steps
- passing observations between steps
- tracking execution history
- stopping safely when a step fails

The Executor does NOT create plans.
The Planner decides WHAT should happen.
The Executor decides HOW the plan is carried out.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from app.planner import PlanStep
from app.tools import ToolRegistry


@dataclass
class ExecutionRecord:
    """Record of one executed plan step."""

    step: int
    description: str
    tool: str | None
    arguments: dict[str, Any] = field(
        default_factory=dict
    )
    result: Any = None
    success: bool = False


class Executor:
    """
    Executes NEXUS plans sequentially.

    The Executor does not decide what to do.
    It only executes the plan it receives.
    """

    _REFERENCE_PATTERN = re.compile(
        r"\{\{step_(\d+)\.result\}\}"
    )

    def __init__(
        self,
        tool_registry: ToolRegistry,
    ) -> None:
        self.tools = tool_registry

    def execute_plan(
        self,
        plan: list[PlanStep],
    ) -> list[ExecutionRecord]:
        """
        Execute every step in a plan.

        Execution stops when a step fails.

        Returns:
            A list containing the execution history.
        """

        history: list[ExecutionRecord] = []

        for index, step in enumerate(
            plan,
            start=1,
        ):
            record = self._execute_step(
                index,
                step,
                history,
            )

            history.append(record)

            if not record.success:
                break

        return history

    def _execute_step(
        self,
        step_number: int,
        step: PlanStep,
        history: list[ExecutionRecord],
    ) -> ExecutionRecord:
        """
        Execute one plan step.
        """

        resolved_arguments = self._resolve_arguments(
            step.arguments,
            history,
        )

        if isinstance(
            resolved_arguments,
            str,
        ):
            return ExecutionRecord(
                step=step_number,
                description=step.description,
                tool=step.tool,
                arguments=step.arguments,
                result=resolved_arguments,
                success=False,
            )

        # A step without a tool is considered
        # successfully completed by the executor.
        if step.tool is None:
            return ExecutionRecord(
                step=step_number,
                description=step.description,
                tool=None,
                arguments=resolved_arguments,
                result=(
                    "Step completed without "
                    "tool execution."
                ),
                success=True,
            )

        tool = self.tools.get(
            step.tool
        )

        if tool is None:
            return ExecutionRecord(
                step=step_number,
                description=step.description,
                tool=step.tool,
                arguments=resolved_arguments,
                result=(
                    f"Error: unknown tool "
                    f"'{step.tool}'."
                ),
                success=False,
            )

        try:
            result = tool.execute(
                **resolved_arguments
            )

        except TypeError as error:
            return ExecutionRecord(
                step=step_number,
                description=step.description,
                tool=step.tool,
                arguments=resolved_arguments,
                result=(
                    "Error: invalid tool arguments: "
                    f"{error}"
                ),
                success=False,
            )

        except Exception as error:
            return ExecutionRecord(
                step=step_number,
                description=step.description,
                tool=step.tool,
                arguments=resolved_arguments,
                result=(
                    "Error: tool execution failed: "
                    f"{error}"
                ),
                success=False,
            )

        success = not self._is_error_result(
            result
        )

        return ExecutionRecord(
            step=step_number,
            description=step.description,
            tool=step.tool,
            arguments=resolved_arguments,
            result=result,
            success=success,
        )

    def _resolve_arguments(
        self,
        arguments: dict[str, Any],
        history: list[ExecutionRecord],
    ) -> dict[str, Any] | str:
        """
        Resolve references to previous step results.

        Example:

            "{{step_1.result}} + 100"

        becomes:

            "425 + 100"
        """

        resolved: dict[str, Any] = {}

        for key, value in arguments.items():
            resolved_value = self._resolve_value(
                value,
                history,
            )

            if isinstance(
                resolved_value,
                _ResolutionError,
            ):
                return resolved_value.message

            resolved[key] = resolved_value

        return resolved

    def _resolve_value(
        self,
        value: Any,
        history: list[ExecutionRecord],
    ) -> Any:
        """
        Resolve references inside any supported value.
        """

        if isinstance(
            value,
            str,
        ):
            matches = list(
                self._REFERENCE_PATTERN.finditer(
                    value
                )
            )

            if not matches:
                return value

            resolved = value

            for match in matches:
                step_number = int(
                    match.group(1)
                )

                if step_number < 1:
                    return _ResolutionError(
                        "Error: invalid step reference."
                    )

                if step_number > len(history):
                    return _ResolutionError(
                        f"Error: step_{step_number} "
                        "result is not available."
                    )

                record = history[
                    step_number - 1
                ]

                if not record.success:
                    return _ResolutionError(
                        f"Error: step_{step_number} "
                        "failed, so its result cannot "
                        "be used."
                    )

                resolved = resolved.replace(
                    match.group(0),
                    str(record.result),
                )

            return resolved

        if isinstance(
            value,
            dict,
        ):
            nested: dict[str, Any] = {}

            for key, nested_value in value.items():
                resolved_value = self._resolve_value(
                    nested_value,
                    history,
                )

                if isinstance(
                    resolved_value,
                    _ResolutionError,
                ):
                    return resolved_value

                nested[key] = resolved_value

            return nested

        if isinstance(
            value,
            list,
        ):
            nested_list: list[Any] = []

            for item in value:
                resolved_value = self._resolve_value(
                    item,
                    history,
                )

                if isinstance(
                    resolved_value,
                    _ResolutionError,
                ):
                    return resolved_value

                nested_list.append(
                    resolved_value
                )

            return nested_list

        return value

    @staticmethod
    def _is_error_result(
        result: Any,
    ) -> bool:
        """
        Determine whether a tool result represents failure.
        """

        if not isinstance(
            result,
            str,
        ):
            return False

        return result.startswith(
            "Error:"
        )


@dataclass
class _ResolutionError:
    """Internal representation of argument resolution failure."""

    message: str
