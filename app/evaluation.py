"""
evaluation.py

Evaluation system for NEXUS.

The Evaluator analyzes execution results and determines:
- whether execution succeeded
- why it succeeded or failed
- whether the failure can be recovered from
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Evaluation:
    """Structured evaluation of an execution."""

    success: bool
    reason: str
    recoverable: bool = False
    step: int | None = None
    result: Any = None


class Evaluator:
    """Evaluates execution history produced by NEXUS Executor."""

    def evaluate(
        self,
        history: list[Any],
    ) -> Evaluation:
        """Evaluate the result of plan execution."""

        if not history:
            return Evaluation(
                success=False,
                reason="No execution occurred.",
                recoverable=False,
            )

        failed_steps = [
            record
            for record in history
            if not record.success
        ]

        if failed_steps:
            failed = failed_steps[-1]

            return Evaluation(
                success=False,
                reason=(
                    f"Step {failed.step} failed."
                ),
                recoverable=True,
                step=failed.step,
                result=failed.result,
            )

        latest = history[-1]

        return Evaluation(
            success=True,
            reason=(
                "All executed steps completed."
            ),
            recoverable=False,
            step=latest.step,
            result=latest.result,
        )
