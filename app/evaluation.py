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
