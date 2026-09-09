"""
Tool system for NEXUS.

V3 provides a safe and extensible tool architecture.
"""

from typing import Any


class Tool:
    """Base class for every NEXUS tool."""

    name: str = ""
    description: str = ""

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool."""
        raise NotImplementedError(
            "Tool must implement execute()."
        )


class CalculatorTool(Tool):
    """Basic calculator tool."""

    name = "calculator"

    description = (
        "Performs basic mathematical calculations "
        "using addition, subtraction, multiplication, "
        "and division."
    )

    def execute(self, expression: str = "") -> str:
        """Calculate a mathematical expression."""

        if not expression:
            return "Error: expression is required."

        allowed_characters = set(
            "0123456789+-*/(). "
        )

        if not set(expression) <= allowed_characters:
            return (
                "Error: expression contains "
                "unsupported characters."
            )

        try:
            result = eval(
                expression,
                {"__builtins__": {}},
                {},
            )

            return str(result)

        except ZeroDivisionError:
            return "Error: division by zero."

        except Exception:
            return (
                "Error: could not calculate "
                "the expression."
            )


class ToolRegistry:
    """Registry containing NEXUS tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""

        if not tool.name:
            raise ValueError(
                "Tool must have a name."
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""

        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""

        return list(self._tools.values())


def create_default_registry() -> ToolRegistry:
    """Create the default NEXUS tool registry."""

    registry = ToolRegistry()

    registry.register(CalculatorTool())

    return registry
