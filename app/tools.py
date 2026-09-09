"""
Tool system for NEXUS.

V3 introduces a simple, extensible tool architecture.

Every tool has:
- a name
- a description
- an execute() method

The ToolRegistry manages all available tools.
"""


from typing import Any


class Tool:
    """
    Base class for every NEXUS tool.

    Future tools should inherit from this class.
    """

    name: str = ""
    description: str = ""

    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool.

        Individual tools must implement this method.
        """
        raise NotImplementedError


class CalculatorTool(Tool):
    """
    Basic calculator tool for NEXUS.
    """

    name = "calculator"

    description = (
        "Performs basic mathematical calculations using "
        "addition, subtraction, multiplication, and division."
    )

    def execute(self, expression: str) -> str:
        """
        Calculate a mathematical expression.
        """

        allowed_characters = set(
            "0123456789+-*/(). "
        )

        if not set(expression) <= allowed_characters:
            return "Error: expression contains unsupported characters."

        try:
            result = eval(
                expression,
                {"__builtins__": {}},
                {},
            )

            return str(result)

        except Exception:
            return "Error: could not calculate the expression."


class ToolRegistry:
    """
    Registry containing all tools available to NEXUS.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.
        """

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.
        """

        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """
        Return all registered tools.
        """

        return list(self._tools.values())


def create_default_registry() -> ToolRegistry:
    """
    Create the default NEXUS tool registry.
    """

    registry = ToolRegistry()

    registry.register(CalculatorTool())

    return registry
