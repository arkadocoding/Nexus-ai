"""
Tools module for NEXUS.

Tools are actions that NEXUS can use when answering a task.

For V1, we are starting with one simple and reliable tool:
a calculator.

Later we can add:
- web search
- file reading
- code execution
- APIs
- custom user tools
"""


class Calculator:
    """
    A simple calculator tool for NEXUS.

    NEXUS will eventually decide when this tool is needed.
    """

    name = "calculator"
    description = "Performs basic mathematical calculations."

    def calculate(self, expression: str) -> str:
        """
        Calculate a mathematical expression.

        Args:
            expression: A mathematical expression such as "25 * 48".

        Returns:
            The calculation result as a string.
        """

        try:
            # Allow only characters commonly used in basic arithmetic.
            allowed_characters = "0123456789+-*/().% "

            if not all(
                character in allowed_characters
                for character in expression
            ):
                return "Error: expression contains unsupported characters."

            result = eval(expression, {"__builtins__": {}}, {})

            return str(result)

        except Exception:
            return "Error: could not calculate the expression."
