"""
Memory module for NEXUS.

This is the foundation for NEXUS's memory system.
The first version is intentionally simple.

We will later expand this to support:
- conversation history
- long-term memory
- semantic search
- persistent storage
"""


class Memory:
    """
    Stores conversation context for NEXUS.

    V1 keeps memory in Python's memory only.
    It will be replaced with a more powerful system later.
    """

    def __init__(self) -> None:
        """Create an empty memory store."""
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        """
        Add a message to memory.

        Args:
            role: Who sent the message, such as "user" or "assistant".
            content: The actual message text.
        """
        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    def get_messages(self) -> list[dict[str, str]]:
        """
        Return all stored messages.
        """
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all stored memories."""
        self.messages.clear()
