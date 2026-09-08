"""
brain.py

This module contains the Brain class: the core orchestration logic for NEXUS.

For now, the Brain only does one thing: take a user's message and pass it
to an LLM client to get a response. Later, this file will grow to include
the full UNDERSTAND -> DECIDE -> (tool or response) -> VERIFY loop described
in the NEXUS architecture. The method names below are placeholders that
show where that logic will eventually go.
"""

from typing import Any


class Brain:
    """
    The Brain is the "thinking" part of NEXUS.

    It does not know how to call any specific LLM provider itself.
    Instead, it receives an already-configured LLM client through its
    constructor. This keeps Brain modular: we can swap LLM providers,
    add memory, add tools, etc. later without rewriting this class.
    """

    def __init__(self, llm_client: Any) -> None:
        """
        Args:
            llm_client: An object that knows how to talk to an LLM.
                It must have a method that Brain can call to get a
                response for a given message. We keep this generic
                (typed as Any) for now, since the exact client
                interface isn't finalized yet.

                No API keys are stored here - the client is expected
                to already be configured with whatever credentials it
                needs before being passed in.
        """
        self.llm_client = llm_client

    def handle_message(self, user_message: str) -> str:
        """
        Main entry point for NEXUS. Takes a raw user message and returns
        a final response string.

        Right now this method just sends the message to the LLM and
        returns whatever comes back. Over time, this is where the full
        orchestration loop will live:

            UNDERSTAND -> DECIDE -> (use tool / generate response) -> VERIFY

        The private helper methods below (_understand, _decide) are
        placeholders that mark where that future logic will go. They do
        not do anything meaningful yet - we're not pretending tool use,
        memory, or planning already exist.

        Args:
            user_message: The raw text the user typed.

        Returns:
            The LLM's response as a string, or a readable error message
            if something went wrong.
        """
        # Step 1: Understand the message (placeholder for now).
        understood_message = self._understand(user_message)

        # Step 2: Decide what to do with it (placeholder for now).
        # In the future, this is where NEXUS will decide whether a tool
        # is needed. For now, it always chooses to just ask the LLM.
        self._decide(understood_message)

        # Step 3: Since there is no tool logic yet, we go straight to
        # generating a response from the LLM.
        try:
            response = self.llm_client.generate(understood_message)
        except Exception as error:
            # Catch any error from the LLM call (network issues, bad
            # responses, etc.) so NEXUS doesn't crash. We return a
            # readable message instead.
            return f"Something went wrong while talking to the LLM: {error}"

        return response

    def _understand(self, user_message: str) -> str:
        """
        Placeholder for the future UNDERSTAND step.

        Eventually this might clean up the message, add context from
        memory, or restructure it for the LLM. For now, it just passes
        the message through unchanged.
        """
        return user_message

    def _decide(self, understood_message: str) -> None:
        """
        Placeholder for the future DECIDE step.

        Eventually this will decide whether a tool is needed to answer
        the message, based on planning logic that doesn't exist yet.
        For now, it does nothing - NEXUS always just asks the LLM
        directly.
        """
        pass
