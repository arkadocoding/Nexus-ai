"""
This is the entry point for NEXUS. It:
1. Creates an LLM client that talks to OpenAI.
2. Passes that client into the Brain.
3. Runs a simple command-line chat loop so you can talk to NEXUS.

No API key is written in this file. It is read from an environment
variable at runtime, so the real key never ends up in the codebase.
"""

import os

from openai import OpenAI

from app.brain import Brain


class OpenAIClient:
    """
    A small wrapper around the OpenAI SDK.

    Brain doesn't know or care which LLM provider we use - it just
    calls generate(message) and expects a string back.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """
        Args:
            api_key: The OpenAI API key.
            model: The OpenAI model to use.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, message: str) -> str:
        """
        Sends a message to the OpenAI API and returns the response.
        """
        response = self.client.responses.create(
            model=self.model,
            input=message,
        )

        return response.output_text


def main() -> None:
    """
    Sets up NEXUS and runs a simple command-line chat loop.
    """

    # Read the API key from an environment variable.
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print(
            "Error: OPENAI_API_KEY environment variable is not set.\n"
            "Set it before running NEXUS."
        )
        return

    # Read the model from an environment variable.
    model = os.environ.get("NEXUS_MODEL", "gpt-4o-mini")

    # Create the LLM client and give it to the Brain.
    llm_client = OpenAIClient(
        api_key=api_key,
        model=model,
    )

    brain = Brain(llm_client=llm_client)

    print("NEXUS is ready. Type 'exit' to quit.\n")

    # Keep chatting until the user types "exit".
    while True:
        user_message = input("You: ")

        if user_message.strip().lower() == "exit":
            print("Goodbye!")
            break

        response = brain.handle_message(user_message)

        print(f"NEXUS: {response}\n")


if __name__ == "__main__":
    main()
