"""Configuration loading for NEXUS."""

import os

from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("NEXUS_MODEL", "gpt-5.6-luna")
