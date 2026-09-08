# NEXUS 🤖

NEXUS is a Python AI agent project built iteratively as an experiment in modular AI architecture.

## Current version

**V1 — Brain + LLM connection**

NEXUS currently:
- accepts a message from the command line
- routes it through a modular `Brain`
- sends it to an OpenAI model
- returns the model's response
- keeps the LLM provider behind a small adapter

The next iterations will add memory, tools, planning, and verification.

## Project structure

```text
nexus-ai/
├── app/
│   ├── brain.py       # Core orchestration layer
│   ├── config.py      # Environment configuration
│   ├── main.py        # CLI + OpenAI adapter
│   ├── memory.py      # Future memory system
│   └── tools.py       # Future tool system
├── tests/
│   └── test_brain.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add your own `OPENAI_API_KEY`.

3. Start NEXUS:

```bash
python -m app.main
```

Type `exit` to quit.

## Roadmap

- [x] Modular Brain
- [x] OpenAI adapter
- [x] CLI chat
- [x] Basic test
- [ ] Conversation memory
- [ ] Calculator tool
- [ ] Tool-selection loop
- [ ] Verification / self-checking
- [ ] More tools
- [ ] Benchmarks and evaluation
