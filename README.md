# AI Engineer Portfolio

Portfolio projects focused on applied GenAI / LLM engineering, themed around food delivery and e-commerce.

**Each subfolder is an independent project**, with its own `README.md`, dependencies, environment and instructions for running and testing. There is no shared code between projects; open the subfolder of the project you care about and follow its README.

## Projects

| Folder | Project | Status |
|--------|---------|--------|
| [`review-triage-llm/`](review-triage-llm/) | Fine-tuning a small LLM to triage PT-BR customer reviews into structured JSON (category, sentiment, urgency), compared against a classic baseline and a large zero-shot LLM | Planning |

## Conventions

- Public or synthetic data only. No third-party brands or logos.
- Every project README covers: problem, architecture, how to run, metrics and a demo.
- Secrets live in local `.env` files, which are never committed.
