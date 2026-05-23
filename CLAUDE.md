# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv sync --group dev
cp .env.example .env   # then fill in SILICONFLOW_API_KEY

# Run
uv run python -m gesture_agent.cli                    # interactive mode
uv run python -m gesture_agent.cli "question text"    # single question
uv run python -m gesture_agent.cli --dry-run          # local parsing only, no API call
uv run python -m gesture_agent.cli --stream           # streaming output

# Inspect knowledge/config
uv run python -m gesture_agent.cli --show-term-inventory
uv run python -m gesture_agent.cli --export-structured-knowledge data/structured_knowledge.json

# Tests
pytest
pytest tests/test_knowledge_base.py   # single file
```

## Architecture

This is a structured learning agent for a gesture interaction design dictionary. It parses user questions into typed intents, retrieves relevant chunks from local Markdown knowledge files, and calls a SiliconFlow (OpenAI-compatible) API to generate structured answers.

### Core pipeline

```
User Input
  → CLI (gesture_agent/cli.py)                    # arg parsing, config loading, loop
  → QuestionParser (learning/question_parser.py)  # regex intent + term/layer extraction
  → ConversationSession (learning/session.py)     # multi-turn clarification, memory (6 turns)
  → KnowledgeBase.search (knowledge/base.py)      # BM25 + term-boost + structured reranking
  → PromptBuilder (learning/prompt_builder.py)    # system prompt + question structure + chunks
  → SiliconFlowClient (providers/siliconflow.py)  # OpenAI SDK wrapper, streaming/vision
  → Output
```

### Intent system

`QuestionParser` classifies each question into one of 11 intents via regex, and extracts `layer`, `terms`, and `focus`. These drive retrieval and prompt construction — every API call includes a `QuestionStructure` JSON so the model answers within the gesture dictionary's framework rather than generically.

Supported intents: `basic_interaction_mechanism`, `advanced_interaction_mechanism`, `control_form`, `basic_property`, `multimodal_interaction`, `voice_interaction`, `podcast_content`, `interaction_compare`, `background_knowledge`, `case_analysis`, `design_evaluation`.

When regex confidence is low, `ClarificationIntentResolver` (learning/llm_intent.py) can delegate intent detection to the LLM.

### Knowledge base

Markdown files under `data/` (1.md, 2.md, 3.md, dic.md, etc.) are chunked by heading and indexed. Retrieval uses BM25 + term-title boosting. `data/structured_knowledge.json` (auto-generated or expert-annotated) is used to expand query terms to related concepts for reranking.

### Configuration layers

| File | Controls |
|------|----------|
| `.env` | `SILICONFLOW_API_KEY`, model names, base URL, timeouts |
| `agent_config.json` | data dir, retrieval top-k, streaming, custom system/response prompts |
| `data/term_inventory.json` | terminology aliases; `merge` mode prepends, `replace` overrides built-in terms |
| `data/output_frames.json` | intent-specific response structure templates shown to the model |

`agent_config.json` supports `//` and `/* */` comments (stripped before JSON parsing in `settings/app_config.py`).

### Key data models (`core/models.py`)

- `QuestionStructure` — parsed question with intent, layers, terms, focus, output frame
- `SourceChunk` — a retrieved Markdown section with score, source, line range, layer
- `TermInventory` — canonical terms + aliases
- `IntentResolution` — intent + confidence from parser or LLM

### Tests

Tests use actual `data/` files (no mocking). Fixtures for KB loading are in `conftest.py`.
