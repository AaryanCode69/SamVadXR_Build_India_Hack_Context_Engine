# Samvad XR — Orchestration Backend

Agentic VR language immersion platform backend. This service is the **Brain & Nervous System** — it orchestrates the full interaction pipeline between the Unity VR client and all AI services.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your real API keys

# 4. Run the server
uvicorn app.main:app --reload --port 8000
```

## Verify

```bash
curl http://localhost:8000/health
# → {"status": "ok", "version": "0.1.0"}
```

## Project Structure

```
app/                    ← Developer A's domain (orchestration + AI brain)
  main.py               API endpoint
  config.py             Pydantic Settings (env vars)
  logging_config.py     Structured JSON logging
  models/               Pydantic request/response models
  services/             AI brain, state engine, mocks
  prompts/              System prompt templates
  dependencies.py       FastAPI dependency injection

services/               ← Developer B's domain (STT, TTS, RAG, memory)
  voice_ops.py          Sarvam STT & TTS
  rag_ops.py            ChromaDB retrieval
  context_memory.py     Conversation memory
  middleware.py         Base64 encoding utilities
  exceptions.py         Custom exception classes

tests/                  ← Test suite
```

## Key Docs

- `ExecutionPlan.md` — Phase-by-phase build plan
- `INTEGRATION_GUIDE.md` — Contract with Developer B
- `rules.md` — Strict project rules (read before coding)
