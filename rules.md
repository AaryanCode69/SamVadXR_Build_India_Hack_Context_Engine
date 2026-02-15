# Samvad XR Orchestration — Project Rules

> **This file is law.** Every contributor reads this before writing a single line of code.
> Violations get flagged in code review. No exceptions.

---

## 1. Architecture Rules

### 1.1 Separation of Concerns

- **App layer** (`app/`) is Dev A's domain — AI brain, state engine, Neo4j persistence, and the `generate_vendor_response()` function.
- **Services layer** (`services/`) is Dev B's domain — STT, TTS, RAG, memory, middleware. Dev A does NOT modify these files.
- **Models** (`app/models/`) contain ONLY Pydantic classes. No business logic. No imports from services.
- **Prompts** (`app/prompts/`) contain ONLY string templates and prompt-building functions. No API calls.
- **State persistence** (`app/services/session_store.py`) is the ONLY module that talks to Neo4j. No other module imports the Neo4j driver directly.
- **Primary interface** (`app/generate.py`) exposes `generate_vendor_response()` — the one function Dev B calls.

### 1.2 The Primary Function is Sacred

> **Architecture v3.0:** Dev B owns the API endpoint. Dev A provides `generate_vendor_response()`.

- `app/generate.py` contains ONE primary function: `generate_vendor_response()`
- This function handles **only Steps 7 + 7½** of the pipeline:
  ```
  Step  Owner   Action
  ────  ──────  ─────────────────────────────────────────────────
   7    Dev A   Parse scene_context → Load Neo4j state → Compose prompt
                → LLM call → Parse JSON → Validate AI decision
   7½   Dev A   Clamp mood ±15, verify stage transition, persist to Neo4j
  ```
- The full pipeline (all 11 steps) is orchestrated by Dev B's endpoint:
  ```
  Step  Owner   Action
  ────  ──────  ─────────────────────────────────────────────────
   1    Dev B   Receive POST /api/interact, parse request
   2    Dev B   base64_to_bytes(request.audio_base64)
   3    Dev B   await transcribe_with_sarvam(bytes, lang)
   4    Dev B   memory.add_turn("user", text, metadata)
   5    Dev B   context_block = memory.get_context_block()
   6    Dev B   rag_ctx = await retrieve_context(text, 3)
   7    Dev A   generate_vendor_response(text, context_block, rag_ctx, scene, session)
   7½   Dev A   (Internal) Validate via Neo4j: clamp mood ±15, verify stage
   8    Dev B   memory.add_turn("vendor", reply, metadata)
   9    Dev B   audio = await speak_with_sarvam(reply, lang)
  10    Dev B   b64 = bytes_to_base64(audio)
  11    Dev B   Return InteractResponse to Unity
  ```
- Every new step inside `generate_vendor_response()` MUST be discussed before insertion.

### 1.3 Server is the Authority

- Unity sends `vendor_happiness`, `vendor_patience`, and `negotiation_stage` via `scene_context` with each request.
- The server's returned values (`new_mood`, `new_stage`, `vendor_happiness`, `vendor_patience`) are **authoritative**.
- Unity MUST overwrite its local state with the server's response values.
- If Unity and server disagree, server wins. Always.
- Dev B forwards Unity's `scene_context` to Dev A's function unchanged. Dev A uses Neo4j as the authoritative source.

### 1.4 No Global Mutable State

- No module-level mutable variables (except the session store, which is explicitly managed).
- Session memory instances are keyed by `session_id` and isolated.
- No function may modify state outside its documented scope.

### 1.5 State Persistence (Neo4j)

- **Neo4j is the authoritative store** for game state: mood, stage, turn count, price history per session.
- Developer B's `context_memory.py` stores **conversation dialogue history** (what was said).
- Neo4j stores **game logic state** (mood numbers, stage transitions, win/loss). These are separate concerns.
- All Neo4j access goes through `app/services/session_store.py`. No Cypher queries scattered in other modules.
- Neo4j connection is initialized at app startup and closed at shutdown (lifespan hook).
- If Neo4j is unreachable at startup, the app MUST fail fast with a clear error.
- If Neo4j is unreachable during a request, return 503 — game state cannot be trusted without it.

---

## 2. Code Style & Standards

### 2.1 Language & Formatting

- Python 3.11+ required.
- All code formatted with **Black** (line length 88).
- Imports sorted with **isort** (profile: black).
- Type hints on ALL function signatures. No `Any` without a comment explaining why.
- Docstrings on all public functions (Google style).

### 2.2 Async Rules

- All functions that perform I/O (HTTP calls, file reads, DB queries) MUST be `async def`.
- NEVER use `requests` library. Use `httpx.AsyncClient` for HTTP calls.
- If a third-party function is synchronous and blocking, wrap it in `asyncio.to_thread()`.
- NEVER call `asyncio.run()` inside the application. FastAPI's event loop handles this.

### 2.3 Naming Conventions

| Thing | Convention | Example |
|-------|-----------|---------|
| Files | `snake_case.py` | `voice_ops.py` |
| Classes | `PascalCase` | `ConversationMemory` |
| Functions | `snake_case` | `transcribe_with_sarvam` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_MOOD_DELTA` |
| Enums | `PascalCase` class, `UPPER_SNAKE_CASE` values | `NegotiationStage.HAGGLING` |
| Env vars | `UPPER_SNAKE_CASE` | `OPENAI_API_KEY` |
| Pydantic models | `PascalCase`, suffix with purpose | `SceneContext`, `VendorResponse`, `AIDecision` |

### 2.4 Import Order

```python
# 1. Standard library
import asyncio
from typing import Optional

# 2. Third-party
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import httpx

# 3. Local — app layer
from app.models.request import InteractionRequest
from app.services.ai_brain import decide

# 4. Local — services layer
from services.voice_ops import transcribe_with_sarvam
```

---

## 3. Data Contract Rules

### 3.1 Pydantic is the Single Source of Truth

- ALL data crossing a boundary (API input, API output, AI response parsing) MUST pass through a Pydantic model.
- No raw `dict` objects passed between functions. Deserialize into a model first.
- No `dict` responses from the API. Always return a Pydantic model instance.

### 3.2 Model Changes Require Notification

- Any change to the `generate_vendor_response()` function signature or return schema is a **breaking change**.
- Breaking changes require:
  1. A message to Developer B.
  2. Updated function contract shared.
- Add new optional return fields freely. Removing or renaming fields requires coordination.
- Changes to `SceneContext` fields must be coordinated with Unity developer (via Dev B).

### 3.3 Enums are Closed

- `NegotiationStage` (`GREETING`, `BROWSING`, `HAGGLING`, `DEAL`, `WALKAWAY`, `CLOSURE`) and `VendorMood` (`enthusiastic`, `neutral`, `annoyed`, `angry`) enums are finite. Adding a new value requires team discussion.
- The AI Brain must NEVER produce a stage or mood value that doesn't exist in the enum.
- Pydantic validation will reject unknown enum values.

### 3.4 Strict Numeric Bounds

- `mood` is always `int`, range `[0, 100]`. Enforced by Pydantic validator AND state engine.
- `vendor_happiness` is always `int`, range `[0, 100]`. Enforced by state engine.
- `vendor_patience` is always `int`, range `[0, 100]`. Enforced by state engine.
- `price_offered` is always `int`, non-negative.
- Mood delta per turn is capped at `±15`. The state engine enforces this even if the AI returns a larger delta.
- `vendor_happiness` and `vendor_patience` deltas also capped at `±15` per turn.

---

## 4. AI Brain Rules

### 4.1 JSON Mode is Non-Negotiable

- Every call to OpenAI MUST use `response_format={"type": "json_object"}`.
- The response MUST be parsed into the `AIDecision` Pydantic model before any further processing.
- If JSON parsing fails: retry once → fallback to safe in-character response. NEVER surface raw AI output.

### 4.2 System Prompt Discipline

- The system prompt (God Prompt) lives in `app/prompts/vendor_system.py`.
- The prompt MUST include the full JSON schema of `AIDecision` so GPT-4o knows the exact output structure.
- The prompt MUST NOT be modified without testing at least 10 sample interactions.
- Every deployed prompt version is tagged in git. Include `prompt_version` in debug logs.

### 4.3 Temperature & Token Limits

- `temperature`: `0.7` (balanced personality variation and coherence).
- `max_tokens`: `200` (vendor replies are short — 2-4 sentences).
- `model`: Read from `OPENAI_MODEL` env var. Default `gpt-4o`. Never hardcode.
- Total latency budget for the AI call: **2000ms**. If consistently exceeding this, reduce prompt size.

### 4.4 The AI Proposes, the Engine Disposes

- The AI's output is a **proposal**. It does NOT directly become the response.
- The state engine validates every proposal:
  - Illegal stage transitions are **rejected** (current stage is kept).
  - Mood values outside `[0, 100]` are **clamped**.
  - Mood deltas exceeding `±15` are **clamped**.
- Every override is logged at `WARN` level.

### 4.5 No Prompt Injection from User Input

- User transcribed text and RAG context are injected into the prompt as clearly delimited data sections.
- Use explicit delimiters (e.g., `--- USER MESSAGE ---`, `--- END ---`) around user-supplied content.
- The system prompt must instruct the AI to treat delimited content as data, not as instructions.
- NEVER concatenate user text directly into the system prompt without delimiters.

---

## 5. State Machine Rules

### 5.1 Legal State Transitions

```
GREETING   → BROWSING
BROWSING   → HAGGLING
BROWSING   → WALKAWAY
HAGGLING   → DEAL
HAGGLING   → WALKAWAY
HAGGLING   → CLOSURE
WALKAWAY   → HAGGLING      (only if vendor_happiness > 40)
WALKAWAY   → CLOSURE
```

- **Updated stages (v3.0):** `GREETING`, `BROWSING`, `HAGGLING`, `DEAL`, `WALKAWAY`, `CLOSURE`
- **Removed:** `WALK_AWAY` (now `WALKAWAY`), `NO_DEAL` (now `CLOSURE`)
- **Everything else is illegal.** No shortcuts. No backward moves (except WALKAWAY → HAGGLING).
- The transition graph is defined in ONE place: `app/services/state_engine.py`. No duplicate definitions.
- State transitions are validated against the graph AND persisted to Neo4j in a single atomic operation.

### 5.2 Terminal States

- `DEAL` and `CLOSURE` are terminal. Once reached, no further interactions on that session.
- Hitting a terminal state MUST trigger a structured session summary log.
- `generate_vendor_response()` should return a flag or distinct stage value so Dev B and Unity know the scenario is over.

### 5.3 Turn Limits

- Maximum turns per session: **30**.
- After turn 25, the AI prompt includes a wrap-up instruction ("start closing the negotiation").
- After turn 30, force the state to `CLOSURE` regardless of AI output.

---

## 6. Error Handling Rules

### 6.1 No Raw Exceptions to Dev B or Unity

- `generate_vendor_response()` either returns a valid `VendorResponse` dict or raises a documented exception.
- Valid exceptions: `BrainServiceError` (LLM failed), `StateStoreError` (Neo4j down).
- Dev B catches these and returns appropriate HTTP errors to Unity.
- Unhandled exceptions MUST NOT leak from `generate_vendor_response()`.

### 6.2 Graceful Degradation Hierarchy

> **Note:** Dev B handles the HTTP response codes. Dev A's function either succeeds or raises exceptions.

| Failed Component | Dev A's Behavior | Dev B Maps To |
|-----------------|------------------|---------------|
| STT (Sarvam) | N/A (Dev B's responsibility) | 503 |
| RAG (ChromaDB) | `rag_context=""` is fine — function works without it | 200 |
| AI Brain (OpenAI) | Retry once → raise `BrainServiceError` | 500 |
| TTS (Sarvam) | N/A (Dev B's responsibility) | 200 (text-only) |
| Neo4j | Raise `StateStoreError` | 503 |

- RAG being empty is a **soft failure** — `generate_vendor_response()` works fine without it.
- Neo4j failure is a **hard failure** — cannot proceed without authoritative state.
- AI failure triggers retry + fallback; only raises exception after retries exhausted.

### 6.3 Timeouts Are Hard Limits

| Component | Timeout | Owner |
|-----------|---------|-------|
| STT | 5 seconds | Dev B |
| RAG | 3 seconds | Dev B |
| AI Brain (OpenAI) | 10 seconds | Dev A |
| TTS | 5 seconds | Dev B |
| Neo4j (per query) | 2 seconds | Dev A |
| `generate_vendor_response()` total | ~3 seconds | Dev A |
| Full pipeline (Dev B's endpoint) | 20 seconds | Dev B |

- Dev A enforces timeouts on OpenAI and Neo4j within `generate_vendor_response()`.
- Dev B enforces timeouts on STT, TTS, RAG, and the overall request budget.
- Timeouts are configured via env vars, not hardcoded.

### 6.4 Retry Policy

- OpenAI API: Max **2 retries** with exponential backoff (1s, 2s).
- Sarvam API (STT/TTS): Max **1 retry** after 500ms.
- RAG: **No retries**. It's either fast or skipped.
- Never retry on 4xx errors (bad request, auth failure). Only on 5xx and timeouts.

---

## 7. Logging Rules

### 7.1 Structured JSON Logs

- All logs MUST be JSON-formatted with these fields:
  ```json
  {
    "timestamp": "2026-02-12T14:30:00Z",
    "level": "INFO",
    "request_id": "uuid",
    "session_id": "session-uuid",
    "step": "stt_complete",
    "duration_ms": 823,
    "message": "STT transcription completed"
  }
  ```
- No `print()` statements. Use Python's `logging` module configured with a JSON formatter.

### 7.2 What to Log at Each Level

| Level | What |
|-------|------|
| `DEBUG` | Full prompts sent to GPT-4o, full AI responses, full RAG results |
| `INFO` | Pipeline step start/complete with timing, session start/end, state transitions |
| `WARN` | AI state overrides (illegal transition blocked, mood clamped), degraded responses (RAG skipped, TTS fallback) |
| `ERROR` | Service failures (STT down, OpenAI timeout), JSON parse failures, unhandled exceptions |

### 7.3 Sensitive Data

- NEVER log the full `audio_base64` (it's huge).
- Log `audio_base64` length in bytes instead.
- NEVER log API keys. Use `***` masking if keys appear in error messages.
- User transcribed text CAN be logged at DEBUG level (needed for prompt debugging).

### 7.4 Request Tracing

- Every incoming request gets a `request_id` (UUID v4) generated at ingestion.
- This `request_id` MUST be attached to every log line for that request's lifecycle.
- The `request_id` is included in the response headers (`X-Request-ID`) so Unity can correlate.

---

## 8. Configuration Rules

### 8.1 Environment Variables Are the Only Config Source

- No hardcoded URLs, keys, model names, or magic numbers in source code.
- All configuration reads from environment variables via a Pydantic `Settings` class.
- The `Settings` class fails fast on startup if required vars are missing.

### 8.2 Required Environment Variables

> **Dev A's variables only.** Dev B has their own set (`SARVAM_API_KEY`, etc.)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | Model to use for AI Brain |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `NEO4J_TIMEOUT_MS` | No | `2000` | Per-query Neo4j timeout |
| `USE_MOCKS` | No | `false` | Toggle mock OpenAI + Neo4j (for Dev A's isolated testing) |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `AI_TIMEOUT_MS` | No | `10000` | AI Brain timeout |
| `MAX_TURNS` | No | `30` | Max turns per session |
| `MAX_MOOD_DELTA` | No | `15` | Max mood change per turn |

### 8.3 Feature Flags

- `USE_MOCKS=true` mocks Dev A's own dependencies (OpenAI, Neo4j) for isolated testing.
- **Removed from Dev A's scope:** `USE_MOCK_STT`, `USE_MOCK_TTS`, `USE_MOCK_RAG`, `SARVAM_API_KEY`, `STT_TIMEOUT_MS`, `TTS_TIMEOUT_MS`, `RAG_TIMEOUT_MS` — all belong to Dev B.
- In production, `USE_MOCKS` MUST be `false`. The startup check should warn if mocks are enabled.

---

## 9. Testing Rules

### 9.1 Test Coverage Requirements

- All Pydantic models: **100% coverage** (valid data, invalid data, edge cases).
- State engine transitions: **100% of legal AND illegal pairs** tested.
- AI Brain: At least **10 diverse scenario tests** (different moods, stages, items, languages).
- API endpoint: Happy path + all error/degradation scenarios.

### 9.2 Test Structure

```
tests/
├── test_models.py            # Pydantic validation tests
├── test_state_engine.py      # State transition tests
├── test_ai_brain.py          # GPT-4o integration tests (can mock OpenAI)
├── test_api.py               # FastAPI endpoint tests (uses mocks)
├── test_mocks.py             # Verify mocks conform to interface
└── test_integration.py       # Full pipeline with real services (run manually)
```

### 9.3 Mock Tests Are First-Class

- Every mock MUST implement the same interface (Protocol/ABC) as the real service.
- Write a test that verifies mock and real implementations both satisfy the interface.
- If a mock diverges from the real service's behavior, the integration will break silently. Prevent this.

### 9.4 No Tests That Call Real External APIs in CI

- Tests that hit OpenAI, Sarvam, or ChromaDB are tagged `@pytest.mark.integration`.
- CI runs only unit tests and mock-based tests.
- Integration tests are run manually before deployment.

---

## 10. Git & Collaboration Rules

### 10.1 Branch Strategy

- `main` — stable, deployable at all times.
- `dev` — integration branch. PRs merge here first.
- Feature branches: `feature/<short-description>` (e.g., `feature/state-engine`).
- Hotfixes: `fix/<short-description>`.

### 10.2 Commit Messages

Format: `<type>(<scope>): <description>`

| Type | When |
|------|------|
| `feat` | New feature or endpoint |
| `fix` | Bug fix |
| `refactor` | Code restructuring, no behavior change |
| `prompt` | System prompt changes (treated as features) |
| `test` | Adding or updating tests |
| `config` | Configuration, env vars, dependencies |
| `docs` | Documentation updates |

Examples:
```
feat(api): add POST /api/interact endpoint with mock pipeline
prompt(vendor): v2 — add mood-based behavioral rules
fix(state): block illegal GREETING→DEAL transition
test(brain): add 10 Hindi haggling scenario tests
```

### 10.3 Code Review Checklist

Before approving any PR, verify:

- [ ] No hardcoded API keys, URLs, or secrets.
- [ ] All new functions have type hints and docstrings.
- [ ] Pydantic models validate edge cases (null, out-of-range, empty string).
- [ ] Error cases return `ErrorResponse`, not raw exceptions.
- [ ] New logs follow JSON format with `request_id`.
- [ ] Prompt changes include before/after test results for at least 5 scenarios.
- [ ] No `print()` statements.
- [ ] No `import requests` (use `httpx`).
- [ ] No `Any` type without justification comment.

---

## 11. Security Rules

### 11.1 Secrets Management

- API keys are NEVER committed to git. Not in code, not in comments, not in `.env` files.
- `.env` is in `.gitignore`. Only `.env.example` (with placeholder values) is committed.
- In production, secrets are injected via environment variables from the hosting platform.

### 11.2 Input Sanitization

- `audio_base64` is validated as legitimate base64 before decoding.
- `session_id` is validated as a UUID format. Arbitrary strings are rejected.
- `held_item` and `looked_at_item` are validated against a known item registry. Unknown items are logged as warnings and treated as `null`.
- String fields have maximum length constraints in Pydantic models (prevent memory abuse).

### 11.3 Rate Limiting

- Implement basic per-session rate limiting: max **2 requests per second** per `session_id`.
- This prevents accidental rapid-fire requests from Unity (e.g., button mashing) from burning OpenAI quota.

### 11.4 CORS

- CORS is configured to allow only known origins (Unity WebXR builds, development localhost).
- Wide-open `allow_origins=["*"]` is permitted ONLY during local development.

---

## 12. Performance Budgets

### 12.1 Target Latencies

> **Dev A's component only.** Dev B's full pipeline has its own budget (~3.5s total).

| Component | Target | Hard Limit | Owner |
|-----------|--------|------------|-------|
| Scene context parsing + validation | < 5ms | 50ms | Dev A |
| Neo4j state load | ~20ms | 2000ms | Dev A |
| AI Brain (GPT-4o) | ~2000ms | 10000ms | Dev A |
| State validation + clamp | < 5ms | 50ms | Dev A |
| Neo4j state persist | ~20ms | 2000ms | Dev A |
| Response assembly | < 5ms | 50ms | Dev A |
| **`generate_vendor_response()` total** | **~2.1s** | **~12s** | **Dev A** |

### 12.2 Optimization Priorities

If `generate_vendor_response()` latency exceeds 3 seconds consistently:
1. Profile each internal step. Identify which component is the bottleneck.
2. Reduce GPT-4o `max_tokens` or prompt length.
3. Consider caching Neo4j session state in memory between calls.
4. Do NOT sacrifice correctness for speed. A correct 3s response beats a wrong 1s response.

---

## 13. Definition of Done

A task is complete when:

- [ ] Code is written and follows all rules in this document.
- [ ] Unit tests pass.
- [ ] No lint errors (Black + isort).
- [ ] Type checking passes (`mypy` or Pylance with no errors).
- [ ] Edge cases are handled (null inputs, out-of-range values, service failures).
- [ ] Logs are structured and include `request_id`.
- [ ] Documentation is updated if the change affects API contracts or architecture.
- [ ] PR is reviewed and approved.

---

*This document is enforced, not aspirational. If a rule doesn't make sense, change it through a PR — don't ignore it.*

*Architecture v3.0 — Feb 13, 2026: Updated for new architecture where Dev B owns the endpoint and Dev A provides `generate_vendor_response()`.*
