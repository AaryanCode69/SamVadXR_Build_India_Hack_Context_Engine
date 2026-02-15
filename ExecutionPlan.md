# Samvad XR — Developer A Execution Plan

## The Brain & Rules Engine of Samvad XR

**Owner:** Developer A (AI Logic & State Engine)
**Stack:** Python · OpenAI GPT-4o · Pydantic · Neo4j · (FastAPI for dev/testing only)
**Last Updated:** 2026-02-13

> **Architecture v3.0:** Developer B owns the FastAPI endpoint (`POST /api/interact`).
> Unity talks directly to Dev B. Dev A provides a single function — `generate_vendor_response()` —
> that Dev B calls mid-pipeline. Dev A is the **brain and rules engine**, not the orchestrator.

---

## Table of Contents

1. [High-Level System Flow](#1-high-level-system-flow)
2. [Phase Breakdown](#2-phase-breakdown)
3. [Phase 0 — Project Scaffolding & Environment](#phase-0--project-scaffolding--environment)
4. [Phase 1 — Data Contracts (Pydantic Models)](#phase-1--data-contracts-pydantic-models)
5. [Phase 2 — Mock Services Layer](#phase-2--mock-services-layer)
6. [Phase 3 — Function Interface & Dev Endpoint](#phase-3--function-interface--dev-endpoint)
7. [Phase 4 — The AI Brain (Game Master)](#phase-4--the-ai-brain-game-master)
8. [Phase 5 — State Machine & Game Logic](#phase-5--state-machine--game-logic)
9. [Phase 6 — Integration with Developer B](#phase-6--integration-with-developer-b)
10. [Phase 7 — End-to-End Testing & Hardening](#phase-7--end-to-end-testing--hardening)
11. [Phase 8 — Deployment & Observability](#phase-8--deployment--observability)
12. [Risk Register](#3-risk-register)
13. [Dependency Map](#4-dependency-map)
14. [Milestones & Deliverables](#5-milestones--deliverables)

---

## 1. High-Level System Flow

> **Architecture v3.0:** Dev B owns the endpoint. Dev A is called as an internal function.

The full request lifecycle from Unity headset to response, showing both Dev A and Dev B responsibilities:

```
UNITY ──► audio_base64 + scene_context JSON
              │
         ┌────▼─────┐
Step 1   │  Dev B    │  Receive POST /api/interact, parse request
         └────┬──────┘
              │
         ┌────▼─────┐
Step 2   │  Dev B    │  base64_to_bytes() → raw WAV bytes
         └────┬──────┘
              │
         ┌────▼─────┐
Step 3   │  Dev B    │  transcribe_with_sarvam() → transcribed text
         └────┬──────┘
              │
         ┌────▼─────┐
Step 4   │  Dev B    │  memory.add_turn("user", text, metadata)
         └────┬──────┘
              │
    ┌─────────┴──────────┐  (parallel — asyncio.gather)
    │                    │
┌───▼────┐          ┌────▼───┐
│ Dev B  │ Step 5   │ Dev B  │ Step 6
│ context│          │ RAG    │
│ _block │          │ context│
└───┬────┘          └────┬───┘
    │                    │
    └─────────┬──────────┘
              │
    ╔═════════▼════════════════════════════════════════╗
    ║  DEV A's DOMAIN (Steps 7 + 7½)                   ║
    ║                                                   ║
    ║  generate_vendor_response(                        ║
    ║      transcribed_text, context_block,             ║
    ║      rag_context, scene_context, session_id       ║
    ║  )                                                ║
    ║                                                   ║
    ║  ┌──────────────────────────────────┐             ║
    ║  │  COGNITION LAYER                 │             ║
    ║  │  • Build GPT-4o prompt           │             ║
    ║  │  • Inject: user_text, rag_ctx,   │             ║
    ║  │    game_state, scene_context      │             ║
    ║  │  • Force JSON-mode response      │             ║
    ║  │  • Parse AI decision             │             ║
    ║  └──────────────┬───────────────────┘             ║
    ║                 │                                  ║
    ║  ┌──────────────▼───────────────────┐             ║
    ║  │  STATE VALIDATION ENGINE         │             ║
    ║  │  • Validate AI's proposed state  │             ║
    ║  │  • Clamp mood ±15 per turn       │             ║
    ║  │  • Enforce legal stage moves     │             ║
    ║  │  • Persist state to Neo4j        │             ║
    ║  └──────────────┬───────────────────┘             ║
    ║                 │                                  ║
    ║  Returns: { reply_text, new_mood, new_stage,      ║
    ║    price_offered, vendor_happiness,                ║
    ║    vendor_patience, vendor_mood }                  ║
    ╚═════════════════╤════════════════════════════════╝
              │
         ┌────▼─────┐
Step 8   │  Dev B    │  memory.add_turn("vendor", reply, metadata)
         └────┬──────┘
              │
         ┌────▼─────┐
Step 9   │  Dev B    │  speak_with_sarvam() → WAV audio bytes
         └────┬──────┘
              │
         ┌────▼─────┐
Step 10  │  Dev B    │  bytes_to_base64() → base64 string
         └────┬──────┘
              │
         ┌────▼─────┐
Step 11  │  Dev B    │  Return InteractResponse JSON to Unity
         └────┬──────┘
              │
              ▼
           UNITY ◄── audio_base64 + vendor reply JSON
```

### Dev A's Scope (This Codebase)

Dev A provides **one function** that encapsulates all AI + state logic:

```python
async def generate_vendor_response(
    transcribed_text: str,    # from Dev B's STT (Step 3)
    context_block: str,       # from Dev B's memory (Step 5)
    rag_context: str,         # from Dev B's RAG (Step 6)
    scene_context: dict,      # from Unity (via Dev B)
    session_id: str           # session identifier
) -> dict:
    """
    Returns:
    {
        "reply_text": str,          # Vendor's spoken response
        "new_mood": int,            # Validated mood (0-100, clamped ±15)
        "new_stage": str,           # GREETING|BROWSING|HAGGLING|DEAL|WALKAWAY|CLOSURE
        "price_offered": int,       # Vendor's current asking price
        "vendor_happiness": int,    # 0-100
        "vendor_patience": int,     # 0-100
        "vendor_mood": str          # enthusiastic|neutral|annoyed|angry
    }
    """
```

### Key Principle: "Never Trust, Always Validate"

Every boundary — incoming arguments, AI output, state transition — gets validated. Dev B relies on our response being clean and validated. The AI hallucinates invalid states. Our job is to be the gatekeeper at every seam.

---

## 2. Phase Breakdown

| Phase | Name | Blocked By | Est. Effort | Priority |
|-------|------|------------|-------------|----------|
| 0 | Project Scaffolding & Environment | Nothing | 0.5 day | P0 |
| 1 | Data Contracts (Pydantic Models) | Nothing | 1 day | P0 |
| 2 | Mock Services Layer | Phase 1 | 0.5 day | P0 |
| 3 | Function Interface & Dev Endpoint | Phase 1, 2 | 0.5 day | P0 |
| 4 | AI Brain (GPT-4o Integration) | Phase 1, 3 | 2 days | P0 |
| 5 | State Machine & Game Logic | Phase 4 | 1.5 days | P0 |
| 6 | Integration with Developer B | Dev B ready | 1-2 days | P1 |
| 7 | End-to-End Testing & Hardening | Phase 5 | 1.5 days | P1 |
| 8 | Deployment & Observability | Phase 7 | 1 day | P2 |

**Critical Path:** 0 → 1 → 2 → 3 → 4 → 5 → 7

> **Note:** Phase 3 is lighter than before — Dev A no longer builds the full API gateway.
> Dev B owns the endpoint. We only expose `generate_vendor_response()` and an optional dev/test endpoint.

---

## Phase 0 — Project Scaffolding & Environment

**Goal:** A clean project structure with proper dependency management. FastAPI is retained for dev/testing (health check, optional test endpoint), but the **primary deliverable** is the `generate_vendor_response()` function that Dev B imports and calls.

> **Architecture v3.0 Note:** Dev B owns the production API endpoint. This codebase is a library/module
> that Dev B imports. FastAPI is kept for isolated development and testing, not as the production entry point.

### Tasks

- [x] **0.1 — Repository & Folder Structure**
  - Define the project layout:
    ```
    SamVadXR-Orchestration/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py              # FastAPI app (dev/testing only) + health check
    │   ├── generate.py          # ★ PRIMARY INTERFACE — generate_vendor_response()
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── enums.py         # NegotiationStage, VendorMood, LanguageCode
    │   │   ├── request.py       # SceneContext model (from Unity via Dev B)
    │   │   └── response.py      # VendorResponse model (returned to Dev B)
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── ai_brain.py      # GPT-4o cognition logic
    │   │   ├── state_engine.py  # State machine / transition logic
    │   │   ├── session_store.py # Neo4j session state read/write
    │   │   └── mocks.py         # Mock OpenAI + Neo4j for testing
    │   ├── prompts/
    │   │   └── vendor_system.py # System prompt templates
    │   ├── config.py            # Settings, env vars, constants
    │   ├── dependencies.py      # FastAPI dependency injection (dev only)
    │   └── logging_config.py    # Structured JSON logging
    ├── services/                # ◄── DEV B's domain (not touched by Dev A)
    │   ├── voice_ops.py         # STT/TTS (Dev B)
    │   ├── rag_ops.py           # RAG (Dev B)
    │   ├── context_memory.py    # Conversation memory (Dev B)
    │   ├── middleware.py        # Base64 encoding (Dev B)
    │   └── exceptions.py       # Dev B's exception classes
    ├── tests/
    │   ├── test_models.py
    │   ├── test_api.py          # Dev test endpoint tests
    │   ├── test_ai_brain.py
    │   └── test_state_engine.py
    ├── .env.example
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md
    ```
  - **Key addition:** `app/generate.py` — This is the single file Dev B imports. It exposes `generate_vendor_response()`.
  - Rationale: Separation of concerns. Models are isolated from logic. Services are independently testable. Prompts live in their own namespace. The `services/` folder is Dev B's domain — we never modify it.

- [x] **0.2 — Dependency Pinning**
  - Lock versions for: `fastapi`, `uvicorn`, `openai`, `pydantic`, `python-dotenv`, `httpx` (for async HTTP), `neo4j` (async driver), `pytest`, `pytest-asyncio`
  - Pin OpenAI SDK to a specific version to avoid breaking changes in structured output APIs
  - Pin `neo4j` driver to a specific version for deterministic builds
  - **Note:** `sarvam` SDK / API client is NOT our dependency — Dev B handles that.

- [x] **0.3 — Environment & Configuration**
  - Define all config via environment variables (12-factor app style)
  - Required env vars: `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o`), `LOG_LEVEL`
  - Neo4j env vars: `NEO4J_URI` (default `bolt://localhost:7687`), `NEO4J_USER` (default `neo4j`), `NEO4J_PASSWORD`, `NEO4J_TIMEOUT_MS` (default `2000`)
  - Game rules: `MAX_TURNS` (default `30`), `MAX_MOOD_DELTA` (default `15`)
  - Create a Pydantic `Settings` class that reads from `.env` and fails fast on missing required vars
  - **Removed from Dev A scope:** `SARVAM_API_KEY`, `USE_MOCK_STT`, `USE_MOCK_TTS`, `USE_MOCK_RAG`, `STT_TIMEOUT_MS`, `TTS_TIMEOUT_MS`, `RAG_TIMEOUT_MS` — all belong to Dev B
  - **Retained:** `USE_MOCKS` flag now controls whether Dev A mocks its own dependencies (OpenAI, Neo4j) for isolated testing

- [x] **0.4 — Health Check Endpoint (Dev/Testing)**
  - `GET /health` — returns `{"status": "ok", "version": "0.1.0"}`
  - Useful for verifying the dev server is running during isolated testing
  - **Not the production health check** — Dev B's server handles that for Unity

- [x] **0.5 — Neo4j Connection Setup**
  - Add Neo4j async driver initialization in the FastAPI lifespan (startup) for dev server
  - Also add a standalone `init_neo4j()` / `close_neo4j()` pair that Dev B can call during their app startup
  - Verify connectivity on startup; log success or hard-fail with a clear error
  - Create `app/services/session_store.py` as the ONLY module that imports the Neo4j driver

### Exit Criteria
- `uvicorn app.main:app --reload` starts without errors (dev mode)
- `/health` returns 200
- All env vars are loaded and validated
- Neo4j driver connects successfully on startup (or logs a clear failure)
- `generate_vendor_response` function is importable: `from app.generate import generate_vendor_response`

---

## Phase 1 — Data Contracts (Pydantic Models)

**Goal:** The single source of truth for what Dev B sends to `generate_vendor_response()` and what we return. Dev B, Unity dev, and our own AI logic all depend on these contracts.

> **Architecture v3.0 Note:** We no longer define Unity's request/response models (that's Dev B's endpoint).
> Our contracts are the **input arguments** and **return value** of `generate_vendor_response()`.

### Tasks

- [x] **1.1 — Define Enums & Constants**
  - `NegotiationStage` enum: `GREETING`, `BROWSING`, `HAGGLING`, `DEAL`, `WALKAWAY`, `CLOSURE`
    - **Changed from v2.0:** `WALK_AWAY` → `WALKAWAY`, `NO_DEAL` removed, `CLOSURE` added
  - `VendorMood` enum: `enthusiastic`, `neutral`, `annoyed`, `angry`
  - `LanguageCode` enum: `hi-IN` (Hindi), `kn-IN` (Kannada), `ta-IN` (Tamil), `en-IN` (English), `hi-EN` (Hinglish)
    - **Changed from v2.0:** Now uses Sarvam-style codes (`hi-IN` not `hi`)
  - `MoodRange`: Constrained integer 0–100
  - `ItemID`: String for items in the bazaar (e.g., `"brass_statue"`, `"silk_scarf"`, `"cheap_keychain"`)

- [x] **1.2 — SceneContext Model (Input from Unity via Dev B)**
  - This is the `scene_context` dict that Dev B forwards from Unity:
    ```python
    class SceneContext(BaseModel):
        items_in_hand: list[str] = []           # What the user holds in VR
        looking_at: Optional[str] = None        # Gaze-tracked item
        distance_to_vendor: float = 1.0         # Physical proximity
        vendor_npc_id: str = "vendor_01"        # Which vendor NPC
        vendor_happiness: int = 50              # 0-100
        vendor_patience: int = 70               # 0-100
        negotiation_stage: NegotiationStage = NegotiationStage.BROWSING
        current_price: int = 0                  # Current asking price
        user_offer: int = 0                     # User's latest offer
    ```
  - **Key changes from v2.0:**
    - `items_in_hand` is now a **list** (user can hold multiple items), replaces single `held_item`
    - `looking_at` replaces `looked_at_item`
    - `distance_to_vendor` is new (proximity awareness)
    - `vendor_happiness` and `vendor_patience` are now separate from `current_mood`
    - `current_price` and `user_offer` are new (explicit price tracking from Unity side)
  - Add field validators: `vendor_happiness` and `vendor_patience` clamped 0–100

- [x] **1.3 — AI Decision Model (Internal)**
  - This is NOT sent to Dev B directly. This is the structured output we demand from GPT-4o:
    - `reply_text: str` — What the NPC says
    - `new_mood: int` — Updated mood after this interaction
    - `new_stage: NegotiationStage` — Updated negotiation stage
    - `price_offered: Optional[int]` — If the NPC is quoting a price
    - `vendor_happiness: int` — Happiness 0–100
    - `vendor_patience: int` — Patience 0–100
    - `vendor_mood: VendorMood` — Mood category string
    - `internal_reasoning: str` — Why the AI made this decision (for debugging/logging only)

- [x] **1.4 — VendorResponse Model (Output to Dev B)**
  - This is what `generate_vendor_response()` returns (as a dict, matching this schema):
    ```python
    class VendorResponse(BaseModel):
        reply_text: str                 # Vendor's spoken response
        new_mood: int                   # Validated mood (0-100, clamped ±15)
        new_stage: str                  # GREETING|BROWSING|HAGGLING|DEAL|WALKAWAY|CLOSURE
        price_offered: int              # Vendor's current asking price
        vendor_happiness: int           # 0-100
        vendor_patience: int            # 0-100
        vendor_mood: str                # enthusiastic|neutral|annoyed|angry
    ```
  - **Note:** We return a plain `dict` (not a Pydantic model instance) to avoid coupling Dev B to our Pydantic version. But we validate internally using this model before returning.

- [x] **1.5 — Error Handling Contract**
  - If `generate_vendor_response()` fails internally (LLM error, Neo4j error), it raises an exception
  - Dev B catches these and returns appropriate HTTP errors to Unity
  - Define: `class BrainServiceError(Exception)` — raised when LLM fails after retries
  - Define: `class StateStoreError(Exception)` — raised when Neo4j is unreachable
  - These are documented in the interface so Dev B knows what to catch

- [x] **1.6 — Share Contract with Dev B**
  - Document the `generate_vendor_response()` function signature and return schema
  - Document the exception classes Dev B should catch
  - This becomes the "constitution" — no changes without a conversation

### Exit Criteria
- All models pass unit tests with valid and invalid data
- `SceneContext` handles the new Unity payload format
- `VendorResponse` matches the schema Dev B expects (per integration_response.md)
- `NegotiationStage` includes the new stages: WALKAWAY, CLOSURE
- Dev B has reviewed and approved the function contract

---

## Phase 2 — Mock Services Layer

**Goal:** Decouple your development velocity from external dependencies. You should be able to test `generate_vendor_response()` end-to-end using mock OpenAI and mock Neo4j.

> **Architecture v3.0 Note:** We no longer mock STT, TTS, RAG, or Memory — those are Dev B's services
> and Dev B's responsibility. We mock our OWN dependencies: OpenAI (LLM) and Neo4j (state persistence).

### Tasks

- [ ] **2.1 — Mock OpenAI / LLM**
  - Input: The composed prompt (system + user message)
  - Output: A valid `AIDecision` JSON matching our schema
  - Returns deterministic test responses based on input keywords:
    - If transcribed text contains "kitne ka" → return haggling response with price
    - If transcribed text contains "namaste" → return greeting response
    - If transcribed text is empty → return "vendor prompts user" response
  - Should simulate ~200ms latency with `asyncio.sleep` to mimic real API behavior

- [ ] **2.2 — Mock Neo4j / Session Store**
  - Input: Session ID + game state operations
  - Output: In-memory dict-based session storage
  - Supports `load_session()`, `save_session()`, `create_session()` with same signatures
  - Stores state in a module-level dict keyed by session_id
  - No actual Neo4j driver or Cypher queries involved

- [ ] **2.3 — Service Interface / Abstract Base**
  - Define a Protocol or ABC for: `LLMService`, `SessionStore`
  - Mock implementations and real implementations both conform to this interface
  - This allows swapping mocks for real services via config toggle (`USE_MOCKS=true/false`)

- [ ] **2.4 — Config Toggle**
  - `USE_MOCKS=true` → inject mock LLM + mock Neo4j
  - `USE_MOCKS=false` → inject real OpenAI + real Neo4j
  - Full `generate_vendor_response()` can run end-to-end with mocks

### Exit Criteria
- Mock services are callable and return expected shapes
- Toggling `USE_MOCKS` swaps implementations without code changes
- `generate_vendor_response()` can run end-to-end with mocks and return a valid `VendorResponse`

---

## Phase 3 — Function Interface & Dev Endpoint

**Goal:** A clean, importable `generate_vendor_response()` function that Dev B can call, plus an optional dev/test HTTP endpoint for isolated testing.

> **Architecture v3.0 Note:** This phase replaces the old "API Gateway" phase. Dev B owns the production
> endpoint. We provide a function, not a server. The dev endpoint is for our own testing only.

### Tasks

- [ ] **3.1 — Primary Function: `generate_vendor_response()`**
  - Create `app/generate.py` with the function signature matching the integration contract:
    ```python
    async def generate_vendor_response(
        transcribed_text: str,
        context_block: str,
        rag_context: str,
        scene_context: dict,
        session_id: str
    ) -> dict
    ```
  - This function orchestrates **only Dev A's steps** (Steps 7 + 7½):
    1. Parse `scene_context` into `SceneContext` Pydantic model (validate)
    2. Load current game state from Neo4j (via `session_store`)
    3. Call AI Brain with all inputs → get `AIDecision`
    4. Validate the AI's decision (clamp mood, verify legal stage transition)
    5. Persist validated state to Neo4j
    6. Return validated `VendorResponse` as a dict
  - Include request correlation logging (generate internal `request_id`)
  - Log: AI started → AI done → state validated → state persisted → response returned

- [ ] **3.2 — Dev Test Endpoint (Optional)**
  - `POST /api/dev/generate` — wraps `generate_vendor_response()` in an HTTP endpoint
  - Accepts the same arguments as the function in a JSON body
  - Useful for testing with curl/Postman without Dev B's pipeline
  - **Not for production use** — Dev B calls the function directly via import
  - Protected with a simple flag: only available when `LOG_LEVEL=DEBUG`

- [ ] **3.3 — Error Handling**
  - `generate_vendor_response()` handles errors internally and either:
    - Returns a valid response (with fallback vendor reply), or
    - Raises `BrainServiceError` (LLM failed after retries) or `StateStoreError` (Neo4j down)
  - Dev B catches these exceptions and maps them to HTTP errors for Unity
  - RAG context being empty is **not an error** — the function works fine without it

- [ ] **3.4 — Logging & Timing**
  - Log each internal step with timing (milliseconds)
  - Include `session_id` and generated `request_id` in all log lines
  - Total latency budget for `generate_vendor_response()`: ~2-3 seconds (LLM is the bottleneck)

### Exit Criteria
- `from app.generate import generate_vendor_response` works
- Function accepts valid inputs and returns a valid `VendorResponse` dict
- Error cases raise documented exceptions
- Dev test endpoint (if enabled) returns valid responses via HTTP
- Logs show full internal pipeline trace with timing

---

## Phase 4 — The AI Brain (Game Master)

**Goal:** GPT-4o behaves as a consistent, stateful Indian street vendor — not a generic chatbot. It returns structured JSON decisions, not freeform text.

### Tasks

- [ ] **4.1 — System Prompt Engineering ("The God Prompt")**
  - This is the single most important piece of text in the entire project. Dedicate real time to it.
  - The system prompt must establish:
    - **Persona**: Name, personality, backstory. E.g., "You are Ramesh, a 55-year-old brass goods vendor in Jaipur's Johari Bazaar. You've been haggling since age 12. You are shrewd but fair."
    - **Behavioral Rules**:
      - "You NEVER break character."
      - "You respond in the language specified by the conversation context. If Hinglish, use natural mixing."
      - "You address the customer directly. You do not narrate."
    - **State-Driven Behavior** (using `vendor_happiness` and `vendor_patience`):
      - "If `vendor_happiness` < 20: You are irritated. Short sentences. Refuse to negotiate further."
      - "If `vendor_happiness` 20-50: You are skeptical but willing to talk."
      - "If `vendor_happiness` 50-80: You are engaged and enjoying the haggle."
      - "If `vendor_happiness` > 80: You are delighted. Offer a good deal."
      - "If `vendor_patience` < 20: You are about to end the conversation."
    - **Item Awareness**:
      - "`items_in_hand` is a list — the user may be holding multiple items. React to combinations."
      - "If `looking_at` differs from items_in_hand, recognize browsing behavior."
      - "`distance_to_vendor` indicates proximity — if far, the vendor may call out."
    - **Stage Transition Rules**:
      - "You may only move the stage forward via legal transitions."
      - Provide the full state transition graph in the prompt.
      - **Updated stages:** GREETING, BROWSING, HAGGLING, DEAL, WALKAWAY, CLOSURE
    - **Price Awareness**:
      - "`current_price` is the last quoted price. `user_offer` is the user's last counter-offer."
      - "Use `rag_context` for factual information about items. Do not invent prices outside realistic ranges."
    - **Output Format**:
      - "You MUST respond with a JSON object matching the provided schema. No markdown. No explanation outside the JSON."
      - Schema now includes: `reply_text`, `new_mood`, `new_stage`, `price_offered`, `vendor_happiness`, `vendor_patience`, `vendor_mood`, `internal_reasoning`

- [ ] **4.2 — Prompt Template System**
  - Build the prompt as a composable template, not a hardcoded string
  - Sections that change per request: `user_text`, `rag_context`, `held_item`, `looked_at_item`, `current_mood`, `current_stage`
  - Sections that are static: persona, behavioral rules, output schema
  - This allows A/B testing different vendor personalities later

- [ ] **4.3 — OpenAI API Integration**
  - Use the OpenAI Python SDK's `chat.completions.create()`
  - Set `response_format={"type": "json_object"}` to enforce JSON mode
  - Use `temperature=0.7` for personality variation while maintaining coherence
  - Set `max_tokens` appropriately (vendor replies should be short — 2-4 sentences max, ~150 tokens)
  - Include the JSON schema of `AIDecision` in the system prompt so GPT-4o knows the exact structure to produce
  - **New fields in schema:** `vendor_happiness`, `vendor_patience`, `vendor_mood` (in addition to existing `reply_text`, `new_mood`, `new_stage`, `price_offered`)

- [ ] **4.4 — Response Parsing & Validation**
  - Parse GPT-4o's response as JSON
  - Validate against the `AIDecision` Pydantic model
  - If parsing fails (malformed JSON): retry once with a simplified prompt, then fall back to a default safe response ("Haan ji, ek minute..." / "Yes, one moment...")
  - If mood value is out of range: clamp to 0–100
  - If `vendor_happiness` or `vendor_patience` out of range: clamp to 0–100
  - If `vendor_mood` not in allowed values: default to `"neutral"`
  - If stage transition is illegal: keep the current stage and log a warning

- [ ] **4.5 — Retry & Fallback Strategy**
  - On OpenAI API timeout or rate limit (429): retry with exponential backoff (max 2 retries)
  - On persistent failure: return a "vendor is distracted" in-character fallback response so the VR experience doesn't break
  - Never surface raw API errors to Unity

- [ ] **4.6 — Prompt Versioning & Logging**
  - Log every prompt sent to GPT-4o and every response received (at DEBUG level)
  - Include a `prompt_version` field in logs so you can correlate behavior changes with prompt edits
  - Store prompt history (even if just in git) so you can roll back

### Exit Criteria
- Given a mock `transcribed_text` of "Yeh kitne ka hai?" with vendor_happiness=50, stage=BROWSING, and items_in_hand=["brass_statue"]:
  - AI returns valid JSON matching `AIDecision` schema (including new fields: vendor_happiness, vendor_patience, vendor_mood)
  - AI stays in character (Hindi/Hinglish response expected)
  - AI acknowledges the held item(s)
  - AI proposes a price (from RAG context or invented within range)
  - AI transitions stage to HAGGLING
- `generate_vendor_response()` returns a valid dict matching `VendorResponse` schema
- Fallback responses work when API is unreachable
- All responses parse successfully into Pydantic models

---

## Phase 5 — State Machine & Game Logic

**Goal:** The AI's decisions are validated and constrained by a deterministic game logic layer. The AI proposes state changes; the State Engine approves or rejects them. All validated state is persisted to Neo4j.

### Tasks

- [ ] **5.0 — Session Store Implementation (Neo4j)**
  - Implement `app/services/session_store.py` with:
    - `async def load_session(session_id: str) -> GameState` — read from Neo4j
    - `async def save_session(session_id: str, state: GameState) -> None` — write to Neo4j
    - `async def create_session(session_id: str) -> GameState` — create with defaults
  - Neo4j node schema: `(:Session {session_id, mood, vendor_happiness, vendor_patience, stage, turn_count, price_history, created_at, updated_at})`
  - All Cypher queries are in this module only — no Neo4j imports anywhere else
  - Add timeout enforcement (`NEO4J_TIMEOUT_MS`) on every query
  - If Neo4j is unreachable, raise a dedicated `StateStoreError` → 503 at API layer

- [ ] **5.1 — Define the State Transition Graph**
  - Legal transitions (updated for v3.0):
    ```
    GREETING  → BROWSING
    BROWSING  → HAGGLING
    BROWSING  → WALKAWAY
    HAGGLING  → DEAL
    HAGGLING  → WALKAWAY
    HAGGLING  → CLOSURE
    WALKAWAY  → HAGGLING  (vendor calls back — only if vendor_happiness > 40)
    WALKAWAY  → CLOSURE   (final exit)
    ```
  - **Changes from v2.0:**
    - `WALK_AWAY` → `WALKAWAY` (no underscore)
    - `NO_DEAL` removed — replaced by `CLOSURE` (covers both unsuccessful endings and natural conversation end)
    - `CLOSURE` is the universal terminal state for non-deal endings
  - Any transition not in this graph is ILLEGAL and must be blocked
  - Log illegal transition attempts as warnings (indicates prompt needs tuning)

- [ ] **5.2 — Mood & Sentiment Mechanics**
  - **Mood** is a number from 0 to 100 representing the NPC vendor's overall disposition
  - **vendor_happiness** (0-100): How happy the vendor is with the interaction
  - **vendor_patience** (0-100): How patient the vendor remains (decreases with frustrating interactions)
  - **vendor_mood** (string): Categorical — `enthusiastic`, `neutral`, `annoyed`, `angry`
  - Mood delta per interaction should be bounded: max ±15 per turn (prevent wild swings)
  - `vendor_happiness` and `vendor_patience` also clamped ±15 per turn
  - `vendor_mood` is derived from happiness/patience ranges:
    - happiness > 70 → `enthusiastic`
    - happiness 40-70 → `neutral`
    - happiness 20-40 → `annoyed`
    - happiness < 20 → `angry`
  - Define mood modifiers for specific actions:
    - User compliments the shop: +5 to +10 happiness
    - User insults the price: -5 to -10 happiness, -5 patience
    - User picks up wrong item: -3 happiness
    - User shows genuine interest: +5 happiness, +3 patience
  - Note: These are GUIDELINES in the prompt. The State Engine enforces the ±15 clamp as a hard rule.

- [ ] **5.3 — Win/Loss Condition Detection**
  - `DEAL` state = user successfully negotiated. Log the final price. Mark session as "won."
  - `CLOSURE` state = negotiation ended without a deal (or natural conversation end). Log the reason. Mark session as "ended."
  - `WALKAWAY` is a temporary state — can recover (→ HAGGLING) or finalize (→ CLOSURE)
  - Emit a structured event (or log entry) when a terminal state is reached, including: session_id, turns_taken, final_price, final_mood, vendor_happiness, vendor_patience

- [ ] **5.4 — Turn Counter & Guardrails**
  - Track the number of interaction turns per session
  - If turns exceed a threshold (e.g., 30), the vendor should start wrapping up ("Bhai, mujhe doosre customers bhi dekhne hain" / "I have other customers too")
  - This prevents infinite loops and encourages players to make decisions

- [ ] **5.5 — State Validation Function**
  - Build a pure function: `validate_transition(current_stage, proposed_stage, vendor_happiness) → (approved_stage, warnings[])`
  - This function sits between the AI Brain output and the return value of `generate_vendor_response()`
  - It can override the AI's stage proposal if it violates rules
  - Also validates `vendor_happiness`, `vendor_patience`, `vendor_mood` consistency
  - All overrides are logged

- [ ] **5.6 — State Persistence Integration**
  - After state validation passes, persist the new state to Neo4j via `session_store.save_session()`
  - On each call to `generate_vendor_response()`, load the authoritative state from Neo4j via `session_store.load_session()`
  - Server-side state (from Neo4j) is authoritative — Unity's `scene_context` values are treated as supplementary context
  - If session not found in Neo4j, create a fresh session with defaults (mood=50, vendor_happiness=50, vendor_patience=70, stage=GREETING, turn_count=0)
  - Price history is appended per turn (list of offered prices stored in Neo4j)
  - Store `vendor_happiness`, `vendor_patience` alongside mood in Neo4j

### Exit Criteria
- State transition graph is encoded and tested with every legal and illegal transition pair
- Mood clamping works (AI proposes mood=150 → engine returns mood=100)
- `vendor_happiness` and `vendor_patience` are clamped and persisted correctly
- Illegal transitions are blocked and logged
- Terminal states (DEAL, CLOSURE) trigger session summary logging
- Turn counter enforces wrap-up behavior
- Session state is correctly persisted to and loaded from Neo4j between turns
- Missing session auto-creates with default values

---

## Phase 6 — Integration with Developer B

**Goal:** Dev B successfully imports and calls `generate_vendor_response()` from their pipeline. The function works seamlessly within Dev B's orchestration.

> **Architecture v3.0:** Integration direction has flipped. Dev B imports our function — we don't import from Dev B.
> Dev B calls `generate_vendor_response()` at Step 7 of their pipeline.

### Tasks

- [ ] **6.1 — Publish the Function Interface**
  - Ensure `generate_vendor_response()` is cleanly importable:
    ```python
    from app.generate import generate_vendor_response
    ```
  - Document all parameters, return schema, and exceptions in the docstring
  - Provide a standalone usage example for Dev B

- [ ] **6.2 — Neo4j Initialization for Dev B's Runtime**
  - Dev B's server needs to initialize our Neo4j driver at their startup
  - Provide helper functions:
    ```python
    from app.services.session_store import init_neo4j, close_neo4j
    ```
  - Dev B calls `init_neo4j(uri, user, password)` during their FastAPI lifespan startup
  - Dev B calls `close_neo4j()` during shutdown
  - If Dev B doesn't call `init_neo4j()`, our function raises `StateStoreError` with a clear message

- [ ] **6.3 — Configuration Handoff**
  - Dev B's `.env` must include Dev A's config vars: `OPENAI_API_KEY`, `OPENAI_MODEL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - Provide an `.env.example` section for Dev B to add to their environment
  - Ensure our `Settings` class can be initialized by Dev B's app startup code

- [ ] **6.4 — Integration Smoke Test**
  - Write a test that simulates Dev B's call pattern:
    ```python
    result = await generate_vendor_response(
        transcribed_text="भाई ये silk scarf कितने का है?",
        context_block="[Turn 1] User: Namaste!\n[Turn 1] Vendor: Aao!",
        rag_context="Silk Scarf: Wholesale ₹150, Retail ₹300-400",
        scene_context={
            "items_in_hand": ["brass_keychain"],
            "looking_at": "silk_scarf",
            "distance_to_vendor": 1.2,
            "vendor_npc_id": "vendor_01",
            "vendor_happiness": 55,
            "vendor_patience": 70,
            "negotiation_stage": "BROWSING",
            "current_price": 0,
            "user_offer": 0
        },
        session_id="vr-session-abc123"
    )
    assert "reply_text" in result
    assert "vendor_happiness" in result
    assert "vendor_patience" in result
    assert "vendor_mood" in result
    ```
  - Verify the return dict matches the contract Dev B expects

- [ ] **6.5 — Latency Profiling**
  - Measure actual latency of `generate_vendor_response()` in isolation
  - Target: < 3 seconds (LLM is the bottleneck)
  - This becomes one segment of Dev B's total pipeline (~3.5s total)
  - Document findings and optimize if needed

- [ ] **6.6 — Error Behavior Documentation**
  - Document for Dev B exactly what exceptions to catch:
    - `BrainServiceError` → Dev B returns 500 to Unity
    - `StateStoreError` → Dev B returns 503 to Unity
  - Document graceful cases: if `rag_context` is empty, function still works fine
  - If `context_block` is empty (first turn), function handles it gracefully

### Exit Criteria
- Dev B can import and call `generate_vendor_response()` from their endpoint handler
- The function returns a dict matching the agreed schema
- Neo4j initialization works within Dev B's server lifecycle
- Integration smoke test passes with real OpenAI + real Neo4j
- Full pipeline latency (Dev B's endpoint) is under 5 seconds for a typical interaction
- Error cases are handled and documented

---

## Phase 7 — End-to-End Testing & Hardening

**Goal:** Confidence that the system won't crash during a VR demo. Every edge case is handled gracefully.

### Tasks

- [ ] **7.1 — Happy Path Test Suite**
  - Full negotiation flow via `generate_vendor_response()`: GREETING → BROWSING → HAGGLING → DEAL
  - Walkaway and recovery: HAGGLING → WALKAWAY → HAGGLING → DEAL
  - Various items_in_hand combinations and their impact on AI behavior
  - Verify all return fields: reply_text, new_mood, new_stage, price_offered, vendor_happiness, vendor_patience, vendor_mood

- [ ] **7.2 — Edge Case Test Suite**
  - Empty transcribed_text ("") → vendor prompts user to speak
  - Empty rag_context ("") → function works without cultural context
  - Empty context_block ("") → function works on first turn
  - Invalid scene_context dict → validation error or sensible defaults
  - Session with 50+ turns → Vendor wrap-up behavior triggers
  - vendor_happiness at extremes (0 and 100) → AI behavior is appropriate
  - vendor_patience at 0 → vendor ends conversation
  - All illegal state transitions attempted → All blocked
  - Stages: WALKAWAY → HAGGLING only when vendor_happiness > 40

- [ ] **7.3 — Chaos Testing**
  - Simulate OpenAI API down → Verify `BrainServiceError` raised
  - Simulate OpenAI rate limiting → Verify retry + fallback
  - Simulate Neo4j down → Verify `StateStoreError` raised
  - Simulate malformed AI JSON → Verify retry and default response
  - All error cases produce clean exceptions with clear messages for Dev B

- [ ] **7.4 — Load Testing (Basic)**
  - Simulate 5-10 concurrent sessions (realistic for a demo)
  - Verify no race conditions, no shared state corruption in Neo4j
  - Verify Neo4j handles concurrent session reads/writes without deadlocks
  - Measure memory usage under sustained load

- [ ] **7.5 — Integration Test with Dev B**
  - Coordinate with Developer B
  - Dev B calls `generate_vendor_response()` from their endpoint handler
  - Verify the function works within Dev B's async event loop
  - Verify Neo4j initialization/shutdown works within Dev B's lifespan
  - Test full pipeline: Unity → Dev B endpoint → our function → Dev B response → Unity
  - Verify vendor_happiness, vendor_patience, vendor_mood flow correctly to Unity

### Exit Criteria
- 100% of happy path scenarios pass
- All edge cases return structured responses (no raw exceptions)
- System survives chaos scenarios gracefully
- Unity developer confirms end-to-end demo works

---

## Phase 8 — Deployment & Observability

**Goal:** The server runs reliably in a hosted environment and you can diagnose issues without SSH-ing into the box.

### Tasks

- [ ] **8.1 — Containerization**
  - Write a Dockerfile for the FastAPI app
  - Multi-stage build: slim Python image, copy only requirements + app code
  - Expose port 8000
  - Health check: `GET /health`
  - Neo4j runs as a separate container (or managed service) — NOT bundled in the app image
  - Provide a `docker-compose.yml` with both `app` and `neo4j` services for local development

- [ ] **8.2 — Environment Configuration**
  - All secrets via environment variables (never in code or Docker image)
  - Document required env vars in README and `.env.example`
  - Neo4j env vars: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_TIMEOUT_MS`

- [ ] **8.3 — Structured Logging**
  - JSON-formatted logs with: timestamp, request_id, session_id, step, duration_ms, level
  - Log levels: DEBUG (full prompts/responses), INFO (pipeline flow), WARN (fallbacks triggered), ERROR (failures)
  - Ensure logs are parseable by any log aggregation tool

- [ ] **8.4 — Metrics Endpoint (Optional but Recommended)**
  - Expose basic metrics: request count, average latency per step, error rate, AI fallback rate
  - Even a simple `/metrics` JSON endpoint is valuable for a demo

- [ ] **8.5 — Deployment Target**
  - Decide hosting: Cloud VM, Cloud Run, or similar
  - Ensure the deployment supports WebSocket (if future streaming is needed)
  - Configure CORS to allow requests from Unity WebXR builds (if applicable)

### Exit Criteria
- Docker image builds and runs successfully
- Server starts and passes health check in deployed environment
- Logs are structured and searchable
- Unity headset can reach the deployed server

---

## 3. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPT-4o returns invalid JSON despite JSON mode | Medium | High | Pydantic validation + retry + fallback response |
| GPT-4o breaks character / goes off-script | Medium | Medium | Strong system prompt + temperature tuning + post-processing |
| `generate_vendor_response()` latency too high | Medium | High | Aggressive LLM timeout + prompt size optimization |
| Dev B's pipeline adds unexpected latency | Medium | Medium | Profile our function independently; optimize our ~2s budget |
| OpenAI rate limits during demo | Low | Critical | Retry with backoff + pre-warm + consider caching common responses |
| Neo4j connection lost mid-session | Low | High | `StateStoreError` raised → Dev B returns 503. Auto-reconnect on next call via driver pool |
| Neo4j adds latency to pipeline | Low | Medium | Neo4j read/write budget is 2s max. Local/containerized Neo4j keeps latency <20ms typically |
| Scene context format changes from Unity | Medium | Medium | Pydantic validation with defaults for missing fields; communicate changes via Dev B |
| New stage enum values not handled | Low | High | Strict enum validation; unknown values rejected with clear error |

---

## 4. Dependency Map

```
Phase 0 (Scaffold)
   │
   ▼
Phase 1 (Data Contracts)  ──────────────────────────┐
   │                                                  │
   ▼                                                  ▼
Phase 2 (Mocks)                              Share schema with
   │                                          Unity Dev + Dev B
   ▼
Phase 3 (API Gateway)
   │
   ▼
Phase 4 (AI Brain) ◄──── Iterative prompt tuning (ongoing)
   │
   ▼
Phase 5 (State Machine)
   │
   ├──────────────────────┐
   ▼                      ▼
Phase 6 (Integration)   Phase 7 (Testing)
   │                      │
   └──────────┬───────────┘
              ▼
        Phase 8 (Deploy)
```

**Developer B Dependency:** Phase 6 only. Everything else is fully independent.

---

## 5. Milestones & Deliverables

| Milestone | Deliverable | Target |
|-----------|------------|--------|
| M1: Skeleton Running | Project structure + `generate_vendor_response()` stub returning mock data | End of Day 2 |
| M2: AI Brain Online | GPT-4o responds in-character with valid JSON, returns all required fields (including vendor_happiness, vendor_patience, vendor_mood) | End of Day 4 |
| M3: State Machine Solid | All state transitions tested (WALKAWAY, CLOSURE), mood mechanics enforced, Neo4j persistence working | End of Day 6 |
| M4: Integration Ready | Dev B can import and call `generate_vendor_response()`, Neo4j init/shutdown helpers work in Dev B's runtime | End of Day 8 |
| M5: Demo Ready | Full pipeline tested (Unity → Dev B → our function → Dev B → Unity), all edge cases handled | End of Day 10 |

---

## Appendix: Design Decisions & Rationale

### Why Server-Side State Authority?
Unity sends `vendor_happiness`, `vendor_patience`, and `negotiation_stage` with each request via `scene_context`, but the **server's response values are authoritative**. If Unity and server ever disagree, Unity must adopt the server's values. This prevents cheating and state desync. Dev B passes Unity's scene_context to our function, and we return the validated state.

### Why Neo4j for State Persistence?
Game state (mood, vendor_happiness, vendor_patience, stage, turn count, price history) is persisted in Neo4j so sessions survive server restarts and enable multi-instance deployments. Neo4j was chosen because the project's future roadmap includes relationship-rich features (vendor networks, item provenance graphs, cultural knowledge graphs) that benefit from a native graph database. For the current MVP, it stores flat session state — but the schema is ready to grow. All Neo4j access is encapsulated in `app/services/session_store.py` to keep the blast radius small.

### Why a Function, Not an API Endpoint? (v3.0 Decision)
In v2.0, Dev A owned the FastAPI endpoint and orchestrated the full pipeline. In v3.0, Dev B owns the endpoint because they control the "senses" (STT, TTS, RAG, Memory) that bookend the pipeline. Dev A provides `generate_vendor_response()` as a function that Dev B imports and calls at Step 7. This avoids an extra network hop (function call vs HTTP call), simplifies deployment (one server, not two), and gives Dev B full control over the request lifecycle.

### Why Mocks First?
The #1 risk for any multi-developer project is integration delay. By building mocks for our OWN dependencies (OpenAI, Neo4j), Developer A can achieve full-pipeline confidence independently. The mock layer costs ~1 hour to build and saves potentially days of blocked time. Note: We no longer mock Dev B's services (STT, TTS, RAG) — those are outside our scope.

### Why Clamp AI Decisions?
LLMs are probabilistic. Even with JSON mode and explicit instructions, GPT-4o might return `vendor_happiness: -50` or transition from `GREETING` directly to `DEAL`. The State Engine (Phase 5) exists as a deterministic safety net. The AI proposes; the engine disposes.

---

*This document is a living plan. Update it as decisions are made and phases are completed.*
*Architecture v3.0 — Feb 13, 2026: Updated to reflect Dev B owning the endpoint, Dev A providing `generate_vendor_response()`.*
