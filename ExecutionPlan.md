# Samvad XR — Developer A Execution Plan

## The Brain & Nervous System of Samvad XR

**Owner:** Developer A (Architect & AI Logic)
**Stack:** FastAPI · OpenAI GPT-4o · Pydantic · Neo4j
**Last Updated:** 2026-02-13

---

## Table of Contents

1. [High-Level System Flow](#1-high-level-system-flow)
2. [Phase Breakdown](#2-phase-breakdown)
3. [Phase 0 — Project Scaffolding & Environment](#phase-0--project-scaffolding--environment)
4. [Phase 1 — Data Contracts (Pydantic Models)](#phase-1--data-contracts-pydantic-models)
5. [Phase 2 — Mock Services Layer](#phase-2--mock-services-layer)
6. [Phase 3 — The API Gateway](#phase-3--the-api-gateway)
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

The entire request lifecycle from Unity headset to response follows this pipeline:

```
Unity VR Client
      │
      ▼
┌─────────────────────────────────┐
│  POST /interact                 │  ◄── API Gateway (FastAPI)
│  (Audio blob + Game Metadata)   │
└──────────────┬──────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  INGESTION LAYER                 │
│  • Validate request via Pydantic │
│  • Extract audio + metadata      │
│  • Assign request correlation ID │
└──────────────┬───────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐  ┌──────────────┐
│  STT Call   │  │  RAG Lookup  │  ◄── Developer B's Services
│  (Sarvam)   │  │  (ChromaDB)  │      (Mocked initially)
└──────┬──────┘  └──────┬───────┘
       │                │
       └───────┬────────┘
               ▼
┌──────────────────────────────────┐
│  COGNITION LAYER (Your Core)     │
│  • Build GPT-4o prompt           │
│  • Inject: user_text, rag_ctx,   │
│    game_state, held_item         │
│  • Force JSON-mode response      │
│  • Parse AI decision             │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  STATE TRANSITION ENGINE         │
│  • Validate AI's proposed state  │
│  • Clamp mood (0-100)            │
│  • Enforce legal stage moves     │
│  • Persist state to Neo4j        │
│  • Log state delta               │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  TTS Call (Sarvam)               │  ◄── Developer B's Service
│  (Mocked initially)             │      (Mocked initially)
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  RESPONSE ASSEMBLY               │
│  • Pack audio_base64             │
│  • Attach new_mood, new_stage    │
│  • Attach subtitle_text          │
│  • Strict Pydantic serialization │
└──────────────┬───────────────────┘
               │
               ▼
         Unity VR Client
```

### Key Principle: "Never Trust, Always Validate"

Every boundary — incoming request, AI output, state transition — gets validated. Unity crashes on bad JSON. The AI hallucinates invalid states. Your job is to be the gatekeeper at every seam.

---

## 2. Phase Breakdown

| Phase | Name | Blocked By | Est. Effort | Priority |
|-------|------|------------|-------------|----------|
| 0 | Project Scaffolding & Environment | Nothing | 0.5 day | P0 |
| 1 | Data Contracts (Pydantic Models) | Nothing | 1 day | P0 |
| 2 | Mock Services Layer | Phase 1 | 0.5 day | P0 |
| 3 | API Gateway (FastAPI Endpoint) | Phase 1, 2 | 1 day | P0 |
| 4 | AI Brain (GPT-4o Integration) | Phase 1, 3 | 2 days | P0 |
| 5 | State Machine & Game Logic | Phase 4 | 1.5 days | P0 |
| 6 | Integration with Developer B | Dev B ready | 1-2 days | P1 |
| 7 | End-to-End Testing & Hardening | Phase 5 | 1.5 days | P1 |
| 8 | Deployment & Observability | Phase 7 | 1 day | P2 |

**Critical Path:** 0 → 1 → 2 → 3 → 4 → 5 → 7

---

## Phase 0 — Project Scaffolding & Environment

**Goal:** A running, empty FastAPI server with proper project structure and dependency management.

### Tasks

- [ ] **0.1 — Repository & Folder Structure**
  - Define the project layout:
    ```
    SamVadXR-Orchestration/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py              # FastAPI app + endpoint
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── request.py       # Pydantic input models
    │   │   └── response.py      # Pydantic output models
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── ai_brain.py      # GPT-4o cognition logic
    │   │   ├── state_engine.py  # State machine / transition logic
    │   │   ├── session_store.py # Neo4j session state read/write
    │   │   └── mocks.py         # Mock STT, TTS, RAG
    │   ├── prompts/
    │   │   └── vendor_system.py # System prompt templates
    │   ├── config.py            # Settings, env vars, constants
    │   └── dependencies.py      # FastAPI dependency injection
    ├── tests/
    │   ├── test_models.py
    │   ├── test_api.py
    │   ├── test_ai_brain.py
    │   └── test_state_engine.py
    ├── .env.example
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md
    ```
  - Rationale: Separation of concerns. Models are isolated from logic. Services are independently testable. Prompts live in their own namespace so they can be versioned and iterated without touching business logic.

- [ ] **0.2 — Dependency Pinning**
  - Lock versions for: `fastapi`, `uvicorn`, `openai`, `pydantic`, `python-dotenv`, `httpx` (for async HTTP), `neo4j` (async driver), `pytest`, `pytest-asyncio`
  - Pin OpenAI SDK to a specific version to avoid breaking changes in structured output APIs
  - Pin `neo4j` driver to a specific version for deterministic builds

- [ ] **0.3 — Environment & Configuration**
  - Define all config via environment variables (12-factor app style)
  - Required env vars: `OPENAI_API_KEY`, `OPENAI_MODEL` (default `gpt-4o`), `USE_MOCKS` (boolean toggle), `LOG_LEVEL`
  - Neo4j env vars: `NEO4J_URI` (default `bolt://localhost:7687`), `NEO4J_USER` (default `neo4j`), `NEO4J_PASSWORD`, `NEO4J_TIMEOUT_MS` (default `2000`)
  - Create a Pydantic `Settings` class that reads from `.env` and fails fast on missing required vars

- [ ] **0.4 — Health Check Endpoint**
  - `GET /health` — returns `{"status": "ok", "version": "0.1.0"}`
  - This is the first thing Unity devs will hit to verify connectivity. Ship it immediately.

- [ ] **0.5 — Neo4j Connection Setup**
  - Add Neo4j async driver initialization in the FastAPI lifespan (startup)
  - Verify connectivity on startup; log success or hard-fail with a clear error
  - Close the driver cleanly in the lifespan shutdown hook
  - Create `app/services/session_store.py` as the ONLY module that imports the Neo4j driver
  - When `USE_MOCKS=true`, Neo4j connection is still initialized but session reads/writes can be stubbed

### Exit Criteria
- `uvicorn app.main:app --reload` starts without errors
- `/health` returns 200
- All env vars are loaded and validated
- Neo4j driver connects successfully on startup (or logs a clear failure)

---

## Phase 1 — Data Contracts (Pydantic Models)

**Goal:** The single source of truth for what Unity sends and what Unity receives. This is the most important artifact you produce — Unity dev, Developer B, and your own AI logic all depend on it.

### Tasks

- [ ] **1.1 — Define Enums & Constants**
  - `NegotiationStage` enum: `GREETING`, `BROWSING`, `HAGGLING`, `DEAL`, `NO_DEAL`, `WALK_AWAY`
  - `LanguageCode` enum: `hi` (Hindi), `kn` (Kannada), `ta` (Tamil), `en` (English), `hi-en` (Hinglish)
  - `MoodRange`: Constrained integer 0–100
  - `ItemID`: String enum or free-form string for items in the bazaar (e.g., `"brass_statue"`, `"silk_scarf"`, `"cheap_keychain"`)

- [ ] **1.2 — InteractionRequest Model (Input)**
  - Fields to define:
    - `audio_base64: str` — Raw audio from Unity mic, base64-encoded
    - `held_item: Optional[str]` — What the user is physically holding in VR (nullable if hands are empty)
    - `looked_at_item: Optional[str]` — What the user's gaze is focused on (for gaze-tracking cues)
    - `current_mood: int` — NPC's current mood (0–100), maintained by Unity between calls
    - `current_stage: NegotiationStage` — Where in the negotiation flow are we
    - `language_code: LanguageCode` — What language the user is speaking
    - `session_id: str` — Unique session identifier for logging/state tracking
  - Add field validators:
    - `current_mood` must be clamped 0–100
    - `audio_base64` must be non-empty
    - `session_id` must be non-empty

- [ ] **1.3 — AI Decision Model (Internal)**
  - This is NOT sent to Unity. This is the structured output you demand from GPT-4o:
    - `reply_text: str` — What the NPC says
    - `new_mood: int` — Updated mood after this interaction
    - `new_stage: NegotiationStage` — Updated negotiation stage
    - `price_offered: Optional[int]` — If the NPC is quoting a price
    - `internal_reasoning: str` — Why the AI made this decision (for debugging/logging only)

- [ ] **1.4 — InteractionResponse Model (Output to Unity)**
  - Fields:
    - `audio_base64: str` — NPC's voice response, base64-encoded
    - `subtitle_text: str` — Text version of NPC's reply (for accessibility/subtitle overlay)
    - `new_mood: int` — Updated NPC mood for Unity to store
    - `new_stage: NegotiationStage` — Updated stage for Unity to store
    - `price_offered: Optional[int]` — Current price on the table (null if not applicable)
    - `language_code: LanguageCode` — Language of the response
  - Ensure all fields have explicit `json_schema_extra` examples for OpenAPI docs

- [ ] **1.5 — Error Response Model**
  - `ErrorResponse`: `error_code: str`, `message: str`, `detail: Optional[str]`
  - Map to HTTP status codes: 400 (bad input), 422 (validation error), 500 (AI failure), 503 (downstream service unavailable)

- [ ] **1.6 — Publish & Share Contract**
  - Generate OpenAPI JSON schema from FastAPI's auto-docs
  - Share with Unity developer and Developer B as a versioned artifact
  - This becomes the "constitution" — no field changes without a conversation

### Exit Criteria
- All models pass unit tests with valid and invalid data
- OpenAPI spec at `/docs` renders correctly with examples
- Unity developer and Developer B have reviewed and approved the contract

---

## Phase 2 — Mock Services Layer

**Goal:** Decouple your development velocity from Developer B's progress. You should be able to test the entire pipeline end-to-end using fake STT, TTS, and RAG.

### Tasks

- [ ] **2.1 — Mock STT (Speech-to-Text)**
  - Input: `audio_base64: str`, `language_code: str`
  - Output: Hardcoded Hindi/English phrases based on a rotation list
  - Examples: `"Yeh kitne ka hai?"`, `"Too expensive, give me a discount"`, `"I like this statue"`
  - Should simulate ~200ms latency with `asyncio.sleep` to mimic real service behavior

- [ ] **2.2 — Mock TTS (Text-to-Speech)**
  - Input: `text: str`, `language_code: str`
  - Output: A small, valid base64-encoded audio stub (can be a tiny silent WAV or a fixed sample)
  - Should simulate ~300ms latency

- [ ] **2.3 — Mock RAG (Retrieval-Augmented Generation)**
  - Input: `query: str`
  - Output: Hardcoded context snippets mapped to keywords
  - Example: If query contains "statue" → return `"This brass statue is from Jaipur, typical retail price ₹800-1200, vendors usually start at ₹1500"`
  - If no keyword match → return `"No specific context available for this item"`

- [ ] **2.4 — Service Interface / Abstract Base**
  - Define an abstract interface (Protocol class or ABC) for each service: `STTService`, `TTSService`, `RAGService`
  - Mock implementations and real implementations (Phase 6) will both conform to this interface
  - This allows swapping mocks for real services via config toggle (`USE_MOCKS=true/false`) without changing any orchestration code

- [ ] **2.5 — Dependency Injection Setup**
  - Use FastAPI's `Depends()` system to inject the correct service implementation
  - When `USE_MOCKS=true` → inject mock services
  - When `USE_MOCKS=false` → inject real services (Developer B's module)

### Exit Criteria
- Mock services are callable and return expected shapes
- Toggling `USE_MOCKS` swaps implementations without code changes
- Full pipeline can run end-to-end with mocks (even if AI brain isn't built yet — use a mock AI too if needed)

---

## Phase 3 — The API Gateway

**Goal:** A single, robust `POST /interact` endpoint that orchestrates the entire pipeline.

### Tasks

- [ ] **3.1 — Endpoint Skeleton**
  - Define `POST /interact` in FastAPI
  - Accept `InteractionRequest` as the request body
  - Return `InteractionResponse`
  - Add proper HTTP status code responses in OpenAPI decorators (200, 400, 422, 500, 503)

- [ ] **3.2 — Request Correlation & Logging**
  - Generate a unique `request_id` (UUID) for every incoming request
  - Attach it to all log lines for that request's lifecycle
  - Log: request received → STT started → STT done → RAG started → RAG done → AI started → AI done → TTS started → TTS done → response sent
  - Include timing for each step (milliseconds)
  - This is critical for debugging latency issues when running on a VR headset where every millisecond matters

- [ ] **3.3 — Orchestration Flow**
  - The endpoint function should follow this strict sequence:
    1. Validate & parse the request (Pydantic does this automatically)
    2. Call STT service with `audio_base64` and `language_code` → get `user_text`
    3. Call RAG service with `user_text` → get `rag_context`
    4. Load current game state from Neo4j (via `session_store`)
    5. Call AI Brain with `user_text`, `rag_context`, and full game state → get `AIDecision`
    6. Validate the AI's decision (clamp mood, verify legal stage transition)
    7. Persist validated state to Neo4j (mood, stage, turn count, price history)
    8. Call TTS service with `reply_text` and `language_code` → get `audio_base64`
    9. Assemble and return `InteractionResponse`

- [ ] **3.4 — Parallelization Consideration**
  - Steps 2 (STT) must complete before Step 3 (RAG), since RAG needs the transcribed text
  - However, consider if STT and any "pre-fetch" can happen in parallel in future iterations
  - Document these dependencies clearly for future optimization

- [ ] **3.5 — Error Handling Strategy**
  - Wrap each service call in try/except
  - If STT fails → return 503 with "Speech recognition unavailable"
  - If RAG fails → continue without context (graceful degradation — the AI can still respond, just without factual grounding)
  - If AI Brain fails → return 500 with "AI processing error"
  - If TTS fails → return the text response with an empty `audio_base64` and a flag (so Unity can fall back to on-screen text)
  - Never let an exception bubble up as a raw 500 — always structured `ErrorResponse`

- [ ] **3.6 — Timeout Management**
  - Set per-service timeouts:
    - STT: 5 seconds
    - RAG: 3 seconds
    - Neo4j read/write: 2 seconds
    - AI Brain: 10 seconds (GPT-4o can be slow)
    - TTS: 5 seconds
  - Total request budget: ~20 seconds max (VR user is standing there waiting)
  - If any service exceeds its timeout, abort and return a partial response or error
  - Neo4j timeout is a hard failure (503) — state integrity cannot be compromised

### Exit Criteria
- `POST /interact` accepts a valid request and returns a valid response (using mocks)
- Logs show full pipeline trace with timing
- Error scenarios return structured errors, not raw exceptions
- Tested via `curl`, Postman, or `pytest` with `httpx.AsyncClient`

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
      - "You respond in the language specified by `language_code`. If `hi-en`, use natural Hinglish mixing."
      - "You address the customer directly. You do not narrate."
    - **State-Driven Behavior**:
      - "If `current_mood` < 20: You are irritated. Short sentences. Refuse to negotiate further."
      - "If `current_mood` 20-50: You are skeptical but willing to talk."
      - "If `current_mood` 50-80: You are engaged and enjoying the haggle."
      - "If `current_mood` > 80: You are delighted. Offer a good deal."
    - **Item Awareness**:
      - "If `held_item` is a cheap item (e.g., keychain) but the user is asking about an expensive item, recognize this as a distraction tactic. React accordingly."
      - "If `held_item` matches the conversation topic, engage more deeply."
    - **Stage Transition Rules**:
      - "You may only move the stage forward, never backward (except WALK_AWAY which can revert to HAGGLING if the vendor calls the customer back)."
      - Provide the full state transition graph in the prompt.
    - **RAG Usage**:
      - "Use `rag_context` for factual information about items. Do not invent prices or histories."
    - **Output Format**:
      - "You MUST respond with a JSON object matching the provided schema. No markdown. No explanation outside the JSON."

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

- [ ] **4.4 — Response Parsing & Validation**
  - Parse GPT-4o's response as JSON
  - Validate against the `AIDecision` Pydantic model
  - If parsing fails (malformed JSON): retry once with a simplified prompt, then fall back to a default safe response ("Haan ji, ek minute..." / "Yes, one moment...")
  - If mood value is out of range: clamp to 0–100
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
- Given a mock `user_text` of "Yeh kitne ka hai?" with mood=50, stage=BROWSING, and held_item="brass_statue":
  - AI returns valid JSON matching `AIDecision` schema
  - AI stays in character (Hindi/Hinglish response expected)
  - AI acknowledges the held item
  - AI proposes a price (from RAG context or invented within range)
  - AI transitions stage to HAGGLING
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
  - Neo4j node schema: `(:Session {session_id, mood, stage, turn_count, price_history, created_at, updated_at})`
  - All Cypher queries are in this module only — no Neo4j imports anywhere else
  - Add timeout enforcement (`NEO4J_TIMEOUT_MS`) on every query
  - If Neo4j is unreachable, raise a dedicated `StateStoreError` → 503 at API layer

- [ ] **5.1 — Define the State Transition Graph**
  - Legal transitions:
    ```
    GREETING  → BROWSING
    BROWSING  → HAGGLING
    BROWSING  → WALK_AWAY
    HAGGLING  → DEAL
    HAGGLING  → NO_DEAL
    HAGGLING  → WALK_AWAY
    WALK_AWAY → HAGGLING  (vendor calls back — only if mood > 40)
    WALK_AWAY → NO_DEAL   (final exit)
    ```
  - Any transition not in this graph is ILLEGAL and must be blocked
  - Log illegal transition attempts as warnings (indicates prompt needs tuning)

- [ ] **5.2 — Mood Mechanics**
  - Mood is a number from 0 to 100 representing the NPC vendor's disposition
  - Mood influences:
    - Willingness to negotiate (mood < 20 = refuses)
    - Price flexibility (mood > 70 = willing to offer discounts)
    - Verbal tone (injected into prompt context)
  - Mood delta per interaction should be bounded: max ±15 per turn (prevent wild swings)
  - Define mood modifiers for specific actions:
    - User compliments the shop: +5 to +10
    - User insults the price: -5 to -10
    - User picks up wrong item: -3
    - User shows genuine interest: +5
  - Note: These are GUIDELINES in the prompt. The State Engine enforces the ±15 clamp as a hard rule.

- [ ] **5.3 — Win/Loss Condition Detection**
  - `DEAL` state = user successfully negotiated. Log the final price. Mark session as "won."
  - `NO_DEAL` state = negotiation collapsed. Log the reason. Mark session as "lost."
  - `WALK_AWAY` is a temporary state — can recover or finalize
  - Emit a structured event (or log entry) when a terminal state is reached, including: session_id, turns_taken, final_price, final_mood

- [ ] **5.4 — Turn Counter & Guardrails**
  - Track the number of interaction turns per session
  - If turns exceed a threshold (e.g., 30), the vendor should start wrapping up ("Bhai, mujhe doosre customers bhi dekhne hain" / "I have other customers too")
  - This prevents infinite loops and encourages players to make decisions

- [ ] **5.5 — State Validation Function**
  - Build a pure function: `validate_transition(current_stage, proposed_stage, current_mood) → (approved_stage, warnings[])`
  - This function sits between the AI Brain output and the final response assembly
  - It can override the AI's stage proposal if it violates rules
  - All overrides are logged

- [ ] **5.6 — State Persistence Integration**
  - After state validation passes, persist the new state to Neo4j via `session_store.save_session()`
  - On each request, load the authoritative state from Neo4j via `session_store.load_session()`
  - Server-side state (from Neo4j) is authoritative — Unity's values are treated as hints only
  - If session not found in Neo4j, create a fresh session with defaults (mood=50, stage=GREETING, turn_count=0)
  - Price history is appended per turn (list of offered prices stored in Neo4j)

### Exit Criteria
- State transition graph is encoded and tested with every legal and illegal transition pair
- Mood clamping works (AI proposes mood=150 → engine returns mood=100)
- Illegal transitions are blocked and logged
- Terminal states (DEAL, NO_DEAL) trigger session summary logging
- Turn counter enforces wrap-up behavior
- Session state is correctly persisted to and loaded from Neo4j between turns
- Missing session auto-creates with default values

---

## Phase 6 — Integration with Developer B

**Goal:** Replace mock services with Developer B's real Sarvam AI (STT/TTS) and ChromaDB (RAG) implementations.

### Tasks

- [ ] **6.1 — Define the Integration Contract with Developer B**
  - Agree on exact function signatures:
    - STT: `async def transcribe(audio_base64: str, language_code: str) -> str`
    - TTS: `async def synthesize(text: str, language_code: str) -> str` (returns base64 audio)
    - RAG: `async def retrieve_context(query: str) -> str` (returns context string)
  - Agree on error behavior: What exceptions do they raise? What does "no result" look like?
  - Document timeout expectations for each service

- [ ] **6.2 — Adapter Pattern Implementation**
  - If Developer B's function signatures don't exactly match your abstract interface (Phase 2.4), write thin adapter wrappers
  - Keep the adapters in a separate module (e.g., `app/services/sarvam_adapter.py`)
  - This insulates your orchestration code from changes in Developer B's API

- [ ] **6.3 — Feature Flag Toggle**
  - Ensure `USE_MOCKS` env var cleanly toggles between mock and real services
  - Support granular toggles if needed: `USE_MOCK_STT`, `USE_MOCK_TTS`, `USE_MOCK_RAG`
  - This allows partial integration (e.g., real STT but mock RAG) during development

- [ ] **6.4 — Integration Smoke Test**
  - Write a test that hits the real `POST /interact` with:
    - A real audio clip (pre-recorded Hindi phrase)
    - Real game state metadata
  - Verify the full loop: real STT → real RAG → AI Brain → real TTS → valid response
  - This is the "moment of truth" test

- [ ] **6.5 — Latency Profiling**
  - After real services are connected, measure actual latency of each step
  - Compare against the timeout budgets set in Phase 3.6
  - Identify bottlenecks (STT and TTS are likely the slowest)
  - Document findings and propose optimizations (e.g., audio streaming, response chunking)

### Exit Criteria
- Real Sarvam STT transcribes actual Hindi audio correctly
- Real ChromaDB RAG returns relevant item context
- Real Sarvam TTS generates audible Hindi speech
- Full pipeline latency is under 15 seconds for a typical interaction
- Feature flags allow instant rollback to mocks if a service breaks

---

## Phase 7 — End-to-End Testing & Hardening

**Goal:** Confidence that the system won't crash during a VR demo. Every edge case is handled gracefully.

### Tasks

- [ ] **7.1 — Happy Path Test Suite**
  - Full negotiation flow: GREETING → BROWSING → HAGGLING → DEAL
  - Each language: Hindi, English, Hinglish, Kannada, Tamil
  - Various held items and their impact on AI behavior

- [ ] **7.2 — Edge Case Test Suite**
  - Empty audio (silence) → AI should prompt user: "Kuch bola aapne?" / "Did you say something?"
  - Garbled/unintelligible STT output → AI should ask to repeat
  - Rapid-fire requests (user mashing interact button) → Rate limiting or debounce
  - Session with 50+ turns → Vendor wrap-up behavior triggers
  - Mood at extremes (0 and 100) → AI behavior is appropriate
  - All illegal state transitions attempted → All blocked

- [ ] **7.3 — Chaos Testing**
  - Simulate STT service down → Verify graceful 503
  - Simulate OpenAI rate limiting → Verify retry + fallback
  - Simulate TTS returning garbage → Verify text-only fallback
  - Simulate malformed AI JSON → Verify retry and default response

- [ ] **7.4 — Load Testing (Basic)**
  - Simulate 5-10 concurrent sessions (realistic for a demo)
  - Verify no race conditions, no shared state corruption in Neo4j
  - Verify Neo4j handles concurrent session reads/writes without deadlocks
  - Measure memory usage under sustained load

- [ ] **7.5 — Unity Integration Test**
  - Coordinate with Unity developer
  - Send real requests from the VR headset
  - Verify audio plays correctly, subtitles render, mood/stage updates are reflected in VR UI
  - Test on-device latency (network + processing)

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
| Sarvam STT/TTS latency is too high for VR | High | High | Aggressive timeouts + text-only fallback + latency profiling |
| Developer B's services aren't ready on time | Medium | Low | Mock layer (Phase 2) completely decouples your timeline |
| Unity sends unexpected/malformed requests | Medium | Medium | Strict Pydantic validation + detailed error responses |
| OpenAI rate limits during demo | Low | Critical | Retry with backoff + pre-warm + consider caching common responses |
| Mood/stage state gets desynchronized between Unity and server | Medium | High | Server is authoritative — Unity must use server's returned values |
| Neo4j connection lost mid-session | Low | High | Hard failure (503) — state integrity is non-negotiable. Auto-reconnect on next request via driver pool |
| Neo4j adds latency to pipeline | Low | Medium | Neo4j read/write budget is 2s max. Local/containerized Neo4j keeps latency <20ms typically |

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
| M1: Skeleton Running | FastAPI server with `/health` + `/interact` (mocked) returning valid schema | End of Day 2 |
| M2: AI Brain Online | GPT-4o responds in-character with valid JSON, state transitions work | End of Day 4 |
| M3: State Machine Solid | All state transitions tested, mood mechanics enforced, guardrails active | End of Day 6 |
| M4: Real Services Connected | Sarvam STT/TTS + ChromaDB RAG integrated and working | End of Day 8 |
| M5: Demo Ready | Full end-to-end flow tested from VR headset, deployed and accessible | End of Day 10 |

---

## Appendix: Design Decisions & Rationale

### Why Server-Side State Authority?
Unity sends `current_mood` and `current_stage` with each request, but the SERVER's response values are authoritative. If Unity and server ever disagree, Unity must adopt the server's values. This prevents cheating and state desync.

### Why Neo4j for State Persistence?
Game state (mood, stage, turn count, price history) is persisted in Neo4j so sessions survive server restarts and enable multi-instance deployments. Neo4j was chosen because the project's future roadmap includes relationship-rich features (vendor networks, item provenance graphs, cultural knowledge graphs) that benefit from a native graph database. For the current MVP, it stores flat session state — but the schema is ready to grow. All Neo4j access is encapsulated in `app/services/session_store.py` to keep the blast radius small.

### Why Not WebSockets (Yet)?
Request-response via HTTP POST is simpler to debug, test, and mock. WebSockets add complexity (connection management, reconnection logic) that isn't justified until we need streaming audio responses. The architecture is designed so that WebSocket support can be added as a transport layer without changing the orchestration logic.

### Why Mocks First?
The #1 risk for any multi-developer project is integration delay. By building mocks that conform to the same interface as real services, Developer A can achieve full-pipeline confidence independently. The mock layer costs ~2 hours to build and saves potentially days of blocked time.

### Why Clamp AI Decisions?
LLMs are probabilistic. Even with JSON mode and explicit instructions, GPT-4o might return `mood: -50` or transition from `GREETING` directly to `DEAL`. The State Engine (Phase 5) exists as a deterministic safety net. The AI proposes; the engine disposes.

---

*This document is a living plan. Update it as decisions are made and phases are completed.*
