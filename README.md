# Samvad XR -- Culturally Aware Immersive Language Learning Platform

**A VR-based language learning system that teaches real-world cultural communication through AI-driven, context-aware vendor negotiation simulations.**

Samvad XR transforms language acquisition from passive memorization into active cultural immersion. Users practice negotiating with a culturally authentic AI street vendor inside a VR environment, learning not just *what* to say, but *how*, *when*, and *why* to say it.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Solution](#the-solution)
3. [System Architecture](#system-architecture)
4. [Repository Map](#repository-map)
5. [This Repository -- Context Engine](#this-repository----context-engine-orchestration-backend)
6. [Core Components](#core-components)
7. [Request Lifecycle -- The 11-Step Pipeline](#request-lifecycle----the-11-step-pipeline)
8. [State Machine -- Negotiation Flow](#state-machine----negotiation-flow)
9. [Neo4j Graph Schema](#neo4j-graph-schema)
10. [AI Brain -- LLM Integration](#ai-brain----llm-integration)
11. [The God Prompt Architecture](#the-god-prompt-architecture)
12. [State Engine -- Validation Layer](#state-engine----validation-layer)
13. [Data Models](#data-models)
14. [Configuration](#configuration)
15. [Error Handling and Resilience](#error-handling-and-resilience)
16. [Testing](#testing)
17. [Getting Started](#getting-started)
18. [Performance Budgets](#performance-budgets)
19. [Technology Stack](#technology-stack)
20. [Related Repositories](#related-repositories)
21. [License](#license)

---

## The Problem

Millions of people study languages through apps, textbooks, and flashcards, yet few can hold a confident conversation in the real world. The reason is straightforward: language is not just vocabulary and grammar. It is culture, context, and behavior.

Traditional methods fail to teach the critical unspoken rules of communication. A negotiation in Tokyo looks entirely different from one in Tamil Nadu or New York. The gestures, the polite indirectness, the acceptable haggling strategies -- these are deeply cultural nuances that flashcards simply cannot teach.

Corporate training for employees relocating abroad is expensive, unscalable, and often relies on static briefings rather than practical experience. Learners are left knowing what to say, but not how, when, or why to say it, leading to anxiety and culture shock when they finally travel.

---

## The Solution

Samvad XR bridges this gap by transforming language learning into a high-fidelity "flight simulator" for cultural interaction. The platform provides a safe, immersive VR environment where users learn by doing, not by reading.

**Context Over Content.** Instead of isolated vocabulary drills, users face real-world scenarios. The MVP scenario is negotiating with an Indian street vendor in Jaipur's Johari Bazaar.

**Cultural RAG Engine.** The backend does not simply translate text. It understands culture. A Retrieval-Augmented Generation system grounded in cultural etiquette data ensures that if you are learning Japanese, the AI vendor expects politeness and indirect refusal. If you are learning a regional Indian language, the AI understands the energetic dynamic of street bargaining.

**State-Aware Interaction.** A Graph Database (Neo4j) tracks the emotional state of the conversation through defined stages (GREETING, INQUIRY, HAGGLING, DEAL, WALKAWAY, CLOSURE). The vendor's behavior is driven by a dynamic Happiness Score that reacts to price offers, conversational tone, and cultural adherence, forcing users to adapt strategy in real-time.

**Scalable Corporate Training.** Companies can deploy Samvad XR to train teams on global business etiquette -- from a boardroom in Berlin to a market in Mumbai -- without hiring expensive consultants.

---

## System Architecture

Samvad XR is composed of three independently developed subsystems that communicate through well-defined interfaces:

```
+------------------------------------------------------+
|                       Unity / XR Client               |
|  (VR headset, scene rendering, audio capture,         |
|   sends POST /api/interact with audio + scene state)  |
+----------------------------+-------------------------+
                             |
                             | HTTP (JSON + Base64 audio)
                             v
+------------------------------------------------------+
|            RAG Intelligence Layer (Dev B)              |
|  Owns: API endpoint, STT, TTS, RAG, Conv. Memory     |
|  Repo: SamvadXR_buildindia_RAG_Intelligence_Layer     |
|                                                       |
|  Step 1:  Receive request, parse body                 |
|  Step 2:  Decode base64 audio                         |
|  Step 3:  Speech-to-Text (Sarvam API)                 |
|  Step 4:  Store user turn in conversation memory      |
|  Step 5:  Build context block from history            |
|  Step 6:  RAG retrieval (ChromaDB)                    |
|       |                                               |
|       |   from app.generate import                    |
|       |       generate_vendor_response                |
|       v                                               |
|  +------------------------------------------------+  |
|  |       Context Engine / Orchestration (Dev A)    |  |
|  |       THIS REPOSITORY                           |  |
|  |                                                 |  |
|  |  Step 7:  Parse scene -> Load Neo4j state ->    |  |
|  |           Compose prompt -> Call GPT-4o ->      |  |
|  |           Parse JSON response                   |  |
|  |  Step 7.5: Validate via state engine: clamp     |  |
|  |            mood, verify stage transition,        |  |
|  |            persist to Neo4j graph               |  |
|  +------------------------------------------------+  |
|       |                                               |
|       v                                               |
|  Step 8:  Store vendor turn in conversation memory    |
|  Step 9:  Text-to-Speech (Sarvam API)                 |
|  Step 10: Encode audio to base64                      |
|  Step 11: Return InteractResponse to Unity            |
+------------------------------------------------------+
                             |
                             | HTTP Response
                             v
+------------------------------------------------------+
|                       Unity / XR Client               |
|  (Plays vendor audio, updates scene state,            |
|   displays negotiation UI)                            |
+------------------------------------------------------+
```

**Ownership boundaries are strict.** Dev B owns 9 of the 11 pipeline steps. This repository (Dev A) provides a single function -- `generate_vendor_response()` -- that Dev B imports and calls at Step 7. There is no reverse dependency; Dev A never calls Dev B.

---

## Repository Map

The Samvad XR platform spans three repositories:

| Repository | Purpose | Owner |
|------------|---------|-------|
| [SamvadXR_BuildIndia](https://github.com/rs0125/SamvadXR_BuildIndia) | Unity/XR client -- VR scene, audio capture, user interface, scene state management | XR Team |
| [SamvadXR_buildindia_RAG_Intelligence_Layer](https://github.com/raghavvag/SamvadXR_buildindia_RAG_Intelligence_Layer.git) | RAG context with middleware -- API endpoint, STT/TTS (Sarvam), ChromaDB RAG retrieval, conversation memory, base64 encoding | Dev B |
| **This repository** (samvad-context-engine) | Orchestration backend -- AI brain (GPT-4o), state machine, Neo4j graph persistence, prompt engineering | Dev A |

---

## This Repository -- Context Engine (Orchestration Backend)

This repository contains the AI decision-making core and game state management for Samvad XR. It is responsible for:

- Generating culturally authentic vendor dialogue via GPT-4o with structured JSON output
- Enforcing negotiation rules through a deterministic state machine
- Persisting session state (mood, stage, turn count, price history) in a Neo4j graph database
- Providing graph-aware context to the LLM so it can reason about conversation history
- Validating and clamping all AI proposals before they reach the user

The entire public interface is a single async function:

```python
from app.generate import generate_vendor_response

result = await generate_vendor_response(
    transcribed_text="Bhaiya ye tomato kitne ka hai?",
    context_block=context_block,       # from conversation memory
    rag_context=rag_context,           # from ChromaDB RAG
    scene_context=request.scene_context,  # from Unity
    session_id=request.session_id
)
# result: {"reply_text": str, "happiness_score": int,
#          "negotiation_state": str, "vendor_mood": str}
```

---

## Core Components

```
samvad-context-engine/
|
|-- app/                              # Dev A's domain
|   |-- main.py                       # FastAPI dev server (testing only)
|   |-- generate.py                   # PRIMARY INTERFACE: generate_vendor_response()
|   |-- config.py                     # Pydantic Settings (env var validation)
|   |-- dependencies.py               # Dependency injection (mock/real toggle)
|   |-- exceptions.py                 # BrainServiceError, StateStoreError
|   |-- logging_config.py             # Structured JSON logging
|   |
|   |-- models/                       # Pydantic data contracts
|   |   |-- enums.py                  # NegotiationStage, VendorMood, LanguageCode
|   |   |-- request.py                # SceneContext (input from Unity)
|   |   |-- response.py               # AIDecision (internal), VendorResponse (output)
|   |
|   |-- prompts/                      # LLM prompt engineering
|   |   |-- vendor_system.py          # The "God Prompt" -- persona, rules, schema
|   |
|   |-- services/                     # Business logic services
|       |-- ai_brain.py               # OpenAI GPT-4o client with retry/fallback
|       |-- state_engine.py           # Deterministic state machine validator
|       |-- session_store.py          # Neo4j persistence (all Cypher lives here)
|       |-- mocks.py                  # Mock LLM + session store for testing
|       |-- protocols.py              # Protocol interfaces (LLMService, SessionStore)
|
|-- services/                         # Dev B's domain (stubs in this repo)
|   |-- voice_ops.py                  # STT/TTS via Sarvam (Dev B owns)
|   |-- rag_ops.py                    # ChromaDB retrieval (Dev B owns)
|   |-- context_memory.py             # Conversation memory (Dev B owns)
|   |-- middleware.py                  # Base64 encoding/decoding (Dev B owns)
|
|-- tests/                            # Test suite
|   |-- test_models.py                # Pydantic validation (100% coverage)
|   |-- test_state_engine.py          # All legal + illegal transitions
|   |-- test_ai_brain.py              # LLM integration tests
|   |-- test_generate.py              # generate_vendor_response() tests
|   |-- test_api.py                   # FastAPI endpoint tests
|   |-- test_mocks.py                 # Mock conformance tests
|   |-- test_neo4j_integration.py     # Live Neo4j tests (manual)
|
|-- rules.md                          # Project rules (enforced, not aspirational)
|-- INTEGRATION_GUIDE.md              # Contract between Dev A and Dev B
|-- requirements.txt                  # Python dependencies
|-- pyproject.toml                    # Project metadata, tool config
```

---

## Request Lifecycle -- The 11-Step Pipeline

Every user interaction follows this exact sequence. Dev A's contribution spans Steps 7 and 7.5.

```
Step  Owner  Module                Action                                      Latency
----  -----  --------------------  ------------------------------------------  --------
 1    Dev B  main.py               Receive POST /api/interact, parse body        ~0ms
 2    Dev B  middleware.py          base64_to_bytes(request.audio_base64)         ~1ms
 3    Dev B  voice_ops.py          transcribe_with_sarvam(bytes, "hi-IN")      ~800ms
 4    Dev B  context_memory.py     memory.add_turn("user", text, metadata)       ~1ms
 5    Dev B  context_memory.py     context_block = memory.get_context_block()    ~1ms
 6    Dev B  rag_ops.py            rag_ctx = retrieve_context(text, 3)          ~50ms
 7    Dev A  generate.py           generate_vendor_response(...)               ~2000ms
 7.5  Dev A  state_engine.py       Validate: clamp mood, verify transition      ~20ms
 8    Dev B  context_memory.py     memory.add_turn("vendor", reply, metadata)    ~1ms
 9    Dev B  voice_ops.py          speak_with_sarvam(reply, "hi-IN")           ~600ms
10    Dev B  middleware.py          bytes_to_base64(audio)                        ~1ms
11    Dev B  main.py               Return InteractResponse to Unity              ~0ms
                                                                       TOTAL  ~3.5s
```

### What Happens Inside Step 7 (generate_vendor_response)

1. **Parse and validate** `scene_context` via Pydantic `SceneContext` model
2. **Load or create** session state from Neo4j (authoritative source)
3. **Retrieve graph context** -- traverse the session's turn history, stage transitions, and item interactions from Neo4j
4. **Check terminal state** -- if session is already DEAL or CLOSURE, return a closure response immediately
5. **Enforce turn limits** -- if turn count exceeds 30, force CLOSURE
6. **Compose the system prompt** -- assemble the God Prompt with dynamic state, graph context, and behavioral rules
7. **Compose the user message** -- delimit transcribed text, conversation history, and RAG context with injection-safe markers
8. **Call GPT-4o** -- JSON mode, structured output, with retry and fallback
9. **Validate via state engine** -- clamp happiness delta to plus/minus 15, verify stage transition is legal, enforce offer-happiness consistency, derive vendor mood from numeric score
10. **Persist updated state** to Neo4j (session node, turn nodes, stage transition nodes, item nodes)
11. **Return validated response** as a plain dict

---

## State Machine -- Negotiation Flow

The negotiation progresses through six defined stages with strictly enforced transition rules:

```
                    +----------+
                    | GREETING |
                    +----+-----+
                         |
                         v
                    +----------+
                    | INQUIRY  |
                    +----+-----+
                         |
                    +----+----+
                    |         |
                    v         v
              +-----------+  +-----------+
              | HAGGLING  |  | WALKAWAY  |<-------+
              +-----+-----+  +-----+-----+        |
                    |               |              |
              +-----+-----+        |  (happiness   |
              |     |     |        |   > 40 only)  |
              v     v     v        +---------------+
          +------+ +---+ +--------+
          | DEAL | |   | | CLOSURE|
          +------+ |   | +--------+
         (terminal)|   | (terminal)
                   v   v
             WALKAWAY / CLOSURE
```

### Transition Rules

| From | Legal Targets | Conditions |
|------|--------------|------------|
| GREETING | INQUIRY | Customer asks about an item or price |
| INQUIRY | HAGGLING, WALKAWAY | Price negotiation begins, or customer disengages |
| HAGGLING | DEAL, WALKAWAY, CLOSURE | Agreement reached, customer leaves, or negotiation fails |
| WALKAWAY | HAGGLING, CLOSURE | Re-entry ONLY if happiness > 40 and customer explicitly re-engages |
| DEAL | (none) | Terminal -- session ends |
| CLOSURE | (none) | Terminal -- session ends |

### Happiness Score Mechanics

The vendor's happiness score (0-100) drives behavioral tone:

| Score Range | Vendor Mood | Behavioral Effect |
|-------------|-------------|-------------------|
| 81-100 | Enthusiastic | Genuinely pleased, ready to give a good deal |
| 61-80 | Friendly | Warm, conversational, willing to chat |
| 41-60 | Neutral | Engaged, casual, matter-of-fact banter |
| 21-40 | Annoyed | Skeptical, slightly guarded |
| 0-20 | Angry | Irritated, short sentences, reluctant to negotiate |

The happiness score is clamped to a maximum delta of plus/minus 15 per turn, preventing dramatic mood swings. Insulting offers (assessed below 25% of quoted price) enforce a minimum happiness drop of 10 points. Lowball offers (25-40% of quoted price) enforce a minimum drop of 6 points.

---

## Neo4j Graph Schema

Neo4j serves as the authoritative store for all game logic state. Conversation dialogue history is stored separately by Dev B's conversation memory module. The graph schema:

### Node Labels

```
(:Session {
    session_id,
    happiness_score,
    negotiation_state,
    turn_count,
    created_at,
    updated_at
})

(:Turn {
    session_id,
    turn_number,
    role,              -- "user" or "vendor"
    text_snippet,      -- truncated to 150 characters
    happiness_score,
    stage,
    object_grabbed,
    timestamp
})

(:Item {
    name,
    session_id
})

(:StageTransition {
    session_id,
    from_stage,
    to_stage,
    at_turn,
    happiness_at_transition,
    timestamp
})
```

### Relationships

```
(:Session)-[:HAS_TURN]->(:Turn)
(:Turn)-[:FOLLOWED_BY]->(:Turn)
(:Turn)-[:ABOUT_ITEM]->(:Item)
(:Session)-[:INVOLVES_ITEM]->(:Item)
(:Session)-[:STAGE_CHANGED]->(:StageTransition)
```

The graph context is traversed before each LLM call to provide the AI with awareness of stage occupancy durations, happiness trends, items discussed, and transition history. This enables graph-aware stage reasoning -- the AI knows not to rush through stages and can calibrate behavior based on how long the conversation has been in its current phase.

---

## AI Brain -- LLM Integration

The AI Brain (`app/services/ai_brain.py`) manages all communication with OpenAI GPT-4o.

### Key Design Decisions

- **JSON mode is mandatory.** Every LLM call uses `response_format={"type": "json_object"}` to guarantee parseable output.
- **Structured output.** The response is parsed into an `AIDecision` Pydantic model before any further processing. Raw AI output never reaches the user.
- **Retry policy.** Up to 2 retries with exponential backoff (1s, 2s) on transient errors (5xx, timeouts, rate limits). Non-retryable errors (4xx) fail immediately.
- **Fallback response.** If all retries are exhausted, the system returns a safe, in-character fallback: *"One minute brother, hold on... yes, what were you saying?"*
- **Timeout enforcement.** Configurable via `AI_TIMEOUT_MS` (default 10,000ms).

### Protocol-Based Abstraction

Services are defined as Python `Protocol` classes (`app/services/protocols.py`):

```python
class LLMService(Protocol):
    async def generate_decision(
        self, system_prompt: str, user_message: str, *,
        temperature: float = 0.7, max_tokens: int = 200,
    ) -> AIDecision: ...

class SessionStore(Protocol):
    async def create_session(self, session_id: str) -> dict[str, Any]: ...
    async def load_session(self, session_id: str) -> Optional[dict[str, Any]]: ...
    async def save_session(self, session_id: str, state: dict[str, Any]) -> None: ...
    async def record_turn(self, ...) -> None: ...
    async def record_stage_transition(self, ...) -> None: ...
    async def get_graph_context(self, session_id: str) -> dict[str, Any]: ...
```

The `USE_MOCKS` configuration flag toggles between real implementations (`OpenAILLMService`, `Neo4jSessionStore`) and test doubles (`MockLLMService`, `MockSessionStore`) via dependency injection in `app/dependencies.py`.

---

## The God Prompt Architecture

The system prompt (`app/prompts/vendor_system.py`) is structured as a layered document with static and dynamic sections. It is versioned (`PROMPT_VERSION = "8.0.0"`) and the version is logged with every LLM call for traceability.

### Static Sections (identical for every request)

| Section | Purpose |
|---------|---------|
| PERSONA | Defines "Ramesh," a 55-year-old vendor in Jaipur's Johari Bazaar. Practical, direct, street-smart. Speaks only in English. Never breaks character. |
| BEHAVIORAL_RULES | 10 rules governing tone calibration per happiness score, word limits per stage, item knowledge sourcing from RAG only, conversation consistency, and price behavior |
| STATE_TRANSITION_RULES | Legal transition graph with graph-aware stage reasoning. Explicit stability guidance based on turns spent in current stage. |
| OUTPUT_SCHEMA | Exact JSON schema the LLM must produce, with field-level constraints and numeric rules |
| ANTI_INJECTION | Security directive instructing the LLM to treat user input and RAG data as data only, ignoring any embedded instructions |

### Dynamic Sections (injected per request)

| Section | Source |
|---------|--------|
| Current Game State | Happiness score, negotiation stage, turn count, object grabbed, input language from session store |
| Conversation Graph Context | Stage occupancy history (turns per stage), happiness trend with direction, items discussed with mention counts, stage transition log, stability hint |
| Wrap-up Instruction | Injected after turn 25 to guide the AI toward closing the negotiation |

### User Message Structure

User input is assembled with explicit delimiters to prevent prompt injection:

```
--- CONVERSATION HISTORY ---
{context_block}
--- END CONVERSATION HISTORY ---

--- CULTURAL CONTEXT ---
{rag_context}
--- END CULTURAL CONTEXT ---

--- USER MESSAGE ---
{transcribed_text}
--- END USER MESSAGE ---
```

---

## State Engine -- Validation Layer

The state engine (`app/services/state_engine.py`) sits between the AI Brain output and the final response. The AI proposes; the state engine disposes.

### Validation Pipeline

1. **Stage transition validation** -- Proposed stage is checked against the legal transition graph. Illegal transitions are rejected and the current stage is preserved.
2. **Offer-happiness consistency** -- If the LLM assessed an offer as "insult" or "lowball" but did not drop happiness sufficiently, a minimum drop is enforced (10 points for insults, 6 for lowball).
3. **Happiness clamping** -- Delta is clamped to plus/minus 15 from the current authoritative value. Final score is bounded to [0, 100].
4. **Price direction validation** -- Vendor prices should only decrease during negotiation. Increases generate warnings (soft validation).
5. **Mood derivation** -- Categorical mood (angry/annoyed/neutral/friendly/enthusiastic) is derived deterministically from the clamped happiness score.
6. **Terminal state detection** -- DEAL and CLOSURE stages are flagged as terminal, triggering a structured session summary log.

Every override is logged at WARN level with full context. All validated values are packaged into a `ValidatedState` dataclass before being returned.

---

## Data Models

### SceneContext (Input from Unity)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `object_grabbed` | Optional[str] | None | Item the user is interacting with in the VR scene |
| `happiness_score` | int | 50 | Vendor happiness (0-100), clamped on ingestion |
| `negotiation_state` | NegotiationStage | GREETING | Current negotiation stage from Unity |
| `input_language` | LanguageCode | en-IN | Language the user is speaking |
| `target_language` | LanguageCode | en-IN | Language the vendor should reply in |

### AIDecision (Internal -- parsed from GPT-4o JSON output)

| Field | Type | Description |
|-------|------|-------------|
| `reply_text` | str | What the vendor says (always in English) |
| `happiness_score` | int | Proposed happiness after this turn (0-100) |
| `negotiation_state` | NegotiationStage | Proposed stage after this turn |
| `vendor_mood` | VendorMood | Categorical mood descriptor |
| `internal_reasoning` | str | AI's reasoning for this decision (debug/logging only) |
| `counter_price` | Optional[int] | Price the vendor is quoting (used for validation) |
| `offer_assessment` | Optional[str] | Assessment of customer's offer: insult, lowball, fair, good, excellent, none |
| `suggested_user_response` | str | Contextual hint for what the user could say next |

### VendorResponse (Output returned to Dev B)

| Field | Type | Description |
|-------|------|-------------|
| `reply_text` | str | Validated vendor dialogue |
| `happiness_score` | int | Clamped and validated happiness score |
| `negotiation_state` | str | Approved negotiation stage |
| `vendor_mood` | str | Derived deterministically from happiness score |

### Supported Languages

| Code | Language |
|------|----------|
| hi-IN | Hindi |
| kn-IN | Kannada |
| ta-IN | Tamil |
| en-IN | Indian English |
| hi-EN | Hinglish |

---

## Configuration

All configuration is loaded from environment variables via Pydantic Settings (`app/config.py`). The application fails fast at startup if required variables are missing.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | -- | OpenAI API key for GPT-4o |
| `OPENAI_MODEL` | No | gpt-4o | LLM model identifier |
| `NEO4J_URI` | No | bolt://localhost:7687 | Neo4j connection URI |
| `NEO4J_USER` | No | neo4j | Neo4j username |
| `NEO4J_PASSWORD` | No | (empty) | Neo4j password |
| `NEO4J_TIMEOUT_MS` | No | 2000 | Per-query Neo4j timeout in milliseconds |
| `USE_MOCKS` | No | false | Toggle mock services for isolated testing |
| `LOG_LEVEL` | No | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `AI_TIMEOUT_MS` | No | 10000 | LLM call timeout in milliseconds |
| `AI_TEMPERATURE` | No | 0.7 | GPT-4o sampling temperature |
| `AI_MAX_TOKENS` | No | 200 | Maximum response tokens |
| `MAX_TURNS` | No | 30 | Hard limit on turns per session |
| `MAX_MOOD_DELTA` | No | 15 | Maximum happiness change per turn |
| `APP_VERSION` | No | 0.1.0 | Application version string |
| `CORS_ORIGINS` | No | * | Comma-separated allowed origins |

---

## Error Handling and Resilience

### Exception Hierarchy

| Exception | Trigger | Downstream Effect |
|-----------|---------|-------------------|
| `BrainServiceError` | LLM call failed after all retries, invalid scene context, JSON parse failure | Dev B returns HTTP 500 to Unity |
| `StateStoreError` | Neo4j unreachable, Cypher query failure, driver not initialized | Dev B returns HTTP 503 to Unity |

### Graceful Degradation

| Failed Component | Behavior |
|-----------------|----------|
| RAG (ChromaDB) | `rag_context=""` is accepted without error. The function operates without cultural context. |
| Conversation Memory | `context_block=""` is treated as a first turn. No error raised. |
| OpenAI API | Retry twice with exponential backoff (1s, 2s). On exhaustion, return an in-character fallback response. Only raise `BrainServiceError` for non-retryable failures. |
| Neo4j | Hard failure. Cannot proceed without authoritative state. Raises `StateStoreError` immediately. |
| Graph context retrieval | Soft failure. If graph traversal fails, the pipeline continues without graph-aware context in the prompt. |

### Retry Policy

| Component | Max Retries | Backoff | Retry Conditions |
|-----------|-------------|---------|------------------|
| OpenAI API | 2 | Exponential (1s, 2s) | 5xx server errors, timeouts, rate limits (429) |
| Neo4j | 0 | -- | Fail immediately on any error |

Non-retryable errors (4xx from OpenAI, authentication failures) are never retried.

---

## Testing

### Test Structure

```
tests/
|-- test_models.py              # Pydantic validation: valid, invalid, edge cases
|-- test_state_engine.py        # All legal and illegal state transitions
|-- test_ai_brain.py            # LLM integration with mock/real OpenAI
|-- test_generate.py            # Full generate_vendor_response() pipeline
|-- test_api.py                 # FastAPI dev endpoint tests
|-- test_mocks.py               # Verify mocks conform to protocol interfaces
|-- test_neo4j_integration.py   # Live Neo4j tests (marked, run manually)
|-- conftest.py                 # Shared fixtures
```

### Running Tests

```bash
# Unit tests only (no external services required)
pytest tests/ -m "not integration and not neo4j_integration"

# All tests including integration
pytest tests/

# With verbose output
pytest tests/ -v --tb=short
```

### Test Markers

| Marker | Purpose |
|--------|---------|
| `integration` | Tests that call real external APIs (OpenAI). Deselect in CI. |
| `neo4j_integration` | Tests that require a live Neo4j instance. Run manually. |

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Neo4j database instance (local or Neo4j Aura cloud)
- OpenAI API key with GPT-4o access

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd samvad-context-engine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```dotenv
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
USE_MOCKS=false
LOG_LEVEL=INFO
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password-here
NEO4J_TIMEOUT_MS=3000
```

### Running the Dev Server

The FastAPI dev server is for isolated testing only. In production, Dev B imports `generate_vendor_response()` directly.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify the server is running:

```bash
curl http://localhost:8000/health
# {"status": "ok", "version": "0.1.0"}
```

### Dev Test Endpoint

When `LOG_LEVEL=DEBUG`, a test endpoint is available at `POST /api/dev/generate`:

```bash
curl -X POST http://localhost:8000/api/dev/generate \
  -H "Content-Type: application/json" \
  -d '{
    "transcribed_text": "Bhaiya ye tomato kitne ka hai?",
    "context_block": "",
    "rag_context": "",
    "scene_context": {
      "object_grabbed": "Tomato",
      "happiness_score": 50,
      "negotiation_state": "GREETING",
      "input_language": "en-IN",
      "target_language": "en-IN"
    },
    "session_id": "test-session-001"
  }'
```

### Mock Mode

Set `USE_MOCKS=true` to run without OpenAI or Neo4j dependencies. The mock LLM returns deterministic responses based on keyword matching in the user's speech. The mock session store uses an in-memory dictionary. This mode is designed for isolated development and testing.

---

## Performance Budgets

| Component | Target Latency | Hard Limit |
|-----------|---------------|------------|
| Scene context parsing | < 5ms | 50ms |
| Neo4j state load | ~20ms | 2,000ms |
| AI Brain (GPT-4o) | ~2,000ms | 10,000ms |
| State validation and clamping | < 5ms | 50ms |
| Neo4j state persist | ~20ms | 2,000ms |
| Response assembly | < 5ms | 50ms |
| **generate_vendor_response() total** | **~2.1s** | **~12s** |
| **Full pipeline (all 11 steps)** | **~3.5s** | **~20s** |

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Runtime | Python | 3.11+ | Async-first backend |
| Web Framework | FastAPI | 0.115 | Dev server for testing |
| ASGI Server | Uvicorn | 0.34 | Development server |
| Data Validation | Pydantic | 2.10 | Models, settings, request/response contracts |
| LLM | OpenAI GPT-4o | -- | AI brain with structured JSON generation |
| Graph Database | Neo4j | 5.27 | Session state persistence, conversation graph |
| Async HTTP | httpx | 0.28 | Async HTTP client |
| Logging | python-json-logger | 3.2 | Structured JSON log output |
| Testing | pytest + pytest-asyncio | 8.3 / 0.25 | Async-aware test runner |
| Configuration | pydantic-settings + python-dotenv | 2.7 / 1.0 | Environment variable management |

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [SamvadXR_BuildIndia](https://github.com/rs0125/SamvadXR_BuildIndia) | Unity/XR client for the immersive VR vendor negotiation experience |
| [SamvadXR_buildindia_RAG_Intelligence_Layer](https://github.com/raghavvag/SamvadXR_buildindia_RAG_Intelligence_Layer.git) | RAG intelligence layer with middleware -- API endpoint, speech-to-text, text-to-speech, ChromaDB retrieval, and conversation memory |

---

## License

This project was developed for the Build India Hackathon.

---

*Architecture v3.0 -- Samvad XR Context Engine*
*Last updated: February 2026*
