# Samvad XR — Integration Guide for Developer B

> **What this document is:** A clear contract between Dev A and Dev B. It tells Dev B exactly what
> function Dev A provides, what arguments it expects, what it returns, and how to initialize
> Dev A's dependencies within Dev B's server.
>
> **TL;DR:** Dev B builds the ears (STT), the mouth (TTS), the cultural memory (RAG), the conversation
> memory, and owns the API endpoint. Dev A builds the brain (LLM agent) and the game rules (state engine).
> Dev B imports Dev A's function and calls it mid-pipeline.
>
> **Architecture v3.0:** Dev B owns `POST /api/interact`. Dev A provides `generate_vendor_response()`.

---

## 1. The Big Picture — Who Owns What

### Complete Request Lifecycle (v3.0 — Finalized)

```
Step  Owner   Module              Action                                         Time
────  ──────  ──────────────────  ─────────────────────────────────────────────   ────
 1    Dev B   main.py             Receive POST /api/interact, parse request        0ms
 2    Dev B   middleware.py        base64_to_bytes(request.audio_base64)           1ms
 3    Dev B   voice_ops.py         await transcribe_with_sarvam(bytes, "hi-IN")  ~800ms
                                   → "भाई ये silk scarf कितने का है?"
 4    Dev B   context_memory.py    memory.add_turn("user", text, metadata)         1ms
 5    Dev B   context_memory.py    context_block = memory.get_context_block()       1ms
 6    Dev B   rag_ops.py           rag_ctx = await retrieve_context(text, 3)      ~50ms
 7    Dev A   generate.py          generate_vendor_response(                       ~2s
                                     transcribed_text, context_block,
                                     rag_context, scene_context, session_id)
 7½   Dev A   state_engine.py      (Internal) Validate via Neo4j: clamp mood,    ~20ms
                                     verify stage transition is legal
 8    Dev B   context_memory.py    memory.add_turn("vendor", reply, metadata)       1ms
 9    Dev B   voice_ops.py         audio = await speak_with_sarvam(reply, "hi-IN") ~600ms
10    Dev B   middleware.py         b64 = bytes_to_base64(audio)                     1ms
11    Dev B   main.py              Return InteractResponse to Unity                 0ms
                                                                    TOTAL ≈ 3.5s
```

**Dev B owns 9 of the 11 steps.** Dev A provides a single function (Steps 7 + 7½) that Dev B calls.

> **Note on Neo4j (Steps 7 & 7½):** Dev A uses Neo4j as the persistent state store for session game
> state (mood, stage, turn count, price history). This is entirely Dev A's domain — Dev B never
> interacts with Neo4j directly. Dev B's conversation memory (`context_memory.py`) handles dialogue
> history; Dev A's Neo4j graph handles game logic state.

---

## 2. What Dev A Provides — The Function Interface

### 2.1 Primary Function: `generate_vendor_response()`

This is the **one function** Dev B calls at Step 7:

```python
# Dev B imports this:
from app.generate import generate_vendor_response

# Dev B calls it like this:
result = await generate_vendor_response(
    transcribed_text="भाई ये silk scarf कितने का है?",   # ← from Step 3 (STT)
    context_block=context_block,                           # ← from Step 5 (memory)
    rag_context=rag_context,                               # ← from Step 6 (RAG)
    scene_context=request.scene_context,                   # ← from Unity
    session_id=request.session_id                          # ← from Unity
)
```

**Parameters:**

| Parameter | Type | Source | Description |
|-----------|------|--------|-------------|
| `transcribed_text` | `str` | Dev B (Step 3 STT) | What the user said, in native script |
| `context_block` | `str` | Dev B (Step 5 memory) | Formatted conversation history string |
| `rag_context` | `str` | Dev B (Step 6 RAG) | Cultural/item knowledge from ChromaDB |
| `scene_context` | `dict` | Unity (via Dev B) | Game state from VR scene (see §2.2) |
| `session_id` | `str` | Unity (via Dev B) | Unique session identifier |

**Returns:** `dict` with the following schema:

```python
{
    "reply_text": str,          # Vendor's spoken response
    "new_mood": int,            # Validated mood (0-100, clamped ±15)
    "new_stage": str,           # "GREETING"|"BROWSING"|"HAGGLING"|"DEAL"|"WALKAWAY"|"CLOSURE"
    "price_offered": int,       # Vendor's current asking price
    "vendor_happiness": int,    # 0-100
    "vendor_patience": int,     # 0-100
    "vendor_mood": str          # "enthusiastic"|"neutral"|"annoyed"|"angry"
}
```

### 2.2 Scene Context Format (from Unity)

Dev B forwards the `scene_context` dict from Unity unchanged:

```python
{
    "items_in_hand": ["brass_keychain"],     # List of items user holds
    "looking_at": "silk_scarf",              # Gaze-tracked item
    "distance_to_vendor": 1.2,              # Physical proximity
    "vendor_npc_id": "vendor_01",           # Which vendor NPC
    "vendor_happiness": 55,                  # 0-100
    "vendor_patience": 70,                   # 0-100
    "negotiation_stage": "BROWSING",        # Current stage from Unity
    "current_price": 0,                      # Last quoted price
    "user_offer": 0                          # User's latest offer
}
```

> **Note:** Dev A's Neo4j state is authoritative. The `scene_context` values from Unity are treated
> as supplementary context (e.g., `items_in_hand`, `looking_at`, `distance_to_vendor`), but mood/stage
> authority comes from Neo4j.

### 2.3 Exception Classes

Dev A raises these exceptions that Dev B should catch:

```python
from app.services.session_store import StateStoreError
from app.services.ai_brain import BrainServiceError
```

| Exception | When Raised | Dev B's Response to Unity |
|-----------|-------------|--------------------------|
| `BrainServiceError` | LLM failed after retries | Return 500: "AI processing error" |
| `StateStoreError` | Neo4j is unreachable | Return 503: "Game state unavailable" |

If `rag_context` is empty (`""`), the function works fine — graceful degradation.
If `context_block` is empty (`""`), the function works fine — it's treated as the first turn.

### 2.4 Neo4j Initialization

Dev A's function requires Neo4j. Dev B must initialize it at server startup:

```python
# In Dev B's FastAPI lifespan:
from app.services.session_store import init_neo4j, close_neo4j

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev B's own initialization...
    
    # Initialize Dev A's Neo4j connection
    await init_neo4j(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    
    yield
    
    # Cleanup Dev A's Neo4j connection
    await close_neo4j()
```

### 2.5 Configuration (Dev A's env vars Dev B must set)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (for LLM) |
| `OPENAI_MODEL` | No | `gpt-4o` | Which model to use |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `NEO4J_TIMEOUT_MS` | No | `2000` | Per-query Neo4j timeout |
| `MAX_TURNS` | No | `30` | Max turns per session |
| `MAX_MOOD_DELTA` | No | `15` | Max mood change per turn |
| `LOG_LEVEL` | No | `INFO` | Log verbosity |

---

## 3. How Dev B Uses Our Function — Conceptual Code

This is what Dev B's endpoint handler should look like (from integration_response.md):

```python
# Dev B's main.py — /api/interact endpoint

from app.generate import generate_vendor_response
from app.services.ai_brain import BrainServiceError
from app.services.session_store import StateStoreError

@app.post("/api/interact")
async def interact(request: InteractRequest) -> InteractResponse:

    # Step 1 — Parse (Pydantic already validated)
    memory = get_memory(request.session_id)

    # Step 2 — Decode audio
    audio_bytes = base64_to_bytes(request.audio_base64)

    # Step 3 — Speech-to-Text
    transcribed_text = await transcribe_with_sarvam(audio_bytes, request.language_code)

    if transcribed_text == "":
        return build_silence_response(request.session_id)

    # Step 4 — Store user turn
    memory.add_turn("user", transcribed_text, extract_metadata(request.scene_context))

    # Steps 5 & 6 — Parallel: context history + RAG
    context_block, rag_context = await asyncio.gather(
        asyncio.to_thread(memory.get_context_block),
        retrieve_context(transcribed_text, n_results=3)
    )

    # Step 7 + 7½ — Call Dev A's brain (LLM + Neo4j validation)
    try:
        result = await generate_vendor_response(
            transcribed_text=transcribed_text,
            context_block=context_block,
            rag_context=rag_context,
            scene_context=request.scene_context,
            session_id=request.session_id
        )
    except BrainServiceError:
        return JSONResponse(status_code=500, content={"error": "AI processing error"})
    except StateStoreError:
        return JSONResponse(status_code=503, content={"error": "Game state unavailable"})

    # Step 8 — Store vendor turn
    memory.add_turn("vendor", result["reply_text"], {
        "vendor_happiness": result["vendor_happiness"],
        "vendor_patience": result["vendor_patience"],
        "stage": result["new_stage"],
        "price": result["price_offered"]
    })

    # Step 9 — Text-to-Speech
    audio_bytes = await speak_with_sarvam(result["reply_text"], request.language_code)

    # Step 10 — Encode audio
    audio_base64 = bytes_to_base64(audio_bytes)

    # Step 11 — Return response
    return InteractResponse(
        session_id=request.session_id,
        transcribed_text=transcribed_text,
        agent_reply_text=result["reply_text"],
        agent_audio_base64=audio_base64,
        vendor_mood=result.get("vendor_mood", "neutral"),
        negotiation_state=build_negotiation_state(result)
    )
```

---

## 4. Data Ownership Summary

| Data | Created By | Consumed By | Format |
|------|-----------|-------------|--------|
| `audio_base64` (input) | Unity | Dev B (Step 2) | Base64 string |
| `transcribed_text` | Dev B (Step 3) | Dev A (via arg) | Native script string |
| `scene_context` | Unity | Dev A (via arg from Dev B) | Dict (see §2.2) |
| `context_block` | Dev B (Step 5) | Dev A (via arg) | Multi-line text string |
| `rag_context` | Dev B (Step 6) | Dev A (via arg) | Multi-line text string |
| `reply_text` | Dev A (Step 7) | Dev B (Steps 8, 9, 11) | Native script string |
| `new_mood / stage / price` | Dev A (Step 7, validated 7½) | Dev B (Steps 8, 11) | Dict fields |
| `vendor_happiness / patience / mood` | Dev A (Step 7) | Dev B (Steps 8, 11) | Dict fields |
| `audio_base64` (output) | Dev B (Step 10) | Unity (Step 11) | Base64 string |

---

## 5. Error Flow — What Happens When Things Break

```
Step 1 fails (bad request body)  → Dev B returns 422 to Unity (Pydantic validation)
Step 2 fails (bad base64)        → Dev B returns 400 to Unity immediately
Step 3 fails (Sarvam STT down)   → Dev B returns 503 to Unity
Step 3 returns "" (silence)      → Dev B skips Steps 4-7, vendor says "Kuch bola?"
Step 6 fails (ChromaDB down)     → Dev B sets rag_context="" and calls Dev A (graceful)
Step 7 fails (LLM error)         → Dev A raises BrainServiceError → Dev B returns 500
Step 7 fails (Neo4j down)        → Dev A raises StateStoreError → Dev B returns 503
Step 9 fails (Sarvam TTS down)   → Dev B sends text-only response (audio_base64="")
No SARVAM_API_KEY set            → Dev B auto-mocks STT/TTS (Dev A unaffected)
No OPENAI_API_KEY set            → Dev A fails at startup with clear error
```

---

## 6. File Structure — Where Each Dev's Code Lives

```
SamVadXR-Orchestration/
│
├── app/                          ◄── DEV A's domain
│   ├── main.py                   # FastAPI dev server (testing only)
│   ├── generate.py               # ★ PRIMARY INTERFACE — generate_vendor_response()
│   ├── models/                   # Pydantic models (SceneContext, VendorResponse, AIDecision)
│   │   ├── enums.py              # NegotiationStage, VendorMood, LanguageCode
│   │   ├── request.py            # SceneContext model
│   │   └── response.py           # VendorResponse model
│   ├── services/
│   │   ├── ai_brain.py           # GPT-4o prompt composition + parsing
│   │   ├── state_engine.py       # State machine validation (Neo4j-backed)
│   │   ├── session_store.py      # Neo4j session state (init_neo4j, close_neo4j, load, save)
│   │   └── mocks.py              # Mock OpenAI + Neo4j for isolated testing
│   ├── prompts/                  # System prompt templates
│   └── config.py                 # Env vars, settings
│
├── services/                     ◄── DEV B's domain
│   ├── voice_ops.py              # transcribe_with_sarvam, speak_with_sarvam
│   ├── rag_ops.py                # retrieve_context
│   ├── context_memory.py         # ConversationMemory class
│   ├── middleware.py             # base64_to_bytes, bytes_to_base64
│   └── exceptions.py            # SarvamServiceError, RAGServiceError
│
├── tests/
│   ├── test_models.py            ◄── Dev A
│   ├── test_ai_brain.py          ◄── Dev A
│   ├── test_state_engine.py      ◄── Dev A
│   ├── test_api.py               ◄── Dev A (dev endpoint tests)
│   └── test_integration.py       ◄── Both (full pipeline test)
│
├── requirements.txt
└── .env.example
```

---

## 7. Answered Questions (from v2.0 Integration Guide)

These questions were open in v2.0. Per integration_response.md (v3.0), they are now resolved:

| # | Question | Answer (v3.0) |
|---|----------|---------------|
| 1 | Are STT/TTS async? | Yes — `async def` (Dev B's concern now) |
| 2 | Is `retrieve_context` async? | Yes — `async def` (Dev B's concern now) |
| 3 | What does STT return on silence? | Empty string `""` → Dev B handles (skips Steps 4-7) |
| 4 | What audio format does TTS return? | WAV 16-bit PCM, 22kHz, mono (Dev B's concern) |
| 5 | What exception for STT failure? | `SarvamServiceError` (Dev B catches this themselves) |
| 6 | What exception for TTS failure? | `SarvamServiceError` (Dev B sends text-only fallback) |
| 7 | What exception for RAG failure? | `RAGServiceError` (Dev B sets rag_context="" and continues) |
| 8 | Is `ConversationMemory` instance or singleton? | Instance per session (Dev B manages lifecycle) |
| 9 | Language code format? | `hi-IN` style (Sarvam format) confirmed |
| 10 | `get_context_block()` format? | Summary + recent dialogue as plain text string |
| 11 | What does Dev A need from Dev B? | Nothing at runtime — Dev B calls us, not the other way around |
| 12 | How does integration work? | Dev B imports `generate_vendor_response()` and calls it at Step 7 |

---

## 8. Timeline & Integration Points

```
Week 1:
  Dev A: Build generate_vendor_response() + AI brain + state engine (with mocks)
  Dev B: Build voice_ops.py (STT/TTS) + endpoint + memory
  
  ✅ Checkpoint: Dev A shares function contract (this document).
     Dev B verifies they can produce the right arguments.

Week 2:
  Dev A: Finish prompt tuning + Neo4j persistence + state machine testing
  Dev B: Build rag_ops.py + context_memory.py + full pipeline
  
  🤝 Integration Point: Dev B imports generate_vendor_response(),
     calls it from their endpoint. We test together with real data.

Week 2-3:
  Together: End-to-end testing, latency profiling, edge case handling.
  Goal: Full loop under 4 seconds, all error cases produce graceful responses.
```

---

*Last updated: 2026-02-13 — Developer A*
*Architecture v3.0: Dev B owns the endpoint, Dev A provides generate_vendor_response()*
