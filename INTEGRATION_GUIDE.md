# Samvad XR — Integration Guide for Developer B

> **What this document is:** A clear contract between us. It tells you exactly what I need from your modules, what I'll do with them, and how our code connects at runtime.
>
> **TL;DR:** You build the ears (STT), the mouth (TTS), the cultural memory (RAG), and the conversation memory. I build the brain (LLM agent), the game rules (state engine), and the API that Unity talks to. I import your modules and call your functions in a specific order.

---

## 1. The Big Picture — Who Owns What

### Complete Request Lifecycle (Finalized)

```
Step  Owner   Module              Action                                         Time
────  ──────  ──────────────────  ─────────────────────────────────────────────   ────
 1    Dev A   main.py             Receive POST /api/interact, parse request        0ms
 2    Dev B   middleware.py        base64_to_bytes(request.audio_base64)           1ms
 3    Dev B   voice_ops.py         await transcribe_with_sarvam(bytes, "hi-IN")  ~800ms
                                   → "भाई ये silk scarf कितने का है?"
 4    Dev B   context_memory.py    memory.add_turn("user", text, metadata)         1ms
 5    Dev B   context_memory.py    context_block = memory.get_context_block()       1ms
 6    Dev B   rag_ops.py           rag_ctx = await retrieve_context(text, 3)      ~50ms
 7    Dev A   ai_brain.py          Compose prompt (context_block + rag_ctx         ~2s
                                     + Neo4j state) → LLM → parse JSON
 7½   Dev A   state_engine.py      Validate via Neo4j: clamp mood ±15,            ~20ms
                                     verify stage transition is legal
 8    Dev B   context_memory.py    memory.add_turn("vendor", reply, metadata)       1ms
 9    Dev B   voice_ops.py         audio = await speak_with_sarvam(reply, "hi-IN") ~600ms
10    Dev B   middleware.py         b64 = bytes_to_base64(audio)                     1ms
11    Dev A   main.py              Return InteractResponse to Unity                 0ms
                                                                    TOTAL ≈ 3.5s
```

**Your modules power 8 of the 11 steps.** But I'm the one calling them, in this exact sequence, inside my endpoint function.

> **Note on Neo4j (Steps 7 & 7½):** I use Neo4j as the persistent state store for session game state (mood, stage, turn count, price history). This is entirely my domain — you never interact with Neo4j directly. Your conversation memory (`context_memory.py`) handles dialogue history; my Neo4j graph handles game logic state.

---

## 2. Your Modules — What I Need From Each

### 2.1 `middleware.py` — Encoding Utilities

These are simple, synchronous helper functions.

```
┌──────────────────────────────────────────────────────┐
│  Function: base64_to_bytes(b64_string: str) -> bytes │
│                                                      │
│  Input:  "SGVsbG8gV29ybGQ="  (base64 string)        │
│  Output: b"Hello World"       (raw bytes)            │
│                                                      │
│  Called at: Step 2 (before STT)                      │
│  Error case: If input is invalid base64, raise       │
│              ValueError with a clear message         │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  Function: bytes_to_base64(audio_bytes: bytes) -> str│
│                                                      │
│  Input:  b"\x00\x01\x02..."  (raw audio bytes)      │
│  Output: "AAEC..."            (base64 string)        │
│                                                      │
│  Called at: Step 10 (after TTS)                      │
│  Error case: Should never fail on valid bytes        │
└──────────────────────────────────────────────────────┘
```

### 2.2 `voice_ops.py` — Sarvam STT & TTS

These are the slowest steps in the pipeline. They MUST be `async`.

```
┌────────────────────────────────────────────────────────────────┐
│  Function: transcribe_with_sarvam(                             │
│      audio_bytes: bytes,                                       │
│      language_code: str        # "hi-IN", "kn-IN", "ta-IN",   │
│  ) -> str                        "en-IN", "hi-EN"             │
│                                                                │
│  Input:  Raw audio bytes from the VR headset mic               │
│  Output: Transcribed text as a string                          │
│          e.g. "भाई ये silk scarf कितने का है?"                    │
│                                                                │
│  Called at: Step 3                                              │
│  Latency budget: ~800ms (your estimate)                        │
│  My timeout: 5 seconds                                         │
│                                                                │
│  ⚠️  QUESTIONS I NEED ANSWERED:                                │
│  1. What do you return if audio is silence/noise?              │
│     → Empty string ""? Or do you raise an exception?           │
│  2. What exception type do you raise on Sarvam API failure?    │
│     → I need to catch it specifically in my error handler      │
│  3. Does this function handle retries internally, or should I? │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Function: speak_with_sarvam(                                  │
│      text: str,                                                │
│      language_code: str                                        │
│  ) -> bytes                                                    │
│                                                                │
│  Input:  Vendor's reply text                                   │
│          e.g. "अरे भाई, ये pure silk है! ₹800 का है"             │
│  Output: Audio bytes (WAV or MP3 — which format?)              │
│                                                                │
│  Called at: Step 9                                              │
│  Latency budget: ~600ms (your estimate)                        │
│  My timeout: 5 seconds                                         │
│                                                                │
│  ⚠️  QUESTIONS I NEED ANSWERED:                                │
│  1. What audio format/encoding? WAV 16-bit PCM? MP3?          │
│     → Unity needs to know what to decode on the other end      │
│  2. What sample rate? 16kHz? 22kHz? 44.1kHz?                  │
│  3. What happens if Sarvam TTS is down?                        │
│     → I'll fall back to sending text-only (no audio),          │
│       but I need to know what exception to catch               │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 `context_memory.py` — Conversation Memory

This tracks the full back-and-forth history within a session.

```
┌────────────────────────────────────────────────────────────────┐
│  Class: ConversationMemory                                     │
│                                                                │
│  Method: add_turn(                                             │
│      role: str,           # "user" or "vendor"                 │
│      text: str,           # what was said                      │
│      metadata: dict       # extra info (see below)             │
│  ) -> None                                                     │
│                                                                │
│  I call this TWICE per request:                                │
│                                                                │
│  Call 1 (Step 4) — after STT:                                  │
│    memory.add_turn("user", "भाई ये silk scarf कितने का है?", { │
│        "held_item": "silk_scarf",                              │
│        "looked_at_item": "brass_statue",                       │
│        "mood": 55,                                             │
│        "stage": "BROWSING"                                     │
│    })                                                          │
│                                                                │
│  Call 2 (Step 8) — after AI decides:                           │
│    memory.add_turn("vendor", "अरे भाई, ये pure silk है!...", { │
│        "mood": 60,                                             │
│        "stage": "HAGGLING",                                    │
│        "price": 700                                            │
│    })                                                          │
│                                                                │
│  ⚠️  QUESTION: Do you need specific metadata keys, or is it   │
│  an arbitrary dict you store as-is?                            │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  Method: get_context_block() -> str                            │
│                                                                │
│  Called at: Step 5 (before I build the LLM prompt)             │
│                                                                │
│  What I expect back: A formatted string of recent conversation │
│  history that I can inject directly into the LLM prompt.       │
│                                                                │
│  Example output:                                               │
│  """                                                           │
│  [Turn 1] User: Namaste bhaiya, kya haal hai?                 │
│  [Turn 1] Vendor: Aao aao! Kya chahiye aapko?                 │
│  [Turn 2] User: भाई ये silk scarf कितने का है?                  │
│  """                                                           │
│                                                                │
│  ⚠️  QUESTIONS:                                                │
│  1. How many turns does this include? Last 5? Last 10? All?    │
│     → I'd suggest last 10 turns max to control token usage     │
│  2. Does the format include metadata (mood, price) or just     │
│     the spoken text?                                           │
│     → I prefer text-only in the context block. I'll inject     │
│       current mood/stage separately from scene_context.        │
│  3. Can I configure the window size (number of turns)?         │
└────────────────────────────────────────────────────────────────┘
```

**Important — Memory Instance Lifecycle:**

I will create and manage the `ConversationMemory` instance per session on my side:

```
# In my orchestration code (conceptual)
sessions = {}   # session_id → ConversationMemory

def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in sessions:
        sessions[session_id] = ConversationMemory()
    return sessions[session_id]
```

This means your `ConversationMemory` class should:
- Be instantiable with no required arguments (or with a `session_id` if you need it)
- Store state in the instance (not in a global/singleton)
- Be safe to create many instances (one per active VR session)

Let me know if you had a different design in mind (e.g., a singleton with internal session routing).

### 2.4 `rag_ops.py` — ChromaDB Retrieval

```
┌────────────────────────────────────────────────────────────────┐
│  Function: retrieve_context(                                   │
│      query: str,           # the user's transcribed text       │
│      n_results: int = 3    # how many chunks to return         │
│  ) -> str                                                      │
│                                                                │
│  Called at: Step 6                                              │
│  Latency budget: ~50ms (your estimate)                         │
│  My timeout: 3 seconds                                         │
│                                                                │
│  What I expect back: A single string with the relevant         │
│  cultural/item knowledge, ready to inject into the LLM prompt. │
│                                                                │
│  Example output:                                               │
│  """                                                           │
│  - Silk scarves from Varanasi are known for Banarasi weave     │
│  - Typical retail price range: ₹500-₹1500                     │
│  - Vendors usually start 2x above their minimum price          │
│  """                                                           │
│                                                                │
│  ⚠️  QUESTIONS:                                                │
│  1. Is the return type a single concatenated string, or a      │
│     list of strings? → I'd prefer a single string I can        │
│     drop directly into the prompt.                             │
│  2. What do you return when there are no relevant results?     │
│     → Empty string ""? Or "No context available"?              │
│  3. Is ChromaDB running in-process or as a separate service?   │
│     → Affects deployment setup                                 │
│  4. Is this function async or sync?                            │
│     → If sync, I'll wrap it in asyncio.to_thread()             │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. The Critical Question: Async or Sync?

My FastAPI server runs on an async event loop. If your functions make blocking I/O calls (HTTP requests to Sarvam, ChromaDB queries), they **must** be either:

| Your Function | Makes Network Call? | Must Be Async? |
|--------------|-------------------|----------------|
| `base64_to_bytes` | No | No (pure computation) |
| `bytes_to_base64` | No | No (pure computation) |
| `transcribe_with_sarvam` | Yes (Sarvam API) | **Yes** |
| `speak_with_sarvam` | Yes (Sarvam API) | **Yes** |
| `retrieve_context` | Yes (ChromaDB) | **Yes, or I'll wrap it** |
| `memory.add_turn` | No (in-memory) | No |
| `memory.get_context_block` | No (in-memory) | No |

**Preferred:** Use `httpx.AsyncClient` or `aiohttp` for your Sarvam API calls instead of `requests`. If you're using `requests` (synchronous), let me know — I'll wrap your calls in `asyncio.to_thread()`, but it's less efficient.

---

## 4. Language Code Format — Let's Standardize

Your pipeline uses Sarvam's format. Let's go with yours:

| Language | Code We'll Both Use |
|----------|-------------------|
| Hindi | `hi-IN` |
| English | `en-IN` |
| Hinglish | `hi-EN` |
| Kannada | `kn-IN` |
| Tamil | `ta-IN` |

I'll update my Pydantic enums to match. This is now the single source of truth.

---

## 5. Error Contract — What Should Happen When Things Break

I need to know what exceptions your functions raise so I can handle them properly. Here's what I propose — please confirm or correct:

| Scenario | Your Function | What You Should Do | What I'll Do |
|----------|--------------|-------------------|-------------|
| Sarvam STT API is down | `transcribe_with_sarvam` | Raise a specific exception (e.g., `SarvamServiceError`) | Return 503 to Unity: "Voice recognition unavailable" |
| Audio is silence/noise | `transcribe_with_sarvam` | Return empty string `""` | Vendor says "Kuch bola aapne?" (Did you say something?) |
| STT returns garbage | `transcribe_with_sarvam` | Return whatever Sarvam returns (your best effort) | My AI brain will handle garbled input gracefully |
| Sarvam TTS API is down | `speak_with_sarvam` | Raise `SarvamServiceError` | Return text-only response (empty audio, subtitle only) |
| ChromaDB has no results | `retrieve_context` | Return empty string `""` | AI brain proceeds without cultural context (still works, just less grounded) |
| ChromaDB is unreachable | `retrieve_context` | Raise `RAGServiceError` | I skip RAG, continue without it (graceful degradation) |

**My ask:** Define two exception classes I can import:
```python
class SarvamServiceError(Exception): ...
class RAGServiceError(Exception): ...
```

Or tell me what you're already using and I'll catch those.

---

## 6. What I Do Between Your Steps (The Invisible Work)

Between steps 6 and 9, there's a lot happening on my side that's invisible to you but critical to the product:

### Step 7 — The AI Brain (~2s)

I take **everything your functions produced**, combine it with **game state from Neo4j**, and compose a prompt for GPT-4o:

```
╔══════════════════════════════════════════════════════════════╗
║  SYSTEM PROMPT (written by me — the "God Prompt")           ║
║  • Vendor persona (Ramesh, 55, from Jaipur)                 ║
║  • Behavioral rules based on mood ranges                    ║
║  • State transition rules (GREETING→BROWSING→HAGGLING→...)  ║
║  • Strict JSON output schema                                ║
╠══════════════════════════════════════════════════════════════╣
║  CONVERSATION HISTORY ← from your get_context_block()       ║
║  [Turn 1] User: Namaste bhaiya!                             ║
║  [Turn 1] Vendor: Aao! Kya chahiye?                         ║
╠══════════════════════════════════════════════════════════════╣
║  CULTURAL CONTEXT ← from your retrieve_context()            ║
║  • Silk scarves retail ₹500-₹1500                           ║
║  • Vendors start at 2x minimum                              ║
╠══════════════════════════════════════════════════════════════╣
║  GAME STATE ← from Neo4j (my domain, not yours)             ║
║  • current_mood: 55                                          ║
║  • current_stage: "BROWSING"                                 ║
║  • turn_count: 3                                             ║
║  • price_history: [1500, 1200]                               ║
╠══════════════════════════════════════════════════════════════╣
║  SCENE CONTEXT ← from Unity request metadata                ║
║  • held_item: "silk_scarf"                                   ║
║  • looked_at_item: "brass_statue"                            ║
╠══════════════════════════════════════════════════════════════╣
║  USER MESSAGE ← from your transcribe_with_sarvam()          ║
║  "भाई ये silk scarf कितने का है?"                               ║
╚══════════════════════════════════════════════════════════════╝
                            │
                            ▼
                    GPT-4o responds with:
            {
              "reply_text": "अरे भाई, ये pure silk है!...",
              "new_mood": 60,
              "new_stage": "HAGGLING",
              "price_offered": 700,
              "internal_reasoning": "User is directly asking..."
            }
```

### Step 7½ — State Validation via Neo4j (~20ms)

I validate the AI's output against the state graph in Neo4j:
- Clamp mood to 0–100, max ±15 change per turn
- Verify the stage transition is legal (e.g., can't jump from GREETING to DEAL)
- If the AI hallucinates an illegal state, I override it and keep the current state
- Write the validated new state back to Neo4j

This is fully my responsibility. You never interact with Neo4j.

---

## 7. What I'm Building While You're Building

So we're not blocked on each other, here's what I'm doing in parallel:

| My Task | Why You Don't Need to Wait |
|---------|---------------------------|
| Mocking all your functions | I have fake versions of `transcribe_with_sarvam`, `speak_with_sarvam`, `retrieve_context`, and `ConversationMemory` that return hardcoded data. I can test my full pipeline without your code. |
| Writing the GPT-4o system prompt | No dependency on you. Pure prompt engineering. |
| Building the state machine + Neo4j graph | No dependency on you. Pure game logic. Neo4j stores session state (mood, stage, turn count, price history). |
| Defining Pydantic models | I'll share the OpenAPI schema with you so you know exactly what the request/response looks like. |

**When you're ready**, I swap the mocks for your real implementations via a config toggle (`USE_MOCKS=true/false`). Zero code changes in my orchestration logic.

---

## 8. File Structure — Where Your Code Lives

```
SamVadXR-Orchestration/
│
├── app/                          ◄── MY domain
│   ├── main.py                   # API endpoint, orchestration
│   ├── models/                   # Pydantic request/response models
│   ├── services/
│   │   ├── ai_brain.py           # GPT-4o prompt + parsing
│   │   ├── state_engine.py       # State machine validation (Neo4j-backed)
│   │   ├── session_store.py      # Neo4j session state read/write
│   │   └── mocks.py              # Mock versions of YOUR functions
│   ├── prompts/                  # System prompt templates
│   └── config.py                 # Env vars, feature flags
│
├── services/                     ◄── YOUR domain
│   ├── voice_ops.py              # transcribe_with_sarvam, speak_with_sarvam
│   ├── rag_ops.py                # retrieve_context
│   ├── context_memory.py         # ConversationMemory class
│   ├── middleware.py             # base64_to_bytes, bytes_to_base64
│   └── exceptions.py            # SarvamServiceError, RAGServiceError
│
├── tests/
│   ├── test_voice_ops.py         ◄── You write these
│   ├── test_rag_ops.py           ◄── You write these
│   ├── test_api.py               ◄── I write these (uses mocks or real)
│   └── test_integration.py       ◄── We write together
│
├── requirements.txt
└── .env.example
```

You can develop your `services/` folder independently. I import from it.

---

## 9. The Handshake Checklist

Before we integrate, let's confirm these decisions. Reply with your answers:

| # | Decision | Options | Your Answer |
|---|----------|---------|-------------|
| 1 | Are `transcribe_with_sarvam` and `speak_with_sarvam` async? | `async def` / `def` | ? |
| 2 | Is `retrieve_context` async? | `async def` / `def` | ? |
| 3 | What HTTP client do you use for Sarvam? | `httpx` / `aiohttp` / `requests` | ? |
| 4 | What does STT return on silence? | `""` / raises exception | ? |
| 5 | What audio format does TTS return? | WAV PCM / MP3 / OGG | ? |
| 6 | What sample rate for TTS audio? | 16kHz / 22kHz / 44.1kHz | ? |
| 7 | Does `retrieve_context` return `str` or `list[str]`? | `str` / `list[str]` | ? |
| 8 | Is `ConversationMemory` instance-based or singleton? | Instance per session / Singleton | ? |
| 9 | What exceptions do you raise for service failures? | Custom class name(s) | ? |
| 10 | Language code format confirmed? | `hi-IN` style | ? |
| 11 | `get_context_block()` — how many turns included? | Last N turns (what N?) | ? |
| 12 | Is ChromaDB in-process or a separate service? | In-process / External | ? |

---

## 10. Timeline & Integration Points

```
Week 1:
  You: Build voice_ops.py (STT/TTS with Sarvam)
  Me:  Build orchestration + AI brain + state engine (all mocked)
  
  ✅ Checkpoint: I send you a mock request/response JSON pair
     so you can verify your functions produce compatible shapes.

Week 2:
  You: Build rag_ops.py + context_memory.py
  Me:  Finish prompt tuning + state machine testing
  
  🤝 Integration Point: I import your modules, toggle USE_MOCKS=false
     We test the full pipeline together with a real audio clip.

Week 2-3:
  Together: End-to-end testing, latency profiling, edge case handling.
  Goal: Full loop under 4 seconds, all error cases produce graceful responses.
```

---

## Quick Reference — Function Signatures I'm Coding Against

```python
# middleware.py
def base64_to_bytes(b64_string: str) -> bytes: ...
def bytes_to_base64(audio_bytes: bytes) -> str: ...

# voice_ops.py
async def transcribe_with_sarvam(audio_bytes: bytes, language_code: str) -> str: ...
async def speak_with_sarvam(text: str, language_code: str) -> bytes: ...

# rag_ops.py
async def retrieve_context(query: str, n_results: int = 3) -> str: ...

# context_memory.py
class ConversationMemory:
    def add_turn(self, role: str, text: str, metadata: dict) -> None: ...
    def get_context_block(self) -> str: ...

# exceptions.py
class SarvamServiceError(Exception): ...
class RAGServiceError(Exception): ...
```

**These are the interfaces I'm mocking now and will swap for your real implementations later. If any signature doesn't work for you, let's discuss before either of us writes too much code.**

---

*Last updated: 2026-02-13 — Developer A*
