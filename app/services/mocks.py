"""
Mock implementations of Dev A's own dependencies (LLM + Session Store).

These conform to the protocols in app.services.protocols and are activated
when USE_MOCKS=true in config. Dev B's mocks (STT/TTS/RAG) are their concern.

Mock behaviour:
    MockLLMService    — deterministic responses based on input keywords,
                        ~200ms simulated latency.
    MockSessionStore  — in-memory dict keyed by session_id, no Neo4j needed.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.models.enums import NegotiationStage, VendorMood
from app.models.response import AIDecision

logger = logging.getLogger("samvadxr")


# ═══════════════════════════════════════════════════════════
#  Mock LLM Service
# ═══════════════════════════════════════════════════════════


class MockLLMService:
    """Deterministic LLM mock — returns canned AIDecision based on keywords.

    Keyword routing (checked in order):
        - "namaste" / "hello"     → GREETING response
        - "kitne" / "price" / "cost" → INQUIRY response
        - "nahi" / "no" / "chhodo" → WALKAWAY response
        - "theek" / "deal" / "done" → DEAL response
        - (empty string)          → vendor prompts the user
        - (default)               → neutral GREETING response

    Simulates ~200ms latency via asyncio.sleep.
    """

    async def generate_decision(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> AIDecision:
        """Return a deterministic AIDecision based on keyword matching."""
        # Simulate LLM latency
        await asyncio.sleep(0.2)

        text_lower = user_message.lower()

        # When called via generate_vendor_response(), the user_message is a
        # composed prompt with delimited sections like:
        #   --- USER MESSAGE ---
        #   <speech>
        #   --- END USER MESSAGE ---
        # Extract only the user's speech for keyword matching to avoid false
        # positives from context/scene fields.
        if "--- user message ---" in text_lower:
            after_marker = text_lower.split("--- user message ---", 1)[1]
            speech = after_marker.split("--- end user message ---", 1)[0].strip()
        elif "user says:" in text_lower:
            speech = text_lower.split("user says:", 1)[1].split("\n", 1)[0].strip()
        else:
            speech = text_lower

        logger.debug(
            "MockLLMService: generating decision",
            extra={"step": "mock_llm", "user_message_snippet": speech[:80]},
        )

        # ── Keyword routing ───────────────────────────────
        if any(kw in speech for kw in ("namaste", "hello", "namaskar")):
            return AIDecision(
                reply_text="Welcome, welcome! Come, come, see what all I have for you!",
                happiness_score=55,
                negotiation_state=NegotiationStage.GREETING,
                vendor_mood=VendorMood.FRIENDLY,
                internal_reasoning="[MOCK] User greeted → greeting response",
                counter_price=None,
                offer_assessment="none",
                suggested_user_response="Can you show me what you have?",
            )

        if any(kw in speech for kw in ("kitne", "price", "cost", "kidhar", "कितने")):
            return AIDecision(
                reply_text="Oh brother, this is the freshest you will find! 60 rupees per kilo, special price just for you!",
                happiness_score=65,
                negotiation_state=NegotiationStage.INQUIRY,
                vendor_mood=VendorMood.FRIENDLY,
                internal_reasoning="[MOCK] User asked price → inquiry response",
                counter_price=60,
                offer_assessment="none",
                suggested_user_response="That seems a bit high. How about 40 rupees?",
            )

        if any(kw in speech for kw in ("nahi", "no", "chhodo", "chalo", "bahut")):
            return AIDecision(
                reply_text="Wait, wait! Don't say that, just listen a little more!",
                happiness_score=35,
                negotiation_state=NegotiationStage.WALKAWAY,
                vendor_mood=VendorMood.ANNOYED,
                internal_reasoning="[MOCK] User rejecting → walkaway response",
                counter_price=None,
                offer_assessment="none",
                suggested_user_response="Okay, what is your best price then?",
            )

        if any(kw in speech for kw in ("theek", "deal", "done", "pakka", "le lo")):
            return AIDecision(
                reply_text="Wonderful! Deal is done! You are a very good customer!",
                happiness_score=85,
                negotiation_state=NegotiationStage.DEAL,
                vendor_mood=VendorMood.ENTHUSIASTIC,
                internal_reasoning="[MOCK] User agreed → deal response",
                counter_price=55,
                offer_assessment="excellent",
                suggested_user_response="Thank you! Please pack it up.",
            )

        if not speech.strip():
            return AIDecision(
                reply_text="Did you say something? Come here brother, let me show you!",
                happiness_score=50,
                negotiation_state=NegotiationStage.GREETING,
                vendor_mood=VendorMood.NEUTRAL,
                internal_reasoning="[MOCK] Empty input → vendor prompts user",
                counter_price=None,
                offer_assessment="none",
                suggested_user_response="Hello! I am looking to buy something.",
            )

        # Default: neutral greeting response
        return AIDecision(
            reply_text="Yes, yes, very good choice! Want to see anything else?",
            happiness_score=50,
            negotiation_state=NegotiationStage.GREETING,
            vendor_mood=VendorMood.NEUTRAL,
            internal_reasoning="[MOCK] Default → greeting response",
            counter_price=None,
            offer_assessment="none",
            suggested_user_response="How much does this cost?",
        )


# ═══════════════════════════════════════════════════════════
#  Mock Session Store
# ═══════════════════════════════════════════════════════════

# Default initial state for a new session
_DEFAULT_SESSION_STATE: dict[str, Any] = {
    "happiness_score": 50,
    "negotiation_state": NegotiationStage.GREETING.value,
    "turn_count": 0,
}


class MockSessionStore:
    """In-memory session store — no Neo4j needed.

    Stores game state in a plain dict keyed by session_id.
    Also maintains in-memory graph structures (turns, items,
    stage transitions) to mirror Neo4jSessionStore's graph behavior.
    Thread-safe for single-process async usage (no locks needed
    since asyncio is single-threaded within the event loop).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._turns: dict[str, list[dict[str, Any]]] = {}
        self._stage_transitions: dict[str, list[dict[str, Any]]] = {}
        self._items: dict[str, dict[str, dict[str, Any]]] = {}  # session_id -> {item_name -> info}

    async def create_session(self, session_id: str) -> dict[str, Any]:
        """Create a new session with default initial state."""
        state = {"session_id": session_id, **_DEFAULT_SESSION_STATE}
        self._sessions[session_id] = state
        logger.info(
            "MockSessionStore: session created",
            extra={"step": "mock_store_create", "session_id": session_id},
        )
        return dict(state)  # return a copy

    async def load_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Load session state, or return None if not found."""
        state = self._sessions.get(session_id)
        if state is None:
            logger.debug(
                "MockSessionStore: session not found",
                extra={"step": "mock_store_load", "session_id": session_id},
            )
            return None
        logger.debug(
            "MockSessionStore: session loaded",
            extra={
                "step": "mock_store_load",
                "session_id": session_id,
                "turn_count": state.get("turn_count", 0),
            },
        )
        return dict(state)  # return a copy

    async def save_session(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist updated session state (in-memory)."""
        self._sessions[session_id] = dict(state)
        logger.debug(
            "MockSessionStore: session saved",
            extra={
                "step": "mock_store_save",
                "session_id": session_id,
                "turn_count": state.get("turn_count", 0),
            },
        )

    async def record_turn(
        self,
        session_id: str,
        turn_number: int,
        role: str,
        text_snippet: str,
        happiness_score: int,
        stage: str,
        object_grabbed: Optional[str] = None,
    ) -> None:
        """Record a conversation turn in the in-memory graph."""
        if session_id not in self._turns:
            self._turns[session_id] = []

        snippet = (text_snippet[:150] + "...") if len(text_snippet) > 150 else text_snippet

        self._turns[session_id].append({
            "turn_number": turn_number,
            "role": role,
            "text_snippet": snippet,
            "happiness_score": happiness_score,
            "stage": stage,
            "object_grabbed": object_grabbed or "",
            "timestamp": "mock",
        })

        # Track item interactions
        if object_grabbed:
            if session_id not in self._items:
                self._items[session_id] = {}
            item_info = self._items[session_id].get(object_grabbed, {
                "item_name": object_grabbed,
                "first_mentioned": turn_number,
                "last_mentioned": turn_number,
                "mention_count": 0,
            })
            item_info["last_mentioned"] = turn_number
            item_info["mention_count"] = item_info.get("mention_count", 0) + 1
            self._items[session_id][object_grabbed] = item_info

        logger.debug(
            "MockSessionStore: turn recorded",
            extra={
                "step": "mock_store_turn",
                "session_id": session_id,
                "turn_number": turn_number,
            },
        )

    async def record_stage_transition(
        self,
        session_id: str,
        from_stage: str,
        to_stage: str,
        turn_number: int,
        happiness_score: int,
    ) -> None:
        """Record a stage transition in the in-memory graph."""
        if session_id not in self._stage_transitions:
            self._stage_transitions[session_id] = []

        self._stage_transitions[session_id].append({
            "from_stage": from_stage,
            "to_stage": to_stage,
            "at_turn": turn_number,
            "happiness_at_transition": happiness_score,
        })

        logger.debug(
            "MockSessionStore: stage transition recorded",
            extra={
                "step": "mock_store_transition",
                "session_id": session_id,
                "from": from_stage,
                "to": to_stage,
            },
        )

    async def get_graph_context(self, session_id: str) -> dict[str, Any]:
        """Return structured graph context from in-memory data."""
        turns = self._turns.get(session_id, [])
        transitions = self._stage_transitions.get(session_id, [])
        items = list(self._items.get(session_id, {}).values())

        return {
            "turns": turns,
            "stage_transitions": transitions,
            "items_discussed": items,
        }

    # ── Test helpers (not part of protocol) ───────────────

    def clear(self) -> None:
        """Clear all sessions and graph data. For testing only."""
        self._sessions.clear()
        self._turns.clear()
        self._stage_transitions.clear()
        self._items.clear()

    @property
    def session_count(self) -> int:
        """Number of active sessions. For testing only."""
        return len(self._sessions)
