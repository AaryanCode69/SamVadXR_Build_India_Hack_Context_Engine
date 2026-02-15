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
        - "kitne" / "price" / "cost" → HAGGLING response with price
        - "nahi" / "no" / "chhodo" → WALKAWAY response
        - "theek" / "deal" / "done" → DEAL response
        - (empty string)          → vendor prompts the user
        - (default)               → neutral BROWSING response

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
        # composed prompt like "User says: <speech>\nContext: ...\nScene: ...".
        # Extract only the user's speech for keyword matching to avoid false
        # positives from context/scene fields (e.g. "price=0" triggering HAGGLING).
        if "user says:" in text_lower:
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
                reply_text="Namaste ji! Aao aao, dekho kya kya hai humare paas!",
                new_mood=55,
                new_stage=NegotiationStage.GREETING,
                price_offered=None,
                vendor_happiness=55,
                vendor_patience=70,
                vendor_mood=VendorMood.ENTHUSIASTIC,
                internal_reasoning="[MOCK] User greeted → greeting response",
            )

        if any(kw in speech for kw in ("kitne", "price", "cost", "kidhar", "कितने")):
            return AIDecision(
                reply_text="Arey bhai, ye pure Banarasi silk hai! ₹800 lagega, lekin aapke liye special price!",
                new_mood=60,
                new_stage=NegotiationStage.HAGGLING,
                price_offered=800,
                vendor_happiness=60,
                vendor_patience=65,
                vendor_mood=VendorMood.ENTHUSIASTIC,
                internal_reasoning="[MOCK] User asked price → haggling response",
            )

        if any(kw in speech for kw in ("nahi", "no", "chhodo", "chalo", "bahut")):
            return AIDecision(
                reply_text="Arey ruko ruko! Itna bhi nahi bolna tha, thoda aur suno!",
                new_mood=35,
                new_stage=NegotiationStage.WALKAWAY,
                price_offered=None,
                vendor_happiness=35,
                vendor_patience=40,
                vendor_mood=VendorMood.ANNOYED,
                internal_reasoning="[MOCK] User rejecting → walkaway response",
            )

        if any(kw in speech for kw in ("theek", "deal", "done", "pakka", "le lo")):
            return AIDecision(
                reply_text="Bahut badhiya! Deal pakki! Aap bahut acche customer ho!",
                new_mood=85,
                new_stage=NegotiationStage.DEAL,
                price_offered=500,
                vendor_happiness=85,
                vendor_patience=90,
                vendor_mood=VendorMood.ENTHUSIASTIC,
                internal_reasoning="[MOCK] User agreed → deal response",
            )

        if not speech.strip():
            return AIDecision(
                reply_text="Kuch bola kya? Arey bhai, yaha aao, dikhata hoon!",
                new_mood=50,
                new_stage=NegotiationStage.BROWSING,
                price_offered=None,
                vendor_happiness=50,
                vendor_patience=60,
                vendor_mood=VendorMood.NEUTRAL,
                internal_reasoning="[MOCK] Empty input → vendor prompts user",
            )

        # Default: neutral browsing response
        return AIDecision(
            reply_text="Haan ji, bahut accha choice hai! Aur kuch dekhna hai?",
            new_mood=50,
            new_stage=NegotiationStage.BROWSING,
            price_offered=None,
            vendor_happiness=50,
            vendor_patience=70,
            vendor_mood=VendorMood.NEUTRAL,
            internal_reasoning="[MOCK] Default → browsing response",
        )


# ═══════════════════════════════════════════════════════════
#  Mock Session Store
# ═══════════════════════════════════════════════════════════

# Default initial state for a new session
_DEFAULT_SESSION_STATE: dict[str, Any] = {
    "vendor_happiness": 50,
    "vendor_patience": 70,
    "negotiation_stage": NegotiationStage.GREETING.value,
    "current_price": 0,
    "turn_count": 0,
    "price_history": [],
}


class MockSessionStore:
    """In-memory session store — no Neo4j needed.

    Stores game state in a plain dict keyed by session_id.
    Thread-safe for single-process async usage (no locks needed
    since asyncio is single-threaded within the event loop).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def create_session(self, session_id: str) -> dict[str, Any]:
        """Create a new session with default initial state."""
        state = {"session_id": session_id, **_DEFAULT_SESSION_STATE}
        # Deep-copy price_history so sessions don't share the same list
        state["price_history"] = []
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

    # ── Test helpers (not part of protocol) ───────────────

    def clear(self) -> None:
        """Clear all sessions. For testing only."""
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        """Number of active sessions. For testing only."""
        return len(self._sessions)
