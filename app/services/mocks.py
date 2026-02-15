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
                reply_text="Namaste ji! Aao aao, dekho kya kya hai humare paas!",
                happiness_score=55,
                negotiation_state=NegotiationStage.GREETING,
                vendor_mood=VendorMood.FRIENDLY,
                internal_reasoning="[MOCK] User greeted → greeting response",
            )

        if any(kw in speech for kw in ("kitne", "price", "cost", "kidhar", "कितने")):
            return AIDecision(
                reply_text="Arey bhai, ye toh sabse fresh hai! ₹60 kilo, special price!",
                happiness_score=65,
                negotiation_state=NegotiationStage.INQUIRY,
                vendor_mood=VendorMood.FRIENDLY,
                internal_reasoning="[MOCK] User asked price → inquiry response",
            )

        if any(kw in speech for kw in ("nahi", "no", "chhodo", "chalo", "bahut")):
            return AIDecision(
                reply_text="Arey ruko ruko! Itna bhi nahi bolna tha, thoda aur suno!",
                happiness_score=35,
                negotiation_state=NegotiationStage.WALKAWAY,
                vendor_mood=VendorMood.ANNOYED,
                internal_reasoning="[MOCK] User rejecting → walkaway response",
            )

        if any(kw in speech for kw in ("theek", "deal", "done", "pakka", "le lo")):
            return AIDecision(
                reply_text="Bahut badhiya! Deal pakki! Aap bahut acche customer ho!",
                happiness_score=85,
                negotiation_state=NegotiationStage.DEAL,
                vendor_mood=VendorMood.ENTHUSIASTIC,
                internal_reasoning="[MOCK] User agreed → deal response",
            )

        if not speech.strip():
            return AIDecision(
                reply_text="Kuch bola kya? Arey bhai, yaha aao, dikhata hoon!",
                happiness_score=50,
                negotiation_state=NegotiationStage.GREETING,
                vendor_mood=VendorMood.NEUTRAL,
                internal_reasoning="[MOCK] Empty input → vendor prompts user",
            )

        # Default: neutral greeting response
        return AIDecision(
            reply_text="Haan ji, bahut accha choice hai! Aur kuch dekhna hai?",
            happiness_score=50,
            negotiation_state=NegotiationStage.GREETING,
            vendor_mood=VendorMood.NEUTRAL,
            internal_reasoning="[MOCK] Default → greeting response",
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
    Thread-safe for single-process async usage (no locks needed
    since asyncio is single-threaded within the event loop).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

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

    # ── Test helpers (not part of protocol) ───────────────

    def clear(self) -> None:
        """Clear all sessions. For testing only."""
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        """Number of active sessions. For testing only."""
        return len(self._sessions)
