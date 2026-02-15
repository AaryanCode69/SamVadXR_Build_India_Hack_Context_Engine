"""
Service protocols (interfaces) for Dev A's dependencies.

Both mock and real implementations conform to these protocols.
This enables swapping via USE_MOCKS config toggle without code changes.

Protocols defined:
    LLMService    — AI Brain (OpenAI or mock)
    SessionStore  — Neo4j state persistence (or in-memory mock)
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from app.models.response import AIDecision


# ═══════════════════════════════════════════════════════════
#  LLM Service Protocol
# ═══════════════════════════════════════════════════════════


@runtime_checkable
class LLMService(Protocol):
    """Protocol for the AI brain — calls an LLM and returns a structured decision.

    Implementations:
        - RealLLMService  (Phase 4) — calls OpenAI GPT-4o
        - MockLLMService  (Phase 2) — returns deterministic responses
    """

    async def generate_decision(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> AIDecision:
        """Send a prompt to the LLM and return a parsed AIDecision.

        Args:
            system_prompt: The full system prompt (God Prompt + context).
            user_message: The user's turn (transcribed text + scene info).
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Max response tokens.

        Returns:
            Parsed AIDecision from the LLM's JSON output.

        Raises:
            BrainServiceError: If LLM call fails after retries.
        """
        ...


# ═══════════════════════════════════════════════════════════
#  Session Store Protocol
# ═══════════════════════════════════════════════════════════


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for game state persistence (Neo4j or in-memory mock).

    Stores per-session state: mood, stage, turn count, price history.
    This is SEPARATE from Dev B's conversation memory (dialogue text).

    Implementations:
        - Neo4jSessionStore  (Phase 5) — real Neo4j Cypher queries
        - MockSessionStore   (Phase 2) — in-memory dict
    """

    async def create_session(self, session_id: str) -> dict[str, Any]:
        """Create a new session with default initial state.

        Args:
            session_id: Unique session identifier.

        Returns:
            Dict with initial game state:
            {
                "session_id": str,
                "vendor_happiness": 50,
                "vendor_patience": 70,
                "negotiation_stage": "GREETING",
                "current_price": 0,
                "turn_count": 0,
                "price_history": [],
            }

        Raises:
            StateStoreError: If store is unreachable.
        """
        ...

    async def load_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Load existing session state.

        Args:
            session_id: Unique session identifier.

        Returns:
            Session state dict if found, None if session doesn't exist.

        Raises:
            StateStoreError: If store is unreachable.
        """
        ...

    async def save_session(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist updated session state.

        Args:
            session_id: Unique session identifier.
            state: Full session state dict to persist.

        Raises:
            StateStoreError: If store is unreachable.
        """
        ...
