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
    Also maintains a session graph: turns, items, stage transitions.
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
                "happiness_score": 50,
                "negotiation_state": "GREETING",
                "turn_count": 0,
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
        """Record a conversation turn as a node in the session graph.

        Creates a turn node linked to the session and chained to
        the previous turn. If object_grabbed is set, links an Item node.

        Args:
            session_id: Unique session identifier.
            turn_number: Sequential turn number (1-based).
            role: "user" or "vendor".
            text_snippet: Truncated text of what was said.
            happiness_score: Happiness score at this turn.
            stage: Negotiation stage at this turn.
            object_grabbed: Item the user is interacting with (if any).
        """
        ...

    async def record_stage_transition(
        self,
        session_id: str,
        from_stage: str,
        to_stage: str,
        turn_number: int,
        happiness_score: int,
    ) -> None:
        """Record a stage transition in the session graph.

        Args:
            session_id: Unique session identifier.
            from_stage: Stage being left.
            to_stage: Stage being entered.
            turn_number: Turn at which the transition occurred.
            happiness_score: Happiness at the moment of transition.
        """
        ...

    async def get_graph_context(self, session_id: str) -> dict[str, Any]:
        """Traverse the session graph and return structured context.

        Returns:
            Dict with keys:
            - "turns": list of turn dicts (ordered by turn_number)
            - "stage_transitions": list of transition dicts
            - "items_discussed": list of item dicts with mention info
        """
        ...
