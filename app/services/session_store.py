"""
Neo4j session state persistence — the ONLY module that imports neo4j.

All Cypher queries live here. No other module touches the driver directly.

Provides:
    - init_neo4j(uri, user, password)  → initialise the async driver
    - close_neo4j()                    → shut down the driver
    - Neo4jSessionStore                → SessionStore protocol impl

Graph schema:
    (:Session {session_id, happiness_score, negotiation_state, turn_count,
               created_at, updated_at})
    (:Turn {session_id, turn_number, role, text_snippet, happiness_score,
            stage, object_grabbed, timestamp})
    (:Item {name, session_id})
    (:StageTransition {session_id, from_stage, to_stage, at_turn,
                       happiness_at_transition, timestamp})

    (:Session)-[:HAS_TURN]->(:Turn)
    (:Turn)-[:FOLLOWED_BY]->(:Turn)
    (:Turn)-[:ABOUT_ITEM]->(:Item)
    (:Session)-[:INVOLVES_ITEM]->(:Item)
    (:Session)-[:STAGE_CHANGED]->(:StageTransition)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from ..exceptions import StateStoreError

logger = logging.getLogger("samvadxr")

# ── Module-level driver singleton ─────────────────────────
_driver: Optional[AsyncDriver] = None


# ═══════════════════════════════════════════════════════════
#  Driver lifecycle (called by Dev B's lifespan or our main.py)
# ═══════════════════════════════════════════════════════════


async def init_neo4j(
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "",
    timeout_ms: int = 2000,
) -> None:
    """Initialise the Neo4j async driver and verify connectivity.

    Must be called once at app startup — either by our FastAPI lifespan
    or by Dev B's lifespan hook.

    Raises:
        StateStoreError: If the driver cannot connect.
    """
    global _driver

    if _driver is not None:
        logger.warning(
            "init_neo4j called but driver already exists — closing old driver",
            extra={"step": "neo4j_init"},
        )
        await close_neo4j()

    try:
        _driver = AsyncGraphDatabase.driver(
            uri,
            auth=(user, password),
            connection_acquisition_timeout=timeout_ms / 1000.0,
            max_connection_lifetime=300,  # 5 min
        )
        await _driver.verify_connectivity()
        logger.info(
            "Neo4j driver initialised and connected",
            extra={"step": "neo4j_init", "uri": uri},
        )
    except Exception as exc:
        _driver = None
        logger.error(
            "Neo4j connection failed",
            extra={"step": "neo4j_init", "error": str(exc)},
        )
        raise StateStoreError(f"Cannot connect to Neo4j: {exc}") from exc


async def close_neo4j() -> None:
    """Shut down the Neo4j driver gracefully."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None
        logger.info("Neo4j driver closed", extra={"step": "neo4j_shutdown"})


def get_driver() -> AsyncDriver:
    """Return the active Neo4j driver, or raise if not initialised."""
    if _driver is None:
        raise StateStoreError(
            "Neo4j driver not initialised. Call init_neo4j() at startup."
        )
    return _driver


# ═══════════════════════════════════════════════════════════
#  Default session state
# ═══════════════════════════════════════════════════════════

_DEFAULT_STATE: dict[str, Any] = {
    "happiness_score": 50,
    "negotiation_state": "GREETING",
    "turn_count": 0,
}


# ═══════════════════════════════════════════════════════════
#  Neo4jSessionStore — conforms to SessionStore protocol
# ═══════════════════════════════════════════════════════════


class Neo4jSessionStore:
    """Real Neo4j-backed session store.

    Every method acquires a session from the driver, runs a Cypher query,
    and returns a plain dict matching the SessionStore protocol.
    """

    # ── Create ────────────────────────────────────────────

    async def create_session(self, session_id: str) -> dict[str, Any]:
        """Create a new Session node with default initial state.

        If a session with the same ID already exists, it is returned
        unchanged (idempotent).
        """
        driver = get_driver()
        now = datetime.now(timezone.utc).isoformat()

        query = """
        MERGE (s:Session {session_id: $session_id})
        ON CREATE SET
            s.happiness_score   = $happiness_score,
            s.negotiation_state = $negotiation_state,
            s.turn_count       = $turn_count,
            s.created_at       = $now,
            s.updated_at       = $now
        RETURN s
        """
        params = {
            "session_id": session_id,
            **_DEFAULT_STATE,
            "now": now,
        }

        try:
            async with driver.session(database="neo4j") as session:
                result = await session.run(query, params)
                record = await result.single()
                if record is None:
                    raise StateStoreError(
                        f"Failed to create session {session_id}"
                    )
                node = record["s"]
                state = self._node_to_dict(node, session_id)

                logger.info(
                    "Neo4j session created",
                    extra={
                        "step": "neo4j_create",
                        "session_id": session_id,
                    },
                )
                return state
        except StateStoreError:
            raise
        except Exception as exc:
            logger.error(
                "Neo4j create_session failed",
                extra={
                    "step": "neo4j_create",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to create session: {exc}"
            ) from exc

    # ── Load ──────────────────────────────────────────────

    async def load_session(self, session_id: str) -> Optional[dict[str, Any]]:
        """Load session state from Neo4j. Returns None if not found."""
        driver = get_driver()

        query = """
        MATCH (s:Session {session_id: $session_id})
        RETURN s
        """

        try:
            async with driver.session(database="neo4j") as session:
                result = await session.run(query, {"session_id": session_id})
                record = await result.single()
                if record is None:
                    logger.debug(
                        "Neo4j session not found",
                        extra={
                            "step": "neo4j_load",
                            "session_id": session_id,
                        },
                    )
                    return None
                node = record["s"]
                state = self._node_to_dict(node, session_id)

                logger.debug(
                    "Neo4j session loaded",
                    extra={
                        "step": "neo4j_load",
                        "session_id": session_id,
                        "turn_count": state.get("turn_count", 0),
                    },
                )
                return state
        except StateStoreError:
            raise
        except Exception as exc:
            logger.error(
                "Neo4j load_session failed",
                extra={
                    "step": "neo4j_load",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to load session: {exc}"
            ) from exc

    # ── Save ──────────────────────────────────────────────

    async def save_session(
        self, session_id: str, state: dict[str, Any]
    ) -> None:
        """Persist updated session state to Neo4j.

        Creates the node if it doesn't exist (upsert).
        """
        driver = get_driver()
        now = datetime.now(timezone.utc).isoformat()

        query = """
        MERGE (s:Session {session_id: $session_id})
        SET s.happiness_score   = $happiness_score,
            s.negotiation_state = $negotiation_state,
            s.turn_count        = $turn_count,
            s.updated_at        = $now
        """
        params = {
            "session_id": session_id,
            "happiness_score": state.get("happiness_score", 50),
            "negotiation_state": state.get("negotiation_state", "GREETING"),
            "turn_count": state.get("turn_count", 0),
            "now": now,
        }

        try:
            async with driver.session(database="neo4j") as session:
                await session.run(query, params)

                logger.debug(
                    "Neo4j session saved",
                    extra={
                        "step": "neo4j_save",
                        "session_id": session_id,
                        "turn_count": params["turn_count"],
                    },
                )
        except StateStoreError:
            raise
        except Exception as exc:
            logger.error(
                "Neo4j save_session failed",
                extra={
                    "step": "neo4j_save",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to save session: {exc}"
            ) from exc

    # ── Delete (for testing / cleanup) ────────────────────

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session node and all its graph children. Returns True if deleted."""
        driver = get_driver()

        query = """
        MATCH (s:Session {session_id: $session_id})
        OPTIONAL MATCH (s)-[r1]->(t:Turn)
        OPTIONAL MATCH (s)-[r2]->(i:Item)
        OPTIONAL MATCH (s)-[r3]->(st:StageTransition)
        OPTIONAL MATCH (t)-[r4]->()
        DETACH DELETE s, t, i, st
        RETURN count(DISTINCT s) AS deleted
        """

        try:
            async with driver.session(database="neo4j") as session:
                result = await session.run(query, {"session_id": session_id})
                record = await result.single()
                deleted = record["deleted"] > 0 if record else False

                logger.info(
                    "Neo4j session deleted",
                    extra={
                        "step": "neo4j_delete",
                        "session_id": session_id,
                        "deleted": deleted,
                    },
                )
                return deleted
        except Exception as exc:
            logger.error(
                "Neo4j delete_session failed",
                extra={
                    "step": "neo4j_delete",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to delete session: {exc}"
            ) from exc

    # ── Delete all sessions (for testing cleanup) ─────────

    async def delete_all_sessions(self) -> int:
        """Delete ALL Session, Turn, Item, StageTransition nodes.

        **For testing only.** Use to reset the database to a clean state.
        """
        driver = get_driver()

        try:
            async with driver.session(database="neo4j") as session:
                # Delete graph children first, then sessions
                await session.run(
                    "MATCH (n) WHERE n:Turn OR n:Item OR n:StageTransition "
                    "DETACH DELETE n"
                )
                result = await session.run(
                    "MATCH (s:Session) WITH s LIMIT 10000 "
                    "DETACH DELETE s RETURN count(*) AS total"
                )
                record = await result.single()
                total = record["total"] if record else 0

                logger.info(
                    "Neo4j all sessions deleted",
                    extra={"step": "neo4j_delete_all", "deleted": total},
                )
                return total
        except Exception as exc:
            logger.error(
                "Neo4j delete_all_sessions failed",
                extra={"step": "neo4j_delete_all", "error": str(exc)},
            )
            raise StateStoreError(
                f"Failed to delete all sessions: {exc}"
            ) from exc

    # ═════════════════════════════════════════════════════
    #  Graph Context — Turn recording & traversal
    # ═════════════════════════════════════════════════════

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
        """Record a conversation turn as a graph node linked to the session.

        Creates a (:Turn) node, links it to the (:Session) via [:HAS_TURN],
        and chains it to the previous turn via [:FOLLOWED_BY].
        If object_grabbed is provided, creates/links an (:Item) node.
        """
        driver = get_driver()
        now = datetime.now(timezone.utc).isoformat()
        # Truncate text_snippet to keep graph nodes lightweight
        snippet = (text_snippet[:150] + "...") if len(text_snippet) > 150 else text_snippet

        query = """
        MATCH (s:Session {session_id: $session_id})
        CREATE (t:Turn {
            session_id: $session_id,
            turn_number: $turn_number,
            role: $role,
            text_snippet: $text_snippet,
            happiness_score: $happiness_score,
            stage: $stage,
            object_grabbed: $object_grabbed,
            timestamp: $now
        })
        CREATE (s)-[:HAS_TURN]->(t)
        WITH s, t
        OPTIONAL MATCH (s)-[:HAS_TURN]->(prev:Turn)
        WHERE prev.turn_number = $prev_turn AND prev.session_id = $session_id
        FOREACH (_ IN CASE WHEN prev IS NOT NULL THEN [1] ELSE [] END |
            CREATE (prev)-[:FOLLOWED_BY]->(t)
        )
        RETURN t
        """
        params = {
            "session_id": session_id,
            "turn_number": turn_number,
            "role": role,
            "text_snippet": snippet,
            "happiness_score": happiness_score,
            "stage": stage,
            "object_grabbed": object_grabbed or "",
            "now": now,
            "prev_turn": turn_number - 1,
        }

        try:
            async with driver.session(database="neo4j") as session:
                await session.run(query, params)

            # If an item was grabbed, record the item interaction
            if object_grabbed:
                await self._record_item_interaction(
                    session_id, turn_number, object_grabbed
                )

            logger.debug(
                "Neo4j turn recorded",
                extra={
                    "step": "neo4j_record_turn",
                    "session_id": session_id,
                    "turn_number": turn_number,
                    "role": role,
                },
            )
        except StateStoreError:
            raise
        except Exception as exc:
            logger.error(
                "Neo4j record_turn failed",
                extra={
                    "step": "neo4j_record_turn",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to record turn: {exc}"
            ) from exc

    async def record_stage_transition(
        self,
        session_id: str,
        from_stage: str,
        to_stage: str,
        turn_number: int,
        happiness_score: int,
    ) -> None:
        """Record a stage transition as a graph node linked to the session.

        Creates a (:StageTransition) node connected to the (:Session) via
        [:STAGE_CHANGED]. This provides a clear history of how the
        negotiation has progressed through stages.
        """
        driver = get_driver()
        now = datetime.now(timezone.utc).isoformat()

        query = """
        MATCH (s:Session {session_id: $session_id})
        CREATE (st:StageTransition {
            session_id: $session_id,
            from_stage: $from_stage,
            to_stage: $to_stage,
            at_turn: $turn_number,
            happiness_at_transition: $happiness_score,
            timestamp: $now
        })
        CREATE (s)-[:STAGE_CHANGED]->(st)
        RETURN st
        """
        params = {
            "session_id": session_id,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "turn_number": turn_number,
            "happiness_score": happiness_score,
            "now": now,
        }

        try:
            async with driver.session(database="neo4j") as session:
                await session.run(query, params)

            logger.info(
                "Neo4j stage transition recorded",
                extra={
                    "step": "neo4j_stage_transition",
                    "session_id": session_id,
                    "from": from_stage,
                    "to": to_stage,
                    "at_turn": turn_number,
                },
            )
        except StateStoreError:
            raise
        except Exception as exc:
            logger.error(
                "Neo4j record_stage_transition failed",
                extra={
                    "step": "neo4j_stage_transition",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to record stage transition: {exc}"
            ) from exc

    async def get_graph_context(self, session_id: str) -> dict[str, Any]:
        """Traverse the session graph and return structured context.

        Queries the graph for:
            - All turns (ordered by turn_number)
            - All stage transitions (ordered by at_turn)
            - All items involved

        Returns a dict with raw graph data that can be formatted into
        a context string for the LLM prompt.
        """
        driver = get_driver()

        # ── Query 1: Get all turns ────────────────────────
        turns_query = """
        MATCH (s:Session {session_id: $session_id})-[:HAS_TURN]->(t:Turn)
        RETURN t.turn_number AS turn_number,
               t.role AS role,
               t.text_snippet AS text_snippet,
               t.happiness_score AS happiness_score,
               t.stage AS stage,
               t.object_grabbed AS object_grabbed,
               t.timestamp AS timestamp
        ORDER BY t.turn_number
        """

        # ── Query 2: Get stage transitions ────────────────
        transitions_query = """
        MATCH (s:Session {session_id: $session_id})-[:STAGE_CHANGED]->(st:StageTransition)
        RETURN st.from_stage AS from_stage,
               st.to_stage AS to_stage,
               st.at_turn AS at_turn,
               st.happiness_at_transition AS happiness_at_transition
        ORDER BY st.at_turn
        """

        # ── Query 3: Get items involved ───────────────────
        items_query = """
        MATCH (s:Session {session_id: $session_id})-[:INVOLVES_ITEM]->(i:Item)
        OPTIONAL MATCH (t:Turn)-[:ABOUT_ITEM]->(i)
        WHERE t.session_id = $session_id
        WITH i.name AS item_name,
             min(t.turn_number) AS first_mentioned,
             max(t.turn_number) AS last_mentioned,
             count(t) AS mention_count
        RETURN item_name, first_mentioned, last_mentioned, mention_count
        ORDER BY first_mentioned
        """

        try:
            turns: list[dict[str, Any]] = []
            transitions: list[dict[str, Any]] = []
            items: list[dict[str, Any]] = []

            async with driver.session(database="neo4j") as session:
                # Turns
                result = await session.run(turns_query, {"session_id": session_id})
                records = await result.data()
                turns = [dict(r) for r in records]

                # Transitions
                result = await session.run(transitions_query, {"session_id": session_id})
                records = await result.data()
                transitions = [dict(r) for r in records]

                # Items
                result = await session.run(items_query, {"session_id": session_id})
                records = await result.data()
                items = [dict(r) for r in records]

            logger.debug(
                "Neo4j graph context retrieved",
                extra={
                    "step": "neo4j_graph_context",
                    "session_id": session_id,
                    "turn_count": len(turns),
                    "transitions": len(transitions),
                    "items": len(items),
                },
            )

            return {
                "turns": turns,
                "stage_transitions": transitions,
                "items_discussed": items,
            }

        except StateStoreError:
            raise
        except Exception as exc:
            logger.error(
                "Neo4j get_graph_context failed",
                extra={
                    "step": "neo4j_graph_context",
                    "session_id": session_id,
                    "error": str(exc),
                },
            )
            raise StateStoreError(
                f"Failed to get graph context: {exc}"
            ) from exc

    # ── Private: item interaction recording ───────────────

    async def _record_item_interaction(
        self,
        session_id: str,
        turn_number: int,
        item_name: str,
    ) -> None:
        """Create or link an Item node for this turn and session."""
        driver = get_driver()

        query = """
        MATCH (s:Session {session_id: $session_id})
        MATCH (s)-[:HAS_TURN]->(t:Turn {turn_number: $turn_number, session_id: $session_id})
        MERGE (i:Item {name: $item_name, session_id: $session_id})
        MERGE (t)-[:ABOUT_ITEM]->(i)
        MERGE (s)-[:INVOLVES_ITEM]->(i)
        RETURN i
        """
        params = {
            "session_id": session_id,
            "turn_number": turn_number,
            "item_name": item_name,
        }

        try:
            async with driver.session(database="neo4j") as session:
                await session.run(query, params)
        except Exception as exc:
            # Item linking is non-critical — log but don't raise
            logger.warning(
                "Neo4j item interaction recording failed (non-critical)",
                extra={
                    "step": "neo4j_item_link",
                    "session_id": session_id,
                    "item": item_name,
                    "error": str(exc),
                },
            )

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _node_to_dict(node: Any, session_id: str) -> dict[str, Any]:
        """Convert a Neo4j node to a plain dict matching the protocol."""
        return {
            "session_id": session_id,
            "happiness_score": node.get("happiness_score", 50),
            "negotiation_state": node.get("negotiation_state", "GREETING"),
            "turn_count": node.get("turn_count", 0),
            "created_at": node.get("created_at"),
            "updated_at": node.get("updated_at"),
        }

