"""
Neo4j session state persistence — the ONLY module that imports neo4j.

All Cypher queries live here. No other module touches the driver directly.

Provides:
    - init_neo4j(uri, user, password)  → initialise the async driver
    - close_neo4j()                    → shut down the driver
    - Neo4jSessionStore                → SessionStore protocol impl

Node schema:
    (:Session {
        session_id, vendor_happiness, vendor_patience,
        negotiation_stage, turn_count, current_price, price_history,
        created_at, updated_at
    })
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.exceptions import StateStoreError

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
    "vendor_happiness": 50,
    "vendor_patience": 70,
    "negotiation_stage": "GREETING",
    "current_price": 0,
    "turn_count": 0,
    "price_history": [],
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
            s.vendor_happiness = $vendor_happiness,
            s.vendor_patience  = $vendor_patience,
            s.negotiation_stage = $negotiation_stage,
            s.current_price    = $current_price,
            s.turn_count       = $turn_count,
            s.price_history    = $price_history,
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
        SET s.vendor_happiness  = $vendor_happiness,
            s.vendor_patience   = $vendor_patience,
            s.negotiation_stage = $negotiation_stage,
            s.current_price     = $current_price,
            s.turn_count        = $turn_count,
            s.price_history     = $price_history,
            s.updated_at        = $now
        """
        params = {
            "session_id": session_id,
            "vendor_happiness": state.get("vendor_happiness", 50),
            "vendor_patience": state.get("vendor_patience", 70),
            "negotiation_stage": state.get("negotiation_stage", "GREETING"),
            "current_price": state.get("current_price", 0),
            "turn_count": state.get("turn_count", 0),
            "price_history": state.get("price_history", []),
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
        """Delete a session node. Returns True if deleted, False if not found."""
        driver = get_driver()

        query = """
        MATCH (s:Session {session_id: $session_id})
        DELETE s
        RETURN count(s) AS deleted
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
        """Delete ALL Session nodes. Returns the count deleted.

        **For testing only.** Use to reset the database to a clean state.
        """
        driver = get_driver()

        try:
            async with driver.session(database="neo4j") as session:
                result = await session.run(
                    "MATCH (s:Session) WITH s LIMIT 10000 "
                    "DELETE s RETURN count(*) AS total"
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

    # ── Internal helpers ──────────────────────────────────

    @staticmethod
    def _node_to_dict(node: Any, session_id: str) -> dict[str, Any]:
        """Convert a Neo4j node to a plain dict matching the protocol."""
        return {
            "session_id": session_id,
            "vendor_happiness": node.get("vendor_happiness", 50),
            "vendor_patience": node.get("vendor_patience", 70),
            "negotiation_stage": node.get("negotiation_stage", "GREETING"),
            "current_price": node.get("current_price", 0),
            "turn_count": node.get("turn_count", 0),
            "price_history": list(node.get("price_history", [])),
            "created_at": node.get("created_at"),
            "updated_at": node.get("updated_at"),
        }

