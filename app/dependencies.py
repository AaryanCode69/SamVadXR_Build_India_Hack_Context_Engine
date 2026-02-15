"""
Dependency injection — service wiring based on USE_MOCKS config toggle.

USE_MOCKS=true  → MockLLMService + MockSessionStore  (no external deps)
USE_MOCKS=false → RealLLMService + Neo4jSessionStore  (Phase 4-5 provide these)

All service access goes through get_llm_service() and get_session_store().
No module should instantiate services directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.config import get_settings
from app.services.mocks import MockLLMService, MockSessionStore
from app.services.protocols import LLMService, SessionStore

# Lazy import to avoid loading OpenAI SDK when mocks are active
# from app.services.ai_brain import OpenAILLMService  # imported in get_llm_service()

logger = logging.getLogger("samvadxr")

# ── Module-level singletons (initialized on first access) ──
_llm_service: Optional[LLMService] = None
_session_store: Optional[SessionStore] = None


def get_llm_service() -> LLMService:
    """Return the configured LLM service (mock or real).

    On first call, instantiates based on USE_MOCKS setting.
    Subsequent calls return the cached singleton.
    """
    global _llm_service
    if _llm_service is None:
        settings = get_settings()
        if settings.use_mocks:
            logger.info(
                "DI: Using MockLLMService",
                extra={"step": "dependency_init"},
            )
            _llm_service = MockLLMService()
        else:
            from app.services.ai_brain import OpenAILLMService

            logger.info(
                "DI: Using OpenAILLMService (real OpenAI)",
                extra={"step": "dependency_init"},
            )
            _llm_service = OpenAILLMService()
    return _llm_service


def get_session_store() -> SessionStore:
    """Return the configured session store (mock or real Neo4j).

    On first call, instantiates based on USE_MOCKS setting.
    Subsequent calls return the cached singleton.
    """
    global _session_store
    if _session_store is None:
        settings = get_settings()
        if settings.use_mocks:
            logger.info(
                "DI: Using MockSessionStore",
                extra={"step": "dependency_init"},
            )
            _session_store = MockSessionStore()
        else:
            # Phase 5 will provide Neo4jSessionStore here.
            raise NotImplementedError(
                "Neo4jSessionStore not implemented yet (Phase 5). "
                "Set USE_MOCKS=true for development."
            )
    return _session_store


def reset_services() -> None:
    """Reset service singletons. For testing only — forces re-initialization."""
    global _llm_service, _session_store
    _llm_service = None
    _session_store = None


def override_llm_service(service: LLMService) -> None:
    """Override the LLM service singleton. For testing only."""
    global _llm_service
    _llm_service = service


def override_session_store(store: SessionStore) -> None:
    """Override the session store singleton. For testing only."""
    global _session_store
    _session_store = store
