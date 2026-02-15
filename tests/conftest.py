"""
Shared test fixtures for Samvad XR test suite.

Provides common mock service setup and environment overrides.
"""

from __future__ import annotations

import os
from typing import Generator

import pytest

from app.dependencies import override_llm_service, override_session_store, reset_services
from app.services.mocks import MockLLMService, MockSessionStore


@pytest.fixture(autouse=True)
def _mock_env(request, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set USE_MOCKS=true and provide a dummy OPENAI_API_KEY for all tests.

    Skipped for tests marked with ``neo4j_integration`` so that real
    database integration tests can use their own environment variables.
    """
    if request.node.get_closest_marker("neo4j_integration"):
        yield
        return
    monkeypatch.setenv("USE_MOCKS", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    yield
    reset_services()


@pytest.fixture()
def mock_llm() -> MockLLMService:
    """Provide a fresh MockLLMService and wire it into DI."""
    service = MockLLMService()
    override_llm_service(service)
    return service


@pytest.fixture()
def mock_store() -> MockSessionStore:
    """Provide a fresh MockSessionStore and wire it into DI."""
    store = MockSessionStore()
    override_session_store(store)
    return store
