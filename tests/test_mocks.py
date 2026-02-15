"""
Phase 2 — Mock Services Layer tests.

Coverage:
    - MockLLMService: all keyword routes + latency simulation
    - MockSessionStore: create/load/save/clear lifecycle
    - Protocol conformance: mocks satisfy LLMService and SessionStore
    - DI wiring: get_llm_service() / get_session_store() toggle
    - End-to-end: generate_vendor_response() runs fully with mocks
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.dependencies import (
    get_llm_service,
    get_session_store,
    override_llm_service,
    override_session_store,
    reset_services,
)
from app.models.enums import NegotiationStage, VendorMood
from app.models.response import AIDecision, VendorResponse
from app.services.mocks import MockLLMService, MockSessionStore
from app.services.protocols import LLMService, SessionStore


# ═══════════════════════════════════════════════════════════
#  1. Protocol Conformance
# ═══════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify mocks satisfy the runtime-checkable protocols."""

    def test_mock_llm_is_llm_service(self) -> None:
        assert isinstance(MockLLMService(), LLMService)

    def test_mock_store_is_session_store(self) -> None:
        assert isinstance(MockSessionStore(), SessionStore)


# ═══════════════════════════════════════════════════════════
#  2. MockLLMService Tests
# ═══════════════════════════════════════════════════════════


class TestMockLLMService:
    """MockLLMService — deterministic keyword-routed responses."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.llm = MockLLMService()

    async def test_greeting_namaste(self) -> None:
        decision = await self.llm.generate_decision("system", "Namaste bhaiya!")
        assert isinstance(decision, AIDecision)
        assert decision.new_stage == NegotiationStage.GREETING
        assert decision.vendor_mood == VendorMood.ENTHUSIASTIC
        assert "Namaste" in decision.reply_text

    async def test_greeting_hello(self) -> None:
        decision = await self.llm.generate_decision("system", "Hello vendor!")
        assert decision.new_stage == NegotiationStage.GREETING

    async def test_price_inquiry_kitne(self) -> None:
        decision = await self.llm.generate_decision("system", "Ye kitne ka hai?")
        assert decision.new_stage == NegotiationStage.HAGGLING
        assert decision.price_offered == 800
        assert decision.vendor_mood == VendorMood.ENTHUSIASTIC

    async def test_price_inquiry_hindi(self) -> None:
        decision = await self.llm.generate_decision("system", "कितने का है?")
        assert decision.new_stage == NegotiationStage.HAGGLING
        assert decision.price_offered is not None

    async def test_price_inquiry_english(self) -> None:
        decision = await self.llm.generate_decision("system", "What's the price?")
        assert decision.new_stage == NegotiationStage.HAGGLING

    async def test_rejection_nahi(self) -> None:
        decision = await self.llm.generate_decision("system", "Nahi bhai, bahut mehnga hai")
        assert decision.new_stage == NegotiationStage.WALKAWAY
        assert decision.vendor_mood == VendorMood.ANNOYED

    async def test_rejection_no(self) -> None:
        decision = await self.llm.generate_decision("system", "No way, too expensive")
        assert decision.new_stage == NegotiationStage.WALKAWAY

    async def test_deal_theek(self) -> None:
        decision = await self.llm.generate_decision("system", "Theek hai, le lo")
        assert decision.new_stage == NegotiationStage.DEAL
        assert decision.vendor_mood == VendorMood.ENTHUSIASTIC
        assert decision.price_offered is not None

    async def test_deal_done(self) -> None:
        decision = await self.llm.generate_decision("system", "Done, deal pakka!")
        assert decision.new_stage == NegotiationStage.DEAL

    async def test_empty_input_prompts_user(self) -> None:
        decision = await self.llm.generate_decision("system", "")
        assert decision.new_stage == NegotiationStage.BROWSING
        assert decision.vendor_mood == VendorMood.NEUTRAL

    async def test_whitespace_input_prompts_user(self) -> None:
        decision = await self.llm.generate_decision("system", "   ")
        assert decision.new_stage == NegotiationStage.BROWSING

    async def test_default_browsing_response(self) -> None:
        decision = await self.llm.generate_decision("system", "Yeh silk acchi hai")
        assert decision.new_stage == NegotiationStage.BROWSING
        assert decision.vendor_mood == VendorMood.NEUTRAL

    async def test_returns_valid_ai_decision(self) -> None:
        """Every response is a valid AIDecision — Pydantic validates."""
        for text in ["namaste", "kitne ka", "nahi", "theek", "", "random text"]:
            decision = await self.llm.generate_decision("system", text)
            assert isinstance(decision, AIDecision)
            assert 0 <= decision.new_mood <= 100
            assert 0 <= decision.vendor_happiness <= 100
            assert 0 <= decision.vendor_patience <= 100

    async def test_simulates_latency(self) -> None:
        """Mock should take ~200ms to simulate real API latency."""
        start = time.monotonic()
        await self.llm.generate_decision("system", "test")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15, f"Expected ≥150ms, got {elapsed*1000:.0f}ms"

    async def test_internal_reasoning_populated(self) -> None:
        decision = await self.llm.generate_decision("system", "Namaste!")
        assert "[MOCK]" in decision.internal_reasoning

    async def test_custom_temperature_accepted(self) -> None:
        """temperature and max_tokens params don't crash the mock."""
        decision = await self.llm.generate_decision(
            "system", "hello", temperature=0.3, max_tokens=100
        )
        assert isinstance(decision, AIDecision)


# ═══════════════════════════════════════════════════════════
#  3. MockSessionStore Tests
# ═══════════════════════════════════════════════════════════


class TestMockSessionStore:
    """MockSessionStore — in-memory dict-based session storage."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.store = MockSessionStore()

    async def test_create_session_returns_defaults(self) -> None:
        state = await self.store.create_session("sess-1")
        assert state["session_id"] == "sess-1"
        assert state["vendor_happiness"] == 50
        assert state["vendor_patience"] == 70
        assert state["negotiation_stage"] == "GREETING"
        assert state["current_price"] == 0
        assert state["turn_count"] == 0
        assert state["price_history"] == []

    async def test_load_nonexistent_returns_none(self) -> None:
        result = await self.store.load_session("no-such-session")
        assert result is None

    async def test_load_returns_created_session(self) -> None:
        await self.store.create_session("sess-2")
        state = await self.store.load_session("sess-2")
        assert state is not None
        assert state["session_id"] == "sess-2"

    async def test_save_and_load_round_trip(self) -> None:
        await self.store.create_session("sess-3")
        updated = {
            "session_id": "sess-3",
            "vendor_happiness": 75,
            "vendor_patience": 60,
            "negotiation_stage": "HAGGLING",
            "current_price": 800,
            "turn_count": 3,
            "price_history": [1000, 900, 800],
        }
        await self.store.save_session("sess-3", updated)
        loaded = await self.store.load_session("sess-3")
        assert loaded is not None
        assert loaded["vendor_happiness"] == 75
        assert loaded["negotiation_stage"] == "HAGGLING"
        assert loaded["price_history"] == [1000, 900, 800]

    async def test_sessions_are_isolated(self) -> None:
        await self.store.create_session("a")
        await self.store.create_session("b")

        state_a = await self.store.load_session("a")
        assert state_a is not None
        state_a["vendor_happiness"] = 99
        await self.store.save_session("a", state_a)

        state_b = await self.store.load_session("b")
        assert state_b is not None
        assert state_b["vendor_happiness"] == 50  # unchanged

    async def test_load_returns_copy(self) -> None:
        """Modifying a loaded state should not affect the store."""
        await self.store.create_session("copy-test")
        loaded = await self.store.load_session("copy-test")
        assert loaded is not None
        loaded["vendor_happiness"] = 999

        reloaded = await self.store.load_session("copy-test")
        assert reloaded is not None
        assert reloaded["vendor_happiness"] == 50  # original untouched

    async def test_create_returns_copy(self) -> None:
        """Modifying the returned state should not affect the store."""
        state = await self.store.create_session("copy-test-2")
        state["turn_count"] = 999

        loaded = await self.store.load_session("copy-test-2")
        assert loaded is not None
        assert loaded["turn_count"] == 0  # original untouched

    async def test_clear(self) -> None:
        await self.store.create_session("x")
        await self.store.create_session("y")
        assert self.store.session_count == 2
        self.store.clear()
        assert self.store.session_count == 0

    async def test_session_count(self) -> None:
        assert self.store.session_count == 0
        await self.store.create_session("1")
        assert self.store.session_count == 1
        await self.store.create_session("2")
        assert self.store.session_count == 2

    async def test_overwrite_existing_session(self) -> None:
        """Creating a session with an existing ID overwrites it."""
        await self.store.create_session("dup")
        state = await self.store.load_session("dup")
        assert state is not None
        state["turn_count"] = 10
        await self.store.save_session("dup", state)

        # Re-create resets to defaults
        fresh = await self.store.create_session("dup")
        assert fresh["turn_count"] == 0


# ═══════════════════════════════════════════════════════════
#  4. Dependency Injection Wiring
# ═══════════════════════════════════════════════════════════


class TestDependencyInjection:
    """DI wiring: USE_MOCKS toggle and override helpers."""

    def setup_method(self) -> None:
        reset_services()

    def test_get_llm_service_returns_mock(self) -> None:
        """With USE_MOCKS=true, get_llm_service() returns MockLLMService."""
        llm = get_llm_service()
        assert isinstance(llm, MockLLMService)

    def test_get_session_store_returns_mock(self) -> None:
        """With USE_MOCKS=true, get_session_store() returns MockSessionStore."""
        store = get_session_store()
        assert isinstance(store, MockSessionStore)

    def test_singleton_returns_same_instance(self) -> None:
        """Repeated calls return the same singleton."""
        llm1 = get_llm_service()
        llm2 = get_llm_service()
        assert llm1 is llm2

        store1 = get_session_store()
        store2 = get_session_store()
        assert store1 is store2

    def test_override_llm_service(self) -> None:
        custom = MockLLMService()
        override_llm_service(custom)
        assert get_llm_service() is custom

    def test_override_session_store(self) -> None:
        custom = MockSessionStore()
        override_session_store(custom)
        assert get_session_store() is custom

    def test_reset_services(self) -> None:
        llm1 = get_llm_service()
        reset_services()
        llm2 = get_llm_service()
        assert llm1 is not llm2  # new instance after reset

    def test_use_mocks_false_creates_real_llm_service(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """USE_MOCKS=false should create OpenAILLMService (Phase 4 implemented)."""
        reset_services()
        monkeypatch.setenv("USE_MOCKS", "false")
        from app.services.ai_brain import OpenAILLMService

        service = get_llm_service()
        assert isinstance(service, OpenAILLMService)

    def test_use_mocks_false_store_creates_neo4j_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """USE_MOCKS=false should create Neo4jSessionStore (Phase 5 implemented)."""
        reset_services()
        monkeypatch.setenv("USE_MOCKS", "false")
        from app.services.session_store import Neo4jSessionStore

        store = get_session_store()
        assert isinstance(store, Neo4jSessionStore)


# ═══════════════════════════════════════════════════════════
#  5. End-to-End: generate_vendor_response with Mocks
# ═══════════════════════════════════════════════════════════


class TestGenerateEndToEnd:
    """Full pipeline: generate_vendor_response() → mock LLM + mock store."""

    @pytest.fixture(autouse=True)
    def _setup(self, mock_llm: MockLLMService, mock_store: MockSessionStore) -> None:
        self.llm = mock_llm
        self.store = mock_store

    def _scene(self, **overrides: object) -> dict:
        """Build a valid scene_context dict with optional overrides."""
        base = {
            "items_in_hand": ["brass_keychain"],
            "looking_at": "silk_scarf",
            "distance_to_vendor": 1.2,
            "vendor_npc_id": "vendor_01",
            "vendor_happiness": 55,
            "vendor_patience": 70,
            "negotiation_stage": "BROWSING",
            "current_price": 0,
            "user_offer": 0,
        }
        base.update(overrides)
        return base

    async def test_greeting_flow(self) -> None:
        from app.generate import generate_vendor_response

        # Pre-populate session with GREETING state (matches what create_session gives us)
        result = await generate_vendor_response(
            transcribed_text="Namaste bhaiya!",
            context_block="",
            rag_context="",
            scene_context=self._scene(negotiation_stage="GREETING"),
            session_id="test-greeting",
        )
        assert isinstance(result, dict)
        # Mock returns GREETING for namaste; same-stage is always legal
        assert result["new_stage"] == "GREETING"
        assert result["vendor_mood"] in ("enthusiastic", "neutral")
        assert len(result["reply_text"]) > 0

    async def test_haggling_flow(self) -> None:
        from app.generate import generate_vendor_response

        # Pre-populate the session at BROWSING so BROWSING → HAGGLING is legal
        await self.store.create_session("test-haggling")
        await self.store.save_session("test-haggling", {
            "vendor_happiness": 55,
            "vendor_patience": 70,
            "negotiation_stage": "BROWSING",
            "current_price": 0,
            "turn_count": 1,
            "price_history": [],
        })

        result = await generate_vendor_response(
            transcribed_text="Ye silk scarf kitne ka hai?",
            context_block="[Turn 1] User: Namaste",
            rag_context="Silk Scarf: Fair price 300-400",
            scene_context=self._scene(),
            session_id="test-haggling",
        )
        assert result["new_stage"] == "HAGGLING"
        assert result["price_offered"] == 800
        assert 0 <= result["new_mood"] <= 100

    async def test_walkaway_flow(self) -> None:
        from app.generate import generate_vendor_response

        # Pre-populate the session at HAGGLING so HAGGLING → WALKAWAY is legal
        await self.store.create_session("test-walkaway")
        await self.store.save_session("test-walkaway", {
            "vendor_happiness": 50,
            "vendor_patience": 60,
            "negotiation_stage": "HAGGLING",
            "current_price": 800,
            "turn_count": 3,
            "price_history": [800],
        })

        result = await generate_vendor_response(
            transcribed_text="Nahi bhai, bahut mehnga hai",
            context_block="",
            rag_context="",
            scene_context=self._scene(negotiation_stage="HAGGLING"),
            session_id="test-walkaway",
        )
        assert result["new_stage"] == "WALKAWAY"
        assert result["vendor_mood"] in ("annoyed", "neutral")

    async def test_deal_flow(self) -> None:
        from app.generate import generate_vendor_response

        # Pre-populate the session at HAGGLING so HAGGLING → DEAL is legal
        await self.store.create_session("test-deal")
        await self.store.save_session("test-deal", {
            "vendor_happiness": 60,
            "vendor_patience": 65,
            "negotiation_stage": "HAGGLING",
            "current_price": 500,
            "turn_count": 5,
            "price_history": [800, 600, 500],
        })

        result = await generate_vendor_response(
            transcribed_text="Theek hai pakka deal!",
            context_block="",
            rag_context="",
            scene_context=self._scene(negotiation_stage="HAGGLING", current_price=500),
            session_id="test-deal",
        )
        assert result["new_stage"] == "DEAL"
        assert result["vendor_mood"] == "enthusiastic"

    async def test_empty_input_flow(self) -> None:
        from app.generate import generate_vendor_response

        result = await generate_vendor_response(
            transcribed_text="",
            context_block="",
            rag_context="",
            scene_context=self._scene(),
            session_id="test-empty",
        )
        assert result["new_stage"] == "BROWSING"
        assert result["vendor_mood"] == "neutral"

    async def test_result_is_valid_vendor_response(self) -> None:
        """Result must be deserializable back to VendorResponse."""
        from app.generate import generate_vendor_response

        result = await generate_vendor_response(
            transcribed_text="Hello vendor",
            context_block="",
            rag_context="",
            scene_context=self._scene(negotiation_stage="GREETING"),
            session_id="test-valid",
        )
        vr = VendorResponse.model_validate(result)
        assert vr.reply_text == result["reply_text"]
        assert vr.new_mood == result["new_mood"]

    async def test_session_state_persisted(self) -> None:
        """After a call, session state should be saved in the store."""
        from app.generate import generate_vendor_response

        await generate_vendor_response(
            transcribed_text="Namaste!",
            context_block="",
            rag_context="",
            scene_context=self._scene(negotiation_stage="GREETING"),
            session_id="persist-test",
        )
        state = await self.store.load_session("persist-test")
        assert state is not None
        assert state["turn_count"] == 1
        assert state["session_id"] == "persist-test"

    async def test_turn_count_increments(self) -> None:
        """Multiple calls should increment turn_count."""
        from app.generate import generate_vendor_response

        scene = self._scene(negotiation_stage="GREETING")
        for i in range(3):
            await generate_vendor_response(
                transcribed_text="Namaste!",
                context_block="",
                rag_context="",
                scene_context=scene,
                session_id="turn-count-test",
            )
        state = await self.store.load_session("turn-count-test")
        assert state is not None
        assert state["turn_count"] == 3

    async def test_price_history_tracked(self) -> None:
        """When a price is offered, it should be appended to price_history."""
        from app.generate import generate_vendor_response

        await generate_vendor_response(
            transcribed_text="Kitne ka hai?",
            context_block="",
            rag_context="",
            scene_context=self._scene(),
            session_id="price-history-test",
        )
        state = await self.store.load_session("price-history-test")
        assert state is not None
        assert 800 in state.get("price_history", [])

    async def test_empty_rag_context_works(self) -> None:
        """rag_context="" is a soft failure — function works fine."""
        from app.generate import generate_vendor_response

        result = await generate_vendor_response(
            transcribed_text="Hello",
            context_block="some history",
            rag_context="",
            scene_context=self._scene(negotiation_stage="GREETING"),
            session_id="no-rag-test",
        )
        assert isinstance(result, dict)
        assert result["reply_text"]

    async def test_response_has_all_expected_keys(self) -> None:
        """Contract: result dict must have all 7 keys."""
        from app.generate import generate_vendor_response

        result = await generate_vendor_response(
            transcribed_text="test",
            context_block="",
            rag_context="",
            scene_context=self._scene(),
            session_id="keys-test",
        )
        expected_keys = {
            "reply_text",
            "new_mood",
            "new_stage",
            "price_offered",
            "vendor_happiness",
            "vendor_patience",
            "vendor_mood",
        }
        assert set(result.keys()) == expected_keys
