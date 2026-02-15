"""
Phase 1 — Pydantic model validation tests.

Coverage targets (rules.md §9.1):
    - All Pydantic models: 100% (valid data, invalid data, edge cases)
    - NegotiationStage: all values + illegal values
    - VendorMood: all values + illegal values
    - SceneContext: defaults, valid overrides, out-of-range clamping
    - AIDecision: valid parsing, missing fields, out-of-range clamping
    - VendorResponse: valid, invalid stage/mood, boundary values
    - Exceptions: instantiation and message
    - LEGAL_TRANSITIONS: coverage of the transition graph
"""

import pytest
from pydantic import ValidationError

from app.exceptions import BrainServiceError, StateStoreError
from app.models.enums import (
    LEGAL_TRANSITIONS,
    MOOD_MAX,
    MOOD_MIN,
    TERMINAL_STAGES,
    LanguageCode,
    NegotiationStage,
    VendorMood,
)
from app.models.request import SceneContext
from app.models.response import AIDecision, VendorResponse


# ═══════════════════════════════════════════════════════════
#  1. Enum Tests
# ═══════════════════════════════════════════════════════════


class TestNegotiationStage:
    """NegotiationStage enum — 6 values, closed set."""

    def test_all_stages_exist(self) -> None:
        expected = {"GREETING", "BROWSING", "HAGGLING", "DEAL", "WALKAWAY", "CLOSURE"}
        actual = {s.value for s in NegotiationStage}
        assert actual == expected

    def test_stage_count(self) -> None:
        assert len(NegotiationStage) == 6

    def test_string_conversion(self) -> None:
        assert NegotiationStage.HAGGLING.value == "HAGGLING"
        assert str(NegotiationStage.HAGGLING) == "NegotiationStage.HAGGLING"

    def test_from_valid_string(self) -> None:
        assert NegotiationStage("WALKAWAY") == NegotiationStage.WALKAWAY

    def test_from_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            NegotiationStage("WALK_AWAY")  # old v2.0 name

    def test_from_invalid_no_deal_raises(self) -> None:
        with pytest.raises(ValueError):
            NegotiationStage("NO_DEAL")  # removed in v3.0


class TestVendorMood:
    """VendorMood enum — 4 values, lowercase."""

    def test_all_moods_exist(self) -> None:
        expected = {"enthusiastic", "neutral", "annoyed", "angry"}
        actual = {m.value for m in VendorMood}
        assert actual == expected

    def test_mood_count(self) -> None:
        assert len(VendorMood) == 4

    def test_from_valid_string(self) -> None:
        assert VendorMood("angry") == VendorMood.ANGRY

    def test_from_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError):
            VendorMood("happy")


class TestLanguageCode:
    """LanguageCode enum — 5 Sarvam-style language codes."""

    def test_all_codes_exist(self) -> None:
        expected = {"hi-IN", "kn-IN", "ta-IN", "en-IN", "hi-EN"}
        actual = {lc.value for lc in LanguageCode}
        assert actual == expected

    def test_hindi_code(self) -> None:
        assert LanguageCode.HI_IN.value == "hi-IN"


# ═══════════════════════════════════════════════════════════
#  2. Legal Transitions & Constants Tests
# ═══════════════════════════════════════════════════════════


class TestLegalTransitions:
    """Transition graph from rules.md §5.1."""

    def test_greeting_to_browsing(self) -> None:
        assert NegotiationStage.BROWSING in LEGAL_TRANSITIONS[NegotiationStage.GREETING]

    def test_greeting_cannot_skip_to_haggling(self) -> None:
        assert NegotiationStage.HAGGLING not in LEGAL_TRANSITIONS[NegotiationStage.GREETING]

    def test_browsing_to_haggling(self) -> None:
        assert NegotiationStage.HAGGLING in LEGAL_TRANSITIONS[NegotiationStage.BROWSING]

    def test_browsing_to_walkaway(self) -> None:
        assert NegotiationStage.WALKAWAY in LEGAL_TRANSITIONS[NegotiationStage.BROWSING]

    def test_haggling_to_deal(self) -> None:
        assert NegotiationStage.DEAL in LEGAL_TRANSITIONS[NegotiationStage.HAGGLING]

    def test_haggling_to_walkaway(self) -> None:
        assert NegotiationStage.WALKAWAY in LEGAL_TRANSITIONS[NegotiationStage.HAGGLING]

    def test_haggling_to_closure(self) -> None:
        assert NegotiationStage.CLOSURE in LEGAL_TRANSITIONS[NegotiationStage.HAGGLING]

    def test_walkaway_to_haggling(self) -> None:
        # Allowed only if vendor_happiness > 40 — graph permits it, engine enforces condition
        assert NegotiationStage.HAGGLING in LEGAL_TRANSITIONS[NegotiationStage.WALKAWAY]

    def test_walkaway_to_closure(self) -> None:
        assert NegotiationStage.CLOSURE in LEGAL_TRANSITIONS[NegotiationStage.WALKAWAY]

    def test_deal_is_terminal(self) -> None:
        assert LEGAL_TRANSITIONS[NegotiationStage.DEAL] == set()
        assert NegotiationStage.DEAL in TERMINAL_STAGES

    def test_closure_is_terminal(self) -> None:
        assert LEGAL_TRANSITIONS[NegotiationStage.CLOSURE] == set()
        assert NegotiationStage.CLOSURE in TERMINAL_STAGES

    def test_no_backward_from_deal(self) -> None:
        assert NegotiationStage.HAGGLING not in LEGAL_TRANSITIONS[NegotiationStage.DEAL]

    def test_all_stages_have_transition_entry(self) -> None:
        for stage in NegotiationStage:
            assert stage in LEGAL_TRANSITIONS


# ═══════════════════════════════════════════════════════════
#  3. SceneContext Tests
# ═══════════════════════════════════════════════════════════


class TestSceneContext:
    """SceneContext — input model for scene_context dict from Unity."""

    def test_defaults(self) -> None:
        sc = SceneContext()
        assert sc.items_in_hand == []
        assert sc.looking_at is None
        assert sc.distance_to_vendor == 1.0
        assert sc.vendor_npc_id == "vendor_01"
        assert sc.vendor_happiness == 50
        assert sc.vendor_patience == 70
        assert sc.negotiation_stage == NegotiationStage.BROWSING
        assert sc.current_price == 0
        assert sc.user_offer == 0

    def test_valid_full_payload(self) -> None:
        data = {
            "items_in_hand": ["brass_keychain", "silk_scarf"],
            "looking_at": "silver_ring",
            "distance_to_vendor": 0.8,
            "vendor_npc_id": "vendor_02",
            "vendor_happiness": 75,
            "vendor_patience": 60,
            "negotiation_stage": "HAGGLING",
            "current_price": 500,
            "user_offer": 300,
        }
        sc = SceneContext.model_validate(data)
        assert sc.items_in_hand == ["brass_keychain", "silk_scarf"]
        assert sc.looking_at == "silver_ring"
        assert sc.distance_to_vendor == 0.8
        assert sc.negotiation_stage == NegotiationStage.HAGGLING
        assert sc.current_price == 500
        assert sc.user_offer == 300

    def test_happiness_clamped_above_100(self) -> None:
        sc = SceneContext(vendor_happiness=150)
        assert sc.vendor_happiness == 100

    def test_happiness_clamped_below_0(self) -> None:
        sc = SceneContext(vendor_happiness=-10)
        assert sc.vendor_happiness == 0

    def test_patience_clamped_above_100(self) -> None:
        sc = SceneContext(vendor_patience=999)
        assert sc.vendor_patience == 100

    def test_patience_clamped_below_0(self) -> None:
        sc = SceneContext(vendor_patience=-50)
        assert sc.vendor_patience == 0

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(negotiation_stage="INVALID_STAGE")

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(current_price=-100)

    def test_negative_user_offer_raises(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(user_offer=-1)

    def test_negative_distance_raises(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(distance_to_vendor=-0.5)

    def test_empty_items_in_hand(self) -> None:
        sc = SceneContext(items_in_hand=[])
        assert sc.items_in_hand == []

    def test_model_validate_from_dict(self) -> None:
        """Simulates how Dev B passes the raw dict from Unity."""
        raw = {"vendor_happiness": 55, "negotiation_stage": "BROWSING"}
        sc = SceneContext.model_validate(raw)
        assert sc.vendor_happiness == 55
        assert sc.negotiation_stage == NegotiationStage.BROWSING


# ═══════════════════════════════════════════════════════════
#  4. AIDecision Tests
# ═══════════════════════════════════════════════════════════


class TestAIDecision:
    """AIDecision — internal model for structured GPT-4o output."""

    @pytest.fixture()
    def valid_data(self) -> dict:
        return {
            "reply_text": "अरे भाई, ये pure Banarasi silk है!",
            "new_mood": 60,
            "new_stage": "HAGGLING",
            "price_offered": 800,
            "vendor_happiness": 65,
            "vendor_patience": 55,
            "vendor_mood": "enthusiastic",
            "internal_reasoning": "User asked about price, transitioning to haggling",
        }

    def test_valid_decision(self, valid_data: dict) -> None:
        d = AIDecision.model_validate(valid_data)
        assert d.reply_text == "अरे भाई, ये pure Banarasi silk है!"
        assert d.new_mood == 60
        assert d.new_stage == NegotiationStage.HAGGLING
        assert d.price_offered == 800
        assert d.vendor_happiness == 65
        assert d.vendor_patience == 55
        assert d.vendor_mood == VendorMood.ENTHUSIASTIC
        assert d.internal_reasoning == "User asked about price, transitioning to haggling"

    def test_price_offered_optional(self) -> None:
        d = AIDecision(
            reply_text="Namaste!",
            new_mood=50,
            new_stage=NegotiationStage.GREETING,
            vendor_happiness=50,
            vendor_patience=70,
            vendor_mood=VendorMood.NEUTRAL,
        )
        assert d.price_offered is None

    def test_internal_reasoning_defaults_empty(self) -> None:
        d = AIDecision(
            reply_text="Haan bhai",
            new_mood=50,
            new_stage=NegotiationStage.BROWSING,
            vendor_happiness=50,
            vendor_patience=70,
            vendor_mood=VendorMood.NEUTRAL,
        )
        assert d.internal_reasoning == ""

    def test_mood_clamped_above_100(self) -> None:
        d = AIDecision(
            reply_text="Hello",
            new_mood=120,
            new_stage=NegotiationStage.BROWSING,
            vendor_happiness=110,
            vendor_patience=200,
            vendor_mood=VendorMood.ENTHUSIASTIC,
        )
        assert d.new_mood == 100
        assert d.vendor_happiness == 100
        assert d.vendor_patience == 100

    def test_mood_clamped_below_0(self) -> None:
        d = AIDecision(
            reply_text="Hmph",
            new_mood=-5,
            new_stage=NegotiationStage.WALKAWAY,
            vendor_happiness=-20,
            vendor_patience=-10,
            vendor_mood=VendorMood.ANGRY,
        )
        assert d.new_mood == 0
        assert d.vendor_happiness == 0
        assert d.vendor_patience == 0

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="",
                new_mood=50,
                new_stage=NegotiationStage.BROWSING,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood=VendorMood.NEUTRAL,
            )

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="Test",
                new_mood=50,
                new_stage="FIGHTING",
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood=VendorMood.NEUTRAL,
            )

    def test_invalid_mood_category_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="Test",
                new_mood=50,
                new_stage=NegotiationStage.BROWSING,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="happy",
            )

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(reply_text="Test")  # type: ignore[call-arg]

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="Test",
                new_mood=50,
                new_stage=NegotiationStage.BROWSING,
                price_offered=-100,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood=VendorMood.NEUTRAL,
            )


# ═══════════════════════════════════════════════════════════
#  5. VendorResponse Tests
# ═══════════════════════════════════════════════════════════


class TestVendorResponse:
    """VendorResponse — validated output dict returned to Dev B."""

    @pytest.fixture()
    def valid_data(self) -> dict:
        return {
            "reply_text": "अरे भाई, ये pure Banarasi silk है!",
            "new_mood": 60,
            "new_stage": "HAGGLING",
            "price_offered": 800,
            "vendor_happiness": 65,
            "vendor_patience": 55,
            "vendor_mood": "enthusiastic",
        }

    def test_valid_response(self, valid_data: dict) -> None:
        r = VendorResponse.model_validate(valid_data)
        assert r.reply_text == "अरे भाई, ये pure Banarasi silk है!"
        assert r.new_mood == 60
        assert r.new_stage == "HAGGLING"
        assert r.price_offered == 800
        assert r.vendor_happiness == 65
        assert r.vendor_patience == 55
        assert r.vendor_mood == "enthusiastic"

    def test_model_dump_returns_dict(self, valid_data: dict) -> None:
        r = VendorResponse.model_validate(valid_data)
        d = r.model_dump()
        assert isinstance(d, dict)
        assert d["new_stage"] == "HAGGLING"
        assert d["vendor_mood"] == "enthusiastic"

    def test_all_stages_valid(self) -> None:
        for stage in NegotiationStage:
            r = VendorResponse(
                reply_text="Test",
                new_mood=50,
                new_stage=stage.value,
                price_offered=0,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="neutral",
            )
            assert r.new_stage == stage.value

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                new_mood=50,
                new_stage="WALK_AWAY",  # old v2.0 name
                price_offered=0,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="neutral",
            )

    def test_invalid_vendor_mood_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                new_mood=50,
                new_stage="BROWSING",
                price_offered=0,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="happy",
            )

    def test_mood_boundary_zero(self) -> None:
        r = VendorResponse(
            reply_text="Hmph",
            new_mood=0,
            new_stage="CLOSURE",
            price_offered=0,
            vendor_happiness=0,
            vendor_patience=0,
            vendor_mood="angry",
        )
        assert r.new_mood == 0

    def test_mood_boundary_100(self) -> None:
        r = VendorResponse(
            reply_text="Wonderful!",
            new_mood=100,
            new_stage="DEAL",
            price_offered=500,
            vendor_happiness=100,
            vendor_patience=100,
            vendor_mood="enthusiastic",
        )
        assert r.new_mood == 100

    def test_mood_below_0_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                new_mood=-1,
                new_stage="BROWSING",
                price_offered=0,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="neutral",
            )

    def test_mood_above_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                new_mood=101,
                new_stage="BROWSING",
                price_offered=0,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="neutral",
            )

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                new_mood=50,
                new_stage="BROWSING",
                price_offered=-50,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="neutral",
            )

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="",
                new_mood=50,
                new_stage="BROWSING",
                price_offered=0,
                vendor_happiness=50,
                vendor_patience=70,
                vendor_mood="neutral",
            )


# ═══════════════════════════════════════════════════════════
#  6. Exception Tests
# ═══════════════════════════════════════════════════════════


class TestExceptions:
    """BrainServiceError and StateStoreError — Dev A's documented exceptions."""

    def test_brain_service_error_default_message(self) -> None:
        err = BrainServiceError()
        assert str(err) == "AI brain service unavailable"
        assert err.message == "AI brain service unavailable"

    def test_brain_service_error_custom_message(self) -> None:
        err = BrainServiceError("OpenAI timeout after 3 retries")
        assert str(err) == "OpenAI timeout after 3 retries"

    def test_state_store_error_default_message(self) -> None:
        err = StateStoreError()
        assert str(err) == "State store unavailable"

    def test_state_store_error_custom_message(self) -> None:
        err = StateStoreError("Neo4j connection refused")
        assert str(err) == "Neo4j connection refused"

    def test_brain_error_is_exception(self) -> None:
        assert issubclass(BrainServiceError, Exception)

    def test_state_error_is_exception(self) -> None:
        assert issubclass(StateStoreError, Exception)

    def test_brain_error_can_be_caught(self) -> None:
        with pytest.raises(BrainServiceError):
            raise BrainServiceError("test")

    def test_state_error_can_be_caught(self) -> None:
        with pytest.raises(StateStoreError):
            raise StateStoreError("test")


# ═══════════════════════════════════════════════════════════
#  7. Integration: generate_vendor_response types
# ═══════════════════════════════════════════════════════════


class TestGenerateImports:
    """Verify generate.py imports and re-exports are correct."""

    def test_import_generate_function(self) -> None:
        from app.generate import generate_vendor_response
        assert callable(generate_vendor_response)

    def test_import_exceptions_from_generate(self) -> None:
        from app.generate import BrainServiceError, StateStoreError
        assert issubclass(BrainServiceError, Exception)
        assert issubclass(StateStoreError, Exception)

    def test_import_from_models_package(self) -> None:
        from app.models import (
            AIDecision,
            LEGAL_TRANSITIONS,
            NegotiationStage,
            SceneContext,
            TERMINAL_STAGES,
            VendorMood,
            VendorResponse,
        )
        assert len(NegotiationStage) == 6
        assert len(VendorMood) == 4
        assert len(LEGAL_TRANSITIONS) == 6
        assert len(TERMINAL_STAGES) == 2
        # Just verify the classes exist
        assert SceneContext is not None
        assert AIDecision is not None
        assert VendorResponse is not None
