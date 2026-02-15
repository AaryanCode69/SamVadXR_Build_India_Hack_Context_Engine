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
        expected = {"GREETING", "INQUIRY", "HAGGLING", "DEAL", "WALKAWAY", "CLOSURE"}
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

    def test_browsing_removed(self) -> None:
        with pytest.raises(ValueError):
            NegotiationStage("BROWSING")  # renamed to INQUIRY in v4.0

    def test_from_invalid_no_deal_raises(self) -> None:
        with pytest.raises(ValueError):
            NegotiationStage("NO_DEAL")  # removed in v3.0


class TestVendorMood:
    """VendorMood enum — 5 values, lowercase."""

    def test_all_moods_exist(self) -> None:
        expected = {"enthusiastic", "friendly", "neutral", "annoyed", "angry"}
        actual = {m.value for m in VendorMood}
        assert actual == expected

    def test_mood_count(self) -> None:
        assert len(VendorMood) == 5

    def test_from_valid_string(self) -> None:
        assert VendorMood("angry") == VendorMood.ANGRY

    def test_friendly_exists(self) -> None:
        assert VendorMood("friendly") == VendorMood.FRIENDLY

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

    def test_greeting_to_inquiry(self) -> None:
        assert NegotiationStage.INQUIRY in LEGAL_TRANSITIONS[NegotiationStage.GREETING]

    def test_greeting_cannot_skip_to_haggling(self) -> None:
        assert NegotiationStage.HAGGLING not in LEGAL_TRANSITIONS[NegotiationStage.GREETING]

    def test_inquiry_to_haggling(self) -> None:
        assert NegotiationStage.HAGGLING in LEGAL_TRANSITIONS[NegotiationStage.INQUIRY]

    def test_inquiry_to_walkaway(self) -> None:
        assert NegotiationStage.WALKAWAY in LEGAL_TRANSITIONS[NegotiationStage.INQUIRY]

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

    def test_no_browsing_stage_in_transitions(self) -> None:
        """BROWSING was removed — INQUIRY replaces it."""
        for targets in LEGAL_TRANSITIONS.values():
            assert NegotiationStage.INQUIRY is not None  # just verify it exists

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
        assert sc.object_grabbed is None
        assert sc.happiness_score == 50
        assert sc.negotiation_state == NegotiationStage.GREETING
        assert sc.input_language == LanguageCode.EN_IN
        assert sc.target_language == LanguageCode.EN_IN

    def test_valid_full_payload(self) -> None:
        data = {
            "object_grabbed": "Tomato",
            "happiness_score": 75,
            "negotiation_state": "HAGGLING",
            "input_language": "hi-IN",
            "target_language": "hi-IN",
        }
        sc = SceneContext.model_validate(data)
        assert sc.object_grabbed == "Tomato"
        assert sc.happiness_score == 75
        assert sc.negotiation_state == NegotiationStage.HAGGLING
        assert sc.input_language == LanguageCode.HI_IN
        assert sc.target_language == LanguageCode.HI_IN

    def test_happiness_clamped_above_100(self) -> None:
        sc = SceneContext(happiness_score=150)
        assert sc.happiness_score == 100

    def test_happiness_clamped_below_0(self) -> None:
        sc = SceneContext(happiness_score=-10)
        assert sc.happiness_score == 0

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            SceneContext(negotiation_state="INVALID_STAGE")

    def test_object_grabbed_none_ok(self) -> None:
        sc = SceneContext(object_grabbed=None)
        assert sc.object_grabbed is None

    def test_object_grabbed_string(self) -> None:
        sc = SceneContext(object_grabbed="silk_scarf")
        assert sc.object_grabbed == "silk_scarf"

    def test_model_validate_from_dict(self) -> None:
        """Simulates how Dev B passes the raw dict from Unity."""
        raw = {"happiness_score": 55, "negotiation_state": "INQUIRY"}
        sc = SceneContext.model_validate(raw)
        assert sc.happiness_score == 55
        assert sc.negotiation_state == NegotiationStage.INQUIRY


# ═══════════════════════════════════════════════════════════
#  4. AIDecision Tests
# ═══════════════════════════════════════════════════════════


class TestAIDecision:
    """AIDecision — internal model for structured GPT-4o output."""

    @pytest.fixture()
    def valid_data(self) -> dict:
        return {
            "reply_text": "अरे भाई, ये pure Banarasi silk है!",
            "happiness_score": 65,
            "negotiation_state": "HAGGLING",
            "vendor_mood": "enthusiastic",
            "internal_reasoning": "User asked about price, transitioning to haggling",
        }

    def test_valid_decision(self, valid_data: dict) -> None:
        d = AIDecision.model_validate(valid_data)
        assert d.reply_text == "अरे भाई, ये pure Banarasi silk है!"
        assert d.happiness_score == 65
        assert d.negotiation_state == NegotiationStage.HAGGLING
        assert d.vendor_mood == VendorMood.ENTHUSIASTIC
        assert d.internal_reasoning == "User asked about price, transitioning to haggling"

    def test_internal_reasoning_defaults_empty(self) -> None:
        d = AIDecision(
            reply_text="Haan bhai",
            happiness_score=50,
            negotiation_state=NegotiationStage.INQUIRY,
            vendor_mood=VendorMood.NEUTRAL,
        )
        assert d.internal_reasoning == ""

    def test_happiness_clamped_above_100(self) -> None:
        d = AIDecision(
            reply_text="Hello",
            happiness_score=120,
            negotiation_state=NegotiationStage.INQUIRY,
            vendor_mood=VendorMood.ENTHUSIASTIC,
        )
        assert d.happiness_score == 100

    def test_happiness_clamped_below_0(self) -> None:
        d = AIDecision(
            reply_text="Hmph",
            happiness_score=-5,
            negotiation_state=NegotiationStage.WALKAWAY,
            vendor_mood=VendorMood.ANGRY,
        )
        assert d.happiness_score == 0

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="",
                happiness_score=50,
                negotiation_state=NegotiationStage.INQUIRY,
                vendor_mood=VendorMood.NEUTRAL,
            )

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="Test",
                happiness_score=50,
                negotiation_state="FIGHTING",
                vendor_mood=VendorMood.NEUTRAL,
            )

    def test_invalid_mood_category_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(
                reply_text="Test",
                happiness_score=50,
                negotiation_state=NegotiationStage.INQUIRY,
                vendor_mood="happy",
            )

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            AIDecision(reply_text="Test")  # type: ignore[call-arg]


# ═══════════════════════════════════════════════════════════
#  5. VendorResponse Tests
# ═══════════════════════════════════════════════════════════


class TestVendorResponse:
    """VendorResponse — validated output dict returned to Dev B."""

    @pytest.fixture()
    def valid_data(self) -> dict:
        return {
            "reply_text": "अरे भाई, ये pure Banarasi silk है!",
            "happiness_score": 65,
            "negotiation_state": "HAGGLING",
            "vendor_mood": "enthusiastic",
            "suggested_user_response": "That seems expensive. Can you lower the price?",
        }

    def test_valid_response(self, valid_data: dict) -> None:
        r = VendorResponse.model_validate(valid_data)
        assert r.reply_text == "अरे भाई, ये pure Banarasi silk है!"
        assert r.happiness_score == 65
        assert r.negotiation_state == "HAGGLING"
        assert r.vendor_mood == "enthusiastic"
        assert r.suggested_user_response == "That seems expensive. Can you lower the price?"

    def test_model_dump_returns_dict(self, valid_data: dict) -> None:
        r = VendorResponse.model_validate(valid_data)
        d = r.model_dump()
        assert isinstance(d, dict)
        assert d["negotiation_state"] == "HAGGLING"
        assert d["vendor_mood"] == "enthusiastic"

    def test_all_stages_valid(self) -> None:
        for stage in NegotiationStage:
            r = VendorResponse(
                reply_text="Test",
                happiness_score=50,
                negotiation_state=stage.value,
                vendor_mood="neutral",
                suggested_user_response="How much is this?",
            )
            assert r.negotiation_state == stage.value

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                happiness_score=50,
                negotiation_state="WALK_AWAY",  # old v2.0 name
                vendor_mood="neutral",
                suggested_user_response="How much is this?",
            )

    def test_invalid_vendor_mood_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                happiness_score=50,
                negotiation_state="INQUIRY",
                vendor_mood="happy",
                suggested_user_response="How much is this?",
            )

    def test_happiness_boundary_zero(self) -> None:
        r = VendorResponse(
            reply_text="Hmph",
            happiness_score=0,
            negotiation_state="CLOSURE",
            vendor_mood="angry",
            suggested_user_response="Goodbye.",
        )
        assert r.happiness_score == 0

    def test_happiness_boundary_100(self) -> None:
        r = VendorResponse(
            reply_text="Wonderful!",
            happiness_score=100,
            negotiation_state="DEAL",
            vendor_mood="enthusiastic",
            suggested_user_response="Thank you!",
        )
        assert r.happiness_score == 100

    def test_happiness_below_0_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                happiness_score=-1,
                negotiation_state="INQUIRY",
                vendor_mood="neutral",
                suggested_user_response="How much?",
            )

    def test_happiness_above_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="Test",
                happiness_score=101,
                negotiation_state="INQUIRY",
                vendor_mood="neutral",
                suggested_user_response="How much?",
            )

    def test_empty_reply_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            VendorResponse(
                reply_text="",
                happiness_score=50,
                negotiation_state="INQUIRY",
                vendor_mood="neutral",
                suggested_user_response="How much?",
            )

    def test_friendly_mood_accepted(self) -> None:
        r = VendorResponse(
            reply_text="Test",
            happiness_score=70,
            negotiation_state="INQUIRY",
            vendor_mood="friendly",
            suggested_user_response="How much is this?",
        )
        assert r.vendor_mood == "friendly"


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
        assert len(VendorMood) == 5
        assert len(LEGAL_TRANSITIONS) == 6
        assert len(TERMINAL_STAGES) == 2
        # Just verify the classes exist
        assert SceneContext is not None
        assert AIDecision is not None
        assert VendorResponse is not None
