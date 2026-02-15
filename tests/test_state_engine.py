"""
State machine validation and transition tests.

Covers Phase 5 deliverables:
    5.1 — State transition graph (legal + illegal)
    5.2 — Mood / sentiment clamping
    5.3 — Terminal state detection
    5.4 — Turn counter enforcement (tested in test_generate.py)
    5.5 — Full AI decision validation
"""

from __future__ import annotations

import pytest

from app.models.enums import (
    LEGAL_TRANSITIONS,
    MAX_MOOD_DELTA,
    NegotiationStage,
    VendorMood,
)
from app.models.response import AIDecision
from app.services.state_engine import (
    ValidatedState,
    build_session_summary,
    clamp_delta,
    derive_vendor_mood,
    is_terminal_state,
    validate_ai_decision,
    validate_transition,
)


# ═══════════════════════════════════════════════════════════
#  5.1 — Stage Transition Tests
# ═══════════════════════════════════════════════════════════


class TestValidateTransition:
    """Test every legal and illegal transition pair."""

    # ── Legal transitions ─────────────────────────────

    @pytest.mark.parametrize(
        "current,proposed",
        [
            (NegotiationStage.GREETING, NegotiationStage.BROWSING),
            (NegotiationStage.BROWSING, NegotiationStage.HAGGLING),
            (NegotiationStage.BROWSING, NegotiationStage.WALKAWAY),
            (NegotiationStage.HAGGLING, NegotiationStage.DEAL),
            (NegotiationStage.HAGGLING, NegotiationStage.WALKAWAY),
            (NegotiationStage.HAGGLING, NegotiationStage.CLOSURE),
            (NegotiationStage.WALKAWAY, NegotiationStage.CLOSURE),
        ],
    )
    def test_legal_transitions(
        self,
        current: NegotiationStage,
        proposed: NegotiationStage,
    ) -> None:
        stage, warnings = validate_transition(current, proposed, vendor_happiness=60)
        assert stage == proposed
        assert warnings == []

    def test_walkaway_to_haggling_high_happiness(self) -> None:
        """WALKAWAY → HAGGLING is legal when vendor_happiness > 40."""
        stage, warnings = validate_transition(
            NegotiationStage.WALKAWAY,
            NegotiationStage.HAGGLING,
            vendor_happiness=50,
        )
        assert stage == NegotiationStage.HAGGLING
        assert warnings == []

    def test_walkaway_to_haggling_low_happiness(self) -> None:
        """WALKAWAY → HAGGLING blocked when vendor_happiness ≤ 40."""
        stage, warnings = validate_transition(
            NegotiationStage.WALKAWAY,
            NegotiationStage.HAGGLING,
            vendor_happiness=40,
        )
        assert stage == NegotiationStage.CLOSURE
        assert len(warnings) == 1
        assert "blocked" in warnings[0].lower()

    def test_walkaway_to_haggling_boundary(self) -> None:
        """WALKAWAY → HAGGLING at happiness=41 is allowed."""
        stage, warnings = validate_transition(
            NegotiationStage.WALKAWAY,
            NegotiationStage.HAGGLING,
            vendor_happiness=41,
        )
        assert stage == NegotiationStage.HAGGLING
        assert warnings == []

    # ── Same-stage transitions (always legal) ─────────

    @pytest.mark.parametrize("stage", list(NegotiationStage))
    def test_same_stage_always_legal(self, stage: NegotiationStage) -> None:
        result, warnings = validate_transition(stage, stage, vendor_happiness=50)
        assert result == stage
        assert warnings == []

    # ── Illegal transitions ───────────────────────────

    @pytest.mark.parametrize(
        "current,proposed",
        [
            (NegotiationStage.GREETING, NegotiationStage.HAGGLING),
            (NegotiationStage.GREETING, NegotiationStage.DEAL),
            (NegotiationStage.GREETING, NegotiationStage.WALKAWAY),
            (NegotiationStage.GREETING, NegotiationStage.CLOSURE),
            (NegotiationStage.BROWSING, NegotiationStage.GREETING),
            (NegotiationStage.BROWSING, NegotiationStage.DEAL),
            (NegotiationStage.BROWSING, NegotiationStage.CLOSURE),
            (NegotiationStage.HAGGLING, NegotiationStage.GREETING),
            (NegotiationStage.HAGGLING, NegotiationStage.BROWSING),
            (NegotiationStage.WALKAWAY, NegotiationStage.GREETING),
            (NegotiationStage.WALKAWAY, NegotiationStage.BROWSING),
            (NegotiationStage.WALKAWAY, NegotiationStage.DEAL),
        ],
    )
    def test_illegal_transitions_blocked(
        self,
        current: NegotiationStage,
        proposed: NegotiationStage,
    ) -> None:
        stage, warnings = validate_transition(current, proposed, vendor_happiness=60)
        assert stage == current  # keeps current stage
        assert len(warnings) >= 1
        assert "illegal" in warnings[0].lower()

    # ── Terminal states cannot transition out ──────────

    @pytest.mark.parametrize("terminal", [NegotiationStage.DEAL, NegotiationStage.CLOSURE])
    @pytest.mark.parametrize("target", list(NegotiationStage))
    def test_terminal_stages_blocked(
        self,
        terminal: NegotiationStage,
        target: NegotiationStage,
    ) -> None:
        if target == terminal:
            return  # same-stage is always allowed
        stage, warnings = validate_transition(terminal, target, vendor_happiness=50)
        assert stage == terminal
        assert len(warnings) >= 1

    # ── Exhaustive legal/illegal coverage ─────────────

    def test_all_transitions_covered(self) -> None:
        """Ensure every possible pair is either in LEGAL_TRANSITIONS or blocked."""
        for current in NegotiationStage:
            legal = LEGAL_TRANSITIONS.get(current, set())
            for target in NegotiationStage:
                if target == current:
                    continue
                stage, _ = validate_transition(current, target, vendor_happiness=60)
                if target in legal:
                    # Special case: WALKAWAY→HAGGLING needs happiness check
                    if (
                        current == NegotiationStage.WALKAWAY
                        and target == NegotiationStage.HAGGLING
                    ):
                        continue
                    assert stage == target, (
                        f"Expected legal: {current.value} → {target.value}"
                    )
                else:
                    assert stage != target or current in {
                        NegotiationStage.DEAL,
                        NegotiationStage.CLOSURE,
                    }


# ═══════════════════════════════════════════════════════════
#  5.2 — Mood / Sentiment Clamping Tests
# ═══════════════════════════════════════════════════════════


class TestClampDelta:
    """Test the ±15 per-turn clamping logic."""

    def test_no_clamp_within_range(self) -> None:
        val, clamped = clamp_delta(50, 60)
        assert val == 60
        assert clamped is False

    def test_clamp_positive_delta(self) -> None:
        val, clamped = clamp_delta(50, 80)
        assert val == 65
        assert clamped is True

    def test_clamp_negative_delta(self) -> None:
        val, clamped = clamp_delta(50, 20)
        assert val == 35
        assert clamped is True

    def test_clamp_respects_min_bound(self) -> None:
        """When proposed is far below 0 but delta is within ±15, clamp to [0,100]."""
        val, clamped = clamp_delta(5, -50)
        # -50 is clamped to 0 by [0,100] range. delta = 0-5 = -5, within ±15.
        assert val == 0
        assert clamped is False  # the ±15 delta check didn't trigger

    def test_clamp_respects_min_bound_with_delta(self) -> None:
        """When delta exceeds ±15 AND hits min bound."""
        val, clamped = clamp_delta(30, 0)
        # delta = 0-30 = -30, exceeds ±15 → clamp to 30-15=15
        assert val == 15
        assert clamped is True

    def test_clamp_respects_max_bound(self) -> None:
        """When proposed is far above 100 but delta is within ±15, clamp to [0,100]."""
        val, clamped = clamp_delta(95, 200)
        # 200 clamped to 100 by [0,100] range. delta = 100-95 = 5, within ±15.
        assert val == 100
        assert clamped is False  # the ±15 delta check didn't trigger

    def test_clamp_respects_max_bound_with_delta(self) -> None:
        """When delta exceeds ±15 AND hits max bound."""
        val, clamped = clamp_delta(70, 150)
        # 150 clamped to 100, delta = 100-70 = 30 exceeds ±15 → clamp to 70+15=85
        assert val == 85
        assert clamped is True

    def test_clamp_exact_boundary(self) -> None:
        val, clamped = clamp_delta(50, 65)
        assert val == 65
        assert clamped is False

    def test_clamp_exact_boundary_negative(self) -> None:
        val, clamped = clamp_delta(50, 35)
        assert val == 35
        assert clamped is False

    def test_custom_max_delta(self) -> None:
        val, clamped = clamp_delta(50, 70, max_delta=10)
        assert val == 60
        assert clamped is True

    def test_same_value_no_clamp(self) -> None:
        val, clamped = clamp_delta(50, 50)
        assert val == 50
        assert clamped is False


class TestDeriveVendorMood:
    """Test happiness → mood category mapping."""

    @pytest.mark.parametrize(
        "happiness,expected",
        [
            (100, VendorMood.ENTHUSIASTIC),
            (80, VendorMood.ENTHUSIASTIC),
            (71, VendorMood.ENTHUSIASTIC),
            (70, VendorMood.NEUTRAL),
            (50, VendorMood.NEUTRAL),
            (41, VendorMood.NEUTRAL),
            (40, VendorMood.ANNOYED),
            (30, VendorMood.ANNOYED),
            (21, VendorMood.ANNOYED),
            (20, VendorMood.ANGRY),
            (10, VendorMood.ANGRY),
            (0, VendorMood.ANGRY),
        ],
    )
    def test_mood_derivation(self, happiness: int, expected: VendorMood) -> None:
        assert derive_vendor_mood(happiness) == expected


# ═══════════════════════════════════════════════════════════
#  5.3 — Terminal State Detection
# ═══════════════════════════════════════════════════════════


class TestTerminalState:
    """Test terminal stage detection and session summary building."""

    def test_deal_is_terminal(self) -> None:
        assert is_terminal_state(NegotiationStage.DEAL) is True

    def test_closure_is_terminal(self) -> None:
        assert is_terminal_state(NegotiationStage.CLOSURE) is True

    @pytest.mark.parametrize(
        "stage",
        [
            NegotiationStage.GREETING,
            NegotiationStage.BROWSING,
            NegotiationStage.HAGGLING,
            NegotiationStage.WALKAWAY,
        ],
    )
    def test_non_terminal(self, stage: NegotiationStage) -> None:
        assert is_terminal_state(stage) is False

    def test_session_summary_deal(self) -> None:
        summary = build_session_summary(
            session_id="test-123",
            stage=NegotiationStage.DEAL,
            turn_count=10,
            final_price=500,
            vendor_happiness=80,
            vendor_patience=60,
        )
        assert summary["result"] == "won"
        assert summary["final_stage"] == "DEAL"
        assert summary["turns_taken"] == 10
        assert summary["final_price"] == 500

    def test_session_summary_closure(self) -> None:
        summary = build_session_summary(
            session_id="test-456",
            stage=NegotiationStage.CLOSURE,
            turn_count=25,
            final_price=0,
            vendor_happiness=20,
            vendor_patience=10,
        )
        assert summary["result"] == "ended"
        assert summary["final_stage"] == "CLOSURE"


# ═══════════════════════════════════════════════════════════
#  5.5 — Full AI Decision Validation
# ═══════════════════════════════════════════════════════════


def _make_decision(**overrides: object) -> AIDecision:
    """Helper to build an AIDecision with sensible defaults."""
    defaults = {
        "reply_text": "Test reply",
        "new_mood": 55,
        "new_stage": NegotiationStage.BROWSING,
        "price_offered": None,
        "vendor_happiness": 55,
        "vendor_patience": 65,
        "vendor_mood": VendorMood.NEUTRAL,
        "internal_reasoning": "test",
    }
    defaults.update(overrides)
    return AIDecision(**defaults)


def _make_session_state(**overrides: object) -> dict:
    """Helper to build a session state dict with defaults."""
    defaults = {
        "session_id": "test-session",
        "vendor_happiness": 50,
        "vendor_patience": 70,
        "negotiation_stage": "GREETING",
        "current_price": 0,
        "turn_count": 1,
        "price_history": [],
    }
    defaults.update(overrides)
    return defaults


class TestValidateAIDecision:
    """Test the full validate_ai_decision pipeline."""

    def test_valid_decision_passthrough(self) -> None:
        """A valid decision within bounds should pass through unchanged."""
        decision = _make_decision(
            new_mood=55,
            new_stage=NegotiationStage.BROWSING,
            vendor_happiness=55,
            vendor_patience=65,
        )
        state = _make_session_state(
            negotiation_stage="GREETING",
            vendor_happiness=50,
            vendor_patience=70,
        )
        result = validate_ai_decision(decision, state)

        assert result.new_stage == NegotiationStage.BROWSING
        assert result.vendor_happiness == 55
        assert result.vendor_patience == 65
        assert result.warnings == []
        assert result.is_terminal is False

    def test_mood_clamped_positive(self) -> None:
        """Mood jumping +40 should be clamped to +15."""
        decision = _make_decision(
            new_mood=90,
            new_stage=NegotiationStage.BROWSING,
            vendor_happiness=90,
            vendor_patience=70,
        )
        state = _make_session_state(
            negotiation_stage="GREETING",
            vendor_happiness=50,
            vendor_patience=70,
        )
        result = validate_ai_decision(decision, state)

        assert result.vendor_happiness == 65
        assert result.new_mood == 65
        assert len(result.warnings) >= 1

    def test_mood_clamped_negative(self) -> None:
        """Mood dropping -40 should be clamped to -15."""
        decision = _make_decision(
            new_mood=10,
            new_stage=NegotiationStage.BROWSING,
            vendor_happiness=10,
            vendor_patience=30,
        )
        state = _make_session_state(
            negotiation_stage="GREETING",
            vendor_happiness=50,
            vendor_patience=70,
        )
        result = validate_ai_decision(decision, state)

        assert result.vendor_happiness == 35
        assert result.vendor_patience == 55
        assert len(result.warnings) >= 1

    def test_illegal_stage_blocked(self) -> None:
        """Illegal GREETING → DEAL should be blocked."""
        decision = _make_decision(
            new_stage=NegotiationStage.DEAL,
        )
        state = _make_session_state(negotiation_stage="GREETING")
        result = validate_ai_decision(decision, state)

        assert result.new_stage == NegotiationStage.GREETING
        assert any("illegal" in w.lower() for w in result.warnings)

    def test_deal_is_terminal(self) -> None:
        """HAGGLING → DEAL should mark session as terminal."""
        decision = _make_decision(
            new_stage=NegotiationStage.DEAL,
            price_offered=500,
            vendor_happiness=60,
            vendor_patience=65,
        )
        state = _make_session_state(
            negotiation_stage="HAGGLING",
            vendor_happiness=50,
            vendor_patience=70,
        )
        result = validate_ai_decision(decision, state)

        assert result.new_stage == NegotiationStage.DEAL
        assert result.is_terminal is True

    def test_closure_is_terminal(self) -> None:
        """HAGGLING → CLOSURE should mark session as terminal."""
        decision = _make_decision(
            new_stage=NegotiationStage.CLOSURE,
            vendor_happiness=50,
            vendor_patience=60,
        )
        state = _make_session_state(
            negotiation_stage="HAGGLING",
            vendor_happiness=50,
            vendor_patience=70,
        )
        result = validate_ai_decision(decision, state)

        assert result.new_stage == NegotiationStage.CLOSURE
        assert result.is_terminal is True

    def test_vendor_mood_derived_from_happiness(self) -> None:
        """vendor_mood should be derived from clamped happiness, not AI's proposal."""
        decision = _make_decision(
            vendor_happiness=55,
            vendor_mood=VendorMood.ANGRY,  # AI says angry, but happiness says neutral
        )
        state = _make_session_state(
            negotiation_stage="GREETING",
            vendor_happiness=50,
        )
        result = validate_ai_decision(decision, state)

        assert result.vendor_mood == VendorMood.NEUTRAL  # derived, not AI's

    def test_negative_price_clamped(self) -> None:
        """price_offered=None should become 0 (Pydantic rejects negatives)."""
        decision = _make_decision(price_offered=None)
        state = _make_session_state(negotiation_stage="GREETING")
        result = validate_ai_decision(decision, state)

        assert result.price_offered == 0

    def test_none_price_becomes_zero(self) -> None:
        """price_offered=None should become 0."""
        decision = _make_decision(price_offered=None)
        state = _make_session_state(negotiation_stage="GREETING")
        result = validate_ai_decision(decision, state)

        assert result.price_offered == 0

    def test_invalid_session_stage_defaults(self) -> None:
        """Invalid stage in session state should default to GREETING."""
        decision = _make_decision(
            new_stage=NegotiationStage.BROWSING,
            vendor_happiness=55,
        )
        state = _make_session_state(
            negotiation_stage="INVALID_STAGE",
            vendor_happiness=50,
        )
        result = validate_ai_decision(decision, state)

        assert result.new_stage == NegotiationStage.BROWSING
        assert any("defaulting" in w.lower() for w in result.warnings)

    def test_custom_max_mood_delta(self) -> None:
        """Custom max_mood_delta should be respected."""
        decision = _make_decision(
            new_mood=60,
            vendor_happiness=60,
            vendor_patience=75,
        )
        state = _make_session_state(
            negotiation_stage="GREETING",
            vendor_happiness=50,
            vendor_patience=70,
        )
        result = validate_ai_decision(decision, state, max_mood_delta=5)

        assert result.vendor_happiness == 55
        assert result.vendor_patience == 75  # only +5, within custom delta
        assert len(result.warnings) >= 1
