"""
State machine validation and transition logic.

The AI proposes state changes; this engine approves or rejects them.
Every override is logged at WARN level.

Public API:
    validate_transition(current, proposed, happiness_score) → (stage, warnings)
    validate_ai_decision(ai_decision, session_state, config) → ValidatedState
    derive_vendor_mood(happiness_score) → VendorMood
    validate_price_consistency(ai_decision, session_state) → list[str]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.models.enums import (
    LEGAL_TRANSITIONS,
    MAX_MOOD_DELTA,
    MOOD_MAX,
    MOOD_MIN,
    TERMINAL_STAGES,
    NegotiationStage,
    VendorMood,
)
from app.models.response import AIDecision

logger = logging.getLogger("samvadxr")

# ── Offer assessment → minimum expected happiness drop ──
# Used by validate_offer_happiness_consistency()
_OFFER_MIN_DROPS: dict[str, int] = {
    "insult": 10,     # Must drop at least 10 for insult offers
    "lowball": 6,     # Must drop at least 6 for lowball offers
    "fair": 0,        # Fair offers — drop is expected but not enforced
    "good": 0,        # Good offers — no drop required
    "excellent": 0,   # Excellent offers — no drop required
    "none": 0,        # No offer made — no constraint
}


# ═══════════════════════════════════════════════════════════
#  Validated output container
# ═══════════════════════════════════════════════════════════


@dataclass
class ValidatedState:
    """Result of passing an AIDecision through the state engine.

    All values are clamped/corrected and safe to persist + return.
    """

    reply_text: str
    happiness_score: int
    negotiation_state: NegotiationStage
    vendor_mood: VendorMood
    suggested_user_response: str = ""
    internal_reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    is_terminal: bool = False


# ═══════════════════════════════════════════════════════════
#  Stage transition validation (5.1 + 5.5)
# ═══════════════════════════════════════════════════════════


def validate_transition(
    current_stage: NegotiationStage,
    proposed_stage: NegotiationStage,
    happiness_score: int,
) -> tuple[NegotiationStage, list[str]]:
    """Validate a proposed stage transition.

    Rules:
        - Transition must be in LEGAL_TRANSITIONS graph.
        - WALKAWAY → HAGGLING only if happiness_score > 40.
        - Terminal stages cannot be left.
        - Staying in the same stage is always allowed.

    Returns:
        (approved_stage, list_of_warnings)
    """
    warnings: list[str] = []

    # Staying in the same stage is always legal
    if proposed_stage == current_stage:
        return proposed_stage, warnings

    # Terminal stages cannot transition out
    if current_stage in TERMINAL_STAGES:
        warnings.append(
            f"Cannot leave terminal stage {current_stage.value}. "
            f"Proposed {proposed_stage.value} blocked."
        )
        return current_stage, warnings

    # Check the transition graph
    legal_targets = LEGAL_TRANSITIONS.get(current_stage, set())
    if proposed_stage not in legal_targets:
        warnings.append(
            f"Illegal transition {current_stage.value} → {proposed_stage.value}. "
            f"Legal targets: {sorted(s.value for s in legal_targets)}. "
            f"Keeping {current_stage.value}."
        )
        return current_stage, warnings

    # Special rule: WALKAWAY → HAGGLING only if happiness_score > 40
    if (
        current_stage == NegotiationStage.WALKAWAY
        and proposed_stage == NegotiationStage.HAGGLING
        and happiness_score <= 40
    ):
        warnings.append(
            f"WALKAWAY → HAGGLING blocked: happiness_score={happiness_score} "
            f"(must be > 40). Forcing CLOSURE."
        )
        return NegotiationStage.CLOSURE, warnings

    return proposed_stage, warnings


# ═══════════════════════════════════════════════════════════
#  Mood / sentiment mechanics (5.2)
# ═══════════════════════════════════════════════════════════


def derive_vendor_mood(happiness_score: int) -> VendorMood:
    """Derive categorical VendorMood from numeric happiness.

    Ranges (from enums.py docstring):
        happiness > 80  → enthusiastic
        happiness 61-80 → friendly
        happiness 41-60 → neutral
        happiness 21-40 → annoyed
        happiness ≤ 20  → angry
    """
    if happiness_score > 80:
        return VendorMood.ENTHUSIASTIC
    if happiness_score > 60:
        return VendorMood.FRIENDLY
    if happiness_score > 40:
        return VendorMood.NEUTRAL
    if happiness_score > 20:
        return VendorMood.ANNOYED
    return VendorMood.ANGRY


def clamp_delta(
    current: int, proposed: int, max_delta: int = MAX_MOOD_DELTA
) -> tuple[int, bool]:
    """Clamp a numeric value's change to ±max_delta from current.

    Returns:
        (clamped_value, was_clamped)
    """
    clamped = max(MOOD_MIN, min(MOOD_MAX, proposed))
    delta = clamped - current
    was_clamped = False

    if abs(delta) > max_delta:
        clamped = current + (max_delta if delta > 0 else -max_delta)
        clamped = max(MOOD_MIN, min(MOOD_MAX, clamped))
        was_clamped = True

    return clamped, was_clamped


# ═══════════════════════════════════════════════════════════
#  Win/loss condition detection (5.3)
# ═══════════════════════════════════════════════════════════


def is_terminal_state(stage: NegotiationStage) -> bool:
    """Check if a negotiation stage is terminal (DEAL or CLOSURE)."""
    return stage in TERMINAL_STAGES


def build_session_summary(
    session_id: str,
    stage: NegotiationStage,
    turn_count: int,
    happiness_score: int,
) -> dict[str, Any]:
    """Build a structured summary for terminal-state logging."""
    result = "won" if stage == NegotiationStage.DEAL else "ended"
    return {
        "session_id": session_id,
        "result": result,
        "final_stage": stage.value,
        "turns_taken": turn_count,
        "final_happiness_score": happiness_score,
    }


# ═══════════════════════════════════════════════════════════
#  Price & offer-happiness consistency validation (v6.0)
# ═══════════════════════════════════════════════════════════


def validate_offer_happiness_consistency(
    ai_decision: AIDecision,
    current_happiness: int,
) -> tuple[int, list[str]]:
    """Enforce that insulting/lowball offers actually cause happiness drops.

    If the LLM assessed an offer as "insult" or "lowball" but didn't drop
    happiness enough, we force a minimum drop. This prevents the vendor
    from being too lenient on terrible offers.

    Args:
        ai_decision: The raw AI decision with offer_assessment.
        current_happiness: The authoritative happiness before this turn.

    Returns:
        (adjusted_happiness, list_of_warnings)
    """
    warnings: list[str] = []
    proposed_happiness = ai_decision.happiness_score
    assessment = (ai_decision.offer_assessment or "none").lower()

    min_drop = _OFFER_MIN_DROPS.get(assessment, 0)
    if min_drop <= 0:
        return proposed_happiness, warnings

    actual_delta = current_happiness - proposed_happiness  # positive = drop

    if actual_delta < min_drop:
        # LLM was too lenient — enforce minimum drop
        forced_happiness = max(MOOD_MIN, current_happiness - min_drop)
        warnings.append(
            f"Offer assessed as '{assessment}' but happiness only dropped "
            f"{actual_delta} (min required: {min_drop}). "
            f"Forcing happiness: {proposed_happiness} → {forced_happiness}"
        )
        return forced_happiness, warnings

    return proposed_happiness, warnings


def validate_price_direction(
    ai_decision: AIDecision,
    session_state: dict[str, Any],
) -> list[str]:
    """Warn if the vendor raised their price (which breaks negotiation realism).

    Prices should only go DOWN during negotiation (vendor concedes),
    never UP (unless re-entering from WALKAWAY, which is a special case).

    This is a soft validation — we log warnings but don't force a change,
    because the LLM might have a valid reason (e.g. switching items).
    """
    warnings: list[str] = []
    counter_price = ai_decision.counter_price
    last_price = session_state.get("last_counter_price")

    if counter_price is None or last_price is None:
        return warnings

    current_stage = session_state.get("negotiation_state", "GREETING")
    # Allow price reset after WALKAWAY → HAGGLING re-entry
    if current_stage == "WALKAWAY":
        return warnings

    if counter_price > last_price:
        warnings.append(
            f"Vendor raised price from {last_price} to {counter_price}. "
            f"Prices should only decrease during negotiation."
        )

    return warnings


# ═══════════════════════════════════════════════════════════
#  Full AI decision validation (5.5 — the main entry point)
# ═══════════════════════════════════════════════════════════


def validate_ai_decision(
    ai_decision: AIDecision,
    session_state: dict[str, Any],
    *,
    max_mood_delta: int = MAX_MOOD_DELTA,
) -> ValidatedState:
    """Validate and clamp the AI's proposed state changes.

    This sits between the AI Brain output and the return value of
    generate_vendor_response(). The AI proposes; this engine disposes.

    Steps:
        1. Validate stage transition (legal + special rules).
        2. Enforce offer-happiness consistency (v6.0 — insult/lowball must hurt).
        3. Clamp happiness_score ±max_mood_delta from current.
        4. Validate price direction (prices should only go down).
        5. Derive vendor_mood from clamped happiness.
        6. Detect terminal state.

    Args:
        ai_decision: Raw proposal from the AI brain.
        session_state: Authoritative state from Neo4j / session store.
        max_mood_delta: Per-turn clamp (default from config).

    Returns:
        ValidatedState with all values corrected and warnings listed.
    """
    warnings: list[str] = []

    # Current authoritative values
    current_happiness = session_state.get("happiness_score", 50)
    current_stage_str = session_state.get("negotiation_state", "GREETING")

    try:
        current_stage = NegotiationStage(current_stage_str)
    except ValueError:
        current_stage = NegotiationStage.GREETING
        warnings.append(
            f"Invalid current stage '{current_stage_str}' in session — "
            f"defaulting to GREETING."
        )

    # ── 1. Stage transition ───────────────────────────
    proposed_stage = ai_decision.negotiation_state
    approved_stage, stage_warnings = validate_transition(
        current_stage, proposed_stage, ai_decision.happiness_score
    )
    warnings.extend(stage_warnings)

    # ── 2. Offer-happiness consistency (v6.0) ─────────
    # If the LLM assessed an offer as "insult" or "lowball" but didn't
    # drop happiness enough, force a minimum drop BEFORE clamping.
    adjusted_happiness, offer_warnings = validate_offer_happiness_consistency(
        ai_decision, current_happiness
    )
    warnings.extend(offer_warnings)

    # Use the adjusted happiness for clamping (may have been forced down)
    happiness_for_clamping = adjusted_happiness

    # ── 3. Clamp happiness_score ─────────────────────
    clamped_happiness, was_clamped = clamp_delta(
        current_happiness, happiness_for_clamping, max_mood_delta
    )
    if was_clamped:
        warnings.append(
            f"happiness_score clamped: {happiness_for_clamping} → "
            f"{clamped_happiness} (current={current_happiness}, "
            f"max_delta=±{max_mood_delta})"
        )

    # ── 4. Price direction validation (v6.0) ──────────
    price_warnings = validate_price_direction(ai_decision, session_state)
    warnings.extend(price_warnings)

    # ── 5. Derive vendor_mood from happiness ──────────
    derived_mood = derive_vendor_mood(clamped_happiness)

    # ── 6. Terminal state detection ───────────────────
    terminal = is_terminal_state(approved_stage)

    # ── Log warnings ──────────────────────────────────
    for w in warnings:
        logger.warning(
            "State engine override",
            extra={
                "step": "state_validation",
                "warning": w,
            },
        )

    if terminal:
        summary = build_session_summary(
            session_id=session_state.get("session_id", "unknown"),
            stage=approved_stage,
            turn_count=session_state.get("turn_count", 0),
            happiness_score=clamped_happiness,
        )
        logger.info(
            "Terminal state reached",
            extra={
                "step": "session_terminal",
                **summary,
            },
        )

    return ValidatedState(
        reply_text=ai_decision.reply_text,
        happiness_score=clamped_happiness,
        negotiation_state=approved_stage,
        vendor_mood=derived_mood,
        suggested_user_response=ai_decision.suggested_user_response,
        internal_reasoning=ai_decision.internal_reasoning,
        warnings=warnings,
        is_terminal=terminal,
    )
