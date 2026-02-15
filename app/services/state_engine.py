"""
State machine validation and transition logic.

The AI proposes state changes; this engine approves or rejects them.
Every override is logged at WARN level.

Public API:
    validate_transition(current, proposed, vendor_happiness) → (stage, warnings)
    validate_ai_decision(ai_decision, session_state, config) → ValidatedState
    derive_vendor_mood(vendor_happiness) → VendorMood
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


# ═══════════════════════════════════════════════════════════
#  Validated output container
# ═══════════════════════════════════════════════════════════


@dataclass
class ValidatedState:
    """Result of passing an AIDecision through the state engine.

    All values are clamped/corrected and safe to persist + return.
    """

    reply_text: str
    new_mood: int
    new_stage: NegotiationStage
    price_offered: int
    vendor_happiness: int
    vendor_patience: int
    vendor_mood: VendorMood
    internal_reasoning: str = ""
    warnings: list[str] = field(default_factory=list)
    is_terminal: bool = False


# ═══════════════════════════════════════════════════════════
#  Stage transition validation (5.1 + 5.5)
# ═══════════════════════════════════════════════════════════


def validate_transition(
    current_stage: NegotiationStage,
    proposed_stage: NegotiationStage,
    vendor_happiness: int,
) -> tuple[NegotiationStage, list[str]]:
    """Validate a proposed stage transition.

    Rules:
        - Transition must be in LEGAL_TRANSITIONS graph.
        - WALKAWAY → HAGGLING only if vendor_happiness > 40.
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

    # Special rule: WALKAWAY → HAGGLING only if vendor_happiness > 40
    if (
        current_stage == NegotiationStage.WALKAWAY
        and proposed_stage == NegotiationStage.HAGGLING
        and vendor_happiness <= 40
    ):
        warnings.append(
            f"WALKAWAY → HAGGLING blocked: vendor_happiness={vendor_happiness} "
            f"(must be > 40). Forcing CLOSURE."
        )
        return NegotiationStage.CLOSURE, warnings

    return proposed_stage, warnings


# ═══════════════════════════════════════════════════════════
#  Mood / sentiment mechanics (5.2)
# ═══════════════════════════════════════════════════════════


def derive_vendor_mood(vendor_happiness: int) -> VendorMood:
    """Derive categorical VendorMood from numeric happiness.

    Ranges (from enums.py docstring):
        happiness > 70  → enthusiastic
        happiness 41-70 → neutral
        happiness 21-40 → annoyed
        happiness ≤ 20  → angry
    """
    if vendor_happiness > 70:
        return VendorMood.ENTHUSIASTIC
    if vendor_happiness > 40:
        return VendorMood.NEUTRAL
    if vendor_happiness > 20:
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
    final_price: int,
    vendor_happiness: int,
    vendor_patience: int,
) -> dict[str, Any]:
    """Build a structured summary for terminal-state logging."""
    result = "won" if stage == NegotiationStage.DEAL else "ended"
    return {
        "session_id": session_id,
        "result": result,
        "final_stage": stage.value,
        "turns_taken": turn_count,
        "final_price": final_price,
        "final_vendor_happiness": vendor_happiness,
        "final_vendor_patience": vendor_patience,
    }


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
        2. Clamp vendor_happiness ±max_mood_delta from current.
        3. Clamp vendor_patience ±max_mood_delta from current.
        4. Clamp new_mood ±max_mood_delta from current.
        5. Derive vendor_mood from clamped happiness.
        6. Ensure price_offered is non-negative.
        7. Detect terminal state.

    Args:
        ai_decision: Raw proposal from the AI brain.
        session_state: Authoritative state from Neo4j / session store.
        max_mood_delta: Per-turn clamp (default from config).

    Returns:
        ValidatedState with all values corrected and warnings listed.
    """
    warnings: list[str] = []

    # Current authoritative values
    current_happiness = session_state.get("vendor_happiness", 50)
    current_patience = session_state.get("vendor_patience", 70)
    current_mood = current_happiness  # new_mood tracks happiness for v3
    current_stage_str = session_state.get("negotiation_stage", "GREETING")

    try:
        current_stage = NegotiationStage(current_stage_str)
    except ValueError:
        current_stage = NegotiationStage.GREETING
        warnings.append(
            f"Invalid current stage '{current_stage_str}' in session — "
            f"defaulting to GREETING."
        )

    # ── 1. Stage transition ───────────────────────────
    proposed_stage = ai_decision.new_stage
    approved_stage, stage_warnings = validate_transition(
        current_stage, proposed_stage, ai_decision.vendor_happiness
    )
    warnings.extend(stage_warnings)

    # ── 2. Clamp vendor_happiness ─────────────────────
    clamped_happiness, was_clamped = clamp_delta(
        current_happiness, ai_decision.vendor_happiness, max_mood_delta
    )
    if was_clamped:
        warnings.append(
            f"vendor_happiness clamped: {ai_decision.vendor_happiness} → "
            f"{clamped_happiness} (current={current_happiness}, "
            f"max_delta=±{max_mood_delta})"
        )

    # ── 3. Clamp vendor_patience ──────────────────────
    clamped_patience, was_clamped = clamp_delta(
        current_patience, ai_decision.vendor_patience, max_mood_delta
    )
    if was_clamped:
        warnings.append(
            f"vendor_patience clamped: {ai_decision.vendor_patience} → "
            f"{clamped_patience} (current={current_patience}, "
            f"max_delta=±{max_mood_delta})"
        )

    # ── 4. Clamp new_mood ─────────────────────────────
    clamped_mood, was_clamped = clamp_delta(
        current_mood, ai_decision.new_mood, max_mood_delta
    )
    if was_clamped:
        warnings.append(
            f"new_mood clamped: {ai_decision.new_mood} → "
            f"{clamped_mood} (current={current_mood}, "
            f"max_delta=±{max_mood_delta})"
        )

    # ── 5. Derive vendor_mood from happiness ──────────
    derived_mood = derive_vendor_mood(clamped_happiness)

    # ── 6. Price sanity ───────────────────────────────
    price = ai_decision.price_offered if ai_decision.price_offered is not None else 0
    if price < 0:
        warnings.append(f"Negative price_offered={price} → clamped to 0")
        price = 0

    # ── 7. Terminal state detection ───────────────────
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
            final_price=price,
            vendor_happiness=clamped_happiness,
            vendor_patience=clamped_patience,
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
        new_mood=clamped_mood,
        new_stage=approved_stage,
        price_offered=price,
        vendor_happiness=clamped_happiness,
        vendor_patience=clamped_patience,
        vendor_mood=derived_mood,
        internal_reasoning=ai_decision.internal_reasoning,
        warnings=warnings,
        is_terminal=terminal,
    )
