"""
Enums and constants shared across the Samvad XR brain & rules engine.

These are CLOSED enums — adding a new value requires team discussion.
Pydantic validation rejects unknown enum values automatically.
"""

from enum import Enum


class NegotiationStage(str, Enum):
    """Finite stages of the vendor–buyer negotiation.

    Transition graph (see rules.md §5.1):
        GREETING  → BROWSING
        BROWSING  → HAGGLING | WALKAWAY
        HAGGLING  → DEAL | WALKAWAY | CLOSURE
        WALKAWAY  → HAGGLING (only if vendor_happiness > 40) | CLOSURE

    Terminal states: DEAL, CLOSURE
    """

    GREETING = "GREETING"
    BROWSING = "BROWSING"
    HAGGLING = "HAGGLING"
    DEAL = "DEAL"
    WALKAWAY = "WALKAWAY"
    CLOSURE = "CLOSURE"


class VendorMood(str, Enum):
    """Categorical mood descriptor returned alongside numeric scores.

    Derived from vendor_happiness:
        0-25   → angry
        26-45  → annoyed
        46-70  → neutral
        71-100 → enthusiastic
    """

    ENTHUSIASTIC = "enthusiastic"
    NEUTRAL = "neutral"
    ANNOYED = "annoyed"
    ANGRY = "angry"


class LanguageCode(str, Enum):
    """Sarvam-style BCP-47 language codes supported by the system."""

    HI_IN = "hi-IN"      # Hindi
    KN_IN = "kn-IN"      # Kannada
    TA_IN = "ta-IN"      # Tamil
    EN_IN = "en-IN"      # Indian English
    HI_EN = "hi-EN"      # Hinglish


# ── Legal state transitions ──────────────────────────────────
# Single source of truth — used by state_engine.py
# Key = current stage, Value = set of valid next stages
LEGAL_TRANSITIONS: dict[NegotiationStage, set[NegotiationStage]] = {
    NegotiationStage.GREETING: {NegotiationStage.BROWSING},
    NegotiationStage.BROWSING: {NegotiationStage.HAGGLING, NegotiationStage.WALKAWAY},
    NegotiationStage.HAGGLING: {
        NegotiationStage.DEAL,
        NegotiationStage.WALKAWAY,
        NegotiationStage.CLOSURE,
    },
    NegotiationStage.WALKAWAY: {NegotiationStage.HAGGLING, NegotiationStage.CLOSURE},
    NegotiationStage.DEAL: set(),      # terminal — no transitions out
    NegotiationStage.CLOSURE: set(),   # terminal — no transitions out
}

TERMINAL_STAGES: frozenset[NegotiationStage] = frozenset(
    {NegotiationStage.DEAL, NegotiationStage.CLOSURE}
)

# ── Numeric constraints ──────────────────────────────────────
MOOD_MIN: int = 0
MOOD_MAX: int = 100
MAX_MOOD_DELTA: int = 15          # per-turn clamp (also in config for override)
MAX_TURNS: int = 30               # hard limit per session
WRAP_UP_TURN_THRESHOLD: int = 25  # AI gets wrap-up instruction after this turn
