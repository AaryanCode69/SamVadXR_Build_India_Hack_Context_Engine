"""
Samvad XR — Pydantic models (the single source of truth for data contracts).

Public API:
    from app.models import (
        NegotiationStage, VendorMood, LanguageCode,
        SceneContext, AIDecision, VendorResponse,
        LEGAL_TRANSITIONS, TERMINAL_STAGES,
    )
"""

from app.models.enums import (
    LEGAL_TRANSITIONS,
    MAX_MOOD_DELTA,
    MAX_TURNS,
    MOOD_MAX,
    MOOD_MIN,
    TERMINAL_STAGES,
    WRAP_UP_TURN_THRESHOLD,
    LanguageCode,
    NegotiationStage,
    VendorMood,
)
from app.models.request import SceneContext
from app.models.response import AIDecision, VendorResponse

__all__ = [
    # Enums
    "NegotiationStage",
    "VendorMood",
    "LanguageCode",
    # Models
    "SceneContext",
    "AIDecision",
    "VendorResponse",
    # Constants
    "LEGAL_TRANSITIONS",
    "TERMINAL_STAGES",
    "MOOD_MIN",
    "MOOD_MAX",
    "MAX_MOOD_DELTA",
    "MAX_TURNS",
    "WRAP_UP_TURN_THRESHOLD",
]