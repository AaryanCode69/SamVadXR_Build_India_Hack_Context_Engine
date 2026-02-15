"""
Pydantic models for inputs to generate_vendor_response().

SceneContext represents the game-state dict that Unity sends via Dev B.
Dev B forwards it unchanged — we parse and validate it here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.enums import MOOD_MAX, MOOD_MIN, LanguageCode, NegotiationStage


class SceneContext(BaseModel):
    """Game state snapshot forwarded from Unity via Dev B's scene_context dict.

    The happiness_score is clamped to [0, 100] on ingestion so downstream
    logic never sees out-of-range values.
    """

    object_grabbed: Optional[str] = Field(
        default=None,
        description="Item the user has grabbed / is interacting with.",
    )
    happiness_score: int = Field(
        default=50,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Vendor happiness score (0-100) from Unity.",
    )
    negotiation_state: NegotiationStage = Field(
        default=NegotiationStage.GREETING,
        description="Current negotiation stage reported by Unity.",
    )
    input_language: LanguageCode = Field(
        default=LanguageCode.EN_IN,
        description="Language the user is speaking.",
    )
    target_language: LanguageCode = Field(
        default=LanguageCode.EN_IN,
        description="Language the vendor should reply in.",
    )

    # ── Validators ────────────────────────────────────────

    @field_validator("happiness_score", mode="before")
    @classmethod
    def clamp_score(cls, v: int) -> int:
        """Clamp happiness score to [0, 100] even if Unity sends garbage."""
        if isinstance(v, (int, float)):
            return max(MOOD_MIN, min(MOOD_MAX, int(v)))
        return v  # let Pydantic's type check handle non-numeric
