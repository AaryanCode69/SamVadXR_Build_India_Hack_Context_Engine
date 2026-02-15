"""
Pydantic models for inputs to generate_vendor_response().

SceneContext represents the game-state dict that Unity sends via Dev B.
Dev B forwards it unchanged — we parse and validate it here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.enums import MOOD_MAX, MOOD_MIN, NegotiationStage


class SceneContext(BaseModel):
    """Game state snapshot forwarded from Unity via Dev B's scene_context dict.

    All numeric scores (vendor_happiness, vendor_patience) are clamped to
    [0, 100] on ingestion so downstream logic never sees out-of-range values.
    """

    items_in_hand: list[str] = Field(
        default_factory=list,
        description="Items the user is currently holding in VR.",
    )
    looking_at: Optional[str] = Field(
        default=None,
        description="Item the user's gaze is focused on (gaze-tracked).",
    )
    distance_to_vendor: float = Field(
        default=1.0,
        ge=0.0,
        description="Physical proximity to the vendor NPC (metres).",
    )
    vendor_npc_id: str = Field(
        default="vendor_01",
        description="Identifier for which vendor NPC the user is interacting with.",
    )
    vendor_happiness: int = Field(
        default=50,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Vendor happiness score (0-100) from Unity.",
    )
    vendor_patience: int = Field(
        default=70,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Vendor patience score (0-100) from Unity.",
    )
    negotiation_stage: NegotiationStage = Field(
        default=NegotiationStage.BROWSING,
        description="Current negotiation stage reported by Unity.",
    )
    current_price: int = Field(
        default=0,
        ge=0,
        description="Vendor's current asking price.",
    )
    user_offer: int = Field(
        default=0,
        ge=0,
        description="User's latest counter-offer.",
    )

    # ── Validators ────────────────────────────────────────

    @field_validator("vendor_happiness", "vendor_patience", mode="before")
    @classmethod
    def clamp_score(cls, v: int) -> int:
        """Clamp happiness/patience scores to [0, 100] even if Unity sends garbage."""
        if isinstance(v, (int, float)):
            return max(MOOD_MIN, min(MOOD_MAX, int(v)))
        return v  # let Pydantic's type check handle non-numeric
