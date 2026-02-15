"""
Pydantic models for outputs from generate_vendor_response().

AIDecision  — internal model for parsing GPT-4o structured JSON output.
VendorResponse — validated output returned to Dev B as a plain dict.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.enums import (
    MOOD_MAX,
    MOOD_MIN,
    NegotiationStage,
    VendorMood,
)


class AIDecision(BaseModel):
    """Structured output demanded from GPT-4o (JSON mode).

    This is an INTERNAL model — never sent directly to Dev B.
    The state engine validates and clamps these values before they
    become a VendorResponse.
    """

    reply_text: str = Field(
        ...,
        min_length=1,
        description="What the vendor NPC says (native-script string).",
    )
    new_mood: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Proposed mood after this turn (0-100).",
    )
    new_stage: NegotiationStage = Field(
        ...,
        description="Proposed negotiation stage after this turn.",
    )
    price_offered: Optional[int] = Field(
        default=None,
        ge=0,
        description="Price the vendor is quoting. None if not quoting.",
    )
    vendor_happiness: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Proposed vendor happiness (0-100).",
    )
    vendor_patience: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Proposed vendor patience (0-100).",
    )
    vendor_mood: VendorMood = Field(
        ...,
        description="Categorical mood: enthusiastic|neutral|annoyed|angry.",
    )
    internal_reasoning: str = Field(
        default="",
        description="AI's reasoning for this decision (debug/logging only).",
    )

    # ── Validators ────────────────────────────────────────

    @field_validator("new_mood", "vendor_happiness", "vendor_patience", mode="before")
    @classmethod
    def clamp_score(cls, v: int) -> int:
        """Clamp numeric scores to [0, 100]."""
        if isinstance(v, (int, float)):
            return max(MOOD_MIN, min(MOOD_MAX, int(v)))
        return v


class VendorResponse(BaseModel):
    """Validated output returned to Dev B as a plain dict.

    The state engine constructs this after validating the AIDecision.
    We call .model_dump() and hand the dict to Dev B — no Pydantic
    coupling across the boundary.

    Schema must match the contract in INTEGRATION_GUIDE.md §2.1.
    """

    reply_text: str = Field(
        ...,
        min_length=1,
        description="Vendor's spoken response (native-script string).",
    )
    new_mood: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Validated mood (0-100, clamped ±15 from previous).",
    )
    new_stage: str = Field(
        ...,
        description="GREETING|BROWSING|HAGGLING|DEAL|WALKAWAY|CLOSURE",
    )
    price_offered: int = Field(
        ...,
        ge=0,
        description="Vendor's current asking price (0 if not quoting).",
    )
    vendor_happiness: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Vendor happiness (0-100).",
    )
    vendor_patience: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Vendor patience (0-100).",
    )
    vendor_mood: str = Field(
        ...,
        description="enthusiastic|neutral|annoyed|angry",
    )

    @field_validator("new_stage")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        """Ensure new_stage is a valid NegotiationStage value."""
        valid = {s.value for s in NegotiationStage}
        if v not in valid:
            raise ValueError(f"Invalid stage '{v}'. Must be one of {valid}")
        return v

    @field_validator("vendor_mood")
    @classmethod
    def validate_mood_category(cls, v: str) -> str:
        """Ensure vendor_mood is a valid VendorMood value."""
        valid = {m.value for m in VendorMood}
        if v not in valid:
            raise ValueError(f"Invalid vendor_mood '{v}'. Must be one of {valid}")
        return v
