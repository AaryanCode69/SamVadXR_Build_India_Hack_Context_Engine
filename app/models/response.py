"""
Pydantic models for outputs from generate_vendor_response().

AIDecision  — internal model for parsing GPT-4o structured JSON output.
VendorResponse — validated output returned to Dev B as a plain dict.
"""

from __future__ import annotations

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
    happiness_score: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Proposed happiness score after this turn (0-100).",
    )
    negotiation_state: NegotiationStage = Field(
        ...,
        description="Proposed negotiation stage after this turn.",
    )
    vendor_mood: VendorMood = Field(
        ...,
        description="Categorical mood: enthusiastic|friendly|neutral|annoyed|angry.",
    )
    internal_reasoning: str = Field(
        default="",
        description="AI's reasoning for this decision (debug/logging only).",
    )

    # ── Validators ────────────────────────────────────────

    @field_validator("happiness_score", mode="before")
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
    happiness_score: int = Field(
        ...,
        ge=MOOD_MIN,
        le=MOOD_MAX,
        description="Validated happiness score (0-100, clamped ±15 from previous).",
    )
    negotiation_state: str = Field(
        ...,
        description="GREETING|INQUIRY|HAGGLING|DEAL|WALKAWAY|CLOSURE",
    )
    vendor_mood: str = Field(
        ...,
        description="enthusiastic|friendly|neutral|annoyed|angry",
    )

    @field_validator("negotiation_state")
    @classmethod
    def validate_stage(cls, v: str) -> str:
        """Ensure negotiation_state is a valid NegotiationStage value."""
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
