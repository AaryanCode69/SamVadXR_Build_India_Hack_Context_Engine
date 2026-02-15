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

    v6.0 additions: counter_price and offer_assessment are optional fields
    that force the LLM to reason about pricing explicitly. They are used
    internally by the state engine for validation but are NOT forwarded
    to Dev B.
    """

    reply_text: str = Field(
        ...,
        min_length=1,
        description="What the vendor NPC says (always in English, phrased for easy translation to target language).",
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
    counter_price: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "The price the vendor is quoting or counter-offering this turn. "
            "None if no price is being discussed (e.g. GREETING stage). "
            "Used internally for price-consistency validation."
        ),
    )
    offer_assessment: Optional[str] = Field(
        default=None,
        description=(
            "Vendor's assessment of the customer's offer: "
            "'insult' (<25% of quoted), 'lowball' (25-40%), "
            "'fair' (40-60%), 'good' (60-75%), 'excellent' (>75%), "
            "or 'none' if no offer was made. Forces LLM to reason "
            "about offer quality before responding."
        ),
    )

    # ── Validators ────────────────────────────────────────

    @field_validator("happiness_score", mode="before")
    @classmethod
    def clamp_score(cls, v: int) -> int:
        """Clamp numeric scores to [0, 100]."""
        if isinstance(v, (int, float)):
            return max(MOOD_MIN, min(MOOD_MAX, int(v)))
        return v

    @field_validator("offer_assessment", mode="before")
    @classmethod
    def validate_offer_assessment(cls, v: Optional[str]) -> Optional[str]:
        """Validate offer_assessment is one of the allowed categories."""
        if v is None:
            return v
        allowed = {"insult", "lowball", "fair", "good", "excellent", "none"}
        if v.lower() not in allowed:
            return "none"  # graceful fallback — don't crash on LLM quirks
        return v.lower()


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
        description="Vendor's spoken response (always in English, phrased for easy translation to target language).",
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
