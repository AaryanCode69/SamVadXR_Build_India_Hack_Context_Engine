"""
Samvad XR — Primary Interface for Developer B.

This module exposes the ONE function Dev B calls from their pipeline:
    generate_vendor_response()

Architecture v3.0: Dev B owns the API endpoint. This function is called
at Step 7 of the pipeline, between Dev B's memory/RAG retrieval and
Dev B's TTS synthesis.

Usage by Dev B:
    from app.generate import generate_vendor_response

    result = await generate_vendor_response(
        transcribed_text="भाई ये silk scarf कितने का है?",
        context_block=context_block,
        rag_context=rag_context,
        scene_context=request.scene_context,
        session_id=request.session_id
    )
"""

import logging
import time
import uuid
from typing import Any

from app.exceptions import BrainServiceError, StateStoreError  # noqa: F401 — re-export
from app.models.enums import VendorMood
from app.models.request import SceneContext
from app.models.response import VendorResponse

logger = logging.getLogger("samvadxr")


async def generate_vendor_response(
    transcribed_text: str,
    context_block: str,
    rag_context: str,
    scene_context: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Generate the vendor's response using AI brain + Neo4j state validation.

    This is the single function Dev B calls at Step 7 of the pipeline.
    Internally it: parses scene_context → loads Neo4j state → composes
    LLM prompt → calls GPT-4o → validates AI decision → persists state.

    Args:
        transcribed_text: What the user said (from STT, Step 3).
        context_block: Formatted conversation history (from memory, Step 5).
        rag_context: Cultural/item knowledge (from RAG, Step 6). Can be "".
        scene_context: Game state from Unity (forwarded by Dev B). Contains
            items_in_hand, looking_at, distance_to_vendor, vendor_npc_id,
            vendor_happiness, vendor_patience, negotiation_stage,
            current_price, user_offer.
        session_id: Unique session identifier for Neo4j state lookup.

    Returns:
        Dict with validated vendor response:
        {
            "reply_text": str,          # Vendor's spoken response
            "new_mood": int,            # Validated mood (0-100, clamped ±15)
            "new_stage": str,           # GREETING|BROWSING|HAGGLING|DEAL|WALKAWAY|CLOSURE
            "price_offered": int,       # Vendor's current asking price
            "vendor_happiness": int,    # 0-100
            "vendor_patience": int,     # 0-100
            "vendor_mood": str          # enthusiastic|neutral|annoyed|angry
        }

    Raises:
        BrainServiceError: If LLM call fails after retries.
        StateStoreError: If Neo4j is unreachable.
    """
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    logger.info(
        "generate_vendor_response called",
        extra={
            "step": "generate_start",
            "request_id": request_id,
            "session_id": session_id,
            "transcribed_text_length": len(transcribed_text),
        },
    )

    # ── Parse & validate scene_context ───────────────────
    parsed_scene = SceneContext.model_validate(scene_context)

    # ── Implementation in Phase 3-5 ──────────────────────
    # Phase 3: Wire up the function skeleton
    # Phase 4: Add AI Brain (GPT-4o prompt + parsing)
    # Phase 5: Add state validation + Neo4j persistence
    #
    # For now, return a mock response validated through VendorResponse:
    response = VendorResponse(
        reply_text="Haan ji, ek minute... (placeholder response)",
        new_mood=parsed_scene.vendor_happiness,
        new_stage=parsed_scene.negotiation_stage.value,
        price_offered=parsed_scene.current_price,
        vendor_happiness=parsed_scene.vendor_happiness,
        vendor_patience=parsed_scene.vendor_patience,
        vendor_mood=VendorMood.NEUTRAL.value,
    )

    elapsed_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "generate_vendor_response completed",
        extra={
            "step": "generate_complete",
            "request_id": request_id,
            "session_id": session_id,
            "duration_ms": round(elapsed_ms, 1),
        },
    )

    return response.model_dump()
