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

from app.dependencies import get_llm_service, get_session_store
from app.exceptions import BrainServiceError, StateStoreError  # noqa: F401 — re-export
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
    Internally it: parses scene_context → loads/creates Neo4j state →
    composes LLM prompt → calls GPT-4o → validates AI decision →
    persists state → returns validated dict.

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

    # ── 1. Parse & validate scene_context ────────────────
    parsed_scene = SceneContext.model_validate(scene_context)

    # ── 2. Load or create session state ──────────────────
    store = get_session_store()
    session_state = await store.load_session(session_id)
    if session_state is None:
        session_state = await store.create_session(session_id)
        logger.info(
            "New session created",
            extra={
                "step": "session_create",
                "request_id": request_id,
                "session_id": session_id,
            },
        )

    # Increment turn count
    session_state["turn_count"] = session_state.get("turn_count", 0) + 1

    # ── 3. Compose prompt & call LLM ─────────────────────
    # Phase 4 will build the real prompt composition.
    # For now, pass a minimal system prompt + user message.
    system_prompt = (
        "You are a street vendor in an Indian bazaar. "
        "Respond in character as a friendly but shrewd vendor."
    )
    user_message = (
        f"User says: {transcribed_text}\n"
        f"Context: {context_block}\n"
        f"Cultural info: {rag_context}\n"
        f"Scene: stage={parsed_scene.negotiation_stage.value}, "
        f"happiness={parsed_scene.vendor_happiness}, "
        f"patience={parsed_scene.vendor_patience}, "
        f"price={parsed_scene.current_price}"
    )

    llm = get_llm_service()
    ai_decision = await llm.generate_decision(
        system_prompt=system_prompt,
        user_message=user_message,
    )

    logger.debug(
        "LLM decision received",
        extra={
            "step": "llm_decision",
            "request_id": request_id,
            "new_stage": ai_decision.new_stage.value,
            "new_mood": ai_decision.new_mood,
            "reasoning": ai_decision.internal_reasoning,
        },
    )

    # ── 4. Validate & build response ─────────────────────
    # Phase 5 will add full state engine validation (clamp ±15, legal transitions).
    # For now, pass through the AI decision with basic validation.
    response = VendorResponse(
        reply_text=ai_decision.reply_text,
        new_mood=ai_decision.new_mood,
        new_stage=ai_decision.new_stage.value,
        price_offered=ai_decision.price_offered if ai_decision.price_offered is not None else 0,
        vendor_happiness=ai_decision.vendor_happiness,
        vendor_patience=ai_decision.vendor_patience,
        vendor_mood=ai_decision.vendor_mood.value,
    )

    # ── 5. Persist state ─────────────────────────────────
    session_state["vendor_happiness"] = response.vendor_happiness
    session_state["vendor_patience"] = response.vendor_patience
    session_state["negotiation_stage"] = response.new_stage
    session_state["current_price"] = response.price_offered
    if response.price_offered > 0:
        session_state.setdefault("price_history", []).append(response.price_offered)
    await store.save_session(session_id, session_state)

    elapsed_ms = (time.monotonic() - start_time) * 1000
    logger.info(
        "generate_vendor_response completed",
        extra={
            "step": "generate_complete",
            "request_id": request_id,
            "session_id": session_id,
            "duration_ms": round(elapsed_ms, 1),
            "turn_count": session_state["turn_count"],
            "stage": response.new_stage,
        },
    )

    return response.model_dump()
