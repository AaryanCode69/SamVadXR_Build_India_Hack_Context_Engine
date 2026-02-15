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
        transcribed_text="Bhaiya ye tomato kitne ka hai?",
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

from pydantic import ValidationError

from app.config import get_settings
from app.dependencies import get_llm_service, get_session_store
from app.exceptions import BrainServiceError, StateStoreError  # noqa: F401 — re-export
from app.models.enums import (
    MAX_TURNS,
    TERMINAL_STAGES,
    WRAP_UP_TURN_THRESHOLD,
    NegotiationStage,
)
from app.models.request import SceneContext
from app.models.response import VendorResponse
from app.prompts.vendor_system import (
    PROMPT_VERSION,
    build_system_prompt,
    build_user_message,
)
from app.services.state_engine import validate_ai_decision

logger = logging.getLogger("samvadxr")


async def generate_vendor_response(
    transcribed_text: str,
    context_block: str,
    rag_context: str,
    scene_context: dict[str, Any],
    session_id: str,
) -> dict[str, Any]:
    """Generate the vendor's response using AI brain + state validation.

    This is the single function Dev B calls at Step 7 of the pipeline.
    Pipeline: parse scene → load/create session → compose prompt →
    call LLM → validate → persist state → return dict.

    Args:
        transcribed_text: What the user said (from STT, Step 3).
        context_block: Formatted conversation history (from memory, Step 5).
        rag_context: Cultural/item knowledge (from RAG, Step 6). Can be "".
        scene_context: Game state from Unity (forwarded by Dev B). Contains
            object_grabbed, happiness_score, negotiation_state,
            input_language, target_language.
        session_id: Unique session identifier for state lookup.

    Returns:
        Dict with validated vendor response:
        {
            "reply_text": str,
            "happiness_score": int,
            "negotiation_state": str,
            "vendor_mood": str
        }

    Raises:
        BrainServiceError: If LLM call fails after retries.
        StateStoreError: If session store is unreachable.
    """
    request_id = str(uuid.uuid4())
    t_start = time.monotonic()

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
    t0 = time.monotonic()
    try:
        parsed_scene = SceneContext.model_validate(scene_context)
    except ValidationError as exc:
        logger.error(
            "Invalid scene_context",
            extra={
                "step": "scene_parse",
                "request_id": request_id,
                "error": str(exc),
            },
        )
        raise BrainServiceError(f"Invalid scene_context: {exc}") from exc

    logger.debug(
        "Scene context parsed",
        extra={
            "step": "scene_parse",
            "request_id": request_id,
            "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            "stage": parsed_scene.negotiation_state.value,
        },
    )

    # ── 2. Load or create session state ──────────────────
    t0 = time.monotonic()
    try:
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
    except (StateStoreError, BrainServiceError):
        raise
    except Exception as exc:
        logger.error(
            "Session store error",
            extra={
                "step": "session_load",
                "request_id": request_id,
                "error": str(exc),
            },
        )
        raise StateStoreError(f"Failed to access session store: {exc}") from exc

    logger.debug(
        "Session state loaded",
        extra={
            "step": "session_load",
            "request_id": request_id,
            "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            "turn_count": session_state.get("turn_count", 0),
        },
    )

    # ── 2½. Check terminal state ─────────────────────────
    current_stage_str = session_state.get(
        "negotiation_state", parsed_scene.negotiation_state.value
    )
    try:
        current_stage = NegotiationStage(current_stage_str)
    except ValueError:
        current_stage = parsed_scene.negotiation_state

    if current_stage in TERMINAL_STAGES:
        logger.warning(
            "Session in terminal state — returning closure response",
            extra={
                "step": "terminal_check",
                "request_id": request_id,
                "session_id": session_id,
                "stage": current_stage.value,
            },
        )
        closure_reply = (
            "Yeh negotiation toh khatam ho chuki hai, bhai. Phir milenge!"
            if current_stage == NegotiationStage.DEAL
            else "Bhai, baat toh khatam ho gayi. Phir kabhi aana!"
        )
        response = VendorResponse(
            reply_text=closure_reply,
            happiness_score=session_state.get("happiness_score", 50),
            negotiation_state=current_stage.value,
            vendor_mood="neutral",
        )
        return response.model_dump()

    # ── 2¾. Increment turn count & enforce limits ────────
    turn_count = session_state.get("turn_count", 0) + 1
    session_state["turn_count"] = turn_count

    force_closure = turn_count > MAX_TURNS

    if force_closure:
        logger.warning(
            "Turn limit exceeded — forcing CLOSURE",
            extra={
                "step": "turn_limit",
                "request_id": request_id,
                "session_id": session_id,
                "turn_count": turn_count,
            },
        )
        response = VendorResponse(
            reply_text="Bahut der ho gayi bhai! Dukaan band karna hai. Phir aana!",
            happiness_score=max(session_state.get("happiness_score", 50) - 10, 0),
            negotiation_state=NegotiationStage.CLOSURE.value,
            vendor_mood="annoyed",
        )
        # Persist forced closure
        session_state["negotiation_state"] = NegotiationStage.CLOSURE.value
        try:
            await store.save_session(session_id, session_state)
        except Exception:
            pass  # best-effort persist on forced closure
        return response.model_dump()

    # ── 3. Compose prompt & call LLM ─────────────────────
    t0 = time.monotonic()
    settings = get_settings()

    # Use authoritative state from session store, fall back to scene for first turn
    effective_happiness = session_state.get(
        "happiness_score", parsed_scene.happiness_score
    )
    effective_stage = session_state.get(
        "negotiation_state", parsed_scene.negotiation_state.value
    )

    system_prompt = build_system_prompt(
        happiness_score=effective_happiness,
        negotiation_state=effective_stage,
        turn_count=turn_count,
        object_grabbed=parsed_scene.object_grabbed,
        input_language=parsed_scene.input_language.value,
        target_language=parsed_scene.target_language.value,
        wrap_up=turn_count >= WRAP_UP_TURN_THRESHOLD,
    )

    user_message = build_user_message(
        transcribed_text=transcribed_text,
        context_block=context_block,
        rag_context=rag_context,
    )

    logger.debug(
        "Prompt composed",
        extra={
            "step": "prompt_compose",
            "request_id": request_id,
            "prompt_version": PROMPT_VERSION,
            "system_prompt_length": len(system_prompt),
            "user_message_length": len(user_message),
        },
    )

    try:
        llm = get_llm_service()
        ai_decision = await llm.generate_decision(
            system_prompt=system_prompt,
            user_message=user_message,
        )
    except BrainServiceError:
        raise
    except Exception as exc:
        logger.error(
            "LLM call failed",
            extra={
                "step": "llm_call",
                "request_id": request_id,
                "error": str(exc),
            },
        )
        raise BrainServiceError(f"LLM service failed: {exc}") from exc

    llm_ms = round((time.monotonic() - t0) * 1000, 1)
    logger.info(
        "LLM decision received",
        extra={
            "step": "llm_decision",
            "request_id": request_id,
            "duration_ms": llm_ms,
            "negotiation_state": ai_decision.negotiation_state.value,
            "happiness_score": ai_decision.happiness_score,
            "reasoning": ai_decision.internal_reasoning,
        },
    )

    # ── 4. Validate via State Engine ───────────────────────
    t0 = time.monotonic()

    validated = validate_ai_decision(
        ai_decision=ai_decision,
        session_state=session_state,
        max_mood_delta=settings.max_mood_delta,
    )

    if validated.warnings:
        logger.warning(
            "State engine overrides applied",
            extra={
                "step": "state_validate",
                "request_id": request_id,
                "overrides": validated.warnings,
            },
        )

    try:
        response = VendorResponse(
            reply_text=validated.reply_text,
            happiness_score=validated.happiness_score,
            negotiation_state=validated.negotiation_state.value,
            vendor_mood=validated.vendor_mood.value,
        )
    except ValidationError as exc:
        logger.error(
            "Validated decision failed Pydantic check",
            extra={
                "step": "response_validate",
                "request_id": request_id,
                "error": str(exc),
            },
        )
        raise BrainServiceError(
            f"State-validated decision is invalid: {exc}"
        ) from exc

    logger.debug(
        "Response validated",
        extra={
            "step": "response_validate",
            "request_id": request_id,
            "duration_ms": round((time.monotonic() - t0) * 1000, 1),
            "is_terminal": validated.is_terminal,
        },
    )

    # ── 5. Persist state ─────────────────────────────────
    t0 = time.monotonic()

    session_state["happiness_score"] = response.happiness_score
    session_state["negotiation_state"] = response.negotiation_state

    try:
        await store.save_session(session_id, session_state)
    except (StateStoreError, BrainServiceError):
        raise
    except Exception as exc:
        logger.error(
            "Failed to persist session state",
            extra={
                "step": "session_save",
                "request_id": request_id,
                "error": str(exc),
            },
        )
        raise StateStoreError(f"Failed to persist state: {exc}") from exc

    logger.debug(
        "Session state persisted",
        extra={
            "step": "session_save",
            "request_id": request_id,
            "duration_ms": round((time.monotonic() - t0) * 1000, 1),
        },
    )

    # ── 6. Done ──────────────────────────────────────────
    total_ms = round((time.monotonic() - t_start) * 1000, 1)
    logger.info(
        "generate_vendor_response completed",
        extra={
            "step": "generate_complete",
            "request_id": request_id,
            "session_id": session_id,
            "duration_ms": total_ms,
            "llm_ms": llm_ms,
            "turn_count": turn_count,
            "stage": response.negotiation_state,
        },
    )

    return response.model_dump()
