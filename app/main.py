"""
Samvad XR Orchestration — FastAPI entry point (dev/testing only).

Architecture v3.0: Dev B owns the production API endpoint.
This FastAPI server is retained for Dev A's isolated development and
testing (health check, optional dev endpoint for testing
generate_vendor_response directly).

The primary deliverable is `app.generate.generate_vendor_response()`,
which Dev B imports and calls from their endpoint handler.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.exceptions import BrainServiceError, StateStoreError
from app.logging_config import setup_logging
from app.services.session_store import init_neo4j, close_neo4j

# ---------------------------------------------------------------------------
# Lifespan — runs once at startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown events for the FastAPI application."""
    settings: Settings = get_settings()
    setup_logging(settings.log_level)

    logger = logging.getLogger("samvadxr")
    logger.info(
        "Samvad XR Orchestration starting",
        extra={
            "step": "startup",
            "version": settings.app_version,
            "model": settings.openai_model,
            "mocks_enabled": settings.use_mocks,
        },
    )

    if settings.use_mocks:
        logger.warning(
            "Running with USE_MOCKS=true — all services are mocked",
            extra={"step": "startup"},
        )

    # ── Neo4j connection ─────────────────────────────
    neo4j_connected = False
    if not settings.use_mocks and settings.neo4j_password:
        try:
            await init_neo4j(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
                timeout_ms=settings.neo4j_timeout_ms,
            )
            neo4j_connected = True
            logger.info(
                "Neo4j connected via init_neo4j()",
                extra={"step": "startup", "uri": settings.neo4j_uri},
            )
        except Exception as exc:
            logger.error(
                "Neo4j connection failed — state persistence unavailable",
                extra={"step": "startup", "error": str(exc)},
            )
    elif settings.use_mocks:
        logger.warning(
            "USE_MOCKS=true — Neo4j disabled (using MockSessionStore)",
            extra={"step": "startup"},
        )
    else:
        logger.warning(
            "NEO4J_PASSWORD not set — Neo4j disabled (state will not be persisted)",
            extra={"step": "startup"},
        )

    yield  # app is running

    # ── Shutdown ─────────────────────────────────────
    if neo4j_connected:
        await close_neo4j()
        logger.info("Neo4j driver closed via close_neo4j()", extra={"step": "shutdown"})

    logger.info("Samvad XR Orchestration shutting down", extra={"step": "shutdown"})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Samvad XR Orchestration — Dev Server",
        description=(
            "Dev A's AI brain and state engine for Samvad XR. "
            "This server is for development/testing only. "
            "In production, Dev B's server imports generate_vendor_response() directly."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────
    origins = [o.strip() for o in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler — no raw 500s ever ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger = logging.getLogger("samvadxr")
        logger.error(
            "Unhandled exception",
            extra={"step": "unhandled_error", "error": str(exc)},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred.",
                "detail": None,
            },
        )

    # ── Health check ─────────────────────────────────
    @app.get("/health", tags=["system"])
    async def health_check() -> dict:
        """Lightweight health probe for dev server."""
        return {
            "status": "ok",
            "version": settings.app_version,
        }

    # ── Dev test endpoint ───────────────────────────────
    # POST /api/dev/generate wraps generate_vendor_response()
    # for isolated curl/Postman testing. Not for production use.
    # Only available when LOG_LEVEL=DEBUG.

    if settings.log_level.upper() == "DEBUG":

        class DevGenerateRequest(BaseModel):
            """Request body for the dev test endpoint."""

            transcribed_text: str = Field(
                default="",
                description="What the user said (from STT).",
            )
            context_block: str = Field(
                default="",
                description="Formatted conversation history.",
            )
            rag_context: str = Field(
                default="",
                description="Cultural/item knowledge from RAG.",
            )
            scene_context: dict[str, Any] = Field(
                default_factory=lambda: {
                    "items_in_hand": [],
                    "looking_at": None,
                    "distance_to_vendor": 1.0,
                    "vendor_npc_id": "vendor_01",
                    "vendor_happiness": 50,
                    "vendor_patience": 70,
                    "negotiation_stage": "BROWSING",
                    "current_price": 0,
                    "user_offer": 0,
                },
                description="Game state from Unity.",
            )
            session_id: str = Field(
                default="dev-test",
                description="Unique session identifier.",
            )

        @app.post("/api/dev/generate", tags=["dev"])
        async def dev_generate(payload: DevGenerateRequest) -> dict:
            """Dev-only endpoint to test generate_vendor_response() directly.

            Only available when LOG_LEVEL=DEBUG.
            Returns the validated VendorResponse dict, or an error JSON
            with the appropriate HTTP status code.
            """
            from app.generate import generate_vendor_response

            try:
                result = await generate_vendor_response(
                    transcribed_text=payload.transcribed_text,
                    context_block=payload.context_block,
                    rag_context=payload.rag_context,
                    scene_context=payload.scene_context,
                    session_id=payload.session_id,
                )
                return result
            except BrainServiceError as exc:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error_code": "BRAIN_SERVICE_ERROR",
                        "message": str(exc),
                    },
                )
            except StateStoreError as exc:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error_code": "STATE_STORE_ERROR",
                        "message": str(exc),
                    },
                )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (uvicorn points here: app.main:app)
# ---------------------------------------------------------------------------

app = create_app()
