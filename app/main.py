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
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from neo4j import AsyncGraphDatabase

from app.config import Settings, get_settings
from app.logging_config import setup_logging

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
    neo4j_driver = None
    if settings.neo4j_password:
        try:
            neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            await neo4j_driver.verify_connectivity()
            logger.info(
                "Neo4j connected",
                extra={"step": "startup", "uri": settings.neo4j_uri},
            )
        except Exception as exc:
            logger.error(
                "Neo4j connection failed — state persistence unavailable",
                extra={"step": "startup", "error": str(exc)},
            )
            neo4j_driver = None
    else:
        logger.warning(
            "NEO4J_PASSWORD not set — Neo4j disabled (state will not be persisted)",
            extra={"step": "startup"},
        )

    # Store driver on app.state for dependency injection in later phases
    app.state.neo4j_driver = neo4j_driver

    yield  # app is running

    # ── Shutdown ─────────────────────────────────────
    if neo4j_driver is not None:
        await neo4j_driver.close()
        logger.info("Neo4j driver closed", extra={"step": "shutdown"})

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

    # ── Dev test endpoint (placeholder — built in Phase 3) ───
    # POST /api/dev/generate wraps generate_vendor_response()
    # for isolated curl/Postman testing. Not for production use.
    @app.post("/api/dev/generate", tags=["dev"])
    async def dev_generate(payload: dict) -> dict:
        """Dev-only endpoint to test generate_vendor_response() directly.

        Expects JSON with: transcribed_text, context_block, rag_context,
        scene_context (dict), session_id.
        """
        from app.generate import generate_vendor_response

        result = await generate_vendor_response(
            transcribed_text=payload.get("transcribed_text", ""),
            context_block=payload.get("context_block", ""),
            rag_context=payload.get("rag_context", ""),
            scene_context=payload.get("scene_context", {}),
            session_id=payload.get("session_id", "dev-test"),
        )
        return result

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (uvicorn points here: app.main:app)
# ---------------------------------------------------------------------------

app = create_app()
