"""
Samvad XR Orchestration — FastAPI entry point.

This is the central nervous system. It receives requests from the Unity
VR client and orchestrates the full pipeline: STT → Memory → RAG →
AI Brain → State Validation (Neo4j) → TTS → Response.
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
        title="Samvad XR Orchestration",
        description="Backend orchestration for the Samvad XR VR language immersion platform.",
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
        """Lightweight health probe — first thing Unity hits to verify connectivity."""
        return {
            "status": "ok",
            "version": settings.app_version,
        }

    # ── Interact endpoint (placeholder — built in Phase 3) ───
    # POST /api/interact will be added here.

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (uvicorn points here: app.main:app)
# ---------------------------------------------------------------------------

app = create_app()
