"""
Application configuration — reads from environment variables.

Uses pydantic-settings to validate and type-check all config at startup.
If a required variable is missing, the app fails fast with a clear error.

Architecture v3.0: Dev A owns AI + State only. STT/TTS/RAG config belongs to Dev B.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for Dev A's AI brain and state engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore Dev B's vars (SARVAM_API_KEY, etc.) in shared .env
    )

    # ── Required secrets ──────────────────────────────────
    openai_api_key: str

    # ── AI model ──────────────────────────────────────────
    openai_model: str = "gpt-4o"
    ai_temperature: float = 0.7
    ai_max_tokens: int = 200

    # ── Feature flags ────────────────────────────────────
    # USE_MOCKS controls Dev A's own dependencies (OpenAI, Neo4j)
    # for isolated testing. Dev B's mocks (STT/TTS/RAG) are their concern.
    use_mocks: bool = False

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_timeout_ms: int = 2000

    # ── Timeouts (milliseconds) — Dev A's components only ─
    ai_timeout_ms: int = 10000

    # ── Game rules ───────────────────────────────────────
    max_turns: int = 30
    max_mood_delta: int = 15

    # ── Logging ──────────────────────────────────────────
    log_level: str = "INFO"

    # ── Server (dev/testing only) ────────────────────────
    app_version: str = "0.1.0"
    cors_origins: str = "*"  # comma-separated in production


def get_settings() -> Settings:
    """Create and return the validated settings instance.

    Raises:
        ValidationError: If required env vars are missing or invalid.
    """
    return Settings()  # type: ignore[call-arg]
