"""
Application configuration — reads from environment variables.

Uses pydantic-settings to validate and type-check all config at startup.
If a required variable is missing, the app fails fast with a clear error.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for the Samvad XR orchestration server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Required secrets ──────────────────────────────────
    openai_api_key: str
    sarvam_api_key: str

    # ── AI model ──────────────────────────────────────────
    openai_model: str = "gpt-4o"
    ai_temperature: float = 0.7
    ai_max_tokens: int = 200

    # ── Feature flags (mocks) ────────────────────────────
    use_mocks: bool = False
    use_mock_stt: bool = False
    use_mock_tts: bool = False
    use_mock_rag: bool = False

    # ── Neo4j ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_timeout_ms: int = 2000

    # ── Timeouts (milliseconds) ──────────────────────────
    stt_timeout_ms: int = 5000
    tts_timeout_ms: int = 5000
    rag_timeout_ms: int = 3000
    ai_timeout_ms: int = 10000

    # ── Game rules ───────────────────────────────────────
    max_turns: int = 30
    max_mood_delta: int = 15

    # ── Logging ──────────────────────────────────────────
    log_level: str = "INFO"

    # ── Server ───────────────────────────────────────────
    app_version: str = "0.1.0"
    cors_origins: str = "*"  # comma-separated in production

    def is_mock_stt(self) -> bool:
        """Whether to use mock STT (global flag overrides individual)."""
        return self.use_mocks or self.use_mock_stt

    def is_mock_tts(self) -> bool:
        """Whether to use mock TTS (global flag overrides individual)."""
        return self.use_mocks or self.use_mock_tts

    def is_mock_rag(self) -> bool:
        """Whether to use mock RAG (global flag overrides individual)."""
        return self.use_mocks or self.use_mock_rag


def get_settings() -> Settings:
    """Create and return the validated settings instance.

    Raises:
        ValidationError: If required env vars are missing or invalid.
    """
    return Settings()  # type: ignore[call-arg]
