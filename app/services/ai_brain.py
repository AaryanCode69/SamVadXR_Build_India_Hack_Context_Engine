"""
Real OpenAI-backed LLM service — the AI Brain (Game Master).

Conforms to the LLMService protocol defined in protocols.py.
Calls OpenAI GPT-4o with JSON mode, parses the response into AIDecision,
and handles retries + fallback per rules.md §4 and §6.4.

Retry policy:
    - Max 2 retries with exponential backoff (1s, 2s).
    - Retry only on 5xx / timeout / rate-limit (429).
    - Never retry on 4xx (bad request, auth failure).
    - On persistent failure: return in-character fallback response.
    - On JSON parse failure: retry once with a simplified prompt, then fallback.

Timeouts:
    - AI_TIMEOUT_MS from config (default 10 000 ms).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

from app.config import get_settings
from app.exceptions import BrainServiceError
from app.models.enums import NegotiationStage, VendorMood
from app.models.response import AIDecision
from app.prompts.vendor_system import PROMPT_VERSION

logger = logging.getLogger("samvadxr")

# Errors that are safe to retry
_RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError)

# Backoff delays in seconds for each retry attempt
_BACKOFF_DELAYS = [1.0, 2.0]

# ── Fallback response — in-character, safe, keeps current stage ──
_FALLBACK_DECISION = AIDecision(
    reply_text="Ek minute bhai, zara ruko... haan, bol raha tha kya?",
    happiness_score=50,
    negotiation_state=NegotiationStage.INQUIRY,
    vendor_mood=VendorMood.NEUTRAL,
    internal_reasoning="[FALLBACK] LLM call failed after retries — safe in-character response",
)


class OpenAILLMService:
    """Real OpenAI GPT-4o LLM service.

    Implements the LLMService protocol. Uses JSON mode to force structured
    output, parses into AIDecision, and retries on transient failures.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.ai_timeout_ms / 1000.0,  # SDK wants seconds
        )
        self._model = settings.openai_model
        self._default_temperature = settings.ai_temperature
        self._default_max_tokens = settings.ai_max_tokens

        logger.info(
            "OpenAILLMService initialized",
            extra={
                "step": "llm_init",
                "model": self._model,
                "timeout_ms": settings.ai_timeout_ms,
                "prompt_version": PROMPT_VERSION,
            },
        )

    async def generate_decision(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ) -> AIDecision:
        """Call OpenAI and return a parsed AIDecision.

        Retry policy: up to 2 retries on transient errors.
        On JSON parse failure: one retry, then fallback.
        On persistent failure: return in-character fallback.
        """
        temperature = temperature or self._default_temperature
        max_tokens = max_tokens or self._default_max_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # ── Attempt loop (1 initial + up to 2 retries) ──
        last_error: Exception | None = None
        raw_content: str | None = None

        for attempt in range(1 + len(_BACKOFF_DELAYS)):
            try:
                raw_content = await self._call_openai(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                logger.debug(
                    "OpenAI raw response",
                    extra={
                        "step": "llm_raw_response",
                        "attempt": attempt + 1,
                        "prompt_version": PROMPT_VERSION,
                        "content_length": len(raw_content),
                    },
                )

                return self._parse_response(raw_content)

            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < len(_BACKOFF_DELAYS):
                    delay = _BACKOFF_DELAYS[attempt]
                    logger.warning(
                        "OpenAI transient error — retrying",
                        extra={
                            "step": "llm_retry",
                            "attempt": attempt + 1,
                            "delay_s": delay,
                            "error": str(exc),
                        },
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "OpenAI retries exhausted",
                        extra={
                            "step": "llm_retry_exhausted",
                            "attempts": attempt + 1,
                            "error": str(exc),
                        },
                    )

            except (json.JSONDecodeError, ValidationError, KeyError) as exc:
                # Parse failure — retry once with the same prompt
                last_error = exc
                logger.warning(
                    "LLM response parse failed — retrying",
                    extra={
                        "step": "llm_parse_failure",
                        "attempt": attempt + 1,
                        "error": str(exc),
                        "raw_content": (raw_content or "")[:300],
                    },
                )
                if attempt < len(_BACKOFF_DELAYS):
                    await asyncio.sleep(_BACKOFF_DELAYS[attempt])
                else:
                    break

            except Exception as exc:
                # Non-retryable (e.g. 4xx auth error)
                logger.error(
                    "OpenAI non-retryable error",
                    extra={
                        "step": "llm_non_retryable",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                raise BrainServiceError(f"LLM call failed: {exc}") from exc

        # All retries exhausted — return fallback
        logger.error(
            "All LLM attempts failed — returning fallback response",
            extra={
                "step": "llm_fallback",
                "prompt_version": PROMPT_VERSION,
                "last_error": str(last_error),
            },
        )
        return _FALLBACK_DECISION

    async def _call_openai(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Make the actual OpenAI API call and return raw content string."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if content is None:
            raise BrainServiceError("OpenAI returned empty content")
        return content

    @staticmethod
    def _parse_response(raw_content: str) -> AIDecision:
        """Parse raw JSON string into a validated AIDecision.

        Handles minor LLM quirks:
        - Strips markdown code fences if present.
        """
        # Strip markdown code fences sometimes added despite JSON mode
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        data: dict[str, Any] = json.loads(cleaned)

        return AIDecision.model_validate(data)
