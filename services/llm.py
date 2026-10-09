"""The single place the app talks to an LLM. Every RAG step (grade, rewrite, generate,
verify) calls `complete()`, which sends the prompt to Groq's OpenAI-compatible chat API."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from core.config import settings

logger = logging.getLogger(__name__)

# Errors worth retrying / falling back on: overloaded, rate-limited, timeouts, model gone.
_TRANSIENT_MARKERS = (
    "503", "overloaded", "over capacity", "service unavailable",
    "429", "rate limit", "rate_limit", "quota",
    "500", "502", "504", "internal", "timed out", "timeout",
    "404", "model_not_found", "decommissioned", "not found",  # retired model -> next one
    "ssl", "eof", "connection", "reset", "temporarily", "getaddrinfo",
)

# Rough per-1M-token pricing, used only to produce an estimate stored with each message.
# Groq's free tier costs nothing; these are approximate pay-as-you-go list prices.
_PRICING_PER_1M_TOKENS = {"groq": {"input": 0.10, "output": 0.50}}


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class LLMResult:
    content: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        rates = _PRICING_PER_1M_TOKENS.get(self.provider, {"input": 0.0, "output": 0.0})
        return (self.input_tokens * rates["input"] + self.output_tokens * rates["output"]) / 1_000_000


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq

    return Groq(
        api_key=settings.groq_api_key,
        max_retries=0,  # retries are handled below so we can also fall back to other models
        timeout=settings.groq_timeout_seconds,
    )


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _call(model: str, system_prompt: str, user_prompt: str) -> LLMResult:
    kwargs: dict = {}
    # gpt-oss models "think" before answering; keep it short so grading/verifying stays fast.
    if model.startswith("openai/gpt-oss"):
        kwargs["reasoning_effort"] = settings.groq_reasoning_effort

    response = _groq_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    usage = getattr(response, "usage", None)
    return LLMResult(
        content=(response.choices[0].message.content or ""),
        provider="groq",
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def complete(system_prompt: str, user_prompt: str) -> LLMResult:
    if not settings.groq_api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not set")

    last_exc: Exception | None = None

    # Try each model in order; retry transient errors with a short exponential backoff.
    for model in settings.groq_model_list:
        for attempt in range(settings.groq_max_retries + 1):
            try:
                return _call(model, system_prompt, user_prompt)
            except Exception as exc:  # noqa: BLE001 - surface any provider error as "unavailable"
                last_exc = exc
                if not _is_transient(exc):
                    raise LLMUnavailableError(f"groq ({model}): {exc}") from exc
                if attempt < settings.groq_max_retries:
                    delay = 0.5 * (2**attempt)  # 0.5s, 1s, ...
                    logger.warning(
                        "Groq %s transient error (attempt %d), retrying in %.1fs: %s",
                        model, attempt + 1, delay, exc,
                    )
                    time.sleep(delay)
        logger.warning("Groq model %s unavailable, trying next fallback model", model)

    raise LLMUnavailableError(f"groq: all models failed. Last error: {last_exc}") from last_exc
