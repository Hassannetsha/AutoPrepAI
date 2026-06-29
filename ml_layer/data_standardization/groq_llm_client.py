# groq_llm_client.py
"""
groq_llm_client.py
==================
Groq implementation of LLMClient.

Wraps the Groq SDK, handles retries with exponential backoff,
and delegates rate limiting to RateLimiter.

Usage
-----
    from groq import Groq
    from rate_limiter import RateLimiter
    from groq_llm_client import GroqLLMClient

    client = GroqLLMClient(
        groq_client=Groq(api_key="..."),
        model="llama3-8b-8192",
        rate_limiter=RateLimiter(
            requests_per_minute=20,
            tokens_per_minute=30_000,
        ),
    )
    response = client.complete([ChatMessage(role="user", content="Hello")])
"""

from __future__ import annotations

import time

from groq import Groq

from .llm_client import ChatMessage, ChatResponse, LLMClient
from .rate_limiter import RateLimiter


class GroqLLMClient(LLMClient):
    """
    Groq-backed LLM client with rate limiting and retries.

    Parameters
    ----------
    groq_client : Groq
        Authenticated Groq SDK instance.
    model : str
        Model identifier (e.g. "llama3-8b-8192").
    rate_limiter : RateLimiter
        Shared rate limiter instance. Can be shared across agents
        if they all hit the same Groq account.
    max_retries : int
        Number of retry attempts on rate-limit errors (429).
        Uses exponential backoff starting at 2 s, capped at 30 s.
    """

    def __init__(
        self,
        groq_client: Groq,
        model: str,
        rate_limiter: RateLimiter,
        max_retries: int = 5,
    ) -> None:
        self._client = groq_client
        self._model = model
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ChatResponse:
        estimated = self.estimate_tokens(messages, max_tokens)
        payload = [{"role": m.role, "content": m.content} for m in messages]

        delay = 2.0
        for attempt in range(self._max_retries + 1):
            self._rate_limiter.acquire(estimated)
            try:
                raw = self._client.chat.completions.create(
                    model=self._model,
                    messages=payload,
                    temperature=temperature,
                )
                return ChatResponse(
                    content=raw.choices[0].message.content,
                    prompt_tokens=raw.usage.prompt_tokens,
                    completion_tokens=raw.usage.completion_tokens,
                )
            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt == self._max_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        raise RuntimeError("Unreachable")  # loop always raises or returns

    def estimate_tokens(self, messages: list[ChatMessage], max_tokens: int) -> int:
        """
        Rough estimation: 1 token ≈ 4 characters of input + completion budget.
        Conservative over-estimate is intentional.
        """
        input_chars = sum(len(m.content) for m in messages)
        return max(1, input_chars // 4) + max_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "rate limit" in msg or "429" in msg or "too many requests" in msg