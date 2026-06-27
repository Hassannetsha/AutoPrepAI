# llm_client.py
"""
llm_client.py
=============
Abstract interface for LLM chat completion clients.

Any LLM provider (Groq, OpenAI, a local mock) must implement this
interface to be used by the standardization pipeline.

Why an interface here?
----------------------
The service currently hard-depends on Groq. Abstracting behind
`LLMClient` means:
  - Tests can inject a `MockLLMClient` with no network calls.
  - Swapping providers requires no changes to the pipeline.
  - Token estimation lives in one place, close to where it's used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class ChatMessage:
    """A single message in a chat conversation."""
    role: str      # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """Normalized response from any LLM provider."""
    content: str           # the assistant's reply text
    prompt_tokens: int     # tokens used by the input
    completion_tokens: int # tokens used by the output


class LLMClient(ABC):
    """
    Abstract base for LLM chat completion.

    Implementors are responsible for:
      - Rate limiting
      - Retries
      - Provider-specific serialization
    """

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ChatResponse:
        """
        Send a chat completion request and return a normalized response.

        Parameters
        ----------
        messages    : Conversation turns, in order.
        max_tokens  : Maximum tokens to generate.
        temperature : Sampling temperature (0.0 = deterministic).

        Raises
        ------
        TimeoutError
            If the rate limiter's max_wait_seconds is exceeded.
        RuntimeError
            If all retry attempts are exhausted.
        """

    @abstractmethod
    def estimate_tokens(self, messages: list[ChatMessage], max_tokens: int) -> int:
        """
        Estimate the total token cost of a request before sending it.

        Used by the rate limiter to pre-check budget. Does not need to
        be exact — a conservative over-estimate is preferable.
        """