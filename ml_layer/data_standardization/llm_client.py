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
        """

    @abstractmethod
    def estimate_tokens(self, messages: list[ChatMessage], max_tokens: int) -> int:
        """
        Estimate the total token cost of a request before sending it.
        """