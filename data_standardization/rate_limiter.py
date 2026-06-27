# rate_limiter.py
"""
rate_limiter.py
===============
Token-bucket–style rate limiter for API calls.

Tracks two independent budgets over a rolling 60-second window:
  - requests per minute (RPM)
  - tokens per minute (TPM)

Usage
-----
    limiter = RateLimiter(requests_per_minute=20, tokens_per_minute=30_000)
    limiter.acquire(estimated_tokens=512)   # blocks until budget is available
"""

from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    """
    Sliding-window rate limiter for LLM API calls.

    Both budgets are enforced over a rolling 60-second window.
    `acquire()` blocks (sleeps) until both budgets have capacity,
    then registers the call before returning.

    Parameters
    ----------
    requests_per_minute : int
        Maximum number of API calls in any 60-second window.
    tokens_per_minute : int
        Maximum number of estimated tokens in any 60-second window.
    """

    WINDOW_SECONDS = 60

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # Each entry: timestamp of the call
        self._request_times: deque[float] = deque()
        # Each entry: (timestamp, estimated_tokens)
        self._token_times: deque[tuple[float, int]] = deque()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def acquire(self, estimated_tokens: int) -> None:
        """
        Block until both RPM and TPM budgets have room, then register
        the call. Always call this before making an API request.

        Parameters
        ----------
        estimated_tokens : int
            Estimated token cost of the upcoming request + completion.
        """
        while True:
            now = time.monotonic()
            self._evict_expired(now)

            rpm_ok = len(self._request_times) < self.requests_per_minute
            tpm_ok = self._tokens_used() + estimated_tokens <= self.tokens_per_minute

            if rpm_ok and tpm_ok:
                self._request_times.append(now)
                self._token_times.append((now, estimated_tokens))
                return

            sleep_for = self._seconds_until_budget(now, tpm_ok)
            time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self, now: float) -> None:
        """Remove entries that have fallen outside the rolling window."""
        cutoff = now - self.WINDOW_SECONDS
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_times and self._token_times[0][0] <= cutoff:
            self._token_times.popleft()

    def _tokens_used(self) -> int:
        return sum(t for _, t in self._token_times)

    def _seconds_until_budget(self, now: float, tpm_ok: bool) -> float:
        """
        Compute how long to sleep before *either* budget might free up.
        Returns at least 0.5 s to avoid tight spinning.
        """
        waits: list[float] = []
        if len(self._request_times) >= self.requests_per_minute and self._request_times:
            waits.append(self.WINDOW_SECONDS - (now - self._request_times[0]))
        if not tpm_ok and self._token_times:
            waits.append(self.WINDOW_SECONDS - (now - self._token_times[0][0]))
        return max(0.5, min(waits) if waits else 1.0)