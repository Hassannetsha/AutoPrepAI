from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    WINDOW_SECONDS = 60

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        # tracks timestamps of recent requests
        self._request_times: deque[float] = deque()
        
        # tracks timestamps and estimated token usage of recent requests.
        self._token_times: deque[tuple[float, int]] = deque()

    def acquire(self, estimated_tokens: int) -> None:
        while True:
            now = time.monotonic()

            # remove requests that are outside the 60-second window
            self._evict_expired(now)

            # Check whether both request and token budgets have capacity.
            rpm_ok = len(self._request_times) < self.requests_per_minute
            tpm_ok = self._tokens_used() + estimated_tokens <= self.tokens_per_minute

            # Reserve capacity and allow the API call to proceed.
            if rpm_ok and tpm_ok:
                self._request_times.append(now)
                self._token_times.append((now, estimated_tokens))
                return

            # Wait until enough budget becomes available.
            sleep_for = self._seconds_until_budget(now, tpm_ok)
            time.sleep(sleep_for)

    def _evict_expired(self, now: float) -> None:
        """Remove entries that have fallen outside the rolling window."""
        
        # Ignore requests older than the current 60-second window.
        cutoff = now - self.WINDOW_SECONDS
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_times and self._token_times[0][0] <= cutoff:
            self._token_times.popleft()

    def _tokens_used(self) -> int:
        # Compute the current token usage within the active window.
        return sum(t for _, t in self._token_times)

    def _seconds_until_budget(self, now: float, tpm_ok: bool) -> float:
        """
        Compute how long to sleep before *either* budget might free up.
        Returns at least 0.5 s to avoid tight spinning.
        """
        # Collect the remaining wait time for each exhausted budget.
        waits: list[float] = []
        if len(self._request_times) >= self.requests_per_minute and self._request_times:
            waits.append(self.WINDOW_SECONDS - (now - self._request_times[0]))
        if not tpm_ok and self._token_times:
            waits.append(self.WINDOW_SECONDS - (now - self._token_times[0][0]))
        
        # Sleep until the earliest budget becomes available.
        return max(0.5, min(waits) if waits else 1.0)