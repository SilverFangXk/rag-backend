"""Tiny in-memory sliding-window rate limiter (per client IP).

Why: the endpoint is public (a browser must call it) and every question costs money.
Limits: it is per server instance (serverless hosts may run several), so the real limit is
looser. Treat it as a speed bump; the hard cap on cost is the spend limit you set in the
Anthropic console.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max, self.window = max_requests, window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> int:
        """Returns 0 if allowed, otherwise the seconds to wait."""
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.max:
            return max(1, int(self.window - (now - q[0])))
        q.append(now)
        if len(self._hits) > 5000:  # keep memory bounded
            for k in [k for k, v in self._hits.items() if not v][:1000]:
                self._hits.pop(k, None)
        return 0
