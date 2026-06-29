"""Connectivity checker with async aiohttp and TTL cache.

Pings the Gemini API endpoint to determine if the system should use
online or offline modules. Non-blocking via aiohttp. On ping failure,
defaults to offline — never raises.
"""

from __future__ import annotations

import time

import aiohttp


class ConnectivityChecker:
    """Async connectivity checker with TTL-cached status.

    Uses aiohttp HEAD request to verify Gemini reachability.
    """

    def __init__(self, check_url: str, ttl_seconds: int = 5) -> None:
        self.check_url = check_url
        self.ttl_seconds = ttl_seconds
        self._last_check: float = 0.0
        self._cached_status: bool = False

    async def is_online(self) -> bool:
        """Return True if the internet is reachable. Result is cached for TTL seconds."""
        now = time.monotonic()
        if now - self._last_check < self.ttl_seconds:
            return self._cached_status

        self._cached_status = await self._ping()
        self._last_check = now
        return self._cached_status

    async def _ping(self) -> bool:
        """Attempt a non-blocking HTTP HEAD request to the check URL."""
        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.head(self.check_url) as resp:
                    return resp.status < 500
        except Exception:
            return False

    async def force_refresh(self) -> bool:
        """Force a fresh connectivity check, bypassing the cache."""
        self._last_check = 0.0
        return await self.is_online()
