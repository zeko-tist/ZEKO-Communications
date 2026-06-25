"""Base interfaces for LLM modules."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

# System prompt shared by all LLM backends.
SYSTEM_PROMPT = (
    "You are ZEKO, a humanoid robot assistant deployed at a college campus. "
    "You interact with students in their preferred language — usually a mix "
    "of Malayalam and English (Manglish). "
    "Rules:\n"
    "1. Keep responses short — maximum 2 sentences.\n"
    "2. Respond in the SAME language the user spoke. If they mixed Malayalam "
    "   and English, you mix too.\n"
    "3. Be natural, friendly, and helpful.\n"
    "4. Never use markdown, bullets, numbered lists, or emojis.\n"
    "5. Your output will be spoken aloud, so write for the ear, not the eye."
)


class BaseLLM(ABC):
    """Abstract base class for LLM backends (async streaming)."""

    @abstractmethod
    async def stream(
        self, user_text: str, language: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Generate a response token-by-token (async generator).

        Args:
            user_text: The user's transcribed speech.
            language: Detected language code ('ml', 'en', etc.).
            history: Previous conversation turns as dicts with 'role' and 'text'.

        Yields:
            Text chunks as they become available.
        """
        ...
        # Marker to make this an async generator for type checkers
        yield  # pragma: no cover

    @abstractmethod
    def preload(self) -> None:
        """Eagerly load model weights / initialize client."""
        ...

    @staticmethod
    async def _sync_gen_to_async(sync_gen, loop=None) -> AsyncIterator[str]:
        """Wrap a synchronous generator into an async one via a thread.

        Runs the sync generator in a thread pool and yields items as they
        arrive via an asyncio.Queue. This avoids blocking the event loop.
        """
        if loop is None:
            loop = asyncio.get_running_loop()

        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()
        _sentinel = object()

        def _consume():
            try:
                for item in sync_gen:
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _sentinel)

        future = loop.run_in_executor(None, _consume)

        while True:
            item = await queue.get()
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

        # Ensure the thread finished cleanly
        await future
