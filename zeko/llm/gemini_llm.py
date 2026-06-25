"""Online LLM using Gemini 2.5 Flash with async streaming.

Primary online brain — streams response tokens via the google-genai SDK.
On API error mid-stream, yields a short fallback phrase and logs — does
not crash the pipeline.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from .base import BaseLLM, SYSTEM_PROMPT


class GeminiLLM(BaseLLM):
    """Online LLM using Gemini 2.5 Flash with streaming responses.

    Uses BaseLLM._sync_gen_to_async to wrap the synchronous
    generate_content_stream into an async generator.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def preload(self) -> None:
        """Initialize the Gemini client."""
        self._get_client()
        print("✅ Gemini LLM client ready.")

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _build_contents(
        self, user_text: str, history: list[dict] | None
    ) -> list[dict]:
        """Build Gemini-native multi-turn contents list."""
        contents = []
        if history:
            for turn in history[-10:]:
                contents.append(
                    {
                        "role": turn["role"],
                        "parts": [{"text": turn["text"]}],
                    }
                )
        contents.append(
            {
                "role": "user",
                "parts": [{"text": user_text.strip()}],
            }
        )
        return contents

    def _sync_stream(self, user_text: str, language: str, history: list[dict] | None):
        """Synchronous streaming generator — will be wrapped to async."""
        client = self._get_client()
        from google.genai import errors as genai_errors

        lang_map = {"ml": "Malayalam", "en": "English", "hi": "Hindi"}
        target_lang = lang_map.get(language, "Malayalam")
        system_instruction = f"{SYSTEM_PROMPT}\n\nRespond in {target_lang}."

        contents = self._build_contents(user_text, history)

        for attempt in range(3):
            try:
                response_stream = client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config={"system_instruction": system_instruction},
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return  # Success — exit retry loop

            except genai_errors.APIError as exc:
                if "PERMISSION_DENIED" in str(exc):
                    print("❌ Gemini permission denied.")
                    yield "Gemini access is not enabled."
                    return
                if ("503" in str(exc) or "UNAVAILABLE" in str(exc)) and attempt < 2:
                    wait = 2**attempt
                    print(f"⚠️  Gemini unavailable, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"❌ Gemini API error: {exc}")
                yield "I could not reach Gemini just now."
                return
            except Exception as exc:
                print(f"❌ Unexpected Gemini error: {exc}")
                yield "I could not process that request."
                return

        yield "I could not reach Gemini after 3 attempts."

    async def stream(
        self, user_text: str, language: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Async stream of Gemini response chunks.

        Wraps the sync generate_content_stream in a thread to avoid
        blocking the event loop, enabling concurrent TTS synthesis.
        """
        sync_gen = self._sync_stream(user_text, language, history)
        async for chunk in self._sync_gen_to_async(sync_gen):
            yield chunk
