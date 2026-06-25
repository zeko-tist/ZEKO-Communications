"""Online STT using Gemini 2.5 Flash multimodal audio input.

Sends raw audio bytes to Gemini with a system prompt tuned for
Manglish transcription. On 429 rate limit or timeout, uses
exponential backoff (2 retries), then falls back to WhisperSTT.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from .base import BaseSTT, TranscriptionResult


_TRANSCRIPTION_PROMPT = (
    "Transcribe this audio exactly. The speaker uses Malayalam "
    "mixed with English (Manglish). Output in the original script."
)


class GeminiSTT(BaseSTT):
    """Online STT engine using Gemini 2.5 Flash multimodal.

    Falls back to a WhisperSTT instance on 429/timeout after 2 retries.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        whisper_fallback: BaseSTT | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.whisper_fallback = whisper_fallback
        self._client = None

    def preload(self) -> None:
        """Initialize the Gemini client."""
        self._get_client()
        print("✅ Gemini STT client ready.")

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio via Gemini multimodal API.

        On 429/timeout: exponential backoff (2 retries), then falls back
        to WhisperSTT. Never crashes the pipeline.
        """
        client = self._get_client()
        audio_bytes = await asyncio.to_thread(Path(audio_path).read_bytes)

        from google import genai
        from google.genai import types

        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model_name,
                    contents=[
                        types.Content(
                            parts=[
                                types.Part.from_bytes(
                                    data=audio_bytes,
                                    mime_type="audio/wav",
                                ),
                                types.Part.from_text(text=_TRANSCRIPTION_PROMPT),
                            ]
                        )
                    ],
                )
                text = response.text.strip() if response.text else ""
                # Gemini STT doesn't provide language detection natively,
                # so we default to "ml" (Malayalam) since that's the primary use.
                return TranscriptionResult(
                    text=text, language="ml", confidence=0.85
                )

            except Exception as exc:
                exc_str = str(exc)
                if ("429" in exc_str or "UNAVAILABLE" in exc_str) and attempt < 2:
                    wait = 2**attempt
                    print(f"⚠️  Gemini STT rate-limited, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                # Exhausted retries or non-retryable error → fall back
                print(f"⚠️  Gemini STT failed: {exc}")
                if self.whisper_fallback is not None:
                    print("   Falling back to Whisper STT...")
                    return await self.whisper_fallback.transcribe(audio_path)

                return TranscriptionResult(text="", language="ml", confidence=0.0)

        # Should not reach here, but safety net
        if self.whisper_fallback is not None:
            return await self.whisper_fallback.transcribe(audio_path)
        return TranscriptionResult(text="", language="ml", confidence=0.0)
