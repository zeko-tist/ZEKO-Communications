"""Fallback online TTS using Gemini 3.1 Flash native text-to-speech.

This is the SECONDARY online TTS — used when Sarvam AI (primary) fails.
Uses the Gemini API with response_modalities=["AUDIO"] to generate
natural-sounding speech. Supports 70+ languages including Malayalam.
Returns raw PCM audio (16-bit, 24kHz).

On API failure, falls back to KokoroTTS (English) as a last resort.
"""

from __future__ import annotations

import asyncio

import numpy as np

from .base import BaseTTS


class GeminiTTS(BaseTTS):
    """Fallback online TTS engine using Gemini 3.1 Flash native TTS.

    Used when Sarvam AI TTS (primary) is unavailable. Leverages the
    same Gemini API key as the LLM. Supports Malayalam and English.
    Falls back to KokoroTTS on failure as a last resort.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.1-flash-tts-preview",
        voice_name: str = "Pulcherrima",
        kokoro_fallback: BaseTTS | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.voice_name = voice_name
        self.kokoro_fallback = kokoro_fallback
        self._client = None

    def preload(self) -> None:
        """Initialize the Gemini client."""
        self._get_client()
        print("✅ Gemini TTS client ready (fallback).")

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis — called from a thread."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice_name,
                    )
                )
            ),
        )

        response = client.models.generate_content(
            model=self.model_name,
            contents=text,
            config=config,
        )

        if not response.candidates:
            raise RuntimeError(f"No candidates returned: {response}")

        content = response.candidates[0].content
        if not content or getattr(content, "parts", None) is None:
            raise RuntimeError(f"Content blocked or empty: {response}")

        # Gemini returns raw PCM 16-bit 24kHz (little-endian)
        audio_data = content.parts[0].inline_data.data
        sample_rate = 24000
        pcm_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = pcm_array.astype(np.float32) / 32768.0

        return audio_float, sample_rate

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech using Gemini's native TTS (async).

        On API failure, falls back to KokoroTTS.
        Returns (audio_float32, sample_rate=24000).
        """
        try:
            return await asyncio.to_thread(self._synthesize_sync, text)
        except Exception as exc:
            print(f"⚠️  Gemini TTS failed: {exc}")
            if self.kokoro_fallback is not None:
                print("   Falling back to Kokoro TTS (English)...")
                return await self.kokoro_fallback.synthesize(text)
            # Return silence as last resort
            return np.zeros(4800, dtype=np.float32), 24000
