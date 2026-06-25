"""Online TTS using Sarvam AI Bulbul v3 — primary online voice.

Sarvam handles Malayalam script natively (no transliteration needed).
Supports 11 Indian languages + English (Indian accent). Returns
base64-encoded WAV which we decode to raw PCM float32 for playback.

On API failure, raises an exception so the pipeline can fall back
to Gemini TTS.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time

import numpy as np

from .base import BaseTTS


class SarvamTTS(BaseTTS):
    """Online TTS engine using Sarvam AI Bulbul v3.

    Primary online TTS — handles Malayalam, English, and Manglish
    code-switching natively. No transliteration required.

    Falls through (raises) on failure so the pipeline can try Gemini TTS.
    """

    def __init__(
        self,
        api_key: str,
        speaker: str = "shubh",
        target_language_code: str = "ml-IN",
        model: str = "bulbul:v3",
    ) -> None:
        self.api_key = api_key
        self.speaker = speaker
        self.target_language_code = target_language_code
        self.model = model
        self._client = None

    def preload(self) -> None:
        """Initialize the Sarvam AI client."""
        self._get_client()
        print(f"✅ Sarvam TTS ready (speaker={self.speaker}, lang={self.target_language_code}).")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from sarvamai import SarvamAI
        except ImportError:
            raise RuntimeError("sarvamai not installed. Run: pip install sarvamai")

        self._client = SarvamAI(api_subscription_key=self.api_key)
        return self._client

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis — called from a thread."""
        client = self._get_client()

        t0 = time.perf_counter()

        response = client.text_to_speech.convert(
            text=text,
            target_language_code=self.target_language_code,
            speaker=self.speaker,
            model=self.model,
            pace=1.0,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Response contains base64-encoded WAV in audios[0]
        if not response.audios or not response.audios[0]:
            raise RuntimeError("Sarvam TTS returned no audio data")

        wav_bytes = base64.b64decode(response.audios[0])

        # Parse WAV to extract raw PCM samples
        audio_float, sample_rate = self._wav_bytes_to_pcm(wav_bytes)

        print(f"✅ Sarvam TTS synthesized in {elapsed_ms:.0f}ms")
        return audio_float, sample_rate

    @staticmethod
    def _wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV bytes to float32 PCM array + sample rate.

        Uses soundfile for reliable WAV parsing (handles various
        bit depths and formats automatically).
        """
        import soundfile as sf

        audio_data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")

        # If stereo, take first channel
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        return audio_data, sample_rate

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech using Sarvam AI (async).

        On failure, raises the exception so the pipeline can
        fall back to Gemini TTS.
        Returns (audio_float32, sample_rate).
        """
        return await asyncio.to_thread(self._synthesize_sync, text)
