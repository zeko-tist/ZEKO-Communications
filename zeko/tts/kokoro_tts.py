"""Offline TTS using Kokoro (82M parameters).

Lightweight, fast TTS for offline English speech synthesis.
Uses ~0.5 GB VRAM on GPU and synthesizes a sentence in ~50ms.

Note: Kokoro does NOT support Malayalam. In offline mode, ZEKO
responds in English and uses Kokoro for speech synthesis.
"""

from __future__ import annotations

import asyncio

import numpy as np

from .base import BaseTTS


class KokoroTTS(BaseTTS):
    """Offline TTS engine using Kokoro for English speech.

    Very lightweight (82M params) and fast. English only — no Malayalam.
    """

    def __init__(
        self,
        lang_code: str = "en-us",
        voice: str = "af_heart",
    ) -> None:
        self.lang_code = lang_code
        self.voice = voice
        self._pipeline = None

    def preload(self) -> None:
        """Load the Kokoro model into memory."""
        print(f"🔄 Loading Kokoro TTS ({self.lang_code}, {self.voice})...")
        self._load_pipeline()
        print("✅ Kokoro TTS ready.")

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from kokoro import KPipeline
            import warnings
        except ImportError:
            raise RuntimeError("Kokoro not installed. Run: pip install kokoro")

        # Suppress PyTorch warnings originating from Kokoro's model definition
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout option adds dropout.*")
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weight_norm is deprecated.*")
            
            self._pipeline = KPipeline(
                lang_code=self.lang_code,
                repo_id="hexgrad/Kokoro-82M",
            )
            
        return self._pipeline

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis — called from a thread."""
        pipeline = self._load_pipeline()
        sample_rate = 24000

        # Kokoro's generate() yields (graphemes, phonemes, audio) tuples
        audio_chunks = []
        for _gs, _ps, audio in pipeline(text, voice=self.voice, speed=1.0):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            print("⚠️  Kokoro produced no audio.")
            return np.zeros(4800, dtype=np.float32), sample_rate

        full_audio = np.concatenate(audio_chunks)

        # Ensure float32 in [-1, 1]
        if full_audio.dtype != np.float32:
            full_audio = full_audio.astype(np.float32)
        if np.max(np.abs(full_audio)) > 1.0:
            full_audio = full_audio / np.max(np.abs(full_audio))

        return full_audio, sample_rate

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize English speech using Kokoro (async)."""
        try:
            return await asyncio.to_thread(self._synthesize_sync, text)
        except Exception as exc:
            print(f"⚠️  Kokoro TTS failed: {exc}")
            return np.zeros(4800, dtype=np.float32), 24000

    def unload(self) -> None:
        """Release the Kokoro model from memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            print("🗑️  Kokoro TTS unloaded.")
