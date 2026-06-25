"""Base interfaces for Text-to-Speech modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]


class BaseTTS(ABC):
    """Abstract base class for TTS engines (async interface)."""

    @abstractmethod
    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text: The text to convert to speech.

        Returns:
            Tuple of (audio_array as float32, sample_rate).
        """
        ...

    @abstractmethod
    def preload(self) -> None:
        """Eagerly load model weights / initialize client."""
        ...

    async def speak(self, text: str) -> None:
        """Synthesize and play through speakers (async)."""
        if not text.strip():
            return

        audio, sample_rate = await self.synthesize(text)

        if sd is None:
            print("⚠️  sounddevice unavailable — cannot play audio.")
            return

        try:
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as exc:
            print(f"⚠️  Could not play audio: {exc}")
