"""Base interfaces for Speech-to-Text modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of a speech transcription."""

    text: str
    language: str
    confidence: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class BaseSTT(ABC):
    """Abstract base class for STT engines (async interface)."""

    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio from a file path.

        Args:
            audio_path: Path to a WAV audio file.

        Returns:
            TranscriptionResult with text, language, and confidence.
        """
        ...

    @abstractmethod
    def preload(self) -> None:
        """Eagerly load model weights / initialize client."""
        ...
