"""Offline STT using faster-whisper (tiny.en, CPU, int8).

Designed for Raspberry Pi 5 (ARM64, no GPU):
    - Model: tiny.en — ~39 MB, 2–3s transcription on RPi5 CPU.
    - Device: cpu, compute_type: int8.
    - Seeded with campus/English vocabulary prompt.

FROZEN: CUDA DLL path injection (Windows-specific) has been removed.
        The original code is archived below for reference.
"""

from __future__ import annotations

import asyncio

from .base import BaseSTT, TranscriptionResult


# English campus vocabulary prompt to bias the Whisper decoder.
_ENGLISH_PROMPT = (
    "college campus, student, professor, class, exam, assignment, "
    "library, cafeteria, schedule, registration, semester, "
    "thank you, please, okay, sorry, excuse me, help"
)

# FROZEN: Manglish prompt — preserved for when Malayalam UPR is re-enabled.
# _MANGLISH_PROMPT = (
#     "college campus, student, professor, class, exam, "
#     "enthaanu, njan, ningal, eppol, evideyaanu, "
#     "enikku, parayoo, shariyaanu, valare nannaayi, "
#     "thank you, please, okay, sorry"
# )

# FROZEN: Windows CUDA DLL path injection — not needed on Linux/RPi5.
# Original code:
#   if sys.platform == "win32":
#       import torch as _torch
#       _torch_lib_dir = os.path.join(os.path.dirname(_torch.__file__), "lib")
#       if os.path.exists(_torch_lib_dir):
#           os.add_dll_directory(_torch_lib_dir)
#           os.environ["PATH"] = _torch_lib_dir + os.pathsep + os.environ.get("PATH", "")


class WhisperSTT(BaseSTT):
    """Offline STT engine using faster-whisper (CPU, int8).

    Async via asyncio.to_thread — wraps the synchronous faster-whisper
    API without blocking the event loop.
    """

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def preload(self) -> None:
        """Eagerly load the Whisper model into memory."""
        print(f"🔄 Loading faster-whisper ({self.model_size}) on {self.device}...")
        self._load_model()
        print(f"✅ faster-whisper ready ({self.model_size}, {self.compute_type}).")

    def _load_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio from file (async, runs in thread)."""
        return await asyncio.to_thread(self._transcribe_sync, audio_path)

    def _transcribe_sync(self, audio_path: str) -> TranscriptionResult:
        """Synchronous transcription — called from a thread."""
        model = self._load_model()

        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=_ENGLISH_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=info.language_probability,
        )

    def unload(self) -> None:
        """Release the Whisper model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            print("🗑️  Whisper STT unloaded.")
