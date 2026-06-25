"""Offline STT using faster-whisper (large-v3-turbo, GPU, float16).

VRAM: ~2GB. Inference: ~0.3–0.5s for 5s audio.
Seeded with Manglish phrases for better Malayalam/English recognition.
On CUDA OOM, retries once on CPU before raising.
"""

from __future__ import annotations

import asyncio
import os
import sys

# ── ctranslate2 Windows CUDA fix ─────────────────────────────────────────────
# ctranslate2 (used by faster-whisper) requires cuBLAS and cuDNN DLLs.
# On Windows, Python 3.8+ does not search PATH for DLLs. We must explicitly
# add PyTorch's `lib` directory (which contains cublas64_12.dll) to the DLL
# search path so ctranslate2 can load them successfully.
if sys.platform == "win32":
    try:
        import torch as _torch
        _torch_lib_dir = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.exists(_torch_lib_dir):
            os.add_dll_directory(_torch_lib_dir)
            os.environ["PATH"] = _torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────────────────────

from .base import BaseSTT, TranscriptionResult


# Common Manglish phrases to bias the decoder toward expected vocabulary.
_MANGLISH_PROMPT = (
    "college campus, student, professor, class, exam, "
    "enthaanu, njan, ningal, eppol, evideyaanu, "
    "enikku, parayoo, shariyaanu, valare nannaayi, "
    "thank you, please, okay, sorry"
)


class WhisperSTT(BaseSTT):
    """Offline STT engine using faster-whisper large-v3-turbo.

    Async via asyncio.to_thread — wraps the synchronous faster-whisper
    API without blocking the event loop.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def preload(self) -> None:
        """Eagerly load the Whisper model into GPU memory."""
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

        try:
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=_MANGLISH_PROMPT,
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return TranscriptionResult(
                text=text,
                language=info.language,
                confidence=info.language_probability,
            )
        except RuntimeError as exc:
            if "CUDA" in str(exc) or "out of memory" in str(exc):
                print("⚠️  CUDA OOM — retrying transcription on CPU...")
                return self._transcribe_cpu_fallback(audio_path)
            raise

    def _transcribe_cpu_fallback(self, audio_path: str) -> TranscriptionResult:
        """One-shot CPU retry on CUDA OOM."""
        from faster_whisper import WhisperModel

        cpu_model = WhisperModel(
            self.model_size, device="cpu", compute_type="int8"
        )
        segments, info = cpu_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=_MANGLISH_PROMPT,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=info.language_probability,
        )

    def unload(self) -> None:
        """Release the Whisper model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("🗑️  Whisper STT unloaded from GPU.")
