"""Voice Activity Detection using silero-vad.

Records from the microphone, detecting speech boundaries via silero-vad.
Returns captured audio as a numpy array and optionally saves to a temp WAV.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch


@dataclass
class VADResult:
    """Result of a VAD recording session."""

    audio: np.ndarray
    sample_rate: int
    speech_detected: bool

    def save_wav(self) -> Path | None:
        """Save captured audio to a temporary WAV file. Returns the path."""
        if not self.speech_detected or len(self.audio) == 0:
            return None
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, self.audio, self.sample_rate, subtype="PCM_16")
        return Path(tmp.name)


class VADRecorder:
    """Microphone recorder with silero-vad speech boundary detection.

    - 512-sample chunks at 16kHz
    - Recording starts on speech_prob > threshold
    - Stops after silence_ms of silence (configurable)
    - Hard cap at max_seconds (configurable)
    - Resets VAD state between recordings
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.3,
        silence_ms: int = 700,
        max_seconds: int = 15,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.max_seconds = max_seconds
        self._chunk_size = 512
        self._model, self._utils = self._load_vad()

    def _load_vad(self):
        """Load silero-vad model via torch.hub."""
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        return model, utils

    def record(self) -> VADResult:
        """Block until speech is detected, record, then return on silence.

        Returns a VADResult with the captured audio numpy array.
        """
        # Reset VAD state
        self._model.reset_states()

        chunks: list[np.ndarray] = []
        is_speaking = False
        silence_chunks = 0
        max_chunks = int(self.max_seconds * self.sample_rate / self._chunk_size)
        silence_limit = int(self.silence_ms * self.sample_rate / 1000 / self._chunk_size)
        total_chunks = 0

        print("🎤 Listening...")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._chunk_size,
        ) as stream:
            while total_chunks < max_chunks:
                audio_chunk, _ = stream.read(self._chunk_size)
                audio_chunk = audio_chunk.flatten()
                total_chunks += 1

                # Compute speech probability
                tensor = torch.from_numpy(audio_chunk)
                speech_prob = self._model(tensor, self.sample_rate).item()

                if not is_speaking:
                    if speech_prob > self.threshold:
                        is_speaking = True
                        silence_chunks = 0
                        chunks.append(audio_chunk)
                        print("🗣️  Speech detected")
                else:
                    chunks.append(audio_chunk)
                    if speech_prob < self.threshold:
                        silence_chunks += 1
                        if silence_chunks >= silence_limit:
                            print("🔇 Silence detected — end of utterance")
                            break
                    else:
                        silence_chunks = 0

        if not chunks:
            return VADResult(
                audio=np.array([], dtype=np.float32),
                sample_rate=self.sample_rate,
                speech_detected=False,
            )

        audio = np.concatenate(chunks)
        return VADResult(
            audio=audio,
            sample_rate=self.sample_rate,
            speech_detected=True,
        )
