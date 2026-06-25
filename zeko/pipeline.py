"""Async streaming pipeline orchestrator.

The heart of the beta engine — replaces alpha's sequential
listen → think → speak with an async pipeline that overlaps
LLM generation and TTS synthesis via asyncio.Queue + asyncio.gather.

Architecture:
    VAD → STT → [async LLM stream → sentence buffer → Queue] ← → [TTS consumer → Speaker]

The key insight: asyncio.gather runs stream_and_split() and
synthesize_and_play() concurrently. TTS synthesizes sentence 1
while the LLM generates sentence 2. Cuts perceived latency 60–70%.

TTS priority chain (online mode):
    Sarvam AI (primary, Malayalam-native) → Gemini 3.1 Flash (fallback) → Kokoro (last resort)
"""

from __future__ import annotations

import asyncio
import re
import time

import numpy as np
import sounddevice as sd

from .connectivity import ConnectivityChecker
from .stt.base import BaseSTT, TranscriptionResult
from .llm.base import BaseLLM
from .tts.base import BaseTTS
from .vad import VADRecorder, VADResult
from .utils.transliterate import transliterate_for_tts


# Regex to split on sentence-ending punctuation followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def _extract_complete_sentence(buffer: str) -> tuple[str | None, str]:
    """Try to extract a complete sentence from the buffer.

    Returns (sentence, remaining_buffer). If no sentence boundary
    is found, returns (None, original_buffer).
    """
    match = _SENTENCE_SPLIT.search(buffer)
    if match:
        sentence = buffer[: match.start() + 1].strip()
        remaining = buffer[match.end() :]
        return sentence, remaining
    return None, buffer


async def _play_audio(audio: np.ndarray, sample_rate: int) -> None:
    """Play audio through speakers (non-blocking via thread)."""
    def _play():
        sd.play(audio, sample_rate)
        sd.wait()

    await asyncio.to_thread(_play)


class Pipeline:
    """Async streaming pipeline orchestrator.

    Coordinates VAD recording, STT transcription, LLM streaming with
    sentence splitting, and TTS synthesis — all running concurrently
    via asyncio.Queue and asyncio.gather.
    """

    def __init__(
        self,
        vad: VADRecorder,
        online_stt: BaseSTT,
        offline_stt: BaseSTT,
        online_llm: BaseLLM,
        offline_llm: BaseLLM | None,
        online_tts: BaseTTS,
        online_tts_fallback: BaseTTS,
        offline_tts: BaseTTS,
        connectivity: ConnectivityChecker,
    ) -> None:
        self.vad = vad
        self.online_stt = online_stt
        self.offline_stt = offline_stt
        self.online_llm = online_llm
        self.offline_llm = offline_llm
        self.online_tts = online_tts                  # Sarvam AI (primary)
        self.online_tts_fallback = online_tts_fallback  # Gemini TTS (fallback)
        self.offline_tts = offline_tts                  # Kokoro (offline)
        self.connectivity = connectivity
        self.history: list[dict] = []

    async def _select_mode(self) -> str:
        """Determine whether to use online or offline modules."""
        if await self.connectivity.is_online():
            return "online"
        if self.offline_llm is not None and self.offline_llm.is_available():
            return "offline"
        print("⚠️  Offline LLM not available, attempting online mode...")
        return "online"

    def listen(self) -> VADResult:
        """Record speech from microphone using VAD (blocking)."""
        return self.vad.record()

    async def transcribe(self, vad_result: VADResult, mode: str) -> TranscriptionResult:
        """Transcribe captured audio using the appropriate STT engine."""
        wav_path = vad_result.save_wav()
        if wav_path is None:
            return TranscriptionResult(text="", language="ml", confidence=0.0)

        try:
            stt = self.online_stt if mode == "online" else self.offline_stt
            t0 = time.perf_counter()
            result = await stt.transcribe(str(wav_path))
            elapsed = time.perf_counter() - t0
            print(f'📝 STT ({mode}): "{result.text}" [{result.language}] ({elapsed:.2f}s)')
            return result
        finally:
            # Clean up temp WAV
            try:
                wav_path.unlink()
            except OSError:
                pass

    async def _synthesize_with_fallback(
        self, text: str, mode: str
    ) -> tuple[np.ndarray, int]:
        """Synthesize speech with the TTS fallback chain.

        Online mode:
            1. Sarvam AI (primary — handles Malayalam natively, no transliteration)
            2. Gemini 3.1 Flash (fallback — pass raw text, model handles Malayalam)
            3. Kokoro (last resort via Gemini's internal fallback — needs transliteration)

        Offline mode:
            1. Kokoro (English only — transliterate Malayalam to Latin)
        """
        if mode == "offline":
            # Offline: Kokoro only, needs transliterated text
            tts_text = transliterate_for_tts(text)
            return await self.offline_tts.synthesize(tts_text)

        # Online: try Sarvam first (no transliteration needed)
        try:
            return await self.online_tts.synthesize(text)
        except Exception as exc:
            print(f"⚠️  Sarvam TTS failed: {exc}")
            print("   Falling back to Gemini TTS...")

        # Gemini fallback (also handles Malayalam natively in 3.1)
        # Gemini's internal fallback to Kokoro will transliterate if needed
        return await self.online_tts_fallback.synthesize(text)

    async def generate_and_speak(
        self, transcript: TranscriptionResult, mode: str
    ) -> str:
        """Stream LLM response, split into sentences, TTS each one concurrently.

        This is the core streaming architecture from the spec:
        1. stream_and_split() — LLM streams tokens into a buffer,
           extracts complete sentences, pushes to asyncio.Queue
        2. synthesize_and_play() — consumes sentences from the queue,
           synthesizes and plays audio

        Both run concurrently via asyncio.gather.

        Returns the full response text.
        """
        llm = self.online_llm if mode == "online" else self.offline_llm

        if llm is None:
            print("❌ No LLM available for this mode.")
            return ""

        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_response_parts: list[str] = []

        t0 = time.perf_counter()
        first_sentence_time: list[float | None] = [None]  # mutable container

        async def stream_and_split():
            """LLM stream → sentence buffer → queue."""
            buffer = ""
            print("🧠 Generating response...")

            async for chunk in llm.stream(transcript.text, transcript.language, self.history):
                full_response_parts.append(chunk)
                buffer += chunk

                # Extract complete sentences from the buffer
                while True:
                    sentence, buffer = _extract_complete_sentence(buffer)
                    if sentence is None:
                        break
                    if first_sentence_time[0] is None:
                        first_sentence_time[0] = time.perf_counter() - t0
                        print(f"⚡ First sentence ready in {first_sentence_time[0]:.2f}s")
                    await sentence_queue.put(sentence)

            # Flush any remaining text in the buffer
            if buffer.strip():
                if first_sentence_time[0] is None:
                    first_sentence_time[0] = time.perf_counter() - t0
                await sentence_queue.put(buffer.strip())

            # Signal end-of-stream
            await sentence_queue.put(None)

        async def synthesize_and_play():
            """Consume sentences from queue, synthesize and play audio."""
            sentence_count = 0
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break

                sentence_count += 1
                print(f'🔊 Speaking sentence {sentence_count}: "{sentence}"')

                try:
                    audio, sample_rate = await self._synthesize_with_fallback(
                        sentence, mode
                    )
                    await _play_audio(audio, sample_rate)
                except Exception as exc:
                    print(f"⚠️  TTS/playback error: {exc}")

            elapsed = time.perf_counter() - t0
            print(f"✅ Response complete ({sentence_count} sentences, {elapsed:.2f}s total)")

        # Run both concurrently — LLM generates while TTS plays
        await asyncio.gather(stream_and_split(), synthesize_and_play())

        return "".join(full_response_parts)

    async def process_utterance(self) -> bool:
        """Run one full listen → transcribe → respond cycle.

        Returns True if speech was detected and processed, False otherwise.
        """
        # 1. Record via VAD (blocking — microphone capture)
        vad_result = await asyncio.to_thread(self.listen)
        if not vad_result.speech_detected:
            return False

        # 2. Determine online/offline mode
        mode = await self._select_mode()
        mode_emoji = "🌐" if mode == "online" else "📴"
        print(f"{mode_emoji} Mode: {mode}")

        # 3. Transcribe (async)
        transcript = await self.transcribe(vad_result, mode)
        if transcript.is_empty:
            print("... No speech recognized ...")
            return False

        # 4. Generate response + speak (streaming, concurrent)
        full_response = await self.generate_and_speak(transcript, mode)

        # 5. Update conversation history (rolling window, max 10 turns)
        if full_response.strip():
            self.history.append({"role": "user", "text": transcript.text})
            self.history.append({"role": "model", "text": full_response.strip()})
            if len(self.history) > 10:
                self.history = self.history[-10:]

        return True
