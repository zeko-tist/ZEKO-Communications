"""ZEKO VRS Engine — top-level engine that wires all modules together.

Initializes all modules from config, manages the main async run loop,
and handles clean shutdown on KeyboardInterrupt.
"""

from __future__ import annotations

from .config import Config
from .connectivity import ConnectivityChecker
from .vad import VADRecorder
from .stt.whisper_stt import WhisperSTT
from .stt.gemini_stt import GeminiSTT
from .llm.gemini_llm import GeminiLLM
from .llm.local_llm import LocalLLM
from .tts.sarvam_tts import SarvamTTS
from .tts.gemini_tts import GeminiTTS
from .tts.kokoro_tts import KokoroTTS
from .pipeline import Pipeline


class VoiceResponseEngine:
    """Top-level engine that initializes all modules and runs the main loop.

    Wires together:
    - VAD (silero-vad) for speech detection
    - STT (Whisper offline / Gemini online) for transcription
    - LLM (Gemini online / Gemma offline) for response generation
    - TTS (Sarvam primary → Gemini fallback → Kokoro last-resort) for speech synthesis
    - Pipeline for async streaming orchestration
    - Connectivity checker for online/offline routing
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pipeline: Pipeline | None = None

    @classmethod
    def from_env(cls) -> "VoiceResponseEngine":
        """Create an engine instance from environment variables."""
        return cls(config=Config())

    async def initialize(self) -> None:
        """Initialize all modules and wire up the pipeline."""
        cfg = self.config

        print("\n" + "=" * 60)
        print("  ZEKO VRS Engine — Phase I Beta (Revised)")
        print("  Initializing components...")
        print("=" * 60)

        # --- Connectivity ---
        connectivity = ConnectivityChecker(
            check_url=cfg.connectivity_check_url,
            ttl_seconds=cfg.connectivity_ttl_seconds,
        )
        online = await connectivity.is_online()
        print(f"{'🌐' if online else '📴'} Connectivity: {'online' if online else 'offline'}")

        # --- VAD ---
        print("\n📦 Loading VAD...")
        vad = VADRecorder(
            sample_rate=cfg.sample_rate,
            threshold=cfg.vad_threshold,
            silence_ms=cfg.vad_silence_ms,
            max_seconds=cfg.vad_max_seconds,
        )

        # --- STT ---
        print("\n📦 Loading STT modules...")
        offline_stt = WhisperSTT(
            model_size=cfg.whisper_model,
            device=cfg.whisper_device,
            compute_type=cfg.whisper_compute_type,
        )
        offline_stt.preload()

        online_stt = GeminiSTT(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_model,
            whisper_fallback=offline_stt,  # Fallback to Whisper on 429/timeout
        )
        if online:
            online_stt.preload()

        # --- TTS ---
        # Offline: Kokoro (English only)
        print("\n📦 Loading TTS modules...")
        offline_tts = KokoroTTS(
            lang_code=cfg.kokoro_lang,
            voice=cfg.kokoro_voice,
        )
        try:
            offline_tts.preload()
        except Exception as exc:
            print(f"⚠️  Kokoro TTS failed to load: {exc}")
            print("   Offline TTS will not be available.")

        # Online fallback: Gemini 3.1 Flash TTS → Kokoro last-resort
        gemini_tts = GeminiTTS(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_tts_model,
            voice_name=cfg.gemini_tts_voice,
            kokoro_fallback=offline_tts,  # Last-resort fallback to Kokoro
        )
        if online:
            gemini_tts.preload()

        # Online primary: Sarvam AI TTS (Malayalam-native)
        online_tts = SarvamTTS(
            api_key=cfg.sarvam_api_key,
            speaker=cfg.sarvam_speaker,
            target_language_code=cfg.sarvam_language,
            model=cfg.sarvam_model,
        )
        if online and cfg.sarvam_api_key:
            try:
                online_tts.preload()
            except Exception as exc:
                print(f"⚠️  Sarvam TTS failed to init: {exc}")
                print("   Will use Gemini TTS as primary online TTS.")

        # --- LLM ---
        print("\n📦 Loading LLM modules...")
        online_llm = GeminiLLM(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_model,
        )
        if online:
            online_llm.preload()

        offline_llm = LocalLLM(
            model_path=cfg.local_llm_model_path,
            n_gpu_layers=cfg.local_llm_gpu_layers,
            n_ctx=cfg.local_llm_ctx_size,
        )
        # Don't preload offline LLM — shares VRAM with Whisper.
        # Loaded on-demand when offline mode is detected.
        if offline_llm.is_available():
            print(f"✅ Offline LLM available: {cfg.local_llm_model_path}")
        else:
            print(f"⚠️  Offline LLM not found: {cfg.local_llm_model_path}")

        # --- Pipeline ---
        self.pipeline = Pipeline(
            vad=vad,
            online_stt=online_stt,
            offline_stt=offline_stt,
            online_llm=online_llm,
            offline_llm=offline_llm,
            online_tts=online_tts,
            online_tts_fallback=gemini_tts,
            offline_tts=offline_tts,
            connectivity=connectivity,
        )

        print("\n" + "=" * 60)
        print("  🚀 ZEKO is online. Speak to begin.")
        print("=" * 60 + "\n")

    async def run(self) -> None:
        """Run the main voice interaction loop."""
        if self.pipeline is None:
            await self.initialize()

        print("Comms System Active")
        print("─" * 40)

        try:
            while True:
                processed = await self.pipeline.process_utterance()
                if not processed:
                    print("... Silence ...")
        except KeyboardInterrupt:
            print("\n\n🛑 ZEKO shutting down.")
