"""ZEKO VRE — top-level engine that wires all modules together.

Hardware target: Raspberry Pi 5 (Linux, ARM64, 8 GB RAM, no GPU).
Runtime mode:    English-only UPR. Sarvam TTS is frozen.
                 GeminiTTS is the online primary TTS.
                 KokoroTTS is the offline TTS.
"""

from __future__ import annotations

from .config import Config
from .connectivity import ConnectivityChecker
from .vad import VADRecorder
from .stt.whisper_stt import WhisperSTT
from .stt.gemini_stt import GeminiSTT
from .llm.gemini_llm import GeminiLLM
from .llm.local_llm import LocalLLM
from .tts.gemini_tts import GeminiTTS
from .tts.kokoro_tts import KokoroTTS
from .pipeline import Pipeline

# FROZEN: SarvamTTS import is preserved but not instantiated when english_only=True.
# from .tts.sarvam_tts import SarvamTTS


class VoiceResponseEngine:
    """Top-level engine that initializes all modules and runs the main loop.

    English-only mode wiring:
    - VAD  : Silero VAD (CPU, 16 kHz)
    - STT  : GeminiSTT (online) / WhisperSTT tiny.en (offline, CPU int8)
    - LLM  : GeminiLLM streaming (online) / Gemma 3 1B GGUF (offline, CPU)
    - TTS  : GeminiTTS (online primary) → KokoroTTS (offline / last-resort)
    - FROZEN: SarvamTTS (Malayalam-native) is not started; no Sarvam API key needed.
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
        print("  ZEKO VRE — English-Only Mode")
        print("  Hardware: Raspberry Pi 5  |  No GPU")
        print("  Initializing components...")
        print("=" * 60)

        if cfg.english_only:
            print("🔒 English-only UPR active. Malayalam paths are frozen.")
            print("🔒 Sarvam TTS is frozen. SARVAM_API_KEY not required.")

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
            whisper_fallback=offline_stt,
        )
        if online:
            online_stt.preload()

        # --- TTS ---
        print("\n📦 Loading TTS modules...")

        # Offline / last-resort: Kokoro 82M (English, CPU)
        offline_tts = KokoroTTS(
            lang_code=cfg.kokoro_lang,
            voice=cfg.kokoro_voice,
        )
        try:
            offline_tts.preload()
        except Exception as exc:
            print(f"⚠️  Kokoro TTS failed to load: {exc}")
            print("   Offline TTS will not be available.")

        # Online primary: Gemini 3.1 Flash TTS (Sarvam is frozen)
        # FROZEN: SarvamTTS is NOT instantiated. To re-enable Malayalam TTS,
        #         set english_only=False and uncomment the SarvamTTS block.
        online_tts = GeminiTTS(
            api_key=cfg.gemini_api_key,
            model_name=cfg.gemini_tts_model,
            voice_name=cfg.gemini_tts_voice,
            kokoro_fallback=offline_tts,
        )
        if online:
            online_tts.preload()

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
            n_gpu_layers=cfg.local_llm_gpu_layers,   # 0 on RPi5
            n_ctx=cfg.local_llm_ctx_size,
        )
        if offline_llm.is_available():
            print(f"✅ Offline LLM available: {cfg.local_llm_model_path}")
        else:
            print(f"⚠️  Offline LLM not found: {cfg.local_llm_model_path}")

        # --- Pipeline ---
        # online_tts_fallback is not needed in English-only mode since
        # GeminiTTS already cascades to Kokoro internally.
        # Passed as None to keep the Pipeline interface stable.
        self.pipeline = Pipeline(
            vad=vad,
            online_stt=online_stt,
            offline_stt=offline_stt,
            online_llm=online_llm,
            offline_llm=offline_llm,
            online_tts=online_tts,
            online_tts_fallback=None,   # Sarvam frozen; GeminiTTS handles fallback internally
            offline_tts=offline_tts,
            connectivity=connectivity,
            english_only=cfg.english_only,
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
