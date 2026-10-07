"""Centralized configuration using Pydantic BaseSettings.

All model paths, API keys, thresholds, and device preferences are loaded
from environment variables and .env files with full type validation.

Hardware target: Raspberry Pi 5 (8 GB RAM, ARM64, no discrete GPU).
Runtime mode:    English-only UPR. Malayalam paths are preserved in code
                 but are frozen at runtime (ENGLISH_ONLY=True).
                 Sarvam TTS is frozen — SARVAM_API_KEY is not required.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Typed, validated application configuration.

    Values are sourced from environment variables or a `.env` file
    in the project root. Pydantic coerces types automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime Mode ---
    # Set to False to re-enable Malayalam UPR and Sarvam TTS paths.
    english_only: bool = True

    # --- API Keys ---
    gemini_api_key: str = ""
    # sarvam_api_key is declared but not loaded at runtime when english_only=True.
    sarvam_api_key: str = ""

    # --- STT ---
    # tiny.en (~39 MB) is recommended for RPi5 CPU.
    # Upgrade to base.en or small.en for better accuracy at the cost of speed.
    whisper_model: str = "tiny.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    sample_rate: int = 16_000

    # --- LLM (online) ---
    gemini_model: str = "gemini-2.5-flash"

    # --- LLM (offline) ---
    # Target: Gemma 3 1B Q4_K_M GGUF (~0.6 GB RAM, feasible on RPi5 CPU).
    # Download: https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF
    local_llm_model_path: str = "models/gemma-3-1b-it-Q4_K_M.gguf"
    local_llm_gpu_layers: int = 0          # No GPU on RPi5
    local_llm_ctx_size: int = 1024         # Reduced from 2048 to conserve RAM

    # --- TTS (online primary — Gemini Flash TTS) ---
    # Gemini TTS is now the online primary since Sarvam is frozen.
    gemini_tts_model: str = "gemini-3.1-flash-tts-preview"
    gemini_tts_voice: str = "Pulcherrima"

    # --- TTS (offline — Kokoro) ---
    # Kokoro 82M runs on CPU; synthesizes a sentence in ~100–200ms on RPi5.
    kokoro_lang: str = "en-us"
    kokoro_voice: str = "am_adam"

    # --- FROZEN: Sarvam TTS (Malayalam-native online primary) ---
    # These values are preserved for future re-enablement but are not used
    # when english_only=True.
    sarvam_speaker: str = "shubh"
    sarvam_language: str = "ml-IN"
    sarvam_model: str = "bulbul:v3"

    # --- VAD ---
    vad_threshold: float = 0.3
    vad_silence_ms: int = 700
    vad_max_seconds: int = 15

    # --- Connectivity ---
    connectivity_check_url: str = "https://generativelanguage.googleapis.com"
    connectivity_ttl_seconds: int = 5
