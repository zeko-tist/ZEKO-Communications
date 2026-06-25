"""Centralized configuration using Pydantic BaseSettings.

All model paths, API keys, thresholds, and device preferences are loaded
from environment variables and .env files with full type validation.
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

    # --- API Keys ---
    gemini_api_key: str = ""

    # --- STT ---
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    sample_rate: int = 16_000

    # --- LLM (online) ---
    gemini_model: str = "gemini-2.5-flash"

    # --- LLM (offline) ---
    local_llm_model_path: str = "models/gemma-3-4b-it-Q4_K_M.gguf"
    local_llm_gpu_layers: int = -1
    local_llm_ctx_size: int = 2048

    # --- TTS (online primary — Sarvam AI) ---
    sarvam_api_key: str = ""
    sarvam_speaker: str = "shubh"
    sarvam_language: str = "ml-IN"
    sarvam_model: str = "bulbul:v3"

    # --- TTS (online fallback — Gemini) ---
    gemini_tts_model: str = "gemini-3.1-flash-tts-preview"
    gemini_tts_voice: str = "Pulcherrima"

    # --- TTS (offline — Kokoro) ---
    kokoro_lang: str = "en-us"
    kokoro_voice: str = "am_adam"

    # --- VAD ---
    vad_threshold: float = 0.3
    vad_silence_ms: int = 700
    vad_max_seconds: int = 15

    # --- Connectivity ---
    connectivity_check_url: str = "https://generativelanguage.googleapis.com"
    connectivity_ttl_seconds: int = 5
