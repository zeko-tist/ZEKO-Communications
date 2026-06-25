"""Offline LLM using Gemma 3 4B via llama-cpp-python.

Fallback brain for offline operation. Runs the quantized Gemma 3 4B
model (Q4_K_M GGUF) on the RTX 3060 (~2.5 GB VRAM).

VRAM management: Only loaded when offline mode is triggered.
WhisperSTT.unload() must be called first to free VRAM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from .base import BaseLLM, SYSTEM_PROMPT


class LocalLLM(BaseLLM):
    """Offline LLM engine using Gemma 3 4B quantized via llama-cpp-python.

    Streaming via create_completion(stream=True), wrapped to async
    via BaseLLM._sync_gen_to_async.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
    ) -> None:
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self._model = None

    def preload(self) -> None:
        """Load the GGUF model into GPU memory."""
        if not Path(self.model_path).exists():
            print(f"⚠️  Local LLM model not found at: {self.model_path}")
            print("   Download the GGUF file and place it at the configured path.")
            print("   Offline mode will not be available.")
            return
        print(f"🔄 Loading local LLM ({self.model_path})...")
        self._load_model()
        print("✅ Local LLM ready.")

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Run:\n"
                '  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python'
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Download the Gemma 3 4B Q4_K_M GGUF from HuggingFace."
            )
        self._model = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        return self._model

    def _build_prompt(
        self, user_text: str, language: str, history: list[dict] | None
    ) -> str:
        """Build a chat-style prompt for the Gemma model."""
        prompt_parts = [
            f"<start_of_turn>system\n{SYSTEM_PROMPT}\n"
            f"CRITICAL INSTRUCTION: Respond EXCLUSIVELY in English. "
            f"Do NOT output Malayalam, Tamil, Hindi, or any other non-English script.<end_of_turn>"
        ]

        if history:
            for turn in history[-10:]:
                role = "user" if turn["role"] == "user" else "model"
                prompt_parts.append(
                    f"<start_of_turn>{role}\n{turn['text']}<end_of_turn>"
                )

        prompt_parts.append(f"<start_of_turn>user\n{user_text.strip()}<end_of_turn>")
        prompt_parts.append("<start_of_turn>model\n")

        return "\n".join(prompt_parts)

    def _sync_stream(self, user_text: str, language: str, history: list[dict] | None):
        """Synchronous streaming generator — will be wrapped to async."""
        model = self._load_model()
        prompt = self._build_prompt(user_text, language, history)

        try:
            output = model(
                prompt,
                max_tokens=150,  # ~2 sentences
                temperature=0.1,
                top_p=0.95,
                stop=["<end_of_turn>", "<start_of_turn>"],
                stream=True,
            )
            for token_data in output:
                choices = token_data.get("choices", [])
                if choices:
                    text = choices[0].get("text", "")
                    if text:
                        yield text
        except Exception as exc:
            print(f"❌ Local LLM error: {exc}")
            yield "I could not process that offline."

    async def stream(
        self, user_text: str, language: str, history: list[dict] | None = None
    ) -> AsyncIterator[str]:
        """Async stream of local LLM tokens.

        Wraps the sync llama.cpp streaming in a thread.
        """
        sync_gen = self._sync_stream(user_text, language, history)
        async for chunk in self._sync_gen_to_async(sync_gen):
            yield chunk

    def is_available(self) -> bool:
        """Check if the model file exists."""
        return Path(self.model_path).exists()

    def unload(self) -> None:
        """Release model from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            print("🗑️  Local LLM unloaded from GPU.")
