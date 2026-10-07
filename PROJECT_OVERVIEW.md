# ZEKO Voice Response Engine (VRE) — Project Overview

**ZEKO VRE** is a real-time conversational voice engine built for the ZEKO humanoid robot in a college campus environment.

> **Current runtime target:** Raspberry Pi 5 (8 GB RAM, ARM64, no GPU), **English-only UPR**.
> Malayalam/Manglish paths and Sarvam TTS are **frozen** (preserved in code, not executed at runtime).

---

## 1. Concurrency & Streaming Architecture

Rather than a sequential `listen → wait → think → wait → speak` cycle, the engine uses an async producer-consumer pipeline:

```
Microphone → Silero VAD → STT → [Async LLM Token Stream]
                                         │
                                [Sentence Boundary Buffer]
                                         │ (asyncio.Queue)
                                         ▼
                                [TTS Worker] → Speaker Playback
```

* **Concurrent Overlap**: Using `asyncio.gather`, LLM generation and TTS synthesis run in parallel. As soon as the LLM completes Sentence 1, TTS synthesizes and plays it while the LLM is generating Sentence 2 — cutting perceived latency by 60–70%.
* **Context Memory**: Rolling multi-turn conversation buffer (max 10 turns).

---

## 2. Online vs. Offline Routing

Connectivity is detected via non-blocking `aiohttp` HEAD pings (5s TTL cache):

| Component | 🌐 Online Mode | 📴 Offline Mode |
| :--- | :--- | :--- |
| **VAD** | Silero VAD (CPU, 16 kHz) | Silero VAD (CPU, 16 kHz) |
| **STT** | Gemini 2.5 Flash Multimodal (Whisper fallback) | `faster-whisper` `tiny.en` (CPU, int8) |
| **LLM** | Gemini 2.5 Flash (streaming) | Gemma 3 1B Q4_K_M GGUF (CPU, `llama-cpp-python`) |
| **TTS** | GeminiTTS `Pulcherrima` → Kokoro fallback | KokoroTTS 82M (CPU, English) |

> **FROZEN (not loaded at runtime):**
> - Sarvam AI TTS (`bulbul:v3`) — Malayalam-native; frozen until `english_only=False`
> - Malayalam transliteration (`transliterate_for_tts`) — preserved in code, not called

---

## 3. Raspberry Pi 5 Hardware Constraints

| Resource | Budget |
| :--- | :--- |
| RAM | 8 GB — Whisper tiny.en (~80 MB) + Gemma 1B (~600 MB) + OS/runtime ≈ ~2 GB total |
| Storage | 32 GB SD card — Gemma 1B GGUF (~0.6 GB), Kokoro model (~200 MB), OS |
| CPU | ARM Cortex-A76 (quad-core) — all inference on CPU, no GPU offload |
| GPU | None — `n_gpu_layers=0`, Whisper `device=cpu`, `compute_type=int8` |

---

## 4. What's Frozen vs. Active

### ✅ Active at Runtime
- Silero VAD (CPU)
- GeminiSTT (English transcription prompt) + WhisperSTT `tiny.en` fallback
- GeminiLLM streaming (English-only system prompt) + Gemma 3 1B GGUF fallback
- GeminiTTS (online primary) → KokoroTTS (offline / last-resort)

### 🔒 Frozen (code preserved, not executed)
| Item | File | How to Re-enable |
| :--- | :--- | :--- |
| Sarvam TTS | `engine.py`, `config.py` | Uncomment `SarvamTTS` block; set `ENGLISH_ONLY=False` |
| Malayalam transliteration | `pipeline.py` | Uncomment `transliterate_for_tts()` call in `_synthesize_with_fallback` |
| Manglish STT prompt | `gemini_stt.py`, `whisper_stt.py` | Swap `_TRANSCRIPTION_PROMPT` / `_ENGLISH_PROMPT` with frozen versions |
| Language-routing in LLM | `gemini_llm.py` | Uncomment `lang_map` block in `_sync_stream` |
| Malayalam/Manglish system prompt | `llm/base.py` | Swap active `SYSTEM_PROMPT` with frozen version |
| Windows CUDA DLL patches | `whisper_stt.py`, `main.py` | Restore archived code for Legion laptop deployment |
| CUDA requirements | `requirements-cuda.txt` | Uncomment entries; reinstall with CUDA index URL |

---

## 5. File Map

```
D:\Projects\ZEKO-Communications/
├── main.py                     # Entry point (Linux/RPi5 clean version)
├── run.sh                      # Bash startup script (RPi5)
├── run.ps1                     # PowerShell startup script (archived, Legion laptop)
├── requirements.txt            # RPi5 dependencies (sarvamai removed)
├── requirements-cuda.txt       # FROZEN — Legion laptop CUDA deps (commented out)
├── models/
│   └── gemma-3-1b-it-Q4_K_M.gguf  # Offline LLM weights (~0.6 GB) — download separately
└── zeko/
    ├── config.py               # Pydantic config — RPi5 defaults, english_only flag
    ├── connectivity.py         # Async connectivity checker (5s TTL cache)
    ├── engine.py               # Module lifecycle — Sarvam frozen, GeminiTTS as primary
    ├── pipeline.py             # Streaming pipeline — Malayalam paths frozen
    ├── vad.py                  # Silero VAD microphone recorder (unchanged)
    ├── stt/
    │   ├── gemini_stt.py       # Online STT — English-only prompt; Manglish frozen
    │   └── whisper_stt.py      # Offline STT — tiny.en, CPU/int8; CUDA patch removed
    ├── llm/
    │   ├── base.py             # SYSTEM_PROMPT — English-only; Manglish version frozen
    │   ├── gemini_llm.py       # Online LLM — forced English; lang-routing frozen
    │   └── local_llm.py        # Offline LLM — Gemma 1B, n_gpu_layers=0
    ├── tts/
    │   ├── gemini_tts.py       # Online primary TTS (unchanged)
    │   ├── kokoro_tts.py       # Offline TTS — English only (unchanged)
    │   └── sarvam_tts.py       # FROZEN — Malayalam TTS; code preserved, not loaded
    └── utils/
        └── transliterate.py    # FROZEN — Malayalam-to-Latin; code preserved, not called
```

---

## 6. Changelog & Redesign History (Legion 5 → Raspberry Pi 5)

This section records the modifications made to scale down the hardware requirements from a Lenovo Legion 5 (RTX 3060, Windows) laptop to a **Raspberry Pi 5 (8 GB RAM, ARM64, 32 GB SD card)**, while shifting UPR to **English-only**:

### 🎯 Key Architectural Changes

1. **English-Only UPR Shift (Malayalam & Manglish Frozen)**
   - **No Code Deleted**: All Malayalam processing, transliteration, and prompt logic were frozen via structured comments and runtime switches rather than deleted.
   - **System Prompt**: [`SYSTEM_PROMPT`](file:///D:/Projects/ZEKO-Communications/zeko/llm/base.py) updated to pure English campus robot persona; Manglish persona archived in comments.
   - **Online LLM**: [`gemini_llm.py`](file:///D:/Projects/ZEKO-Communications/zeko/llm/gemini_llm.py) forces English responses regardless of input; language detection routing frozen.
   - **STT Prompts**: [`gemini_stt.py`](file:///D:/Projects/ZEKO-Communications/zeko/stt/gemini_stt.py) prompt updated to English-only transcription; Manglish transcription prompt frozen.
   - **TTS Pipeline**: Sarvam AI TTS (`bulbul:v3`) frozen; `GeminiTTS` promoted to online primary with `KokoroTTS` (English, 82M) as offline fallback. Malayalam transliteration in [`pipeline.py`](file:///D:/Projects/ZEKO-Communications/zeko/pipeline.py) disabled.
   - **API Dependencies**: Sarvam API key is no longer required in `.env` or during runtime.

2. **Raspberry Pi 5 Hardware & Resource Scaling**
   - **GPU Offload Removed**: `n_gpu_layers` set to `0` across config and model loaders; CUDA cache clearing logic removed from [`local_llm.py`](file:///D:/Projects/ZEKO-Communications/zeko/llm/local_llm.py).
   - **Model Downscaling**:
     - **Offline LLM**: Replaced 4B GGUF with **Gemma 3 1B Q4_K_M GGUF** (~0.6 GB RAM, down from ~2.5 GB).
     - **Context Window**: Reduced `n_ctx` from 2048 to 1024 tokens to conserve RAM on RPi5.
     - **Offline STT**: Replaced Whisper base/small multilingual with **`tiny.en`** (`int8` CPU quantization, ~80 MB RAM).
   - **OS & Startup Adaptation**:
     - Created [`run.sh`](file:///D:/Projects/ZEKO-Communications/run.sh) for Linux/Raspberry Pi OS startup.
     - Removed Windows-specific CUDA DLL directory additions (`os.add_dll_directory`) in [`whisper_stt.py`](file:///D:/Projects/ZEKO-Communications/zeko/stt/whisper_stt.py) and [`main.py`](file:///D:/Projects/ZEKO-Communications/main.py).
   - **Dependency Cleanup**:
     - `sarvamai` removed from active [`requirements.txt`](file:///D:/Projects/ZEKO-Communications/requirements.txt).
     - [`requirements-cuda.txt`](file:///D:/Projects/ZEKO-Communications/requirements-cuda.txt) frozen and archived for future reference.

### 📋 Detailed File-by-File Modification Log

| File | Status | Modifications |
| :--- | :--- | :--- |
| [`.env.example`](file:///D:/Projects/ZEKO-Communications/.env.example) | Modified | Added `ENGLISH_ONLY=True`; switched defaults to `tiny.en` and Gemma 1B GGUF; commented out Sarvam API keys. |
| [`requirements.txt`](file:///D:/Projects/ZEKO-Communications/requirements.txt) | Modified | Commented out `sarvamai`; marked `indic-transliteration` as frozen; removed CUDA build flags for `llama-cpp-python`. |
| [`requirements-cuda.txt`](file:///D:/Projects/ZEKO-Communications/requirements-cuda.txt) | Frozen | Entire file commented out and preserved as reference for Legion GPU deployment. |
| [`run.sh`](file:///D:/Projects/ZEKO-Communications/run.sh) | Added | Bash startup script for RPi5 / Linux with virtual environment activation. |
| [`main.py`](file:///D:/Projects/ZEKO-Communications/main.py) | Modified | Removed Windows-specific CUDA DLL searching; cleaned platform-specific launch banners. |
| [`zeko/config.py`](file:///D:/Projects/ZEKO-Communications/zeko/config.py) | Modified | Added `english_only: bool = True`; updated defaults for RPi5 (`whisper_model_size="tiny.en"`, `whisper_device="cpu"`, `whisper_compute_type="int8"`, `local_llm_model_path="models/gemma-3-1b-it-Q4_K_M.gguf"`, `local_llm_n_gpu_layers=0`, `local_llm_n_ctx=1024`). |
| [`zeko/engine.py`](file:///D:/Projects/ZEKO-Communications/zeko/engine.py) | Modified | Froze `SarvamTTS` initialization; made `GeminiTTS` the online primary engine with `KokoroTTS` fallback. |
| [`zeko/pipeline.py`](file:///D:/Projects/ZEKO-Communications/zeko/pipeline.py) | Modified | Froze Malayalam transliteration step (`transliterate_for_tts`); streamlined queueing directly for English speech. |
| [`zeko/llm/base.py`](file:///D:/Projects/ZEKO-Communications/zeko/llm/base.py) | Modified | Replaced `SYSTEM_PROMPT` with English-only persona; original Manglish prompt preserved as comment. |
| [`zeko/llm/gemini_llm.py`](file:///D:/Projects/ZEKO-Communications/zeko/llm/gemini_llm.py) | Modified | Enforced English responses in Gemini system instructions; froze multilingual language routing logic. |
| [`zeko/llm/local_llm.py`](file:///D:/Projects/ZEKO-Communications/zeko/llm/local_llm.py) | Modified | Configured defaults for CPU execution (`n_gpu_layers=0`, `n_ctx=1024`); simplified `unload()` to CPU-only memory cleanup. |
| [`zeko/stt/gemini_stt.py`](file:///D:/Projects/ZEKO-Communications/zeko/stt/gemini_stt.py) | Modified | Updated transcription prompt to English-only; archived Manglish prompt in comments. |
| [`zeko/stt/whisper_stt.py`](file:///D:/Projects/ZEKO-Communications/zeko/stt/whisper_stt.py) | Modified | Updated defaults to `tiny.en` on CPU (`int8`); stripped Windows CUDA DLL directory injection. |
