# ZEKO VRS Engine — Phase I Beta (Revised)

A real-time voice response engine for the ZEKO humanoid robot, built for a college campus environment. Speaks and understands **Malayalam** (and English) with sub-10s end-to-end latency.

## Architecture

```
Microphone → VAD → STT → LLM (streaming) → TTS → Speaker
                              ↓
                    asyncio.Queue (sentence buffer)
                              ↓
                    TTS consumer (concurrent playback)
```

### Online Mode
| Component | Engine |
|-----------|--------|
| STT | Gemini multimodal API (Whisper fallback) |
| LLM | Gemini 2.5 Flash |
| TTS | Sarvam AI Bulbul v3 → Gemini 3.1 Flash TTS → Kokoro |

### Offline Mode
| Component | Engine |
|-----------|--------|
| STT | faster-whisper large-v3-turbo (CUDA) |
| LLM | Gemma 3 4B (Q4_K_M GGUF, llama-cpp-python) |
| TTS | Kokoro 82M (English) |

## Hardware Target

- **Lenovo Legion 5** — AMD R7 5800H, 16GB RAM, RTX 3060 6GB
- VRAM budget: ~2.5GB Whisper + ~2.5GB Gemma 4B = ~5GB (fits in 6GB)

## Quick Start

### 1. Clone and set up the virtual environment
```powershell
python -m venv .build2r
.\.build2r\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-cuda.txt
```

### 2. Download the offline LLM model
Download [`google_gemma-3-4b-it-Q4_K_M.gguf`](https://huggingface.co/bartowski/google_gemma-3-4b-it-GGUF) and place it in `models/`.

### 3. Configure environment
```powershell
cp .env.example .env
# Edit .env with your API keys:
#   GEMINI_API_KEY=your-gemini-key
#   SARVAM_API_KEY=your-sarvam-key
```

### 4. Run
```powershell
.\run.ps1
```

## Project Structure

```
├── main.py                  # Entry point
├── run.ps1                  # PowerShell launch script
├── zeko/
│   ├── engine.py            # Component initialization & wiring
│   ├── pipeline.py          # Async streaming pipeline orchestrator
│   ├── config.py            # Pydantic-based configuration
│   ├── connectivity.py      # Online/offline detection
│   ├── vad.py               # Voice Activity Detection (Silero)
│   ├── stt/                 # Speech-to-Text engines
│   │   ├── gemini_stt.py    # Online: Gemini multimodal API
│   │   └── whisper_stt.py   # Offline: faster-whisper (CUDA)
│   ├── llm/                 # Language Model engines
│   │   ├── gemini_llm.py    # Online: Gemini 2.5 Flash
│   │   └── local_llm.py     # Offline: Gemma 3 4B (GGUF)
│   ├── tts/                 # Text-to-Speech engines
│   │   ├── sarvam_tts.py    # Online primary: Sarvam AI Bulbul v3
│   │   ├── gemini_tts.py    # Online fallback: Gemini 3.1 Flash TTS
│   │   └── kokoro_tts.py    # Offline: Kokoro 82M
│   └── utils/
│       └── transliterate.py # Malayalam→Latin for Kokoro
├── models/                  # GGUF model weights (gitignored)
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
└── requirements-cuda.txt    # CUDA-specific dependencies
```

## TTS Fallback Chain

```
Online:  Sarvam AI (Malayalam-native) → Gemini 3.1 Flash → Kokoro (English)
Offline: Kokoro (English only, with transliteration)
```

## License

Private — ZEKO Project.
