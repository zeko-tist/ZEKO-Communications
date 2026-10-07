"""ZEKO VRE — entry point.

Designed for Raspberry Pi 5 (Linux, ARM64, no discrete GPU).
English-only UPR mode: Malayalam interaction paths are preserved in code
but frozen at runtime. Sarvam TTS is also frozen (SARVAM_API_KEY not needed).
"""

import asyncio

from zeko.engine import VoiceResponseEngine


async def main() -> None:
    engine = VoiceResponseEngine.from_env()
    await engine.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 ZEKO shutting down. Goodbye.")
