"""ZEKO VRE Build Beta — entry point."""

# ── espeakng-loader Windows fix ──────────────────────────────────────────────
# The espeakng-loader Windows wheel hardcodes the GitHub Actions build path
# (D:/a/espeakng-loader/...) into its compiled DLL. The env vars alone are not
# enough — phonemizer's EspeakWrapper must be explicitly told the library and
# data path via its class-level API BEFORE any `import kokoro` happens.
import os
import sys

if sys.platform == "win32":
    try:
        import espeakng_loader as _esl
        from pathlib import Path as _Path
        _pkg_dir = _Path(_esl.__file__).parent
        _lib_path = str(_pkg_dir / "espeak-ng.dll")
        _data_path = str(_pkg_dir / "espeak-ng-data")

        # Tell phonemizer's wrapper where the DLL and its data actually live
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(_lib_path)
        EspeakWrapper.set_data_path(_data_path)

        # Belt-and-suspenders: also set env vars that espeak-ng reads internally
        os.environ["ESPEAK_DATA_PATH"] = _data_path
        os.environ["OUPUT_DIR"] = _data_path  # Intentional typo in espeak-ng source

        # Make the DLL findable on Windows (adds it to the DLL search path)
        _esl.make_library_available()
    except Exception as _e:
        print(f"⚠️  espeakng-loader fix failed: {_e}")
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
from zeko.engine import VoiceResponseEngine


async def main() -> None:
    engine = VoiceResponseEngine.from_env()
    await engine.run()


if __name__ == "__main__":
    # Use a manual event loop instead of asyncio.run() to avoid a Python 3.10
    # Windows bug where asyncio's ProactorEventLoop throws:
    #   AttributeError: 'NoneType' object has no attribute 'close'
    # ...when Ctrl+C is pressed, because the IOCP self-pipe is already None
    # by the time the loop teardown runs.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\n\n🛑 ZEKO shutting down. Goodbye.")
    finally:
        try:
            # Cancel all remaining tasks gracefully
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        try:
            loop.close()
        except Exception:
            pass
