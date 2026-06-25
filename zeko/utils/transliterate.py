"""Malayalam-to-Latin transliteration utility for TTS engines.

Converts Malayalam Unicode script (U+0D00–U+0D7F) to romanized Latin text
so that TTS engines (Gemini, Kokoro) can pronounce it. English text passes
through unchanged. Mixed-script text is handled word-by-word.

Uses the `indic-transliteration` library with ITRANS output scheme.
"""

from __future__ import annotations

import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Malayalam Unicode block: U+0D00 – U+0D7F
_MALAYALAM_RE = re.compile(r"[\u0D00-\u0D7F]")


def contains_malayalam(text: str) -> bool:
    """Check if the text contains any Malayalam Unicode characters."""
    return bool(_MALAYALAM_RE.search(text))


# Malayalam chillu letters — modern Unicode additions that indic-transliteration
# does not always map. We provide explicit Latin fallbacks.
_CHILLU_MAP = {
    "\u0D7B": "n",   # ൻ (chillu-n)
    "\u0D7A": "N",   # ൺ (chillu-NN)
    "\u0D7E": "L",   # ൾ (chillu-L)
    "\u0D7D": "l",   # ൽ (chillu-l)
    "\u0D7C": "r",   # ർ (chillu-r)
    "\u0D7F": "k",   # ൿ (chillu-k)
}


def _cleanup_residual_malayalam(text: str) -> str:
    """Replace any residual Malayalam characters that survived transliteration."""
    for char, latin in _CHILLU_MAP.items():
        text = text.replace(char, latin)
    # Catch any remaining Malayalam Unicode as a safety net
    text = _MALAYALAM_RE.sub("", text)
    return text


def transliterate_for_tts(text: str) -> str:
    """Transliterate Malayalam script to Latin (IAST) for TTS engines.

    - Pure English text passes through unchanged.
    - Pure Malayalam text is fully transliterated.
    - Mixed-script text is handled word-by-word: only words containing
      Malayalam characters are transliterated; English words are kept as-is.

    Returns:
        Romanized text suitable for TTS pronunciation.
    """
    if not contains_malayalam(text):
        return text

    words = text.split()
    result = []
    for word in words:
        if _MALAYALAM_RE.search(word):
            romanized = transliterate(word, sanscript.MALAYALAM, sanscript.IAST)
            romanized = _cleanup_residual_malayalam(romanized)
            result.append(romanized)
        else:
            result.append(word)

    return " ".join(result)
