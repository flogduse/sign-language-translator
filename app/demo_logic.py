"""Text rules shared by the Streamlit demo and its tests.

The live loop in ``app/realtime.py`` uses ``Stabilizer`` over many frames. The
demo classifies one photo at a time, so it needs only the text step: given one
committed symbol, update the text buffer. Keeping that rule here (free of
Streamlit, MediaPipe and OpenCV) is what makes it testable.
"""
from __future__ import annotations

from .translator import TextSession

CONTROL_SYMBOLS = ("SPACE", "DELETE", "NOTHING")


def apply_symbol(session: TextSession, symbol: str) -> str:
    """Apply one model symbol to the session and return the new text.

    ``SPACE`` inserts a word break, ``DELETE`` removes the last character, and
    ``NOTHING`` is the model's "no letter" class. Anything else that is a single
    letter is appended. Unknown values are ignored rather than guessed.
    """
    if symbol == "SPACE":
        session.space()
    elif symbol == "DELETE":
        session.delete()
    elif len(symbol) == 1 and symbol.isalpha():
        session.push(symbol.upper())
    return session.text


__all__ = ["CONTROL_SYMBOLS", "apply_symbol"]
