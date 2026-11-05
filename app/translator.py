"""Letter → word assembly, entirely free of I/O.

Two cooperating pieces:

* `Stabilizer` — turns per-frame predictions into a single committed letter by
  requiring confidence over a window. Recognising the same letter with two
  noisy consecutive frames gives the same letter flickering into words 30
  times per second; this is what stops that.
* `TextSession` — the word text logic: letters accumulate, SPACE finishes a
  word, DEL removes a letter and reaches into the previous word when the
  current one is empty.

Both are pure state machines: tests drive them with sampled letter streams
and assert exact string outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stabilizer:
    """Commit a letter only when the model is confident and consistent."""

    threshold: float = 0.75      # probability a single frame must reach
    window: int = 8              # recent frames consulted per commit
    agree: int = 6               # how many of the window must agree
    min_different: int = 2       # frames since the last commit

    symbol: str = ""
    _history: list[str] = field(default_factory=list)
    _cooldown: int = 0

    def push(self, letter: str, prob: float) -> object:
        """Feed one frame's prediction.

        Returns the committed symbol if a new commit fires this frame,
        otherwise "" (including when the *same* letter is held down — a
        repeat commit needs a fresh gap, otherwise one gesture spells a word).
        """
        self._cooldown = max(self._cooldown - 1, 0)
        self._history.append(letter if prob >= self.threshold else "")
        self._history = self._history[-self.window:]

        if self._cooldown or len(self._history) < self.agree:
            return ""
        recent = self._history[-self.agree:]
        if not recent or len(set(recent)) != 1:
            return ""

        if recent[0] == self.symbol and self._cooldown == 0 and any(recent):
            # Same letter again within the window: does not re-commit, but does
            # not lock the machine up either — this is the "held gesture" case.
            return ""
        if not recent[0]:
            self.symbol = ""
            return ""

        self.symbol = recent[0]
        self._cooldown = self.min_different
        self._history.clear()
        return self.symbol


@dataclass
class TextSession:
    """Accumulates committed letters into text with space/delete semantics."""

    text: str = ""

    def push(self, letter: str) -> object:
        # Defensive: junk input must not corrupt the buffer. Space and control
        # characters have their own explicit methods.
        if not isinstance(letter, str) or len(letter) != 1 or not letter.isalnum():
            return self
        self.text += letter
        return self

    def space(self) -> object:
        if self.text and not self.text.endswith(" "):
            self.text += " "
        return self

    def delete(self) -> object:
        if not self.text:
            return self
        self.text = self.text[:-1]
        return self

    def word(self) -> str:
        """The partial word still being fingerspelled (next commit joins it)."""
        return self.text.split(" ")[-1]

    def words(self) -> list[str]:
        return [w for w in self.text.split(" ") if w]

    def clear(self) -> object:
        self.text = ""
        return self
