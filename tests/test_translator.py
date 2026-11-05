"""The letter/word logic alone, driven by synthetic prediction streams."""
from __future__ import annotations

from app.translator import Stabilizer, TextSession


def _letters(stabilizer: Stabilizer, frames):
    """Run a frames list (letter, prob) and return committed symbols in order."""
    committed = []
    for letter, prob in frames:
        c = stabilizer.push(letter, prob)
        if c:
            committed.append(c)
    return committed


def test_stabilizer_commits_when_confident_and_consistent():
    p = Stabilizer(threshold=0.7, window=6, agree=4)
    frames = [("A", 0.9)] * 4
    assert _letters(p, frames) == ["A"]
    assert p.symbol == "A"


def test_stabilizer_ignores_low_confidence_noise():
    p = Stabilizer(threshold=0.7, window=6, agree=4)
    frames = [("A", 0.2)] * 8
    assert _letters(p, frames) == []


def test_stabilizer_i_handles_freakish_change():
    """Two confident letters in a row commit both."""
    p = Stabilizer(threshold=0.7, window=6, agree=3)
    frames = [("A", 0.95)] * 3 + [("S", 0.95)] * 3
    committed = _letters(p, frames)
    assert committed == ["A", "S"]


def test_stabilizer_hold_does_not_spammer_a_word():
    """The same gesture held down commits ONCE, not per frame."""
    p = Stabilizer(threshold=0.7, window=6, agree=4)
    frames = [("A", 0.9)] * 14
    assert _letters(p, frames) == ["A"]


def test_session_composes_words_spaces_and_deletes():
    s = TextSession()
    for c in "HI":
        s.push(c)
    s.space()
    for c in "THERE":
        s.push(c)
    assert s.text == "HI THERE"
    s.delete()
    assert s.text == "HI THER"
    assert s.word() == "THER"


def test_delete_reaches_into_previous_word():
    s = TextSession()
    s.push("G")
    s.push("O")
    s.space()
    assert s.text == "GO "
    s.delete()
    assert s.text == "GO"


def test_session_ignores_junk():
    s = TextSession()
    s.push("A")
    s.push(" ")
    s.push("!!")
    s.push("")
    assert s.text == "A"


def test_end_to_end_letters_become_text():
    p = Stabilizer(threshold=0.7, window=8, agree=5)
    s = TextSession()
    for letter in "CAT":
        for _ in range(5):
            symbol = p.push(letter, 0.93)
            if symbol == letter:
                break
        s.push(letter)
    assert s.text == "CAT"
