"""Tests for the demo's text rules (no Streamlit, no camera, no model)."""
from __future__ import annotations

from app.demo_logic import apply_symbol
from app.translator import TextSession


def test_letters_append_uppercase():
    session = TextSession()
    for symbol in ("A", "S", "L"):
        assert apply_symbol(session, symbol) == session.text
    assert session.text == "ASL"


def test_space_finishes_a_word_and_does_not_double_up():
    session = TextSession()
    apply_symbol(session, "H")
    apply_symbol(session, "I")
    apply_symbol(session, "SPACE")
    apply_symbol(session, "SPACE")
    apply_symbol(session, "Y")
    assert session.text == "HI Y"
    assert session.words() == ["HI", "Y"]


def test_delete_removes_one_character_and_crosses_words():
    session = TextSession(text="HI ")
    apply_symbol(session, "DELETE")
    apply_symbol(session, "DELETE")
    assert session.text == "H"
    apply_symbol(session, "DELETE")
    apply_symbol(session, "DELETE")
    assert session.text == ""


def test_nothing_is_ignored():
    session = TextSession(text="OK")
    assert apply_symbol(session, "NOTHING") == "OK"


def test_unknown_symbols_are_ignored_not_guessed():
    session = TextSession(text="A")
    assert apply_symbol(session, "") == "A"
    assert apply_symbol(session, "THUMBS_UP") == "A"
    assert apply_symbol(session, "1") == "A"
    assert session.text == "A"
