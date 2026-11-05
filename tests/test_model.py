"""Contract test: the bundled model loads and produces a letter.

This is the no-camera smoke test that the real-time runtime uses (`--selftest`
adds a full data pass; this is the CI-sized version).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.model import LETTERS, FingerspellingModel


@pytest.fixture(scope="module")
def model() -> FingerspellingModel:
    return FingerspellingModel()


def test_labels_define_the_29_class_contract(model):
    assert len(model.labels) == 29
    assert model.labels[0] == "A"
    assert model.labels[25] == "Z"
    assert model.labels[26] == "DELETE"
    assert model.labels[27] == "NOTHING"
    assert model.labels[28] == "SPACE"


def test_predict_returns_a_letter_and_confidence(model):
    rng = np.random.default_rng(5)
    landmarks = np.abs(rng.normal(0.5, 0.08, size=(21, 3)))
    letter, prob = model.predict(landmarks)
    assert isinstance(letter, str)
    assert prob == pytest.approx(prob, abs=1e-6)
    if letter:
        assert letter in LETTERS


def test_degenerate_landmarks_yield_no_letter(model):
    letter, prob = model.predict(np.zeros((21, 3)))
    assert letter == ""
    assert prob == 0.0


def test_registry_order_matches_dataset_letters():
    assert LETTERS[0] == "A" and LETTERS[25] == "Z"
