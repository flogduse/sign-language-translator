"""Dataset-contract tests: a dropped class must be reported, never silent.

`metrics.json` publishes three things that have to agree: `class_ids` (what the
classifier emits), `classes` (their display names) and `dropped_classes` (what
`MIN_CLASS_SAMPLES` removed). These tests pin them together so the published
class count cannot drift away from what was actually measured.
"""
from __future__ import annotations

import numpy as np

from app.train import LETTERS, load_dataset


def _write_corpus(path, per_class: dict[int, int]) -> None:
    rng = np.random.default_rng(0)
    labels = np.concatenate(
        [np.full(n, cls, dtype=np.int64) for cls, n in per_class.items()]
    )
    landmarks = np.abs(rng.normal(0.5, 0.05, size=(len(labels), 21, 3)))
    np.savez(path, landmarks=landmarks, labels=labels)


def test_under_populated_class_is_reported_not_silently_dropped(tmp_path):
    path = tmp_path / "corpus.npz"
    _write_corpus(path, {0: 8, 1: 8, 27: 1})

    ds = load_dataset(path)

    assert ds.dropped == {"NOTHING": 1}
    assert ds.class_ids == [0, 1]
    assert ds.class_names == ["A", "B"]
    assert len(ds.y) == 16  # the single NOTHING sample left the split
    assert len(ds.X) == len(ds.y)


def test_class_ids_keep_their_real_ids_instead_of_being_renumbered(tmp_path):
    """A gap in the id space must survive as a gap: 26 then 28, not 26 then 27."""
    path = tmp_path / "corpus.npz"
    _write_corpus(path, {26: 6, 28: 6})

    ds = load_dataset(path)

    assert ds.class_ids == [26, 28]
    assert ds.class_names == ["DELETE", "SPACE"]
    assert ds.dropped == {}
    # Display names come from the 29-entry contract, not from list position.
    assert LETTERS[ds.class_ids[-1]] == "SPACE"
