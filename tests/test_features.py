"""Feature invariance: normalization must survive what a webcam does to a hand."""
from __future__ import annotations

import numpy as np
import pytest

from app.features import N_FEATURES, to_features

RNG = np.random.default_rng(3)


def _hand(scale=1.0, center=(0.0, 0.0, 0.0)):
    """A random-but-plausible landmark cloud (no uniform noise cloud: distances
    would degenerate)."""
    base = np.zeros((21, 3))
    # palm direction wrist -> middle MCP, plus per-finger offsets
    rng = np.random.default_rng(11)
    joints = rng.uniform(-0.4, 0.4, size=(20, 3))
    joints[9] = np.array([1.0, 0.35, 0.1])  # middle MCP sets palm direction
    pts = np.vstack([np.zeros(3), joints]) * scale
    pts = pts + np.array(center)
    return pts


def test_features_are_translation_invariant():
    a = to_features(_hand())
    b = to_features(_hand(center=(3.2, -1.7, 0.5)))
    assert np.allclose(a, b, atol=1e-9)


def test_features_are_scale_invariant():
    a = to_features(_hand(scale=1.0))
    b = to_features(_hand(scale=7.5))
    assert np.allclose(a, b, atol=1e-9)


def test_features_are_roll_invariant():
    pts = _hand()
    angle = np.deg2rad(63)
    rot = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    rotated = pts.copy()
    rotated[:, :2] = rotated[:, :2] @ rot.T
    assert np.allclose(to_features(pts), to_features(rotated), atol=1e-9)


def test_feature_vector_shape_and_degeneracy():
    assert to_features(_hand()).shape == (N_FEATURES,)
    degenerate = np.zeros((21, 3))  # no palm direction: must not crash
    assert np.allclose(to_features(degenerate), np.zeros(N_FEATURES))


def test_nan_landmarks_are_rejected():
    bad = _hand()
    bad[4][0] = np.nan
    with pytest.raises(ValueError):
        to_features(bad)
