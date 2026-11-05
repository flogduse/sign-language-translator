"""Pure landmark geometry: MediaPipe hand landmarks in, ML-ready features out.

This module has no I/O and no model: every function is trivially testable, which
is what lets the real-time loop stay thin and the maths stay honest.
"""
from __future__ import annotations

import numpy as np

# MediaPipe hand landmark indices (21 points, x/y/z).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
PINKY_MCP = 17

N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3  # 63


def to_features(points) -> np.ndarray:
    """Convert a 21x3 landmark array to a translation/scale/roll invariant vector.

    Steps, each of which targets a real failure mode seen on a live webcam:

    1. **Translate** — subtract the wrist. Without this, the same sign means
       something completely different depending on where the hand sits in frame.
    2. **Scale** — divide by `wrist -> middle-finger MCP` distance. Distance of
       the hand from the camera otherwise dominates every coordinate.
    3. **Rotate** — align the palm direction (wrist -> middle MCP) onto the +X
       axis on the image plane. Raises the camera or leans back and the sign
       stays recognised.

    z is scaled with the same factor, relative to the wrist, so depth still
    informs shape without contributing absolute-camera noise.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(N_LANDMARKS, 3)
    if not np.isfinite(pts).all():
        raise ValueError("landmarks contain NaN or infinite values")

    wrist = pts[WRIST]
    rel = pts - wrist

    scale = float(np.linalg.norm(rel[MIDDLE_MCP][:2]))
    if scale < 1e-6:
        # Degenerate detection (pointer tracking a static image). Returning a
        # near-zero vector is safer than crashing mid-stream.
        return np.zeros(N_FEATURES, dtype=np.float64)
    rel = rel / scale

    # Rotate the image plane so the palm direction is always +X.
    dx, dy = rel[MIDDLE_MCP][0], rel[MIDDLE_MCP][1]
    angle = np.arctan2(dy, dx)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rel[:, :2] = rel[:, :2] @ rot.T

    return rel.reshape(-1)


FINGER_CHAINS = (
    # (tip index, [joint indices from MCP outward]) used to compute curl/cross
    # features by consumers if they need geometric facts about the hand.
    (THUMB_TIP, [1, 2, 3, THUMB_TIP]),
    (INDEX_TIP, [INDEX_MCP, 6, 7, INDEX_TIP]),
    (MIDDLE_TIP, [MIDDLE_MCP, 10, 11, MIDDLE_TIP]),
    (16, [RING_MCP, 14, 15, 16]),
    (20, [PINKY_MCP, 18, 19, 20]),
)
