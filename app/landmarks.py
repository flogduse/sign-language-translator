"""MediaPipe hand-landmark extraction for the live camera loop.

One class with one method keeps the real-time loop readable: `detect(bgr_frame)`
returns zero or more `(21, 3)` landmark arrays. Drawing utilities live beside it
so the loop is nothing but frames → landmarks → letters → words.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# (start, end) landmark index pairs — draws the correct 21-point skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def ensure_model(path: Path) -> Path:
    """Download the MediaPipe hand landmarker model if it is not on disk."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand landmarker model ({path.parent.name}/{path.name}) …")
    urllib.request.urlretrieve(MODEL_URL, str(path))
    return path


class HandTracker:
    """Hands-in, landmarks-out. Operating in VIDEO mode keeps tracking stable."""

    def __init__(self, model_path: Path | None = None, num_hands: int = 1) -> None:
        path = ensure_model(model_path or Path("models/hand_landmarker.task"))
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp = 0

    def detect(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 1  # monotonic, required by the VIDEO runner
        result = self._detector.detect_for_video(image, self._timestamp)
        return [
            np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float64).reshape(21, 3)
            for lm in result.hand_landmarks or ()
        ]


def draw_landmarks(frame: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """Skeleton overlay, hand-rolled because the mp.solutions drawing helpers
    do not exist in the Tasks API."""
    height, width = frame.shape[:2]
    pts = hand[:, :2] * np.array([width, height])
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), (40, 220, 140), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, tuple(np.int32(p)), 4, (250, 100, 80), -1, cv2.LINE_AA)
    return frame
