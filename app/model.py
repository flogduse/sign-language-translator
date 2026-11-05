"""Runtime inference: load the trained model, map class ids to letters.

Kept deliberately boring: everything here is a thin, typed wrapper around the
artefacts `python -m app.train` produces, so the real-time loop reads clean.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from .features import N_FEATURES, to_features

LETTERS = [chr(ord("A") + i) for i in range(26)] + ["DELETE", "NOTHING", "SPACE"]


class FingerspellingModel:
    """Wraps a joblib estimator and the class-id → symbol contract."""

    def __init__(
        self,
        model_path: Path | None = None,
        labels_path: Path | None = None,
    ) -> None:
        root = Path(__file__).resolve().parent.parent
        self.model_path = model_path or root / "models" / "asl_letters.joblib"
        self.labels_path = labels_path or root / "models" / "labels.json"
        self.model = joblib.load(self.model_path)
        self.labels: dict[int, str] = {
            int(k): v for k, v in json.loads(self.labels_path.read_text(encoding="utf-8")).items()
        }

    def predict(self, landmarks) -> tuple[str, float]:
        """Return `(symbol, confidence)` for one 21x3 landmark frame."""
        features = to_features(landmarks)
        if not np.any(features):
            return "", 0.0
        features = features.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            best = int(np.argmax(proba))
            return self.labels.get(self._class_id(best), ""), float(proba[best])
        return self.labels.get(self._class_id(int(self.model.predict(features)[0])), ""), 1.0

    def _class_id(self, position: int) -> int:
        """`predict_proba` columns follow `model.classes_`, not dataset order."""
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return position
        classes = np.asarray(classes).ravel()
        if len(classes) and classes.dtype.kind in "iu":
            return int(classes[position])
        return self._label_to_id(str(classes[position]))

    def _label_to_id(self, label: str) -> int:
        for key, value in self.labels.items():
            if value == label:
                return key
        return LETTERS.index(label)


__all__ = ["FingerspellingModel", "LETTERS", "N_FEATURES", "to_features"]
