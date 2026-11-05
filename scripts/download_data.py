#!/usr/bin/env python3
"""Fetch the landmark corpus the project is trained against.

    python scripts/download_data.py

Downloads `data/asl_landmarks.npz` — 63,580 MediaPipe hand-landmark samples
across 29 classes (A-Z, del, nothing, space) — from the derivation repo it was
published in, with the source dataset credited in the README.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data" / "asl_landmarks.npz"
SOURCE = (
    "https://raw.githubusercontent.com/AbhisumatK/Real-Time-ASL-Alphabet-"
    "Recognition-System/HEAD/data/asl_landmarks.npz"
)


def main() -> int:
    if TARGET.exists():
        print(f"already present: {TARGET}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print("downloading ~14 MB …")
    urllib.request.urlretrieve(SOURCE, str(TARGET))
    size_mb = TARGET.stat().st_size / (1024 * 1024)
    print(f"saved {TARGET} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
