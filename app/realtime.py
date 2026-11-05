"""Real-time fingerspelling: webcam → landmarks → letters → words.

    python -m app.realtime                 # camera 0
    python -m app.realtime --camera 1
    python -m app.realtime --selftest      # no camera: re-scores the test split

Controls (in the open window):
    SPACE  finish the current word
    X      delete the last letter
    C      clear the text
    Q      quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .features import to_features
from .landmarks import HandTracker, draw_landmarks
from .model import FingerspellingModel
from .translator import Stabilizer, TextSession

_ACCENT = (60, 200, 120)


def run_loop(camera: int, stabilizer: Stabilizer, session: TextSession, model: FingerspellingModel) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {camera} — run with --camera, or try --selftest")
    try:
        tracker = HandTracker()
    except Exception as exc:  # model download/setup failures surface readably
        cap.release()
        raise SystemExit(f"could not initialise hand tracking: {exc}")

    last_commit = ""
    fps, last_t = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # selfie view: gestures match intuition
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-6))
        last_t = now

        hands = tracker.detect(frame)
        if hands:
            draw_landmarks(frame, hands[0])
            letter, prob = model.predict(hands[0])
            committed = stabilizer.push(letter, prob)
            if committed == "SPACE":
                session.space()
                last_commit = "⟨SPACE⟩"
            elif committed == "DELETE":
                session.delete()
                last_commit = "⟨DELETE⟩"
            elif committed:
                session.push(committed)
                last_commit = committed
            draw_hud(frame, letter, prob, stabilizer.symbol, last_commit, session.text, fps)
        else:
            stabilizer.symbol = ""
            draw_hud(frame, "", 0.0, "", "no hand", session.text, fps)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == 32:  # space key
            session.space()
        elif key in (ord("x"), 8):
            session.delete()
        elif key == ord("c"):
            session.clear()

    cap.release()
    cv2.destroyAllWindows()


def draw_hud(
    frame: np.ndarray,
    letter: str,
    prob: float,
    held: str,
    last_commit: str,
    text: str,
    fps: float,
) -> np.ndarray:
    """Overlay: big current guess, bars, committed text, controls."""
    height, width = frame.shape[:2]
    band = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (0.82 * frame + 0.18 * band).astype(np.uint8)

    def put(string: str, pos, scale: float, color: tuple = (255, 255, 255)) -> None:
        cv2.putText(frame, string, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    put(f"fps {fps:0.1f}", (10, 30), 0.7)
    if letter:
        color = _ACCENT if prob >= 0.75 else (0, 160, 255)
        put(f"{letter}  {prob * 100:0.0f}%", (18, 108), 1.9, color)
    else:
        put("show a hand to the camera", (18, 108), 0.9, (0, 160, 255))

    if held:
        put(f"committed: {held}", (18, 156), 0.7, _ACCENT)
    put(f"text: {text[-40:]}", (18, height - 60), 1.1, (255, 255, 210))
    put("q quit · space word · x delete · c clear", (18, height - 20), 0.6, (180, 180, 180))
    return frame


def selftest(model: FingerspellingModel, data: Path, samples: int = 400, seed: int = 7) -> int:
    """Re-score the bundled model on a held-out slice of the landmark corpus.

    This is the no-camera smoke test: it catches a broken download, a feature
    mismatch between the training script and the runtime, or a stale model
    before someone spends an evening picking a bug apart in front of a webcam.
    """
    raw = np.load(data)
    landmarks, labels = raw["landmarks"], raw["labels"].astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=min(samples, len(labels)), replace=False)
    correct = 0
    misses: dict[str, tuple[str, str]] = {}
    for i in idx:
        letter, _prob = model.predict(landmarks[i])
        label = model.labels.get(int(labels[i]), "…")
        if letter == label:
            correct += 1
        else:
            misses.setdefault(f"{label}->{letter}", (label, letter))
    acc = correct / len(idx)
    print(f"selftest: {correct}/{len(idx)} = {acc * 100:.1f}% on bundled corpus sample")
    if misses:
        pairs = ", ".join(sorted(misses)[:12])
        print(f"typical misses: {pairs}")
    return 0 if acc >= 0.90 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--selftest", action="store_true", help="no camera: score the bundled model on a data slice")
    parser.add_argument("--data", type=Path, default=Path("data/asl_landmarks.npz"))
    args = parser.parse_args(argv)

    model = FingerspellingModel()
    session = TextSession()
    stabilizer = Stabilizer()

    if args.selftest:
        if not args.data.exists():
            print(
                "selftest needs the landmark corpus — run "
                "`python scripts/download_data.py` first",
                file=sys.stderr,
            )
            return 2
        return selftest(model, args.data)

    run_loop(args.camera, stabilizer, session, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
