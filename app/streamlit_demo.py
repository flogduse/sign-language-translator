"""Streamlit demo: photograph a hand, get one fingerspelled letter.

This is deliberately narrower than the real-time app:

* the live loop (``python -m app.realtime``) sees ~30 frames per second and uses
  ``Stabilizer`` to suppress flicker;
* this demo classifies a single photo per letter and appends it to a text
  buffer when the model is confident.

That difference is stated in the UI so nobody mistakes it for continuous
translation. It exists because a hosted Space cannot open a local camera the
way ``cv2.VideoCapture`` does, but it *can* use the browser camera through
``st.camera_input``.

Run locally:
    streamlit run app/streamlit_demo.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# `streamlit run app/streamlit_demo.py` puts app/ on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.demo_logic import apply_symbol  # noqa: E402
from app.landmarks import HandTracker, draw_landmarks  # noqa: E402
from app.model import FingerspellingModel  # noqa: E402
from app.translator import TextSession  # noqa: E402

MIN_CONFIDENCE = 0.55
DISPLAY_WARNING = 0.75

st.set_page_config(page_title="ASL Fingerspelling", page_icon="🤟", layout="wide")


@st.cache_resource(show_spinner="Loading classifier…")
def load_model() -> FingerspellingModel:
    return FingerspellingModel()


@st.cache_resource(show_spinner="Loading hand tracker…")
def load_tracker() -> HandTracker:
    return HandTracker()


def analyse(image: Image.Image) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return an annotated RGB frame and the detected 21x3 landmarks."""
    rgb = np.asarray(image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    hands = load_tracker().detect(bgr)
    for hand in hands:
        draw_landmarks(bgr, hand)
    return bgr[:, :, ::-1], hands


def reset_text() -> None:
    st.session_state.text = ""
    st.session_state.pop("last_hash", None)


def forget_last_image() -> None:
    if "last_hash" in st.session_state:
        del st.session_state["last_hash"]


if "text" not in st.session_state:
    st.session_state.text = ""

st.title("🤟 ASL Fingerspelling — single-frame demo")
st.caption(
    "Photograph one handshape, get one letter. The full project recognises a live "
    "webcam stream with temporal stabilisation — this hosted version trades that "
    "for a browser camera."
)

with st.sidebar:
    st.subheader("How to use it")
    st.markdown(
        "1. Allow the camera (or upload a photo).\n"
        "2. Hold **one** hand in frame, palm toward the camera.\n"
        "3. A confident prediction is appended automatically.\n"
        "4. Use the buttons for space, delete and clear."
    )
    st.warning(
        "Single-frame scope: **J and Z are motion letters** and will not be "
        "recognised. Only one hand is tracked. Confidence below "
        f"{DISPLAY_WARNING:.0%} is shown but can be wrong."
    )
    st.markdown(
        "Source and evaluation: "
        "[github.com/flogduse/sign-language-translator](https://github.com/flogduse/sign-language-translator)"
    )

left, right = st.columns([3, 2], gap="large")

with left:
    camera = st.camera_input("Camera")
    upload = st.file_uploader("…or upload a photo", type=["jpg", "jpeg", "png"])
    source = camera or upload

annotated = None
symbol, confidence, hand_count = "", 0.0, 0

if source is not None:
    raw = source.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    image = Image.open(source)

    annotated, hands = analyse(image)
    hand_count = len(hands)

    if not hands:
        st.error("No hand found — keep the whole hand in frame and try again.")
        forget_last_image()
    else:
        symbol, confidence = load_model().predict(hands[0])

        # Append exactly once per captured image, not on every Streamlit rerun
        # (button clicks rerun the whole script).
        if digest != st.session_state.get("last_hash"):
            st.session_state.last_hash = digest
            if symbol and confidence >= MIN_CONFIDENCE:
                session = TextSession(text=st.session_state.text)
                apply_symbol(session, symbol)
                st.session_state.text = session.text

with right:
    if annotated is not None:
        st.image(annotated, caption="Landmarks used by the classifier", use_container_width=True)

    st.metric("Detected symbol", symbol or "—")
    st.progress(min(max(confidence, 0.0), 1.0), text=f"confidence {confidence:.0%}")
    st.caption(f"Hands in frame: {hand_count}")

    if symbol and confidence < DISPLAY_WARNING:
        st.info("Below the comfortable range — hold the hand steadier and retake.")

st.divider()
st.subheader("Text")
st.code(st.session_state.text or " ", language=None)

session = TextSession(text=st.session_state.text)
col_space, col_delete, col_clear = st.columns(3)
if col_space.button("Space", use_container_width=True):
    apply_symbol(session, "SPACE")
    st.session_state.text = session.text
    st.rerun()
if col_delete.button("Delete", use_container_width=True):
    apply_symbol(session, "DELETE")
    st.session_state.text = session.text
    st.rerun()
if col_clear.button("Clear", use_container_width=True):
    reset_text()
    st.rerun()
