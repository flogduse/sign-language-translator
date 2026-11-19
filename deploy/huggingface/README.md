---
title: ASL Fingerspelling
emoji: 🤟
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app/streamlit_demo.py
pinned: false
---

# ASL Fingerspelling — single-frame demo

Photograph one handshape and get the predicted letter, built from the
[sign-language-translator](https://github.com/flogduse/sign-language-translator)
model: MediaPipe hand landmarks, translation/scale/roll-invariant features, and a
HistGradientBoosting classifier (97.8% top-1 on the held-out split).

**Scope:** one photo per letter, one hand. The motion letters J and Z are not
recognised. The full project runs a live webcam loop with temporal
stabilisation; this Space exists so the model can be tried without installing
anything.
