"""Smoke test: the Streamlit demo boots with no camera attached.

Skipped unless the demo extras are installed (``pip install -r
requirements-demo.txt``), so the core CI keeps its small dependency set.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="install requirements-demo.txt to run this")

from streamlit.testing.v1 import AppTest  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent / "app" / "streamlit_demo.py"


def test_demo_boots_and_renders_controls():
    at = AppTest.from_file(str(DEMO), default_timeout=120).run()
    assert not at.exception
    assert [b.label for b in at.button] == ["Space", "Delete", "Clear"]
