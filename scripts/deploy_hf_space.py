#!/usr/bin/env python3
"""Publish the single-frame Streamlit demo as a Hugging Face Space.

The Space runs the same classifier and the same MediaPipe tracker as the local
app; only the input changes (browser camera instead of ``cv2.VideoCapture``).
No secret or GPU is required.

Usage
-----
    pip install huggingface_hub
    hf auth login                       # or pass --token

    python scripts/deploy_hf_space.py --repo-id flogduse/asl-fingerspelling
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE_README = ROOT / "deploy" / "huggingface" / "README.md"
DEMO_REQUIREMENTS = ROOT / "requirements-demo.txt"

IGNORE = [
    ".git/*",
    ".github/*",
    ".venv/*",
    ".env",
    "__pycache__/*",
    "**/__pycache__/*",
    "*.pyc",
    ".pytest_cache/*",
    "tests/*",
    "docs/*",
    "scripts/*",
    "deploy/*",
    "requirements-dev.txt",
    "requirements-demo.txt",
    "*.log",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_SPACE_ID", "flogduse/asl-fingerspelling"),
        help="Space id, e.g. username/space-name",
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF access token")
    parser.add_argument("--private", action="store_true", help="Create a private Space")
    args = parser.parse_args()

    if not args.token:
        print(
            "error: pass --token or set HF_TOKEN "
            "(create one at https://huggingface.co/settings/tokens)",
            file=sys.stderr,
        )
        return 2
    for required in (SPACE_README, DEMO_REQUIREMENTS):
        if not required.exists():
            print(f"error: missing {required}", file=sys.stderr)
            return 2

    api = HfApi(token=args.token)

    print(f"→ creating/updating Space {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="streamlit",
        private=args.private,
        exist_ok=True,
    )

    print("→ uploading app, model and labels")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Deploy ASL fingerspelling demo",
    )

    # Spaces read requirements from the root requirements.txt; the GitHub one is
    # the core runtime only, so the demo extras are uploaded under that name.
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="space",
        path_or_fileobj=str(DEMO_REQUIREMENTS),
        path_in_repo="requirements.txt",
        commit_message="Install Streamlit and Pillow",
    )

    # The Space also reads its metadata from the root README.md.
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="space",
        path_or_fileobj=str(SPACE_README),
        path_in_repo="README.md",
        commit_message="Add Space metadata",
    )

    print(f"✓ done: https://huggingface.co/spaces/{args.repo_id}")
    print("  first build installs MediaPipe and Streamlit; give it a few minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
