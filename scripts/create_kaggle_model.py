"""
create_kaggle_model.py — Documents and automates uploading model artefacts to Kaggle.

This script does NOT train a model.  It packages a local HuggingFace checkpoint
directory and uploads it as a Kaggle model so that the submission notebook can
load it from /kaggle/input/...

Requirements:
    pip install kaggle

Usage:
    python scripts/create_kaggle_model.py \
        --model-dir /path/to/qwen3-4b-checkpoint \
        --model-slug qwen3-4b-grids15-sft139 \
        --variation-slug sft139 \
        --username joyjeni

Environment variables required:
    KAGGLE_USERNAME  — your Kaggle username
    KAGGLE_KEY       — your Kaggle API key  (from https://kaggle.com/settings → API)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upload a HuggingFace checkpoint to Kaggle Models."
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Local directory with HuggingFace model files.")
    p.add_argument("--model-slug", default="qwen3-4b-grids15-sft139",
                   help="Kaggle model slug (lowercase, hyphens).")
    p.add_argument("--variation-slug", default="sft139",
                   help="Variation / version slug.")
    p.add_argument("--framework", default="transformers",
                   help="Framework label (default: transformers).")
    p.add_argument("--username", default=os.environ.get("KAGGLE_USERNAME", ""),
                   help="Kaggle username (or set KAGGLE_USERNAME env var).")
    return p


def _write_model_meta(
    meta_path: Path,
    username: str,
    slug: str,
    variation: str,
    framework: str,
) -> None:
    meta = {
        "ownerSlug": username,
        "title": slug.replace("-", " ").title(),
        "slug": slug,
        "subtitle": "ARC-AGI-2 fine-tuned Qwen3-4B for arc-jeni-2026",
        "isPrivate": False,
        "licenseName": "Apache 2.0",
        "modelInstanceVariations": [
            {
                "framework": framework,
                "overview": f"Variation {variation}",
                "variations": [
                    {
                        "variationSlug": variation,
                        "variationTitle": variation,
                    }
                ],
            }
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = _build_parser().parse_args()

    if not args.username:
        print("ERROR: KAGGLE_USERNAME not set.  Export it or pass --username.",
              file=sys.stderr)
        sys.exit(1)

    if not args.model_dir.exists():
        print(f"ERROR: Model dir not found: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    # Write model metadata alongside model files
    meta_path = args.model_dir / "dataset-metadata.json"
    _write_model_meta(
        meta_path,
        username=args.username,
        slug=args.model_slug,
        variation=args.variation_slug,
        framework=args.framework,
    )
    print(f"Metadata written: {meta_path}")

    # ------------------------------------------------------------------
    # Push via Kaggle CLI
    # ------------------------------------------------------------------
    cmd = [
        "kaggle", "models", "instances", "versions", "create",
        f"{args.username}/{args.model_slug}/{args.framework}/{args.variation_slug}",
        "--path", str(args.model_dir),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("UPLOAD FAILED:")
        print(result.stderr)
        sys.exit(result.returncode)
    print("Upload successful!")
    print(result.stdout)

    print(
        f"\nModel available at:\n"
        f"  https://kaggle.com/models/{args.username}/{args.model_slug}\n"
        f"\nKaggle input path in notebook:\n"
        f"  /kaggle/input/{args.model_slug}/{args.framework}/{args.variation_slug}/1/"
    )


if __name__ == "__main__":
    main()
