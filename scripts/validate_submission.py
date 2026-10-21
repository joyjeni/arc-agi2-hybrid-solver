"""
validate_submission.py — Standalone CLI validator for ARC-AGI-2 submission.json.

Usage:
    python scripts/validate_submission.py --file /kaggle/working/submission.json
    python scripts/validate_submission.py --file submission.json --tasks 240
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.submission import validate_submission


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate an ARC-AGI-2 submission.json against competition format rules."
    )
    p.add_argument("--file", type=Path, required=True, help="Path to submission.json.")
    p.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Expected number of task entries (warn if mismatch).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-task validation details.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if not args.file.exists():
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(args.file) as f:
        submission = json.load(f)

    print(f"Loaded submission: {len(submission)} task entries")

    errors = validate_submission(submission)

    if args.tasks is not None and len(submission) != args.tasks:
        print(
            f"WARNING: Expected {args.tasks} tasks, found {len(submission)}."
        )

    if args.verbose:
        for task_id, preds in submission.items():
            status = "OK"
            for entry in preds:
                if "attempt_1" not in entry or "attempt_2" not in entry:
                    status = "MISSING_ATTEMPTS"
                    break
            print(f"  {task_id}: {len(preds)} test input(s) — {status}")

    if errors:
        print(f"\nVALIDATION FAILED — {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(2)
    else:
        print("\nVALIDATION PASSED — submission.json is competition-ready.")


if __name__ == "__main__":
    main()
