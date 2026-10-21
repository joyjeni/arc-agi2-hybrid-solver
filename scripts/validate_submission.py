"""
validate_submission.py — Standalone CLI validator for ARC-AGI-2 submission.json.

Usage:
    python scripts/validate_submission.py --file /kaggle/working/submission.json
    python scripts/validate_submission.py --file submission.json --challenges /tmp/arcdata/arc-agi_evaluation_challenges.json
    python scripts/validate_submission.py --file submission.json --tasks 240
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate an ARC-AGI-2 submission.json against competition format rules."
    )
    p.add_argument("--file", type=Path, required=True, help="Path to submission.json.")
    p.add_argument(
        "--challenges",
        type=Path,
        default=None,
        help="Optional path to challenges JSON for full per-task validation.",
    )
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


def _format_only_check(submission: dict) -> list[str]:
    """Basic structure check — does not require a TaskContext map."""
    errors: list[str] = []
    for task_id, preds in submission.items():
        if not isinstance(preds, list):
            errors.append(f"{task_id}: value is not a list")
            continue
        for i, entry in enumerate(preds):
            if not isinstance(entry, dict):
                errors.append(f"{task_id}[{i}]: entry is not a dict")
                continue
            for key in ("attempt_1", "attempt_2"):
                if key not in entry:
                    errors.append(f"{task_id}[{i}]: missing '{key}'")
                    continue
                grid = entry[key]
                if not isinstance(grid, list) or not grid:
                    errors.append(f"{task_id}[{i}].{key}: not a non-empty list")
                    continue
                if not all(isinstance(r, list) for r in grid):
                    errors.append(f"{task_id}[{i}].{key}: rows must be lists")
                    continue
                w0 = len(grid[0])
                for r in grid:
                    if len(r) != w0:
                        errors.append(f"{task_id}[{i}].{key}: non-rectangular grid")
                    for c in r:
                        if not isinstance(c, int) or not (0 <= c <= 9):
                            errors.append(
                                f"{task_id}[{i}].{key}: cell {c!r} out of range [0-9]"
                            )
    return errors


def main() -> None:
    args = _build_parser().parse_args()

    if not args.file.exists():
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    with open(args.file) as f:
        submission = json.load(f)

    print(f"Loaded submission: {len(submission)} task entries")

    # ------------------------------------------------------------------
    # Validation — full (with tasks) or format-only
    # ------------------------------------------------------------------
    if args.challenges and args.challenges.exists():
        from src.core.submission import validate_submission
        from src.pipeline import load_challenges

        tasks = load_challenges(str(args.challenges))
        errors = validate_submission(submission, tasks)
    else:
        errors = _format_only_check(submission)

    if args.tasks is not None and len(submission) != args.tasks:
        print(f"WARNING: Expected {args.tasks} tasks, found {len(submission)}.")

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
        for err in errors[:20]:
            print(f"  - {err}")
        sys.exit(2)
    else:
        print("\nVALIDATION PASSED — submission.json is competition-ready.")


if __name__ == "__main__":
    main()
