"""
benchmark.py — CLI benchmark runner for arc-jeni-2026.

Usage:
    python scripts/benchmark.py --split eval --config configs/pipeline.yaml
    python scripts/benchmark.py --split train --tier 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root on path when invoked directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.submission import score_submission
from src.pipeline import (
    load_challenges,
    load_solutions,
    run_pipeline,
    run_tier1,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark arc-jeni-2026 solver tiers against ARC-AGI-2 data."
    )
    p.add_argument(
        "--split",
        choices=["train", "eval"],
        default="eval",
        help="Dataset split to benchmark (default: eval).",
    )
    p.add_argument(
        "--tier",
        type=int,
        choices=[0, 1, 2, 3],
        default=0,
        help="Tier to run: 0=full pipeline, 1=DSA only, 2=+ProgSynth, 3=+Chakra (default: 0).",
    )
    p.add_argument(
        "--challenges",
        type=Path,
        default=None,
        help="Override path to challenges JSON.",
    )
    p.add_argument(
        "--solutions",
        type=Path,
        default=None,
        help="Override path to solutions JSON (required for scoring).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pipeline.yaml",
        help="Path to pipeline YAML config (default: configs/pipeline.yaml).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit benchmark to first N tasks (for quick sanity checks).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write submission JSON to this path.",
    )
    return p


# ---------------------------------------------------------------------------
# Tier routing helpers
# ---------------------------------------------------------------------------

def _run_tier1_only(challenges: dict) -> dict:
    """Run DSA Tier-1 only and return predictions dict."""
    from src.solvers import REGISTRY  # noqa: F401 — triggers registration
    predictions: dict = {}
    for task_id, task_data in challenges.items():
        result = run_tier1(task_id, task_data)
        predictions[task_id] = result
    return predictions


def _print_tier_breakdown(results: dict) -> None:
    counts: dict[str, int] = {}
    for preds in results.values():
        for entry in preds:
            tier = entry.get("tier", "unknown")
            counts[tier] = counts.get(tier, 0) + 1
    print("\n--- Tier Distribution ---")
    for tier, cnt in sorted(counts.items()):
        print(f"  {tier}: {cnt} predictions")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve data paths
    # ------------------------------------------------------------------
    base_dir = Path("/kaggle/input/arc-prize-2026-arc-agi-2")
    if not base_dir.exists():
        # Local dev fallback
        base_dir = ROOT / "data"

    if args.challenges:
        challenges_path = args.challenges
    elif args.split == "eval":
        challenges_path = base_dir / "arc-agi_evaluation-challenges.json"
    else:
        challenges_path = base_dir / "arc-agi_training-challenges.json"

    if args.solutions:
        solutions_path = args.solutions
    elif args.split == "eval":
        solutions_path = base_dir / "arc-agi_evaluation-solutions.json"
    else:
        solutions_path = base_dir / "arc-agi_training-solutions.json"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading challenges from: {challenges_path}")
    challenges = load_challenges(str(challenges_path))
    if args.limit:
        task_ids = list(challenges.keys())[: args.limit]
        challenges = {k: challenges[k] for k in task_ids}
    print(f"Tasks loaded: {len(challenges)}")

    solutions = {}
    if solutions_path.exists():
        solutions = load_solutions(str(solutions_path))
        if args.limit:
            solutions = {k: solutions[k] for k in challenges if k in solutions}

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------
    t0 = time.time()

    if args.tier == 1:
        print("\nRunning Tier-1 (DSA) only ...")
        predictions = _run_tier1_only(challenges)
    else:
        print(f"\nRunning full pipeline (tier={args.tier or 'all'}) ...")
        predictions = run_pipeline(challenges, config_path=str(args.config))

    elapsed = time.time() - t0
    print(f"\nBenchmark completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    if solutions:
        stats = score_submission(predictions, solutions)
        total = stats.get("total", len(challenges))
        correct = stats.get("correct", 0)
        pct = 100.0 * correct / total if total else 0.0
        print(f"\n=== Score: {correct}/{total} ({pct:.1f}%) ===")

        _print_tier_breakdown(predictions)
    else:
        print("\n(No solutions file — skipping scoring)")

    # ------------------------------------------------------------------
    # Optionally write submission
    # ------------------------------------------------------------------
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(predictions, f)
        print(f"Submission written to: {args.output}")


if __name__ == "__main__":
    main()
