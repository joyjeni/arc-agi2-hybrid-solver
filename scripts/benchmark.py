"""
benchmark.py — CLI benchmark runner for arc-jeni-2026.

Usage:
    python scripts/benchmark.py --split eval --config configs/pipeline.yaml
    python scripts/benchmark.py --split train --tier 1 --limit 20
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

from src.core.submission import build_submission, score_submission
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
    """Run DSA Tier-1 only; return competition-format submission dict."""
    from src.solvers import REGISTRY  # noqa: F401 — triggers solver registration

    tier1_results = run_tier1(challenges, already_solved=set(), registry=REGISTRY)
    return build_submission(challenges, tier1_results, fallback_input=True)


def _print_tier_summary(tier1_results_count: int, total: int) -> None:
    print("\n--- Tier Distribution ---")
    print(f"  Tier-1 (DSA):    {tier1_results_count} solved")
    print(f"  Fallback:        {total - tier1_results_count} (identity/black)")


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
        base_dir = ROOT / "data"

    if args.challenges:
        challenges_path = args.challenges
    elif args.split == "eval":
        challenges_path = base_dir / "arc-agi_evaluation_challenges.json"
    else:
        challenges_path = base_dir / "arc-agi_training_challenges.json"

    if args.solutions:
        solutions_path = args.solutions
    elif args.split == "eval":
        solutions_path = base_dir / "arc-agi_evaluation_solutions.json"
    else:
        solutions_path = base_dir / "arc-agi_training_solutions.json"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading challenges from: {challenges_path}")
    challenges = load_challenges(str(challenges_path))
    if args.limit:
        task_ids = list(challenges.keys())[: args.limit]
        challenges = {k: challenges[k] for k in task_ids}
    print(f"Tasks loaded: {len(challenges)}")

    solutions: dict = {}
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
        from src.solvers import REGISTRY  # noqa: F401 — triggers registration

        tier1_results = run_tier1(challenges, already_solved=set(), registry=REGISTRY)
        predictions = build_submission(challenges, tier1_results, fallback_input=True)
        t1_solved = len(tier1_results)
    else:
        print(f"\nRunning full pipeline (tier={args.tier or 'all'}) ...")
        out_path = str(args.output) if args.output else "/tmp/_pipeline_submission.json"
        predictions = run_pipeline(
            challenges_path=str(challenges_path),
            output_path=out_path,
            solutions_path=str(solutions_path) if solutions_path.exists() else None,
        )
        t1_solved = 0  # breakdown not available for full pipeline here

    elapsed = time.time() - t0
    print(f"\nBenchmark completed in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    if solutions:
        accuracy = score_submission(predictions, solutions)
        # Count total test inputs in solutions
        total_inputs = sum(len(v) for v in solutions.values() if isinstance(v, list))
        correct = round(accuracy * total_inputs)
        pct = 100.0 * accuracy
        print(f"\n=== Score: {correct}/{total_inputs} test-inputs ({pct:.1f}%) ===")
        n_tasks = len(challenges)
        print(f"    Tasks benchmarked: {n_tasks}")
        if args.tier == 1:
            _print_tier_summary(t1_solved, n_tasks)
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
