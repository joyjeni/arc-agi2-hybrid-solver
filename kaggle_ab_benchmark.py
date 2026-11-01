#!/usr/bin/env python3
"""
kaggle_ab_benchmark.py  —  A/B style benchmark for jeni_llm_fast.py
=====================================================================
Runs on Kaggle (4x L4 GPUs) to compare LLM prompt styles on ARC training tasks.
Usage:
    python kaggle_ab_benchmark.py --tasks 100 --styles nvarc compact --out /kaggle/working/ab_result.json
"""
import argparse, json, subprocess, sys, tempfile, os
from pathlib import Path

_TRAINING_CHALLENGES = "/kaggle/input/arc-prize-2026/arc-agi_training_challenges.json"
_TRAINING_SOLUTIONS  = "/kaggle/input/arc-prize-2026/arc-agi_training_solutions.json"
_MODEL_PATH          = "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1"

def score_predictions(preds: dict, solutions: dict) -> dict:
    """Return per-task correctness dict."""
    correct = {}
    for tid, task_preds in preds.items():
        if tid not in solutions:
            continue
        sol = solutions[tid]
        task_correct = False
        for panel_pred, panel_sol in zip(task_preds, sol):
            a1 = panel_pred.get("attempt_1")
            a2 = panel_pred.get("attempt_2")
            if a1 == panel_sol or a2 == panel_sol:
                task_correct = True
                break
        correct[tid] = task_correct
    return correct

def run_style(style: str, task_ids: list, challenges: dict, model_path: str,
              timeout: int = 3600) -> dict:
    """Run jeni_llm_fast.py with a forced single style override."""
    # Write a temp challenges file with only the selected tasks
    tmp_challenges = {tid: challenges[tid] for tid in task_ids if tid in challenges}
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tmp_challenges, f)
        tmp_in = f.name
    tmp_out = tmp_in.replace(".json", "_preds.json")
    
    # Patch env to inject forced style
    env = dict(os.environ,
               ARC_MODEL_PATH=model_path,
               ARC_LLM_FORCE_STYLE=style,  # jeni_llm_fast.py reads this to override
               CUDA_VISIBLE_DEVICES="0,1,2,3")
    
    cmd = [sys.executable, "jeni_llm_fast.py",
           "--challenges", tmp_in,
           "--output",     tmp_out,
           "--model-path", model_path]
    
    print(f"\n[AB] Running style='{style}' on {len(task_ids)} tasks…")
    rc = subprocess.call(cmd, env=env, timeout=timeout)
    print(f"[AB] style='{style}' exit code: {rc}")
    
    if Path(tmp_out).exists():
        with open(tmp_out) as f:
            preds = json.load(f)
    else:
        preds = {}
    
    Path(tmp_in).unlink(missing_ok=True)
    Path(tmp_out).unlink(missing_ok=True)
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  type=int, default=100,
                    help="Number of training tasks to benchmark per style")
    ap.add_argument("--styles", nargs="+", default=["nvarc", "compact"],
                    help="Styles to compare")
    ap.add_argument("--out", default="/kaggle/working/ab_result.json")
    ap.add_argument("--model-path", default=_MODEL_PATH)
    args = ap.parse_args()
    
    # Load data
    with open(_TRAINING_CHALLENGES) as f:
        challenges = json.load(f)
    with open(_TRAINING_SOLUTIONS) as f:
        solutions = json.load(f)
    
    # Select unsolved-by-DSA tasks for a fair LLM-only comparison
    # (or just the first N for speed)
    task_ids = list(challenges.keys())[:args.tasks]
    print(f"[AB] Benchmark: {len(task_ids)} tasks, styles: {args.styles}")
    
    results = {}
    for style in args.styles:
        preds   = run_style(style, task_ids, challenges, args.model_path)
        correct = score_predictions(preds, solutions)
        n_correct = sum(correct.values())
        n_total   = len(correct)
        accuracy  = n_correct / n_total if n_total else 0.0
        results[style] = {
            "correct":  n_correct,
            "total":    n_total,
            "accuracy": accuracy,
            "task_ids_correct": [t for t, v in correct.items() if v],
        }
        print(f"  {style:12s}: {n_correct}/{n_total}  ({accuracy*100:.1f}%)")
    
    # Summary
    print("\n══ A/B Result ══")
    styles = list(results.keys())
    if len(styles) == 2:
        a, b = styles
        delta = results[b]["accuracy"] - results[a]["accuracy"]
        print(f"  {b} vs {a}: {'+'if delta>=0 else ''}{delta*100:.1f}% accuracy delta")
    
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[AB] Results saved → {args.out}")

if __name__ == "__main__":
    main()
