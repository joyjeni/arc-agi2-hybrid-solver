"""
fp_audit.py — False-positive audit for advanced_solvers 16–26.

For every solver, tries it on every training task.
Reports: solved, wrong (false positives), skipped (not attempted).
"""
from __future__ import annotations
import sys, json, re as _re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.solvers import REGISTRY
from src.pipeline import load_challenges, load_solutions

# ── data ────────────────────────────────────────────────────────────────────
challenges = load_challenges(str(ROOT / "data/arc-agi_training_challenges.json"))
solutions  = load_solutions (str(ROOT / "data/arc-agi_training_solutions.json"))

# ── solver name → module mapping ────────────────────────────────────────────
SOLVER_MODULE: dict[str, int] = {}
for num in range(16, 27):
    txt = (ROOT / f"src/solvers/advanced_solvers{num}.py").read_text()
    for cls in _re.findall(r'@register_solver\s+class\s+(\w+)', txt, _re.S):
        SOLVER_MODULE[cls] = num

TARGET_SOLVERS = list(SOLVER_MODULE.keys())
print(f"Auditing {len(TARGET_SOLVERS)} solvers across advanced_solvers16-26")
print(f"Training tasks: {len(challenges)}\n")

# ── audit ────────────────────────────────────────────────────────────────────
results: dict[str, dict] = {}

for solver_name in TARGET_SOLVERS:
    solver = REGISTRY.get(solver_name)
    solved = []
    wrong  = []
    errors = []

    for task_id, ctx in challenges.items():
        try:
            result = solver.solve(ctx)
        except Exception as e:
            errors.append((task_id, str(e)[:100]))
            continue

        if result is None or not result.predictions:
            continue  # not attempted

        pred = result.predictions[0].attempt_1  # Grid for first test input

        gt_list = solutions.get(task_id)
        if gt_list is None:
            continue

        gt = gt_list[0]  # first test output [[int,...],...]

        # normalise to plain list
        if hasattr(pred, 'tolist'):
            pred_list = pred.tolist()
        elif isinstance(pred, list):
            pred_list = [list(r) if not isinstance(r, list) else r for r in pred]
        else:
            pred_list = list(pred)

        if pred_list == gt:
            solved.append(task_id)
        else:
            wrong.append(task_id)

    results[solver_name] = {
        "module":    SOLVER_MODULE[solver_name],
        "solved":    len(solved),
        "wrong":     len(wrong),
        "errors":    len(errors),
        "wrong_ids": wrong[:20],
        "error_ids": errors[:3],
    }

# ── report ───────────────────────────────────────────────────────────────────
print(f"{'Solver':<42} {'Mod':>4} {'Solved':>7} {'Wrong':>6} {'Errors':>7}")
print("-" * 74)
total_wrong = 0
fp_solvers = []
for name in TARGET_SOLVERS:
    r = results[name]
    flag = "  *** FP ***" if r['wrong'] > 0 else ""
    print(f"{name:<42} {r['module']:>4} {r['solved']:>7} {r['wrong']:>6} {r['errors']:>7}{flag}")
    total_wrong += r['wrong']
    if r['wrong'] > 0:
        fp_solvers.append(name)
        for tid in r['wrong_ids']:
            print(f"    FP: {tid}")

print("-" * 74)
print(f"\nTotal false positives: {total_wrong}")
print(f"Solvers with wrong > 0: {fp_solvers}")

out = ROOT / "scripts/fp_audit_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nFull results saved to: {out}")
