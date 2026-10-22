"""
Benchmark all T1 solvers (excl. IdentitySolver) on 240 test task IDs.
Pipeline convention: lower priority number = tried first (ascending sort).
Verifies against arc-agi_training_solutions.json.
"""
import json, sys, time
from pathlib import Path

ROOT = Path("/home/sandbox/arc-jeni-2026")
sys.path.insert(0, str(ROOT))

from src.solvers import REGISTRY
from src.core.interfaces import TaskContext

# Load data
test_challenges = json.loads((ROOT / "data/arc-agi_test_challenges.json").read_text())
train_solutions  = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())

task_ids = list(test_challenges.keys())
print(f"Tasks to evaluate : {len(task_ids)}")

solvers = [s for s in REGISTRY.all_solvers() if s.name != "IdentitySolver"]
solvers.sort(key=lambda s: s.priority)   # ascending: lower = higher priority
print(f"Solvers (excl. Id): {len(solvers)}")
print(f"Priority order (first 8): {[s.name for s in solvers[:8]]}")

correct = 0
wrong   = 0
no_attempt = 0
solver_hits:     dict[str, int] = {}
solver_wrongs:   dict[str, int] = {}
solver_attempts: dict[str, int] = {}
failed_tasks = []

t0 = time.time()
for task_id in task_ids:
    task_data     = test_challenges[task_id]
    ground_truths = train_solutions.get(task_id, [])
    if not ground_truths:
        no_attempt += 1
        continue

    ctx = TaskContext(
        task_id=task_id,
        train_pairs=task_data.get("train", []),
        test_inputs=[t["input"] for t in task_data.get("test", [])],
    )

    attempted = False
    solved    = False
    for solver in solvers:
        try:
            result = solver.solve(ctx)
        except Exception:
            continue
        if result is None:
            continue
        # First solver that returns a result wins (highest priority)
        attempted = True
        solver_attempts[solver.name] = solver_attempts.get(solver.name, 0) + 1
        if result.predictions:
            pred_grid = result.predictions[0].attempt_1
            gt_grid   = ground_truths[0] if ground_truths else None
            if gt_grid is not None and pred_grid == gt_grid:
                solver_hits[solver.name] = solver_hits.get(solver.name, 0) + 1
                solved = True
            else:
                solver_wrongs[solver.name] = solver_wrongs.get(solver.name, 0) + 1
        break  # highest-priority match wins

    if not attempted:
        no_attempt += 1
    elif solved:
        correct += 1
    else:
        wrong += 1
        failed_tasks.append(task_id)

elapsed = time.time() - t0
total   = len(task_ids)
print(f"\n=== Benchmark Results ({elapsed:.1f}s) ===")
print(f"Correctly solved  : {correct} / {total}  ({correct/total*100:.1f}%)")
print(f"Attempted wrong   : {wrong}")
print(f"Not attempted     : {no_attempt}")

print(f"\n--- Solver breakdown ---")
all_names = sorted(
    set(list(solver_hits.keys()) + list(solver_wrongs.keys())),
    key=lambda x: -(solver_hits.get(x, 0))
)
for name in all_names:
    h = solver_hits.get(name, 0)
    w = solver_wrongs.get(name, 0)
    a = solver_attempts.get(name, 0)
    print(f"  {name:<42s} {h:3d} correct  {w:3d} wrong  ({a} attempts)")

print(f"\n--- First 20 failed tasks ---")
for tid in failed_tasks[:20]:
    print(f"  {tid}")
