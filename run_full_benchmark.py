"""
Full benchmark — try ALL solvers on every task (don't stop at first match).
Report: how many tasks are solvable by at least one solver.
"""
import json, sys, time, traceback
from pathlib import Path

ROOT = Path("/home/sandbox/arc-jeni-2026")
sys.path.insert(0, str(ROOT))

from src.solvers import REGISTRY
from src.core.interfaces import TaskContext

train_challenges = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
train_solutions  = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())
test_challenges  = json.loads((ROOT / "data/arc-agi_test_challenges.json").read_text())

# Run on all 1000 training tasks (full coverage check)
all_tasks = {**train_challenges}
task_ids  = list(all_tasks.keys())
print(f"Total tasks: {len(task_ids)}")

solvers = [s for s in REGISTRY.all_solvers() if s.name != "IdentitySolver"]
solvers.sort(key=lambda s: s.priority)
print(f"Solvers: {len(solvers)}")

solver_correct: dict[str, int] = {s.name: 0 for s in solvers}
solver_wrong:   dict[str, int] = {s.name: 0 for s in solvers}
solver_errors:  dict[str, list] = {s.name: [] for s in solvers}

solvable_tasks = set()
best_solver: dict[str, str] = {}

t0 = time.time()
for task_id in task_ids:
    task_data     = all_tasks[task_id]
    ground_truths = train_solutions.get(task_id, [])
    if not ground_truths:
        continue

    ctx = TaskContext(
        task_id=task_id,
        train_pairs=task_data.get("train", []),
        test_inputs=[t["input"] for t in task_data.get("test", [])],
    )

    for solver in solvers:
        try:
            result = solver.solve(ctx)
        except Exception as e:
            solver_errors[solver.name].append((task_id, str(e)))
            continue
        if result is None:
            continue
        if not result.predictions:
            continue
        pred = result.predictions[0].attempt_1
        gt   = ground_truths[0]
        if pred == gt:
            solver_correct[solver.name] += 1
            solvable_tasks.add(task_id)
            best_solver[task_id] = solver.name
        else:
            solver_wrong[solver.name] += 1

elapsed = time.time() - t0
print(f"\n=== Full Coverage ({elapsed:.1f}s) ===")
print(f"Tasks solvable by at least one solver: {len(solvable_tasks)} / {len(task_ids)}  ({len(solvable_tasks)/len(task_ids)*100:.1f}%)")

print(f"\n--- Per-solver breakdown (correct / wrong) ---")
for s in solvers:
    c = solver_correct[s.name]
    w = solver_wrong[s.name]
    e = len(solver_errors[s.name])
    if c > 0 or w > 0 or e > 0:
        print(f"  {s.name:<42s} {c:3d} correct  {w:3d} wrong  {e:2d} errors")

# Show error summary
for s in solvers:
    errs = solver_errors[s.name]
    if errs:
        print(f"\n  Errors in {s.name}:")
        for tid, msg in errs[:3]:
            print(f"    task={tid}: {msg}")
