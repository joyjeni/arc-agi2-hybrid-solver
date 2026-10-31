"""Benchmark that saves solved/unsolved IDs."""
import json, sys, time, traceback
from pathlib import Path
ROOT = Path("/home/sandbox/arc-jeni-2026")
sys.path.insert(0, str(ROOT))
from src.solvers import REGISTRY
from src.core.interfaces import TaskContext

train_challenges = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
train_solutions  = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())
task_ids = list(train_challenges.keys())
solvers = [s for s in REGISTRY.all_solvers() if s.name != "IdentitySolver"]
solvers.sort(key=lambda s: s.priority)

solvable_tasks = set()
best_solver = {}
for task_id in task_ids:
    task_data = train_challenges[task_id]
    ground_truths = train_solutions.get(task_id, [])
    if not ground_truths: continue
    ctx = TaskContext(task_id=task_id,
                      train_pairs=task_data.get("train", []),
                      test_inputs=[t["input"] for t in task_data.get("test", [])])
    for solver in solvers:
        try:
            result = solver.solve(ctx)
        except: continue
        if not result or not result.predictions: continue
        if result.predictions[0].attempt_1 == ground_truths[0]:
            solvable_tasks.add(task_id)
            best_solver[task_id] = solver.name
            break  # first correct solver wins

print(f"Solved: {len(solvable_tasks)} / {len(task_ids)}")
unsolved = [tid for tid in task_ids if tid not in solvable_tasks]
print(f"Unsolved: {len(unsolved)}")

with open('/home/sandbox/solved_ids.json', 'w') as f:
    json.dump(list(solvable_tasks), f)
with open('/home/sandbox/unsolved_ids.json', 'w') as f:
    json.dump(unsolved, f)
with open('/home/sandbox/best_solver_map.json', 'w') as f:
    json.dump(best_solver, f)
print("Saved solved_ids.json, unsolved_ids.json, best_solver_map.json")
