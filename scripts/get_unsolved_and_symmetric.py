"""
1. Run all solvers on all 1000 training tasks → save solved/unsolved IDs
2. Among unsolved, find tasks where ALL outputs in train pairs are symmetric
   (horizontally, vertically, or rotationally)
3. Save results to /tmp/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/sandbox/arc-jeni-2026")
sys.path.insert(0, str(ROOT))

from src.core.interfaces import TaskContext
from src.solvers import REGISTRY

train_challenges = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
train_solutions  = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())

task_ids = list(train_challenges.keys())
print(f"Total tasks: {len(task_ids)}")

solvers = [s for s in REGISTRY.all_solvers() if s.name != "IdentitySolver"]
solvers.sort(key=lambda s: s.priority)
print(f"Solvers: {len(solvers)}")

solvable = set()
t0 = time.time()

for task_id in task_ids:
    task_data     = train_challenges[task_id]
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
        except Exception:
            continue
        if result is None or not result.predictions:
            continue
        pred = result.predictions[0].attempt_1
        gt   = ground_truths[0]
        if pred == gt:
            solvable.add(task_id)
            break  # first correct solver is enough

elapsed = time.time() - t0
unsolved = [tid for tid in task_ids if tid not in solvable]
print(f"\nSolved: {len(solvable)}/{len(task_ids)} ({len(solvable)/len(task_ids)*100:.1f}%) in {elapsed:.1f}s")
print(f"Unsolved: {len(unsolved)}")

# Save lists
Path("/tmp/solved_ids.json").write_text(json.dumps(sorted(solvable)))
Path("/tmp/unsolved_ids.json").write_text(json.dumps(unsolved))
print("Saved /tmp/solved_ids.json and /tmp/unsolved_ids.json")

# --- Symmetry analysis on unsolved tasks ---
def is_symmetric(grid):
    """Check h-flip, v-flip, 180-rot, 4-fold rotation symmetry."""
    g = np.array(grid)
    return (
        np.array_equal(g, g[::-1])           or  # h-flip (vertical axis)
        np.array_equal(g, g[:, ::-1])        or  # v-flip (horizontal axis)
        np.array_equal(g, g[::-1, ::-1])     or  # 180 rotation
        (g.shape[0] == g.shape[1] and np.array_equal(g, np.rot90(g)))  # 90 rot
    )

def all_outputs_symmetric(task_data):
    for pair in task_data.get("train", []):
        if not is_symmetric(pair["output"]):
            return False
    return True

sym_tasks = []
for tid in unsolved:
    td = train_challenges[tid]
    if all_outputs_symmetric(td):
        sym_tasks.append(tid)

print(f"\nSymmetric-output unsolved tasks: {len(sym_tasks)}")
Path("/tmp/symmetric_unsolved.json").write_text(json.dumps(sym_tasks))
print("Saved /tmp/symmetric_unsolved.json")

# Also bucket the unsolved by shape relationship
grow, shrink, same_size, mixed = [], [], [], []
for tid in unsolved:
    td = train_challenges[tid]
    pairs = td.get("train", [])
    rels = []
    for p in pairs:
        ih, iw = len(p["input"]), len(p["input"][0])
        oh, ow = len(p["output"]), len(p["output"][0])
        if ih == oh and iw == ow:
            rels.append("same")
        elif oh > ih or ow > iw:
            rels.append("grow")
        else:
            rels.append("shrink")
    if all(r == "same" for r in rels):
        same_size.append(tid)
    elif all(r == "grow" for r in rels):
        grow.append(tid)
    elif all(r == "shrink" for r in rels):
        shrink.append(tid)
    else:
        mixed.append(tid)

print("\nBuckets (fresh, truly unsolved):")
print(f"  same_size: {len(same_size)}")
print(f"  grow:      {len(grow)}")
print(f"  shrink:    {len(shrink)}")
print(f"  mixed:     {len(mixed)}")

Path("/tmp/fresh_buckets.json").write_text(json.dumps({
    "same_size": same_size, "grow": grow, "shrink": shrink, "mixed": mixed
}))
print("Saved /tmp/fresh_buckets.json")
