"""
Inspect sample tasks from each sub-bucket to understand actual patterns.
"""
import json
from pathlib import Path

ROOT = Path("/home/sandbox/arc-jeni-2026")
train_ch  = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
train_sol = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())
dc = json.loads(open('/tmp/deep_cluster.json').read())

def show_task(tid, max_pairs=2):
    task = train_ch[tid]
    sol  = train_sol[tid]
    pairs = task["train"]
    print(f"\n{'='*60}")
    print(f"Task: {tid}  ({len(pairs)} train pairs)")
    for i, p in enumerate(pairs[:max_pairs]):
        inp = p["input"]
        out = p["output"]
        print(f"  Train {i}:  IN {len(inp)}x{len(inp[0])}  OUT {len(out)}x{len(out[0])}")
        # Show as compact grid
        print(f"    INPUT:  {[r for r in inp[:5]]}")
        print(f"    OUTPUT: {[r for r in out[:5]]}")
    print(f"  Test IN: {len(task['test'][0]['input'])}x{len(task['test'][0]['input'][0])}")
    print(f"  Test SOL: {sol[0] if sol else 'N/A'}")

print("=== FEW_CELLS_CHANGE samples ===")
for tid in dc["recolor_same_size_sub"].get("FEW_CELLS_CHANGE", [])[:5]:
    show_task(tid)

print("\n\n=== UNKNOWN_RECOLOR samples ===")
for tid in dc["recolor_same_size_sub"].get("UNKNOWN_RECOLOR", [])[:5]:
    show_task(tid)

print("\n\n=== NEW_COLORS_APPEAR|SMALL_CHANGE samples ===")
for tid in dc["recolor_diff_colors_sub"].get("NEW_COLORS_APPEAR|SMALL_CHANGE", [])[:5]:
    show_task(tid)

print("\n\n=== NEW_COLORS_APPEAR samples ===")
for tid in dc["recolor_diff_colors_sub"].get("NEW_COLORS_APPEAR", [])[:5]:
    show_task(tid)

print("\n\n=== SHRINK samples ===")
import scripts.pattern_audit  # noqa: F401  (side effect: runs the audit, writing /tmp/pattern_audit.json)

audit = json.loads(open('/tmp/pattern_audit.json').read())
for tid in audit["buckets"].get("SHRINK", [])[:5]:
    show_task(tid)
