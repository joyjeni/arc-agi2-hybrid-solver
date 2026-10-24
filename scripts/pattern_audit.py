"""
Pattern Audit — analyzes all 864 unsolved tasks, extracts structural features,
clusters them into buckets, and reports the top patterns to target.
"""
import json, sys, time, collections, math
from pathlib import Path

ROOT = Path("/home/sandbox/arc-jeni-2026")
sys.path.insert(0, str(ROOT))

from src.solvers import REGISTRY
from src.core.interfaces import TaskContext

train_ch  = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
train_sol = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())

# ── 1. Find unsolved tasks ─────────────────────────────────────────────────
solvers = [s for s in REGISTRY.all_solvers() if s.name != "IdentitySolver"]
solved_ids = set()

for task_id, task_data in train_ch.items():
    gts = train_sol.get(task_id, [])
    if not gts:
        continue
    ctx = TaskContext(
        task_id=task_id,
        train_pairs=task_data.get("train", []),
        test_inputs=[t["input"] for t in task_data.get("test", [])],
    )
    for solver in solvers:
        try:
            res = solver.solve(ctx)
        except:
            continue
        if res and res.predictions:
            if res.predictions[0].attempt_1 == gts[0]:
                solved_ids.add(task_id)
                break

unsolved_ids = [tid for tid in train_ch if tid not in solved_ids]
print(f"Solved: {len(solved_ids)}  |  Unsolved: {len(unsolved_ids)}")

# ── 2. Feature extractor ───────────────────────────────────────────────────
def grid_shape(g):
    return (len(g), len(g[0]) if g else 0)

def unique_colors(g):
    s = set()
    for row in g:
        s.update(row)
    return s

def count_nonzero(g):
    return sum(1 for r in g for c in r if c != 0)

def is_same_size(inp, out):
    return grid_shape(inp) == grid_shape(out)

def size_ratio(inp, out):
    ih, iw = grid_shape(inp)
    oh, ow = grid_shape(out)
    if ih == 0 or iw == 0:
        return (0, 0)
    return (oh / ih if ih else 0, ow / iw if iw else 0)

def color_change(inp, out):
    cin = unique_colors(inp)
    cout = unique_colors(out)
    return (cin == cout, len(cout - cin), len(cin - cout))

def is_single_color_out(out):
    colors = unique_colors(out)
    colors.discard(0)
    return len(colors) == 1

def detect_transform_type(pairs):
    """Heuristic: classify the dominant transformation in the examples."""
    tags = collections.Counter()
    for p in pairs:
        inp, out = p["input"], p["output"]
        ih, iw = grid_shape(inp)
        oh, ow = grid_shape(out)
        same_size = (ih == oh and iw == ow)
        # Size change
        if oh == ih and ow == iw:
            tags["same_size"] += 1
        elif oh * ow < ih * iw:
            tags["shrink"] += 1
        elif oh * ow > ih * iw:
            tags["grow"] += 1
        # Ratio patterns
        rh, rw = oh / max(ih, 1), ow / max(iw, 1)
        if abs(rh - round(rh)) < 0.05 and abs(rw - round(rw)) < 0.05 and rh > 1:
            tags["scale_up"] += 1
        if abs(rh - round(rh)) < 0.05 and abs(rw - round(rw)) < 0.05 and rh < 1 and rh > 0:
            tags["scale_down"] += 1
        # Color preservation
        cin = unique_colors(inp)
        cout = unique_colors(out)
        if cin == cout:
            tags["color_preserve"] += 1
        elif cout.issubset(cin):
            tags["color_reduce"] += 1
        elif cin.issubset(cout):
            tags["color_add"] += 1
        # Object-like: few non-zero in output
        nz_in  = count_nonzero(inp)
        nz_out = count_nonzero(out)
        if nz_out < 0.2 * (oh * ow):
            tags["sparse_out"] += 1
        if nz_in < 0.2 * (ih * iw):
            tags["sparse_in"] += 1
        # Symmetry check for output
        if same_size:
            # horizontal flip
            flipped_h = [row[::-1] for row in out]
            if flipped_h == inp:
                tags["flip_h"] += 1
            flipped_v = out[::-1]
            if flipped_v == inp:
                tags["flip_v"] += 1
    return tags.most_common()

def output_is_single_row_or_col(out):
    h, w = grid_shape(out)
    return h == 1 or w == 1

def num_colors_in(g):
    c = unique_colors(g)
    c.discard(0)
    return len(c)

# ── 3. Feature extraction for each unsolved task ──────────────────────────
features = {}
for tid in unsolved_ids:
    task = train_ch[tid]
    pairs = task["train"]
    test_in = task["test"][0]["input"]

    # Size signature
    in_sizes  = [grid_shape(p["input"])  for p in pairs]
    out_sizes = [grid_shape(p["output"]) for p in pairs]
    consistent_size = len(set(in_sizes)) == 1 and len(set(out_sizes)) == 1

    same_io_size = all(grid_shape(p["input"]) == grid_shape(p["output"]) for p in pairs)
    out_smaller  = all(grid_shape(p["output"])[0] * grid_shape(p["output"])[1] <
                       grid_shape(p["input"])[0]  * grid_shape(p["input"])[1]  for p in pairs)
    out_larger   = all(grid_shape(p["output"])[0] * grid_shape(p["output"])[1] >
                       grid_shape(p["input"])[0]  * grid_shape(p["input"])[1]  for p in pairs)

    # Color signature
    n_colors_in  = [num_colors_in(p["input"])  for p in pairs]
    n_colors_out = [num_colors_in(p["output"]) for p in pairs]
    same_colors  = all(unique_colors(p["input"]) == unique_colors(p["output"]) for p in pairs)
    fewer_colors = all(num_colors_in(p["output"]) <= num_colors_in(p["input"]) for p in pairs)
    single_color_out = all(is_single_color_out(p["output"]) for p in pairs)

    # Ratio
    ratios = [size_ratio(p["input"], p["output"]) for p in pairs]
    rh_vals = [r[0] for r in ratios]
    rw_vals = [r[1] for r in ratios]
    integer_scale = all(rh == rw and rh == round(rh) and rh > 0 for rh, rw in ratios)

    # tiny output (counting/selection)
    tiny_out = all(grid_shape(p["output"])[0] * grid_shape(p["output"])[1] <= 4 for p in pairs)
    single_cell = all(grid_shape(p["output"]) == (1, 1) for p in pairs)

    # transform tags
    ttags = detect_transform_type(pairs)

    features[tid] = {
        "same_io_size": same_io_size,
        "out_smaller": out_smaller,
        "out_larger": out_larger,
        "same_colors": same_colors,
        "fewer_colors": fewer_colors,
        "single_color_out": single_color_out,
        "single_cell_out": single_cell,
        "tiny_out": tiny_out,
        "integer_scale": integer_scale,
        "avg_in_size": (sum(s[0] for s in in_sizes) / len(in_sizes),
                        sum(s[1] for s in in_sizes) / len(in_sizes)),
        "n_train_pairs": len(pairs),
        "transform_tags": dict(ttags),
    }

# ── 4. Pattern buckets ────────────────────────────────────────────────────
buckets = collections.defaultdict(list)
for tid, f in features.items():
    tags = f["transform_tags"]
    bucket = []
    if f["same_io_size"] and f["same_colors"]:
        bucket.append("RECOLOR_SAME_SIZE")
    elif f["same_io_size"] and not f["same_colors"]:
        bucket.append("RECOLOR_DIFF_COLORS")
    if f["single_cell_out"]:
        bucket.append("SINGLE_CELL_OUT")
    elif f["tiny_out"]:
        bucket.append("TINY_OUT")
    elif f["out_smaller"] and not f["tiny_out"]:
        bucket.append("SHRINK")
    elif f["out_larger"] and f["integer_scale"]:
        bucket.append("SCALE_UP_INTEGER")
    elif f["out_larger"]:
        bucket.append("GROW_NON_INTEGER")
    if f["single_color_out"] and not f["single_cell_out"]:
        bucket.append("SINGLE_COLOR_OUTPUT")
    if not bucket:
        bucket.append("OTHER")
    key = "|".join(sorted(set(bucket)))
    buckets[key].append(tid)

print("\n=== Pattern Buckets (sorted by size) ===")
for bname, tids in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(tids):3d}] {bname}")

# ── 5. Drill-down on top buckets ──────────────────────────────────────────
TOP_BUCKETS = ["RECOLOR_SAME_SIZE", "RECOLOR_SAME_SIZE|RECOLOR_DIFF_COLORS",
               "SINGLE_CELL_OUT", "SHRINK", "SCALE_UP_INTEGER"]

print("\n=== Top 5 task IDs per major bucket ===")
for bname, tids in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print(f"\n  {bname} ({len(tids)} tasks):")
    for tid in tids[:5]:
        f = features[tid]
        sz = f["avg_in_size"]
        nc = [num_colors_in(train_ch[tid]["train"][i]["input"]) for i in range(len(train_ch[tid]["train"]))]
        print(f"    {tid}  in~{sz[0]:.0f}x{sz[1]:.0f}  n_colors_in={nc}")

# Save for downstream
out = {"unsolved": unsolved_ids, "features": {k: {**v, "transform_tags": v["transform_tags"]} for k, v in features.items()}, "buckets": {k: v for k, v in buckets.items()}}
with open('/tmp/pattern_audit.json', 'w') as f:
    json.dump(out, f)
print(f"\nSaved audit to /tmp/pattern_audit.json")
