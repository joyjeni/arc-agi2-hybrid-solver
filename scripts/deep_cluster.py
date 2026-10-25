"""
Deep sub-clustering of the RECOLOR_SAME_SIZE and RECOLOR_DIFF_COLORS buckets.
Identifies concrete sub-patterns for specialized solver construction.
"""
import json, sys, collections, itertools
from pathlib import Path

ROOT = Path("/home/sandbox/arc-jeni-2026")
sys.path.insert(0, str(ROOT))

train_ch  = json.loads((ROOT / "data/arc-agi_training_challenges.json").read_text())
train_sol = json.loads((ROOT / "data/arc-agi_training_solutions.json").read_text())
audit     = json.loads(open('/tmp/pattern_audit.json').read())

RECOLOR_SAME_SIZE  = audit["buckets"].get("RECOLOR_SAME_SIZE", [])
RECOLOR_DIFF_COLOR = audit["buckets"].get("RECOLOR_DIFF_COLORS", [])
SHRINK_BUCKET      = audit["buckets"].get("SHRINK", [])

def grid_to_tuple(g):
    return tuple(tuple(r) for r in g)

def unique_colors(g):
    s = set()
    for r in g: s.update(r)
    return s

def color_mapping(inp, out):
    """Returns per-color mapping {c_in: c_out} if consistent, else None."""
    m = {}
    for r_in, r_out in zip(inp, out):
        for ci, co in zip(r_in, r_out):
            if ci in m:
                if m[ci] != co:
                    return None
            else:
                m[ci] = co
    return m

def is_identity(mapping):
    return all(k == v for k, v in mapping.items())

def n_cells_changed(inp, out):
    changed = 0
    total = 0
    for r_in, r_out in zip(inp, out):
        for ci, co in zip(r_in, r_out):
            total += 1
            if ci != co:
                changed += 1
    return changed, total

def detect_border_fill(inp, out):
    """Detect if only border cells are changed."""
    H, W = len(inp), len(inp[0])
    border_changed = 0
    interior_changed = 0
    for r in range(H):
        for c in range(W):
            if inp[r][c] != out[r][c]:
                if r == 0 or r == H-1 or c == 0 or c == W-1:
                    border_changed += 1
                else:
                    interior_changed += 1
    return border_changed > 0 and interior_changed == 0

def detect_flood_fill(inp, out):
    """Detect if a flood-fill-like change happened (connected region recolored)."""
    H, W = len(inp), len(inp[0])
    changed = []
    for r in range(H):
        for c in range(W):
            if inp[r][c] != out[r][c]:
                changed.append((r, c, inp[r][c], out[r][c]))
    if not changed:
        return False
    # Check if changed cells form a connected region
    changed_cells = set((r, c) for r, c, ci, co in changed)
    if len(changed_cells) == 0:
        return False
    start = next(iter(changed_cells))
    visited = {start}
    queue = [start]
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if (nr, nc) in changed_cells and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return len(visited) == len(changed_cells)

def detect_global_color_map(pairs):
    """Check if same color mapping holds across ALL train pairs."""
    mappings = []
    for p in pairs:
        m = color_mapping(p["input"], p["output"])
        if m is None:
            return None
        mappings.append(m)
    if not mappings:
        return None
    # Check consistency: same src->dst for colors present in multiple pairs
    merged = {}
    for m in mappings:
        for k, v in m.items():
            if k in merged and merged[k] != v:
                return None
            merged[k] = v
    if is_identity(merged):
        return None
    return merged

def count_changed_fraction(pairs):
    fracs = []
    for p in pairs:
        changed, total = n_cells_changed(p["input"], p["output"])
        fracs.append(changed / total if total else 0)
    return sum(fracs) / len(fracs)

def detect_single_object_recolor(pairs):
    """Detect: one object in input changes color, rest stays."""
    for p in pairs:
        changed, total = n_cells_changed(p["input"], p["output"])
        if changed == 0 or changed == total:
            return False
    return True

def detect_noise_removal(pairs):
    """Detect: sparse pixels removed/changed (noise cleaning)."""
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        nz_in = sum(1 for r in inp for c in r if c != 0)
        nz_out = sum(1 for r in out for c in r if c != 0)
        if nz_out >= nz_in:
            return False
    return True

def detect_symmetry_completion(pairs):
    """Detect: output is more symmetric than input."""
    def sym_score(g):
        H, W = len(g), len(g[0])
        # horizontal
        h_match = sum(g[r][c] == g[r][W-1-c] for r in range(H) for c in range(W)) / (H*W)
        v_match = sum(g[r][c] == g[H-1-r][c] for r in range(H) for c in range(W)) / (H*W)
        return max(h_match, v_match)
    scores = [(sym_score(p["output"]) - sym_score(p["input"])) for p in pairs]
    return all(s > 0.05 for s in scores)

# Sub-classify the RECOLOR_SAME_SIZE bucket
sub_buckets = collections.defaultdict(list)

print(f"Analyzing {len(RECOLOR_SAME_SIZE)} RECOLOR_SAME_SIZE tasks...")
for tid in RECOLOR_SAME_SIZE:
    task = train_ch[tid]
    pairs = task["train"]
    tags = []

    # 1. Global color mapping
    gmap = detect_global_color_map(pairs)
    if gmap:
        non_id = {k: v for k, v in gmap.items() if k != v}
        n_mapped = len(non_id)
        if n_mapped == 1:
            tags.append("SINGLE_COLOR_SWAP")
        elif n_mapped == 2:
            tags.append("TWO_COLOR_SWAP")
        elif n_mapped <= 4:
            tags.append("FEW_COLOR_MAP")
        else:
            tags.append("MULTI_COLOR_MAP")
    else:
        # 2. Flood fill
        all_flood = all(detect_flood_fill(p["input"], p["output"]) for p in pairs)
        if all_flood:
            tags.append("FLOOD_FILL")

        # 3. Border fill
        all_border = all(detect_border_fill(p["input"], p["output"]) for p in pairs)
        if all_border:
            tags.append("BORDER_FILL")

        # 4. Symmetry completion
        sym = detect_symmetry_completion(pairs)
        if sym:
            tags.append("SYMMETRY_COMPLETE")

        # 5. Noise removal
        noise = detect_noise_removal(pairs)
        if noise:
            tags.append("NOISE_REMOVE")

    frac = count_changed_fraction(pairs)
    if frac < 0.1:
        tags.append("FEW_CELLS_CHANGE")
    elif frac > 0.7:
        tags.append("MOST_CELLS_CHANGE")

    if not tags:
        tags.append("UNKNOWN_RECOLOR")

    key = "|".join(sorted(set(tags)))
    sub_buckets[key].append(tid)

print("\n=== RECOLOR_SAME_SIZE sub-buckets ===")
for bname, tids in sorted(sub_buckets.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(tids):3d}] {bname}")
    for tid in tids[:3]:
        print(f"        {tid}")

# Save sub-bucket results
results = {
    "RECOLOR_SAME_SIZE_sub": {k: v for k, v in sub_buckets.items()}
}

# Also quick check RECOLOR_DIFF_COLORS
diff_sub = collections.defaultdict(list)
print(f"\nAnalyzing {len(RECOLOR_DIFF_COLOR)} RECOLOR_DIFF_COLORS tasks...")
for tid in RECOLOR_DIFF_COLOR:
    task = train_ch[tid]
    pairs = task["train"]
    tags = []

    gmap = detect_global_color_map(pairs)
    if gmap:
        non_id = {k: v for k, v in gmap.items() if k != v}
        tags.append(f"GLOBAL_MAP_{len(non_id)}c")

    all_flood = all(detect_flood_fill(p["input"], p["output"]) for p in pairs)
    if all_flood:
        tags.append("FLOOD_FILL")

    frac = count_changed_fraction(pairs)
    if frac < 0.05:
        tags.append("TINY_CHANGE")
    elif frac < 0.2:
        tags.append("SMALL_CHANGE")
    elif frac > 0.8:
        tags.append("LARGE_CHANGE")

    # New colors appear?
    for p in pairs:
        cin = unique_colors(p["input"])
        cout = unique_colors(p["output"])
        new_colors = cout - cin
        if new_colors:
            tags.append("NEW_COLORS_APPEAR")
            break

    if not tags:
        tags.append("UNKNOWN_DIFF")

    key = "|".join(sorted(set(tags)))
    diff_sub[key].append(tid)

print("\n=== RECOLOR_DIFF_COLORS sub-buckets ===")
for bname, tids in sorted(diff_sub.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(tids):3d}] {bname}")
    for tid in tids[:3]:
        print(f"        {tid}")

with open('/tmp/deep_cluster.json', 'w') as f:
    json.dump({
        "recolor_same_size_sub": {k: v for k, v in sub_buckets.items()},
        "recolor_diff_colors_sub": {k: v for k, v in diff_sub.items()},
    }, f)
print("\nSaved to /tmp/deep_cluster.json")
