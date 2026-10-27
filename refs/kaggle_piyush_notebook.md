[Piyush kumar's profile](https://www.kaggle.com/piyush295) Piyush kumar  · 17h ago · 47 views

arrow\_drop\_up2

Copy & Edit
1

file\_downloadDownload

more\_vert

# ARC AGI2 Advanced 30 Strategies

## ARC AGI2 Advanced 30 Strategies

[Notebook](https://www.kaggle.com/code/piyush295/arc-agi2-advanced-30-strategies/notebook) [Input](https://www.kaggle.com/code/piyush295/arc-agi2-advanced-30-strategies/input) [Output](https://www.kaggle.com/code/piyush295/arc-agi2-advanced-30-strategies/output) [Logs](https://www.kaggle.com/code/piyush295/arc-agi2-advanced-30-strategies/log) [Comments (0)](https://www.kaggle.com/code/piyush295/arc-agi2-advanced-30-strategies/comments)

historyVersion 1 of 1

## Runtime

play\_arrow

38s

## Input

Competitions

![](https://www.kaggle.com/competitions/133469/images/thumbnail)

ARC Prize 2026 - ARC-AGI-2

## Language

Python

![Profile picture for undefined](https://storage.googleapis.com/kaggle-competitions/kaggle/133469/logos/thumb76_76.png?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1793130329&Signature=TThtSUJHtEeTd07TWOchTo%2FPHBLFGLBjd%2FLK%2Bm6fx%2BPnFPzVFXNdq%2FL7vlqDzr%2FTh5rgnrUQW0MPlG2ssUBrVj2ot%2Bk%2FhI2eXWw1r32%2FBJ8nAxuB8gy7uQp7RDbETDqhcUk16vIuvGJAVCR5cQ75xrbJVkILcV5zL%2BGTlqcJuveoTr%2FHkc%2BIn654lyRvqYVcgHflOlld%2BbcRq%2BXooHPWfoT5vsf7mJjW54CQLyb4j9cp1kesYN4rwdX7mI2Xo8IqEh%2Fm6epNpza7mqYL0qkHkapTUQTp8lbfBAvENTrgj0sBu2pe5%2FGADodvaeGDjEPONZZuySVKpSRm%2FkyNZGMy7A%3D%3D&t=2026-03-10-16-17-03)

Competition Notebook

[ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)

\_\_script\_\_

Loading \[MathJax\]/jax/output/CommonHTML/fonts/TeX/fontdata.js

linkcode

```
#!/usr/bin/env python3
"""
ARC-AGI-2 Advanced Submission Generator
=======================================

A pure-stdlib ARC-AGI-2 solver that goes well beyond the 13-heuristic
baseline. The core philosophy is UNCHANGED and is the only thing that can
score on a hidden set: every candidate transformation is a *program* that is
FIT and then VERIFIED against the task's train demonstration pairs. A program
is only trusted if it reproduces EVERY train output exactly. Trusted programs
are applied to the test input to produce attempt_1 / attempt_2 (pass@2).

Why this beats the baseline
---------------------------
The baseline only had fixed geometric transforms + simple tiling/recolor.
ARC-AGI-2 eval defeats those. This solver adds genuine reasoning primitives
grouped by the dominant task families from the pattern analysis:

  * Object detection & segmentation (connected components, 4/8 connectivity,
    color-split, bounding boxes).
  * Object-based transformations (select by size/color/uniqueness/shape,
    crop to object, recolor by property, replicate/fractal).
  * Symmetry completion (mirror/rotation repair, occlusion in-fill,
    periodic-tiling continuation, symmetrize overlay).
  * Spatial / grid operations (crop to content, border add, gravity, panel
    split + boolean combine, panel select, color mapping, denoise).

Each family contributes one or more *builder* functions. A builder inspects
the train pairs and returns either callable solver(s) (grid->grid) or [].
Every returned solver is re-verified on every train pair before being
trusted. No program is used unless it is demonstrably consistent with the
task specification, which is what makes a non-zero hidden score possible.

Output schema (identical to baseline / sample_submission.json):

    {
      "<task_id>": [\
        {"attempt_1": <grid>, "attempt_2": <grid>},   # one dict per test input\
        ...\
      ],
      ...
    }

Usage:  python3 advanced_submission.py
Reads : arc-agi_test_challenges.json  (same dir)
Writes: submission.json

Pure Python, stdlib only. Works offline.
"""

import json
import os
import sys
from collections import Counter, deque

sys.setrecursionlimit(100000)

def find_test_challenges():
    """Find test challenges file in Kaggle environment."""
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            for file in files:
                if "test" in file and "challenges" in file and file.endswith(".json"):
                    return os.path.join(root, file)
    return "./arc-agi_test_challenges.json"

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_CHALLENGES = find_test_challenges()
SAMPLE_SUBMISSION = "./sample_submission.json"
OUTPUT = "./submission.json"

# ============================================================================
# Grid primitives
# ============================================================================
def dims(g):
    return len(g), (len(g[0]) if g else 0)

def copy_grid(g):
    return [row[:] for row in g]

def rot90(g):
    return [list(row) for row in zip(*g[::-1])]

def rot180(g):
    return [row[::-1] for row in g[::-1]]

def rot270(g):
    return [list(row) for row in zip(*g)][::-1]

def flip_h(g):
    return [row[::-1] for row in g]

def flip_v(g):
    return [row[:] for row in g[::-1]]

def transpose(g):
    return [list(row) for row in zip(*g)]

def anti_transpose(g):
    return [list(row) for row in zip(*g[::-1])][::-1]

GEOMETRIC = {
    "identity": copy_grid,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "transpose": transpose,
    "anti_transpose": anti_transpose,
}

def valid_grid(g):
    if not isinstance(g, list) or len(g) == 0:
        return False
    if not all(isinstance(r, list) and len(r) > 0 for r in g):
        return False
    width = len(g[0])
    if not all(len(r) == width for r in g):
        return False
    for r in g:
        for v in r:
            if not isinstance(v, int) or v < 0 or v > 9:
                return False
    return True

def flatten(g):
    return [v for row in g for v in row]

def color_counts(g):
    return Counter(flatten(g))

def background_color(g):
    """Most common color is treated as background. 0 preferred on ties."""
    c = color_counts(g)
    if not c:
        return 0
    best = c.most_common()
    top = best[0][1]
    tied = [col for col, n in best if n == top]
    return 0 if 0 in tied else tied[0]

# ============================================================================
# Object detection & segmentation
# ============================================================================
def neighbours(r, c, connectivity):
    if connectivity == 4:
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    return [\
        (r + dr, c + dc)\
        for dr in (-1, 0, 1)\
        for dc in (-1, 0, 1)\
        if not (dr == 0 and dc == 0)\
    ]

def connected_components(g, bg, connectivity=4, same_color=True):
    """Return list of objects (dicts). Non-bg cells only."""
    h, w = dims(g)
    seen = [[False] * w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or g[r][c] == bg:
                continue
            start_color = g[r][c]
            q = deque([(r, c)])
            seen[r][c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbours(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc]:
                        if g[nr][nc] == bg:
                            continue
                        if same_color and g[nr][nc] != start_color:
                            continue
                        seen[nr][nc] = True
                        q.append((nr, nc))
            objs.append(make_object(g, cells))
    return objs

def make_object(g, cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    cols = Counter(g[r][c] for r, c in cells)
    return {
        "cells": cells,
        "r0": r0,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "size": len(cells),
        "colors": cols,
        "main_color": cols.most_common(1)[0][0],
    }

def crop_object(g, obj, bg=None):
    r0, c0, h, w = obj["r0"], obj["c0"], obj["h"], obj["w"]
    cellset = set(obj["cells"])
    out = []
    for r in range(r0, r0 + h):
        row = []
        for c in range(c0, c0 + w):
            if bg is not None and (r, c) not in cellset:
                row.append(bg)
            else:
                row.append(g[r][c])
        out.append(row)
    return out

def subgrid(g, r0, c0, h, w):
    return [row[c0 : c0 + w] for row in g[r0 : r0 + h]]

# ============================================================================
# Verification helper
# ============================================================================
def verify(solver, train):
    try:
        for pair in train:
            out = solver(pair["input"])
            if out is None or out != pair["output"]:
                return False
        return True
    except Exception:
        return False

# ============================================================================
# BUILDERS
# ============================================================================
def build_geometric(train):
    return [(n, fn) for n, fn in GEOMETRIC.items() if verify(fn, train)]

def build_color_remap(train):
    mapping = {}
    for pair in train:
        gi, go = pair["input"], pair["output"]
        if dims(gi) != dims(go):
            return []
        for ri, ro in zip(gi, go):
            for ci, co in zip(ri, ro):
                if ci in mapping and mapping[ci] != co:
                    return []
                mapping[ci] = co
    if not mapping:
        return []

    def solver(g):
        return [[mapping.get(v, v) for v in row] for row in g]

    return [("color_remap", solver)] if verify(solver, train) else []

def build_tiling(train):
    ratios = set()
    for pair in train:
        hi, wi = dims(pair["input"])
        ho, wo = dims(pair["output"])
        if hi == 0 or wi == 0 or ho % hi != 0 or wo % wi != 0:
            return []
        ratios.add((ho // hi, wo // wi))
    if len(ratios) != 1:
        return []
    ty, tx = ratios.pop()
    if ty == 1 and tx == 1:
        return []

    def plain(g):
        out = []
        for _ in range(ty):
            for row in g:
                out.append(row * tx)
        return out

    def mirrored(g):
        h, w = dims(g)
        out = [[0] * (w * tx) for _ in range(h * ty)]
        for by in range(ty):
            for bx in range(tx):
                block = g
                if by % 2 == 1:
                    block = flip_v(block)
                if bx % 2 == 1:
                    block = flip_h(block)
                for r in range(h):
                    for c in range(w):
                        out[by * h + r][bx * w + c] = block[r][c]
        return out

    results = []
    if verify(plain, train):
        results.append(("tile_plain", plain))
    if verify(mirrored, train):
        results.append(("tile_mirror", mirrored))
    return results

def build_scale(train):
    results = []
    up = set()
    ok = True
    for pair in train:
        hi, wi = dims(pair["input"])
        ho, wo = dims(pair["output"])
        if hi == 0 or wi == 0 or ho % hi != 0 or wo % wi != 0:
            ok = False
            break
        up.add((ho // hi, wo // wi))
    if ok and len(up) == 1:
        ky, kx = up.pop()
        if not (ky == 1 and kx == 1):

            def upscale(g):
                out = []
                for row in g:
                    newrow = []
                    for v in row:
                        newrow.extend([v] * kx)
                    for _ in range(ky):
                        out.append(newrow[:])
                return out

            if verify(upscale, train):
                results.append(("upscale", upscale))

    down = set()
    ok = True
    for pair in train:
        hi, wi = dims(pair["input"])
        ho, wo = dims(pair["output"])
        if ho == 0 or wo == 0 or hi % ho != 0 or wi % wo != 0:
            ok = False
            break
        down.add((hi // ho, wi // wo))
    if ok and len(down) == 1:
        ky, kx = down.pop()
        if not (ky == 1 and kx == 1):

            def downscale(g):
                ho, wo = len(g) // ky, len(g[0]) // kx
                out = []
                for i in range(ho):
                    row = []
                    for j in range(wo):
                        block = [\
                            g[i * ky + a][j * kx + b]\
                            for a in range(ky)\
                            for b in range(kx)\
                        ]
                        row.append(Counter(block).most_common(1)[0][0])
                    out.append(row)
                return out

            if verify(downscale, train):
                results.append(("downscale", downscale))
    return results

def build_crop_content(train):
    results = []

    def crop_nonbg(g):
        bg = background_color(g)
        cells = [\
            (r, c)\
            for r in range(len(g))\
            for c in range(len(g[0]))\
            if g[r][c] != bg\
        ]
        if not cells:
            return None
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return subgrid(
            g, min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        )

    if verify(crop_nonbg, train):
        results.append(("crop_content", crop_nonbg))

    for pick, name in (("max", "crop_largest_obj"), ("min", "crop_smallest_obj")):
        placed = False
        for conn in (4, 8):
            for sc in (True, False):

                def make(pick=pick, conn=conn, sc=sc):
                    def solver(g):
                        bg = background_color(g)
                        objs = connected_components(g, bg, conn, sc)
                        if not objs:
                            return None
                        obj = (max if pick == "max" else min)(
                            objs, key=lambda o: o["size"]
                        )
                        return subgrid(g, obj["r0"], obj["c0"], obj["h"], obj["w"])

                    return solver

                s = make()
                if verify(s, train):
                    results.append((name, s))
                    placed = True
                    break
            if placed:
                break
    return results

def build_object_select(train):
    results = []

    def norm_shape(o):
        return frozenset((r - o["r0"], c - o["c0"]) for r, c in o["cells"])

    def pick_unique_color(objs):
        colcount = Counter(o["main_color"] for o in objs)
        uniq = [o for o in objs if colcount[o["main_color"]] == 1]
        return uniq[0] if len(uniq) == 1 else None

    def pick_unique_shape(objs):
        shapes = Counter(norm_shape(o) for o in objs)
        uniq = [o for o in objs if shapes[norm_shape(o)] == 1]
        return uniq[0] if len(uniq) == 1 else None

    def pick_common_shape(objs):
        shapes = Counter(norm_shape(o) for o in objs)
        if not shapes:
            return None
        common = shapes.most_common(1)[0][0]
        for o in objs:
            if norm_shape(o) == common:
                return o
        return None

    pickers = {
        "largest": lambda objs: max(objs, key=lambda o: o["size"]),
        "smallest": lambda objs: min(objs, key=lambda o: o["size"]),
        "unique_color": pick_unique_color,
        "unique_shape": pick_unique_shape,
        "most_common_shape": pick_common_shape,
    }

    for pname, picker in pickers.items():
        placed = False
        for conn in (4, 8):
            for sc in (True, False):
                for masked in (False, True):

                    def make(picker=picker, conn=conn, sc=sc, masked=masked):
                        def solver(g):
                            bg = background_color(g)
                            objs = connected_components(g, bg, conn, sc)
                            if not objs:
                                return None
                            obj = picker(objs)
                            if obj is None:
                                return None
                            return crop_object(g, obj, bg if masked else None)

                        return solver

                    s = make()
                    if verify(s, train):
                        results.append(("select_%s" % pname, s))
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break
    return results

def build_symmetry_repair(train):
    results = []

    def learn_occluder():
        cand = None
        for pair in train:
            gi, go = pair["input"], pair["output"]
            if dims(gi) != dims(go):
                return None
            diff_colors = set()
            for ri, ro in zip(gi, go):
                for ci, co in zip(ri, ro):
                    if ci != co:
                        diff_colors.add(ci)
            if len(diff_colors) != 1:
                return None
            oc = next(iter(diff_colors))
            if cand is None:
                cand = oc
            elif cand != oc:
                return None
        return cand

    occ = learn_occluder()
    if occ is None:
        return results

    def repair_same_shape(g):
        h, w = dims(g)
        out = copy_grid(g)
        maps = [\
            lambda r, c: (r, w - 1 - c),\
            lambda r, c: (h - 1 - r, c),\
            lambda r, c: (h - 1 - r, w - 1 - c),\
        ]
        if h == w:
            maps.append(lambda r, c: (c, r))
            maps.append(lambda r, c: (w - 1 - c, h - 1 - r))
        valid_maps = []
        for m in maps:
            consistent = True
            for r in range(h):
                for c in range(w):
                    rr, cc = m(r, c)
                    if g[r][c] != occ and g[rr][cc] != occ and g[r][c] != g[rr][cc]:
                        consistent = False
                        break
                if not consistent:
                    break
            if consistent:
                valid_maps.append(m)
        changed = True
        while changed:
            changed = False
            for r in range(h):
                for c in range(w):
                    if out[r][c] == occ:
                        for m in valid_maps:
                            rr, cc = m(r, c)
                            if out[rr][cc] != occ:
                                out[r][c] = out[rr][cc]
                                changed = True
                                break
        return out

    if verify(repair_same_shape, train):
        results.append(("symmetry_repair", repair_same_shape))

    def repair_periodic(g):
        h, w = dims(g)
        out = copy_grid(g)

        def find_period(n, axis):
            for p in range(1, n):
                ok = True
                for r in range(h):
                    for c in range(w):
                        if g[r][c] == occ:
                            continue
                        if axis == 0:
                            r2 = r + p
                            if r2 < h and g[r2][c] != occ and g[r2][c] != g[r][c]:
                                ok = False
                                break
                        else:
                            c2 = c + p
                            if c2 < w and g[r][c2] != occ and g[r][c2] != g[r][c]:
                                ok = False
                                break
                    if not ok:
                        break
                if ok:
                    return p
            return None

        pr = find_period(h, 0)
        pc = find_period(w, 1)
        changed = True
        while changed:
            changed = False
            for r in range(h):
                for c in range(w):
                    if out[r][c] != occ:
                        continue
                    val = None
                    if pr:
                        for k in range(-(h // pr) - 1, (h // pr) + 2):
                            rr = r + k * pr
                            if 0 <= rr < h and out[rr][c] != occ:
                                val = out[rr][c]
                                break
                    if val is None and pc:
                        for k in range(-(w // pc) - 1, (w // pc) + 2):
                            cc = c + k * pc
                            if 0 <= cc < w and out[r][cc] != occ:
                                val = out[r][cc]
                                break
                    if val is not None:
                        out[r][c] = val
                        changed = True
        return out

    if verify(repair_periodic, train):
        results.append(("periodic_repair", repair_periodic))

    repair_fn = None
    for name, fn in results:
        if name in ("symmetry_repair", "periodic_repair"):
            repair_fn = fn
            break
    if repair_fn is not None:

        def repair_crop(g):
            h, w = dims(g)
            cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] == occ]
            if not cells:
                return None
            full = repair_fn(g)
            if full is None:
                return None
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            return subgrid(
                full, min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
            )

        if verify(repair_crop, train):
            results.append(("repair_crop", repair_crop))

    return results

def build_symmetrize(train):
    results = []

    def make_overlay(ops):
        def solver(g):
            bg = background_color(g)
            h, w = dims(g)
            out = copy_grid(g)
            for op in ops:
                t = op(g)
                if dims(t) != (h, w):
                    return None
                for r in range(h):
                    for c in range(w):
                        if out[r][c] == bg and t[r][c] != bg:
                            out[r][c] = t[r][c]
            return out

        return solver

    for name, ops in (
        ("sym_h", [flip_h]),
        ("sym_v", [flip_v]),
        ("sym_hv", [flip_h, flip_v, rot180]),
        ("sym_rot", [rot90, rot180, rot270]),
    ):
        s = make_overlay(ops)
        if verify(s, train):
            results.append((name, s))
    return results

def find_separator_lines(g):
    h, w = dims(g)
    line_colors = Counter()
    for r in range(h):
        if len(set(g[r])) == 1:
            line_colors[g[r][0]] += 1
    for c in range(w):
        colset = set(g[r][c] for r in range(h))
        if len(colset) == 1:
            line_colors[next(iter(colset))] += 1
    if not line_colors:
        return [], [], None
    sep_color = line_colors.most_common(1)[0][0]
    row_seps = [r for r in range(h) if all(v == sep_color for v in g[r])]
    col_seps = [c for c in range(w) if all(g[r][c] == sep_color for r in range(h))]
    return row_seps, col_seps, sep_color

def split_panels(g):
    h, w = dims(g)
    row_seps, col_seps, sep = find_separator_lines(g)
    if row_seps or col_seps:

        def segments(n, seps):
            segs = []
            prev = 0
            for s in seps:
                if s > prev:
                    segs.append((prev, s))
                prev = s + 1
            if prev < n:
                segs.append((prev, n))
            return segs

        rsegs = segments(h, row_seps) if row_seps else [(0, h)]
        csegs = segments(w, col_seps) if col_seps else [(0, w)]
        panels = []
        for (r0, r1) in rsegs:
            for (c0, c1) in csegs:
                panels.append(subgrid(g, r0, c0, r1 - r0, c1 - c0))
        if panels and len(set(dims(p) for p in panels)) == 1 and len(panels) >= 2:
            return panels, "sep"
    if w % 2 == 0:
        return [subgrid(g, 0, 0, h, w // 2), subgrid(g, 0, w // 2, h, w // 2)], "halfw"
    if h % 2 == 0:
        return [subgrid(g, 0, 0, h // 2, w), subgrid(g, h // 2, 0, h // 2, w)], "halfh"
    return None, None

def build_panel_combine(train):
    results = []

    def panels_of(g):
        panels, _ = split_panels(g)
        return panels

    for pair in train:
        panels = panels_of(pair["input"])
        if not panels or len(panels) < 2 or dims(panels[0]) != dims(pair["output"]):
            return results

    def cell_bool(panels, mode):
        h, w = dims(panels[0])
        allflat = [v for p in panels for row in p for v in row]
        bg = Counter(allflat).most_common(1)[0][0]
        res = [[False] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                vals = [p[r][c] != bg for p in panels]
                if mode == "and":
                    res[r][c] = all(vals)
                elif mode == "or":
                    res[r][c] = any(vals)
                elif mode == "xor":
                    res[r][c] = (sum(vals) % 2) == 1
                elif mode == "nand":
                    res[r][c] = not all(vals)
                elif mode == "nor":
                    res[r][c] = not any(vals)
                elif mode == "diff":
                    res[r][c] = vals[0] and not any(vals[1:])
        return res

    for mode in ("and", "or", "xor", "nand", "nor", "diff"):
        on_color = off_color = None
        consistent = True
        for pair in train:
            panels = panels_of(pair["input"])
            mask = cell_bool(panels, mode)
            go = pair["output"]
            h, w = dims(go)
            for r in range(h):
                for c in range(w):
                    if mask[r][c]:
                        if on_color is None:
                            on_color = go[r][c]
                        elif on_color != go[r][c]:
                            consistent = False
                    else:
                        if off_color is None:
                            off_color = go[r][c]
                        elif off_color != go[r][c]:
                            consistent = False
            if not consistent:
                break
        if consistent and on_color is not None and off_color is not None:

            def make(mode=mode, on_color=on_color, off_color=off_color):
                def solver(g):
                    panels = panels_of(g)
                    if not panels or len(panels) < 2:
                        return None
                    mask = cell_bool(panels, mode)
                    return [\
                        [\
                            on_color if mask[r][c] else off_color\
                            for c in range(len(mask[0]))\
                        ]\
                        for r in range(len(mask))\
                    ]

                return solver

            s = make()
            if verify(s, train):
                results.append(("panel_%s" % mode, s))
    return results

def build_panel_select(train):
    results = []

    def panels_of(g):
        panels, _ = split_panels(g)
        return panels

    for pair in train:
        panels = panels_of(pair["input"])
        if not panels or len(panels) < 2 or dims(panels[0]) != dims(pair["output"]):
            return results

    def pick_odd(g):
        panels = panels_of(g)
        if not panels:
            return None
        keys = [json.dumps(p) for p in panels]
        cnt = Counter(keys)
        odd = [p for p, k in zip(panels, keys) if cnt[k] == 1]
        return odd[0] if len(odd) == 1 else None

    def pick_common(g):
        panels = panels_of(g)
        if not panels:
            return None
        keys = [json.dumps(p) for p in panels]
        cnt = Counter(keys)
        return json.loads(cnt.most_common(1)[0][0])

    for fn, name in ((pick_odd, "panel_odd"), (pick_common, "panel_common")):
        if verify(fn, train):
            results.append((name, fn))
    return results

def build_border(train):
    results = []
    diffs = set()
    for pair in train:
        hi, wi = dims(pair["input"])
        ho, wo = dims(pair["output"])
        diffs.add((ho - hi, wo - wi))
    if len(diffs) == 1:
        dh, dw = diffs.pop()
        if dh == 2 and dw == 2:
            bc = None
            for pair in train:
                go = pair["output"]
                border = set(go[0]) | set(go[-1]) | set(r[0] for r in go) | set(
                    r[-1] for r in go
                )
                if len(border) == 1:
                    col = next(iter(border))
                    if bc is None:
                        bc = col
                    elif bc != col:
                        bc = None
                        break
                else:
                    bc = None
                    break
            if bc is not None:

                def add_border(g, bc=bc):
                    w = len(g[0])
                    out = [[bc] * (w + 2)]
                    for row in g:
                        out.append([bc] + row[:] + [bc])
                    out.append([bc] * (w + 2))
                    return out

                if verify(add_border, train):
                    results.append(("add_border", add_border))
    return results

def build_gravity(train):
    results = []
    for direction in ("down", "up", "left", "right"):

        def make(direction=direction):
            def solver(g):
                bg = background_color(g)
                h, w = dims(g)
                out = [[bg] * w for _ in range(h)]
                if direction in ("down", "up"):
                    for c in range(w):
                        col = [g[r][c] for r in range(h) if g[r][c] != bg]
                        if direction == "down":
                            for i, v in enumerate(reversed(col)):
                                out[h - 1 - i][c] = v
                        else:
                            for i, v in enumerate(col):
                                out[i][c] = v
                else:
                    for r in range(h):
                        row = [v for v in g[r] if v != bg]
                        if direction == "right":
                            for i, v in enumerate(reversed(row)):
                                out[r][w - 1 - i] = v
                        else:
                            for i, v in enumerate(row):
                                out[r][i] = v
                return out

            return solver

        s = make()
        if verify(s, train):
            results.append(("gravity_%s" % direction, s))
    return results

def build_constant(train):
    outs = [pair["output"] for pair in train]
    first = outs[0]
    if all(o == first for o in outs):
        const = copy_grid(first)
        return [("constant", lambda g: copy_grid(const))]
    return []

def build_color_summary(train):
    results = []
    if not all(dims(p["output"]) == (1, 1) for p in train):
        return results

    def mk(fn):
        def solver(g):
            v = fn(g)
            return [[v]] if v is not None else None

        return solver

    def most_freq_nonbg(g):
        bg = background_color(g)
        c = Counter(v for row in g for v in row if v != bg)
        return c.most_common(1)[0][0] if c else None

    def least_freq_nonbg(g):
        bg = background_color(g)
        c = Counter(v for row in g for v in row if v != bg)
        return c.most_common()[-1][0] if c else None

    def largest_obj_color(g):
        bg = background_color(g)
        objs = connected_components(g, bg, 4, True)
        return max(objs, key=lambda o: o["size"])["main_color"] if objs else None

    def smallest_obj_color(g):
        bg = background_color(g)
        objs = connected_components(g, bg, 4, True)
        return min(objs, key=lambda o: o["size"])["main_color"] if objs else None

    for fn, name in (
        (most_freq_nonbg, "most_freq_color"),
        (least_freq_nonbg, "least_freq_color"),
        (largest_obj_color, "largest_obj_color"),
        (smallest_obj_color, "smallest_obj_color"),
    ):
        s = mk(fn)
        if verify(s, train):
            results.append((name, s))
    return results

def build_fractal(train):
    results = []
    for pair in train:
        hi, wi = dims(pair["input"])
        ho, wo = dims(pair["output"])
        if hi == 0 or wi == 0 or ho != hi * hi or wo != wi * wi:
            return results
    for invert in (False, True):

        def make(invert=invert):
            def solver(g):
                bg = background_color(g)
                h, w = dims(g)
                out = [[bg] * (w * w) for _ in range(h * h)]
                for r in range(h):
                    for c in range(w):
                        on = g[r][c] != bg
                        if invert:
                            on = not on
                        if on:
                            for rr in range(h):
                                for cc in range(w):
                                    out[r * h + rr][c * w + cc] = g[rr][cc]
                return out

            return solver

        s = make()
        if verify(s, train):
            results.append(("fractal%s" % ("_inv" if invert else ""), s))
    return results

def build_denoise(train):
    results = []
    if not all(dims(p["input"]) == dims(p["output"]) for p in train):
        return results

    def mk(rule, conn, sc):
        def solver(g):
            bg = background_color(g)
            objs = connected_components(g, bg, conn, sc)
            if not objs:
                return copy_grid(g)
            keep = rule(objs)
            keepcells = set()
            for o in keep:
                keepcells.update(o["cells"])
            h, w = dims(g)
            return [\
                [g[r][c] if (r, c) in keepcells else bg for c in range(w)]\
                for r in range(h)\
            ]

        return solver

    def rule_largest(objs):
        m = max(o["size"] for o in objs)
        return [o for o in objs if o["size"] == m]

    def rule_remove_singletons(objs):
        return [o for o in objs if o["size"] > 1]

    def rule_most_common_size(objs):
        common = Counter(o["size"] for o in objs).most_common(1)[0][0]
        return [o for o in objs if o["size"] == common]

    def rule_remove_most_common_color(objs):
        common = Counter(o["main_color"] for o in objs).most_common(1)[0][0]
        return [o for o in objs if o["main_color"] != common]

    for rule, name in (
        (rule_largest, "keep_largest"),
        (rule_remove_singletons, "remove_noise"),
        (rule_most_common_size, "keep_common_size"),
        (rule_remove_most_common_color, "remove_common_color"),
    ):
        placed = False
        for conn in (4, 8):
            for sc in (True, False):
                s = mk(rule, conn, sc)
                if verify(s, train):
                    results.append((name, s))
                    placed = True
                    break
            if placed:
                break
    return results

def build_object_recolor(train):
    results = []
    if not all(dims(p["input"]) == dims(p["output"]) for p in train):
        return results

    def norm_shape(o):
        return frozenset((r - o["r0"], c - o["c0"]) for r, c in o["cells"])

    for key_name, keyfn in (
        ("size", lambda o: o["size"]),
        ("shape", lambda o: norm_shape(o)),
        ("dims", lambda o: (o["h"], o["w"])),
    ):
        placed = False
        for conn in (4, 8):
            for sc in (True, False):
                mapping = {}
                consistent = True
                for pair in train:
                    gi, go = pair["input"], pair["output"]
                    bg = background_color(gi)
                    objs = connected_components(gi, bg, conn, sc)
                    for o in objs:
                        outcolors = set(go[r][c] for r, c in o["cells"])
                        if len(outcolors) != 1:
                            consistent = False
                            break
                        oc = next(iter(outcolors))
                        k = keyfn(o)
                        if k in mapping and mapping[k] != oc:
                            consistent = False
                            break
                        mapping[k] = oc
                    if not consistent:
                        break
                if not consistent or not mapping:
                    continue

                def make(keyfn=keyfn, conn=conn, sc=sc, mapping=dict(mapping)):
                    def solver(g):
                        bg = background_color(g)
                        objs = connected_components(g, bg, conn, sc)
                        out = copy_grid(g)
                        for o in objs:
                            k = keyfn(o)
                            if k not in mapping:
                                return None
                            col = mapping[k]
                            for r, c in o["cells"]:
                                out[r][c] = col
                        return out

                    return solver

                s = make()
                if verify(s, train):
                    results.append(("recolor_by_%s" % key_name, s))
                    placed = True
                    break
            if placed:
                break
    return results

def build_pattern_denoise(train):
    """Same-shape tasks where the output enforces a periodic regularity along
    rows and/or columns: cells that deviate from the dominant period are
    rewritten to the period-consistent value. Covers ARC 'repair the broken
    pattern' tasks that have no single occluder color."""
    results = []
    if not all(dims(p["input"]) == dims(p["output"]) for p in train):
        return results

    def best_period_1d(seq):
        n = len(seq)
        best_p, best_score = None, -1.0
        for p in range(1, n // 2 + 1):
            total = 0
            agree = 0
            for phase in range(p):
                col = [seq[j] for j in range(phase, n, p)]
                if col:
                    total += len(col)
                    agree += Counter(col).most_common(1)[0][1]
            if total == 0:
                continue
            ratio = agree / total
            # require strong regularity; prefer smaller period on ties
            if ratio < 0.75:
                continue
            score = ratio - 0.001 * p
            if score > best_score:
                best_score, best_p = score, p
        return best_p

    def repair_axis(g, axis):
        h, w = dims(g)
        out = copy_grid(g)
        if axis == "row":
            for r in range(h):
                seq = g[r]
                p = best_period_1d(seq)
                if not p:
                    continue
                for phase in range(p):
                    idx = list(range(phase, w, p))
                    col = [seq[j] for j in idx]
                    maj = Counter(col).most_common(1)[0][0]
                    for j in idx:
                        out[r][j] = maj
        else:
            for c in range(w):
                seq = [g[r][c] for r in range(h)]
                p = best_period_1d(seq)
                if not p:
                    continue
                for phase in range(p):
                    idx = list(range(phase, h, p))
                    col = [seq[j] for j in idx]
                    maj = Counter(col).most_common(1)[0][0]
                    for j in idx:
                        out[j][c] = maj
        return out

    def repair_both(g):
        return repair_axis(repair_axis(g, "row"), "col")

    for name, fn in (
        ("pattern_repair_row", lambda g: repair_axis(g, "row")),
        ("pattern_repair_col", lambda g: repair_axis(g, "col")),
        ("pattern_repair_both", repair_both),
    ):
        if verify(fn, train):
            results.append((name, fn))
    return results

# ============================================================================
# Master
# ============================================================================
BUILDERS = [\
    build_geometric,\
    build_color_remap,\
    build_tiling,\
    build_scale,\
    build_fractal,\
    build_crop_content,\
    build_object_select,\
    build_symmetry_repair,\
    build_symmetrize,\
    build_panel_combine,\
    build_panel_select,\
    build_border,\
    build_gravity,\
    build_denoise,\
    build_object_recolor,\
    build_pattern_denoise,\
    build_color_summary,\
    build_constant,\
]

PREFERENCE = [\
    "color_remap",\
    "recolor_by_shape",\
    "recolor_by_dims",\
    "recolor_by_size",\
    "symmetry_repair",\
    "periodic_repair",\
    "pattern_repair_both",\
    "pattern_repair_row",\
    "pattern_repair_col",\
    "repair_crop",\
    "panel_and",\
    "panel_or",\
    "panel_xor",\
    "panel_nand",\
    "panel_nor",\
    "panel_diff",\
    "panel_odd",\
    "panel_common",\
    "fractal",\
    "fractal_inv",\
    "select_unique_color",\
    "select_unique_shape",\
    "select_most_common_shape",\
    "select_largest",\
    "select_smallest",\
    "crop_largest_obj",\
    "crop_smallest_obj",\
    "crop_content",\
    "keep_largest",\
    "remove_noise",\
    "keep_common_size",\
    "remove_common_color",\
    "sym_hv",\
    "sym_rot",\
    "sym_h",\
    "sym_v",\
    "add_border",\
    "gravity_down",\
    "gravity_up",\
    "gravity_left",\
    "gravity_right",\
    "tile_mirror",\
    "tile_plain",\
    "upscale",\
    "downscale",\
    "most_freq_color",\
    "least_freq_color",\
    "largest_obj_color",\
    "smallest_obj_color",\
    "constant",\
    "rot180",\
    "rot90",\
    "rot270",\
    "flip_h",\
    "flip_v",\
    "transpose",\
    "anti_transpose",\
    "identity",\
]

def rank_key(name):
    return PREFERENCE.index(name) if name in PREFERENCE else len(PREFERENCE)

def fitted_solvers(train):
    fitted = []
    for builder in BUILDERS:
        try:
            for name, fn in builder(train):
                fitted.append((name, fn))
        except Exception:
            continue
    fitted.sort(key=lambda t: rank_key(t[0]))
    if fitted:
        return fitted
    # No single-stage program fit. Try COMPOSITIONAL search: apply a
    # grid-reducing / normalizing PRE-stage, then re-run the full builder
    # library on the transformed train pairs. Verify the composite end-to-end.
    comp = composed_solvers(train)
    comp.sort(key=lambda t: rank_key(t[0].split("+")[-1]))
    return comp

# ============================================================================
# Compositional search
# ============================================================================
def _pre_stage_candidates(train):
    """Return list of (name, pre_fn) grid->grid transforms that are applied
    to BOTH train input and (implicitly) carried into the next stage. These
    are task-agnostic normalizers that often expose a learnable second stage
    on ARC-AGI-2 (crop to content, panel split halves, geometric views)."""
    stages = []

    def crop_content(g):
        bg = background_color(g)
        cells = [\
            (r, c)\
            for r in range(len(g))\
            for c in range(len(g[0]))\
            if g[r][c] != bg\
        ]
        if not cells:
            return None
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return subgrid(g, min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    stages.append(("crop", crop_content))

    def largest_obj_crop(g):
        bg = background_color(g)
        objs = connected_components(g, bg, 8, False)
        if not objs:
            return None
        o = max(objs, key=lambda x: x["size"])
        return subgrid(g, o["r0"], o["c0"], o["h"], o["w"])

    stages.append(("objcrop", largest_obj_crop))

    for name, fn in GEOMETRIC.items():
        if name != "identity":
            stages.append(("pre_" + name, fn))

    # panel halves (left/top) as a pre-stage
    def panel0(g):
        panels, _ = split_panels(g)
        return panels[0] if panels else None

    def panel1(g):
        panels, _ = split_panels(g)
        return panels[1] if panels and len(panels) > 1 else None

    stages.append(("panel0", panel0))
    stages.append(("panel1", panel1))
    return stages

def composed_solvers(train, max_results=4):
    """Two-stage composition: pre-stage then full builder library.
    Verified end-to-end on all train pairs."""
    results = []
    for pre_name, pre_fn in _pre_stage_candidates(train):
        # Build transformed train set
        try:
            new_train = []
            ok = True
            for pair in train:
                ti = pre_fn(pair["input"])
                if ti is None or not valid_grid(ti):
                    ok = False
                    break
                new_train.append({"input": ti, "output": pair["output"]})
            if not ok:
                continue
        except Exception:
            continue
        # Run builders on transformed inputs
        for builder in BUILDERS:
            try:
                built = builder(new_train)
            except Exception:
                continue
            for name, fn in built:
                def make(pre_fn=pre_fn, fn=fn):
                    def solver(g):
                        ti = pre_fn(g)
                        if ti is None or not valid_grid(ti):
                            return None
                        return fn(ti)
                    return solver
                composite = make()
                if verify(composite, train):
                    results.append(("%s+%s" % (pre_name, name), composite))
                    if len(results) >= max_results:
                        return results
    return results

def most_common_train_output(train):
    counts = Counter(json.dumps(pair["output"]) for pair in train)
    return json.loads(counts.most_common(1)[0][0])

def dominant_output_shape(train):
    shapes = Counter((len(p["output"]), len(p["output"][0])) for p in train)
    return shapes.most_common(1)[0][0]

def predict_for_test_input(train, test_input):
    candidates = []
    seen = set()

    def push(grid):
        if grid is None or not valid_grid(grid):
            return
        key = json.dumps(grid)
        if key not in seen:
            seen.add(key)
            candidates.append(grid)

    for _name, fn in fitted_solvers(train):
        try:
            push(fn(test_input))
        except Exception:
            pass
        if len(candidates) >= 2:
            break

    push(copy_grid(test_input))
    try:
        push(most_common_train_output(train))
    except Exception:
        pass
    try:
        h, w = dominant_output_shape(train)
        flat = flatten(test_input)
        fill = Counter(flat).most_common(1)[0][0] if flat else 0
        push([[fill] * w for _ in range(h)])
    except Exception:
        pass

    while len(candidates) < 2:
        if not candidates:
            candidates.append([[0]])
        else:
            candidates.append([[0, 0]] if candidates[0] != [[0, 0]] else [[0]])

    return candidates[0], candidates[1]

def build_submission(challenges):
    submission = {}
    for task_id, task in challenges.items():
        train = task["train"]
        entries = []
        for test_case in task["test"]:
            a1, a2 = predict_for_test_input(train, test_case["input"])
            entries.append({"attempt_1": a1, "attempt_2": a2})
        submission[task_id] = entries
    return submission

def validate_submission(submission, challenges, sample):
    errors = []
    if set(submission.keys()) != set(challenges.keys()):
        errors.append("Task id set does not match test challenges.")
    if sample is not None and set(submission.keys()) != set(sample.keys()):
        errors.append("Task id set does not match sample_submission.")
    for task_id, task in challenges.items():
        if task_id not in submission:
            errors.append(f"{task_id}: missing from submission")
            continue
        entries = submission[task_id]
        n_test = len(task["test"])
        if not isinstance(entries, list):
            errors.append(f"{task_id}: value is not a list")
            continue
        if len(entries) != n_test:
            errors.append(f"{task_id}: expected {n_test} entries, got {len(entries)}")
        for i, entry in enumerate(entries):
            if set(entry.keys()) != {"attempt_1", "attempt_2"}:
                errors.append(f"{task_id}[{i}]: wrong attempt keys {set(entry.keys())}")
                continue
            for att in ("attempt_1", "attempt_2"):
                if not valid_grid(entry[att]):
                    errors.append(f"{task_id}[{i}].{att}: invalid grid")
    return errors

def main():
    with open(TEST_CHALLENGES) as f:
        challenges = json.load(f)
    sample = None
    if os.path.exists(SAMPLE_SUBMISSION):
        with open(SAMPLE_SUBMISSION) as f:
            sample = json.load(f)

    print(f"Loaded {len(challenges)} test tasks.")
    submission = build_submission(challenges)

    errors = validate_submission(submission, challenges, sample)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for e in errors[:20]:
            print("  -", e)
        raise SystemExit(1)

    print("VALIDATION PASSED: schema matches sample_submission.json.")
    with open(OUTPUT, "w") as f:
        json.dump(submission, f)
    print(f"Wrote {OUTPUT} ({os.path.getsize(OUTPUT)} bytes).")

if __name__ == "__main__":
    main()
```

## License

This Notebook has been released under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.

## Continue exploring

- input







Input

1 file

- output







Output

1 file

- list\_alt







Logs

37.8 second run - successful

- comment







Comments

0 comments


reCAPTCHA

Recaptcha requires verification.

protected by **reCAPTCHA**