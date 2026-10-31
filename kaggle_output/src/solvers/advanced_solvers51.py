"""
advanced_solvers51.py — CropSubgridSolver
Session 53: Crops an isolated non-background rectangular sub-grid from a grid.

DSA modules: BFS connected-components (isolation detection), heuristic scoring
(minority-count, max-color-count, unique-size disambiguation).
Tier 3.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter
from src.solvers.registry import register_solver
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction

Grid = List[List[int]]
Rect = Tuple[int, int, int, int]   # (r0, c0, h, w)


# ── helpers ──────────────────────────────────────────────────────────────────

def _s51_bg(grid: Grid) -> int:
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _s51_find_rects(grid: Grid, bg: int) -> List[Rect]:
    """
    Return every maximal axis-aligned rectangle where:
      • Every cell inside is non-bg (bbox fully filled).
      • Every cell immediately bordering outside is bg or out-of-bounds (isolated).

    Algorithm: BFS connected-components on non-bg cells → bbox per component →
    validate full-fill + isolation.
    """
    rows, cols = len(grid), len(grid[0])
    nonbg: set = {(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg}

    visited: set = set()
    rects: List[Rect] = []

    for start in sorted(nonbg):
        if start in visited:
            continue
        # BFS component
        comp: List[Tuple[int, int]] = []
        stack = [start]
        visited.add(start)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in nonbg and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1

        # Validate: bbox fully non-bg
        if any(grid[r][c] == bg
               for r in range(r0, r1 + 1)
               for c in range(c0, c1 + 1)):
            continue

        # Validate: all orthogonally adjacent cells outside bbox are bg or OOB
        # (4-connectivity isolation: left/right edges per row, top/bottom edges per col)
        isolated = True
        for r in range(r0, r1 + 1):
            if (c0 > 0 and grid[r][c0 - 1] != bg) or \
               (c1 < cols - 1 and grid[r][c1 + 1] != bg):
                isolated = False
                break
        if isolated:
            for c in range(c0, c1 + 1):
                if (r0 > 0 and grid[r0 - 1][c] != bg) or \
                   (r1 < rows - 1 and grid[r1 + 1][c] != bg):
                    isolated = False
                    break
        if isolated:
            rects.append((r0, c0, h, w))

    return rects


def _s51_extract(grid: Grid, r0: int, c0: int, h: int, w: int) -> Grid:
    return [row[c0:c0 + w] for row in grid[r0:r0 + h]]


def _s51_score_max_color(sub: Grid, color: int) -> int:
    return sum(cell == color for row in sub for cell in row)


def _s51_score_max_distinct(sub: Grid, bg: int) -> int:
    """Count of distinct non-bg colors in the sub-grid."""
    return len({c for row in sub for c in row if c != bg})


def _s51_score_max_minority(sub: Grid, bg: int) -> int:
    flat = [c for row in sub for c in row if c != bg]
    if not flat:
        return 0
    dominant = Counter(flat).most_common(1)[0][0]
    return sum(c != dominant for c in flat)


# ── rule discovery ────────────────────────────────────────────────────────────

def _s51_discover_rule(
    train_pairs: list,
    bg_list: list,
) -> Optional[Dict[str, Any]]:
    """
    Try rules in priority order:
      1. unique_size  — exactly one isolated rect per training pair matches output size.
      2. max_minority — rect with most non-dominant non-bg cells equals output.
      3. max_color_X  — rect with most cells of color X equals output (for every X).

    Pre-condition: every training pair must have its exact output present as one of the
    isolated rects; otherwise None is returned immediately.
    """
    n = len(train_pairs)
    all_rects: List[List[Rect]] = []
    all_subs: List[List[Grid]] = []
    out_grids: List[Grid] = []

    for pair, bg in zip(train_pairs, bg_list):
        rects = _s51_find_rects(pair["input"], bg)
        subs = [_s51_extract(pair["input"], r, c, h, w) for r, c, h, w in rects]
        all_rects.append(rects)
        all_subs.append(subs)
        out_grids.append(pair["output"])

    # Pre-check: correct output must be among the isolated rects
    for i in range(n):
        if out_grids[i] not in all_subs[i]:
            return None

    # ── Rule 1: unique_size ──────────────────────────────────────────────────
    out_sizes = [(len(og), len(og[0])) for og in out_grids]
    if all(
        sum(1 for _, _, h, w in all_rects[i]
            if h == out_sizes[i][0] and w == out_sizes[i][1]) == 1
        for i in range(n)
    ):
        return {"type": "unique_size", "train_out_sizes": out_sizes}

    # Collect all non-bg colors
    all_colors: set = set()
    for pair, bg in zip(train_pairs, bg_list):
        for row in pair["input"]:
            for c in row:
                if c != bg:
                    all_colors.add(c)

    def _check(score_fn) -> bool:
        for i in range(n):
            subs = all_subs[i]
            if not subs:
                return False
            scores = [score_fn(s, i) for s in subs]
            best = max(scores)
            winners = [j for j, s in enumerate(scores) if s == best]
            if len(winners) != 1 or all_subs[i][winners[0]] != out_grids[i]:
                return False
        return True

    # ── Rule 2: max_distinct ─────────────────────────────────────────────────
    # Rect with the most distinct non-bg color values (e.g. 3 colors vs 2)
    if _check(lambda s, i: _s51_score_max_distinct(s, bg_list[i])):
        return {"type": "max_distinct"}

    # ── Rule 3: max_minority ─────────────────────────────────────────────────
    if _check(lambda s, i: _s51_score_max_minority(s, bg_list[i])):
        return {"type": "max_minority"}

    # ── Rule 4: max_color_X ──────────────────────────────────────────────────
    for color in sorted(all_colors):
        if _check(lambda s, i, col=color: _s51_score_max_color(s, col)):
            return {"type": "max_color", "color": color}

    return None


# ── prediction ────────────────────────────────────────────────────────────────

def _s51_predict(inp: Grid, bg: int, rule: Dict[str, Any]) -> Optional[Grid]:
    rects = _s51_find_rects(inp, bg)
    if not rects:
        return None
    subs = [_s51_extract(inp, r, c, h, w) for r, c, h, w in rects]

    rtype = rule["type"]

    if rtype == "unique_size":
        # Find all unique training output sizes that match exactly 1 rect in this input.
        # When multiple such sizes exist (ambiguity), pick the largest-area match.
        seen_sizes: set = set()
        unique_matches: List[Tuple[int, int]] = []  # (area, rect_idx)
        for sz in rule["train_out_sizes"]:
            oh, ow = (sz[0], sz[1])
            if (oh, ow) in seen_sizes:
                continue
            seen_sizes.add((oh, ow))
            idxs = [j for j, (_, _, h, w) in enumerate(rects) if h == oh and w == ow]
            if len(idxs) == 1:
                unique_matches.append((oh * ow, idxs[0]))
        if not unique_matches:
            if len(rects) == 1:
                return subs[0]
            return None   # ambiguous — abstain
        # Largest-area unique-matching rect wins ties
        _, best_idx = max(unique_matches)
        return subs[best_idx]

    if rtype == "max_distinct":
        scores = [_s51_score_max_distinct(s, bg) for s in subs]
        return subs[scores.index(max(scores))]

    if rtype == "max_minority":
        scores = [_s51_score_max_minority(s, bg) for s in subs]
        return subs[scores.index(max(scores))]

    if rtype == "max_color":
        color = rule["color"]
        scores = [_s51_score_max_color(s, color) for s in subs]
        return subs[scores.index(max(scores))]

    return None


# ── solver ────────────────────────────────────────────────────────────────────

@register_solver
class CropSubgridSolver(BaseSolver):
    """
    Detects isolated non-background rectangular sub-grids (BFS + isolation check),
    discovers a discrimination rule from training pairs (unique_size / max_minority /
    max_color_X), validates on all training pairs, then applies to test inputs.

    DSA: BFS connected-components, Counter heuristics, exact grid matching.
    Tier 3.
    """
    name = "CropSubgridSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train_pairs = ctx.train_pairs
        test_inputs = ctx.test_inputs

        if not train_pairs or not test_inputs:
            return None

        bg_list = [_s51_bg(p["input"]) for p in train_pairs]

        # Guard: at least one training input must have ≥2 isolated rects
        # (ensures the discrimination step is non-trivial)
        if not any(
            len(_s51_find_rects(p["input"], bg)) >= 2
            for p, bg in zip(train_pairs, bg_list)
        ):
            return None

        rule = _s51_discover_rule(train_pairs, bg_list)
        if rule is None:
            return None

        # Full training-pair validation gate
        for pair, bg in zip(train_pairs, bg_list):
            pred = _s51_predict(pair["input"], bg, rule)
            if pred is None or pred != pair["output"]:
                return None

        # Apply to test inputs
        predictions = []
        for test_inp in test_inputs:
            test_bg = _s51_bg(test_inp)
            pred = _s51_predict(test_inp, test_bg, rule)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=3,
        )
