"""
advanced_solvers50.py — Session 50 Solvers
=========================================
DSA Modules: Spiral traversal, BFS/line-scan, Rectangle intersection geometry,
             Bounding-box analysis, Row/column scan

Confirmed target tasks:
  SpiralDrawSolver          → 28e73c20  (all-zero input → clockwise rectangular spiral)
  ConnectAlignedPairsSolver → ded97339  (connect same-color aligned pairs in row/col)
  TwoRectGapFillSolver      → d6ad076f  (fill interior gap between 2 solid rectangles)
  GapLineFillSolver         → da2b0fe3  (fill all-bg gap row/col inside pattern bbox)
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import List, Optional, Tuple

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

Grid = List[List[int]]
Pair = dict


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _bg_color(grid: Grid) -> int:
    """Return the most-frequent color (background)."""
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _nonbg_colors(grid: Grid, bg: int):
    return [c for row in grid for c in row if c != bg]


def _bounding_box(grid: Grid, bg: int) -> Optional[Tuple[int, int, int, int]]:
    """Return (r1, c1, r2, c2) bounding box of non-bg cells, or None."""
    rows, cols = len(grid), len(grid[0])
    r1, c1, r2, c2 = rows, cols, -1, -1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                r1 = min(r1, r)
                c1 = min(c1, c)
                r2 = max(r2, r)
                c2 = max(c2, c)
    if r2 == -1:
        return None
    return r1, c1, r2, c2


def _find_solid_rect(grid: Grid, bg: int) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Find the first contiguous solid rectangle of a single non-bg color.
    Returns (r1, c1, r2, c2, color) or None.
    A solid rectangle: every cell in that row/col range has the same color.
    """
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                # Expand rectangle
                r2 = r
                while r2 + 1 < rows and all(grid[r2 + 1][cc] == color for cc in range(c, c + 1)):
                    r2 += 1
                c2 = c
                while c2 + 1 < cols and all(grid[rr][c2 + 1] == color for rr in range(r, r2 + 1)):
                    c2 += 1
                # Verify solid
                solid = all(grid[rr][cc] == color for rr in range(r, r2 + 1) for cc in range(c, c2 + 1))
                if solid:
                    for rr in range(r, r2 + 1):
                        for cc in range(c, c2 + 1):
                            visited[rr][cc] = True
                    return r, c, r2, c2, color
    return None


def _find_all_solid_rects(grid: Grid, bg: int) -> List[Tuple[int, int, int, int, int]]:
    """Find all solid rectangles of non-bg colors."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    rects = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                # Grow r2
                r2 = r
                while r2 + 1 < rows and grid[r2 + 1][c] == color:
                    r2 += 1
                # Grow c2
                c2 = c
                while c2 + 1 < cols and all(grid[rr][c2 + 1] == color for rr in range(r, r2 + 1)):
                    c2 += 1
                # Verify entire rectangle
                solid = all(grid[rr][cc] == color for rr in range(r, r2 + 1) for cc in range(c, c2 + 1))
                # Verify no stray cells of same color outside this rect
                if solid:
                    for rr in range(rows):
                        for cc in range(cols):
                            if grid[rr][cc] == color and not (r <= rr <= r2 and c <= cc <= c2):
                                solid = False
                                break
                        if not solid:
                            break
                if solid:
                    for rr in range(r, r2 + 1):
                        for cc in range(c, c2 + 1):
                            visited[rr][cc] = True
                    rects.append((r, c, r2, c2, color))
    return rects


# ─────────────────────────────────────────────────────────────────────────────
# 1. SpiralDrawSolver
# ─────────────────────────────────────────────────────────────────────────────

def _draw_spiral(h: int, w: int, color: int) -> Grid:
    """
    Draw a clockwise rectangular spiral on an h×w grid.
    Sequence: RIGHT w, then pairs (DOWN, LEFT), (UP, RIGHT) with step counts
    decreasing by 2 each revolution.
    Verified against all 5 training pairs of task 28e73c20.
    """
    grid = [[0] * w for _ in range(h)]
    r, c = 0, 0

    DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    def fill(dr, dc, steps):
        nonlocal r, c
        for _ in range(steps):
            if 0 <= r < h and 0 <= c < w:
                grid[r][c] = color
            r += dr
            c += dc
        r -= dr; c -= dc  # undo last over-advance

    def pivot(dr, dc):
        nonlocal r, c
        r += dr; c += dc

    # First segment: RIGHT w
    fill(0, 1, w)

    h_k = h - 1   # step count for vertical moves
    w_k = w - 1   # step count for horizontal moves
    d_idx = 1      # start with DOWN

    while h_k > 0 and w_k > 0:
        for _ in range(2):
            dr, dc = DIRS[d_idx]
            k = h_k if d_idx in (1, 3) else w_k
            pivot(dr, dc)
            fill(dr, dc, k)
            d_idx = (d_idx + 1) % 4
        h_k -= 2
        w_k -= 2

    return grid


@register_solver
class SpiralDrawSolver(BaseSolver):
    name = "SpiralDrawSolver"
    """
    DSA: Spiral traversal.
    All-zero input grid → output is a clockwise rectangular spiral
    with a single learned color.
    Condition: all training inputs are all-zero; outputs have exactly 1 non-zero color.
    """

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Learn color from training pairs
        learned_color = None
        for pair in ctx.train_pairs:
            inp = pair["input"]
            out = pair["output"]
            # Input must be all zeros
            if any(c != 0 for row in inp for c in row):
                return None
            # Output must have exactly one non-zero color
            nz = [c for row in out for c in row if c != 0]
            if not nz:
                return None
            colors = set(nz)
            if len(colors) != 1:
                return None
            fc = colors.pop()
            if learned_color is None:
                learned_color = fc
            elif learned_color != fc:
                return None
            # Verify spiral matches
            h, w = len(out), len(out[0])
            expected = _draw_spiral(h, w, fc)
            if expected != out:
                return None

        if learned_color is None:
            return None

        preds = []
        for inp in ctx.test_inputs:
            # Input must be all zeros
            if any(c != 0 for row in inp for c in row):
                return None
            h, w = len(inp), len(inp[0])
            out = _draw_spiral(h, w, learned_color)
            preds.append(Prediction(attempt_1=out, attempt_2=out))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConnectAlignedPairsSolver
# ─────────────────────────────────────────────────────────────────────────────

def _connect_aligned_pairs(grid: Grid, fg: int, bg: int) -> Grid:
    """
    For each fg cell, if there is another fg cell in the same row or same column
    with only bg cells between them, fill those bg cells with fg.
    Uses scan-line approach (1D interval fill).
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Row scan
    for r in range(rows):
        fg_cols = [c for c in range(cols) if grid[r][c] == fg]
        for i in range(len(fg_cols)):
            for j in range(i + 1, len(fg_cols)):
                c1, c2 = fg_cols[i], fg_cols[j]
                # Check all cells between them are bg
                between = [grid[r][c] for c in range(c1 + 1, c2)]
                if all(v == bg for v in between):
                    for c in range(c1 + 1, c2):
                        result[r][c] = fg

    # Column scan
    for c in range(cols):
        fg_rows = [r for r in range(rows) if grid[r][c] == fg]
        for i in range(len(fg_rows)):
            for j in range(i + 1, len(fg_rows)):
                r1, r2 = fg_rows[i], fg_rows[j]
                between = [grid[r][c] for r in range(r1 + 1, r2)]
                if all(v == bg for v in between):
                    for r in range(r1 + 1, r2):
                        result[r][c] = fg

    return result


@register_solver
class ConnectAlignedPairsSolver(BaseSolver):
    name = "ConnectAlignedPairsSolver"
    """
    DSA: Line-scan, interval fill.
    Input has sparse fg cells on bg. Connect any two fg cells in the same
    row or column (with only bg between them) by filling the gap with fg.
    Condition: exactly 2 non-bg colors (bg + one fg); output same size;
               output matches _connect_aligned_pairs on all training pairs.
    """

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            bg = _bg_color(inp)
            nz_colors = set(c for row in inp for c in row if c != bg)
            if len(nz_colors) != 1:
                return None
            fg = nz_colors.pop()
            expected = _connect_aligned_pairs(inp, fg, bg)
            if expected != out:
                return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg_color(inp)
            nz_colors = set(c for row in inp for c in row if c != bg)
            if len(nz_colors) != 1:
                return None
            fg = nz_colors.pop()
            _ca = _connect_aligned_pairs(inp, fg, bg)
            preds.append(Prediction(attempt_1=_ca, attempt_2=_ca))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TwoRectGapFillSolver
# ─────────────────────────────────────────────────────────────────────────────

def _gap_fill_two_rects(grid: Grid) -> Optional[Grid]:
    """
    Find exactly 2 solid rectangles (non-bg).
    Fill the strictly interior gap region between their row spans AND col spans
    with the fill color learned from the grid itself (if grid has a 3rd color)
    — but for test inputs, fill_color is passed externally.
    Returns (result_grid, fill_color) where fill_color is the gap fill color,
    or None if geometry doesn't produce a valid interior.
    """
    bg = _bg_color(grid)
    rects = _find_all_solid_rects(grid, bg)
    if len(rects) != 2:
        return None, None

    r1a, c1a, r2a, c2a, _ = rects[0]
    r1b, c1b, r2b, c2b, _ = rects[1]

    # Sort row edges
    row_edges = sorted([r1a, r2a, r1b, r2b])
    col_edges = sorted([c1a, c2a, c1b, c2b])

    gap_r1 = row_edges[1] + 1
    gap_r2 = row_edges[2] - 1
    gap_c1 = col_edges[1] + 1
    gap_c2 = col_edges[2] - 1

    if gap_r1 > gap_r2 or gap_c1 > gap_c2:
        return None, None

    return (gap_r1, gap_c1, gap_r2, gap_c2), (rects[0], rects[1])


def _apply_gap_fill(grid: Grid, gap: Tuple, fill_color: int) -> Grid:
    gap_r1, gap_c1, gap_r2, gap_c2 = gap
    result = [row[:] for row in grid]
    for r in range(gap_r1, gap_r2 + 1):
        for c in range(gap_c1, gap_c2 + 1):
            result[r][c] = fill_color
    return result


@register_solver
class TwoRectGapFillSolver(BaseSolver):
    name = "TwoRectGapFillSolver"
    """
    DSA: Rectangle intersection geometry, bounding-box analysis.
    Input has exactly 2 solid rectangular objects (different non-bg colors).
    Output fills the strictly interior gap region (between their row spans
    AND between their col spans) with a learned fill color.
    Condition: all training pairs verified; fill_color learned from training.
    """

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        fill_color = None

        for pair in ctx.train_pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None

            gap_info, rects = _gap_fill_two_rects(inp)
            if gap_info is None:
                return None

            gap_r1, gap_c1, gap_r2, gap_c2 = gap_info
            # Find fill color: any color in output that's in the gap region but not in input
            candidates = set()
            bg = _bg_color(inp)
            for r in range(gap_r1, gap_r2 + 1):
                for c in range(gap_c1, gap_c2 + 1):
                    if inp[r][c] == bg and out[r][c] != bg:
                        candidates.add(out[r][c])

            if len(candidates) != 1:
                return None
            fc = candidates.pop()

            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return None

            # Verify output
            expected = _apply_gap_fill(inp, gap_info, fc)
            if expected != out:
                return None

        if fill_color is None:
            return None

        preds = []
        for inp in ctx.test_inputs:
            gap_info, rects = _gap_fill_two_rects(inp)
            if gap_info is None:
                return None
            _gf = _apply_gap_fill(inp, gap_info, fill_color)
            preds.append(Prediction(attempt_1=_gf, attempt_2=_gf))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. GapLineFillSolver
# ─────────────────────────────────────────────────────────────────────────────

def _find_gap_line(grid: Grid, bg: int) -> Optional[Tuple[str, int]]:
    """
    Find a gap row or column: an all-bg row/col that is strictly inside
    the bounding box of non-bg cells.
    Returns ('row', r) or ('col', c), or None.
    """
    bb = _bounding_box(grid, bg)
    if bb is None:
        return None
    r1, c1, r2, c2 = bb
    rows, cols = len(grid), len(grid[0])

    # Check interior rows (strictly inside bbox)
    for r in range(r1 + 1, r2):
        if all(grid[r][c] == bg for c in range(cols)):
            return ('row', r)

    # Check interior cols (strictly inside bbox)
    for c in range(c1 + 1, c2):
        if all(grid[r][c] == bg for r in range(rows)):
            return ('col', c)

    return None


def _apply_gap_line_fill(grid: Grid, gap: Tuple[str, int], fill_color: int) -> Grid:
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    kind, idx = gap
    if kind == 'row':
        for c in range(cols):
            result[idx][c] = fill_color
    else:
        for r in range(rows):
            result[r][idx] = fill_color
    return result


@register_solver
class GapLineFillSolver(BaseSolver):
    name = "GapLineFillSolver"
    """
    DSA: Bounding-box analysis, row/column scan.
    Input has 1+ non-bg pattern with an all-bg gap row or col inside its bbox.
    Output fills that entire row/col with a learned fill color.
    Condition: gap row/col exists in all training inputs; fill_color consistent.
    """

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        fill_color = None

        for pair in ctx.train_pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None

            bg = _bg_color(inp)
            # Input must have exactly 1 non-bg color (besides potential fill)
            nz_inp = set(c for row in inp for c in row if c != bg)
            if len(nz_inp) != 1:
                return None

            gap = _find_gap_line(inp, bg)
            if gap is None:
                return None

            # Find fill color from diff
            kind, idx = gap
            rows, cols = len(inp), len(inp[0])
            cands = set()
            if kind == 'row':
                for c in range(cols):
                    if inp[idx][c] == bg and out[idx][c] != bg:
                        cands.add(out[idx][c])
            else:
                for r in range(rows):
                    if inp[r][idx] == bg and out[r][idx] != bg:
                        cands.add(out[r][idx])

            if len(cands) != 1:
                return None
            fc = cands.pop()

            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return None

            expected = _apply_gap_line_fill(inp, gap, fc)
            if expected != out:
                return None

        if fill_color is None:
            return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg_color(inp)
            gap = _find_gap_line(inp, bg)
            if gap is None:
                return None
            _gl = _apply_gap_line_fill(inp, gap, fill_color)
            preds.append(Prediction(attempt_1=_gl, attempt_2=_gl))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
