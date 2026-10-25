"""
advanced_solvers21.py — 4 pattern solvers for ARC-AGI-2
  1. FloodTowardBorderSolver        (task 13713586)
  2. ColumnPartitionSolver          (task 17b80ad2)
  3. FilledRectanglePartitionSolver (task 1c02dbbe)
  4. CrossAroundPixelSolver         (task 4258a5f9)
"""

from collections import Counter
from typing import Optional

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from .registry import register_solver


def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _bg(grid: Grid) -> int:
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _verify_exact(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


# ===========================================================================
# 1. FloodTowardBorderSolver — task 13713586
# ===========================================================================
#
# One full-edge row or column is uniform non-background → "border".
# For each perpendicular row/col:
#   - Collect non-bg, non-border-color pixels, sort nearest-to-border first
#   - Each pixel floods from its position toward the border up to just before
#     the previous (nearer) pixel; last (farthest) pixel stops at itself.
#
# ===========================================================================

def _detect_border(grid: Grid):
    """Return (side, idx, border_color, bg) or None."""
    H = len(grid)
    W = len(grid[0]) if H else 0
    bg = _bg(grid)

    def row_uniform_nonbg(r):
        return len(set(grid[r])) == 1 and grid[r][0] != bg

    def col_uniform_nonbg(c):
        col = [grid[r][c] for r in range(H)]
        return len(set(col)) == 1 and grid[0][c] != bg

    if row_uniform_nonbg(0):
        return ("top", 0, grid[0][0], bg)
    if row_uniform_nonbg(H - 1):
        return ("bottom", H - 1, grid[H - 1][0], bg)
    if col_uniform_nonbg(0):
        return ("left", 0, grid[0][0], bg)
    if col_uniform_nonbg(W - 1):
        return ("right", W - 1, grid[0][W - 1], bg)
    return None


def _solve_flood_toward_border(grid: Grid) -> Optional[Grid]:
    info = _detect_border(grid)
    if info is None:
        return None
    side, border_idx, border_color, bg = info
    H = len(grid)
    W = len(grid[0])
    out = _copy_grid(grid)

    if side == "top":
        for c in range(W):
            pixels = [(r, grid[r][c]) for r in range(1, H) if grid[r][c] != bg]
            if not pixels:
                continue
            pixels.sort(key=lambda x: x[0])  # nearest to top first
            prev_stop = border_idx  # flood doesn't include border row
            for (r, color) in pixels:
                for fill_r in range(prev_stop + 1, r + 1):
                    out[fill_r][c] = color
                prev_stop = r

    elif side == "bottom":
        for c in range(W):
            pixels = [(r, grid[r][c]) for r in range(0, H - 1) if grid[r][c] != bg]
            if not pixels:
                continue
            pixels.sort(key=lambda x: x[0], reverse=True)  # nearest to bottom first
            prev_stop = border_idx  # flood doesn't include border row
            for (r, color) in pixels:
                for fill_r in range(r, prev_stop):
                    out[fill_r][c] = color
                prev_stop = r

    elif side == "left":
        for r in range(H):
            pixels = [(c, grid[r][c]) for c in range(1, W) if grid[r][c] != bg]
            if not pixels:
                continue
            pixels.sort(key=lambda x: x[0])  # nearest to left first
            prev_stop = border_idx
            for (c, color) in pixels:
                for fill_c in range(prev_stop + 1, c + 1):
                    out[r][fill_c] = color
                prev_stop = c

    elif side == "right":
        for r in range(H):
            pixels = [(c, grid[r][c]) for c in range(0, W - 1) if grid[r][c] != bg]
            if not pixels:
                continue
            pixels.sort(key=lambda x: x[0], reverse=True)  # nearest to right first
            prev_stop = border_idx
            for (c, color) in pixels:
                for fill_c in range(c, prev_stop):
                    out[r][fill_c] = color
                prev_stop = c

    return out


@register_solver
class FloodTowardBorderSolver(BaseSolver):
    """Flood pixels toward the uniform full-edge border (task 13713586)."""

    @property
    def name(self) -> str:
        return "FloodTowardBorderSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        try:
            for pair in ctx.train_pairs:
                if _detect_border(pair["input"]) is None:
                    return None
                result = _solve_flood_toward_border(pair["input"])
                if result is None or not _verify_exact(result, pair["output"]):
                    return None

            predictions = []
            for test_input in ctx.test_inputs:
                out = _solve_flood_toward_border(test_input)
                if out is None:
                    return None
                predictions.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name))

            return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)
        except Exception:
            return None


# ===========================================================================
# 2. ColumnPartitionSolver — task 17b80ad2
# ===========================================================================
#
# Columns with non-background markers are partitioned into color segments.
# Markers (sorted top-to-bottom) define segment boundaries:
#   - Topmost marker: fills rows 0 to its own row
#   - Each subsequent marker: fills from (prev_marker_row + 1) to its own row
#   - Bottommost marker: fills from its row to (H - 1)
# Columns without markers stay as background.
#
# ===========================================================================

def _solve_column_partition(grid: Grid) -> Optional[Grid]:
    """
    Only columns whose last (bottom-most) marker sits at row H-1 are processed.
    Unprocessed columns are kept as-is from input.
    Fill rule for processed columns:
      - marker i fills from (markers[i-1].row + 1) to markers[i].row (own row)
      - first marker fills from row 0 to its own row
      - last marker fills from (prev + 1) to H-1
    """
    H = len(grid)
    W = len(grid[0]) if H else 0
    bg = _bg(grid)
    out = _copy_grid(grid)   # unprocessed columns keep input values

    has_any_processed = False
    for c in range(W):
        markers = [(r, grid[r][c]) for r in range(H) if grid[r][c] != bg]
        if not markers:
            continue
        markers.sort(key=lambda x: x[0])
        # Only process columns where the last marker is at the bottom row
        if markers[-1][0] != H - 1:
            continue
        has_any_processed = True
        for i, (r, color) in enumerate(markers):
            start = 0 if i == 0 else markers[i - 1][0] + 1
            end = r if i < len(markers) - 1 else H - 1
            for fill_r in range(start, end + 1):
                out[fill_r][c] = color

    if not has_any_processed:
        return None
    return out


@register_solver
class ColumnPartitionSolver(BaseSolver):
    """Partition columns by marker colors top-to-bottom (task 17b80ad2)."""

    @property
    def name(self) -> str:
        return "ColumnPartitionSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        try:
            for pair in ctx.train_pairs:
                result = _solve_column_partition(pair["input"])
                if result is None or not _verify_exact(result, pair["output"]):
                    return None

            predictions = []
            for test_input in ctx.test_inputs:
                out = _solve_column_partition(test_input)
                if out is None:
                    return None
                predictions.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name))

            return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)
        except Exception:
            return None


# ===========================================================================
# 3. FilledRectanglePartitionSolver — task 1c02dbbe
# ===========================================================================
#
# A large rectangle filled with color 5 is partitioned into quadrants.
# For each non-0, non-5 color C:
#   - Corner marker at one of the 4 geometric corners → which quadrant (TL/TR/BL/BR)
#   - External LEFT or RIGHT marker (cell just outside rect boundary) → row_bnd
#   - External ABOVE or BELOW marker (cell just outside rect) → col_bnd
# Fill that quadrant of the rect with color C; unfilled quadrants stay 5.
#
# ===========================================================================

def _find_rect_of_fives(grid: Grid):
    """Return (top, left, bottom, right) bounding box of all 5-cells, or None."""
    H = len(grid)
    W = len(grid[0]) if H else 0
    fives = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 5]
    if not fives:
        return None
    rows = [r for r, c in fives]
    cols = [c for r, c in fives]
    return (min(rows), min(cols), max(rows), max(cols))


def _solve_filled_rectangle_partition(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0]) if H else 0
    rect = _find_rect_of_fives(grid)
    if rect is None:
        return None
    rt, rl, rb, rr = rect

    # Find non-0, non-5 special colors
    special_colors = {grid[r][c] for r in range(H) for c in range(W)
                      if grid[r][c] != 0 and grid[r][c] != 5}
    if not special_colors:
        return None

    # Start output: copy input, but clear ALL special-color cells to 0 first
    # (external markers must be erased; corner markers inside rect get overwritten by fill)
    out = _copy_grid(grid)
    for r in range(H):
        for c in range(W):
            if out[r][c] not in (0, 5):
                out[r][c] = 0

    for color in special_colors:
        # Identify corner quadrant
        corner_quad = None
        for (r, c, quad) in [(rt, rl, "TL"), (rt, rr, "TR"), (rb, rl, "BL"), (rb, rr, "BR")]:
            if grid[r][c] == color:
                corner_quad = quad
                break
        if corner_quad is None:
            return None

        # Find row boundary from left/right external marker
        row_bnd = None
        for r in range(rt, rb + 1):
            if rl > 0 and grid[r][rl - 1] == color:
                row_bnd = r
                break
            if rr < W - 1 and grid[r][rr + 1] == color:
                row_bnd = r
                break

        # Find col boundary from above/below external marker
        col_bnd = None
        for c in range(rl, rr + 1):
            if rt > 0 and grid[rt - 1][c] == color:
                col_bnd = c
                break
            if rb < H - 1 and grid[rb + 1][c] == color:
                col_bnd = c
                break

        if row_bnd is None or col_bnd is None:
            return None

        if corner_quad == "TL":
            r_range = range(rt, row_bnd + 1)
            c_range = range(rl, col_bnd + 1)
        elif corner_quad == "TR":
            r_range = range(rt, row_bnd + 1)
            c_range = range(col_bnd, rr + 1)
        elif corner_quad == "BL":
            r_range = range(row_bnd, rb + 1)
            c_range = range(rl, col_bnd + 1)
        else:  # BR
            r_range = range(row_bnd, rb + 1)
            c_range = range(col_bnd, rr + 1)

        for r in r_range:
            for c in c_range:
                out[r][c] = color

    return out


@register_solver
class FilledRectanglePartitionSolver(BaseSolver):
    """Partition a color-5-filled rectangle into quadrants (task 1c02dbbe)."""

    @property
    def name(self) -> str:
        return "FilledRectanglePartitionSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        try:
            for pair in ctx.train_pairs:
                if _find_rect_of_fives(pair["input"]) is None:
                    return None
                result = _solve_filled_rectangle_partition(pair["input"])
                if result is None or not _verify_exact(result, pair["output"]):
                    return None

            predictions = []
            for test_input in ctx.test_inputs:
                out = _solve_filled_rectangle_partition(test_input)
                if out is None:
                    return None
                predictions.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name))

            return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)
        except Exception:
            return None


# ===========================================================================
# 4. CrossAroundPixelSolver — task 4258a5f9
# ===========================================================================
#
# For each pixel with color 5 in the input at (r, c):
#   - All 8 surrounding cells become color 1
#   - Center (r, c) stays color 5
# Pixels of color 5 are preserved; boundary cells (out-of-bounds) are skipped.
#
# ===========================================================================

def _solve_cross_around_pixel(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0]) if H else 0
    fives = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 5]
    if not fives:
        return None
    out = _copy_grid(grid)
    for (r, c) in fives:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if dr == 0 and dc == 0:
                        out[nr][nc] = 5
                    else:
                        out[nr][nc] = 1
    return out


@register_solver
class CrossAroundPixelSolver(BaseSolver):
    """Place 3×3 block of 1s around each color-5 pixel (task 4258a5f9)."""

    @property
    def name(self) -> str:
        return "CrossAroundPixelSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        try:
            for pair in ctx.train_pairs:
                result = _solve_cross_around_pixel(pair["input"])
                if result is None or not _verify_exact(result, pair["output"]):
                    return None

            predictions = []
            for test_input in ctx.test_inputs:
                out = _solve_cross_around_pixel(test_input)
                if out is None:
                    return None
                predictions.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name))

            return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)
        except Exception:
            return None
