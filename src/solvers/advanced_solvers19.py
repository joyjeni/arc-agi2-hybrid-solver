"""
advanced_solvers19.py — 3 pattern solvers
  1. StrayToAdjacentDividerSolver  (task 1a07d186)
  2. IntersectionColorSwapSolver   (task ba97ae07)
  3. DiagonalRayFromInteriorSolver (task ec883f72)
"""

from typing import List, Tuple, Dict, Optional
from collections import Counter

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from .registry import register_solver

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]

def _verify_exact(out: Grid, expected: Grid) -> bool:
    if len(out) != len(expected):
        return False
    for r in range(len(out)):
        if len(out[r]) != len(expected[r]):
            return False
        for c in range(len(out[r])):
            if out[r][c] != expected[r][c]:
                return False
    return True


# ===========================================================================
# 1. StrayToAdjacentDividerSolver  —  task 1a07d186
# ===========================================================================
#
# Rule:
#   Background = 0.
#   Full-row dividers: an entire row filled with one non-bg color.
#   Full-col dividers: an entire column filled with one non-bg color.
#   For each stray cell (non-bg, not part of any divider):
#     - If its color matches a row divider: snap to the divider's near side
#       (divider_row - 1 if stray is above, divider_row + 1 if below),
#       same column. Remove original.
#     - Else if its color matches a col divider: snap to the divider's near
#       side (divider_col - 1 if stray is left, divider_col + 1 if right),
#       same row. Remove original.
#     - No matching divider: delete the stray (set to bg).
#   When multiple dividers of the same color exist, snap to the nearest one.

def _solve_stray_divider(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    bg = 0

    # Find full-row dividers
    row_dividers: Dict[int, int] = {}  # row -> color
    for r in range(H):
        non_bg = [grid[r][c] for c in range(W) if grid[r][c] != bg]
        if len(non_bg) == W and len(set(non_bg)) == 1:
            row_dividers[r] = non_bg[0]

    # Find full-col dividers
    col_dividers: Dict[int, int] = {}  # col -> color
    for c in range(W):
        non_bg = [grid[r][c] for r in range(H) if grid[r][c] != bg]
        if len(non_bg) == H and len(set(non_bg)) == 1:
            col_dividers[c] = non_bg[0]

    # No dividers found → reject
    if not row_dividers and not col_dividers:
        return None

    divider_cells = set()
    for r in row_dividers:
        for c in range(W):
            divider_cells.add((r, c))
    for c in col_dividers:
        for r in range(H):
            divider_cells.add((r, c))

    result = _copy_grid(grid)

    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == bg or (r, c) in divider_cells:
                continue

            # Find nearest matching row divider
            best_dr: Optional[int] = None
            best_dist = float('inf')
            for dr, dv in row_dividers.items():
                if dv == v:
                    d = abs(r - dr)
                    if d < best_dist:
                        best_dist = d
                        best_dr = dr
            if best_dr is not None:
                side = -1 if r < best_dr else 1   # near side of divider
                nr = best_dr + side
                result[r][c] = bg
                if 0 <= nr < H and result[nr][c] == bg:
                    result[nr][c] = v
                continue

            # Find nearest matching col divider
            best_dc: Optional[int] = None
            best_dist = float('inf')
            for dc, dv in col_dividers.items():
                if dv == v:
                    d = abs(c - dc)
                    if d < best_dist:
                        best_dist = d
                        best_dc = dc
            if best_dc is not None:
                side = -1 if c < best_dc else 1
                nc = best_dc + side
                result[r][c] = bg
                if 0 <= nc < W and result[r][nc] == bg:
                    result[r][nc] = v
                continue

            # No matching divider — delete the stray
            result[r][c] = bg

    return result


@register_solver
class StrayToAdjacentDividerSolver(BaseSolver):
    """Snaps stray cells to their matching full-row/col divider's near side (1a07d186)."""

    TARGET_TASK = "1a07d186"

    @property
    def name(self) -> str:
        return "StrayToAdjacentDividerSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_stray_divider(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_stray_divider(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ===========================================================================
# 2. IntersectionColorSwapSolver  —  task ba97ae07
# ===========================================================================
#
# Rule:
#   Background = most common color.
#   H-bar: a row where a single non-bg color occupies ≥ 40 % of cells.
#   V-bar: a col where a single non-bg color occupies ≥ 40 % of cells.
#   At each intersection (H-bar row r, V-bar col c) where bar colors differ:
#     If grid[r][c] == H-bar color → swap to V-bar color.
#     If grid[r][c] == V-bar color → swap to H-bar color.

def _solve_intersection_swap(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])

    flat = [grid[r][c] for r in range(H) for c in range(W)]
    if not flat:
        return None
    mc = Counter(flat)
    bg = mc.most_common(1)[0][0]

    THRESH = 0.40

    # H-bars
    h_bars: List[Tuple[int, int]] = []  # (row, color)
    for r in range(H):
        row_vals = [grid[r][c] for c in range(W) if grid[r][c] != bg]
        if not row_vals:
            continue
        top = Counter(row_vals).most_common(1)[0]
        if top[1] / W >= THRESH:
            h_bars.append((r, top[0]))

    # V-bars
    v_bars: List[Tuple[int, int]] = []  # (col, color)
    for c in range(W):
        col_vals = [grid[r][c] for r in range(H) if grid[r][c] != bg]
        if not col_vals:
            continue
        top = Counter(col_vals).most_common(1)[0]
        if top[1] / H >= THRESH:
            v_bars.append((c, top[0]))

    if not h_bars or not v_bars:
        return None

    result = _copy_grid(grid)
    for (hr, hc) in h_bars:
        for (vc, vc_color) in v_bars:
            if hc == vc_color:
                continue
            cell = grid[hr][vc]
            if cell == hc:
                result[hr][vc] = vc_color
            elif cell == vc_color:
                result[hr][vc] = hc
    return result


@register_solver
class IntersectionColorSwapSolver(BaseSolver):
    """Swaps colors at H-bar / V-bar intersections (ba97ae07)."""

    TARGET_TASK = "ba97ae07"

    @property
    def name(self) -> str:
        return "IntersectionColorSwapSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_intersection_swap(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_intersection_swap(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ===========================================================================
# 3. DiagonalRayFromInteriorSolver  —  task ec883f72
# ===========================================================================
#
# Rule:
#   Background = most common color (= 0 in all known cases).
#   Two non-bg colors: wall (border-like L/U/rectangle, fill-ratio < 0.70)
#   and interior (compact filled block or single cell, fill-ratio ≥ 0.70).
#   Wall corners = wall cells that have ≥ 1 horizontal AND ≥ 1 vertical
#   wall neighbor (i.e., the bending point of the wall structure).
#   For each pair (interior corner, wall corner) that are diagonally aligned
#   at 45°, with only bg/wall cells in between:
#     Cast a ray from the wall corner outward in the same diagonal direction,
#     painting bg cells with the interior color until the ray exits the grid.

def _solve_diagonal_ray(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])

    flat = [grid[r][c] for r in range(H) for c in range(W)]
    if not flat:
        return None
    mc = Counter(flat)
    bg = mc.most_common(1)[0][0]

    colors = [c for c in mc if c != bg]
    if len(colors) != 2:
        return None

    # Classify wall vs interior by fill ratio of bounding box
    wall_color: Optional[int] = None
    interior_color: Optional[int] = None
    for col in colors:
        rows_c = [r for r in range(H) for cc in range(W) if grid[r][cc] == col]
        cols_c = [cc for r in range(H) for cc in range(W) if grid[r][cc] == col]
        n = len(rows_c)
        if n == 0:
            return None
        rmin, rmax = min(rows_c), max(rows_c)
        cmin, cmax = min(cols_c), max(cols_c)
        bb = (rmax - rmin + 1) * (cmax - cmin + 1)
        fill = n / bb
        if fill < 0.70:
            wall_color = col
        else:
            interior_color = col

    if wall_color is None or interior_color is None:
        return None

    wall_cells = {(r, c) for r in range(H) for c in range(W)
                  if grid[r][c] == wall_color}

    # Interior bounding-box corners
    int_rows = [r for r in range(H) for c in range(W) if grid[r][c] == interior_color]
    int_cols = [c for r in range(H) for c in range(W) if grid[r][c] == interior_color]
    r0, r1 = min(int_rows), max(int_rows)
    c0, c1 = min(int_cols), max(int_cols)
    int_corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]

    # Wall corners: wall cells with ≥1 H and ≥1 V wall neighbor
    wall_corners = [
        (wr, wc) for (wr, wc) in wall_cells
        if any((wr, wc + dc) in wall_cells for dc in (-1, 1))
        and any((wr + dr, wc) in wall_cells for dr in (-1, 1))
    ]

    result = _copy_grid(grid)

    for (wc_r, wc_c) in wall_corners:
        for (ir, ic) in int_corners:
            dr_total = wc_r - ir
            dc_total = wc_c - ic
            if dr_total == 0 or dc_total == 0:
                continue
            if abs(dr_total) != abs(dc_total):
                continue
            step_r = 1 if dr_total > 0 else -1
            step_c = 1 if dc_total > 0 else -1
            # Verify all cells between interior corner and wall corner are bg or wall
            r, c = ir + step_r, ic + step_c
            valid = True
            while (r, c) != (wc_r, wc_c):
                if not (0 <= r < H and 0 <= c < W):
                    valid = False
                    break
                if grid[r][c] != bg and grid[r][c] != wall_color:
                    valid = False
                    break
                r += step_r
                c += step_c
            if not valid:
                continue
            # Paint past the wall corner in the same diagonal direction
            r, c = wc_r + step_r, wc_c + step_c
            while 0 <= r < H and 0 <= c < W:
                if result[r][c] == bg:
                    result[r][c] = interior_color
                r += step_r
                c += step_c

    return result


@register_solver
class DiagonalRayFromInteriorSolver(BaseSolver):
    """Shoots diagonal rays from interior block through wall corners (ec883f72)."""

    TARGET_TASK = "ec883f72"

    @property
    def name(self) -> str:
        return "DiagonalRayFromInteriorSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_diagonal_ray(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_diagonal_ray(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
