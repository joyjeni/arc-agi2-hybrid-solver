"""
advanced_solvers18.py — 2 pattern solvers
  1. MirrorAcrossDividerSolver  (task ad3b40cf)
  2. PerimeterTilingSolver       (task 30f42897)
"""

from typing import List, Tuple, Dict, Set, Optional

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from .registry import register_solver

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]

def _make_result(grids: List[Grid], ctx: TaskContext, solver_name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=solver_name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds)

def _verify_exact(out: Grid, expected: Grid) -> bool:
    if len(out) != len(expected):
        return False
    return all(out[r][c] == expected[r][c]
               for r in range(len(out)) for c in range(len(out[0])))


# ===========================================================================
# 1. MirrorAcrossDividerSolver  —  task ad3b40cf
# ===========================================================================
#
# Rule:
#   Background = 8.  A full divider line of 1s separates two regions.
#   Two non-background (not 0, 1, 8) colors appear; the one with **fewer**
#   cells is reflected across the divider onto background cells.
#
#   Divider types:
#     'col'       — all 1-cells share the same column  (r → r, c → 2*d - c)
#     'row'       — all 1-cells share the same row     (r → 2*d - r, c → c)
#     'main_diag' — all 1-cells satisfy r == c          (r,c) → (c, r)
#     'anti_diag' — all 1-cells satisfy r+c == K        (r,c) → (K-c, K-r)

def _detect_divider(grid: Grid) -> Tuple[str, int]:
    """Return (kind, param): kind in {'col','row','main_diag','anti_diag'}, param = d or K."""
    H, W = len(grid), len(grid[0])
    ones = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 1]
    if not ones:
        raise ValueError("No 1-cells found")
    rows = [r for r, c in ones]
    cols = [c for r, c in ones]
    diag_main = [r - c for r, c in ones]
    diag_anti = [r + c for r, c in ones]
    if len(set(cols)) == 1:
        return 'col', cols[0]
    if len(set(rows)) == 1:
        return 'row', rows[0]
    if len(set(diag_main)) == 1 and diag_main[0] == 0:
        return 'main_diag', 0
    if len(set(diag_anti)) == 1:
        return 'anti_diag', diag_anti[0]
    raise ValueError("Cannot classify divider")

def _reflect_cell(r: int, c: int, kind: str, param: int) -> Tuple[int, int]:
    if kind == 'col':
        return r, 2 * param - c
    if kind == 'row':
        return 2 * param - r, c
    if kind == 'main_diag':
        return c, r
    # anti_diag
    K = param
    return K - c, K - r

def _solve_mirror(grid: Grid) -> Optional[Grid]:
    IGNORE = {0, 1, 8}
    H, W = len(grid), len(grid[0])
    try:
        kind, param = _detect_divider(grid)
    except ValueError:
        return None

    # Collect non-background colors and their cells
    color_cells: Dict[int, List[Tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v not in IGNORE:
                color_cells.setdefault(v, []).append((r, c))

    if len(color_cells) < 2:
        return None

    # Reflect the color with fewer cells
    counts = {col: len(cells) for col, cells in color_cells.items()}
    reflect_color = min(counts, key=counts.get)
    reflect_cells = color_cells[reflect_color]

    result = _copy_grid(grid)
    for (r, c) in reflect_cells:
        nr, nc = _reflect_cell(r, c, kind, param)
        if 0 <= nr < H and 0 <= nc < W and result[nr][nc] == 8:
            result[nr][nc] = reflect_color
    return result


@register_solver
class MirrorAcrossDividerSolver(BaseSolver):
    """Reflects the minority color across a 1-cell divider line (ad3b40cf)."""

    TARGET_TASK = "ad3b40cf"

    @property
    def name(self) -> str:
        return "MirrorAcrossDividerSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate on training pairs
        for pair in ctx.train_pairs:
            pred = _solve_mirror(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        # Predict on test inputs
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_mirror(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=preds)


# ===========================================================================
# 2. PerimeterTilingSolver  —  task 30f42897
# ===========================================================================
#
# Rule:
#   Background = 8.  All shape cells (non-8) are on the grid perimeter.
#   Perimeter P[0..N-1] clockwise: top L→R, right T→B, bottom R→L, left B→T.
#   Shape length L; period = 2*L.  N must be divisible by period.
#   Find pos0 = start of shape (wrapping-aware: position after largest gap).
#   Tile: for k in 0..N/period-1, place color at P[(pos0+k*period+off)%N]
#         for off in 0..L-1.

def _build_perimeter(H: int, W: int) -> List[Tuple[int, int]]:
    P = []
    for c in range(W):            P.append((0, c))
    for r in range(1, H):         P.append((r, W - 1))
    for c in range(W - 2, -1, -1): P.append((H - 1, c))
    for r in range(H - 2, 0, -1): P.append((r, 0))
    return P

def _find_start_pos(positions: List[int], N: int) -> int:
    """Return the start of a (possibly wrapping) contiguous run on a circular perimeter."""
    if not positions:
        raise ValueError("Empty positions")
    sorted_pos = sorted(positions)
    L = len(sorted_pos)
    # Find the largest gap between consecutive positions (mod N)
    max_gap = -1
    max_gap_idx = 0
    for i in range(L):
        gap = (sorted_pos[(i + 1) % L] - sorted_pos[i]) % N
        if gap > max_gap:
            max_gap = gap
            max_gap_idx = i
    # Start is the element right after the largest gap
    return sorted_pos[(max_gap_idx + 1) % L]

def _solve_perimeter(grid: Grid) -> Optional[Grid]:
    H, W = len(grid), len(grid[0])
    P = _build_perimeter(H, W)
    N = len(P)
    pos_map = {pos: i for i, pos in enumerate(P)}

    # Find shape cells (non-8 on perimeter)
    shape_cells = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 8 and (r, c) in pos_map:
                shape_cells.append((r, c))

    if not shape_cells:
        return None

    # All shape cells must be a single color
    colors = {grid[r][c] for r, c in shape_cells}
    if len(colors) != 1:
        return None
    color = colors.pop()

    L = len(shape_cells)
    period = 2 * L
    if N % period != 0:
        return None

    shape_positions = [pos_map[cell] for cell in shape_cells]
    try:
        pos0 = _find_start_pos(shape_positions, N)
    except ValueError:
        return None

    # Build result
    result = _copy_grid(grid)
    reps = N // period
    for k in range(reps):
        for offset in range(L):
            p = (pos0 + k * period + offset) % N
            r, c = P[p]
            result[r][c] = color
    return result


@register_solver
class PerimeterTilingSolver(BaseSolver):
    """Tiles the grid perimeter with a shape at period=2*shape_length (30f42897)."""

    TARGET_TASK = "30f42897"

    @property
    def name(self) -> str:
        return "PerimeterTilingSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate on training pairs
        for pair in ctx.train_pairs:
            pred = _solve_perimeter(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        # Predict on test inputs
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_perimeter(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=preds)
