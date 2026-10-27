"""
advanced_solvers31.py — Two solvers from Robina Mirbahar reference notebook

1. RayGuidedFaceRecolorSolver  (task 3d588dc9)
   — Locate gray-5 directional wedge; find largest black-0 connected component
     that horizontally overlaps with the wedge's row range; prune 1-cell-wide
     spikes (non-core cells → bg 7); recolor the face column facing the wedge
     to magenta 6.

2. PerimeterBFSCavitySolver    (task 00d62c1b)
   — BFS from every border 0-cell outward; any interior 0-cell never reached
     is a cavity → recolor it to yellow 4.
"""

from typing import List, Optional, Tuple
from collections import deque, Counter

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from src.solvers.registry import register_solver


def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    for r in range(len(pred)):
        if len(pred[r]) != len(expected[r]):
            return False
        for c in range(len(pred[r])):
            if pred[r][c] != expected[r][c]:
                return False
    return True


# ---------------------------------------------------------------------------
# Solver 1 — Ray-Guided Face Recoloring & Spike Pruning  (task 3d588dc9)
# ---------------------------------------------------------------------------

def _solve_3d588dc9(g: Grid) -> Optional[Grid]:
    h = len(g)
    if h == 0:
        return None
    w = len(g[0])

    # Background = most-common color (expect 7=orange)
    flat = [v for row in g for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find gray (5) wedge cells
    gray_cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] == 5]
    if not gray_cells:
        return None
    g_rmin = min(r for r, _ in gray_cells)
    g_rmax = max(r for r, _ in gray_cells)
    g_cmean = sum(c for _, c in gray_cells) / len(gray_cells)

    # BFS: find all black (0) connected components ≥6 cells
    vis = [[False] * w for _ in range(h)]
    best_comp: Optional[List[Tuple[int, int]]] = None
    best_overlap = -1

    for sr in range(h):
        for sc in range(w):
            if g[sr][sc] == 0 and not vis[sr][sc]:
                q: deque = deque([(sr, sc)])
                vis[sr][sc] = True
                cells: List[Tuple[int, int]] = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not vis[nr][nc] and g[nr][nc] == 0:
                            vis[nr][nc] = True
                            q.append((nr, nc))
                if len(cells) < 6:
                    continue
                rs = [r for r, _ in cells]
                overlap = max(0, min(max(rs), g_rmax) - max(min(rs), g_rmin) + 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_comp = cells

    if best_comp is None:
        return None

    out = _copy_grid(g)

    # Count how many component cells per row/col
    row_cnt: Counter = Counter(r for r, _ in best_comp)
    col_cnt: Counter = Counter(c for _, c in best_comp)

    # Core = rows with ≥3 cells AND cols with ≥3 cells
    core_rs = {r for r, n in row_cnt.items() if n >= 3}
    core_cs = {c for c, n in col_cnt.items() if n >= 3}

    # Prune spike cells (not fully in core)
    for r, c in best_comp:
        if r not in core_rs or c not in core_cs:
            out[r][c] = bg

    if not core_rs or not core_cs:
        return out

    # Face column: whichever side of core faces the wedge
    core_cs_sorted = sorted(core_cs)
    core_cs_mean = sum(core_cs_sorted) / len(core_cs_sorted)
    target_c = core_cs_sorted[0] if g_cmean < core_cs_mean else core_cs_sorted[-1]

    for r in sorted(core_rs):
        out[r][target_c] = 6

    return out


@register_solver
class RayGuidedFaceRecolorSolver(BaseSolver):
    """
    task 3d588dc9 — Gray-5 wedge points at black-0 object on orange-7 field.
    Prune 1-cell-wide spikes from the object, then recolor the face column
    nearest the wedge to magenta-6.
    """

    @property
    def name(self) -> str:
        return "RayGuidedFaceRecolorSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Verify on all train pairs
        for pair in ctx.train_pairs:
            pred = _solve_3d588dc9(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        # Solve test inputs
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_3d588dc9(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ---------------------------------------------------------------------------
# Solver 2 — Perimeter BFS Cavity Enclosure  (task 00d62c1b)
# ---------------------------------------------------------------------------

def _solve_00d62c1b(g: Grid) -> Optional[Grid]:
    h = len(g)
    if h == 0:
        return None
    w = len(g[0])

    vis = [[False] * w for _ in range(h)]
    q: deque = deque()

    # Seed BFS from all border 0-cells
    for r in range(h):
        for c in [0, w - 1]:
            if g[r][c] == 0 and not vis[r][c]:
                vis[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if g[r][c] == 0 and not vis[r][c]:
                vis[r][c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not vis[nr][nc] and g[nr][nc] == 0:
                vis[nr][nc] = True
                q.append((nr, nc))

    # Interior 0-cells not reached → cavity → color 4
    out = _copy_grid(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 and not vis[r][c]:
                out[r][c] = 4
    return out


@register_solver
class PerimeterBFSCavitySolver(BaseSolver):
    """
    task 00d62c1b — Flood-fill exterior background (0) from the grid perimeter
    via 4-connected BFS; any 0-cell not reached is an enclosed interior cavity
    → recolor to yellow 4.
    """

    @property
    def name(self) -> str:
        return "PerimeterBFSCavitySolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Verify on all train pairs
        for pair in ctx.train_pairs:
            pred = _solve_00d62c1b(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        # Solve test inputs
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_00d62c1b(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
