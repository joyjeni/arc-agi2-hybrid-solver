"""
advanced_solvers32.py — Three targeted solvers for specific ARC patterns.

1. VerticalBarHeightSolver    (a61f2674 family)
   — Vertical bars (single non-zero color); tallest → 1, shortest → 2, rest erased.

2. ConnectedComponentSize6Solver (d2abd087 family)
   — CCs of non-zero color; size exactly 6 → color 2, all other CCs → color 1.

3. IsolatedZeroToEightSolver  (e0fb7511 family)
   — On 1-background grid: isolated 0s stay 0; connected 0-groups (size ≥ 2) → 8.
"""

from collections import deque
from typing import List, Optional

from src.core.interfaces import BaseSolver, Grid, Prediction, SolverResult, TaskContext
from src.solvers.registry import register_solver


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    return all(pred[r] == expected[r] for r in range(len(pred)))


def _connected_components_4(grid: Grid, match_fn) -> List[List[tuple]]:
    """4-connected CCs; match_fn(cell_value) → include in component."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components: List[List[tuple]] = []
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and match_fn(grid[r][c]):
                q = deque([(r, c)])
                visited[r][c] = True
                cc: List[tuple] = []
                while q:
                    rr, rc = q.popleft()
                    cc.append((rr, rc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, rc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and match_fn(grid[nr][nc]):
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append(cc)
    return components


# ─────────────────────────────────────────────────────────────────────────────
# Solver 1 — Vertical Bar Height Recoloring  (a61f2674 family)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_vbar(g: Grid) -> Optional[Grid]:
    rows, cols = len(g), len(g[0])

    # Find columns that contain any non-zero cell
    bar_info = {}  # col -> (height, color_set)
    for c in range(cols):
        cells = [(r, g[r][c]) for r in range(rows) if g[r][c] != 0]
        if not cells:
            continue
        colors = {v for _, v in cells}
        if len(colors) != 1:
            return None  # mixed colors in one column
        bar_info[c] = (len(cells), list(colors)[0])

    if len(bar_info) < 2:
        return None

    heights = {c: h for c, (h, _) in bar_info.items()}
    max_h = max(heights.values())
    min_h = min(heights.values())
    if max_h == min_h:
        return None  # all bars same height — ambiguous

    tall_cols = [c for c, h in heights.items() if h == max_h]
    short_cols = [c for c, h in heights.items() if h == min_h]
    if len(tall_cols) != 1 or len(short_cols) != 1:
        return None  # tie — skip

    tall_c, short_c = tall_cols[0], short_cols[0]

    out = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        if g[r][tall_c] != 0:
            out[r][tall_c] = 1
        if g[r][short_c] != 0:
            out[r][short_c] = 2
    return out


@register_solver
class VerticalBarHeightSolver(BaseSolver):
    """
    Vertical-bar height recoloring (a61f2674 family).
    Tallest bar column → color 1, shortest bar column → color 2;
    all other bar columns erased (→ 0).
    """

    @property
    def name(self) -> str:
        return "VerticalBarHeightSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_vbar(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_vbar(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 2 — Connected Component Size-6 Recoloring  (d2abd087 family)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_cc_size6(g: Grid) -> Optional[Grid]:
    rows, cols = len(g), len(g[0])

    # Determine foreground color (most frequent non-zero)
    freq: dict = {}
    for row in g:
        for v in row:
            if v != 0:
                freq[v] = freq.get(v, 0) + 1
    if not freq:
        return None
    fg = max(freq, key=lambda k: freq[k])

    ccs = _connected_components_4(g, lambda v: v == fg)
    if not ccs:
        return None

    # Pattern requires at least one CC of size exactly 6
    if not any(len(cc) == 6 for cc in ccs):
        return None

    out = [[0] * cols for _ in range(rows)]
    for cc in ccs:
        color = 2 if len(cc) == 6 else 1
        for r, c in cc:
            out[r][c] = color
    return out


@register_solver
class ConnectedComponentSize6Solver(BaseSolver):
    """
    CC size-6 recoloring (d2abd087 family).
    Non-zero CCs of size exactly 6 → color 2; all other non-zero CCs → color 1.
    """

    @property
    def name(self) -> str:
        return "ConnectedComponentSize6Solver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_cc_size6(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_cc_size6(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 3 — Isolated Zero to 8  (e0fb7511 family)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_isolated_zero(g: Grid) -> Optional[Grid]:
    rows, cols = len(g), len(g[0])

    # Input must be purely 0s and 1s (no 8s yet)
    for row in g:
        for v in row:
            if v not in (0, 1):
                return None

    ones = sum(v == 1 for row in g for v in row)
    zeros = sum(v == 0 for row in g for v in row)
    if ones == 0 or zeros == 0:
        return None
    if ones < zeros:
        return None  # 1 must be the dominant background

    ccs = _connected_components_4(g, lambda v: v == 0)

    # Need both isolated (size 1) and connected (size ≥ 2) zero CCs
    has_isolated = any(len(cc) == 1 for cc in ccs)
    has_connected = any(len(cc) >= 2 for cc in ccs)
    if not (has_isolated and has_connected):
        return None

    out = _copy_grid(g)
    for cc in ccs:
        if len(cc) >= 2:
            for r, c in cc:
                out[r][c] = 8
    return out


@register_solver
class IsolatedZeroToEightSolver(BaseSolver):
    """
    Isolated-zero-to-eight recoloring (e0fb7511 family).
    On a 1-background grid: isolated 0s (size-1 CC) stay 0;
    connected 0 groups (size ≥ 2) → color 8.
    """

    @property
    def name(self) -> str:
        return "IsolatedZeroToEightSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_isolated_zero(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_isolated_zero(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
