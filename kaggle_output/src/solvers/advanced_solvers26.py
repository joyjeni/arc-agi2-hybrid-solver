"""
advanced_solvers26.py — Four new solvers for ARC-AGI-2

1. QuadrantFillSolver        (task: 87ab05b8)
2. CornerTriangleCWSolver    (task: 880c1354)
3. RectangleFillByAreaSolver (task: 694f12f3)
4. RingToCrossReplaceSolver  (task: 6c434453)
"""

from collections import Counter, deque
from typing import List, Optional
import math

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from src.solvers.registry import register_solver


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    for r in range(len(pred)):
        if pred[r] != list(expected[r]):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. QuadrantFillSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class QuadrantFillSolver(BaseSolver):
    """
    4×4 grid.  bg = most common value in training inputs.
    fill_val = the single non-bg color consistent across all training outputs.
    Locate fill_val in input (appears exactly once) → determine quadrant (r//2, c//2)
    → fill that entire 2×2 quadrant with fill_val, rest with bg.
    """

    @property
    def name(self) -> str:
        return "QuadrantFillSolver"

    def _bg_and_fill(self, train_pairs) -> tuple:
        all_vals = [v for p in train_pairs for row in p["input"] for v in row]
        bg = Counter(all_vals).most_common(1)[0][0]
        out_colors = {v for p in train_pairs for row in p["output"] for v in row if v != bg}
        if len(out_colors) != 1:
            return None, None
        return bg, out_colors.pop()

    def _apply(self, inp: Grid, bg: int, fill_val: int) -> Optional[Grid]:
        pos = [(r, c) for r in range(4) for c in range(4) if inp[r][c] == fill_val]
        if len(pos) != 1:
            return None
        qr, qc = pos[0][0] // 2, pos[0][1] // 2
        return [[fill_val if (r // 2 == qr and c // 2 == qc) else bg
                 for c in range(4)] for r in range(4)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Must be 4×4 everywhere
        all_grids = ([p["input"] for p in ctx.train_pairs] +
                     [p["output"] for p in ctx.train_pairs] +
                     ctx.test_inputs)
        if any(len(g) != 4 or any(len(r) != 4 for r in g) for g in all_grids):
            return None

        bg, fill_val = self._bg_and_fill(ctx.train_pairs)
        if bg is None:
            return None

        # Validate on all training pairs
        for p in ctx.train_pairs:
            pred = self._apply(p["input"], bg, fill_val)
            if pred is None or not _verify_exact(pred, p["output"]):
                return None

        # Apply to test inputs
        predictions = []
        for test_input in ctx.test_inputs:
            out = self._apply(test_input, bg, fill_val)
            if out is None:
                return None
            predictions.append(Prediction(attempt_1=out, attempt_2=out,
                                           solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CornerTriangleCWSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class CornerTriangleCWSolver(BaseSolver):
    """
    8×8 grid.  Divider value=7, center value=4.
    Non-4, non-7 cells form triangular corner regions (monochromatic, 4-connected).
    CW rotation: sort regions by angle from grid center (ascending = CW order),
    then each region receives the color of the previous region in that order.
    """
    DIV, CTR = 7, 4

    @property
    def name(self) -> str:
        return "CornerTriangleCWSolver"

    def _regions(self, grid: Grid) -> List:
        H, W = len(grid), len(grid[0])
        vis: set = set()
        regs = []
        for r in range(H):
            for c in range(W):
                v = grid[r][c]
                if v == self.DIV or v == self.CTR or (r, c) in vis:
                    continue
                q = deque([(r, c)])
                cells = []
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in vis:
                        continue
                    vis.add((cr, cc))
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis:
                            nv = grid[nr][nc]
                            if nv != self.DIV and nv != self.CTR:
                                q.append((nr, nc))
                colors = {grid[cr][cc] for cr, cc in cells}
                if len(colors) == 1:
                    col = colors.pop()
                    cr_avg = sum(r for r, _ in cells) / len(cells)
                    cc_avg = sum(c for _, c in cells) / len(cells)
                    regs.append((col, cells, cr_avg, cc_avg))
        return regs

    def _rotate(self, grid: Grid) -> Optional[Grid]:
        H, W = len(grid), len(grid[0])
        cr, cc = (H - 1) / 2.0, (W - 1) / 2.0
        regs = self._regions(grid)
        if not regs:
            return None
        # Sort ascending by atan2(dr, dc) — gives CW order starting from TL
        regs.sort(key=lambda x: math.atan2(x[2] - cr, x[3] - cc))
        n = len(regs)
        out = [list(row) for row in grid]
        for i, (_, cells, _, _) in enumerate(regs):
            new_col = regs[(i - 1) % n][0]   # receive from CCW predecessor
            for r, c in cells:
                out[r][c] = new_col
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        all_grids = ([g for p in ctx.train_pairs for g in (p["input"], p["output"])] +
                     ctx.test_inputs)
        if any(len(g) != 8 or any(len(r) != 8 for r in g) for g in all_grids):
            return None

        inp0 = ctx.train_pairs[0]["input"]
        flat = {v for row in inp0 for v in row}
        if self.DIV not in flat or self.CTR not in flat:
            return None
        if len(self._regions(inp0)) < 2:
            return None

        for p in ctx.train_pairs:
            pred = self._rotate(p["input"])
            if pred is None or not _verify_exact(pred, p["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            out = self._rotate(test_input)
            if out is None:
                return None
            predictions.append(Prediction(attempt_1=out, attempt_2=out,
                                           solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. RectangleFillByAreaSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class RectangleFillByAreaSolver(BaseSolver):
    """
    Grid with bg=0, fg=4.  Exactly 2 solid rectangles of fg.
    Fill each rectangle's interior (shrink 1 on each side) with rank color:
    smaller area → 1, larger area → 2.
    """

    @property
    def name(self) -> str:
        return "RectangleFillByAreaSolver"

    def _find_rects(self, grid: Grid) -> List:
        H, W = len(grid), len(grid[0])
        vis: set = set()
        rects = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == 0 or (r, c) in vis:
                    continue
                q = deque([(r, c)])
                cells = []
                while q:
                    cr, cc = q.popleft()
                    if (cr, cc) in vis:
                        continue
                    vis.add((cr, cc))
                    cells.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis and grid[nr][nc] != 0:
                            q.append((nr, nc))
                r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
                c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
                rects.append((r0, r1, c0, c1))
        return rects

    def _apply(self, grid: Grid, rects: List) -> Grid:
        out = [list(row) for row in grid]
        areas = [(r1 - r0 + 1) * (c1 - c0 + 1) for r0, r1, c0, c1 in rects]
        for rank, (_, (r0, r1, c0, c1)) in enumerate(sorted(zip(areas, rects)), 1):
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    out[r][c] = rank
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for p in ctx.train_pairs:
            inp = p["input"]
            if not {v for row in inp for v in row}.issubset({0, 4}):
                return None
            rects = self._find_rects(inp)
            if len(rects) != 2:
                return None
            pred = self._apply(inp, rects)
            if not _verify_exact(pred, p["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            rects = self._find_rects(test_input)
            out = self._apply(test_input, rects)
            predictions.append(Prediction(attempt_1=out, attempt_2=out,
                                           solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RingToCrossReplaceSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class RingToCrossReplaceSolver(BaseSolver):
    """
    Grid with bg=0, fg=1.  3×3 hollow rings (8 border cells = fg=1, center = 0)
    → replaced with + crosses of value 2 (center + 4 cardinal neighbors).
    Ring corner cells become 0.  Non-ring fg shapes remain unchanged.
    """
    FG, CROSS = 1, 2

    @property
    def name(self) -> str:
        return "RingToCrossReplaceSolver"

    def _is_ring(self, grid: Grid, r: int, c: int) -> bool:
        H, W = len(grid), len(grid[0])
        if r < 1 or r >= H - 1 or c < 1 or c >= W - 1 or grid[r][c] != 0:
            return False
        return all(grid[r + dr][c + dc] == self.FG
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0))

    def _apply(self, grid: Grid) -> Grid:
        H, W = len(grid), len(grid[0])
        centers = [(r, c) for r in range(H) for c in range(W) if self._is_ring(grid, r, c)]
        out = [list(row) for row in grid]
        # Pass 1: erase all ring borders
        for r, c in centers:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not (dr == 0 and dc == 0):
                        out[r + dr][c + dc] = 0
        # Pass 2: draw crosses
        for r, c in centers:
            out[r][c] = self.CROSS
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                out[r + dr][c + dc] = self.CROSS
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for p in ctx.train_pairs:
            inp = p["input"]
            if not {v for row in inp for v in row}.issubset({0, 1}):
                return None
            if not any(self._is_ring(inp, r, c)
                       for r in range(len(inp)) for c in range(len(inp[0]))):
                return None
            pred = self._apply(inp)
            if not _verify_exact(pred, p["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            out = self._apply(test_input)
            predictions.append(Prediction(attempt_1=out, attempt_2=out,
                                           solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=2)
