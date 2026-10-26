"""
advanced_solvers29.py — Four solvers for ARC-AGI-2

1. CrossMoveAwayFromFiveSolver    (task: e48d4e1a) — full-row/col cross moves away from 5s
2. DiagonalRaysFromBlockCornersSolver (task: 5c0a986e) — diagonal rays from 2x2 block corners
3. FiveGravityToBlockSolver       (task: a48eeaf7) — 5s gravitate toward 2x2 block, stop at Chebyshev-1
4. SpineExtrusionSolver           (task: 9c1e755f) — partial 5-ruler extrudes adjacent colors
"""

from typing import List, Optional, Tuple

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from src.solvers.registry import register_solver


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    for r in range(len(pred)):
        if list(pred[r]) != list(expected[r]):
            return False
    return True


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CrossMoveAwayFromFiveSolver  (e48d4e1a)
# ─────────────────────────────────────────────────────────────────────────────
#
# Input: 10x10 grid with one cross (a full row + full column of same non-zero
#        non-5 color X) and N cells containing value 5.
# Algorithm:
#   - Cross center (cr, cc): the row r where all 10 cells = X, the column c
#     where all 10 cells = X.
#   - N = count of 5-cells; centroid (mean_r5, mean_c5) of all 5-positions.
#   - Direction: dr = sign(cr - mean_r5), dc = sign(cc - mean_c5).
#   - New center = (cr + N*dr, cc + N*dc).
#   - Output = blank 10x10 with new cross drawn (full new_row + full new_col).
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class CrossMoveAwayFromFiveSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "CrossMoveAwayFromFiveSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])

        # Find cross row: a row where ALL cells are the same non-zero non-5 color
        cross_row = None
        cross_color = None
        for r in range(rows):
            vals = set(inp[r][c] for c in range(cols))
            if len(vals) == 1 and 0 not in vals and 5 not in vals:
                cross_row = r
                cross_color = inp[r][0]
                break

        if cross_row is None or cross_color is None:
            return None

        # Find cross column: all cells = cross_color
        cross_col = None
        for c in range(cols):
            if all(inp[r][c] == cross_color for r in range(rows)):
                cross_col = c
                break

        if cross_col is None:
            return None

        # Collect 5-cells
        fives = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 5]
        N = len(fives)
        if N == 0:
            return None

        mean_r5 = sum(r for r, c in fives) / N
        mean_c5 = sum(c for r, c in fives) / N

        dr = _sign(cross_row - mean_r5)
        dc = _sign(cross_col - mean_c5)

        new_row = cross_row + N * dr
        new_col = cross_col + N * dc

        if not (0 <= new_row < rows and 0 <= new_col < cols):
            return None

        # Build output: blank grid, draw new cross
        out = [[0] * cols for _ in range(rows)]
        for c in range(cols):
            out[new_row][c] = cross_color
        for r in range(rows):
            out[r][new_col] = cross_color

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DiagonalRaysFromBlockCornersSolver  (5c0a986e)
# ─────────────────────────────────────────────────────────────────────────────
#
# Input: contains a 2x2 solid block of INTEGER color 1 and a 2x2 solid block
#        of INTEGER color 2.
# Rules (hardcoded by integer color value):
#   - Color 1 block top-left at (r1, c1):
#       ray starts at (r1-1, c1-1), direction (-1,-1), color 1.
#   - Color 2 block top-left at (r2, c2):
#       ray starts at (r2+2, c2+2), direction (+1,+1), color 2.
# Output = copy of input + ray cells filled with respective color.
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class DiagonalRaysFromBlockCornersSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "DiagonalRaysFromBlockCornersSolver"

    def _find_2x2(self, inp: Grid, color: int) -> Optional[Tuple[int, int]]:
        rows, cols = len(inp), len(inp[0])
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (inp[r][c] == color and inp[r][c + 1] == color and
                        inp[r + 1][c] == color and inp[r + 1][c + 1] == color):
                    return (r, c)
        return None

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])

        block1 = self._find_2x2(inp, 1)
        block2 = self._find_2x2(inp, 2)

        if block1 is None or block2 is None:
            return None

        out = [list(row) for row in inp]

        # Color 1: ray from (r1-1, c1-1) going (-1, -1)
        r1, c1 = block1
        r, c = r1 - 1, c1 - 1
        while 0 <= r < rows and 0 <= c < cols:
            out[r][c] = 1
            r -= 1
            c -= 1

        # Color 2: ray from (r2+2, c2+2) going (+1, +1)
        r2, c2 = block2
        r, c = r2 + 2, c2 + 2
        while 0 <= r < rows and 0 <= c < cols:
            out[r][c] = 2
            r += 1
            c += 1

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FiveGravityToBlockSolver  (a48eeaf7)
# ─────────────────────────────────────────────────────────────────────────────
#
# Input: a 2x2 solid-color block (non-zero, non-5) and scattered 5-cells.
# For each 5-cell at (r5, c5):
#   - Find nearest block cell by Euclidean distance → target (tr, tc).
#   - Direction: dr=sign(tr-r5), dc=sign(tc-c5).
#   - Advance step-by-step until Chebyshev distance to ANY block cell = 1.
#   - Clear original 5-position, place 5 at final position.
# ─────────────────────────────────────────────────────────────────────────────

def _chebyshev(r1: int, c1: int, r2: int, c2: int) -> int:
    return max(abs(r1 - r2), abs(c1 - c2))


@register_solver
class FiveGravityToBlockSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "FiveGravityToBlockSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])

        # Find 2x2 solid-color block (non-zero, non-5)
        block_cells: List[Tuple[int, int]] = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                v = inp[r][c]
                if v != 0 and v != 5:
                    if (inp[r][c + 1] == v and inp[r + 1][c] == v and
                            inp[r + 1][c + 1] == v):
                        block_cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
                        break
            if block_cells:
                break

        if not block_cells:
            return None

        fives = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 5]
        if not fives:
            return None

        out = [list(row) for row in inp]

        for r5, c5 in fives:
            # Already adjacent to block?
            init_min_cheb = min(_chebyshev(r5, c5, br, bc) for br, bc in block_cells)
            if init_min_cheb <= 1:
                continue

            # Find nearest block cell by Euclidean distance
            best_dist_sq = None
            tr, tc = block_cells[0]
            for br, bc in block_cells:
                dsq = (r5 - br) ** 2 + (c5 - bc) ** 2
                if best_dist_sq is None or dsq < best_dist_sq:
                    best_dist_sq = dsq
                    tr, tc = br, bc

            dr = _sign(tr - r5)
            dc = _sign(tc - c5)

            if dr == 0 and dc == 0:
                continue

            # Walk step-by-step until Chebyshev distance to any block cell = 1
            r_cur, c_cur = r5, c5
            while True:
                r_next = r_cur + dr
                c_next = c_cur + dc
                if not (0 <= r_next < rows and 0 <= c_next < cols):
                    break
                min_cheb = min(_chebyshev(r_next, c_next, br, bc)
                               for br, bc in block_cells)
                if min_cheb == 0:
                    break  # would land on block cell, stop before
                r_cur, c_cur = r_next, c_next
                if min_cheb == 1:
                    break  # reached adjacency, stop here

            # Move 5
            out[r5][c5] = 0
            out[r_cur][c_cur] = 5

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SpineExtrusionSolver  (9c1e755f)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rulers are PARTIAL contiguous runs of 5s within a row (horizontal) or
# column (vertical) — NOT necessarily full rows/columns.
#
# Horizontal ruler (row r, cols c1..c2):
#   Valid if some col in [c1..c2] has non-zero non-5 at another row.
#   tc = first col in [c1..c2] that has any non-zero non-5 at rows ≠ r.
#   For each row rr ≠ r where inp[rr][tc] != 0 and != 5:
#     fill out[rr][c1..c2] entirely with inp[rr][tc].
#   Ruler cells (row r, cols c1..c2) stay as 5.
#
# Vertical ruler (col cc, rows r1..r2):
#   Valid if some row in [r1..r2] has non-zero non-5 at cc-1 or cc+1.
#   template_rows = rows rr in [r1..r2] where adj cell (cc-1 or cc+1) is
#                   non-zero non-5.
#   fill_cols = cols c_t (c_t ≠ cc) where any template_row has non-zero non-5.
#   For template rows: out[rr][c_t] = inp[rr][c_t] for c_t in fill_cols.
#   Non-template rows in [r1..r2] (ordered), index i:
#     r_t = template_rows[i % len(template_rows)]
#     out[rr][c_t] = inp[r_t][c_t] for c_t in fill_cols.
#   Ruler cells (col cc, rows r1..r2) stay as 5.
#
# Handles multiple rulers per grid; output starts blank.
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class SpineExtrusionSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "SpineExtrusionSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])
        out = [[0] * cols for _ in range(rows)]
        applied_any = False

        # ── Horizontal rulers ──────────────────────────────────────────────
        for r in range(rows):
            c = 0
            while c < cols:
                if inp[r][c] == 5:
                    c1 = c
                    while c < cols and inp[r][c] == 5:
                        c += 1
                    c2 = c - 1

                    # Find first template column in [c1..c2]
                    tc = None
                    for cc in range(c1, c2 + 1):
                        for rr in range(rows):
                            if rr != r and inp[rr][cc] != 0 and inp[rr][cc] != 5:
                                tc = cc
                                break
                        if tc is not None:
                            break

                    if tc is None:
                        continue  # no data; skip this run

                    # Place ruler
                    for cc in range(c1, c2 + 1):
                        out[r][cc] = 5

                    # Fill data rows
                    for rr in range(rows):
                        if rr != r and inp[rr][tc] != 0 and inp[rr][tc] != 5:
                            color = inp[rr][tc]
                            for cc in range(c1, c2 + 1):
                                out[rr][cc] = color

                    applied_any = True
                else:
                    c += 1

        # ── Vertical rulers ────────────────────────────────────────────────
        for cc in range(cols):
            r = 0
            while r < rows:
                if inp[r][cc] == 5:
                    r1 = r
                    while r < rows and inp[r][cc] == 5:
                        r += 1
                    r2 = r - 1

                    # Check validity: some row in [r1..r2] has adj non-zero non-5
                    def _adj_nz(rr: int) -> bool:
                        for adj_c in (cc - 1, cc + 1):
                            if 0 <= adj_c < cols and inp[rr][adj_c] != 0 and inp[rr][adj_c] != 5:
                                return True
                        return False

                    if not any(_adj_nz(rr) for rr in range(r1, r2 + 1)):
                        continue  # no adjacent data; skip this run

                    # Template rows: rows in [r1..r2] with adjacent non-zero non-5
                    template_rows = [rr for rr in range(r1, r2 + 1) if _adj_nz(rr)]

                    # fill_cols: columns (≠ cc) where any template row has non-zero non-5
                    fill_cols = [
                        fc for fc in range(cols)
                        if fc != cc and any(
                            inp[rt][fc] != 0 and inp[rt][fc] != 5
                            for rt in template_rows
                        )
                    ]

                    if not fill_cols:
                        continue

                    # Place ruler
                    for rr in range(r1, r2 + 1):
                        out[rr][cc] = 5

                    # Place template rows as-is
                    for rt in template_rows:
                        for fc in fill_cols:
                            out[rt][fc] = inp[rt][fc]

                    # Fill non-template rows cyclically
                    non_template = [rr for rr in range(r1, r2 + 1)
                                    if rr not in template_rows]
                    for i, rr in enumerate(non_template):
                        rt = template_rows[i % len(template_rows)]
                        for fc in fill_cols:
                            out[rr][fc] = inp[rt][fc]

                    applied_any = True
                else:
                    r += 1

        if not applied_any:
            return None

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)
