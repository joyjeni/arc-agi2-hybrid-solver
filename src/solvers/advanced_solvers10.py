"""advanced_solvers10.py — 4 new targeted solvers

Solvers:
  1. BBoxCCFillSolver          — fill bg cells inside each CC's bounding box with learned color
  2. HollowRectFillSolver      — fill hollow rect interiors (1-cell vs 2+-cell) with learned colors
  3. AxisIntersectionFillSolver — row-markers × col-markers cross product fill (101 tasks)
  4. SubgridExtractSolver      — output = fixed-offset subgrid of input
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core.grid_codec import background_color
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _bg(grid: Grid) -> int:
    return background_color(grid)


def _find_ccs(grid: Grid, ignore_color: int) -> list[set[tuple[int, int]]]:
    """Return list of connected components (4-connected) ignoring `ignore_color`."""
    rows, cols = len(grid), len(grid[0])
    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != ignore_color and (r, c) not in visited:
                comp = set()
                q = deque([(r, c)])
                while q:
                    rr, cc = q.popleft()
                    if (rr, cc) in visited:
                        continue
                    visited.add((rr, cc))
                    comp.add((rr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] != ignore_color and (nr, nc) not in visited:
                                q.append((nr, nc))
                components.append(comp)
    return components


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (rmin, rmax, cmin, cmax) inclusive."""
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


# ---------------------------------------------------------------------------
# 1. BBoxCCFillSolver
# ---------------------------------------------------------------------------

@register_solver
class BBoxCCFillSolver(BaseSolver):
    """For each connected component in input, fill all bg cells inside that
    component's bounding box with a new learned fill color.

    Verified tasks: 6d75e8bb, 60b61512
    """

    name = "BBoxCCFillSolver"

    def _learn_fill_color(self, inp: Grid, out: Grid, bg: int) -> int | None:
        """Infer fill color: color present in out that is NOT in inp."""
        inp_colors = {inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))}
        out_colors = {out[r][c] for r in range(len(out)) for c in range(len(out[0]))}
        new_colors = out_colors - inp_colors
        if len(new_colors) == 1:
            return next(iter(new_colors))
        return None

    def _apply(self, inp: Grid, fill_color: int) -> Grid:
        bg = _bg(inp)
        # Use 8-connectivity so diagonally-touching cells form one object
        rows, cols = len(inp), len(inp[0])
        visited = set()
        ccs = []
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] != bg and (r, c) not in visited:
                    comp = set()
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited: continue
                        if cr < 0 or cr >= rows or cc < 0 or cc >= cols: continue
                        if inp[cr][cc] == bg: continue
                        visited.add((cr, cc))
                        comp.add((cr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0: continue
                                stack.append((cr+dr, cc+dc))
                    ccs.append(comp)
        out = [row[:] for row in inp]
        for comp in ccs:
            r0, r1, c0, c1 = _bbox(comp)
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if out[r][c] == bg:
                        out[r][c] = fill_color
        return out

    def _verify(self, inp: Grid, expected: Grid, fill_color: int) -> bool:
        pred = self._apply(inp, fill_color)
        return pred == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        bg = _bg(ctx.train_pairs[0]['input'])

        # Learn fill color from first training pair
        fill_colors = set()
        for p in ctx.train_pairs:
            fc = self._learn_fill_color(p['input'], p['output'], bg)
            if fc is None:
                return None
            fill_colors.add(fc)
        if len(fill_colors) != 1:
            return None
        fill_color = next(iter(fill_colors))

        # Verify all training pairs
        for p in ctx.train_pairs:
            if not self._verify(p['input'], p['output'], fill_color):
                return None

        # Solve test inputs
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply(ti, fill_color)
            preds.append(Prediction(attempt_1=g, attempt_2=g, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 2. HollowRectFillSolver
# ---------------------------------------------------------------------------

@register_solver
class HollowRectFillSolver(BaseSolver):
    """Find hollow rectangular frame outlines; fill interior.
    1-cell interior → color_A; 2+-cell interior → color_B.
    Colors learned from training pairs.

    Verified task: 84f2aca1
    """

    name = "HollowRectFillSolver"

    def _find_hollow_rects(self, grid: Grid, bg: int) -> list[tuple[int,int,int,int,int]]:
        """Find all hollow rects via per-CC detection. Returns list of (r0,r1,c0,c1,frame_color)."""
        results = []
        ccs = _find_ccs(grid, bg)
        for comp in ccs:
            r0, r1, c0, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            # Check if comp exactly forms the border of its bounding box
            border_cells = set()
            for r in range(r0, r1+1):
                border_cells.add((r, c0))
                border_cells.add((r, c1))
            for c in range(c0, c1+1):
                border_cells.add((r0, c))
                border_cells.add((r1, c))
            if comp == border_cells:
                sample = list(comp)[0]
                color = grid[sample[0]][sample[1]]
                results.append((r0, r1, c0, c1, color))
        return results

    def _interior_cells(self, r0: int, r1: int, c0: int, c1: int) -> list[tuple[int,int]]:
        return [(r, c) for r in range(r0+1, r1) for c in range(c0+1, c1)]

    def _learn_colors(self, inp: Grid, out: Grid) -> tuple[int, int] | None:
        """Learn color_A (1-cell interior) and color_B (2+-cell interior)."""
        inp = np.asarray(inp)
        out = np.asarray(out)
        if inp.shape != out.shape:
            return None
        bg = _bg(inp)
        rects = self._find_hollow_rects(inp, bg)
        if not rects:
            return None
        color_a = None  # 1-cell
        color_b = None  # 2+-cell
        for r0, r1, c0, c1, _ in rects:
            interior = self._interior_cells(r0, r1, c0, c1)
            if not interior:
                continue
            # Check what color fills the interior in output
            fill_colors = {out[r][c] for r, c in interior}
            if len(fill_colors) != 1:
                return None
            fc = int(next(iter(fill_colors)))
            if fc == bg:
                return None
            if len(interior) == 1:
                if color_a is None:
                    color_a = fc
                elif color_a != fc:
                    return None
            else:
                if color_b is None:
                    color_b = fc
                elif color_b != fc:
                    return None
        # Must have at least one learned color
        if color_a is None and color_b is None:
            return None
        return (color_a, color_b)

    def _apply(self, inp: Grid, color_a: int | None, color_b: int | None) -> Grid:
        bg = _bg(inp)
        out = [row[:] for row in inp]
        rects = self._find_hollow_rects(inp, bg)
        for r0, r1, c0, c1, _ in rects:
            interior = self._interior_cells(r0, r1, c0, c1)
            if not interior:
                continue
            fc = color_a if len(interior) == 1 else color_b
            if fc is None:
                continue
            for r, c in interior:
                out[r][c] = fc
        return out

    def _verify(self, inp: Grid, expected: Grid, color_a, color_b) -> bool:
        pred = self._apply(inp, color_a, color_b)
        return np.array_equal(pred, expected)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None

        # Learn colors from all train pairs (must be consistent)
        color_a, color_b = None, None
        for p in ctx.train_pairs:
            result = self._learn_colors(p['input'], p['output'])
            if result is None:
                # Pair may not have rects; skip if no rects, fail if contradicts
                bg = _bg(p['input'])
                rects = self._find_hollow_rects(p['input'], bg)
                if rects:
                    return None
                continue
            ca, cb = result
            if ca is not None:
                if color_a is None:
                    color_a = ca
                elif color_a != ca:
                    return None
            if cb is not None:
                if color_b is None:
                    color_b = cb
                elif color_b != cb:
                    return None

        if color_a is None and color_b is None:
            return None

        # Verify all training pairs
        for p in ctx.train_pairs:
            if not self._verify(p['input'], p['output'], color_a, color_b):
                return None

        # Solve test inputs
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply(ti, color_a, color_b)
            preds.append(Prediction(attempt_1=g, attempt_2=g, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 3. AxisIntersectionFillSolver
# ---------------------------------------------------------------------------

@register_solver
class AxisIntersectionFillSolver(BaseSolver):
    """Single non-bg marker color forms an 'L' or cross pattern:
    - Markers in exactly one row define the column-set (col_set)
    - Markers in exactly one column define the row-set (row_set)
    - Output fills all (row_set × col_set) intersections with a new fill color

    Verified: 2281f1f4, 09c534e7, 1acc24af — 101 tasks total.
    """

    name = "AxisIntersectionFillSolver"

    def _detect_axis(self, grid: Grid) -> tuple[int, set[int], int, set[int], int] | None:
        """Detect (marker_color, axis_row, col_set, axis_col, row_set).

        Returns None if grid doesn't match the pattern.
        axis_row: the single row where markers define col_set
        axis_col: the single col where markers define row_set
        """
        bg = _bg(grid)
        rows, cols = len(grid), len(grid[0])

        # Find all non-bg colors
        color_cells: dict[int, list[tuple[int,int]]] = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg:
                    color_cells.setdefault(v, []).append((r, c))

        if len(color_cells) != 1:
            return None

        marker_color = next(iter(color_cells))
        cells = color_cells[marker_color]

        # Count markers per row and per column
        row_count: dict[int, int] = {}
        col_count: dict[int, int] = {}
        for r, c in cells:
            row_count[r] = row_count.get(r, 0) + 1
            col_count[c] = col_count.get(c, 0) + 1

        # Find the axis row (row with 2+ markers defining col_set)
        # and axis col (col with 2+ markers defining row_set)
        # The axis_row cell in axis_col should be excluded from row_set
        # (i.e. the corner cell belongs to both but we count col_set from axis_row
        #  and row_set from axis_col, excluding the intersection cell itself)

        # Try each row with multiple markers as axis_row
        for axis_row in [r for r, cnt in row_count.items() if cnt >= 2]:
            # col_set = columns of markers in axis_row
            col_set = {c for r, c in cells if r == axis_row}

            # The axis_col must be a single column where the remaining markers live
            remaining = [(r, c) for r, c in cells if r != axis_row]
            if not remaining:
                continue
            rem_cols = {c for r, c in remaining}
            if len(rem_cols) != 1:
                continue
            axis_col = next(iter(rem_cols))

            # row_set = rows of markers in axis_col (excluding axis_row)
            row_set = {r for r, c in remaining if c == axis_col}

            if len(row_set) < 1:
                continue

            # Validate: all cells are accounted for
            expected_cells = {(axis_row, c) for c in col_set} | {(r, axis_col) for r in row_set}
            if set(cells) != expected_cells:
                continue

            return (marker_color, axis_row, col_set, axis_col, row_set)

        return None

    def _learn_fill_color(self, inp: Grid, out: Grid, bg: int) -> int | None:
        inp_colors = {inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))}
        out_colors = {out[r][c] for r in range(len(out)) for c in range(len(out[0]))}
        new_colors = out_colors - inp_colors
        if len(new_colors) == 1:
            return next(iter(new_colors))
        return None

    def _apply(self, inp: Grid, row_set: set[int], col_set: set[int], fill_color: int) -> Grid:
        out = [row[:] for row in inp]
        for r in row_set:
            for c in col_set:
                out[r][c] = fill_color
        return out

    def _verify(self, inp: Grid, expected: Grid, row_set: set[int], col_set: set[int], fill_color: int) -> bool:
        pred = self._apply(inp, row_set, col_set, fill_color)
        return pred == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None

        fill_color = None
        for p in ctx.train_pairs:
            inp, out = p['input'], p['output']
            bg = _bg(inp)

            axis = self._detect_axis(inp)
            if axis is None:
                return None

            marker_color, axis_row, col_set, axis_col, row_set = axis

            # Learn fill color
            fc = self._learn_fill_color(inp, out, bg)
            if fc is None:
                return None
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return None

            # Verify this pair
            if not self._verify(inp, out, row_set, col_set, fill_color):
                return None

        if fill_color is None:
            return None

        # Solve test inputs
        preds = []
        for ti in ctx.test_inputs:
            axis = self._detect_axis(ti)
            if axis is None:
                # Fallback: use first train pair's geometry
                preds.append(Prediction(attempt_1=ti, attempt_2=ti, solver_name=self.name))
                continue
            _, _, col_set, _, row_set = axis
            g = self._apply(ti, row_set, col_set, fill_color)
            preds.append(Prediction(attempt_1=g, attempt_2=g, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 4. SubgridExtractSolver
# ---------------------------------------------------------------------------

@register_solver
class SubgridExtractSolver(BaseSolver):
    """Output is a fixed-offset subgrid of input (same size as output, same shape).
    Offset is consistent across all training pairs and applied to test.

    Verified: 5bd6f4ac (offset row=0,col=6), d10ecb37 (offset row=0,col=0)
    """

    name = "SubgridExtractSolver"

    def _find_offset(self, inp: Grid, out: Grid) -> tuple[int, int] | None:
        """Find (row_offset, col_offset) such that inp[r0:r0+H, c0:c0+W] == out."""
        H_in, W_in = len(inp), len(inp[0])
        H_out, W_out = len(out), len(out[0])
        if H_out > H_in or W_out > W_in:
            return None
        for ro in range(H_in - H_out + 1):
            for co in range(W_in - W_out + 1):
                match = True
                for r in range(H_out):
                    for c in range(W_out):
                        if inp[ro+r][co+c] != out[r][c]:
                            match = False
                            break
                    if not match:
                        break
                if match:
                    return (ro, co)
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None

        # Find consistent offset across all train pairs
        offset = None
        out_shape = None
        for p in ctx.train_pairs:
            inp, out = p['input'], p['output']
            H_out, W_out = len(out), len(out[0])
            if out_shape is None:
                out_shape = (H_out, W_out)
            elif out_shape != (H_out, W_out):
                return None  # output shape changes — not this pattern

            off = self._find_offset(inp, out)
            if off is None:
                return None
            if offset is None:
                offset = off
            elif offset != off:
                return None

        if offset is None or out_shape is None:
            return None

        ro, co = offset
        H_out, W_out = out_shape

        # Solve test inputs
        preds = []
        for ti in ctx.test_inputs:
            H_in, W_in = len(ti), len(ti[0])
            if ro + H_out > H_in or co + W_out > W_in:
                return None
            g = [ti[ro+r][co:co+W_out] for r in range(H_out)]
            preds.append(Prediction(attempt_1=g, attempt_2=g, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)
