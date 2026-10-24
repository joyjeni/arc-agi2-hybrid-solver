"""advanced_solvers6.py — BorderGravitySortSolver and friends.

Pattern: a grid has a solid top-border row (color A) and a solid bottom-border
row (color B), background color BG.  Inner pixels of color A are 8-connected
components that get rigidly shifted so their topmost row touches row 1.  Inner
pixels of color B are 8-connected components that get rigidly shifted so their
bottommost row touches row H-2.
"""
from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Optional

from ..core.interfaces import BaseSolver, SolverResult, TaskContext, Prediction
from ..core.grid_codec import background_color
from .registry import register_solver

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _np(g): return np.array(g, dtype=np.int32)
def _to_grid(a): return a.tolist()
def _clone(a): return a.copy()
def _grids_equal(a, b): return np.array_equal(np.array(a), np.array(b))
def _same_shape(a, b): return np.array(a).shape == np.array(b).shape
def _bg(grid): return background_color(_to_grid(grid) if isinstance(grid, np.ndarray) else grid)

def _result(ctx, grids, name):
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _find_components_8conn(grid: np.ndarray, color: int) -> list[list[tuple[int,int]]]:
    """Return list of 8-connected components of `color` in grid."""
    H, W = grid.shape
    visited = np.zeros((H, W), bool)
    components = []
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for r in range(H):
        for c in range(W):
            if grid[r, c] == color and not visited[r, c]:
                comp: list[tuple[int,int]] = []
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    cr, cc = queue.pop(0)
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr,nc] and grid[nr,nc] == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                components.append(comp)
    return components


def _apply_border_gravity(inp: np.ndarray, bg: int, A: int, B: int) -> Optional[np.ndarray]:
    """Apply border-gravity rule and return output grid, or None if invalid."""
    H, W = inp.shape
    # validate borders
    if not all(inp[0, c] == A for c in range(W)):
        return None
    if not all(inp[H-1, c] == B for c in range(W)):
        return None

    out = np.full((H, W), bg, dtype=np.int32)
    out[0, :] = A
    out[H-1, :] = B

    # --- A components (shift top → row 1) ---
    comps_A = _find_components_8conn(inp, A)
    for comp in comps_A:
        inner = [(r, c) for r, c in comp if r != 0 and r != H-1]
        if not inner:
            continue
        top_row = min(r for r, c in inner)
        shift = 1 - top_row
        for r, c in inner:
            nr = r + shift
            if 1 <= nr <= H-2:
                out[nr, c] = A

    # --- B components (shift bottom → row H-2) ---
    comps_B = _find_components_8conn(inp, B)
    for comp in comps_B:
        inner = [(r, c) for r, c in comp if r != 0 and r != H-1]
        if not inner:
            continue
        bot_row = max(r for r, c in inner)
        shift = (H - 2) - bot_row
        for r, c in inner:
            nr = r + shift
            if 1 <= nr <= H-2:
                out[nr, c] = B

    return out


def _detect_border_colors(inp: np.ndarray, bg: int):
    """Return (A, B) top/bottom border colors, or (None, None) on failure."""
    H, W = inp.shape
    top_row = [inp[0, c] for c in range(W) if inp[0, c] != bg]
    bot_row = [inp[H-1, c] for c in range(W) if inp[H-1, c] != bg]
    if not top_row or not bot_row:
        return None, None
    A = Counter(top_row).most_common(1)[0][0]
    B = Counter(bot_row).most_common(1)[0][0]
    if A == B or A == bg or B == bg:
        return None, None
    return A, B


# ---------------------------------------------------------------------------
# BorderGravitySortSolver
# ---------------------------------------------------------------------------

@register_solver
class BorderGravitySortSolver(BaseSolver):
    """Detect top/bottom border colors; rigidly shift inner 8-connected
    components of each color so they touch the border row (row 1 for top
    color, row H-2 for bottom color).  Verified against all train pairs."""

    @property
    def name(self) -> str:
        return "BorderGravitySortSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # --- verify all train pairs ---
        for pair in ctx.train_pairs:
            inp = _np(pair['input'])
            out_expected = _np(pair['output'])
            if not _same_shape(inp, out_expected):
                return None
            bg = _bg(pair['input'])
            A, B = _detect_border_colors(inp, bg)
            if A is None:
                return None
            predicted = _apply_border_gravity(inp, bg, A, B)
            if predicted is None:
                return None
            if not _grids_equal(predicted, out_expected):
                return None

        # --- apply to test inputs ---
        results = []
        for test_inp in ctx.test_inputs:
            inp = _np(test_inp)
            bg = _bg(test_inp)
            A, B = _detect_border_colors(inp, bg)
            if A is None:
                return None
            predicted = _apply_border_gravity(inp, bg, A, B)
            if predicted is None:
                return None
            results.append(_to_grid(predicted))

        if not results:
            return None
        return _result(ctx, results, self.name)


# ---------------------------------------------------------------------------
# ColumnGravitySortSolver  (per-column gravity variant)
# ---------------------------------------------------------------------------
# Some tasks have pixels that fall column-by-column (like Tetris gravity)
# toward the border row, packing tightly.  Top-border color falls down to
# row 1 and stacks upward; bottom-border color falls down to row H-2 and
# stacks downward.

@register_solver
class ColumnGravitySortSolver(BaseSolver):
    """Per-column gravity: top-border-color pixels in each column stack up
    from row 1 downward; bottom-border-color pixels stack down from row H-2
    upward.  Verified against all train pairs."""

    @property
    def name(self) -> str:
        return "ColumnGravitySortSolver"

    def _apply(self, inp: np.ndarray, bg: int, A: int, B: int) -> Optional[np.ndarray]:
        H, W = inp.shape
        if not all(inp[0, c] == A for c in range(W)):
            return None
        if not all(inp[H-1, c] == B for c in range(W)):
            return None

        out = np.full((H, W), bg, dtype=np.int32)
        out[0, :] = A
        out[H-1, :] = B

        for c in range(W):
            inner = inp[1:H-1, c]
            a_pixels = int(np.sum(inner == A))
            b_pixels = int(np.sum(inner == B))
            # A pixels fall toward row 1 (stack from row 1 downward)
            for k in range(a_pixels):
                out[1 + k, c] = A
            # B pixels fall toward row H-2 (stack from H-2 upward)
            for k in range(b_pixels):
                out[H-2 - k, c] = B

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            inp = _np(pair['input'])
            out_e = _np(pair['output'])
            if not _same_shape(inp, out_e):
                return None
            bg = _bg(pair['input'])
            A, B = _detect_border_colors(inp, bg)
            if A is None:
                return None
            pred = self._apply(inp, bg, A, B)
            if pred is None or not _grids_equal(pred, out_e):
                return None

        results = []
        for test_inp in ctx.test_inputs:
            inp = _np(test_inp)
            bg = _bg(test_inp)
            A, B = _detect_border_colors(inp, bg)
            if A is None:
                return None
            pred = self._apply(inp, bg, A, B)
            if pred is None:
                return None
            results.append(_to_grid(pred))

        if not results:
            return None
        return _result(ctx, results, self.name)


# ---------------------------------------------------------------------------
# SingleBorderGravitySolver  (only one border color, single direction)
# ---------------------------------------------------------------------------
# Tasks with only a top OR bottom border row; objects float toward that border.

@register_solver
class SingleBorderGravitySolver(BaseSolver):
    """One border row (top or bottom); inner 8-connected components of the
    border color shift rigidly so they touch row 1 (top) or row H-2 (bottom).
    Verified against all train pairs."""

    @property
    def name(self) -> str:
        return "SingleBorderGravitySolver"

    def _apply(self, inp: np.ndarray) -> Optional[np.ndarray]:
        H, W = inp.shape
        bg = background_color(inp.tolist())
        # Detect top border
        top_row_vals = [inp[0, c] for c in range(W)]
        bot_row_vals = [inp[H-1, c] for c in range(W)]
        top_uniform = len(set(top_row_vals)) == 1 and top_row_vals[0] != bg
        bot_uniform = len(set(bot_row_vals)) == 1 and bot_row_vals[0] != bg

        # Exactly one border present
        if top_uniform == bot_uniform:
            return None  # either both or neither

        out = np.full((H, W), bg, dtype=np.int32)

        if top_uniform:
            A = top_row_vals[0]
            out[0, :] = A
            comps = _find_components_8conn(inp, A)
            for comp in comps:
                inner = [(r, c) for r, c in comp if r != 0]
                if not inner:
                    continue
                top_r = min(r for r, c in inner)
                shift = 1 - top_r
                for r, c in inner:
                    nr = r + shift
                    if 1 <= nr < H:
                        out[nr, c] = A
        else:
            B = bot_row_vals[0]
            out[H-1, :] = B
            comps = _find_components_8conn(inp, B)
            for comp in comps:
                inner = [(r, c) for r, c in comp if r != H-1]
                if not inner:
                    continue
                bot_r = max(r for r, c in inner)
                shift = (H - 2) - bot_r
                for r, c in inner:
                    nr = r + shift
                    if 0 <= nr <= H-2:
                        out[nr, c] = B

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            inp = _np(pair['input'])
            out_e = _np(pair['output'])
            if not _same_shape(inp, out_e):
                return None
            pred = self._apply(inp)
            if pred is None or not _grids_equal(pred, out_e):
                return None

        results = []
        for test_inp in ctx.test_inputs:
            inp = _np(test_inp)
            pred = self._apply(inp)
            if pred is None:
                return None
            results.append(_to_grid(pred))

        if not results:
            return None
        return _result(ctx, results, self.name)
