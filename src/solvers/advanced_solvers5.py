"""
advanced_solvers5.py — Fifth batch of targeted solvers for ARC-AGI-2.

Three fully-analyzed structural patterns:
  1. ColorKeyStampSolver       — color-key strip drives vertical stacking of shapes
  2. GridCellStampSolver       — all-0 dividers create cells; stamp shape into triggered cells
  3. NoisyRepeatGridCleanSolver — repeated cells with noise; majority-vote canonical cell
"""

from __future__ import annotations

import numpy as np
from collections import Counter
from typing import Optional, List, Tuple, Dict

from ..core.interfaces import BaseSolver, SolverResult, TaskContext, Prediction
from ..core.grid_codec import background_color
from .registry import register_solver

Grid = List[List[int]]


# ── shared helpers ─────────────────────────────────────────────────────────────

def _sh(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)


def _bg(g: Grid) -> int:
    flat = [v for r in g for v in r]
    return Counter(flat).most_common(1)[0][0] if flat else 0


def _clone(g: Grid) -> Grid:
    return [r[:] for r in g]


def _np(g: Grid) -> np.ndarray:
    return np.array(g, dtype=np.int32)


def _to_grid(a: np.ndarray) -> Grid:
    return a.tolist()


def _grids_equal(a: Grid, b: Grid) -> bool:
    return a == b


def _same_shape(a: Grid, b: Grid) -> bool:
    return _sh(a) == _sh(b)


def _result(ctx: TaskContext, grids: List[Grid], name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ColorKeyStampSolver  (task 94be5b80)
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ColorKeyStampSolver(BaseSolver):
    """
    Detects a horizontal color-key strip (≥3 consecutive identical rows,
    each containing ≥2 distinct non-bg colors) and stacks matching shapes
    in key order, removing the strip from the output.

    Case A: key-colored shapes already exist in the grid — used as anchors
            to derive stride; missing shapes are generated at computed positions.
    Case B: no key-colored shapes exist — a non-key template provides geometry;
            shapes are stacked starting from row 1 downward.
    """

    @property
    def name(self) -> str:
        return "ColorKeyStampSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        grid = _np(inp)
        H, W = grid.shape
        bg = _bg(inp)

        # --- 1. Detect key strip ---
        key_order: Optional[List[int]] = None
        strip_rows: set = set()

        for start in range(H):
            row = grid[start]
            non_bg_vals = row[row != bg]
            if len(set(non_bg_vals.tolist())) < 2:
                continue
            count = 1
            for r in range(start + 1, H):
                if np.array_equal(grid[r], row):
                    count += 1
                else:
                    break
            if count >= 3:
                seen: List[int] = []
                for v in row.tolist():
                    if v != bg and v not in seen:
                        seen.append(v)
                key_order = seen
                strip_rows = set(range(start, start + count))
                break

        if key_order is None or len(key_order) < 2:
            return None

        # --- 2. Collect non-strip, non-bg cells grouped by color ---
        color_cells: Dict[int, List[Tuple[int, int]]] = {}
        for r in range(H):
            if r in strip_rows:
                continue
            for c in range(W):
                val = int(grid[r, c])
                if val != bg:
                    color_cells.setdefault(val, []).append((r, c))

        # --- 3. Key shapes: which key colors already have shapes ---
        key_shapes: Dict[int, Dict] = {}
        for kc in key_order:
            if kc in color_cells:
                cells = color_cells[kc]
                rows_r = [r for r, _ in cells]
                key_shapes[kc] = {
                    "cells": cells,
                    "min_row": min(rows_r),
                    "max_row": max(rows_r),
                }

        out = np.full((H, W), bg, dtype=np.int32)

        if len(key_shapes) >= 1:
            # --- Case A: anchor on existing key-colored shapes ---
            anchors: List[Tuple[int, Dict]] = [
                (key_order.index(kc), s) for kc, s in key_shapes.items()
            ]
            anchors.sort(key=lambda x: x[0])
            idx0, shape0 = anchors[0]

            # Stride from first two anchors, or fallback to shape height
            if len(anchors) >= 2:
                _, shape1 = anchors[1]
                stride = shape1["min_row"] - shape0["min_row"]
            else:
                stride = shape0["max_row"] - shape0["min_row"] + 1

            if stride <= 0:
                return None

            ref_cells = shape0["cells"]
            ref_min_row = shape0["min_row"]
            ref_min_col = min(c for _, c in ref_cells)
            rel_cells = [(r - ref_min_row, c - ref_min_col) for r, c in ref_cells]

            for ki, kc in enumerate(key_order):
                row_start = shape0["min_row"] + (ki - idx0) * stride
                for dr, dc in rel_cells:
                    nr, nc = row_start + dr, ref_min_col + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = kc

        else:
            # --- Case B: no key colors present — use non-key template ---
            template_cells: Optional[List[Tuple[int, int]]] = None
            for color, cells in color_cells.items():
                if color not in key_order:
                    template_cells = cells
                    break
            if template_cells is None:
                return None

            t_min_row = min(r for r, _ in template_cells)
            t_max_row = max(r for r, _ in template_cells)
            t_min_col = min(c for _, c in template_cells)
            shape_h = t_max_row - t_min_row + 1
            rel_cells = [(r - t_min_row, c - t_min_col) for r, c in template_cells]

            for i, kc in enumerate(key_order):
                row_start = 1 + i * shape_h
                for dr, dc in rel_cells:
                    nr, nc = row_start + dr, t_min_col + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr, nc] = kc

        return _to_grid(out)

    def _verify(self, inp: Grid, out: Grid) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for p in train:
            if not self._verify(p["input"], p["output"]):
                return None

        results: List[Grid] = []
        for ti in ctx.test_inputs:
            out = self._apply(ti)
            if out is None:
                return None
            results.append(out)

        return _result(ctx, results, self.name) if results else None


# ══════════════════════════════════════════════════════════════════════════════
# 2. GridCellStampSolver  (task e734a0e8)
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class GridCellStampSolver(BaseSolver):
    """
    Grid divided by all-zero rows/cols into rectangular cells.
    One cell contains a non-bg shape ('shape cell').
    Cells containing a 0 pixel inside ('trigger cells') receive the shape
    stamped into them at identical local positions.
    Clean cells (all-bg, no 0) are left unchanged.
    """

    @property
    def name(self) -> str:
        return "GridCellStampSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        grid = _np(inp)
        H, W = grid.shape
        bg = _bg(inp)

        # --- 1. All-zero divider rows and cols ---
        div_rows = [r for r in range(H) if np.all(grid[r] == 0)]
        div_cols = [c for c in range(W) if np.all(grid[:, c] == 0)]

        if not div_rows or not div_cols:
            return None

        def get_bands(size: int, divs: List[int]) -> List[Tuple[int, int]]:
            bands: List[Tuple[int, int]] = []
            prev = 0
            for d in sorted(set(divs)):
                if d > prev:
                    bands.append((prev, d))
                prev = d + 1
            if prev < size:
                bands.append((prev, size))
            return bands

        row_bands = get_bands(H, div_rows)
        col_bands = get_bands(W, div_cols)

        if len(row_bands) < 2 or len(col_bands) < 2:
            return None

        # --- 2. Identify shape cell and trigger cells ---
        shape_content: Optional[np.ndarray] = None
        shape_h = shape_w = 0
        trigger_list: List[Tuple[int, int, int, int]] = []

        for r1, r2 in row_bands:
            for c1, c2 in col_bands:
                region = grid[r1:r2, c1:c2]
                has_shape = bool(((region != bg) & (region != 0)).any())
                has_trigger = bool((region == 0).any())
                if has_shape:
                    if shape_content is not None:
                        return None  # ambiguous — multiple shape cells
                    shape_content = region.copy()
                    shape_h = r2 - r1
                    shape_w = c2 - c1
                elif has_trigger:
                    trigger_list.append((r1, r2, c1, c2))

        if shape_content is None or not trigger_list:
            return None

        # --- 3. Stamp shape into each same-size trigger cell ---
        out = grid.copy()
        for r1, r2, c1, c2 in trigger_list:
            if (r2 - r1) == shape_h and (c2 - c1) == shape_w:
                out[r1:r2, c1:c2] = shape_content

        return _to_grid(out)

    def _verify(self, inp: Grid, out: Grid) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for p in train:
            if not self._verify(p["input"], p["output"]):
                return None

        results: List[Grid] = []
        for ti in ctx.test_inputs:
            out = self._apply(ti)
            if out is None:
                return None
            results.append(out)

        return _result(ctx, results, self.name) if results else None


# ══════════════════════════════════════════════════════════════════════════════
# 3. NoisyRepeatGridCleanSolver  (task 0607ce86)
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class NoisyRepeatGridCleanSolver(BaseSolver):
    """
    Grid arranged as N×M identical cells separated by 0-divider rows/cols
    (possibly doubled), with scattered noise pixels inside and outside cells.
    Output: replace every cell with the majority-vote canonical cell;
    all separator and exterior rows/cols → 0.
    """

    @property
    def name(self) -> str:
        return "NoisyRepeatGridCleanSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        grid = _np(inp)
        H, W = grid.shape

        # --- 1. Noise-robust separator detection (>50 % zeros in row/col) ---
        sep_rows = set(r for r in range(H) if (grid[r] == 0).mean() > 0.5)
        sep_cols = set(c for c in range(W) if (grid[:, c] == 0).mean() > 0.5)

        if len(sep_rows) < 2 or len(sep_cols) < 2:
            return None

        def get_bands(size: int, seps: set) -> List[Tuple[int, int]]:
            bands: List[Tuple[int, int]] = []
            in_band = False
            start = 0
            for i in range(size):
                if i in seps:
                    if in_band:
                        bands.append((start, i))
                        in_band = False
                else:
                    if not in_band:
                        start = i
                        in_band = True
            if in_band:
                bands.append((start, size))
            return bands

        row_bands = get_bands(H, sep_rows)
        col_bands = get_bands(W, sep_cols)

        if len(row_bands) < 2 or len(col_bands) < 2:
            return None

        # --- 2. Dominant cell size (noise may create extra small bands) ---
        cell_h = Counter(r2 - r1 for r1, r2 in row_bands).most_common(1)[0][0]
        cell_w = Counter(c2 - c1 for c1, c2 in col_bands).most_common(1)[0][0]

        main_rb = [(r1, r2) for r1, r2 in row_bands if (r2 - r1) == cell_h]
        main_cb = [(c1, c2) for c1, c2 in col_bands if (c2 - c1) == cell_w]

        if len(main_rb) < 2 or len(main_cb) < 2:
            return None

        # --- 3. Majority-vote canonical cell ---
        canonical = np.zeros((cell_h, cell_w), dtype=grid.dtype)
        for i in range(cell_h):
            for j in range(cell_w):
                votes = [
                    int(grid[r1 + i, c1 + j])
                    for r1, _ in main_rb
                    for c1, _ in main_cb
                ]
                canonical[i, j] = Counter(votes).most_common(1)[0][0]

        # --- 4. Build output: zeros everywhere; fill cell regions with canonical ---
        out = np.zeros_like(grid)
        for r1, r2 in main_rb:
            for c1, c2 in main_cb:
                out[r1:r2, c1:c2] = canonical

        return _to_grid(out)

    def _verify(self, inp: Grid, out: Grid) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for p in train:
            if not self._verify(p["input"], p["output"]):
                return None

        results: List[Grid] = []
        for ti in ctx.test_inputs:
            out = self._apply(ti)
            if out is None:
                return None
            results.append(out)

        return _result(ctx, results, self.name) if results else None
