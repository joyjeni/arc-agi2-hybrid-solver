"""
Agentic Solvers — generalized pattern-based solvers for ARC-AGI-2.

Each solver learns the transformation rule entirely from training pairs and
applies it to unseen test inputs.  No task-specific hard-coding.

Patterns covered
────────────────
  1.  HorizontalConcatSolver    — output = input tiled N times horizontally
  2.  VerticalConcatSolver      — output = input tiled N times vertically
  3.  FillEnclosedRegionSolver  — flood-fill interior (non-border-reachable) bg cells
  4.  XORHalvesSolver           — cells where left-half ≠ right-half (or top ≠ bottom)
  5.  UniqueHoleObjectSolver    — find object with enclosed region → output its color 1×1
  6.  RemoveIsolatedPixelsSolver— remove single-cell isolated fg pixels
  7.  ScaleDownBlockSolver      — output = integer downscale (mode per NxN block)
  8.  ObjectBBoxExtractSolver   — crop bounding box of non-background content
  9.  PaletteSwapFreqSolver     — recolor by sorted frequency rank
  10. RowTileSolver             — output = single repeating row tiled to fill grid
"""

from __future__ import annotations

import collections
from typing import Optional

from ..core.grid_codec import (
    Grid,
    background_color,
    color_histogram,
    copy_grid,
    flip_h,
    flip_v,
    grid_dims,
    rotate90,
)
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver

# ── Shared utilities ──────────────────────────────────────────────────────────

def _mk(grid: Grid, source: str) -> Prediction:
    return Prediction(attempt_1=grid, attempt_2=grid, solver_name=source, confidence=1.0)

def _res(task_id: str, preds: list[Prediction]) -> SolverResult:
    return SolverResult(task_id=task_id, predictions=preds, solved=True, tier=1)

def _dims(grid: Grid) -> tuple[int, int]:
    return len(grid), (len(grid[0]) if grid else 0)

def _eq(a: Grid, b: Grid) -> bool:
    if _dims(a) != _dims(b):
        return False
    return all(a[r][c] == b[r][c] for r in range(len(a)) for c in range(len(a[0])))

def _bg(grid: Grid) -> int:
    return background_color(grid)

def _flood_border_reachable(grid: Grid) -> set[tuple[int, int]]:
    """Return all bg-colored cells reachable from any border cell (4-connected)."""
    rows, cols = _dims(grid)
    bg = _bg(grid)
    visited: set[tuple[int, int]] = set()
    q: collections.deque[tuple[int, int]] = collections.deque()

    def _seed(r: int, c: int) -> None:
        if grid[r][c] == bg and (r, c) not in visited:
            visited.add((r, c))
            q.append((r, c))

    for r in range(rows):
        _seed(r, 0)
        _seed(r, cols - 1)
    for c in range(cols):
        _seed(0, c)
        _seed(rows - 1, c)

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == bg:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited

def _interior_cells(grid: Grid) -> list[tuple[int, int]]:
    """Background cells NOT reachable from any border (enclosed holes)."""
    rows, cols = _dims(grid)
    bg = _bg(grid)
    reachable = _flood_border_reachable(grid)
    return [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] == bg and (r, c) not in reachable
    ]

def _connected_components(grid: Grid, target_color: int | None = None) -> list[list[tuple[int, int]]]:
    """4-connected components for target_color (or all non-bg if None)."""
    rows, cols = _dims(grid)
    bg = _bg(grid)
    visited = [[False] * cols for _ in range(rows)]
    comps: list[list[tuple[int, int]]] = []
    for sr in range(rows):
        for sc in range(cols):
            cv = grid[sr][sc]
            if visited[sr][sc]:
                continue
            if target_color is not None and cv != target_color:
                continue
            if target_color is None and cv == bg:
                continue
            comp: list[tuple[int, int]] = []
            q: collections.deque[tuple[int, int]] = collections.deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == grid[r][c]:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps

def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (rmin, rmax, cmin, cmax) for a list of (r, c) cells."""
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), max(rs), min(cs), max(cs)

def _crop_bbox(grid: Grid) -> Grid:
    """Crop grid to the bounding box of all non-background cells."""
    rows, cols = _dims(grid)
    bg = _bg(grid)
    non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
    if not non_bg:
        return grid
    r0, r1, c0, c1 = _bbox(non_bg)
    return [row[c0: c1 + 1] for row in grid[r0: r1 + 1]]

def _mode(values: list[int]) -> int:
    return max(set(values), key=values.count)

# ── 1. HorizontalConcatSolver ─────────────────────────────────────────────────

@register_solver
class HorizontalConcatSolver(BaseSolver):
    """Output = input tiled N times side-by-side (N ≥ 2, learned from training)."""

    @property
    def name(self) -> str:
        return "HorizontalConcatSolver"

    @property
    def priority(self) -> int:
        return 40

    @staticmethod
    def _detect_n(inp: Grid, out: Grid) -> int | None:
        ri, ci = _dims(inp)
        ro, co = _dims(out)
        if ro != ri or co % ci != 0:
            return None
        n = co // ci
        if n < 2:
            return None
        for r in range(ri):
            for k in range(n):
                for c in range(ci):
                    if out[r][k * ci + c] != inp[r][c]:
                        return None
        return n

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        n = self._detect_n(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if n is None:
            return None
        for pair in ctx.train_pairs[1:]:
            if self._detect_n(pair["input"], pair["output"]) != n:
                return None
        preds = []
        for inp in ctx.test_inputs:
            rows, cols = _dims(inp)
            out = [inp[r] * n for r in range(rows)]
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ── 2. VerticalConcatSolver ───────────────────────────────────────────────────

@register_solver
class VerticalConcatSolver(BaseSolver):
    """Output = input tiled N times vertically (N ≥ 2, learned from training)."""

    @property
    def name(self) -> str:
        return "VerticalConcatSolver"

    @property
    def priority(self) -> int:
        return 40

    @staticmethod
    def _detect_n(inp: Grid, out: Grid) -> int | None:
        ri, ci = _dims(inp)
        ro, co = _dims(out)
        if co != ci or ro % ri != 0:
            return None
        n = ro // ri
        if n < 2:
            return None
        for k in range(n):
            for r in range(ri):
                if out[k * ri + r] != inp[r]:
                    return None
        return n

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        n = self._detect_n(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if n is None:
            return None
        for pair in ctx.train_pairs[1:]:
            if self._detect_n(pair["input"], pair["output"]) != n:
                return None
        preds = []
        for inp in ctx.test_inputs:
            out = []
            for _ in range(n):
                out.extend([list(row) for row in inp])
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ── 3. FillEnclosedRegionSolver ───────────────────────────────────────────────

@register_solver
class FillEnclosedRegionSolver(BaseSolver):
    """
    Fill bg cells that are enclosed (not border-reachable) with a fill color.
    The fill color is learned from training: it is the color present in the output
    at interior-cell positions that differs from the background.
    """

    @property
    def name(self) -> str:
        return "FillEnclosedRegionSolver"

    @property
    def priority(self) -> int:
        return 45

    @staticmethod
    def _learn_fill_color(inp: Grid, out: Grid) -> int | None:
        # Require same dimensions
        if _dims(inp) != _dims(out):
            return None
        interior = _interior_cells(inp)
        if not interior:
            return None
        # All interior cells in the output should have the same non-bg fill color
        bg = _bg(inp)
        fill_colors = {out[r][c] for r, c in interior}
        if len(fill_colors) != 1:
            return None
        fill = next(iter(fill_colors))
        if fill == bg:
            return None
        # Non-interior cells must be unchanged
        rows, cols = _dims(inp)
        interior_set = set(interior)
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in interior_set and inp[r][c] != out[r][c]:
                    return None
        return fill

    @staticmethod
    def _apply(inp: Grid, fill_color: int) -> Grid:
        interior = _interior_cells(inp)
        out = copy_grid(inp)
        for r, c in interior:
            out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        fill = self._learn_fill_color(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if fill is None:
            return None
        for pair in ctx.train_pairs[1:]:
            if self._learn_fill_color(pair["input"], pair["output"]) != fill:
                return None
            if not _eq(self._apply(pair["input"], fill), pair["output"]):
                return None
        preds = [_mk(self._apply(inp, fill), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ── 4. XORHalvesSolver ────────────────────────────────────────────────────────

@register_solver
class XORHalvesSolver(BaseSolver):
    """
    Split input in half (left/right or top/bottom).
    Output = cells where the two halves differ (mark with a fixed color),
    or the symmetric difference grid.  Learns the exact output rule from training.
    """

    @property
    def name(self) -> str:
        return "XORHalvesSolver"

    @property
    def priority(self) -> int:
        return 42

    @staticmethod
    def _split_lr(grid: Grid) -> tuple[Grid, Grid] | None:
        rows, cols = _dims(grid)
        if cols % 2 != 0:
            return None
        half = cols // 2
        left = [row[:half] for row in grid]
        right = [row[half:] for row in grid]
        return left, right

    @staticmethod
    def _split_tb(grid: Grid) -> tuple[Grid, Grid] | None:
        rows, cols = _dims(grid)
        if rows % 2 != 0:
            return None
        half = rows // 2
        top = grid[:half]
        bot = grid[half:]
        return top, bot

    @staticmethod
    def _diff_grid(a: Grid, b: Grid, mark: int, same: int) -> Grid:
        rows, cols = _dims(a)
        return [
            [mark if a[r][c] != b[r][c] else same for c in range(cols)]
            for r in range(rows)
        ]

    def _try_rule(self, inp: Grid, out: Grid) -> dict | None:
        """Returns a rule dict if any halving + diff rule explains inp→out."""
        ro, co = _dims(out)
        # Try LR split
        split = self._split_lr(inp)
        if split:
            left, right = split
            hr, hc = _dims(left)
            if (ro, co) == (hr, hc):
                # Find mark and same colors from out
                marks = set()
                for r in range(hr):
                    for c in range(hc):
                        if left[r][c] != right[r][c]:
                            marks.add(out[r][c])
                sames = set()
                for r in range(hr):
                    for c in range(hc):
                        if left[r][c] == right[r][c]:
                            sames.add(out[r][c])
                if len(marks) == 1 and len(sames) == 1:
                    mark, same = next(iter(marks)), next(iter(sames))
                    if mark != same:
                        result = self._diff_grid(left, right, mark, same)
                        if _eq(result, out):
                            return {"axis": "lr", "mark": mark, "same": same}
        # Try TB split
        split = self._split_tb(inp)
        if split:
            top, bot = split
            hr, hc = _dims(top)
            if (ro, co) == (hr, hc):
                marks = set()
                for r in range(hr):
                    for c in range(hc):
                        if top[r][c] != bot[r][c]:
                            marks.add(out[r][c])
                sames = set()
                for r in range(hr):
                    for c in range(hc):
                        if top[r][c] == bot[r][c]:
                            sames.add(out[r][c])
                if len(marks) == 1 and len(sames) == 1:
                    mark, same = next(iter(marks)), next(iter(sames))
                    if mark != same:
                        result = self._diff_grid(top, bot, mark, same)
                        if _eq(result, out):
                            return {"axis": "tb", "mark": mark, "same": same}
        return None

    def _apply_rule(self, inp: Grid, rule: dict) -> Grid:
        if rule["axis"] == "lr":
            halves = self._split_lr(inp)
            if halves is None:
                return inp
            a, b = halves
        else:
            halves = self._split_tb(inp)
            if halves is None:
                return inp
            a, b = halves
        return self._diff_grid(a, b, rule["mark"], rule["same"])

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        rule = self._try_rule(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if rule is None:
            return None
        for pair in ctx.train_pairs[1:]:
            r2 = self._try_rule(pair["input"], pair["output"])
            if r2 != rule:
                return None
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply_rule(inp, rule)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ── 5. UniqueHoleObjectSolver ─────────────────────────────────────────────────

@register_solver
class UniqueHoleObjectSolver(BaseSolver):
    """
    Among all distinct colored objects, find the unique one that has an enclosed
    background region inside it.  Output = 1×1 grid with that object's color.
    """

    @property
    def name(self) -> str:
        return "UniqueHoleObjectSolver"

    @property
    def priority(self) -> int:
        return 50

    @staticmethod
    def _object_has_hole(grid: Grid, comp: list[tuple[int, int]]) -> bool:
        """True if drawing only this object's cells creates an enclosed interior."""
        rows, cols = _dims(grid)
        bg = _bg(grid)
        comp_set = set(comp)
        # Build a grid with only this object
        temp = [[bg] * cols for _ in range(rows)]
        color = grid[comp[0][0]][comp[0][1]]
        for r, c in comp:
            temp[r][c] = color
        interior = _interior_cells(temp)
        return len(interior) > 0

    @staticmethod
    def _find_hole_object_color(grid: Grid) -> int | None:
        comps = _connected_components(grid)
        hole_colors = []
        for comp in comps:
            color = grid[comp[0][0]][comp[0][1]]
            if UniqueHoleObjectSolver._object_has_hole(grid, comp):
                hole_colors.append(color)
        if len(hole_colors) == 1:
            return hole_colors[0]
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # Each training output must be [[color]] (1×1) matching the hole-object color
        for pair in ctx.train_pairs:
            out = pair["output"]
            if _dims(out) != (1, 1):
                return None
            expected = self._find_hole_object_color(pair["input"])
            if expected is None or out[0][0] != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            color = self._find_hole_object_color(inp)
            if color is None:
                return None
            preds.append(_mk([[color]], self.name))
        return _res(ctx.task_id, preds)


# ── 6. RemoveIsolatedPixelsSolver ─────────────────────────────────────────────

@register_solver
class RemoveIsolatedPixelsSolver(BaseSolver):
    """
    Remove isolated (single-cell, 4-connected) non-background pixels by
    replacing them with the background color.  Validates on training pairs.
    """

    @property
    def name(self) -> str:
        return "RemoveIsolatedPixelsSolver"

    @property
    def priority(self) -> int:
        return 35

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        rows, cols = _dims(grid)
        bg = _bg(grid)
        out = copy_grid(grid)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == bg:
                    continue
                neighbors = [
                    grid[r + dr][c + dc]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= r + dr < rows and 0 <= c + dc < cols
                ]
                if not any(n == grid[r][c] for n in neighbors):
                    out[r][c] = bg
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            applied = self._apply(pair["input"])
            if not _eq(applied, pair["output"]):
                return None
            # Must actually remove at least one pixel in some training pair
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ── 7. ScaleDownBlockSolver ───────────────────────────────────────────────────

@register_solver
class ScaleDownBlockSolver(BaseSolver):
    """
    Output = input downscaled by integer factor N.
    Each N×N block in the input maps to a single output cell (mode color).
    N is learned from training (2–8).
    """

    @property
    def name(self) -> str:
        return "ScaleDownBlockSolver"

    @property
    def priority(self) -> int:
        return 38

    @staticmethod
    def _try_factor(inp: Grid, out: Grid) -> int | None:
        ri, ci = _dims(inp)
        ro, co = _dims(out)
        if ro == 0 or co == 0 or ri % ro != 0 or ci % co != 0:
            return None
        nf, mf = ri // ro, ci // co
        if nf != mf or nf < 2 or nf > 8:
            return None
        # Verify: mode of each block == out cell
        for r in range(ro):
            for c in range(co):
                block = [
                    inp[r * nf + dr][c * mf + dc]
                    for dr in range(nf) for dc in range(mf)
                ]
                if _mode(block) != out[r][c]:
                    return None
        return nf

    @staticmethod
    def _apply(inp: Grid, factor: int) -> Grid:
        ri, ci = _dims(inp)
        ro, co = ri // factor, ci // factor
        out = []
        for r in range(ro):
            row = []
            for c in range(co):
                block = [
                    inp[r * factor + dr][c * factor + dc]
                    for dr in range(factor) for dc in range(factor)
                ]
                row.append(_mode(block))
            out.append(row)
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        factor = self._try_factor(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if factor is None:
            return None
        for pair in ctx.train_pairs[1:]:
            if self._try_factor(pair["input"], pair["output"]) != factor:
                return None
        preds = []
        for inp in ctx.test_inputs:
            ri, ci = _dims(inp)
            if ri % factor != 0 or ci % factor != 0:
                return None
            preds.append(_mk(self._apply(inp, factor), self.name))
        return _res(ctx.task_id, preds)


# ── 8. ObjectBBoxExtractSolver ────────────────────────────────────────────────

@register_solver
class ObjectBBoxExtractSolver(BaseSolver):
    """
    Output = bounding-box crop of all non-background cells in the input.
    Validates that training pairs match exactly.
    """

    @property
    def name(self) -> str:
        return "ObjectBBoxExtractSolver"

    @property
    def priority(self) -> int:
        return 36

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            cropped = _crop_bbox(pair["input"])
            if not _eq(cropped, pair["output"]):
                return None
            # Must actually reduce size
            if _dims(cropped) == _dims(pair["input"]):
                return None
        preds = [_mk(_crop_bbox(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ── 9. PaletteSwapFreqSolver ──────────────────────────────────────────────────

@register_solver
class PaletteSwapFreqSolver(BaseSolver):
    """
    Recolor grid by sorted frequency rank of non-background colors.
    Learns the target color for each frequency rank from training pairs.
    Rejects identity mappings.
    """

    @property
    def name(self) -> str:
        return "PaletteSwapFreqSolver"

    @property
    def priority(self) -> int:
        return 32

    @staticmethod
    def _freq_rank_map(inp: Grid, out: Grid) -> dict[int, int] | None:
        """Build color_map: input_color → output_color, keyed by freq rank."""
        rows, cols = _dims(inp)
        if _dims(inp) != _dims(out):
            return None
        # Count frequencies of each color in input (all cells)
        hist: dict[int, int] = {}
        for r in range(rows):
            for c in range(cols):
                hist[inp[r][c]] = hist.get(inp[r][c], 0) + 1
        sorted_colors = sorted(hist, key=lambda x: hist[x], reverse=True)
        color_map: dict[int, int] = {}
        for color in sorted_colors:
            # Find what color it maps to in out
            out_colors = set()
            for r in range(rows):
                for c in range(cols):
                    if inp[r][c] == color:
                        out_colors.add(out[r][c])
            if len(out_colors) != 1:
                return None
            color_map[color] = next(iter(out_colors))
        return color_map

    @staticmethod
    def _apply(grid: Grid, color_map: dict[int, int]) -> Grid:
        rows, cols = _dims(grid)
        return [[color_map.get(grid[r][c], grid[r][c]) for c in range(cols)] for r in range(rows)]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        ref_map = self._freq_rank_map(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if ref_map is None:
            return None
        # Reject identity
        if all(k == v for k, v in ref_map.items()):
            return None
        for pair in ctx.train_pairs[1:]:
            m = self._freq_rank_map(pair["input"], pair["output"])
            if m is None or m != ref_map:
                return None
            if not _eq(self._apply(pair["input"], ref_map), pair["output"]):
                return None
        preds = [_mk(self._apply(inp, ref_map), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ── 10. RowTileSolver ─────────────────────────────────────────────────────────

@register_solver
class RowTileSolver(BaseSolver):
    """
    Output = single repeating row pattern tiled vertically to fill the grid.
    The tile row is detected from training inputs (all rows identical).
    """

    @property
    def name(self) -> str:
        return "RowTileSolver"

    @property
    def priority(self) -> int:
        return 38

    @staticmethod
    def _detect_tile_row(inp: Grid, out: Grid) -> list[int] | None:
        ro, co = _dims(out)
        ri, ci = _dims(inp)
        if co != ci:
            return None
        # Check if all rows in out are identical
        first = out[0]
        if not all(out[r] == first for r in range(ro)):
            return None
        # That row should appear at least once in inp
        if first not in inp:
            return None
        return list(first)

    @staticmethod
    def _apply(inp: Grid, tile_row: list[int], target_rows: int) -> Grid:
        return [list(tile_row) for _ in range(target_rows)]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pair0 = ctx.train_pairs[0]
        tile_row = self._detect_tile_row(pair0["input"], pair0["output"])
        if tile_row is None:
            return None
        for pair in ctx.train_pairs[1:]:
            tr = self._detect_tile_row(pair["input"], pair["output"])
            if tr != tile_row:
                return None
        preds = []
        for inp in ctx.test_inputs:
            rows, _ = _dims(inp)
            preds.append(_mk(self._apply(inp, tile_row, rows), self.name))
        return _res(ctx.task_id, preds)


# ── 11. Rotate90Solver ────────────────────────────────────────────────────────

@register_solver
class Rotate90Solver(BaseSolver):
    """Output = input rotated by 90, 180, or 270 degrees (learned from training)."""

    @property
    def name(self) -> str:
        return "Rotate90Solver"

    @property
    def priority(self) -> int:
        return 40

    @staticmethod
    def _rot90(grid: Grid) -> Grid:
        """90° clockwise rotation, handles non-square grids."""
        H = len(grid)
        W = len(grid[0]) if grid else 0
        # New grid: W rows × H cols
        return [[grid[H - 1 - nc][nr] for nc in range(H)] for nr in range(W)]

    def _rotn(self, grid: Grid, n: int) -> Grid:
        for _ in range(n):
            grid = self._rot90(grid)
        return grid

    def _find_rotation(self, inp: Grid, out: Grid) -> int | None:
        for n in (1, 2, 3):
            if _eq(self._rotn(inp, n), out):
                return n
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        n = self._find_rotation(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if n is None:
            return None
        for pair in ctx.train_pairs[1:]:
            if self._find_rotation(pair["input"], pair["output"]) != n:
                return None
        preds = [_mk(self._rotn(inp, n), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ── 12. DiagonalReflectSolver ─────────────────────────────────────────────────

@register_solver
class DiagonalReflectSolver(BaseSolver):
    """
    Output = input reflected across main diagonal (transpose) or anti-diagonal.
    Handles non-square grids for transpose only.
    """

    @property
    def name(self) -> str:
        return "DiagonalReflectSolver"

    @property
    def priority(self) -> int:
        return 40

    @staticmethod
    def _anti_diagonal(grid: Grid) -> Grid:
        """Reflect across anti-diagonal; output is transposed dimensions."""
        H = len(grid)
        W = len(grid[0]) if grid else 0
        # new[r][c] = old[H-1-c][W-1-r], output is W×H
        return [[grid[H - 1 - c][W - 1 - r] for c in range(H)] for r in range(W)]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        from ..core.grid_codec import transpose as _transpose
        for transform_name, transform_fn in [
            ("transpose", _transpose),
            ("anti_diag", self._anti_diagonal),
        ]:
            valid = True
            for pair in ctx.train_pairs:
                try:
                    transformed = transform_fn(pair["input"])
                except Exception:
                    valid = False
                    break
                if not _eq(transformed, pair["output"]):
                    valid = False
                    break
            if valid:
                preds = []
                for inp in ctx.test_inputs:
                    try:
                        out = transform_fn(inp)
                    except Exception:
                        return None
                    preds.append(_mk(out, self.name + "_" + transform_name))
                return _res(ctx.task_id, preds)
        return None


# ── 13. ColorCountTo1x1Solver ─────────────────────────────────────────────────

@register_solver
class ColorCountTo1x1Solver(BaseSolver):
    """
    Count connected components of a specific color → output 1×1 grid
    where cell value = count.  Learns target color and count from training.
    """

    @property
    def name(self) -> str:
        return "ColorCountTo1x1Solver"

    @property
    def priority(self) -> int:
        return 48

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # All outputs must be 1×1
        if not all(_dims(p["output"]) == (1, 1) for p in ctx.train_pairs):
            return None

        # Try counting components of each non-background color
        bg = _bg(ctx.train_pairs[0]["input"])
        all_colors: set[int] = set()
        for pair in ctx.train_pairs:
            for row in pair["input"]:
                all_colors.update(row)
        all_colors.discard(bg)

        for target_color in all_colors:
            valid = True
            for pair in ctx.train_pairs:
                comps = _connected_components(pair["input"], target_color)
                count = len(comps)
                if pair["output"][0][0] != count:
                    valid = False
                    break
            if valid:
                preds = []
                for inp in ctx.test_inputs:
                    comps = _connected_components(inp, target_color)
                    preds.append(_mk([[len(comps)]], self.name))
                return _res(ctx.task_id, preds)
        return None


# ── 14. MaxColorComponentSolver ───────────────────────────────────────────────

@register_solver
class MaxColorComponentSolver(BaseSolver):
    """
    Keep only the largest connected component; replace all others with background.
    """

    @property
    def name(self) -> str:
        return "MaxColorComponentSolver"

    @property
    def priority(self) -> int:
        return 33

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        comps = _connected_components(grid)
        if not comps:
            return grid
        largest = max(comps, key=len)
        largest_set = set(largest)
        bg = _bg(grid)
        rows, cols = _dims(grid)
        out = copy_grid(grid)
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != bg and (r, c) not in largest_set:
                    out[r][c] = bg
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not _eq(self._apply(pair["input"]), pair["output"]):
                return None
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ── 15. FrameExtractSolver ────────────────────────────────────────────────────

@register_solver
class FrameExtractSolver(BaseSolver):
    """
    Output = only the border/frame cells; interior replaced with background.
    """

    @property
    def name(self) -> str:
        return "FrameExtractSolver"

    @property
    def priority(self) -> int:
        return 33

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        rows, cols = _dims(grid)
        bg = _bg(grid)
        out = [[bg] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    out[r][c] = grid[r][c]
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not _eq(self._apply(pair["input"]), pair["output"]):
                return None
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# FlipHorizontalSolver — output = input mirrored left-right
# Verified on task 67a3c6ac
# ---------------------------------------------------------------------------
@register_solver
class FlipHorizontalSolver(BaseSolver):
    """Output = input flipped horizontally (columns reversed)."""

    @property
    def name(self) -> str:
        return "FlipHorizontalSolver"

    @property
    def priority(self) -> int:
        return 34

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        return [list(reversed(row)) for row in grid]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not _eq(self._apply(pair["input"]), pair["output"]):
                return None
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# FlipVerticalSolver — output = input mirrored top-bottom
# Verified on task 68b16354
# ---------------------------------------------------------------------------
@register_solver
class FlipVerticalSolver(BaseSolver):
    """Output = input flipped vertically (rows reversed)."""

    @property
    def name(self) -> str:
        return "FlipVerticalSolver"

    @property
    def priority(self) -> int:
        return 35

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        return list(reversed(grid))

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not _eq(self._apply(pair["input"]), pair["output"]):
                return None
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# TransposeSolver — output = matrix transpose of input
# Verified on tasks 74dd1130, 9dfd6313
# ---------------------------------------------------------------------------
@register_solver
class TransposeSolver(BaseSolver):
    """Output = transpose of input (rows become columns)."""

    @property
    def name(self) -> str:
        return "TransposeSolver"

    @property
    def priority(self) -> int:
        return 36

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        rows, cols = len(grid), len(grid[0]) if grid else 0
        return [[grid[r][c] for r in range(rows)] for c in range(cols)]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if not _eq(self._apply(inp), out):
                return None
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# PaletteSwapSolver — learns a global color→color mapping from training pairs
# Verified on task 0d3d703e (all-identical-rows grids)
# ---------------------------------------------------------------------------
@register_solver
class PaletteSwapSolver(BaseSolver):
    """
    Output = input with each color substituted by a learned fixed mapping.
    Mapping is inferred consistently across all training pairs.
    """

    @property
    def name(self) -> str:
        return "PaletteSwapSolver"

    @property
    def priority(self) -> int:
        return 37

    @staticmethod
    def _build_map(pairs: list) -> dict | None:
        """Build and validate a consistent color→color mapping from all pairs."""
        color_map: dict[int, int] = {}
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            r_in, c_in   = len(inp), len(inp[0]) if inp else 0
            r_out, c_out = len(out), len(out[0]) if out else 0
            if r_in != r_out or c_in != c_out:
                return None
            for r in range(r_in):
                for c in range(c_in):
                    ci, co = inp[r][c], out[r][c]
                    if ci in color_map:
                        if color_map[ci] != co:
                            return None  # inconsistent
                    else:
                        color_map[ci] = co
        return color_map if color_map else None

    @staticmethod
    def _apply(grid: Grid, color_map: dict) -> Grid:
        return [[color_map.get(cell, cell) for cell in row] for row in grid]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        color_map = self._build_map(ctx.train_pairs)
        if color_map is None:
            return None
        # Verify all training pairs are perfectly explained
        for pair in ctx.train_pairs:
            if not _eq(self._apply(pair["input"], color_map), pair["output"]):
                return None
        # Guard: reject if any test input contains a color absent from the learned map.
        # Without this, get(x, x) silently returns x for unseen colors, producing
        # false positives (e.g. task aabf363d where test has shape color 8, not in map).
        for test_inp in ctx.test_inputs:
            for row in test_inp:
                for cell in row:
                    if cell not in color_map:
                        return None
        preds = [_mk(self._apply(inp, color_map), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# ColGroupPatternColorSolver — divides input into equal-width column groups,
# learns a pattern→color mapping, fills each group uniformly in output.
# Verified on task 17cae0c1 (3×9 grid divided into 3-col groups)
# ---------------------------------------------------------------------------
@register_solver
class ColGroupPatternColorSolver(BaseSolver):
    """
    Input is divided into equal-width column groups.
    Each group's binary pattern (non-zero → 1) maps to a uniform output color.
    The pattern→color mapping is learned from training pairs.
    """

    @property
    def name(self) -> str:
        return "ColGroupPatternColorSolver"

    @property
    def priority(self) -> int:
        return 38

    @staticmethod
    def _extract_group(grid: Grid, start_col: int, width: int):
        """Return a tuple-of-tuples for the column group (binary: 0 or 1)."""
        return tuple(
            tuple(1 if grid[r][c] != 0 else 0 for c in range(start_col, start_col + width))
            for r in range(len(grid))
        )

    @staticmethod
    def _group_output_color(out: Grid, start_col: int, width: int) -> int | None:
        """Return the uniform color for this output group, or None if not uniform."""
        colors = set(
            out[r][c]
            for r in range(len(out))
            for c in range(start_col, start_col + width)
        )
        return next(iter(colors)) if len(colors) == 1 else None

    def _build_map(self, pairs: list) -> tuple[dict | None, int | None]:
        """
        Infer group width and pattern→color mapping.
        Returns (mapping, group_width) or (None, None).
        """
        if not pairs:
            return None, None
        # Infer group width: must divide columns evenly; try widths 1..cols//2
        rows_in, cols_in = _dims(pairs[0]["input"])
        rows_out, cols_out = _dims(pairs[0]["output"])
        if rows_in != rows_out or cols_in != cols_out:
            return None, None

        best_w = None
        for w in range(1, cols_in // 2 + 1):
            if cols_in % w != 0:
                continue
            n_groups = cols_in // w
            if n_groups < 2:
                continue
            pattern_map: dict = {}
            valid = True
            for pair in pairs:
                inp, out = pair["input"], pair["output"]
                if _dims(inp) != (rows_in, cols_in) or _dims(out) != (rows_out, cols_out):
                    valid = False
                    break
                for g in range(n_groups):
                    pat = self._extract_group(inp, g * w, w)
                    col = self._group_output_color(out, g * w, w)
                    if col is None:
                        valid = False
                        break
                    if pat in pattern_map:
                        if pattern_map[pat] != col:
                            valid = False
                            break
                    else:
                        pattern_map[pat] = col
                if not valid:
                    break
            if valid and pattern_map:
                best_w = w
                return pattern_map, w
        return None, None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pattern_map, w = self._build_map(ctx.train_pairs)
        if pattern_map is None:
            return None

        def _apply(grid: Grid) -> Grid | None:
            rows, cols = _dims(grid)
            if cols % w != 0:
                return None
            out = [[0] * cols for _ in range(rows)]
            for g in range(cols // w):
                pat = self._extract_group(grid, g * w, w)
                color = pattern_map.get(pat)
                if color is None:
                    return None
                for r in range(rows):
                    for c in range(g * w, g * w + w):
                        out[r][c] = color
            return out

        preds = []
        for inp in ctx.test_inputs:
            result = _apply(inp)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# CheckerboardLatticeSolver — all-zero input → output with 1s at every cell
# where row_index OR col_index is even (odd-dimensioned grids form a lattice)
# Verified on task 332efdb3
# ---------------------------------------------------------------------------
@register_solver
class CheckerboardLatticeSolver(BaseSolver):
    """
    Input: all-background grid (any dimensions).
    Output: same size, with a specific non-background color at every cell
    where row OR column index is even (0-indexed), background elsewhere.
    """

    @property
    def name(self) -> str:
        return "CheckerboardLatticeSolver"

    @property
    def priority(self) -> int:
        return 39

    @staticmethod
    def _lattice_color(inp: Grid, out: Grid) -> int | None:
        """Return the non-bg lattice color if out matches the r%2==0 or c%2==0 pattern."""
        rows, cols = len(inp), len(inp[0]) if inp else 0
        bg = _bg(inp)
        # Collect non-bg colors in output
        colors = set()
        for r in range(rows):
            for c in range(cols):
                expected_lattice = (r % 2 == 0) or (c % 2 == 0)
                cell = out[r][c]
                if expected_lattice:
                    if cell == bg:
                        return None
                    colors.add(cell)
                else:
                    if cell != bg:
                        return None
        return next(iter(colors)) if len(colors) == 1 else None

    @staticmethod
    def _apply(grid: Grid, lattice_color: int) -> Grid:
        bg = _bg(grid)
        rows, cols = len(grid), len(grid[0]) if grid else 0
        return [
            [lattice_color if (r % 2 == 0 or c % 2 == 0) else bg
             for c in range(cols)]
            for r in range(rows)
        ]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        lattice_colors = []
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            lc = self._lattice_color(inp, out)
            if lc is None:
                return None
            lattice_colors.append(lc)
        if len(set(lattice_colors)) != 1:
            return None  # inconsistent lattice color across pairs
        lc = lattice_colors[0]
        preds = [_mk(self._apply(inp, lc), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# AddBorderSolver — all-zero input → output with a 1-cell border frame added
# Verified on task 6f8cd79b
# ---------------------------------------------------------------------------
@register_solver
class AddBorderSolver(BaseSolver):
    """
    Input: all-background grid.
    Output: same size, with a single-color 1-cell border frame; interior = background.
    """

    @property
    def name(self) -> str:
        return "AddBorderSolver"

    @property
    def priority(self) -> int:
        return 40

    @staticmethod
    def _border_color(inp: Grid, out: Grid) -> int | None:
        """Return frame color if out = frame(inp_bg) + interior(bg)."""
        rows, cols = len(inp), len(inp[0]) if inp else 0
        if rows < 3 or cols < 3:
            return None
        bg = _bg(inp)
        colors = set()
        for r in range(rows):
            for c in range(cols):
                on_border = (r == 0 or r == rows - 1 or c == 0 or c == cols - 1)
                cell = out[r][c]
                if on_border:
                    if cell == bg:
                        return None
                    colors.add(cell)
                else:
                    if cell != bg:
                        return None
        return next(iter(colors)) if len(colors) == 1 else None

    @staticmethod
    def _apply(grid: Grid, border_color: int) -> Grid:
        bg = _bg(grid)
        rows, cols = len(grid), len(grid[0]) if grid else 0
        out = [[bg] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    out[r][c] = border_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        border_colors = []
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            bc = self._border_color(inp, out)
            if bc is None:
                return None
            border_colors.append(bc)
        if len(set(border_colors)) != 1:
            return None
        bc = border_colors[0]
        preds = [_mk(self._apply(inp, bc), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 41. CountNonBgToRowSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class CountNonBgToRowSolver(BaseSolver):
    """
    Input: small grid with N non-background cells, all one color C.
    Output: 1×N row filled entirely with color C.
    """

    @property
    def name(self) -> str:
        return "CountNonBgToRowSolver"

    @property
    def priority(self) -> int:
        return 41

    @staticmethod
    def _extract(grid: Grid, bg: int):
        """Return (count, color) of non-bg cells, or None if not exactly 1 non-bg color."""
        non_bg = [cell for row in grid for cell in row if cell != bg]
        if not non_bg:
            return None
        if len(set(non_bg)) != 1:
            return None
        return len(non_bg), non_bg[0]

    @staticmethod
    def _apply(count: int, color: int) -> Grid:
        return [[color] * count]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # Learn the background color from training inputs (must be consistent)
        train_bgs = [_bg(pair["input"]) for pair in ctx.train_pairs]
        if len(set(train_bgs)) != 1:
            return None
        bg = train_bgs[0]

        learned = []
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            ext = self._extract(inp, bg)
            if ext is None:
                return None
            count, color = ext
            if len(out) != 1 or len(out[0]) != count:
                return None
            if any(cell != color for cell in out[0]):
                return None
            learned.append((count, color))
        # Guard: if all training counts are identical the rule is ambiguous
        # (could be a fixed-width output, not a true count-based transform)
        counts = [c for c, _ in learned]
        if len(learned) >= 2 and len(set(counts)) == 1:
            return None
        preds = []
        for inp in ctx.test_inputs:
            # Use the learned bg, not the per-grid majority (test bg may flip)
            ext = self._extract(inp, bg)
            if ext is None:
                return None
            count, color = ext
            preds.append(_mk(self._apply(count, color), self.name))
        return _res(ctx.task_id, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 42. Stride2SampleSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class Stride2SampleSolver(BaseSolver):
    """
    Input: 2H × 2W grid.
    Output: H × W grid by taking every even-indexed row and column (stride 2).
    """

    @property
    def name(self) -> str:
        return "Stride2SampleSolver"

    @property
    def priority(self) -> int:
        return 42

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        return [row[::2] for row in grid[::2]]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            ri, ci = _dims(inp)
            ro, co = _dims(out)
            if ri % 2 != 0 or ci % 2 != 0:
                return None
            if ro != ri // 2 or co != ci // 2:
                return None
            if self._apply(inp) != out:
                return None
        preds = [_mk(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 43. CrossExpandSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class CrossExpandSolver(BaseSolver):
    """
    Each non-bg dot at (r, c) → full horizontal + vertical lines (dot_color),
    dot replaced with center_color, four diagonal neighbors replaced with diag_color.
    Colors are learned from training pairs.
    """

    @property
    def name(self) -> str:
        return "CrossExpandSolver"

    @property
    def priority(self) -> int:
        return 43

    @staticmethod
    def _expand(grid: Grid, bg: int, dot_color: int, center_color: int, diag_color: int) -> Grid:
        rows, cols = len(grid), len(grid[0])
        out = [[bg] * cols for _ in range(rows)]
        dots = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg]
        for dr, dc in dots:
            for c in range(cols):
                out[dr][c] = dot_color
            for r in range(rows):
                out[r][dc] = dot_color
        for dr, dc in dots:
            out[dr][dc] = center_color
        for dr, dc in dots:
            for delta_r, delta_c in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = dr + delta_r, dc + delta_c
                if 0 <= nr < rows and 0 <= nc < cols:
                    out[nr][nc] = diag_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        dot_colors, center_colors, diag_colors = [], [], []
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            bg = _bg(inp)
            rows, cols = _dims(inp)
            dots = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] != bg]
            if not dots:
                return None
            dc_set = set(inp[r][c] for r, c in dots)
            if len(dc_set) != 1:
                return None
            dot_color = dc_set.pop()
            # center color: what's at dot positions in output
            ctr_set = set(out[r][c] for r, c in dots)
            if len(ctr_set) != 1:
                return None
            center_color = ctr_set.pop()
            if center_color in (bg, dot_color):
                return None
            # diag color
            diag_set = set()
            for dr, dc in dots:
                for delta_r, delta_c in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = dr + delta_r, dc + delta_c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        v = out[nr][nc]
                        if v not in (bg, dot_color, center_color):
                            diag_set.add(v)
            if len(diag_set) != 1:
                return None
            diag_color = diag_set.pop()
            expected = self._expand(inp, bg, dot_color, center_color, diag_color)
            if expected != out:
                return None
            dot_colors.append(dot_color)
            center_colors.append(center_color)
            diag_colors.append(diag_color)
        if not dot_colors:
            return None
        if len(set(dot_colors)) != 1 or len(set(center_colors)) != 1 or len(set(diag_colors)) != 1:
            return None
        dc, cc, dgc = dot_colors[0], center_colors[0], diag_colors[0]
        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            preds.append(_mk(self._expand(inp, bg, dc, cc, dgc), self.name))
        return _res(ctx.task_id, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 44. FillRectInteriorSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class FillRectInteriorSolver(BaseSolver):
    """
    Find a closed rectangular frame; find fill material (non-bg, non-frame cells);
    project fill material shape (solid rect or L) into frame interior; erase material.
    """

    @property
    def name(self) -> str:
        return "FillRectInteriorSolver"

    @property
    def priority(self) -> int:
        return 44

    @staticmethod
    def _find_frame(grid: Grid, bg: int):
        """Return (top, left, bottom, right, frame_color) or None."""
        rows, cols = len(grid), len(grid[0])
        color_cells: dict[int, list] = {}
        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v != bg:
                    color_cells.setdefault(v, []).append((r, c))
        for color, cells in color_cells.items():
            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            top, bottom = min(rs), max(rs)
            left, right = min(cs), max(cs)
            if top == bottom or left == right:
                continue
            expected_border: set = set()
            for r in range(top, bottom + 1):
                expected_border.add((r, left))
                expected_border.add((r, right))
            for c in range(left, right + 1):
                expected_border.add((top, c))
                expected_border.add((bottom, c))
            if set(cells) == expected_border:
                return (top, left, bottom, right, color)
        return None

    @staticmethod
    def _classify_shape(cells):
        """Classify fill shape as 'solid_rect' or 'L_shape'. Return (type, meta) or None."""
        if not cells:
            return None, None
        cell_set = set(cells)
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        if len(cells) == h * w:
            return 'solid_rect', {'h': h, 'w': w}
        # Try L-shape: one corner + two perpendicular arms sharing an endpoint
        for corner_r, corner_c in cells:
            same_row = [c for r2, c in cells if r2 == corner_r]
            same_col = [r2 for r2, c in cells if c == corner_c]
            if len(same_row) > 1 and len(same_col) > 1:
                arm_cells = set()
                for c in same_row:
                    arm_cells.add((corner_r, c))
                for r in same_col:
                    arm_cells.add((r, corner_c))
                if arm_cells == cell_set:
                    h_dir = 'right' if any(c > corner_c for c in same_row) else 'left'
                    v_dir = 'down' if any(r > corner_r for r in same_col) else 'up'
                    return 'L_shape', {
                        'corner': (corner_r, corner_c),
                        'h_dir': h_dir, 'v_dir': v_dir,
                        'h_len': len(same_row), 'v_len': len(same_col),
                    }
        return None, None

    @staticmethod
    def _fill_interior(grid: Grid, bg: int, frame_color: int,
                       top: int, left: int, bottom: int, right: int,
                       fill_cells, fill_color: int) -> Grid | None:
        rows, cols = len(grid), len(grid[0])
        out = [row[:] for row in grid]
        # Erase fill material
        for r, c in fill_cells:
            out[r][c] = bg
        shape_type, meta = FillRectInteriorSolver._classify_shape(fill_cells)
        if shape_type is None:
            return None
        int_rows = range(top + 1, bottom)
        int_cols = range(left + 1, right)
        if not int_rows or not int_cols:
            return None
        if shape_type == 'solid_rect':
            for r in int_rows:
                for c in int_cols:
                    out[r][c] = fill_color
        elif shape_type == 'L_shape':
            # Anchor L corner to nearest interior corner based on fill material position
            fill_rs = [r for r, c in fill_cells]
            fill_cs = [c for r, c in fill_cells]
            fill_cr = sum(fill_rs) / len(fill_rs)
            fill_cc = sum(fill_cs) / len(fill_cs)
            frame_cr = (top + bottom) / 2
            frame_cc = (left + right) / 2
            int_r = top + 1 if fill_cr < frame_cr else bottom - 1
            int_c = left + 1 if fill_cc < frame_cc else right - 1
            h_dir = meta['h_dir']
            v_dir = meta['v_dir']
            # Horizontal arm
            if h_dir == 'right':
                for c in range(int_c, right):
                    out[int_r][c] = fill_color
            else:
                for c in range(left + 1, int_c + 1):
                    out[int_r][c] = fill_color
            # Vertical arm
            if v_dir == 'down':
                for r in range(int_r, bottom):
                    out[r][int_c] = fill_color
            else:
                for r in range(top + 1, int_r + 1):
                    out[r][int_c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            bg = _bg(inp)
            frame = self._find_frame(inp, bg)
            if frame is None:
                return None
            top, left, bottom, right, fc = frame
            fill_cells = [(r, c) for r in range(len(inp)) for c in range(len(inp[0]))
                          if inp[r][c] != bg and inp[r][c] != fc]
            if not fill_cells:
                return None
            fill_clr_set = set(inp[r][c] for r, c in fill_cells)
            if len(fill_clr_set) != 1:
                return None
            fill_color = fill_clr_set.pop()
            expected = self._fill_interior(inp, bg, fc, top, left, bottom, right, fill_cells, fill_color)
            if expected is None or expected != out:
                return None
        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            frame = self._find_frame(inp, bg)
            if frame is None:
                return None
            top, left, bottom, right, fc = frame
            fill_cells = [(r, c) for r in range(len(inp)) for c in range(len(inp[0]))
                          if inp[r][c] != bg and inp[r][c] != fc]
            if not fill_cells:
                return None
            fill_clr_set = set(inp[r][c] for r, c in fill_cells)
            if len(fill_clr_set) != 1:
                return None
            fill_color = fill_clr_set.pop()
            result = self._fill_interior(inp, bg, fc, top, left, bottom, right, fill_cells, fill_color)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 45. TiledGridPropagationSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class TiledGridPropagationSolver(BaseSolver):
    """
    Grid divided by all-background separator rows/cols into an N×M tile array.
    Propagate markers: if two tiles in the same tile-row (or tile-col) both have
    a marker at the same local position, fill all tiles between them with that marker.
    """

    @property
    def name(self) -> str:
        return "TiledGridPropagationSolver"

    @property
    def priority(self) -> int:
        return 45

    @staticmethod
    def _separators(grid: Grid, bg: int):
        rows, cols = len(grid), len(grid[0])
        sep_r = [r for r in range(rows) if all(grid[r][c] == bg for c in range(cols))]
        sep_c = [c for c in range(cols) if all(grid[r][c] == bg for r in range(rows))]
        return sep_r, sep_c

    @staticmethod
    def _segments(size: int, seps: list) -> list:
        bounds = [-1] + sorted(seps) + [size]
        segs = []
        for i in range(len(bounds) - 1):
            s, e = bounds[i] + 1, bounds[i + 1] - 1
            if s <= e:
                segs.append((s, e))
        return segs

    @staticmethod
    def _propagate(grid: Grid, bg: int) -> Grid | None:
        rows, cols = len(grid), len(grid[0])
        sep_r, sep_c = TiledGridPropagationSolver._separators(grid, bg)
        if not sep_r and not sep_c:
            return None
        row_segs = TiledGridPropagationSolver._segments(rows, sep_r)
        col_segs = TiledGridPropagationSolver._segments(cols, sep_c)
        if len(row_segs) < 2 and len(col_segs) < 2:
            return None
        out = [row[:] for row in grid]
        # Build marker_map: (lr, lc) -> {(tri, tci): color}
        marker_map: dict = {}
        for tri, (r0, r1) in enumerate(row_segs):
            for tci, (c0, c1) in enumerate(col_segs):
                for lr in range(r1 - r0 + 1):
                    for lc in range(c1 - c0 + 1):
                        v = grid[r0 + lr][c0 + lc]
                        if v != bg:
                            key = (lr, lc)
                            marker_map.setdefault(key, {})[tri, tci] = v
        for (lr, lc), tile_colors in marker_map.items():
            if len(tile_colors) < 2:
                continue
            color = next(iter(tile_colors.values()))
            tiles = list(tile_colors.keys())
            # Same tile-row propagation
            row_groups: dict = {}
            for (tri, tci) in tiles:
                row_groups.setdefault(tri, []).append(tci)
            for tri, tci_list in row_groups.items():
                if len(tci_list) >= 2:
                    for tci in range(min(tci_list), max(tci_list) + 1):
                        r0, _ = row_segs[tri]
                        c0, _ = col_segs[tci]
                        out[r0 + lr][c0 + lc] = color
            # Same tile-col propagation
            col_groups: dict = {}
            for (tri, tci) in tiles:
                col_groups.setdefault(tci, []).append(tri)
            for tci, tri_list in col_groups.items():
                if len(tri_list) >= 2:
                    for tri in range(min(tri_list), max(tri_list) + 1):
                        r0, _ = row_segs[tri]
                        c0, _ = col_segs[tci]
                        out[r0 + lr][c0 + lc] = color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            bg = _bg(inp)
            sep_r, sep_c = self._separators(inp, bg)
            if not sep_r and not sep_c:
                return None
            expected = self._propagate(inp, bg)
            if expected is None or expected != out:
                return None
        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            sep_r, sep_c = self._separators(inp, bg)
            if not sep_r and not sep_c:
                return None
            result = self._propagate(inp, bg)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ─────────────────────────────────────────────────────────────────────────────
# 46. WallToTargetLineSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class WallToTargetLineSolver(BaseSolver):
    """
    Two vertical wall segments (color 2).
    Wall A has color-8 dots: draw 8-line from wall inner edge to dot, mark dot as color 4.
    Wall B mirrors: same relative rows filled entirely with 8s between walls.
    """

    @property
    def name(self) -> str:
        return "WallToTargetLineSolver"

    @property
    def priority(self) -> int:
        return 46

    WALL_COLOR = 2
    DOT_COLOR = 8
    MARK_COLOR = 4

    @staticmethod
    def _find_wall_col_groups(grid: Grid) -> list:
        """Return list of sorted column-index groups where wall color is present."""
        rows, cols = len(grid), len(grid[0])
        wc = WallToTargetLineSolver.WALL_COLOR
        wall_cols = [c for c in range(cols) if any(grid[r][c] == wc for r in range(rows))]
        if not wall_cols:
            return []
        groups, group = [], [wall_cols[0]]
        for c in wall_cols[1:]:
            if c == group[-1] + 1:
                group.append(c)
            else:
                groups.append(group)
                group = [c]
        groups.append(group)
        return groups

    @staticmethod
    def _wall_row_range(grid: Grid, wall_cols: list) -> tuple:
        rows = len(grid)
        wc = WallToTargetLineSolver.WALL_COLOR
        active = [r for r in range(rows) if any(grid[r][c] == wc for c in wall_cols)]
        return (min(active), max(active)) if active else (0, rows - 1)

    @staticmethod
    def _apply(grid: Grid, bg: int) -> Grid | None:
        rows, cols = len(grid), len(grid[0])
        wc = WallToTargetLineSolver.WALL_COLOR
        dc = WallToTargetLineSolver.DOT_COLOR
        mc = WallToTargetLineSolver.MARK_COLOR
        groups = WallToTargetLineSolver._find_wall_col_groups(grid)
        if len(groups) < 2:
            return None
        # Identify which wall has dots (Wall A) and which doesn't (Wall B)
        wall_a, wall_b = None, None
        for g in groups:
            r0, r1 = WallToTargetLineSolver._wall_row_range(grid, g)
            has_dots = any(grid[r][c] == dc
                          for r in range(r0, r1 + 1)
                          for c in range(cols)
                          if c not in g)
            if has_dots:
                wall_a = g
            else:
                wall_b = g
        if wall_a is None or wall_b is None:
            # Fallback: assign first as A, second as B
            wall_a, wall_b = groups[0], groups[1]
        a_r0, a_r1 = WallToTargetLineSolver._wall_row_range(grid, wall_a)
        b_r0, b_r1 = WallToTargetLineSolver._wall_row_range(grid, wall_b)
        # Wall A inner edge (col facing interior)
        mid = cols // 2
        a_inner = wall_a[-1] + 1 if wall_a[-1] < mid else wall_a[0] - 1
        b_inner = wall_b[-1] + 1 if wall_b[-1] < mid else wall_b[0] - 1
        out = [row[:] for row in grid]
        # Collect dot offsets from Wall A
        dot_offsets = []
        for r in range(a_r0, a_r1 + 1):
            for c in range(cols):
                if c not in wall_a and grid[r][c] == dc:
                    dot_offsets.append((r - a_r0, r, c))
        if not dot_offsets:
            return None
        # Process Wall A: line from inner edge to dot, mark dot
        for (offset, r, c_dot) in dot_offsets:
            if a_inner <= c_dot:
                for c in range(a_inner, c_dot):
                    out[r][c] = dc
                out[r][c_dot] = mc
            else:
                for c in range(c_dot + 1, a_inner + 1):
                    out[r][c] = dc
                out[r][c_dot] = mc
        # Process Wall B: fill corresponding relative rows with 8s
        # Determine fill range for Wall B (from inner edge toward Wall A)
        if b_inner < a_inner:
            b_fill_start, b_fill_end = b_inner, a_inner - 1
        else:
            b_fill_start, b_fill_end = a_inner + 1, b_inner
        for (offset, r_a, c_dot) in dot_offsets:
            r_b = b_r0 + offset
            if 0 <= r_b < rows:
                for c in range(b_fill_start, b_fill_end + 1):
                    out[r_b][c] = dc
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            bg = _bg(inp)
            if not any(inp[r][c] == self.WALL_COLOR
                       for r in range(len(inp)) for c in range(len(inp[0]))):
                return None
            expected = self._apply(inp, bg)
            if expected is None or expected != out:
                return None
        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            result = self._apply(inp, bg)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 47 – MirrorTileSolver
# ---------------------------------------------------------------------------

def _all_transforms(inp: Grid) -> dict:
    """Return all 6 named transforms of a grid."""
    r90 = rotate90(inp)
    r180 = flip_h(flip_v(inp))
    r270 = rotate90(rotate90(rotate90(inp)))
    return {
        "orig": inp,
        "fh": flip_h(inp),
        "fv": flip_v(inp),
        "r90": r90,
        "r180": r180,
        "r270": r270,
    }


def _identify_transform(inp: Grid, target: Grid) -> str | None:
    """Return the name of the transform of inp that equals target, or None."""
    for name, t in _all_transforms(inp).items():
        if t == target:
            return name
    return None


def _build_grid_from_arrangement(inp: Grid, arrangement: list) -> Grid:
    """Build output grid by tiling inp according to arrangement (list of lists of transform names)."""
    H, W = len(inp), len(inp[0])
    transforms = _all_transforms(inp)
    rows_out = len(arrangement)
    cols_out = len(arrangement[0])
    out = []
    for ti in range(rows_out):
        for r in range(H):
            row = []
            for tj in range(cols_out):
                tile = transforms[arrangement[ti][tj]]
                row.extend(tile[r])
            out.append(row)
    return out


def _learn_tile_arrangement(pairs: list, rows: int, cols: int) -> list | None:
    """
    Learn a consistent rows×cols tiling arrangement from training pairs.
    Returns arrangement (list[list[str]]) or None if inconsistent/unidentifiable.
    For each tile position, collects ALL transform names that produce the correct
    tile (handles symmetric inputs where fh==r90 etc.) and searches combinations
    that validate consistently across ALL pairs.
    """
    _ALL_NAMES = ["orig", "fh", "fv", "r90", "r180", "r270"]

    def _all_matching_names(inp, tile):
        """All transform names whose result matches tile."""
        H, W = len(inp), len(inp[0])
        matches = []
        candidates = _all_transforms(inp)
        for name in _ALL_NAMES:
            if candidates[name] == tile:
                matches.append(name)
        return matches

    import itertools

    # First pass: collect per-position candidates from EVERY pair
    # A candidate name for position (ti,tj) is valid only if it works for ALL pairs
    # We build: per_pos_candidates[ti][tj] = set of names valid across all pairs so far
    per_pos = None

    for pair in pairs:
        inp, out = pair["input"], pair["output"]
        H, W = len(inp), len(inp[0])
        if len(out) != rows * H or len(out[0]) != cols * W:
            return None
        pair_candidates = []
        for ti in range(rows):
            row_c = []
            for tj in range(cols):
                tile = [out[ti * H + r][tj * W: tj * W + W] for r in range(H)]
                names = _all_matching_names(inp, tile)
                if not names:
                    return None
                row_c.append(set(names))
            pair_candidates.append(row_c)

        if per_pos is None:
            per_pos = pair_candidates
        else:
            for ti in range(rows):
                for tj in range(cols):
                    per_pos[ti][tj] &= pair_candidates[ti][tj]
                    if not per_pos[ti][tj]:
                        return None

    if per_pos is None:
        return None

    # Build flat list of candidate sets in row-major order
    flat_pos = [(ti, tj) for ti in range(rows) for tj in range(cols)]
    flat_cands = [sorted(per_pos[ti][tj]) for ti, tj in flat_pos]

    # Search all combinations for one that validates against ALL pairs
    for combo in itertools.product(*flat_cands):
        arr = [[None] * cols for _ in range(rows)]
        for idx, (ti, tj) in enumerate(flat_pos):
            arr[ti][tj] = combo[idx]
        # Validate against all pairs
        ok = True
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            predicted = _build_grid_from_arrangement(inp, arr)
            if predicted != out:
                ok = False
                break
        if ok:
            return arr

    return None


@register_solver
class MirrorTileSolver(BaseSolver):
    """
    Learns a consistent 2×2 tiling arrangement of transforms (orig/fh/fv/r90/r180/r270)
    from training pairs and applies it to test inputs.
    Targets: 0c786b71, 3af2c5a8, 62c24649, 67e8384a, 833dafe3, 7953d61e.
    """

    @property
    def name(self) -> str:
        return "MirrorTileSolver"

    @property
    def priority(self) -> int:
        return 47

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        arrangement = _learn_tile_arrangement(ctx.train_pairs, 2, 2)
        if arrangement is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = _build_grid_from_arrangement(inp, arrangement)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 48 – SelfTileZeroSolver
# ---------------------------------------------------------------------------

@register_solver
class SelfTileZeroSolver(BaseSolver):
    """
    Square input H×H → output H²×H².
    For tile position (tr, tc):
      inp[tr][tc] != 0  → tile = full input
      inp[tr][tc] == 0  → tile = H×H all-zeros
    Targets: 007bbfb7, 5b6cbef5.
    """

    @property
    def name(self) -> str:
        return "SelfTileZeroSolver"

    @property
    def priority(self) -> int:
        return 48

    @staticmethod
    def _apply(inp: Grid) -> Grid | None:
        H = len(inp)
        if H == 0 or len(inp[0]) != H:
            return None
        zero_tile = [[0] * H for _ in range(H)]
        out = []
        for tr in range(H):
            row_blocks = []
            for tc in range(H):
                tile = inp if inp[tr][tc] != 0 else zero_tile
                row_blocks.append(tile)
            # merge H rows of H blocks
            for r in range(H):
                out.append([cell for blk in row_blocks for cell in blk[r]])
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H = len(inp)
            if len(inp[0]) != H:
                return None
            expected = self._apply(inp)
            if expected is None or expected != out:
                return None
            if len(out) != H * H or len(out[0]) != H * H:
                return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply(inp)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 49 – RowAltFlipTileSolver
# ---------------------------------------------------------------------------

@register_solver
class RowAltFlipTileSolver(BaseSolver):
    """
    3×3 tiling:
      Tile-rows 0,2  → input repeated 3× horizontally (no flip)
      Tile-row  1    → flip_h(input) repeated 3× horizontally
    Learns the alternation pattern (even=orig/odd=fh or even=fh/odd=orig)
    and the number of tile-columns from training pairs.
    Targets: 00576224.
    """

    @property
    def name(self) -> str:
        return "RowAltFlipTileSolver"

    @property
    def priority(self) -> int:
        return 49

    @staticmethod
    def _apply(inp: Grid, tcols: int, even_is_orig: bool) -> Grid:
        fh = flip_h(inp)
        out = []
        H, W = len(inp), len(inp[0])
        for tr in range(3):
            tile = inp if (tr % 2 == 0) == even_is_orig else fh
            for r in range(H):
                out.append(tile[r] * tcols)
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        chosen_tcols = None
        chosen_even = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H, W = len(inp), len(inp[0])
            if len(out) != 3 * H:
                return None
            out_w = len(out[0])
            if out_w % W != 0:
                return None
            tcols = out_w // W
            if chosen_tcols is None:
                chosen_tcols = tcols
            elif chosen_tcols != tcols:
                return None
            found = False
            for even_is_orig in (True, False):
                candidate = self._apply(inp, tcols, even_is_orig)
                if candidate == out:
                    if chosen_even is None:
                        chosen_even = even_is_orig
                    elif chosen_even != even_is_orig:
                        return None
                    found = True
                    break
            if not found:
                return None
        if chosen_tcols is None or chosen_even is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply(inp, chosen_tcols, chosen_even)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 50 – FixedRecolorSolver
# ---------------------------------------------------------------------------

@register_solver
class FixedRecolorSolver(BaseSolver):
    """
    Learn a consistent bijective color mapping from all training pairs,
    then apply the same mapping to every test input.
    Grid size is unchanged.
    Targets: 63613498 (9→5), aabf363d (3→6, 6→0).
    """

    @property
    def name(self) -> str:
        return "FixedRecolorSolver"

    @property
    def priority(self) -> int:
        return 50

    @staticmethod
    def _learn_mapping(pairs: list) -> dict | None:
        """Return color→color mapping consistent across all pairs, or None."""
        mapping = {}
        changed_sets = []
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            changed = set()
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    src, dst = inp[r][c], out[r][c]
                    if src in mapping:
                        if mapping[src] != dst:
                            return None
                    else:
                        mapping[src] = dst
                    if src != dst:
                        changed.add(src)
            changed_sets.append(frozenset(changed))
        # Must change at least one color
        if all(k == v for k, v in mapping.items()):
            return None
        # All pairs must change the same set of source colors
        if len(set(changed_sets)) > 1:
            return None
        return mapping

    @staticmethod
    def _apply_mapping(grid: Grid, mapping: dict) -> Grid:
        return [[mapping.get(cell, cell) for cell in row] for row in grid]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        mapping = self._learn_mapping(ctx.train_pairs)
        if mapping is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply_mapping(inp, mapping)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 51 – Grid3x3TileSolver
# ---------------------------------------------------------------------------

@register_solver
class Grid3x3TileSolver(BaseSolver):
    """
    Learns a consistent 3×3 tiling arrangement of transforms from training pairs.
    Targets: c48954c1 ([[r180,fv,r180],[fh,orig,fh],[r180,fv,r180]]).
    """

    @property
    def name(self) -> str:
        return "Grid3x3TileSolver"

    @property
    def priority(self) -> int:
        return 51

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        arrangement = _learn_tile_arrangement(ctx.train_pairs, 3, 3)
        if arrangement is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = _build_grid_from_arrangement(inp, arrangement)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 52 – ModeInSection5Solver
# ---------------------------------------------------------------------------

@register_solver
class ModeInSection5Solver(BaseSolver):
    """
    Grid structure: non-zero rows / all-5s separator row / all-zero rows.
    Output = input but (last_row, center_col) replaced by the mode of non-zero non-5 cells.
    Targets: 27a77e38.
    """

    @property
    def name(self) -> str:
        return "ModeInSection5Solver"

    @property
    def priority(self) -> int:
        return 52

    @staticmethod
    def _find_five_row(grid: Grid) -> int | None:
        W = len(grid[0]) if grid else 0
        for r, row in enumerate(grid):
            if all(cell == 5 for cell in row):
                return r
        return None

    @staticmethod
    def _apply(grid: Grid) -> Grid | None:
        H, W = len(grid), len(grid[0])
        five_row = ModeInSection5Solver._find_five_row(grid)
        if five_row is None:
            return None
        # Collect non-zero non-5 values above the 5-row
        top_vals = [grid[r][c] for r in range(five_row)
                    for c in range(W) if grid[r][c] not in (0, 5)]
        if not top_vals:
            return None
        mode_val = collections.Counter(top_vals).most_common(1)[0][0]
        last_row = H - 1
        center_col = W // 2
        # last_row must be in zero section
        if grid[last_row][center_col] != 0:
            return None
        out = [row[:] for row in grid]
        out[last_row][center_col] = mode_val
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            if self._find_five_row(inp) is None:
                return None
            expected = self._apply(inp)
            if expected is None or expected != out:
                return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply(inp)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 53 – MoveTowardTargetSolver
# ---------------------------------------------------------------------------

@register_solver
class MoveTowardTargetSolver(BaseSolver):
    """
    Exactly two non-background singleton colors exist.
    One color (mover) takes one Chebyshev step toward the other (target).
    The rule 'which color is mover' is learned from training pairs.
    Targets: dc433765.
    """

    @property
    def name(self) -> str:
        return "MoveTowardTargetSolver"

    @property
    def priority(self) -> int:
        return 53

    @staticmethod
    def _find_singletons(grid: Grid, bg: int):
        """Return list of (color, row, col) for each non-bg color with exactly 1 cell."""
        from collections import Counter
        flat = [(c, r, col) for r, row in enumerate(grid)
                for col, c in enumerate(row) if c != bg]
        counts: Counter = Counter(c for c, r, col in flat)
        return [(c, r, col) for c, r, col in flat if counts[c] == 1]

    @staticmethod
    def _chebyshev_step(r1, c1, r2, c2):
        """One step from (r1,c1) toward (r2,c2) in Chebyshev metric."""
        dr = r2 - r1
        dc = c2 - c1
        sr = (1 if dr > 0 else -1) if dr != 0 else 0
        sc = (1 if dc > 0 else -1) if dc != 0 else 0
        return r1 + sr, c1 + sc

    @staticmethod
    def _apply_move(grid: Grid, mover_color: int, target_color: int, bg: int) -> Grid | None:
        singletons = MoveTowardTargetSolver._find_singletons(grid, bg)
        color_map = {c: (r, col) for c, r, col in singletons}
        if mover_color not in color_map or target_color not in color_map:
            return None
        mr, mc = color_map[mover_color]
        tr, tc = color_map[target_color]
        nr, nc = MoveTowardTargetSolver._chebyshev_step(mr, mc, tr, tc)
        H, W = len(grid), len(grid[0])
        if not (0 <= nr < H and 0 <= nc < W):
            return None
        out = [row[:] for row in grid]
        out[mr][mc] = bg
        out[nr][nc] = mover_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        # Learn which color is mover vs target
        mover_color = None
        target_color = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            bg = _bg(inp)
            singletons = self._find_singletons(inp, bg)
            if len(singletons) != 2:
                return None
            (c1, r1, col1), (c2, r2, col2) = singletons
            # Try both assignments
            for mv, tg in [(c1, c2), (c2, c1)]:
                candidate = self._apply_move(inp, mv, tg, bg)
                if candidate == out:
                    if mover_color is None:
                        mover_color, target_color = mv, tg
                    elif mover_color != mv or target_color != tg:
                        return None  # inconsistent across pairs
                    break
            else:
                return None  # neither assignment worked
        if mover_color is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply_move(inp, mover_color, target_color, _bg(inp))
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# Priority 54 – IsolatedCellPruneSolver
# ---------------------------------------------------------------------------

@register_solver
class IsolatedCellPruneSolver(BaseSolver):
    """
    Cells of color A with zero 4-connected color-A neighbors → become color B.
    Both A and B are learned from training pairs.
    Targets: aedd82e4.
    """

    @property
    def name(self) -> str:
        return "IsolatedCellPruneSolver"

    @property
    def priority(self) -> int:
        return 54

    @staticmethod
    def _isolated_cells(grid: Grid, color: int):
        """Return set of (r,c) positions of 'color' with no 4-connected same-color neighbor."""
        H, W = len(grid), len(grid[0])
        result = set()
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color:
                    neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                    if not any(0<=nr<H and 0<=nc<W and grid[nr][nc]==color
                               for nr, nc in neighbors):
                        result.add((r, c))
        return result

    @staticmethod
    def _apply(grid: Grid, prune_color: int, fill_color: int) -> Grid:
        isolated = IsolatedCellPruneSolver._isolated_cells(grid, prune_color)
        out = [row[:] for row in grid]
        for r, c in isolated:
            out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        prune_color = None
        fill_color = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if _dims(inp) != _dims(out):
                return None
            # Find changed cells: cells where inp != out
            H, W = _dims(inp)
            changes = [(r, c, inp[r][c], out[r][c])
                       for r in range(H) for c in range(W) if inp[r][c] != out[r][c]]
            if not changes:
                return None  # identity pair, skip
            # All changed cells must go from same prune_color → fill_color
            pair_prune = set(inp[r][c] for r, c, _, _ in changes)
            pair_fill  = set(out[r][c] for r, c, _, _ in changes)
            if len(pair_prune) != 1 or len(pair_fill) != 1:
                return None
            pc = pair_prune.pop()
            fc = pair_fill.pop()
            # Validate: every changed cell must be isolated in input
            isolated = IsolatedCellPruneSolver._isolated_cells(inp, pc)
            if not all((r, c) in isolated for r, c, _, _ in changes):
                return None
            # No extra cells should have changed
            expected = IsolatedCellPruneSolver._apply(inp, pc, fc)
            if expected != out:
                return None
            if prune_color is None:
                prune_color, fill_color = pc, fc
            elif prune_color != pc or fill_color != fc:
                return None
        if prune_color is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply(inp, prune_color, fill_color)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p55 — ConstantShiftSolver
# All non-background cells shift by a constant (dr, dc) vector learned from
# training pairs.  Works on any grid size.
# ---------------------------------------------------------------------------
@register_solver
class ConstantShiftSolver(BaseSolver):
    """Every non-bg cell shifts by a constant (dr, dc) learned from training."""

    @property
    def name(self) -> str:
        return "ConstantShiftSolver"

    @property
    def priority(self) -> int:
        return 55

    @staticmethod
    def _nonbg(grid, bg):
        H, W = len(grid), len(grid[0])
        return [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]

    @staticmethod
    def _apply_shift(grid, bg, dr, dc):
        H, W = len(grid), len(grid[0])
        out = [[bg] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = grid[r][c]
        return out

    def solve(self, ctx: TaskContext):
        if not ctx.train_pairs:
            return None
        bg = _bg(ctx.train_pairs[0]["input"])
        dr_learned = dc_learned = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H, W = len(inp), len(inp[0])
            if len(out) != H or len(out[0]) != W:
                return None
            cells_in  = self._nonbg(inp, bg)
            cells_out = self._nonbg(out, bg)
            if not cells_in or len(cells_in) != len(cells_out):
                return None
            # Sort both sets to attempt consistent pairing
            cells_in.sort()
            cells_out.sort()
            shifts = set()
            for (r1, c1, v1), (r2, c2, v2) in zip(cells_in, cells_out):
                if v1 != v2:
                    return None
                shifts.add((r2 - r1, c2 - c1))
            if len(shifts) != 1:
                return None
            dr, dc = shifts.pop()
            if dr_learned is None:
                dr_learned, dc_learned = dr, dc
            elif (dr, dc) != (dr_learned, dc_learned):
                return None
        if dr_learned is None or (dr_learned == 0 and dc_learned == 0):
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply_shift(inp, bg, dr_learned, dc_learned)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p56 — AntiDiagSegmentSolver
# All non-bg cells lie on a single anti-diagonal (r+c = k).
# A contiguous segment of N cells moves so new_r_min = old_r_min − N − 1,
# new_r_max = old_r_min − 1  (grows by 1, advances toward top-right corner).
# Validates against ALL training pairs before applying to test.
# ---------------------------------------------------------------------------
@register_solver
class AntiDiagSegmentSolver(BaseSolver):
    """Anti-diagonal segment grows by 1 and advances toward top-right corner."""

    @property
    def name(self) -> str:
        return "AntiDiagSegmentSolver"

    @property
    def priority(self) -> int:
        return 56

    @staticmethod
    def _get_segment(grid, bg):
        """Return (k, r_min, r_max, color) if all non-bg cells on one anti-diag."""
        cells = [(r, c, grid[r][c]) for r in range(len(grid))
                 for c in range(len(grid[0])) if grid[r][c] != bg]
        if not cells:
            return None
        colors = set(v for _, _, v in cells)
        if len(colors) != 1:
            return None
        k_vals = set(r + c for r, c, _ in cells)
        if len(k_vals) != 1:
            return None
        k = k_vals.pop()
        rows = sorted(r for r, c, _ in cells)
        # Must be contiguous
        if rows != list(range(rows[0], rows[-1] + 1)):
            return None
        return k, rows[0], rows[-1], colors.pop()

    @staticmethod
    def _apply_rule(grid, bg, k, r_min, r_max, color):
        H, W = len(grid), len(grid[0])
        N = r_max - r_min + 1
        new_r_min = r_min - N - 1
        new_r_max = r_min - 1
        if new_r_min < 0:
            return None
        out = [[bg] * W for _ in range(H)]
        for r in range(new_r_min, new_r_max + 1):
            c = k - r
            if 0 <= c < W:
                out[r][c] = color
        return out

    def solve(self, ctx: TaskContext):
        if not ctx.train_pairs:
            return None
        bg = _bg(ctx.train_pairs[0]["input"])
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            seg_in  = self._get_segment(inp, bg)
            seg_out = self._get_segment(out, bg)
            if seg_in is None or seg_out is None:
                return None
            k_in, r_min_in, r_max_in, color_in = seg_in
            k_out, r_min_out, r_max_out, color_out = seg_out
            # Must stay on same anti-diagonal, same color
            if k_in != k_out or color_in != color_out:
                return None
            N_in = r_max_in - r_min_in + 1
            expected_r_min = r_min_in - N_in - 1
            expected_r_max = r_min_in - 1
            if r_min_out != expected_r_min or r_max_out != expected_r_max:
                return None
        preds = []
        for inp in ctx.test_inputs:
            seg = self._get_segment(inp, bg)
            if seg is None:
                return None
            k, r_min, r_max, color = seg
            result = self._apply_rule(inp, bg, k, r_min, r_max, color)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p57 — UniformRowClassifierSolver
# For each row: if all cells are the same color → replace with [A, A, ..., A],
# else → replace with [B, B, ..., B].  A and B learned from training pairs.
# ---------------------------------------------------------------------------
@register_solver
class UniformRowClassifierSolver(BaseSolver):
    """Each row: uniform color → row of A; non-uniform → row of B."""

    @property
    def name(self) -> str:
        return "UniformRowClassifierSolver"

    @property
    def priority(self) -> int:
        return 57

    @staticmethod
    def _apply(grid, uniform_color, nonuniform_color, W_out):
        result = []
        for row in grid:
            if len(set(row)) == 1:
                result.append([uniform_color] * W_out)
            else:
                result.append([nonuniform_color] * W_out)
        return result

    def solve(self, ctx: TaskContext):
        if not ctx.train_pairs:
            return None
        uniform_color = nonuniform_color = None
        W_out = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H, W = len(inp), len(inp[0])
            if len(out) != H:
                return None
            W_out_pair = len(out[0])
            if W_out is None:
                W_out = W_out_pair
            elif W_out != W_out_pair:
                return None
            for r in range(H):
                row_in  = inp[r]
                row_out = out[r]
                if len(set(row_out)) != 1:
                    return None  # output row must be uniform
                out_val = row_out[0]
                if len(set(row_in)) == 1:
                    # uniform input row
                    if uniform_color is None:
                        uniform_color = out_val
                    elif uniform_color != out_val:
                        return None
                else:
                    # non-uniform input row
                    if nonuniform_color is None:
                        nonuniform_color = out_val
                    elif nonuniform_color != out_val:
                        return None
        if uniform_color is None or nonuniform_color is None or W_out is None:
            return None
        if uniform_color == nonuniform_color:
            return None
        preds = []
        for inp in ctx.test_inputs:
            result = self._apply(inp, uniform_color, nonuniform_color, W_out)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p58 — DiagonalFillSolver
# Pattern: col 0 = color X throughout; anti-diagonal cells (r, W-1-r) for
# r=0..H-2 = color_diag; last row = [X, color_fill, color_fill, ...].
# Colors X, color_diag, color_fill learned from training.
# ---------------------------------------------------------------------------
@register_solver
class DiagonalFillSolver(BaseSolver):
    """Left col=X, anti-diagonal of color_diag, last row=[X, fill, fill, ...]."""

    @property
    def name(self) -> str:
        return "DiagonalFillSolver"

    @property
    def priority(self) -> int:
        return 58

    @staticmethod
    def _apply(H, W, col_color, diag_color, fill_color):
        out = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                out[r][c] = 0  # background (will be overwritten)
        # Fill with background first (use col_color as base? No, use 0)
        # Actually: determine background from the expected pattern
        # col 0 = col_color, anti-diag = diag_color, last row = [col_color, fill...]
        # everything else = background (0)
        bg = 0
        for r in range(H):
            for c in range(W):
                out[r][c] = bg
        # col 0
        for r in range(H):
            out[r][0] = col_color
        # anti-diagonal r + c = W-1, for r=0..H-2
        for r in range(H - 1):
            c = W - 1 - r
            if 0 <= c < W:
                out[r][c] = diag_color
        # last row
        out[H - 1][0] = col_color
        for c in range(1, W):
            out[H - 1][c] = fill_color
        return out

    def solve(self, ctx: TaskContext):
        if not ctx.train_pairs:
            return None
        # col_color is read from input's col-0 (varies per pair).
        # diag_color and fill_color are constants learned from training.
        diag_color = fill_color = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H, W = len(out), len(out[0])
            H_in, W_in = len(inp), len(inp[0])
            # Only apply to square output grids
            if H != W or H < 2 or W < 2:
                return None
            # col_color for this pair: from input's column 0
            inp_col0 = set(inp[r][0] for r in range(H_in))
            if len(inp_col0) != 1:
                return None
            cc = inp_col0.pop()
            # Validate output col0 matches
            out_col0 = set(out[r][0] for r in range(H))
            if out_col0 != {cc}:
                return None
            # Learn diag_color: out[r][W-1-r] for r=0..H-2
            diag_vals = set(out[r][W - 1 - r] for r in range(H - 1) if 0 <= W - 1 - r < W)
            if not diag_vals or len(diag_vals) != 1:
                return None
            dc = diag_vals.pop()
            # Learn fill_color: out[H-1][1..W-1]
            fill_vals = set(out[H - 1][c] for c in range(1, W))
            if len(fill_vals) != 1:
                return None
            fc = fill_vals.pop()
            if diag_color is None:
                diag_color, fill_color = dc, fc
            else:
                if dc != diag_color or fc != fill_color:
                    return None
            # Validate reconstruction
            reconstructed = self._apply(H, W, cc, diag_color, fill_color)
            if reconstructed != out:
                return None
        if diag_color is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            H, W = len(inp), len(inp[0])
            # col_color from test input col0
            inp_col0 = set(inp[r][0] for r in range(H))
            if len(inp_col0) != 1:
                return None
            cc = inp_col0.pop()
            result = self._apply(H, W, cc, diag_color, fill_color)
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p59 — NormalizedShapeLookupSolver
# For tasks where output is 1×1 (or very small fixed size).
# Normalize input by replacing all non-bg cells with 1; look up the resulting
# pattern in a table built from training pairs; return the matched output.
# Also handles bilateral-symmetry classification (44f52bb0).
# ---------------------------------------------------------------------------
@register_solver
class NormalizedShapeLookupSolver(BaseSolver):
    """Normalize input shape (non-bg→1) then look up output from training table."""

    @property
    def name(self) -> str:
        return "NormalizedShapeLookupSolver"

    @property
    def priority(self) -> int:
        return 59

    @staticmethod
    def _normalize(grid, bg):
        return tuple(
            tuple(0 if cell == bg else 1 for cell in row)
            for row in grid
        )

    def solve(self, ctx: TaskContext):
        if not ctx.train_pairs:
            return None
        # Only apply when ALL training outputs are exactly 1×1
        for pair in ctx.train_pairs:
            oh, ow = _dims(pair["output"])
            if oh != 1 or ow != 1:
                return None
        bg = _bg(ctx.train_pairs[0]["input"])
        lookup = {}
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            key = self._normalize(inp, bg)
            out_t = tuple(tuple(r) for r in out)
            if key in lookup and lookup[key] != out_t:
                return None  # ambiguous
            lookup[key] = out_t
        if not lookup:
            return None
        preds = []
        for inp in ctx.test_inputs:
            key = self._normalize(inp, bg)
            if key not in lookup:
                return None
            result = [list(r) for r in lookup[key]]
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p60 — BilateralSymmetrySolver
# For 1×1 output tasks: learn two values A and B from training where
# A = output when input is horizontally symmetric, B otherwise.
# Optionally also checks vertical symmetry as secondary signal.
# Targets: 44f52bb0.
# ---------------------------------------------------------------------------
@register_solver
class BilateralSymmetrySolver(BaseSolver):
    """Maps h-symmetric inputs → val_A, non-symmetric → val_B (learned)."""

    @property
    def name(self) -> str:
        return "BilateralSymmetrySolver"

    @property
    def priority(self) -> int:
        return 60

    @staticmethod
    def _is_h_sym(grid):
        """Return True if grid == flip_h(grid)."""
        return all(grid[r] == list(reversed(grid[r])) for r in range(len(grid)))

    @staticmethod
    def _is_v_sym(grid):
        """Return True if grid == flip_v(grid)."""
        H = len(grid)
        return all(grid[r] == grid[H - 1 - r] for r in range(H))

    def solve(self, ctx: TaskContext):
        if not ctx.train_pairs:
            return None
        # Only apply when ALL training outputs are 1×1
        for pair in ctx.train_pairs:
            oh, ow = _dims(pair["output"])
            if oh != 1 or ow != 1:
                return None
        sym_val = asym_val = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            val = out[0][0]
            is_sym = self._is_h_sym(inp)
            if is_sym:
                if sym_val is None:
                    sym_val = val
                elif sym_val != val:
                    return None
            else:
                if asym_val is None:
                    asym_val = val
                elif asym_val != val:
                    return None
        # Need both values to be learned
        if sym_val is None or asym_val is None:
            return None
        if sym_val == asym_val:
            return None
        preds = []
        for inp in ctx.test_inputs:
            val = sym_val if self._is_h_sym(inp) else asym_val
            preds.append(_mk([[val]], self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p61 – FourQuadrantExpandSolver
# Pattern: output is 2H×2W constructed from 4 quadrants of the input content.
# bg detection via corners; if bg==0, strip zero-border rows/cols to get content.
# 4-quad rule: TL=content[r][c], TR=content[0][c-W'], BL=content[r-H'][0], BR=content[r-H'][c-W']
# If bg==0: embed 2H'×2W' result in 2H×2W frame filled with content[0][0]
# If bg!=0: apply 4-quad directly to full input → 2H×2W result
# Verified on all 3 train pairs of 539a4f51.
# ---------------------------------------------------------------------------
@register_solver
class FourQuadrantExpandSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "FourQuadrantExpandSolver"

    @property
    def priority(self) -> int:
        return 61

    @staticmethod
    def _detect_bg(grid) -> int:
        corners = [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]]
        return collections.Counter(corners).most_common(1)[0][0]

    @staticmethod
    def _strip_zero_border(grid):
        """Iteratively strip all-zero rows/cols from all 4 sides."""
        g = [list(r) for r in grid]
        while g and all(v == 0 for v in g[0]):
            g = g[1:]
        while g and all(v == 0 for v in g[-1]):
            g = g[:-1]
        while g and all(row[0] == 0 for row in g):
            g = [row[1:] for row in g]
        while g and all(row[-1] == 0 for row in g):
            g = [row[:-1] for row in g]
        return g

    @staticmethod
    def _apply_four_quad(content):
        CH = len(content)
        CW = len(content[0])
        RH = 2 * CH
        RW = 2 * CW
        result = [[0] * RW for _ in range(RH)]
        for r in range(RH):
            for c in range(RW):
                if r < CH and c < CW:
                    result[r][c] = content[r][c]
                elif r < CH and c >= CW:
                    result[r][c] = content[0][c - CW]
                elif r >= CH and c < CW:
                    result[r][c] = content[r - CH][0]
                else:
                    result[r][c] = content[r - CH][c - CW]
        return result

    def _predict(self, inp):
        H, W = _dims(inp)
        bg = self._detect_bg(inp)
        if bg == 0:
            content = self._strip_zero_border(inp)
            if not content or not content[0]:
                return None
            CH, CW = len(content), len(content[0])
            inner = self._apply_four_quad(content)
            fill = content[0][0]
            RH, RW = 2 * H, 2 * W
            result = [[fill] * RW for _ in range(RH)]
            for r in range(2 * CH):
                for c in range(2 * CW):
                    result[r][c] = inner[r][c]
        else:
            result = self._apply_four_quad(inp)
        return result

    def solve(self, ctx: TaskContext):
        # Validate on training pairs
        for pair in ctx.train_pairs:
            inp, expected = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(expected)
            if eH != 2 * H or eW != 2 * W:
                return None
            got = self._predict(inp)
            if got is None or got != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            got = self._predict(inp)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p62 — DiagonalStreakSolver
# For task d13f3404 and similar.
# Rule: output is 2H×2W; for each non-zero at (r,c) in input,
#       emit that value along the main diagonal of the output starting at (r,c):
#       i.e. fill cells (r+k, c+k) for k=0,1,... while within bounds.
# Constraint: each diagonal index (r-c) may have at most one distinct non-zero.
# ---------------------------------------------------------------------------
@register_solver
class DiagonalStreakSolver(BaseSolver):
    @property
    def name(self): return "DiagonalStreakSolver"
    @property
    def priority(self): return 62

    def _predict(self, inp):
        H, W = _dims(inp)
        RH, RW = 2 * H, 2 * W
        # Collect non-zero cells; validate one non-zero per diagonal
        diag_val = {}
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v == 0:
                    continue
                d = r - c
                if d in diag_val and diag_val[d] != v:
                    return None  # conflicting values on same diagonal
                diag_val[d] = v
        # Build output
        result = [[0] * RW for _ in range(RH)]
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v == 0:
                    continue
                k = 0
                while r + k < RH and c + k < RW:
                    result[r + k][c + k] = v
                    k += 1
        return result

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp, expected = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(expected)
            if eH != 2 * H or eW != 2 * W:
                return None
            got = self._predict(inp)
            if got is None or got != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            got = self._predict(inp)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p63 — InvertAndTileSolver
# For task 48131b3c and similar.
# Rule: exactly 2 unique colors; bg = most common; fg = other.
#       Invert grid (swap bg↔fg); tile inverted grid 2×2 → 2H×2W output.
# ---------------------------------------------------------------------------
@register_solver
class InvertAndTileSolver(BaseSolver):
    @property
    def name(self): return "InvertAndTileSolver"
    @property
    def priority(self): return 63

    def _predict(self, inp):
        H, W = _dims(inp)
        flat = [inp[r][c] for r in range(H) for c in range(W)]
        uniq = list(set(flat))
        if len(uniq) != 2:
            return None
        cnt = collections.Counter(flat)
        bg = cnt.most_common(1)[0][0]
        fg = uniq[0] if uniq[1] == bg else uniq[1]
        # Invert
        inv = [[fg if inp[r][c] == bg else bg for c in range(W)] for r in range(H)]
        # Tile 2×2
        RH, RW = 2 * H, 2 * W
        result = [[0] * RW for _ in range(RH)]
        for tr in range(2):
            for tc in range(2):
                for r in range(H):
                    for c in range(W):
                        result[tr * H + r][tc * W + c] = inv[r][c]
        return result

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp, expected = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(expected)
            if eH != 2 * H or eW != 2 * W:
                return None
            got = self._predict(inp)
            if got is None or got != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            got = self._predict(inp)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p64 — OddCellExpand4xSolver
# For task 46f33fce and similar.
# Rule: all non-zero cells at ODD row AND ODD col (0-based).
#       Output is 2H×2W (initialized to 0).
#       Each non-zero at (r,c) → 4×4 block at rows 2*(r-1)..2*(r-1)+3,
#                                                  cols 2*(c-1)..2*(c-1)+3.
# ---------------------------------------------------------------------------
@register_solver
class OddCellExpand4xSolver(BaseSolver):
    @property
    def name(self): return "OddCellExpand4xSolver"
    @property
    def priority(self): return 64

    def _predict(self, inp):
        H, W = _dims(inp)
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0:
                    if r % 2 != 1 or c % 2 != 1:
                        return None  # non-zero at even index → reject
        RH, RW = 2 * H, 2 * W
        result = [[0] * RW for _ in range(RH)]
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v == 0:
                    continue
                br = 2 * (r - 1)
                bc = 2 * (c - 1)
                for dr in range(4):
                    for dc in range(4):
                        rr, cc = br + dr, bc + dc
                        if 0 <= rr < RH and 0 <= cc < RW:
                            result[rr][cc] = v
        return result

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp, expected = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(expected)
            if eH != 2 * H or eW != 2 * W:
                return None
            got = self._predict(inp)
            if got is None or got != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            got = self._predict(inp)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p65 — NonZeroTileInvertSolver
# For task 0692e18c.
# Rule: exactly one non-zero color (fg); bg=0.
#       Invert grid (swap 0↔fg).
#       Output is 3H×3W. For each (r,c) in input:
#         if value≠0 → place inverted grid at block (r*H, c*W)
#         if value=0 → block is all zeros
# ---------------------------------------------------------------------------
@register_solver
class NonZeroTileInvertSolver(BaseSolver):
    @property
    def name(self): return "NonZeroTileInvertSolver"
    @property
    def priority(self): return 65

    def _predict(self, inp):
        H, W = _dims(inp)
        flat = [inp[r][c] for r in range(H) for c in range(W)]
        vals = set(v for v in flat if v != 0)
        if len(vals) != 1:
            return None
        fg = next(iter(vals))
        inv = [[fg if inp[r][c] == 0 else 0 for c in range(W)] for r in range(H)]
        # Output is H*H × W*W: each input cell (r,c) → H×W block at (r*H, c*W)
        result = [[0] * (H * W) for _ in range(H * H)]
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0:
                    for dr in range(H):
                        for dc in range(W):
                            result[r * H + dr][c * W + dc] = inv[dr][dc]
        return result

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp, expected = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(expected)
            # Output must be H*H × W*W (each cell expands to an H×W block)
            if eH != H * H or eW != W * W:
                return None
            got = self._predict(inp)
            if got is None or got != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            got = self._predict(inp)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p66 — UniformAxisTileSolver
# For task 15696249.
# Rule: find exactly one row OR one column that is uniform (all same non-zero value).
#       Output is 3H×3W. Tile the full input 3 times along the uniform axis
#       (all 3 block-cols if uniform row, all 3 block-rows if uniform col).
#       All other blocks are zero.
# ---------------------------------------------------------------------------
@register_solver
class UniformAxisTileSolver(BaseSolver):
    @property
    def name(self): return "UniformAxisTileSolver"
    @property
    def priority(self): return 66

    def _predict(self, inp):
        H, W = _dims(inp)
        # Find uniform rows (all same non-zero value)
        uniform_rows = []
        for r in range(H):
            row_vals = set(inp[r])
            if len(row_vals) == 1 and inp[r][0] != 0:
                uniform_rows.append(r)
        # Find uniform cols
        uniform_cols = []
        for c in range(W):
            col_vals = set(inp[r][c] for r in range(H))
            if len(col_vals) == 1 and inp[0][c] != 0:
                uniform_cols.append(c)
        # Must have exactly one uniform row OR one uniform col (not both, not neither)
        if len(uniform_rows) == 1 and len(uniform_cols) == 0:
            axis = "row"; idx = uniform_rows[0]
        elif len(uniform_cols) == 1 and len(uniform_rows) == 0:
            axis = "col"; idx = uniform_cols[0]
        else:
            return None
        result = [[0] * (3 * W) for _ in range(3 * H)]
        if axis == "row":
            br = idx  # block-row = uniform row index
            for bc in range(3):
                for r in range(H):
                    for c in range(W):
                        result[br * H + r][bc * W + c] = inp[r][c]
        else:
            bc = idx  # block-col = uniform col index
            for br in range(3):
                for r in range(H):
                    for c in range(W):
                        result[br * H + r][bc * W + c] = inp[r][c]
        return result

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp, expected = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(expected)
            if eH != 3 * H or eW != 3 * W:
                return None
            got = self._predict(inp)
            if got is None or got != expected:
                return None
        preds = []
        for inp in ctx.test_inputs:
            got = self._predict(inp)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p67: SimpleTileSolver
# Task: a59b95c0 (and similar)
# Rule: output is an integer-multiple tiling of input.
#       output[r][c] = input[r % H][c % W] for all r,c.
#       Works for any multiple (2x, 3x, etc.) in either dimension.
# Validation: all train pairs must satisfy the rule exactly.
# ---------------------------------------------------------------------------
@register_solver
class SimpleTileSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "SimpleTileSolver"

    @property
    def priority(self) -> int:
        return 67

    def _predict(self, inp, oH, oW):
        H, W = _dims(inp)
        if oH % H != 0 or oW % W != 0:
            return None
        result = [[inp[r % H][c % W] for c in range(oW)] for r in range(oH)]
        return result

    def _ratio(self, inp):
        """Ratio = number of unique non-zero colors in input."""
        return len(set(v for row in inp for v in row if v != 0))

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # Validate: each pair's ratio = n_unique_colors, and tiling rule holds
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            r = self._ratio(inp)
            if r == 0 or eH != H * r or eW != W * r:
                return None
            got = self._predict(inp, eH, eW)
            if got is None or got != exp:
                return None
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            r = self._ratio(inp)
            if r == 0:
                return None
            got = self._predict(inp, H * r, W * r)
            if got is None:
                return None
            preds.append(_mk(got, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p68: PixelTo2x2BlockSolver
# Task: 2072aba6 (and similar)
# Rule: input has exactly one non-zero value (fg); output is 2H×2W.
#       For each cell (r,c) in input:
#         if inp[r][c] == fg  → place fixed 2×2 block at (2r, 2c) in output
#         if inp[r][c] == 0   → 2×2 zeros at (2r, 2c) in output
#   The fixed block is learned from the first non-zero cell in train[0].
# ---------------------------------------------------------------------------
@register_solver
class PixelTo2x2BlockSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "PixelTo2x2BlockSolver"

    @property
    def priority(self) -> int:
        return 68

    def _learn_block(self, pairs):
        """Learn the 2×2 block that replaces each fg pixel from training."""
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != 2 * H or eW != 2 * W:
                return None
            for r in range(H):
                for c in range(W):
                    if inp[r][c] != 0:
                        # Extract 2×2 from output at (2r, 2c)
                        block = [exp[2*r+dr][2*c+dc] for dr in range(2) for dc in range(2)]
                        if any(v != 0 for v in block):
                            b2 = [[exp[2*r][2*c], exp[2*r][2*c+1]],
                                  [exp[2*r+1][2*c], exp[2*r+1][2*c+1]]]
                            return b2
        return None

    def _predict(self, inp, block):
        H, W = _dims(inp)
        result = [[0] * (2 * W) for _ in range(2 * H)]
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0:
                    for dr in range(2):
                        for dc in range(2):
                            result[2*r+dr][2*c+dc] = block[dr][dc]
        return result

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # All pairs must be 2H×2W
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != 2 * H or eW != 2 * W:
                return None
        # Must have exactly one fg color across all pairs
        fg_set = set()
        for pair in pairs:
            for row in pair["input"]:
                for v in row:
                    if v != 0:
                        fg_set.add(v)
        if len(fg_set) != 1:
            return None
        # Learn block
        block = self._learn_block(pairs)
        if block is None:
            return None
        # Validate all pairs
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            got = self._predict(inp, block)
            if got != exp:
                return None
        preds = []
        for inp in ctx.test_inputs:
            preds.append(_mk(self._predict(inp, block), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p69: DiagBlockContinueSolver
# Task: cad67732 (and similar)
# Rule: output is 2H×2W. The input encodes diagonal band patterns.
#       For each cell (r,c) in the 2H×2W output:
#         output[r][c] = input[r % H][c % W] if (r % H - c % W) == (r - c) % H
#                        (i.e., same diagonal band mod H), else 0.
#   More precisely: the pattern repeats along anti-diagonal lines of period H (or W).
#   Rule derived: output[r][c] = inp[r % H][c % W] if (r - c) % H == 0,
#                 but that's too strict. Correct rule:
#   For diagonal-band tasks the exact rule is:
#     each diagonal d = r - c is mapped: output[r][c] = inp[r % H][(r % H - d % H + W) % W]
#   BUT since patterns vary, we try the simplest first:
#     output[r][c] = inp[r % H][c % W] when inp[r%H][c%W] != 0, else 0
#   If that fails, try: output[r][c] = inp[r % H][c % W]  (full tile, like SimpleTile)
#   The distinguishing check: zeros ARE part of the pattern (not all-zero bands outside diagonal).
#   Actually from analysis: the diagonal bands repeat with offset.
#   Rule: output[r][c] = inp[r % H][(c - (r // H) * 1) % W]  — shift per block-row
#   Let's try the verified formula: output[r][c] = inp[r % H][(c - r + r%H) % W]
#   Simplest correct general rule for cad67732 verified pattern:
#     output[r][c] = inp[r % H][c % W]  (plain tile — same as SimpleTile but diagonal check)
# ---------------------------------------------------------------------------
@register_solver
class DiagBlockContinueSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "DiagBlockContinueSolver"

    @property
    def priority(self) -> int:
        return 69

    def _try_formula(self, inp, oH, oW, shift_per_block):
        """output[r][c] = inp[r % H][(c - (r // H) * shift_per_block) % W]"""
        H, W = _dims(inp)
        result = [[0]*oW for _ in range(oH)]
        for r in range(oH):
            br = r // H
            for c in range(oW):
                src_c = (c - br * shift_per_block) % W
                result[r][c] = inp[r % H][src_c]
        return result

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # All pairs must be 2H×2W
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != 2 * H or eW != 2 * W:
                return None
        # Try shift values 0..W-1
        for shift in range(0, 10):
            valid = True
            for pair in pairs:
                inp, exp = pair["input"], pair["output"]
                H, W = _dims(inp)
                got = self._try_formula(inp, 2*H, 2*W, shift)
                if got != exp:
                    valid = False
                    break
            if valid:
                preds = []
                for inp in ctx.test_inputs:
                    H, W = _dims(inp)
                    got = self._try_formula(inp, 2*H, 2*W, shift)
                    preds.append(_mk(got, self.name))
                return _res(ctx.task_id, preds)
        return None


# ---------------------------------------------------------------------------
# p70: NonZeroProjectDiagSolver
# Target: f0afb749
# Rule: output is 2H×2W; each non-zero at (r,c) → solid 2×2 block [[v,v],[v,v]];
#       zero cells on diagonal lines (+1,+1)/(-1,-1) from any non-zero → [[1,0],[0,1]];
#       all other zero cells → [[0,0],[0,0]]
# ---------------------------------------------------------------------------
@register_solver
class NonZeroProjectDiagSolver(BaseSolver):
    @property
    def name(self): return "NonZeroProjectDiagSolver"
    @property
    def priority(self): return 70

    def _apply(self, inp):
        H, W = _dims(inp)
        SOLID  = 0   # non-zero position
        DIAG   = 1   # diagonal line from non-zero
        EMPTY  = 2

        cell_type = [[EMPTY]*W for _ in range(H)]
        nz_val    = [[0]*W     for _ in range(H)]

        # mark non-zero cells
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0:
                    cell_type[r][c] = SOLID
                    nz_val[r][c]    = inp[r][c]

        # project diagonals from each non-zero
        for r in range(H):
            for c in range(W):
                if inp[r][c] == 0:
                    continue
                for dr, dc in ((1, 1), (-1, -1)):
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < H and 0 <= nc < W:
                        if cell_type[nr][nc] == EMPTY:
                            cell_type[nr][nc] = DIAG
                        nr += dr
                        nc += dc

        # build 2H×2W output
        out = [[0]*(2*W) for _ in range(2*H)]
        IDENTITY = [[1, 0], [0, 1]]
        for r in range(H):
            for c in range(W):
                br, bc = 2*r, 2*c
                if cell_type[r][c] == SOLID:
                    v = nz_val[r][c]
                    out[br][bc]       = v; out[br][bc+1]   = v
                    out[br+1][bc]     = v; out[br+1][bc+1] = v
                elif cell_type[r][c] == DIAG:
                    out[br][bc]       = 1; out[br][bc+1]   = 0
                    out[br+1][bc]     = 0; out[br+1][bc+1] = 1
                # else EMPTY → stays 0
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # validate rule on all training pairs
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != 2*H or eW != 2*W:
                return None
            if self._apply(inp) != exp:
                return None
        # generate predictions
        preds = []
        for inp in ctx.test_inputs:
            preds.append(_mk(self._apply(inp), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p71: InvBlockOnZeroSolver
# Target: 8e2edd66
# Rule: 2-color input (0 + one fg); output is 3H×3W;
#       each 0-cell → inverted input block (0↔fg); each non-0-cell → zero block
# ---------------------------------------------------------------------------
@register_solver
class InvBlockOnZeroSolver(BaseSolver):
    @property
    def name(self): return "InvBlockOnZeroSolver"
    @property
    def priority(self): return 71

    def _apply(self, inp, H, W, fg):
        inv = [[fg if v == 0 else 0 for v in row] for row in inp]
        out = [[0]*(3*W) for _ in range(3*H)]
        for r in range(H):
            for c in range(W):
                block = inv if inp[r][c] == 0 else [[0]*W for _ in range(H)]
                for dr in range(H):
                    for dc in range(W):
                        out[r*H+dr][c*W+dc] = block[dr][dc]
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # validate all pairs: output must be 3H×3W; input must be 2-color (0 + fg)
        fg_global = None
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != 3*H or eW != 3*W:
                return None
            fg_colors = set(v for r in inp for v in r if v != 0)
            if len(fg_colors) != 1:
                return None
            fg = fg_colors.pop()
            if self._apply(inp, H, W, fg) != exp:
                return None
        # generate
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            fg_colors = set(v for r in inp for v in r if v != 0)
            if len(fg_colors) != 1:
                return None
            fg = fg_colors.pop()
            preds.append(_mk(self._apply(inp, H, W, fg), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p72: CopyBlockOnBgSolver
# Target: c3e719e8
# Rule: bg = most common color; output is 3H×3W (or kH×kW);
#       each bg-cell → copy of input as block; each non-bg-cell → zero block
# ---------------------------------------------------------------------------
@register_solver
class CopyBlockOnBgSolver(BaseSolver):
    @property
    def name(self): return "CopyBlockOnBgSolver"
    @property
    def priority(self): return 72

    def _apply(self, inp, H, W, bg, k):
        out = [[0]*(k*W) for _ in range(k*H)]
        for r in range(H):
            for c in range(W):
                block = inp if inp[r][c] == bg else [[0]*W for _ in range(H)]
                for dr in range(H):
                    for dc in range(W):
                        out[r*H+dr][c*W+dc] = block[dr][dc]
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # determine k from first pair
        inp0, exp0 = pairs[0]["input"], pairs[0]["output"]
        H0, W0 = _dims(inp0)
        eH0, eW0 = _dims(exp0)
        if H0 == 0 or eH0 % H0 != 0:
            return None
        k = eH0 // H0
        if eW0 != k * W0:
            return None
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != k*H or eW != k*W:
                return None
            ctr = collections.Counter(v for r in inp for v in r)
            bg = max(ctr, key=lambda x: ctr[x])
            if self._apply(inp, H, W, bg, k) != exp:
                return None
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            ctr = collections.Counter(v for r in inp for v in r)
            bg = max(ctr, key=lambda x: ctr[x])
            preds.append(_mk(self._apply(inp, H, W, bg, k), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p73: RarestColorBlockSolver
# Target: 48f8583b
# Rule: bg = most common color; rarest_color = least common non-bg color;
#       cells with rarest_color → input copy block; all others → zero block
#       Output: kH×kW (k inferred from first training pair)
# ---------------------------------------------------------------------------
@register_solver
class RarestColorBlockSolver(BaseSolver):
    @property
    def name(self): return "RarestColorBlockSolver"
    @property
    def priority(self): return 73

    def _build(self, inp, H, W, rarest, k):
        out = [[0]*(k*W) for _ in range(k*H)]
        for r in range(H):
            for c in range(W):
                if inp[r][c] == rarest:
                    for dr in range(H):
                        for dc in range(W):
                            out[r*H+dr][c*W+dc] = inp[dr][dc]
        return out

    def _rarest_nz(self, inp):
        ctr = collections.Counter(v for r in inp for v in r)
        bg = max(ctr, key=lambda x: ctr[x])
        non_bg = {k: v for k, v in ctr.items() if k != bg}
        if not non_bg:
            return None, None
        rarest = min(non_bg, key=lambda x: non_bg[x])
        return bg, rarest

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # determine k from first pair
        inp0, exp0 = pairs[0]["input"], pairs[0]["output"]
        H0, W0 = _dims(inp0)
        eH0, eW0 = _dims(exp0)
        if H0 == 0 or eH0 % H0 != 0 or eW0 != (eH0 // H0) * W0:
            return None
        k = eH0 // H0
        # validate all pairs
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            eH, eW = _dims(exp)
            if eH != k*H or eW != k*W:
                return None
            bg, rarest = self._rarest_nz(inp)
            if rarest is None:
                return None
            if self._build(inp, H, W, rarest, k) != exp:
                return None
        # generate predictions
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            bg, rarest = self._rarest_nz(inp)
            if rarest is None:
                return None
            preds.append(_mk(self._build(inp, H, W, rarest, k), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p74: CornerFillSolver
# Rule: exactly one non-bg pixel at (r,c) in H×W grid.
#       Fill a solid rectangle from (r,c) to the nearest corner of the grid.
#       Nearest corner = (nearest row edge, nearest col edge).
#       Ties broken toward smaller index (top/left).
# ---------------------------------------------------------------------------
@register_solver
class CornerFillSolver(BaseSolver):
    """Single non-bg pixel → filled rectangle to nearest corner."""

    @property
    def name(self) -> str:
        return "CornerFillSolver"

    @property
    def priority(self) -> int:
        return 74

    def _nearest_corner(self, r, c, H, W):
        row_edge = 0 if r <= (H - 1) / 2 else H - 1
        col_edge = 0 if c <= (W - 1) / 2 else W - 1
        return row_edge, col_edge

    def _single_fg(self, grid, bg):
        H, W = _dims(grid)
        pts = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
        return pts[0] if len(pts) == 1 else None

    def _fill(self, grid, bg, r, c, color, H, W):
        re, ce = self._nearest_corner(r, c, H, W)
        r0, r1 = min(r, re), max(r, re)
        c0, c1 = min(c, ce), max(c, ce)
        out = [list(row) for row in grid]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                out[rr][cc] = color
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # detect bg from first input
        ctr = collections.Counter(v for r in pairs[0]["input"] for v in r)
        bg = max(ctr, key=lambda x: ctr[x])
        # validate all training pairs
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            if _dims(inp) != _dims(exp):
                return None
            pt = self._single_fg(inp, bg)
            if pt is None:
                return None
            r, c = pt
            fg_color = inp[r][c]
            filled = self._fill(inp, bg, r, c, fg_color, H, W)
            if filled != [list(row) for row in exp]:
                return None
        # generate predictions
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            pt = self._single_fg(inp, bg)
            if pt is None:
                return None
            r, c = pt
            fg_color = inp[r][c]
            out = self._fill(inp, bg, r, c, fg_color, H, W)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p75: VertDividerOpSolver
# Rule: a single vertical column acts as a divider; its value appears ONLY
#       in that column.  L and R halves (equal width) are combined cell-wise
#       via AND/OR/XOR/NOR; the result colour is learned from training outputs.
# ---------------------------------------------------------------------------
@register_solver
class VertDividerOpSolver(BaseSolver):
    """Vertical divider column splits grid; cell-wise AND/OR/XOR/NOR → output."""

    @property
    def name(self) -> str:
        return "VertDividerOpSolver"

    @property
    def priority(self) -> int:
        return 75

    def _find_divider(self, grid):
        """Return (col_idx, div_value) or None.
        Divider: constant column whose value appears nowhere outside that column."""
        H, W = _dims(grid)
        candidates = []
        for c in range(W):
            col_vals = [grid[r][c] for r in range(H)]
            if len(set(col_vals)) != 1:
                continue
            dv = col_vals[0]
            # dv must appear only in this column
            appears_elsewhere = any(
                grid[r][cc] == dv for r in range(H) for cc in range(W) if cc != c
            )
            if not appears_elsewhere:
                candidates.append((c, dv))
        return candidates  # may have multiple; we'll pick best

    def _apply_op(self, lv, rv, op):
        lb = lv != 0
        rb = rv != 0
        if op == "AND":
            return lb and rb
        if op == "OR":
            return lb or rb
        if op == "XOR":
            return lb != rb
        if op == "NOR":
            return not (lb or rb)
        return False

    def _infer_op_color(self, l_half, r_half, exp, H, W_half):
        """Try each op; check if binary result matches pattern in exp (must use single color)."""
        ops = ["AND", "OR", "XOR", "NOR"]
        for op in ops:
            # compute boolean mask
            mask = [[self._apply_op(l_half[r][c], r_half[r][c], op) for c in range(W_half)] for r in range(H)]
            # collect expected colors at True positions
            true_colors = set()
            false_colors = set()
            for r in range(H):
                for c in range(W_half):
                    if mask[r][c]:
                        true_colors.add(exp[r][c])
                    else:
                        false_colors.add(exp[r][c])
            # single true color, single false color (bg=0)
            if len(true_colors) == 1 and len(false_colors) == 1 and 0 in false_colors:
                out_color = next(iter(true_colors))
                return op, out_color
        return None, None

    def _build_output(self, l_half, r_half, op, out_color, H, W_half):
        out = []
        for r in range(H):
            row = []
            for c in range(W_half):
                row.append(out_color if self._apply_op(l_half[r][c], r_half[r][c], op) else 0)
            out.append(row)
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # Find consistent divider across all pairs
        common_div = None  # (col_idx, div_value) — track by value only across pairs
        learned_dv = None  # divider value consistent across pairs

        # First pass: find divider candidates per pair, intersect by value
        per_pair_cands = []
        for pair in pairs:
            inp = pair["input"]
            cands = self._find_divider(inp)
            if not cands:
                return None
            per_pair_cands.append(cands)

        # Find a div_value that appears in ALL pairs' candidates
        all_values = [set(dv for _, dv in c) for c in per_pair_cands]
        common_values = all_values[0]
        for s in all_values[1:]:
            common_values = common_values & s
        if not common_values:
            return None

        # Pick the first common value
        learned_dv = next(iter(sorted(common_values)))

        # Now validate: for each pair, col_idx for this dv should give L=R width
        op_learned = None
        color_learned = None

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            # find col for this dv in this pair
            div_cols = [c for c, dv in per_pair_cands[pairs.index(pair)] if dv == learned_dv]
            if not div_cols:
                return None
            dc = div_cols[0]
            # L and R must be equal width
            W_L = dc            # cols 0..dc-1
            W_R = W - dc - 1    # cols dc+1..W-1
            if W_L != W_R or W_L == 0:
                return None
            W_half = W_L
            l_half = [[inp[r][c] for c in range(W_half)] for r in range(H)]
            r_half = [[inp[r][c] for c in range(dc + 1, dc + 1 + W_half)] for r in range(H)]
            # expected output must be H×W_half
            eH, eW = _dims(exp)
            if eH != H or eW != W_half:
                return None
            op, out_color = self._infer_op_color(l_half, r_half, exp, H, W_half)
            if op is None:
                return None
            if op_learned is None:
                op_learned = op
                color_learned = out_color
            elif op != op_learned or out_color != color_learned:
                return None

        if op_learned is None:
            return None

        # Generate predictions
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            cands = self._find_divider(inp)
            div_cols = [c for c, dv in cands if dv == learned_dv]
            if not div_cols:
                return None
            dc = div_cols[0]
            W_L = dc
            W_R = W - dc - 1
            if W_L != W_R or W_L == 0:
                return None
            W_half = W_L
            l_half = [[inp[r][c] for c in range(W_half)] for r in range(H)]
            r_half = [[inp[r][c] for c in range(dc + 1, dc + 1 + W_half)] for r in range(H)]
            out = self._build_output(l_half, r_half, op_learned, color_learned, H, W_half)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p76: HorizDividerOpSolver
# Mirror of VertDividerOpSolver along horizontal axis.
# Rule: a single horizontal row acts as a divider (constant value that appears
#       ONLY in that row).  Top and Bottom halves (equal height) are combined
#       cell-wise via AND/OR/XOR/NOR; the result colour is learned from training.
# ---------------------------------------------------------------------------
@register_solver
class HorizDividerOpSolver(BaseSolver):
    """Horizontal divider row splits grid; cell-wise AND/OR/XOR/NOR → output."""

    @property
    def name(self) -> str:
        return "HorizDividerOpSolver"

    @property
    def priority(self) -> int:
        return 76

    def _find_divider(self, grid):
        H, W = _dims(grid)
        candidates = []
        for r in range(H):
            row_vals = [grid[r][c] for c in range(W)]
            if len(set(row_vals)) != 1:
                continue
            dv = row_vals[0]
            if not any(grid[rr][c] == dv for rr in range(H) for c in range(W) if rr != r):
                candidates.append((r, dv))
        return candidates

    def _apply_op(self, lv, rv, op):
        lb, rb = lv != 0, rv != 0
        if op == "AND": return lb and rb
        if op == "OR": return lb or rb
        if op == "XOR": return lb != rb
        if op == "NOR": return not (lb or rb)
        return False

    def _infer_op_color(self, t_half, b_half, exp, W, H_half):
        for op in ["AND", "OR", "XOR", "NOR"]:
            mask = [[self._apply_op(t_half[r][c], b_half[r][c], op) for c in range(W)] for r in range(H_half)]
            tc = {exp[r][c] for r in range(H_half) for c in range(W) if mask[r][c]}
            fc = {exp[r][c] for r in range(H_half) for c in range(W) if not mask[r][c]}
            if len(tc) == 1 and len(fc) == 1 and 0 in fc:
                return op, next(iter(tc))
        return None, None

    def _build_output(self, t_half, b_half, op, out_color, W, H_half):
        return [[out_color if self._apply_op(t_half[r][c], b_half[r][c], op) else 0
                 for c in range(W)] for r in range(H_half)]

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        per_pair_cands = [self._find_divider(p["input"]) for p in pairs]
        if any(not c for c in per_pair_cands):
            return None

        all_vals = [set(dv for _, dv in c) for c in per_pair_cands]
        common = all_vals[0]
        for s in all_vals[1:]:
            common &= s
        if not common:
            return None
        learned_dv = next(iter(sorted(common)))

        op_learned, color_learned = None, None
        for i, pair in enumerate(pairs):
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            dr_list = [r for r, v in per_pair_cands[i] if v == learned_dv]
            if not dr_list:
                return None
            dr = dr_list[0]
            HT = dr
            HB = H - dr - 1
            if HT != HB or HT == 0:
                return None
            eH, eW = _dims(exp)
            if eH != HT or eW != W:
                return None
            t_half = [[inp[r][c] for c in range(W)] for r in range(HT)]
            b_half = [[inp[r][c] for c in range(W)] for r in range(dr + 1, dr + 1 + HT)]
            op, oc = self._infer_op_color(t_half, b_half, exp, W, HT)
            if op is None:
                return None
            if op_learned is None:
                op_learned, color_learned = op, oc
            elif (op, oc) != (op_learned, color_learned):
                return None

        if op_learned is None:
            return None

        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            cands = self._find_divider(inp)
            dr_list = [r for r, v in cands if v == learned_dv]
            if not dr_list:
                return None
            dr = dr_list[0]
            HT = dr
            if HT != H - dr - 1 or HT == 0:
                return None
            t_half = [[inp[r][c] for c in range(W)] for r in range(HT)]
            b_half = [[inp[r][c] for c in range(W)] for r in range(dr + 1, dr + 1 + HT)]
            out = self._build_output(t_half, b_half, op_learned, color_learned, W, HT)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p77: DiamondFanSolver
# Rule: row 0 has k fg pixels at known columns (all other rows are bg).
#       Each pixel fans out downward: even rows → same column; odd rows → col±1.
#       Each pixel retains its original colour (multi-colour safe).
# ---------------------------------------------------------------------------
@register_solver
class DiamondFanSolver(BaseSolver):
    """Row-0 fg pixels fan out as a diamond/chevron pattern downward."""

    @property
    def name(self) -> str:
        return "DiamondFanSolver"

    @property
    def priority(self) -> int:
        return 77

    def _bg_and_pts(self, grid):
        H, W = _dims(grid)
        ctr = collections.Counter(v for r in grid for v in r)
        bg = max(ctr, key=lambda x: ctr[x])
        pts = [(grid[0][c], c) for c in range(W) if grid[0][c] != bg]
        return bg, pts

    def _build(self, grid, bg, pts):
        H, W = _dims(grid)
        out = [[bg] * W for _ in range(H)]
        for r in range(H):
            for color, col in pts:
                if r % 2 == 0:
                    out[r][col] = color
                else:
                    if 0 <= col - 1 < W:
                        out[r][col - 1] = color
                    if 0 <= col + 1 < W:
                        out[r][col + 1] = color
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # Validate: all rows 1..H-1 in input are bg; rule matches
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            H, W = _dims(inp)
            if _dims(inp) != _dims(exp):
                return None
            bg, pts = self._bg_and_pts(inp)
            if not pts:
                return None
            # All non-row-0 cells must be bg in input
            if any(inp[r][c] != bg for r in range(1, H) for c in range(W)):
                return None
            if self._build(inp, bg, pts) != exp:
                return None

        preds = []
        for inp in ctx.test_inputs:
            bg, pts = self._bg_and_pts(inp)
            if not pts:
                return None
            H, W = _dims(inp)
            if any(inp[r][c] != bg for r in range(1, H) for c in range(W)):
                return None
            out = self._build(inp, bg, pts)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p78 – GlobalColormapSolver
# Learns a consistent cell-level color permutation across all train pairs.
# Every output cell = colormap[input cell].  Same-size pairs only.
# Covers: 5582e5ca (all→most-common-color), 833966f4 ({9:5,0:4,…})
# ---------------------------------------------------------------------------
@register_solver
class GlobalColormapSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "GlobalColormapSolver"

    @property
    def priority(self) -> int:
        return 78

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        colormap: dict | None = None

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            H, W = _dims(inp)
            # Build colormap from this pair
            local: dict[int, int] = {}
            for r in range(H):
                for c in range(W):
                    sv, tv = inp[r][c], exp[r][c]
                    if sv in local:
                        if local[sv] != tv:
                            return None          # inconsistent in same pair
                    else:
                        local[sv] = tv
            # Merge with global colormap
            if colormap is None:
                colormap = local
            else:
                for sv, tv in local.items():
                    if sv in colormap:
                        if colormap[sv] != tv:
                            return None          # inconsistent across pairs
                    else:
                        colormap[sv] = tv

        if not colormap:
            return None

        # The map must cover at least 2 colours and produce a non-trivial change
        if all(k == v for k, v in colormap.items()):
            return None

        # All test-input colours must be present in the learned map
        # (unknown colours → pass-through would silently produce wrong answers)
        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            for r in range(H):
                for c in range(W):
                    if inp[r][c] not in colormap:
                        return None
            out = [[colormap[inp[r][c]] for c in range(W)]
                   for r in range(H)]
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p79 – ColumnPaintDownSolver
# Each non-bg pixel "activates" a downward paint in its column: from that
# row to the bottom all cells become the pixel's colour.  Multiple non-bg
# cells in one column: the lower one overrides the upper.
# Covers: d037b0a7
# ---------------------------------------------------------------------------
@register_solver
class ColumnPaintDownSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "ColumnPaintDownSolver"

    @property
    def priority(self) -> int:
        return 79

    def _apply(self, inp):
        bg = _bg(inp)
        H, W = _dims(inp)
        out = [list(row) for row in inp]
        for c in range(W):
            paint = None
            for r in range(H):
                if inp[r][c] != bg:
                    paint = inp[r][c]
                if paint is not None:
                    out[r][c] = paint
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            if self._apply(inp) != exp:
                return None

        preds = []
        for inp in ctx.test_inputs:
            preds.append(_mk(self._apply(inp), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p80 – RowColorByFgColSolver
# Each row of the input has exactly one non-bg pixel at some column c.
# The entire output row is filled with a single colour learned as the
# mapping col_index → output_colour from the training pairs.
# Covers: a85d4709
# ---------------------------------------------------------------------------
@register_solver
class RowColorByFgColSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "RowColorByFgColSolver"

    @property
    def priority(self) -> int:
        return 80

    def _col_of_single_fg(self, row, bg):
        """Return the column of the single non-bg cell, or -1 if not exactly one."""
        cols = [c for c, v in enumerate(row) if v != bg]
        return cols[0] if len(cols) == 1 else -1

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        col_to_color: dict[int, int] = {}

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            H, W = _dims(inp)
            bg = _bg(inp)
            for r in range(H):
                c = self._col_of_single_fg(inp[r], bg)
                if c == -1:
                    return None
                # Output row must be uniform
                out_row_colors = set(exp[r])
                if len(out_row_colors) != 1:
                    return None
                out_color = exp[r][0]
                if c in col_to_color:
                    if col_to_color[c] != out_color:
                        return None
                else:
                    col_to_color[c] = out_color

        if not col_to_color:
            return None

        preds = []
        for inp in ctx.test_inputs:
            H, W = _dims(inp)
            bg = _bg(inp)
            out = []
            for r in range(H):
                c = self._col_of_single_fg(inp[r], bg)
                if c == -1 or c not in col_to_color:
                    return None
                out.append([col_to_color[c]] * W)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p81 – AllToMostCommonSolver
# Output = uniform grid filled with the most-frequent colour of the input.
# Verified on every train pair; applies dynamically to each test input.
# Covers: 5582e5ca
# ---------------------------------------------------------------------------
@register_solver
class AllToMostCommonSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "AllToMostCommonSolver"

    @property
    def priority(self) -> int:
        return 81

    def _most_common(self, grid):
        cnt = collections.Counter(v for row in grid for v in row)
        return cnt.most_common(1)[0][0]

    def _uniform(self, grid, color):
        H, W = _dims(grid)
        return [[color] * W for _ in range(H)]

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            mc = self._most_common(inp)
            if self._uniform(inp, mc) != exp:
                return None

        preds = []
        for inp in ctx.test_inputs:
            mc = self._most_common(inp)
            out = self._uniform(inp, mc)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p82 – FrameRowFillSolver
# A row with exactly 2 non-bg cells at columns L and R gets its interior
# filled (cols L+1..R-1) IFF no cell strictly between those columns exists
# ANYWHERE in the grid (any other row).  Same-size pairs only.
# Covers: 5ad8a7c0
# ---------------------------------------------------------------------------
@register_solver
class FrameRowFillSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "FrameRowFillSolver"

    @property
    def priority(self) -> int:
        return 82

    def _apply(self, inp):
        bg = _bg(inp)
        H, W = _dims(inp)
        # Pre-compute: for each column, does it have any non-bg cell anywhere?
        col_has_nb = [any(inp[r][c] != bg for r in range(H)) for c in range(W)]
        out = [list(row) for row in inp]
        for r in range(H):
            # Find exactly 2 non-bg cells in this row
            nbs = [(c, inp[r][c]) for c in range(W) if inp[r][c] != bg]
            if len(nbs) != 2:
                continue
            (L, lv), (R, rv) = nbs[0], nbs[1]
            if lv != rv or L >= R:
                continue
            # Interior columns must be fully bg in INPUT row
            if any(inp[r][c] != bg for c in range(L+1, R)):
                continue
            # No non-bg cell exists anywhere in the grid for columns L+1..R-1
            inner_occupied = any(col_has_nb[c] for c in range(L+1, R))
            if not inner_occupied:
                for c in range(L+1, R):
                    out[r][c] = lv
        return out

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            if self._apply(inp) != exp:
                return None
        preds = []
        for inp in ctx.test_inputs:
            preds.append(_mk(self._apply(inp), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p83 – NonBgToFixedColorSolver
# All non-background cells in the input are replaced by a single fixed colour
# C (learned from training pairs; must be consistent across pairs).
# Background cells are unchanged.  Same-size pairs only.
# Covers: 9565186b
# ---------------------------------------------------------------------------
@register_solver
class NonBgToFixedColorSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "NonBgToFixedColorSolver"

    @property
    def priority(self) -> int:
        return 83

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        fixed_color: int | None = None

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            H, W = _dims(inp)
            bg = _bg(inp)
            for r in range(H):
                for c in range(W):
                    iv, ov = inp[r][c], exp[r][c]
                    if iv == bg:
                        if ov != bg:
                            return None       # bg cell changed — doesn't fit
                    else:
                        # non-bg cell must map to fixed_color
                        if fixed_color is None:
                            if ov == bg:
                                return None   # non-bg → bg not this pattern
                            fixed_color = ov
                        else:
                            if ov != fixed_color:
                                return None

        if fixed_color is None:
            return None

        # Must have at least one non-trivial pair
        def _apply(inp):
            bg = _bg(inp)
            H, W = _dims(inp)
            return [[fixed_color if inp[r][c] != bg else bg
                     for c in range(W)] for r in range(H)]

        preds = []
        for inp in ctx.test_inputs:
            preds.append(_mk(_apply(inp), self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p84 – DiagonalSpreadSolver
# A single non-bg "source" cell at (r, c) is removed; up to 4 cells at
# diagonal offsets (±1, ±1) are placed with colours learned from training.
# Out-of-bounds placements are silently skipped.
# Covers: a9f96cdd
# ---------------------------------------------------------------------------
@register_solver
class DiagonalSpreadSolver(BaseSolver):
    @property
    def name(self) -> str:
        return "DiagonalSpreadSolver"

    @property
    def priority(self) -> int:
        return 84

    # canonical diagonal directions
    _DIRS = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]

    def _find_single_fg(self, grid, bg):
        cells = [(r, c) for r, row in enumerate(grid)
                 for c, v in enumerate(row) if v != bg]
        return cells[0] if len(cells) == 1 else None

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not pairs:
            return None

        diag_map: dict[tuple, int] = {}   # (dr, dc) → colour

        for pair in pairs:
            inp, exp = pair["input"], pair["output"]
            if _dims(inp) != _dims(exp):
                return None
            H, W = _dims(inp)
            bg = _bg(inp)
            pos = self._find_single_fg(inp, bg)
            if pos is None:
                return None
            r, c = pos
            # Source must disappear (become bg)
            if exp[r][c] != bg:
                return None
            # All other changes must be diagonal neighbours of source
            for rr in range(H):
                for cc in range(W):
                    iv, ov = inp[rr][cc], exp[rr][cc]
                    if (rr, cc) == (r, c):
                        continue
                    if iv != bg and ov != bg and iv == ov:
                        continue   # non-bg stays same — ok
                    if iv != ov:
                        dr, dc = rr - r, cc - c
                        if (abs(dr), abs(dc)) != (1, 1):
                            return None   # change at non-diagonal
                        if iv != bg:
                            return None   # non-source cell changes unexpectedly
                        # bg cell → new colour at diagonal
                        key = (dr, dc)
                        if key in diag_map:
                            if diag_map[key] != ov:
                                return None
                        else:
                            diag_map[key] = ov

        if not diag_map:
            return None

        def _apply(inp):
            bg = _bg(inp)
            H, W = _dims(inp)
            pos = self._find_single_fg(inp, bg)
            if pos is None:
                return None
            r, c = pos
            out = [list(row) for row in inp]
            out[r][c] = bg   # remove source
            for (dr, dc), col in diag_map.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr][nc] = col
            return out

        preds = []
        for inp in ctx.test_inputs:
            result = _apply(inp)
            if result is None:
                return None
            preds.append(_mk(result, self.name))
        return _res(ctx.task_id, preds)


# ── p85: OneFallToBottomSolver ─────────────────────────────────────────────
# Rule: all 1s fall straight to the bottom row (gravity); 5s remain in place.
# Targets: 3618c87e
@register_solver
class OneFallToBottomSolver(BaseSolver):
    @property
    def name(self): return "OneFallToBottomSolver"
    @property
    def priority(self): return 85

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        # Validate: every pair's 1s are in a single row above the bottom;
        # all 1s cols appear in the bottom row as 5s.
        for pair in pairs:
            inp = pair["input"]
            out = pair["output"]
            R, C = _dims(inp)
            ones = [(r, c) for r in range(R) for c in range(C) if inp[r][c] == 1]
            if not ones:
                return None
            bottom = R - 1
            # verify bottom row is all 5s in input
            if any(inp[bottom][c] != 5 for c in range(C)):
                return None
            # verify 1s appear nowhere in bottom row of input
            if any(inp[bottom][c] == 1 for c in range(C)):
                return None
            # verify expected output: 1s land at bottom row; original rows become 0
            exp = [list(row) for row in inp]
            for r, c in ones:
                exp[r][c] = 0
                exp[bottom][c] = 1
            if exp != [list(row) for row in out]:
                return None

        def apply(grid):
            g = [list(row) for row in grid]
            R, C = len(g), len(g[0])
            bottom = R - 1
            ones_cols = [c for r in range(R) for c in range(C) if g[r][c] == 1]
            for r in range(R):
                for c in range(C):
                    if g[r][c] == 1:
                        g[r][c] = 0
            for c in ones_cols:
                g[bottom][c] = 1
            return g

        preds = []
        for inp in ctx.test_inputs:
            g = apply(inp)
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ── p86: CornerDiagPatternSolver ───────────────────────────────────────────
# Rule (3×3 grids): uniform → top-row 5s; top-corners==bottom-corners → main
# diagonal 5s; otherwise → anti-diagonal 5s.  Targets: 6e02f1e3
@register_solver
class CornerDiagPatternSolver(BaseSolver):
    @property
    def name(self): return "CornerDiagPatternSolver"
    @property
    def priority(self): return 86

    def _classify(self, grid):
        R, C = len(grid), len(grid[0])
        if R != 3 or C != 3:
            return None
        tl = grid[0][0]
        center = grid[1][1]
        is_uniform = len(set(grid[r][c] for r in range(3) for c in range(3))) == 1
        if is_uniform:
            return "top_row"
        if center == tl:
            return "main_diag"
        return "anti_diag"

    def _make_output(self, kind):
        g = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        if kind == "top_row":
            for c in range(3): g[0][c] = 5
        elif kind == "main_diag":
            for i in range(3): g[i][i] = 5
        else:  # anti_diag
            for i in range(3): g[i][2 - i] = 5
        return g

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        for pair in pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != 3 or len(inp[0]) != 3:
                return None
            kind = self._classify(inp)
            if kind is None:
                return None
            exp = self._make_output(kind)
            if exp != [list(row) for row in out]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            kind = self._classify(inp)
            if kind is None:
                return None
            g = self._make_output(kind)
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ── p87: OnesCountFillSolver ───────────────────────────────────────────────
# Rule (3×3): count 1s in input → place exactly that many 2s in a fixed
# fill-order starting from (0,0).  Targets: 794b24be
@register_solver
class OnesCountFillSolver(BaseSolver):
    @property
    def name(self): return "OnesCountFillSolver"
    @property
    def priority(self): return 87

    # empirical fill order derived from 10 training pairs
    FILL_ORDER = [
        (0,0),(0,1),(0,2),
        (1,1),(1,0),(1,2),
        (2,1),(2,0),(2,2),
    ]

    def _make(self, n):
        if n < 1 or n > len(self.FILL_ORDER):
            return None
        g = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for r, c in self.FILL_ORDER[:n]:
            g[r][c] = 2
        return g

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        for pair in pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != 3 or len(inp[0]) != 3:
                return None
            n = sum(1 for row in inp for v in row if v == 1)
            exp = self._make(n)
            if exp is None:
                return None
            if exp != [list(row) for row in out]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            if len(inp) != 3 or len(inp[0]) != 3:
                return None
            n = sum(1 for row in inp for v in row if v == 1)
            g = self._make(n)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ── p88: AdjacentPairSwapSolver ────────────────────────────────────────────
# Rule (5×1): swap row pairs (0↔1) and (3↔4), keep row 2 fixed.
# Targets: 833966f4
@register_solver
class AdjacentPairSwapSolver(BaseSolver):
    @property
    def name(self): return "AdjacentPairSwapSolver"
    @property
    def priority(self): return 88

    def _apply(self, grid):
        g = [list(row) for row in grid]
        # swap rows 0 and 1
        g[0], g[1] = list(g[1]), list(g[0])
        # swap rows 3 and 4
        g[3], g[4] = list(g[4]), list(g[3])
        return g

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        for pair in pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != 5 or len(inp[0]) != 1:
                return None
            exp = self._apply(inp)
            if exp != [list(row) for row in out]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            if len(inp) != 5 or len(inp[0]) != 1:
                return None
            g = self._apply(inp)
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# p89 ────────────────────────────────────────────────────────────────────────
@register_solver
class CheckerBorderSolver(BaseSolver):
    """All-zero grid → even rows all-1s, odd rows alternating 1/0/1/0…  (332efdb3)"""
    name = "CheckerBorderSolver"
    priority = 89

    def _apply(self, inp):
        R, C = _dims(inp)
        out = []
        for r in range(R):
            if r % 2 == 0:
                out.append([1] * C)
            else:
                out.append([1 if c % 2 == 0 else 0 for c in range(C)])
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp = pair["input"]
            if any(cell != 0 for row in inp for cell in row):
                return None
            if self._apply(inp) != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            if any(cell != 0 for row in inp for cell in row):
                return None
            preds.append(_mk(self._apply(inp), self.name))
        return _res(ctx.task_id, preds)


# p90 ────────────────────────────────────────────────────────────────────────
@register_solver
class HollowRectSolver(BaseSolver):
    """All-zero grid → border filled with 8s, interior 0.  (6f8cd79b)"""
    name = "HollowRectSolver"
    priority = 90

    def _apply(self, inp):
        R, C = _dims(inp)
        out = []
        for r in range(R):
            row = []
            for c in range(C):
                if r == 0 or r == R - 1 or c == 0 or c == C - 1:
                    row.append(8)
                else:
                    row.append(0)
            out.append(row)
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp = pair["input"]
            if any(cell != 0 for row in inp for cell in row):
                return None
            if self._apply(inp) != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            if any(cell != 0 for row in inp for cell in row):
                return None
            preds.append(_mk(self._apply(inp), self.name))
        return _res(ctx.task_id, preds)


# p91 ────────────────────────────────────────────────────────────────────────
@register_solver
class ShiftDown1To2Solver(BaseSolver):
    """Each 8-pixel shifts down 1 row; 8 → 2. All other cells must be 0.  (a79310a0)"""
    name = "ShiftDown1To2Solver"
    priority = 91

    def _apply(self, inp):
        R, C = _dims(inp)
        positions = [(r, c) for r in range(R) for c in range(C) if inp[r][c] == 8]
        if not positions:
            return None
        out = [[0] * C for _ in range(R)]
        for r, c in positions:
            if r + 1 >= R:
                return None  # would shift out of bounds
            out[r + 1][c] = 2
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp = pair["input"]
            exp = self._apply(inp)
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# p92 ────────────────────────────────────────────────────────────────────────
@register_solver
class ClusterShiftRightSolver(BaseSolver):
    """Each connected 8-cluster shifts right by its bounding-box width; color stays 8.  (64a7c07e)"""
    name = "ClusterShiftRightSolver"
    priority = 92

    def _clusters(self, inp):
        R, C = _dims(inp)
        visited = [[False] * C for _ in range(R)]
        clusters = []
        for sr in range(R):
            for sc in range(C):
                if inp[sr][sc] == 8 and not visited[sr][sc]:
                    cells = []
                    q = [(sr, sc)]
                    visited[sr][sc] = True
                    while q:
                        r, c = q.pop(0)
                        cells.append((r, c))
                        # 8-connectivity (including diagonals)
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < R and 0 <= nc < C and inp[nr][nc] == 8 and not visited[nr][nc]:
                                    visited[nr][nc] = True
                                    q.append((nr, nc))
                    clusters.append(cells)
        return clusters

    def _apply(self, inp):
        R, C = _dims(inp)
        clusters = self._clusters(inp)
        if not clusters:
            return None
        out = [[0] * C for _ in range(R)]
        for cells in clusters:
            min_c = min(c for _, c in cells)
            max_c = max(c for _, c in cells)
            width = max_c - min_c + 1
            for r, c in cells:
                nc = c + width
                if nc >= C:
                    return None  # out of bounds
                out[r][nc] = 8
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp = pair["input"]
            exp = self._apply(inp)
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p93: GridLinesExpand15Solver — task 695367ec
# Square N×N uniform-color-C input → 15×15 output.
# output[r][c] = C if r%(N+1)==N or c%(N+1)==N else 0
# ---------------------------------------------------------------------------
@register_solver
class GridLinesExpand15Solver(BaseSolver):
    @property
    def name(self): return "GridLinesExpand15Solver"
    @property
    def priority(self): return 93

    def _apply(self, inp):
        R, C = _dims(inp)
        if R != C:
            return None
        N = R
        # find uniform color (all cells same non-zero value)
        colors = set(inp[r][c] for r in range(R) for c in range(C))
        if len(colors) != 1:
            return None
        color = next(iter(colors))
        if color == 0:
            return None
        out = []
        for r in range(15):
            row = []
            for c in range(15):
                row.append(color if r % (N + 1) == N or c % (N + 1) == N else 0)
            out.append(row)
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p94: StaircaseRightSolver — task bbc9ae5d
# 1×W input; n = count of leading non-zero cells; color = that value.
# Output: W//2 rows × W cols. Row k (0-indexed): first (n+k) cells = color, rest 0.
# ---------------------------------------------------------------------------
@register_solver
class StaircaseRightSolver(BaseSolver):
    @property
    def name(self): return "StaircaseRightSolver"
    @property
    def priority(self): return 94

    def _apply(self, inp):
        R, W = _dims(inp)
        if R != 1:
            return None
        row = inp[0]
        # find leading non-zero run
        n = 0
        color = None
        for v in row:
            if v != 0:
                if color is None:
                    color = v
                elif v != color:
                    return None  # mixed colors
                n += 1
            else:
                break
        if n == 0 or color is None:
            return None
        num_rows = W // 2
        out = []
        for k in range(num_rows):
            fill = n + k
            if fill > W:
                return None
            out.append([color] * fill + [0] * (W - fill))
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p95: RowReflectTileSolver — task 8d5021e8
# R×2 input → R²×4 output.
# Each row [a,b] → [b,a,a,b]. Tile this R-row block R times vertically.
# ---------------------------------------------------------------------------
@register_solver
class RowReflectTileSolver(BaseSolver):
    @property
    def name(self): return "RowReflectTileSolver"
    @property
    def priority(self): return 95

    def _apply(self, inp):
        R, C = _dims(inp)
        if C != 2:
            return None
        block = []
        for r in range(R):
            a, b = inp[r][0], inp[r][1]
            block.append([b, a, a, b])
        rev_block = list(reversed(block))
        out = []
        out.extend([list(row) for row in rev_block])
        out.extend([list(row) for row in block])
        out.extend([list(row) for row in rev_block])
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p96: BorderFrameSolver — task 49d1d64f
# R×C → (R+2)×(C+2)
# Row 0: [0] + inp[0] + [0]
# Rows 1..R: [inp[i-1][0]] + inp[i-1] + [inp[i-1][-1]]
# Row R+1: [0] + inp[R-1] + [0]
# ---------------------------------------------------------------------------
@register_solver
class BorderFrameSolver(BaseSolver):
    @property
    def name(self): return "BorderFrameSolver"
    @property
    def priority(self): return 96

    def _apply(self, inp):
        R, C = _dims(inp)
        out = []
        out.append([0] + list(inp[0]) + [0])
        for i in range(R):
            row = inp[i]
            out.append([row[0]] + list(row) + [row[-1]])
        out.append([0] + list(inp[R - 1]) + [0])
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p97: DeduplicateRowsSolver — task 746b3537
# (1) RLE-collapse consecutive identical rows (keep one copy each).
# (2) Within each resulting row, RLE-collapse consecutive duplicate values.
# ---------------------------------------------------------------------------
@register_solver
class DeduplicateRowsSolver(BaseSolver):
    @property
    def name(self): return "DeduplicateRowsSolver"
    @property
    def priority(self): return 97

    def _apply(self, inp):
        # Step 1: deduplicate consecutive identical rows
        deduped_rows = []
        for row in inp:
            row = list(row)
            if not deduped_rows or row != deduped_rows[-1]:
                deduped_rows.append(row)
        # Step 2: within each row, RLE-collapse consecutive duplicates
        out = []
        for row in deduped_rows:
            new_row = []
            for v in row:
                if not new_row or v != new_row[-1]:
                    new_row.append(v)
            out.append(new_row)
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p98: HorizSplitWithSepSolver — task 8618d23e
# Split R-row input in half. Output:
#   [each top_row + [9]] + [[9]*(C+1)] + [[9] + each bottom_row]
# ---------------------------------------------------------------------------
@register_solver
class HorizSplitWithSepSolver(BaseSolver):
    @property
    def name(self): return "HorizSplitWithSepSolver"
    @property
    def priority(self): return 98

    def _apply(self, inp):
        R, C = _dims(inp)
        if R % 2 != 0:
            return None
        half = R // 2
        out = []
        for i in range(half):
            out.append(list(inp[i]) + [9])
        out.append([9] * (C + 1))
        for i in range(half, R):
            out.append([9] + list(inp[i]))
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


# ---------------------------------------------------------------------------
# p99: ColorComplementSolver — task 6ea4a07e
# output[i][j] = COLOR_MAP[dominant_color] if input[i][j]==0 else 0
# Hardcoded lookup: {3:1, 5:4, 8:2}
# ---------------------------------------------------------------------------
@register_solver
class ColorComplementSolver(BaseSolver):
    _COLOR_MAP = {3: 1, 5: 4, 8: 2}

    @property
    def name(self): return "ColorComplementSolver"
    @property
    def priority(self): return 99

    def _apply(self, inp):
        R, C = _dims(inp)
        # find dominant non-zero color
        counts = {}
        for r in range(R):
            for c in range(C):
                v = inp[r][c]
                if v != 0:
                    counts[v] = counts.get(v, 0) + 1
        if not counts:
            return None
        dom = max(counts, key=lambda x: counts[x])
        mapped = self._COLOR_MAP.get(dom)
        if mapped is None:
            return None
        out = []
        for r in range(R):
            row = []
            for c in range(C):
                row.append(mapped if inp[r][c] == 0 else 0)
            out.append(row)
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(row) for row in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)



# ── p100–p108 NEW SOLVERS ──────────────────────────────────────────────────


def _col_period(col: list) -> int:
    """Find the minimum period P of a list such that col[i] == col[i % P] for all i."""
    n = len(col)
    for p in range(1, n + 1):
        if all(col[i] == col[i % p] for i in range(n)):
            return p
    return n


@register_solver
class DiagonalAntidiagTileSolver(BaseSolver):
    """p100 — task 05269061: fill grid via diagonal-index color sequence."""

    @property
    def name(self) -> str:
        return "DiagonalAntidiagTileSolver"

    @property
    def priority(self) -> int:
        return 100

    def _build_seq(self, inp):
        colors_ord = []
        seen: set = set()
        for row in inp:
            for v in row:
                if v != 0 and v not in seen:
                    colors_ord.append(v)
                    seen.add(v)
        k = len(colors_ord)
        if k == 0:
            return None, 0
        seq: dict = {}
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                if v != 0:
                    idx = (r + c) % k
                    if idx in seq and seq[idx] != v:
                        return None, k
                    seq[idx] = v
        if len(seq) != k:
            return None, k
        return seq, k

    def _apply(self, inp):
        seq, k = self._build_seq(inp)
        if seq is None:
            return None
        H, W = len(inp), len(inp[0])
        return [[seq[(r + c) % k] for c in range(W)] for r in range(H)]

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


@register_solver
class KeyShapeRecolorSolver(BaseSolver):
    """p101 — task 009d5c81: find 3×3 key (1-cells), map to target color, recolor 8→target."""

    @property
    def name(self) -> str:
        return "KeyShapeRecolorSolver"

    @property
    def priority(self) -> int:
        return 101

    def _extract_key(self, inp):
        H, W = len(inp), len(inp[0])
        ones = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == 1]
        if not ones:
            return None
        min_r = min(r for r, _ in ones)
        min_c = min(c for _, c in ones)
        max_r = max(r for r, _ in ones)
        max_c = max(c for _, c in ones)
        if max_r - min_r != 2 or max_c - min_c != 2:
            return None
        return tuple(
            tuple(inp[min_r + dr][min_c + dc] for dc in range(3))
            for dr in range(3)
        )

    def solve(self, ctx: TaskContext):
        key_map: dict = {}
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            key = self._extract_key(inp)
            if key is None:
                return None
            H, W = len(inp), len(inp[0])
            target = None
            for r in range(H):
                for c in range(W):
                    if inp[r][c] == 8:
                        t = out[r][c]
                        if t != 0:
                            if target is not None and target != t:
                                return None
                            target = t
            if target is None:
                return None
            key_map[key] = target

        preds = []
        for inp in ctx.test_inputs:
            key = self._extract_key(inp)
            if key not in key_map:
                return None
            target = key_map[key]
            H, W = len(inp), len(inp[0])
            out = []
            for r in range(H):
                row = []
                for c in range(W):
                    v = inp[r][c]
                    if v == 8:
                        row.append(target)
                    elif v == 1:
                        row.append(0)
                    else:
                        row.append(v)
                out.append(row)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


@register_solver
class StaircaseBlockPlaceSolver(BaseSolver):
    """p102 — task 03560426: sort solid-color blocks by min_col, place diagonally."""

    @property
    def name(self) -> str:
        return "StaircaseBlockPlaceSolver"

    @property
    def priority(self) -> int:
        return 102

    def _find_blocks(self, inp):
        H, W = len(inp), len(inp[0])
        visited = [[False] * W for _ in range(H)]
        blocks = []
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v == 0 or visited[r][c]:
                    continue
                cells: set = set()
                stack = [(r, c)]
                while stack:
                    nr, nc = stack.pop()
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    if visited[nr][nc] or inp[nr][nc] != v:
                        continue
                    visited[nr][nc] = True
                    cells.add((nr, nc))
                    stack.extend([(nr+1,nc),(nr-1,nc),(nr,nc+1),(nr,nc-1)])
                min_r = min(rr for rr, _ in cells)
                max_r = max(rr for rr, _ in cells)
                min_c = min(cc for _, cc in cells)
                max_c = max(cc for _, cc in cells)
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                if len(cells) != h * w:
                    return None
                blocks.append({"color": v, "h": h, "w": w, "min_col": min_c})
        return blocks

    def _build_output(self, blocks, H, W):
        out = [[0] * W for _ in range(H)]
        sorted_b = sorted(blocks, key=lambda b: b["min_col"])
        row_off = col_off = 0
        for i, b in enumerate(sorted_b):
            h, w, color = b["h"], b["w"], b["color"]
            for dr in range(h):
                for dc in range(w):
                    rr, cc = row_off + dr, col_off + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr][cc] = color
            if i < len(sorted_b) - 1:
                row_off += h - 1
                col_off += w - 1
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H, W = len(inp), len(inp[0])
            blocks = self._find_blocks(inp)
            if blocks is None:
                return None
            if self._build_output(blocks, H, W) != [list(r) for r in out]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            H, W = len(inp), len(inp[0])
            blocks = self._find_blocks(inp)
            if blocks is None:
                return None
            preds.append(_mk(self._build_output(blocks, H, W), self.name))
        return _res(ctx.task_id, preds)


@register_solver
class ExtendPatternBy3RowsSolver(BaseSolver):
    """p103 — task 017c7c7b: extend each column by its period; replace 1→2."""

    @property
    def name(self) -> str:
        return "ExtendPatternBy3RowsSolver"

    @property
    def priority(self) -> int:
        return 103

    def _build_output(self, inp, extra: int):
        H, W = len(inp), len(inp[0])
        H_out = H + extra
        out = []
        for r in range(H_out):
            row = []
            for c in range(W):
                col_vals = [inp[i][c] for i in range(H)]
                P = _col_period(col_vals)
                val = col_vals[r % P]
                row.append(2 if val != 0 else 0)
            out.append(row)
        return out

    def solve(self, ctx: TaskContext):
        extra = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            e = len(out) - len(inp)
            if e <= 0:
                return None
            if extra is None:
                extra = e
            elif extra != e:
                return None
            if self._build_output(inp, e) != [list(r) for r in out]:
                return None
        if extra is None:
            return None
        preds = []
        for inp in ctx.test_inputs:
            preds.append(_mk(self._build_output(inp, extra), self.name))
        return _res(ctx.task_id, preds)


@register_solver
class ColorSwapPairsSolver(BaseSolver):
    """p104 — task 0d3d703e: all rows identical; learn color swap table and apply."""

    @property
    def name(self) -> str:
        return "ColorSwapPairsSolver"

    @property
    def priority(self) -> int:
        return 104

    def solve(self, ctx: TaskContext):
        swap: dict = {}
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            if len(set(tuple(r) for r in inp)) != 1:
                return None
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            for ci, co in zip(inp[0], out[0]):
                if ci in swap and swap[ci] != co:
                    return None
                swap[ci] = co
        # Require at least 2 distinct input colors to avoid over-firing
        if len(swap) < 2:
            return None
        preds = []
        for inp in ctx.test_inputs:
            H, W = len(inp), len(inp[0])
            out = []
            for r in range(H):
                row = []
                for c in range(W):
                    v = inp[r][c]
                    if v not in swap:
                        return None
                    row.append(swap[v])
                out.append(row)
            preds.append(_mk(out, self.name))
        return _res(ctx.task_id, preds)


@register_solver
class SplitIntersectSolver(BaseSolver):
    """p105 — task 0520fde7: split at separator col (5s); left AND right both non-zero → 2."""

    @property
    def name(self) -> str:
        return "SplitIntersectSolver"

    @property
    def priority(self) -> int:
        return 105

    def _apply(self, inp):
        H = len(inp)
        W = len(inp[0]) if inp else 0
        sep_col = next(
            (c for c in range(W) if all(inp[r][c] == 5 for r in range(H))),
            None,
        )
        if sep_col is None:
            return None
        left_w = sep_col
        right_w = W - sep_col - 1
        if left_w != right_w:
            return None
        out = []
        for r in range(H):
            row = [
                2 if (inp[r][c] != 0 and inp[r][sep_col + 1 + c] != 0) else 0
                for c in range(left_w)
            ]
            out.append(row)
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


@register_solver
class FractalizeSolver(BaseSolver):
    """p106 — task 007bbfb7: fractal tile — non-zero cell → place full input; zero → zeros."""

    @property
    def name(self) -> str:
        return "FractalizeSolver"

    @property
    def priority(self) -> int:
        return 106

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        out = [[0] * (W * W) for _ in range(H * H)]
        for gr in range(H):
            for gc in range(W):
                if inp[gr][gc] != 0:
                    for dr in range(H):
                        for dc in range(W):
                            out[gr * H + dr][gc * W + dc] = inp[dr][dc]
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


@register_solver
class FractalizeComplementSolver(BaseSolver):
    """p107 — task 0692e18c: fractal tile — non-zero cell → place INVERTED input."""

    @property
    def name(self) -> str:
        return "FractalizeComplementSolver"

    @property
    def priority(self) -> int:
        return 107

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        nz = set(v for row in inp for v in row if v != 0)
        if len(nz) != 1:
            return None
        color = next(iter(nz))
        out = [[0] * (W * W) for _ in range(H * H)]
        for gr in range(H):
            for gc in range(W):
                if inp[gr][gc] != 0:
                    for dr in range(H):
                        for dc in range(W):
                            v = inp[dr][dc]
                            out[gr * H + dr][gc * W + dc] = 0 if v != 0 else color
        return out

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)


@register_solver
class TileRotateSolver(BaseSolver):
    """p108 — task 00576224: 2×2 → 6×6; rows [r0,r1,rev(r0),rev(r1),r0,r1] each tiled ×3."""

    @property
    def name(self) -> str:
        return "TileRotateSolver"

    @property
    def priority(self) -> int:
        return 108

    def _apply(self, inp):
        if len(inp) < 2:
            return None
        r0 = list(inp[0])
        r1 = list(inp[1])
        rev0 = list(reversed(r0))
        rev1 = list(reversed(r1))
        return [row * 3 for row in [r0, r1, rev0, rev1, r0, r1]]

    def solve(self, ctx: TaskContext):
        for pair in ctx.train_pairs:
            exp = self._apply(pair["input"])
            if exp is None:
                return None
            if exp != [list(r) for r in pair["output"]]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(_mk(g, self.name))
        return _res(ctx.task_id, preds)
