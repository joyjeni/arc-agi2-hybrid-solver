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

@register_solver
class MirrorTileSolver(BaseSolver):
    """
    2×2 tile arrangement using input, flip_h, flip_v, rot180.
    Learns which of the 4 quadrant-arrangement variants matches training pairs.
    Targets: 0c786b71, 3af2c5a8, 62c24649, 67e8384a, 833dafe3.
    """

    @property
    def name(self) -> str:
        return "MirrorTileSolver"

    @property
    def priority(self) -> int:
        return 47

    @staticmethod
    def _variants(inp: Grid) -> dict:
        """Return all 4 canonical 2×2-tile arrangements keyed by variant id."""
        o = inp
        h = flip_h(inp)
        v = flip_v(inp)
        r = flip_h(flip_v(inp))   # rot180
        R, C = len(inp), len(inp[0])

        def tile(tl, tr, bl, br):
            top = [tl[row] + tr[row] for row in range(R)]
            bot = [bl[row] + br[row] for row in range(R)]
            return top + bot

        return {
            "OHVR": tile(o, h, v, r),
            "HOVR": tile(h, o, r, v),   # orig shifted to TR
            "VROH": tile(v, r, o, h),   # orig shifted to BL
            "RVHO": tile(r, v, h, o),   # orig shifted to BR
        }

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        # Determine which variant is consistent across all training pairs
        chosen = None
        for pair in ctx.train_pairs:
            inp, out = pair["input"], pair["output"]
            H, W = len(inp), len(inp[0])
            if len(out) != 2 * H or len(out[0]) != 2 * W:
                return None
            variants = self._variants(inp)
            matches = {k for k, v in variants.items() if v == out}
            if not matches:
                return None
            if chosen is None:
                chosen = matches
            else:
                chosen = chosen & matches
        if not chosen:
            return None
        variant_key = next(iter(chosen))
        preds = []
        for inp in ctx.test_inputs:
            variants = self._variants(inp)
            result = variants[variant_key]
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
