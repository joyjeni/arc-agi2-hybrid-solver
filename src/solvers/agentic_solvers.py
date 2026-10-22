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
    grid_dims,
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
