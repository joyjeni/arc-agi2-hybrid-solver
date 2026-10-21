"""
DSA Deterministic Solvers — Tier-1 fast solvers using classical algorithms.

Each solver is independent and registered automatically via @register_solver.
Patterns covered:
  - Color-frequency identity rule
  - Grid reflection / symmetry completion
  - Gravity fill (objects fall to bottom)
  - Connected-component recolor by size
  - Row/column sort
  - Bounding-box extraction
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ..core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from ..core.grid_codec import (
    Grid, copy_grid, color_histogram, background_color,
    rotate90, flip_h, flip_v, transpose, grid_equal, grid_dims,
)
from .registry import register_solver


# ── Utilities ─────────────────────────────────────────────────────────────────

def _bfs_connected(grid: Grid, r: int, c: int, color: int) -> list[tuple[int, int]]:
    h, w = len(grid), len(grid[0])
    visited: set = set()
    queue = [(r, c)]
    visited.add((r, c))
    while queue:
        cr, cc = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                if grid[nr][nc] == color:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return list(visited)


def _connected_components(grid: Grid, exclude_bg: bool = True) -> list[list[tuple[int, int]]]:
    h, w = len(grid), len(grid[0])
    bg = background_color(grid) if exclude_bg else -1
    visited: set = set()
    comps: list = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in visited and grid[r][c] != bg:
                comp = _bfs_connected(grid, r, c, grid[r][c])
                visited.update(comp)
                comps.append(comp)
    return comps


def _make_prediction(grid: Grid, solver_name: str, confidence: float = 1.0) -> Prediction:
    return Prediction(
        attempt_1=grid,
        attempt_2=grid,
        solver_name=solver_name,
        confidence=confidence,
    )


def _result(task_id: str, preds: list[Prediction], tier: int = 1) -> SolverResult:
    return SolverResult(task_id=task_id, predictions=preds, solved=True, tier=tier)


# ── Solvers ────────────────────────────────────────────────────────────────────

@register_solver
class IdentitySolver(BaseSolver):
    """Output = Input (sanity baseline; very low priority)."""

    @property
    def name(self) -> str:
        return "IdentitySolver"

    @property
    def priority(self) -> int:
        return 999  # last resort

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        preds = [_make_prediction(inp, self.name, 0.01) for inp in ctx.test_inputs]
        return _result(ctx.task_id, preds)


@register_solver
class ColorFrequencyRuleSolver(BaseSolver):
    """
    Output is a 1×1 grid containing the color that appears with a
    consistent frequency rule across all training pairs.
    E.g. 'output = most-common color in input'.
    """

    @property
    def name(self) -> str:
        return "ColorFrequencyRuleSolver"

    @property
    def priority(self) -> int:
        return 10

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Learn: does output == [[most_common_color(input)]] for all pairs?
        for rule_fn in [
            lambda g: [[max(color_histogram(g), key=color_histogram(g).get)]],
            lambda g: [[min(color_histogram(g), key=color_histogram(g).get)]],
        ]:
            if all(
                rule_fn(p["input"]) == p["output"]
                for p in ctx.train_pairs
            ):
                preds = [_make_prediction(rule_fn(inp), self.name) for inp in ctx.test_inputs]
                return _result(ctx.task_id, preds)
        return None


@register_solver
class GravityFallSolver(BaseSolver):
    """Non-background pixels fall to the bottom row (gravity)."""

    @property
    def name(self) -> str:
        return "GravityFallSolver"

    @property
    def priority(self) -> int:
        return 20

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        bg = background_color(grid)
        result = [[bg] * w for _ in range(h)]
        for c in range(w):
            col = [grid[r][c] for r in range(h) if grid[r][c] != bg]
            for i, v in enumerate(reversed(col)):
                result[h - 1 - i][c] = v
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            if self._apply(pair["input"]) != pair["output"]:
                return None
        preds = [_make_prediction(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _result(ctx.task_id, preds)


@register_solver
class ReflectHorizontalSolver(BaseSolver):
    """Output = horizontal flip of input."""

    @property
    def name(self) -> str:
        return "ReflectHorizontalSolver"

    @property
    def priority(self) -> int:
        return 15

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if all(flip_h(p["input"]) == p["output"] for p in ctx.train_pairs):
            preds = [_make_prediction(flip_h(inp), self.name) for inp in ctx.test_inputs]
            return _result(ctx.task_id, preds)
        return None


@register_solver
class ReflectVerticalSolver(BaseSolver):
    """Output = vertical flip of input."""

    @property
    def name(self) -> str:
        return "ReflectVerticalSolver"

    @property
    def priority(self) -> int:
        return 15

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if all(flip_v(p["input"]) == p["output"] for p in ctx.train_pairs):
            preds = [_make_prediction(flip_v(inp), self.name) for inp in ctx.test_inputs]
            return _result(ctx.task_id, preds)
        return None


@register_solver
class Rotate90Solver(BaseSolver):
    """Output = 90° clockwise rotation of input."""

    @property
    def name(self) -> str:
        return "Rotate90Solver"

    @property
    def priority(self) -> int:
        return 15

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if all(rotate90(p["input"]) == p["output"] for p in ctx.train_pairs):
            preds = [_make_prediction(rotate90(inp), self.name) for inp in ctx.test_inputs]
            return _result(ctx.task_id, preds)
        return None


@register_solver
class TransposeSolver(BaseSolver):
    """Output = transpose of input."""

    @property
    def name(self) -> str:
        return "TransposeSolver"

    @property
    def priority(self) -> int:
        return 15

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if all(transpose(p["input"]) == p["output"] for p in ctx.train_pairs):
            preds = [_make_prediction(transpose(inp), self.name) for inp in ctx.test_inputs]
            return _result(ctx.task_id, preds)
        return None


@register_solver
class CCRecolorBySizeSolver(BaseSolver):
    """
    Recolor connected components by size rank:
    largest → color A, second → color B, etc.
    Learns color mapping from training pairs.
    """

    @property
    def name(self) -> str:
        return "CCRecolorBySizeSolver"

    @property
    def priority(self) -> int:
        return 30

    @staticmethod
    def _rank_color_map(inp: Grid, out: Grid) -> Optional[dict[int, int]]:
        comps = _connected_components(inp)
        if not comps:
            return None
        comps_sorted = sorted(comps, key=len, reverse=True)
        color_map: dict[int, int] = {}
        for comp in comps_sorted:
            r, c = comp[0]
            old_c = inp[r][c]
            new_c = out[r][c]
            if old_c in color_map and color_map[old_c] != new_c:
                return None
            color_map[old_c] = new_c
        return color_map

    @staticmethod
    def _apply(inp: Grid, color_map: dict[int, int]) -> Grid:
        bg = background_color(inp)
        result = copy_grid(inp)
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                result[r][c] = color_map.get(inp[r][c], inp[r][c])
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        ref_map = self._rank_color_map(
            ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"]
        )
        if ref_map is None:
            return None
        for pair in ctx.train_pairs[1:]:
            m = self._rank_color_map(pair["input"], pair["output"])
            if m != ref_map:
                return None
        preds = [
            _make_prediction(self._apply(inp, ref_map), self.name)
            for inp in ctx.test_inputs
        ]
        return _result(ctx.task_id, preds)


@register_solver
class RemoveSmallObjectsSolver(BaseSolver):
    """Replace connected components smaller than the largest with background."""

    @property
    def name(self) -> str:
        return "RemoveSmallObjectsSolver"

    @property
    def priority(self) -> int:
        return 25

    @staticmethod
    def _apply(grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        bg = background_color(grid)
        comps = _connected_components(grid)
        if not comps:
            return copy_grid(grid)
        max_size = max(len(c) for c in comps)
        keep: set[tuple[int, int]] = set()
        for comp in comps:
            if len(comp) == max_size:
                keep.update(comp)
        result = [[bg] * w for _ in range(h)]
        for r, c in keep:
            result[r][c] = grid[r][c]
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            if self._apply(pair["input"]) != pair["output"]:
                return None
        preds = [_make_prediction(self._apply(inp), self.name) for inp in ctx.test_inputs]
        return _result(ctx.task_id, preds)


@register_solver
class ColorSwapSolver(BaseSolver):
    """Learn a fixed color→color mapping and apply it."""

    @property
    def name(self) -> str:
        return "ColorSwapSolver"

    @property
    def priority(self) -> int:
        return 12

    @staticmethod
    def _learn_map(inp: Grid, out: Grid) -> Optional[dict[int, int]]:
        if grid_dims(inp) != grid_dims(out):
            return None
        h, w = grid_dims(inp)
        cmap: dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                i, o = inp[r][c], out[r][c]
                if i in cmap and cmap[i] != o:
                    return None
                cmap[i] = o
        return cmap

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        ref = self._learn_map(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if ref is None:
            return None
        for pair in ctx.train_pairs[1:]:
            m = self._learn_map(pair["input"], pair["output"])
            if m != ref:
                return None
        def apply(g: Grid) -> Grid:
            return [[ref.get(c, c) for c in row] for row in g]
        preds = [_make_prediction(apply(inp), self.name) for inp in ctx.test_inputs]
        return _result(ctx.task_id, preds)


@register_solver
class RowSortSolver(BaseSolver):
    """Output = rows sorted by their first non-background cell color."""

    @property
    def name(self) -> str:
        return "RowSortSolver"

    @property
    def priority(self) -> int:
        return 40

    @staticmethod
    def _sorted_rows(grid: Grid) -> Grid:
        bg = background_color(grid)
        def key(row):
            for c in row:
                if c != bg:
                    return c
            return -1
        return sorted(grid, key=key)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            if self._sorted_rows(pair["input"]) != pair["output"]:
                return None
        preds = [_make_prediction(self._sorted_rows(inp), self.name) for inp in ctx.test_inputs]
        return _result(ctx.task_id, preds)
