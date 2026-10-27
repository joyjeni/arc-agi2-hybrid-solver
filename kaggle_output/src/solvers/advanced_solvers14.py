"""
advanced_solvers14.py — 5 new crop-rule solvers (all rules confirmed)

  1. Color5BBoxExpandSolver        (task 3f7978a0)
  2. MedianColorObjectSolver       (task 72ca375d)
  3. AsymmetricComponentCropSolver (task 73ccf9c2)
  4. MaxColor2InBBoxSolver         (task 8efcae92)
  5. SeparatorGridCropSolver       (task a644e277)
"""

from collections import Counter, deque
from typing import List, Optional, Tuple

from src.core.interfaces import (
    BaseSolver, TaskContext, SolverResult, Prediction, Grid
)
from .registry import register_solver


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bbox(cells) -> Tuple[int, int, int, int]:
    """Return (r0, c0, r1, c1) bounding box for a list of (r,c) cells."""
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    r1 = max(r for r, c in cells)
    c1 = max(c for r, c in cells)
    return r0, c0, r1, c1


def _extract(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    return [list(row[c0: c1 + 1]) for row in grid[r0: r1 + 1]]


def _components_8(grid: Grid, color: int) -> List[List[Tuple[int, int]]]:
    """BFS 8-connected components for a specific color (pure Python)."""
    H, W = len(grid), len(grid[0])
    visited = [[False] * W for _ in range(H)]
    comps = []
    for sr in range(H):
        for sc in range(W):
            if grid[sr][sc] == color and not visited[sr][sc]:
                queue = deque([(sr, sc)])
                visited[sr][sc] = True
                comp = []
                while queue:
                    r, c = queue.popleft()
                    comp.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and \
                               not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                comps.append(comp)
    return comps


def _make_result(ctx: TaskContext, preds: List[Grid]) -> SolverResult:
    predictions = [
        Prediction(attempt_1=p, attempt_2=p, solver_name=ctx.task_id)
        for p in preds
    ]
    return SolverResult(task_id=ctx.task_id, predictions=predictions,
                        solved=True, tier=1)


# ---------------------------------------------------------------------------
# 1. Color5BBoxExpandSolver  —  task 3f7978a0
# ---------------------------------------------------------------------------

@register_solver
class Color5BBoxExpandSolver(BaseSolver):
    """
    Find tight bounding box of all color-5 cells.
    Expand by exactly +1 row above and +1 row below (NO column expansion).
    Clamp to grid boundaries.
    Confirmed on: 3f7978a0.
    """

    @property
    def name(self) -> str:
        return "Color5BBoxExpandSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    def _predict(self, grid: Grid) -> Optional[Grid]:
        H, W = len(grid), len(grid[0])
        cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 5]
        if not cells:
            return None
        r0, c0, r1, c1 = _bbox(cells)
        # Expand rows only
        r0 = max(0, r0 - 1)
        r1 = min(H - 1, r1 + 1)
        return _extract(grid, r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# 2. MedianColorObjectSolver  —  task 72ca375d
# ---------------------------------------------------------------------------

@register_solver
class MedianColorObjectSolver(BaseSolver):
    """
    Exactly 3 colored (non-background) objects per pair.
    Select the unique object whose bounding-box crop is left-right symmetric.
    Return that object's bounding-box crop (preserving background zeros).
    Confirmed on: 72ca375d.
    """

    @property
    def name(self) -> str:
        return "MedianColorObjectSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    @staticmethod
    def _is_lr_symmetric(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> bool:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mirror_c = c1 - (c - c0)
                if grid[r][c] != grid[r][mirror_c]:
                    return False
        return True

    def _predict(self, grid: Grid) -> Optional[Grid]:
        H, W = len(grid), len(grid[0])
        flat = [grid[r][c] for r in range(H) for c in range(W)]
        bg = Counter(flat).most_common(1)[0][0]
        colors = sorted(set(flat) - {bg})
        if len(colors) != 3:
            return None
        symmetric = []
        for col in colors:
            cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == col]
            r0, c0, r1, c1 = _bbox(cells)
            if self._is_lr_symmetric(grid, r0, c0, r1, c1):
                symmetric.append((r0, c0, r1, c1))
        if len(symmetric) != 1:
            return None
        r0, c0, r1, c1 = symmetric[0]
        return _extract(grid, r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# 3. AsymmetricComponentCropSolver  —  task 73ccf9c2
# ---------------------------------------------------------------------------

@register_solver
class AsymmetricComponentCropSolver(BaseSolver):
    """
    Find the unique 8-connected component (of any non-zero color) that is
    NOT left-right (horizontally) symmetric within its own bounding box.
    Return that component's bounding-box crop of the raw input.
    Confirmed on: 73ccf9c2.
    """

    @property
    def name(self) -> str:
        return "AsymmetricComponentCropSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    @staticmethod
    def _is_lr_symmetric(grid: Grid, r0: int, c0: int,
                         r1: int, c1: int) -> bool:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                mirror_c = c1 - (c - c0)
                if grid[r][c] != grid[r][mirror_c]:
                    return False
        return True

    def _predict(self, grid: Grid) -> Optional[Grid]:
        H, W = len(grid), len(grid[0])
        flat = [grid[r][c] for r in range(H) for c in range(W)]
        bg = Counter(flat).most_common(1)[0][0]
        # All non-bg colors
        colors = set(flat) - {bg}
        if not colors:
            return None
        # Collect all 8-connected components across all non-bg colors
        all_comps: List[Tuple[List, int, int, int, int]] = []
        for col in colors:
            for comp in _components_8(grid, col):
                r0, c0, r1, c1 = _bbox(comp)
                all_comps.append((comp, r0, c0, r1, c1))
        if not all_comps:
            return None
        asymmetric = []
        for comp, r0, c0, r1, c1 in all_comps:
            if not self._is_lr_symmetric(grid, r0, c0, r1, c1):
                asymmetric.append((r0, c0, r1, c1))
        if len(asymmetric) != 1:
            return None
        r0, c0, r1, c1 = asymmetric[0]
        return _extract(grid, r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# 4. MaxColor2InBBoxSolver  —  task 8efcae92
# ---------------------------------------------------------------------------

@register_solver
class MaxColor2InBBoxSolver(BaseSolver):
    """
    Find all 8-connected components of color-1 cells.
    For each, count color-2 single-pixel cells inside the component's bbox.
    Pick the component whose bbox contains the MOST color-2 pixels.
    Crop the RAW input to that bbox (output preserves embedded color-2 cells).
    Confirmed on: 8efcae92.
    """

    @property
    def name(self) -> str:
        return "MaxColor2InBBoxSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    def _predict(self, grid: Grid) -> Optional[Grid]:
        comps = _components_8(grid, 1)
        if not comps:
            return None
        best_count = -1
        best_bbox = None
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            # Count color-2 single-pixel cells inside this bbox
            count = 0
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    if grid[r][c] == 2:
                        # Verify it is a single pixel (no 4-connected color-2 neighbors)
                        H, W = len(grid), len(grid[0])
                        neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                        isolated = all(
                            not (0 <= nr < H and 0 <= nc < W and
                                 grid[nr][nc] == 2)
                            for nr, nc in neighbors
                        )
                        if isolated:
                            count += 1
            if count > best_count:
                best_count = count
                best_bbox = (r0, c0, r1, c1)
        if best_bbox is None:
            return None
        r0, c0, r1, c1 = best_bbox
        return _extract(grid, r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# 5. SeparatorGridCropSolver  —  task a644e277
# ---------------------------------------------------------------------------

@register_solver
class SeparatorGridCropSolver(BaseSolver):
    """
    The grid contains a background color, a separator color, and an inner
    cell color.  Separator rows/cols form a lattice:
      - Strong separators: fully uniform rows/cols of sep_color.
      - Weak separators:  rows/cols with only {bg, sep} values, more sep than
                          bg, AND at least one bg pixel (occurs at intersections
                          with other weak separators).
    Return input[r1:r2+1, c1:c2+1] where r1,r2 are the first two weak-sep
    rows and c1,c2 are the first two weak-sep cols.
    Confirmed on: a644e277.
    """

    @property
    def name(self) -> str:
        return "SeparatorGridCropSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    @staticmethod
    def _find_sep_color(grid: Grid, bg: int) -> Optional[int]:
        H, W = len(grid), len(grid[0])
        candidates = set()
        # Any color that fills at least one complete row or col
        for r in range(H):
            row_vals = set(grid[r])
            if len(row_vals) == 1 and list(row_vals)[0] != bg:
                candidates.add(list(row_vals)[0])
        for c in range(W):
            col_vals = set(grid[r][c] for r in range(H))
            if len(col_vals) == 1 and list(col_vals)[0] != bg:
                candidates.add(list(col_vals)[0])
        if len(candidates) == 1:
            return candidates.pop()
        if not candidates:
            return None
        # Multiple candidates: pick most frequent among them
        flat = [grid[r][c] for r in range(H) for c in range(W)]
        cnt = Counter(flat)
        return max(candidates, key=lambda v: cnt[v])

    @staticmethod
    def _weak_separators(grid: Grid, bg: int, sep: int):
        H, W = len(grid), len(grid[0])
        allowed = {bg, sep}
        weak_rows = []
        for r in range(H):
            row = [grid[r][c] for c in range(W)]
            if set(row) <= allowed:
                n_sep = row.count(sep)
                n_bg = row.count(bg)
                if n_sep > n_bg > 0:
                    weak_rows.append(r)
        weak_cols = []
        for c in range(W):
            col = [grid[r][c] for r in range(H)]
            if set(col) <= allowed:
                n_sep = col.count(sep)
                n_bg = col.count(bg)
                if n_sep > n_bg > 0:
                    weak_cols.append(c)
        return weak_rows, weak_cols

    def _predict(self, grid: Grid) -> Optional[Grid]:
        H, W = len(grid), len(grid[0])
        flat = [grid[r][c] for r in range(H) for c in range(W)]
        bg = Counter(flat).most_common(1)[0][0]
        sep = self._find_sep_color(grid, bg)
        if sep is None:
            return None
        weak_rows, weak_cols = self._weak_separators(grid, bg, sep)
        if len(weak_rows) < 2 or len(weak_cols) < 2:
            return None
        r1, r2 = weak_rows[0], weak_rows[1]
        c1, c2 = weak_cols[0], weak_cols[1]
        return _extract(grid, r1, c1, r2, c2)
