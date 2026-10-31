"""
advanced_solvers33.py — Four pattern-targeted solvers for ARC-AGI-2.

1. FloodFillBorderSolver       (T19, ~165 tasks)
   — BFS from border zeros through connected zeros; recolor them to a
     learned fill-color inferred from training pair diffs.

2. MirrorCompletionSolver      (T12, ~124 tasks)
   — Half-filled grid (one half zeros, the other half non-zero); fill the
     zero half by reflecting the non-zero half along H or V axis.

3. ConnectSameColorObjectsSolver (T20, ~30 tasks)
   — Draw straight H / V / diagonal lines of each color between same-color
     disconnected objects whose centroids are axis- or diagonally-aligned.

4. UniqueColorRemarkSolver     (T18, ~52 tasks)
   — Cells whose color appears exactly once in the input get recolored to a
     learned target color inferred from training pair diffs.
"""

from collections import deque, Counter
from typing import List, Optional, Tuple

from src.core.interfaces import BaseSolver, Grid, Prediction, SolverResult, TaskContext
from src.solvers.registry import register_solver


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    return all(pred[r] == expected[r] for r in range(len(pred)))


def _bfs_border_zeros(grid: Grid) -> List[Tuple[int, int]]:
    """Return all (r,c) positions reachable from the border via 4-connected zero cells."""
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    queue: deque = deque()
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == 0:
                visited[r][c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == 0:
                visited[nr][nc] = True
                queue.append((nr, nc))
    return [(r, c) for r in range(h) for c in range(w) if visited[r][c]]


# ─────────────────────────────────────────────────────────────────────────────
# 1. EnclosedFloodFillSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class EnclosedFloodFillSolver(BaseSolver):
    """
    T19 pattern: zeros that are NOT reachable from the border (enclosed/interior
    zeros) are replaced by a single non-zero color in the output.
    Everything else stays the same.
    Learns the fill color from training pair diffs; self-validates on all pairs.
    """

    @property
    def name(self) -> str:
        return "EnclosedFloodFillSolver"

    def _enclosed_zeros(self, grid: Grid) -> List[Tuple[int, int]]:
        """Return (r,c) positions of interior zeros not reachable from border."""
        h, w = len(grid), len(grid[0])
        border_reach = set(_bfs_border_zeros(grid))
        return [(r, c) for r in range(h) for c in range(w)
                if grid[r][c] == 0 and (r, c) not in border_reach]

    def _learn_fill_color(self, train_pairs) -> Optional[int]:
        """Infer the consistent fill color from training pair diffs."""
        target_colors = []
        for pair in train_pairs:
            inp = pair["input"]
            out = pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            enc = self._enclosed_zeros(inp)
            if not enc:
                return None
            # What did those enclosed zeros become in the output?
            colors_found = {out[r][c] for r, c in enc if out[r][c] != 0}
            if len(colors_found) != 1:
                return None
            target_colors.append(next(iter(colors_found)))
        if not target_colors or len(set(target_colors)) != 1:
            return None
        return target_colors[0]

    def _apply(self, grid: Grid, fill_color: int) -> Grid:
        result = _copy_grid(grid)
        for (r, c) in self._enclosed_zeros(grid):
            result[r][c] = fill_color
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        fill_color = self._learn_fill_color(ctx.train_pairs)
        if fill_color is None:
            return None

        # Validate on all training pairs
        for pair in ctx.train_pairs:
            pred = self._apply(pair["input"], fill_color)
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = self._apply(test_input, fill_color)
            predictions.append(
                Prediction(
                    attempt_1=result,
                    attempt_2=result,
                    solver_name=self.name,
                    confidence=1.0,
                )
            )
        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. MirrorCompletionSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class MirrorCompletionSolver(BaseSolver):
    """
    T12 pattern: one half of the grid is all zeros; the other half contains the
    pattern.  Output is the full grid with the zero half filled by reflecting the
    non-zero half across the horizontal or vertical mid-axis.
    """

    @property
    def name(self) -> str:
        return "MirrorCompletionSolver"

    def _detect_axis_and_apply(self, grid: Grid) -> Optional[Grid]:
        """Return completed grid if exactly one axis/half split works, else None."""
        h, w = len(grid), len(grid[0])
        result = _copy_grid(grid)

        def all_zero(cells) -> bool:
            return all(grid[r][c] == 0 for r, c in cells)

        def has_nonzero(cells) -> bool:
            return any(grid[r][c] != 0 for r, c in cells)

        # --- Horizontal split: top half zeros, reflect bottom → top ---
        if h >= 2:
            mid_h = h // 2
            top_cells = [(r, c) for r in range(mid_h) for c in range(w)]
            bot_cells = [(r, c) for r in range(mid_h, h) for c in range(w)]
            if all_zero(top_cells) and has_nonzero(bot_cells):
                for r in range(mid_h):
                    for c in range(w):
                        mirror_r = h - 1 - r
                        if 0 <= mirror_r < h:
                            result[r][c] = grid[mirror_r][c]
                return result
            # Bottom half zeros, reflect top → bottom
            result2 = _copy_grid(grid)
            if all_zero(bot_cells) and has_nonzero(top_cells):
                for r in range(mid_h, h):
                    for c in range(w):
                        mirror_r = h - 1 - r
                        if 0 <= mirror_r < h:
                            result2[r][c] = grid[mirror_r][c]
                return result2

        # --- Vertical split: left half zeros, reflect right → left ---
        if w >= 2:
            mid_w = w // 2
            left_cells = [(r, c) for r in range(h) for c in range(mid_w)]
            right_cells = [(r, c) for r in range(h) for c in range(mid_w, w)]
            result3 = _copy_grid(grid)
            if all_zero(left_cells) and has_nonzero(right_cells):
                for r in range(h):
                    for c in range(mid_w):
                        mirror_c = w - 1 - c
                        if 0 <= mirror_c < w:
                            result3[r][c] = grid[r][mirror_c]
                return result3
            # Right half zeros, reflect left → right
            result4 = _copy_grid(grid)
            if all_zero(right_cells) and has_nonzero(left_cells):
                for r in range(h):
                    for c in range(mid_w, w):
                        mirror_c = w - 1 - c
                        if 0 <= mirror_c < w:
                            result4[r][c] = grid[r][mirror_c]
                return result4

        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate on all training pairs
        for pair in ctx.train_pairs:
            pred = self._detect_axis_and_apply(pair["input"])
            if pred is None:
                return None
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = self._detect_axis_and_apply(test_input)
            if result is None:
                return None
            predictions.append(
                Prediction(
                    attempt_1=result,
                    attempt_2=result,
                    solver_name=self.name,
                    confidence=1.0,
                )
            )
        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ConnectSameColorObjectsSolver
# ─────────────────────────────────────────────────────────────────────────────

def _find_objects(grid: Grid, bg: int = 0) -> List[List[Tuple[int, int]]]:
    """4-connected connected components of non-bg cells."""
    h, w = len(grid), len(grid[0])
    visited = [[False] * w for _ in range(h)]
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and not visited[r][c]:
                color = grid[r][c]
                component = []
                queue: deque = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w
                                and not visited[nr][nc]
                                and grid[nr][nc] == color):
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                objects.append(component)
    return objects


def _centroid(cells: List[Tuple[int, int]]) -> Tuple[float, float]:
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return sum(rs) / len(rs), sum(cs) / len(cs)


def _draw_hv_diagonal(result: Grid, r1: int, c1: int, r2: int, c2: int, color: int) -> None:
    """Draw line (H, V, or 45° diagonal) between (r1,c1) and (r2,c2)."""
    dr = r2 - r1
    dc = c2 - c1
    h, w = len(result), len(result[0])
    if dr == 0:  # horizontal
        for c in range(min(c1, c2), max(c1, c2) + 1):
            result[r1][c] = color
    elif dc == 0:  # vertical
        for r in range(min(r1, r2), max(r1, r2) + 1):
            result[r][c1] = color
    elif abs(dr) == abs(dc):  # diagonal
        steps = abs(dr)
        step_r = 1 if dr > 0 else -1
        step_c = 1 if dc > 0 else -1
        for i in range(steps + 1):
            nr, nc = r1 + i * step_r, c1 + i * step_c
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = color


def _apply_connections(grid: Grid, bg: int = 0) -> Grid:
    """Connect same-color objects that share a row, column, or 45° diagonal."""
    result = _copy_grid(grid)
    objects = _find_objects(grid, bg)
    if not objects:
        return result

    # Group by color
    from collections import defaultdict
    by_color: dict = defaultdict(list)
    for obj in objects:
        color = grid[obj[0][0]][obj[0][1]]
        by_color[color].append(obj)

    for color, objs in by_color.items():
        if len(objs) < 2:
            continue
        centroids = [_centroid(o) for o in objs]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                ri, ci_ = centroids[i]
                rj, cj = centroids[j]
                dr = rj - ri
                dc = cj - ci_
                # Same row (H), same column (V), or 45° diagonal
                if abs(dr) < 0.5:  # same row
                    _draw_hv_diagonal(result, round(ri), round(ci_), round(rj), round(cj), color)
                elif abs(dc) < 0.5:  # same column
                    _draw_hv_diagonal(result, round(ri), round(ci_), round(rj), round(cj), color)
                elif abs(abs(dr) - abs(dc)) < 0.5:  # diagonal
                    _draw_hv_diagonal(result, round(ri), round(ci_), round(rj), round(cj), color)
    return result


@register_solver
class ConnectSameColorObjectsSolver(BaseSolver):
    """
    T20 pattern (H/V/diagonal sub-types): same-color disconnected objects whose
    centroids share a row, column, or 45° diagonal are connected by a line of
    the same color.  Background is assumed 0.
    """

    @property
    def name(self) -> str:
        return "ConnectSameColorObjectsSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate on all training pairs
        for pair in ctx.train_pairs:
            pred = _apply_connections(pair["input"])
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _apply_connections(test_input)
            predictions.append(
                Prediction(
                    attempt_1=result,
                    attempt_2=result,
                    solver_name=self.name,
                    confidence=1.0,
                )
            )
        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. UniqueColorRemarkSolver
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class UniqueColorRemarkSolver(BaseSolver):
    """
    T18 pattern: cells whose color appears exactly once in the entire input grid
    get recolored to a single learned target color in the output.  Everything
    else stays the same.  Target color is inferred from training pair diffs and
    must be consistent across all pairs.
    """

    @property
    def name(self) -> str:
        return "UniqueColorRemarkSolver"

    def _unique_color_cells(self, grid: Grid) -> List[Tuple[int, int]]:
        """Return (r,c) positions whose color value appears exactly once in the grid."""
        h, w = len(grid), len(grid[0])
        counts: Counter = Counter()
        for row in grid:
            counts.update(row)
        unique_colors = {color for color, cnt in counts.items() if cnt == 1}
        return [(r, c) for r in range(h) for c in range(w) if grid[r][c] in unique_colors]

    def _learn_target_color(self, train_pairs) -> Optional[int]:
        target_colors = []
        for pair in train_pairs:
            inp = pair["input"]
            out = pair["output"]
            # Input and output must have the same shape
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            unique_cells = self._unique_color_cells(inp)
            if not unique_cells:
                return None
            # What did those cells become in output?
            new_colors = set()
            for (r, c) in unique_cells:
                new_colors.add(out[r][c])
            if len(new_colors) != 1:
                return None
            target_colors.append(new_colors.pop())
        if not target_colors:
            return None
        if len(set(target_colors)) != 1:
            return None
        return target_colors[0]

    def _apply(self, grid: Grid, target_color: int) -> Grid:
        result = _copy_grid(grid)
        for (r, c) in self._unique_color_cells(grid):
            result[r][c] = target_color
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        target_color = self._learn_target_color(ctx.train_pairs)
        if target_color is None:
            return None

        # Validate on all training pairs
        for pair in ctx.train_pairs:
            pred = self._apply(pair["input"], target_color)
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = self._apply(test_input, target_color)
            predictions.append(
                Prediction(
                    attempt_1=result,
                    attempt_2=result,
                    solver_name=self.name,
                    confidence=1.0,
                )
            )
        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )
