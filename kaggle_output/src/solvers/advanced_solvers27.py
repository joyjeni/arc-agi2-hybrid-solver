"""
advanced_solvers27.py — Four gravity/ring solvers for ARC-AGI-2

1. GravityDownLargeSolver      (task: 6ca952ad) — size≥4 components fall to grid bottom
2. GravityUpBySizeSolver       (task: b2bc3ffd) — each component moves UP by its size
3. GravityDownOneRowGapSolver  (task: 825aa9e9) — components fall to floor_c-2 with stacking
4. ConcentricRingsColorCycleSolver (task: bda2d7a6) — ring colors rotate right by 1
"""

from collections import Counter, deque
from typing import List, Optional

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from src.solvers.registry import register_solver


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    for r in range(len(pred)):
        if list(pred[r]) != list(expected[r]):
            return False
    return True


def _get_bg(grid: Grid) -> int:
    flat = [v for row in grid for v in row]
    return Counter(flat).most_common(1)[0][0]


def _connected_components(grid: Grid, bg: int, exclude: int = None):
    """Return list of components (list of (r,c)) for non-bg, non-exclude cells."""
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []
    for sr in range(rows):
        for sc in range(cols):
            v = grid[sr][sc]
            if v == bg or v == exclude or visited[sr][sc]:
                continue
            comp = []
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        nv = grid[nr][nc]
                        if nv != bg and nv != exclude:
                            visited[nr][nc] = True
                            q.append((nr, nc))
            components.append(comp)
    return components


# ─────────────────────────────────────────────────────────────────────────────
# 1. GravityDownLargeSolver  (6ca952ad)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class GravityDownLargeSolver(BaseSolver):
    """
    Components with size >= 4 fall to the absolute grid bottom (preserving shape).
    Components with size <= 3 stay in place.
    BG = most common color.
    """

    @property
    def name(self) -> str:
        return "GravityDownLargeSolver"

    def _apply(self, inp: Grid, bg: int) -> Grid:
        rows, cols = len(inp), len(inp[0])
        components = _connected_components(inp, bg)
        pred = [list(row) for row in inp]

        for comp in components:
            if len(comp) < 4:
                continue
            color = inp[comp[0][0]][comp[0][1]]
            max_row = max(r for r, c in comp)
            for r, c in comp:
                pred[r][c] = bg
            shift = (rows - 1) - max_row
            for r, c in comp:
                pred[r + shift][c] = color

        return pred

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        bg = _get_bg(train[0]["input"])

        for pair in train:
            pred = self._apply(pair["input"], bg)
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            predictions.append(Prediction(attempt_1=self._apply(test_inp, bg), attempt_2=self._apply(test_inp, bg), solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GravityUpBySizeSolver  (b2bc3ffd)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class GravityUpBySizeSolver(BaseSolver):
    """
    BG = most common color.
    floor_color = most common non-BG color in the bottom row.
    Each non-BG, non-floor connected component moves UP by exactly its size
    (delta_row = -size); shape preserved.
    """

    @property
    def name(self) -> str:
        return "GravityUpBySizeSolver"

    def _get_floor_color(self, inp: Grid, bg: int) -> Optional[int]:
        rows = len(inp)
        non_bg = [v for v in inp[rows - 1] if v != bg]
        if not non_bg:
            return None
        return Counter(non_bg).most_common(1)[0][0]

    def _apply(self, inp: Grid, bg: int, floor_color: int) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])
        components = _connected_components(inp, bg, exclude=floor_color)
        pred = [list(row) for row in inp]

        for comp in components:
            color = inp[comp[0][0]][comp[0][1]]
            size = len(comp)
            delta = -size  # move UP by size

            new_cells = [(r + delta, c) for r, c in comp]
            if any(nr < 0 or nr >= rows for nr, nc in new_cells):
                return None

            for r, c in comp:
                pred[r][c] = bg
            for nr, nc in new_cells:
                pred[nr][nc] = color

        return pred

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        bg = _get_bg(train[0]["input"])

        for pair in train:
            fc = self._get_floor_color(pair["input"], bg)
            if fc is None:
                return None
            pred = self._apply(pair["input"], bg, fc)
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            fc = self._get_floor_color(test_inp, bg)
            if fc is None:
                return None
            p = self._apply(test_inp, bg, fc)
            if p is None:
                return None
            predictions.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GravityDownOneRowGapSolver  (825aa9e9)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class GravityDownOneRowGapSolver(BaseSolver):
    """
    BG = most common color in all rows except the bottom row.
    struct_color = most common non-BG color in the bottom row (or None).
    floor_c[col] = first row (top-down) where struct appears; or rows if none.
    Each non-BG, non-struct component's target_bottom = min(floor_c for its cols) - 2.
    Components moved DOWN (delta > 0 only). Bottom-to-top processing with stacking.
    """

    @property
    def name(self) -> str:
        return "GravityDownOneRowGapSolver"

    def _detect_bg_struct(self, inp: Grid):
        rows, cols = len(inp), len(inp[0])
        non_bottom = [inp[r][c] for r in range(rows - 1) for c in range(cols)]
        if not non_bottom:
            return None, None
        bg = Counter(non_bottom).most_common(1)[0][0]
        bottom_non_bg = [v for v in inp[rows - 1] if v != bg]
        struct_color = Counter(bottom_non_bg).most_common(1)[0][0] if bottom_non_bg else None
        return bg, struct_color

    def _floor_cols(self, inp: Grid, struct_color) -> List[int]:
        rows, cols = len(inp), len(inp[0])
        result = []
        for c in range(cols):
            fc = rows
            if struct_color is not None:
                for r in range(rows):
                    if inp[r][c] == struct_color:
                        fc = r
                        break
            result.append(fc)
        return result

    def _apply(self, inp: Grid, bg: int, struct_color) -> Grid:
        rows, cols = len(inp), len(inp[0])
        floor_c = self._floor_cols(inp, struct_color)
        components = _connected_components(inp, bg, exclude=struct_color)

        pred = [list(row) for row in inp]
        for comp in components:
            for r, c in comp:
                pred[r][c] = bg

        # Sort: largest bottom first (process bottom components first)
        sorted_comps = sorted(components, key=lambda comp: -max(r for r, c in comp))
        placed = [[False] * cols for _ in range(rows)]

        for comp in sorted_comps:
            color = inp[comp[0][0]][comp[0][1]]
            current_bottom = max(r for r, c in comp)
            comp_cols = {c for r, c in comp}
            tf = min(floor_c[c] for c in comp_cols)
            target_bottom = tf - 2
            delta = target_bottom - current_bottom

            if delta <= 0:
                for r, c in comp:
                    pred[r][c] = color
                    placed[r][c] = True
                continue

            # Try from target downward, back up on collision
            for extra_shift in range(delta + 1):
                new_cells = [(r + delta - extra_shift, c) for r, c in comp]
                if all(0 <= nr < rows and not placed[nr][nc] for nr, nc in new_cells):
                    break
            else:
                new_cells = [(r, c) for r, c in comp]

            for nr, nc in new_cells:
                if 0 <= nr < rows:
                    pred[nr][nc] = color
                    placed[nr][nc] = True

        return pred

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            bg, struct = self._detect_bg_struct(pair["input"])
            pred = self._apply(pair["input"], bg, struct)
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            bg, struct = self._detect_bg_struct(test_inp)
            predictions.append(Prediction(attempt_1=self._apply(test_inp, bg, struct), attempt_2=self._apply(test_inp, bg, struct), solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ConcentricRingsColorCycleSolver  (bda2d7a6)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class ConcentricRingsColorCycleSolver(BaseSolver):
    """
    Detect concentric rings (layer 0 = outermost border, layer 1 = next, ...).
    Collect unique colors in layer order (outermost to innermost).
    Rotate that list RIGHT by 1: [c0,c1,...,cn-1] → [cn-1, c0, c1, ..., cn-2].
    Apply as a color-mapping to every cell.
    """

    @property
    def name(self) -> str:
        return "ConcentricRingsColorCycleSolver"

    def _get_ring_colors(self, inp: Grid) -> Optional[List[int]]:
        rows, cols = len(inp), len(inp[0])
        colors = []
        seen = set()
        for layer in range(min(rows, cols) // 2 + 1):
            layer_cells = []
            for r in range(layer, rows - layer):
                for c in range(layer, cols - layer):
                    if (r == layer or r == rows - 1 - layer or
                            c == layer or c == cols - 1 - layer):
                        layer_cells.append((r, c))
            if not layer_cells:
                break
            layer_colors = {inp[r][c] for r, c in layer_cells}
            if len(layer_colors) != 1:
                return None
            color = next(iter(layer_colors))
            if color in seen:
                break
            seen.add(color)
            colors.append(color)
        return colors if len(colors) >= 2 else None

    def _apply(self, inp: Grid, color_map: dict) -> Grid:
        return [[color_map.get(v, v) for v in row] for row in inp]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            ring_colors = self._get_ring_colors(pair["input"])
            if ring_colors is None:
                return None
            rotated = [ring_colors[-1]] + ring_colors[:-1]
            color_map = {ring_colors[i]: rotated[i] for i in range(len(ring_colors))}
            pred = self._apply(pair["input"], color_map)
            if not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            ring_colors = self._get_ring_colors(test_inp)
            if ring_colors is None:
                return None
            rotated = [ring_colors[-1]] + ring_colors[:-1]
            color_map = {ring_colors[i]: rotated[i] for i in range(len(ring_colors))}
            predictions.append(Prediction(attempt_1=self._apply(test_inp, color_map), attempt_2=self._apply(test_inp, color_map), solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)
