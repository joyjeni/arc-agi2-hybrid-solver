"""
advanced_solvers16.py — 4 pattern solvers
  1. IsolatedCellsToColorSolver   (task 67385a82)
  2. NearestPixelToFiveBlockSolver (task 6df30ad6)
  3. SizeRankRecolorSolver         (task b230c067)
  4. RaycastBlockEdgeSolver        (task 1f642eb9)
"""

from collections import deque
from math import sqrt

from src.core.interfaces import BaseSolver, Grid, Prediction, SolverResult, TaskContext

from .registry import register_solver

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]

def _make_result(grids: list[Grid], ctx: TaskContext, solver_name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=solver_name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)

def _bfs_components_color(grid: Grid, color: int) -> list[list[tuple[int, int]]]:
    """Return list of connected components (4-connected) for cells with the given color."""
    rows, cols = len(grid), len(grid[0])
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == color and (r, c) not in visited:
                comp: list[tuple[int, int]] = []
                q = deque([(r, c)])
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append(comp)
    return components

def _all_colors(grid: Grid) -> set[int]:
    return {cell for row in grid for cell in row}

def _verify_exact(out: Grid, expected: Grid) -> bool:
    if len(out) != len(expected):
        return False
    for r, row in enumerate(out):
        if row != list(expected[r]):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. IsolatedCellsToColorSolver
# ---------------------------------------------------------------------------

def _apply_isolated_cells_to_color(grid: Grid) -> Grid:
    """
    Single non-0 color in input.
    Connected components of that color:
      - size > 1  → recolor every cell to 8
      - size == 1 → keep original color
    """
    colors = _all_colors(grid) - {0}
    assert len(colors) == 1
    color = next(iter(colors))
    comps = _bfs_components_color(grid, color)
    result = _copy_grid(grid)
    for comp in comps:
        new_color = 8 if len(comp) > 1 else color
        for r, c in comp:
            result[r][c] = new_color
    return result


@register_solver
class IsolatedCellsToColorSolver(BaseSolver):
    """
    Single non-0 color; connected components ≥2 cells → 8; isolated cells → keep color.
    Confirmed on task 67385a82.
    """

    @property
    def name(self) -> str:
        return "IsolatedCellsToColorSolver"

    def can_solve(self, ctx: TaskContext) -> bool:
        try:
            train = ctx.train_pairs
            if not train:
                return False
            for pair in train:
                inp = pair["input"]
                expected = pair["output"]
                if len(inp) != len(expected) or len(inp[0]) != len(expected[0]):
                    return False
                colors = _all_colors(inp) - {0}
                if len(colors) != 1:
                    return False
                color = next(iter(colors))
                if color == 8:
                    return False
                if 8 in _all_colors(inp):
                    return False
                out = _apply_isolated_cells_to_color(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not self.can_solve(ctx):
            return None
        outputs = [_apply_isolated_cells_to_color(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)


# ---------------------------------------------------------------------------
# 2. NearestPixelToFiveBlockSolver
# ---------------------------------------------------------------------------

def _apply_nearest_pixel_to_five_block(grid: Grid) -> Grid:
    """
    One connected component of 5s (the block).
    Isolated non-0 non-5 pixels are markers.
    Block → color of nearest marker (min Euclidean distance, block cell ↔ marker cell).
    Everything else → 0.
    """
    rows, cols = len(grid), len(grid[0])

    five_cells: list[tuple[int, int]] = [
        (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5
    ]

    markers: list[tuple[int, int, int]] = []  # (r, c, color)
    marker_colors = _all_colors(grid) - {0, 5}
    for mc in marker_colors:
        comps = _bfs_components_color(grid, mc)
        for comp in comps:
            if len(comp) == 1:
                r, c = comp[0]
                markers.append((r, c, mc))

    if not markers or not five_cells:
        return _copy_grid(grid)

    result = [[0] * cols for _ in range(rows)]

    best_color = None
    best_dist = float("inf")
    for br, bc in five_cells:
        for mr, mc_col, mcolor in markers:
            d = sqrt((br - mr) ** 2 + (bc - mc_col) ** 2)
            if d < best_dist:
                best_dist = d
                best_color = mcolor

    for br, bc in five_cells:
        result[br][bc] = best_color

    return result


@register_solver
class NearestPixelToFiveBlockSolver(BaseSolver):
    """
    One connected 5-block + isolated non-0 non-5 markers.
    Block recolored to nearest marker's color; everything else → 0.
    Confirmed on task 6df30ad6.
    """

    @property
    def name(self) -> str:
        return "NearestPixelToFiveBlockSolver"

    def can_solve(self, ctx: TaskContext) -> bool:
        try:
            train = ctx.train_pairs
            if not train:
                return False
            for pair in train:
                inp = pair["input"]
                expected = pair["output"]
                if len(inp) != len(expected) or len(inp[0]) != len(expected[0]):
                    return False
                five_comps = _bfs_components_color(inp, 5)
                if len(five_comps) != 1 or len(five_comps[0]) < 2:
                    return False
                marker_colors = _all_colors(inp) - {0, 5}
                found_marker = False
                for mc in marker_colors:
                    comps = _bfs_components_color(inp, mc)
                    for comp in comps:
                        if len(comp) == 1:
                            found_marker = True
                            break
                    if found_marker:
                        break
                if not found_marker:
                    return False
                out = _apply_nearest_pixel_to_five_block(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not self.can_solve(ctx):
            return None
        outputs = [_apply_nearest_pixel_to_five_block(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)


# ---------------------------------------------------------------------------
# 3. SizeRankRecolorSolver
# ---------------------------------------------------------------------------

def _apply_size_rank_recolor(grid: Grid) -> Grid:
    """
    Input colors ⊆ {0, 8}.
    BFS components of 8; normalize each shape (translate to origin).
    Components whose normalized shape appears ≥2 times → color 1.
    Components with a unique shape → color 2.
    Background → 0.
    """
    rows, cols = len(grid), len(grid[0])
    comps = _bfs_components_color(grid, 8)

    # Normalize each component: translate so (min_r, min_c) = (0, 0)
    def normalize(cells):
        min_r = min(r for r, c in cells)
        min_c = min(c for r, c in cells)
        return frozenset((r - min_r, c - min_c) for r, c in cells)

    shapes = [normalize(comp) for comp in comps]

    # Count occurrences of each shape
    from collections import Counter
    shape_counts = Counter(shapes)

    result = [[0] * cols for _ in range(rows)]
    for comp, shape in zip(comps, shapes):
        color = 1 if shape_counts[shape] >= 2 else 2
        for r, c in comp:
            result[r][c] = color
    return result


@register_solver
class SizeRankRecolorSolver(BaseSolver):
    """
    Input only 0 and 8; multiple 8-components ranked by size desc → recolored to rank.
    Confirmed on task b230c067.
    """

    @property
    def name(self) -> str:
        return "SizeRankRecolorSolver"

    def can_solve(self, ctx: TaskContext) -> bool:
        try:
            train = ctx.train_pairs
            if not train:
                return False
            for pair in train:
                inp = pair["input"]
                expected = pair["output"]
                if len(inp) != len(expected) or len(inp[0]) != len(expected[0]):
                    return False
                if not (_all_colors(inp) <= {0, 8}):
                    return False
                comps = _bfs_components_color(inp, 8)
                if len(comps) < 2:
                    return False
                out = _apply_size_rank_recolor(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not self.can_solve(ctx):
            return None
        outputs = [_apply_size_rank_recolor(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)


# ---------------------------------------------------------------------------
# 4. RaycastBlockEdgeSolver
# ---------------------------------------------------------------------------

def _find_8_block_bbox(grid: Grid):
    """
    Return (top, bottom, left, right) of the unique filled rectangular block of 8s.
    Returns None if no such block exists.
    """
    rows, cols = len(grid), len(grid[0])
    eight_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    if not eight_cells:
        return None
    top = min(r for r, c in eight_cells)
    bottom = max(r for r, c in eight_cells)
    left = min(c for r, c in eight_cells)
    right = max(c for r, c in eight_cells)
    expected_count = (bottom - top + 1) * (right - left + 1)
    if len(eight_cells) != expected_count:
        return None
    return top, bottom, left, right


def _apply_raycast_block_edge(grid: Grid) -> Grid:
    """
    Single filled rectangular block of 8s.
    Isolated non-0 non-8 markers outside the block.
    Marker aligned with block by row XOR column:
      - row ∈ [top, bottom] and col < left  → paint (row, left)
      - row ∈ [top, bottom] and col > right → paint (row, right)
      - col ∈ [left, right] and row < top   → paint (top, col)
      - col ∈ [left, right] and row > bottom→ paint (bottom, col)
    Diagonal → skip.
    """
    bbox = _find_8_block_bbox(grid)
    if bbox is None:
        return _copy_grid(grid)
    top, bottom, left, right = bbox

    marker_colors = _all_colors(grid) - {0, 8}
    markers: list[tuple[int, int, int]] = []
    for mc in marker_colors:
        comps = _bfs_components_color(grid, mc)
        for comp in comps:
            if len(comp) == 1:
                r, c = comp[0]
                markers.append((r, c, mc))

    result = _copy_grid(grid)
    for mr, mc_col, mcolor in markers:
        row_in = top <= mr <= bottom
        col_in = left <= mc_col <= right
        if row_in and not col_in:
            if mc_col < left:
                result[mr][left] = mcolor
            else:
                result[mr][right] = mcolor
        elif col_in and not row_in:
            if mr < top:
                result[top][mc_col] = mcolor
            else:
                result[bottom][mc_col] = mcolor
        # diagonal → skip
    return result


@register_solver
class RaycastBlockEdgeSolver(BaseSolver):
    """
    Single filled rectangular 8-block; markers outside cast ray to nearest block edge cell.
    Confirmed on task 1f642eb9.
    """

    @property
    def name(self) -> str:
        return "RaycastBlockEdgeSolver"

    def can_solve(self, ctx: TaskContext) -> bool:
        try:
            train = ctx.train_pairs
            if not train:
                return False
            for pair in train:
                inp = pair["input"]
                expected = pair["output"]
                if len(inp) != len(expected) or len(inp[0]) != len(expected[0]):
                    return False
                bbox = _find_8_block_bbox(inp)
                if bbox is None:
                    return False
                marker_colors = _all_colors(inp) - {0, 8}
                if not marker_colors:
                    return False
                found_marker = False
                for mc in marker_colors:
                    comps = _bfs_components_color(inp, mc)
                    for comp in comps:
                        if len(comp) == 1:
                            found_marker = True
                            break
                    if found_marker:
                        break
                if not found_marker:
                    return False
                out = _apply_raycast_block_edge(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not self.can_solve(ctx):
            return None
        outputs = [_apply_raycast_block_edge(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)
