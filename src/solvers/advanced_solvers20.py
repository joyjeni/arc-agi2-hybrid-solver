"""
advanced_solvers20.py — 5 pattern solvers
  1. DiagonalPairConnectSolver   (task 1f876c06)
  2. TriangleMidpointSolver      (task 11e1fe23)
  3. ClosestDividerRecolorSolver (task 2204b7a8)
  4. RectangleGapFillSolver      (task 292dd178)
  5. PatternRowProjectionSolver  (task 2685904e)
"""

from typing import List, Tuple, Dict, Optional, Set
from collections import Counter, defaultdict

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from .registry import register_solver

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _bg_color(grid: Grid) -> int:
    cnt = Counter(cell for row in grid for cell in row)
    return cnt.most_common(1)[0][0]


def _verify_exact(out: Grid, expected: Grid) -> bool:
    if len(out) != len(expected):
        return False
    return all(
        out[r][c] == expected[r][c]
        for r in range(len(out))
        for c in range(len(out[0]))
    )


# ===========================================================================
# 1. DiagonalPairConnectSolver — task 1f876c06
# ===========================================================================
#
# Rule:
#   Background = most common color.
#   For each non-bg color appearing exactly twice: if |dr|==|dc| (45° diagonal),
#     fill all cells along the diagonal with that color (inclusive of both endpoints).
#     If not diagonal, return None.
#   For each non-bg color appearing exactly once at (r,c): among 4 diagonal directions,
#     compute steps = min(steps_to_row_boundary, steps_to_col_boundary) for each direction;
#     pick direction with minimum steps; fill from cell to boundary (exclusive of start).
#   Return None if any color appears 3+ times.

def _solve_diagonal_pair_connect(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    bg = _bg_color(grid)

    # Collect non-bg cells by color
    color_cells: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                color_cells[grid[r][c]].append((r, c))

    # Reject if any non-bg color appears 3+ times
    for color, cells in color_cells.items():
        if len(cells) >= 3:
            return None

    out = _copy_grid(grid)

    for color, cells in color_cells.items():
        if len(cells) == 2:
            (r1, c1), (r2, c2) = cells
            dr = r2 - r1
            dc = c2 - c1
            if abs(dr) != abs(dc):
                return None  # non-diagonal pair — not our task
            steps = abs(dr)
            sr = 1 if dr > 0 else -1
            sc = 1 if dc > 0 else -1
            for i in range(steps + 1):
                out[r1 + i * sr][c1 + i * sc] = color

        elif len(cells) == 1:
            (r, c) = cells[0]
            # For each of 4 diagonal directions, compute steps until boundary
            best_steps = float('inf')
            best_dir = None
            for dr, dc in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]:
                steps_r = r if dr < 0 else H - 1 - r
                steps_c = c if dc < 0 else W - 1 - c
                steps = min(steps_r, steps_c)
                if steps < best_steps:
                    best_steps = steps
                    best_dir = (dr, dc)
            if best_dir is not None and best_steps > 0:
                dr, dc = best_dir
                for i in range(1, best_steps + 1):
                    out[r + i * dr][c + i * dc] = color

    return out


@register_solver
class DiagonalPairConnectSolver(BaseSolver):
    """Fill 45° diagonals between paired same-color cells (task 1f876c06)."""

    @property
    def name(self) -> str:
        return "DiagonalPairConnectSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            result = _solve_diagonal_pair_connect(pair["input"])
            if result is None or not _verify_exact(result, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            out = _solve_diagonal_pair_connect(test_input)
            if out is None:
                return None
            predictions.append(
                Prediction(attempt_1=out, attempt_2=out, solver_name=self.name)
            )

        return SolverResult(task_id=ctx.task_id, predictions=predictions)


# ===========================================================================
# 2. TriangleMidpointSolver — task 11e1fe23
# ===========================================================================
#
# Rule:
#   Background = 0. Exactly 3 non-bg cells, each a different color.
#   Among the 3 pairs, find the one where |dr|==|dc| AND |dr| is even (integer midpoint).
#   Midpoint M = ((r1+r2)//2, (c1+c2)//2) → paint with color 5.
#   For each of the 3 source cells: paint M + (sign(r-Mr), sign(c-Mc)) with that cell's color.

def _solve_triangle_midpoint(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    bg = 0

    non_bg = [(grid[r][c], r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]

    if len(non_bg) != 3:
        return None

    # All 3 cells must have distinct colors
    if len({cell[0] for cell in non_bg}) != 3:
        return None

    # Find the diagonal pair: |dr|==|dc| AND |dr| is even AND |dr|>0
    diagonal_idx = None
    for i in range(3):
        for j in range(i + 1, 3):
            _, r1, c1 = non_bg[i]
            _, r2, c2 = non_bg[j]
            dr = abs(r2 - r1)
            dc = abs(c2 - c1)
            if dr == dc and dr > 0 and dr % 2 == 0:
                diagonal_idx = (i, j)
                break
        if diagonal_idx is not None:
            break

    if diagonal_idx is None:
        return None

    i, j = diagonal_idx
    _, r1, c1 = non_bg[i]
    _, r2, c2 = non_bg[j]
    Mr = (r1 + r2) // 2
    Mc = (c1 + c2) // 2

    out = _copy_grid(grid)
    # Paint midpoint with color 5
    out[Mr][Mc] = 5

    # For each source cell, paint one step from M toward that cell
    for color, r, c in non_bg:
        sr = 1 if r > Mr else (-1 if r < Mr else 0)
        sc = 1 if c > Mc else (-1 if c < Mc else 0)
        nr, nc = Mr + sr, Mc + sc
        if 0 <= nr < H and 0 <= nc < W:
            out[nr][nc] = color

    return out


@register_solver
class TriangleMidpointSolver(BaseSolver):
    """Paint midpoint and adjacent cells for a 3-cell triangle (task 11e1fe23)."""

    @property
    def name(self) -> str:
        return "TriangleMidpointSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            result = _solve_triangle_midpoint(pair["input"])
            if result is None or not _verify_exact(result, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            out = _solve_triangle_midpoint(test_input)
            if out is None:
                return None
            predictions.append(
                Prediction(attempt_1=out, attempt_2=out, solver_name=self.name)
            )

        return SolverResult(task_id=ctx.task_id, predictions=predictions)


# ===========================================================================
# 3. ClosestDividerRecolorSolver — task 2204b7a8
# ===========================================================================
#
# Rule:
#   Background = 0.
#   Dividers = full rows (all W cells same non-bg color) or full cols (all H cells same non-bg color).
#   Non-bg, non-divider cells (neutral): for each, find nearest divider by perpendicular distance
#     (|r−divrow| for row dividers, |c−divcol| for col dividers); recolor with that divider's color.
#   Return None if no dividers found.

def _solve_closest_divider_recolor(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    bg = 0

    # Find full-row dividers
    row_dividers: Dict[int, int] = {}
    for r in range(H):
        non_bg = [grid[r][c] for c in range(W) if grid[r][c] != bg]
        if len(non_bg) == W and len(set(non_bg)) == 1:
            row_dividers[r] = non_bg[0]

    # Find full-col dividers
    col_dividers: Dict[int, int] = {}
    for c in range(W):
        non_bg = [grid[r][c] for r in range(H) if grid[r][c] != bg]
        if len(non_bg) == H and len(set(non_bg)) == 1:
            col_dividers[c] = non_bg[0]

    if not row_dividers and not col_dividers:
        return None

    out = _copy_grid(grid)

    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg:
                continue
            # Skip divider cells
            if r in row_dividers or c in col_dividers:
                continue
            # Neutral non-bg cell — find nearest divider by perpendicular distance
            best_dist = float('inf')
            best_color = None
            for dr, dcolor in row_dividers.items():
                dist = abs(r - dr)
                if dist < best_dist:
                    best_dist = dist
                    best_color = dcolor
            for dc, dcolor in col_dividers.items():
                dist = abs(c - dc)
                if dist < best_dist:
                    best_dist = dist
                    best_color = dcolor
            if best_color is not None:
                out[r][c] = best_color

    return out


@register_solver
class ClosestDividerRecolorSolver(BaseSolver):
    """Recolor neutral cells with nearest row/col divider color (task 2204b7a8)."""

    @property
    def name(self) -> str:
        return "ClosestDividerRecolorSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        try:
            for pair in ctx.train_pairs:
                result = _solve_closest_divider_recolor(pair["input"])
                if result is None or not _verify_exact(result, pair["output"]):
                    return None

            predictions = []
            for test_input in ctx.test_inputs:
                out = _solve_closest_divider_recolor(test_input)
                if out is None:
                    return None
                predictions.append(
                    Prediction(attempt_1=out, attempt_2=out, solver_name=self.name)
                )

            return SolverResult(task_id=ctx.task_id, predictions=predictions)
        except Exception:
            return None


# ===========================================================================
# 4. RectangleGapFillSolver — task 292dd178
# ===========================================================================
#
# Rule:
#   Frame color = 1. Fill color = 2. Background = everything else.
#   Find connected components of 1-cells.
#   For each component: compute bounding box (r1,c1)-(r2,c2); verify all cells are on the
#     border (no interior 1s); find exactly 1 missing border cell (gap).
#   Fill interior cells (r1+1..r2-1, c1+1..c2-1) with 2.
#   Set gap cell to 2; extend line of 2s outward from gap until grid boundary:
#     - Gap on top border (r==r1):    extend upward
#     - Gap on bottom border (r==r2): extend downward
#     - Gap on left border (c==c1):   extend leftward
#     - Gap on right border (c==c2):  extend rightward
#   If gap is already at grid edge, no extension (loop is empty).

def _bfs_components(grid: Grid, color: int) -> List[Set[Tuple[int, int]]]:
    H = len(grid)
    W = len(grid[0])
    visited: Set[Tuple[int, int]] = set()
    components = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] == color and (r, c) not in visited:
                comp: Set[Tuple[int, int]] = set()
                queue = [(r, c)]
                visited.add((r, c))
                comp.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    for nr, nc in [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]:
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            comp.add((nr, nc))
                            queue.append((nr, nc))
                components.append(comp)

    return components


def _solve_rectangle_gap_fill(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    FRAME = 1
    FILL = 2

    components = _bfs_components(grid, FRAME)
    if not components:
        return None

    out = _copy_grid(grid)

    for comp in components:
        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        r1, r2 = min(rows), max(rows)
        c1, c2 = min(cols), max(cols)

        # Rectangle must be at least 3x3 to have an interior
        if r2 - r1 < 2 or c2 - c1 < 2:
            continue

        # Verify all component cells are on the border of the bounding box
        for r, c in comp:
            if r != r1 and r != r2 and c != c1 and c != c2:
                return None  # interior frame cell — wrong task

        # Build expected full border
        expected_border: Set[Tuple[int, int]] = set()
        for cc in range(c1, c2 + 1):
            expected_border.add((r1, cc))
            expected_border.add((r2, cc))
        for rr in range(r1 + 1, r2):
            expected_border.add((rr, c1))
            expected_border.add((rr, c2))

        missing = expected_border - comp
        if len(missing) != 1:
            continue  # not exactly one gap — skip this component

        gr, gc = list(missing)[0]

        # Fill interior with FILL
        for rr in range(r1 + 1, r2):
            for cc in range(c1 + 1, c2):
                out[rr][cc] = FILL

        # Set gap cell to FILL
        out[gr][gc] = FILL

        # Extend outward from gap until grid boundary
        if gr == r1:
            # Top border — extend upward
            for rr in range(gr - 1, -1, -1):
                out[rr][gc] = FILL
        elif gr == r2:
            # Bottom border — extend downward
            for rr in range(gr + 1, H):
                out[rr][gc] = FILL
        elif gc == c1:
            # Left border — extend leftward
            for cc in range(gc - 1, -1, -1):
                out[gr][cc] = FILL
        elif gc == c2:
            # Right border — extend rightward
            for cc in range(gc + 1, W):
                out[gr][cc] = FILL

    return out


@register_solver
class RectangleGapFillSolver(BaseSolver):
    """Fill near-complete rectangle frames and extend a 2-line from the gap (task 292dd178)."""

    @property
    def name(self) -> str:
        return "RectangleGapFillSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        try:
            for pair in ctx.train_pairs:
                result = _solve_rectangle_gap_fill(pair["input"])
                if result is None or not _verify_exact(result, pair["output"]):
                    return None

            predictions = []
            for test_input in ctx.test_inputs:
                out = _solve_rectangle_gap_fill(test_input)
                if out is None:
                    return None
                predictions.append(
                    Prediction(attempt_1=out, attempt_2=out, solver_name=self.name)
                )

            return SolverResult(task_id=ctx.task_id, predictions=predictions)
        except Exception:
            return None


# ===========================================================================
# 5. PatternRowProjectionSolver — task 2685904e
# ===========================================================================
#
# Rule:
#   Background = 0. Divider = full row of 5s. Counter row = row containing N 8s (N > 0).
#   Pattern row = non-divider, non-counter row with ≥2 distinct non-{0,5,8} colors.
#   Count occurrences of each color in pattern row (excluding 0,5,8).
#   Target colors = colors whose count == N.
#   For i in 1..N: at row (divider_row − i), for each column where a target color
#     appears in the pattern row, paint that cell with the target color.

def _solve_pattern_row_projection(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    IGNORED = {0, 5, 8}

    # Find divider row (full row of 5s)
    divider_row = None
    for r in range(H):
        if all(grid[r][c] == 5 for c in range(W)):
            divider_row = r
            break
    if divider_row is None:
        return None

    # Find counter row (row containing N >= 1 cells with value 8, not the divider)
    counter_row = None
    N = 0
    for r in range(H):
        if r == divider_row:
            continue
        count_8 = sum(1 for c in range(W) if grid[r][c] == 8)
        if count_8 > 0:
            counter_row = r
            N = count_8
            break
    if counter_row is None or N == 0:
        return None

    # Find pattern row (non-divider, non-counter, ≥2 distinct non-{0,5,8} colors)
    pattern_row = None
    for r in range(H):
        if r == divider_row or r == counter_row:
            continue
        colors_in_row = {grid[r][c] for c in range(W) if grid[r][c] not in IGNORED}
        if len(colors_in_row) >= 2:
            pattern_row = r
            break
    if pattern_row is None:
        return None

    # Count each non-ignored color in pattern row
    color_count: Counter = Counter(
        grid[pattern_row][c] for c in range(W) if grid[pattern_row][c] not in IGNORED
    )

    # Target colors: those whose count equals N
    target_colors = {color for color, cnt in color_count.items() if cnt == N}
    if not target_colors:
        return None

    out = _copy_grid(grid)

    for i in range(1, N + 1):
        target_r = divider_row - i
        if 0 <= target_r < H:
            for c in range(W):
                if grid[pattern_row][c] in target_colors:
                    out[target_r][c] = grid[pattern_row][c]

    return out


@register_solver
class PatternRowProjectionSolver(BaseSolver):
    """Project pattern-row target colors N rows above divider row (task 2685904e)."""

    @property
    def name(self) -> str:
        return "PatternRowProjectionSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            result = _solve_pattern_row_projection(pair["input"])
            if result is None or not _verify_exact(result, pair["output"]):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            out = _solve_pattern_row_projection(test_input)
            if out is None:
                return None
            predictions.append(
                Prediction(attempt_1=out, attempt_2=out, solver_name=self.name)
            )

        return SolverResult(task_id=ctx.task_id, predictions=predictions)
