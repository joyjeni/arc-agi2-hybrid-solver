"""
Advanced solvers batch 25.

Solvers:
  1. ColParityRecolorSolver       — task d406998b
     3-row grid; each column has exactly one non-zero value v1.
     Rule: cell at (r,c) gets recolored to v2 if (W-c) is ODD, stays v1 if EVEN.
     v2 is learned from training outputs (new color not in input).

  2. CellularAutomatonPairSolver  — task b5bb5719
     Grid has background bg; row 0 has exactly 2 non-bg seed values {v1,v2}.
     All rows 1..H-1 are all-bg in input.
     For each row r>=1, col c: find minimum d>=1 so prev[c-d]!=bg AND prev[c+d]!=bg.
       same left/right → output the OTHER seed value.
       different left/right → output the right value (prev[c+d]).
       no valid pair → output bg.

  3. ArrowDirectedMirrorSolver    — task 760b3cac
     Background 0; two non-zero shapes: V_arrow and V_reflect.
     V_arrow has a horizontal bar (3+ cells in one row) plus one extra cell
     offset from bar's center column.
     extra_col > center → direction RIGHT; extra_col < center → LEFT.
     V_reflect is mirrored:
       RIGHT: new_col = 2*(rightmost+1)-1 - original_col
       LEFT:  new_col = 2*leftmost-1 - original_col
     Output = input + mirrored V_reflect cells (clipped to bounds).

  4. BoundingBoxFillSolver        — task 60b61512
     Input has exactly 2 distinct values: background (0) and cluster_val.
     Output fills all 0s inside each cluster's bounding box with fill_val
     (a new value learned from training outputs).
"""

from __future__ import annotations

from collections import Counter, deque

from src.core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from src.solvers.registry import register_solver

Grid = list[list[int]]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verify_exact(pred: Grid, expected: Grid) -> bool:
    return pred == expected


def _grid_vals(grid: Grid) -> set:
    return {v for row in grid for v in row}


# ─────────────────────────────────────────────────────────────────────────────
# 1. ColParityRecolorSolver  (d406998b)
# ─────────────────────────────────────────────────────────────────────────────

def _col_parity_solve(grid: Grid, v1: int, v2: int) -> Grid | None:
    """Apply parity recolor: (W-c) odd → v2, even → v1 for each v1 cell."""
    H = len(grid)
    W = len(grid[0]) if grid else 0
    if H != 3 or W == 0:
        return None

    result = [[0] * W for _ in range(H)]
    for c in range(W):
        col_vals = [grid[r][c] for r in range(H)]
        nonzero = [v for v in col_vals if v != 0]
        if len(nonzero) != 1 or nonzero[0] != v1:
            return None
        for r in range(H):
            if grid[r][c] == v1:
                result[r][c] = v2 if (W - c) % 2 == 1 else v1
    return result


@register_solver
class ColParityRecolorSolver(BaseSolver):
    """d406998b: recolor v1→v2 based on column-width parity."""

    @property
    def name(self) -> str:
        return "ColParityRecolorSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None

        # --- learn v1 and v2 from training pairs ---
        v1: int | None = None
        v2: int | None = None

        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            out = pair["output"] if isinstance(pair, dict) else pair[1]
            H = len(inp)
            if H != 3:
                return None
            in_nonzero = {v for row in inp for v in row if v != 0}
            if len(in_nonzero) != 1:
                return None
            cur_v1 = next(iter(in_nonzero))
            out_vals = {v for row in out for v in row if v != 0}
            new_vals = out_vals - in_nonzero
            if len(new_vals) != 1:
                return None
            cur_v2 = next(iter(new_vals))
            if v1 is None:
                v1, v2 = cur_v1, cur_v2
            elif v1 != cur_v1 or v2 != cur_v2:
                return None

        if v1 is None or v2 is None:
            return None

        # --- validate all training pairs ---
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _col_parity_solve(inp, v1, v2)
            if pred is None or not _verify_exact(pred, exp):
                return None

        # --- apply to test inputs ---
        predictions = []
        for test_input in ctx.test_inputs:
            result = _col_parity_solve(test_input, v1, v2)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CellularAutomatonPairSolver  (b5bb5719)
# ─────────────────────────────────────────────────────────────────────────────

def _ca_pair_solve(grid: Grid, v1: int, v2: int, bg: int) -> Grid | None:
    """Cascade from row 0 using symmetric pair lookup."""
    H = len(grid)
    W = len(grid[0]) if grid else 0
    if H < 2 or W < 3:
        return None

    # Rows 1..H-1 must all be bg in input
    for r in range(1, H):
        if any(grid[r][c] != bg for c in range(W)):
            return None

    result = [row[:] for row in grid]

    for r in range(1, H):
        prev = result[r - 1]
        for c in range(W):
            placed = False
            for d in range(1, W):
                lc = c - d
                rc = c + d
                if lc < 0 or rc >= W:
                    break
                left = prev[lc]
                right = prev[rc]
                if left != bg and right != bg:
                    if left == right:
                        result[r][c] = v2 if left == v1 else v1
                    else:
                        result[r][c] = right
                    placed = True
                    break
            if not placed:
                result[r][c] = bg

    return result


@register_solver
class CellularAutomatonPairSolver(BaseSolver):
    """b5bb5719: cascade from row-0 seeds using symmetric pair rule."""

    @property
    def name(self) -> str:
        return "CellularAutomatonPairSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None

        # --- learn bg from all training inputs ---
        all_vals = [v for pair in ctx.train_pairs
                    for row in (pair["input"] if isinstance(pair, dict) else pair[0])
                    for v in row]
        if not all_vals:
            return None
        bg = Counter(all_vals).most_common(1)[0][0]

        # Collect seed values from row 0 of ALL training pairs (some may have
        # only one of the two seed values in row 0 — pool them to find both)
        seed_set: set = set()
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            for v in inp[0]:
                if v != bg:
                    seed_set.add(v)
        if len(seed_set) != 2:
            return None
        v1, v2 = sorted(seed_set)

        # Verify bg consistent; each pair's row-0 seeds must be a subset of {v1,v2}
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            cur_bg = Counter(v for row in inp for v in row).most_common(1)[0][0]
            if cur_bg != bg:
                return None
            cur_seeds = {v for v in inp[0] if v != bg}
            if not cur_seeds.issubset({v1, v2}):
                return None

        # --- validate all training pairs ---
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _ca_pair_solve(inp, v1, v2, bg)
            if pred is None or not _verify_exact(pred, exp):
                return None

        # --- apply to test inputs ---
        predictions = []
        for test_input in ctx.test_inputs:
            result = _ca_pair_solve(test_input, v1, v2, bg)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ArrowDirectedMirrorSolver  (760b3cac)
# ─────────────────────────────────────────────────────────────────────────────

def _arrow_mirror_solve(grid: Grid) -> Grid | None:
    """Detect arrow direction and horizontally mirror the other shape."""
    H = len(grid)
    W = len(grid[0]) if grid else 0

    # Collect cells per non-zero value
    shapes: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != 0:
                shapes.setdefault(v, []).append((r, c))

    if len(shapes) != 2:
        return None

    vals = list(shapes.keys())

    def is_arrow(v: int) -> bool:
        """True iff shape v has a 3+-cell bar row AND exactly 1 non-bar cell
        that is NOT at the bar's center column (the single arrow tip)."""
        cells = shapes[v]
        row_counts = Counter(r for r, c in cells)
        if max(row_counts.values()) < 3:
            return False
        b_row = max(row_counts, key=row_counts.get)
        b_cols = sorted(c for r, c in cells if r == b_row)
        if len(b_cols) < 3:
            return False
        ctr = sum(b_cols) / len(b_cols)
        extra = [(r, c) for r, c in cells if r != b_row and abs(c - ctr) > 0.01]
        return len(extra) == 1

    # Exactly one shape must satisfy is_arrow
    arrow_candidates = [v for v in vals if is_arrow(v)]
    if len(arrow_candidates) != 1:
        return None

    v_arrow = arrow_candidates[0]
    v_reflect = next(v for v in vals if v != v_arrow)

    # Horizontal bar row and center col for v_arrow
    arrow_cells = shapes[v_arrow]
    row_counts = Counter(r for r, c in arrow_cells)
    bar_row = max(row_counts, key=row_counts.get)
    bar_cols = sorted(c for r, c in arrow_cells if r == bar_row)
    if len(bar_cols) < 3:
        return None
    center_col = sum(bar_cols) / len(bar_cols)

    # Extra arm: non-bar cell NOT at center column
    extra_arm: tuple[int, int] | None = None
    for r, c in arrow_cells:
        if r != bar_row and abs(c - center_col) > 0.01:
            extra_arm = (r, c)
            break

    if extra_arm is None:
        return None

    direction = "right" if extra_arm[1] > center_col else "left"

    # Mirror reflect shape
    reflect_cells = shapes[v_reflect]
    if direction == "right":
        rightmost = max(c for r, c in reflect_cells)
        def mirror_col(c: int) -> int:
            return 2 * (rightmost + 1) - 1 - c
    else:
        leftmost = min(c for r, c in reflect_cells)
        def mirror_col(c: int) -> int:
            return 2 * leftmost - 1 - c

    result = [row[:] for row in grid]
    for r, c in reflect_cells:
        nc = mirror_col(c)
        if 0 <= nc < W:
            result[r][nc] = v_reflect

    return result


@register_solver
class ArrowDirectedMirrorSolver(BaseSolver):
    """760b3cac: arrow extra-arm direction determines horizontal mirror axis."""

    @property
    def name(self) -> str:
        return "ArrowDirectedMirrorSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # --- validate all training pairs ---
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _arrow_mirror_solve(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        # --- apply to test inputs ---
        predictions = []
        for test_input in ctx.test_inputs:
            result = _arrow_mirror_solve(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. BoundingBoxFillSolver  (60b61512)
# ─────────────────────────────────────────────────────────────────────────────

def _connected_components(grid: Grid, val: int,
                           connectivity: int = 4) -> list[list[tuple[int, int]]]:
    """Return list of connected components of cells equal to val.

    connectivity=4 uses cardinal neighbours; connectivity=8 adds diagonals.
    """
    H = len(grid)
    W = len(grid[0]) if grid else 0
    target = {(r, c) for r in range(H) for c in range(W) if grid[r][c] == val}
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dirs8 = [(-1, -1), (-1, 0), (-1, 1),
             (0,  -1),          (0,  1),
             (1,  -1), (1,  0), (1,  1)]
    dirs = dirs8 if connectivity == 8 else dirs4
    visited: set = set()
    components = []
    for start in target:
        if start in visited:
            continue
        comp: list[tuple[int, int]] = []
        queue = deque([start])
        visited.add(start)
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in dirs:
                nb = (r + dr, c + dc)
                if nb in target and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def _bbox_fill_solve(grid: Grid, cluster_val: int, fill_val: int) -> Grid | None:
    """Fill 0s inside each cluster's bounding box with fill_val."""
    len(grid)
    len(grid[0]) if grid else 0
    components = _connected_components(grid, cluster_val, connectivity=8)
    if not components:
        return None

    result = [row[:] for row in grid]
    for comp in components:
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r][c] == 0:
                    result[r][c] = fill_val
    return result


@register_solver
class BoundingBoxFillSolver(BaseSolver):
    """60b61512: fill 0s inside each cluster's bounding box with learned fill_val."""

    @property
    def name(self) -> str:
        return "BoundingBoxFillSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None

        # --- learn cluster_val and fill_val from training pairs ---
        first_inp = (ctx.train_pairs[0]["input"]
                     if isinstance(ctx.train_pairs[0], dict)
                     else ctx.train_pairs[0][0])
        first_out = (ctx.train_pairs[0]["output"]
                     if isinstance(ctx.train_pairs[0], dict)
                     else ctx.train_pairs[0][1])

        in_vals = _grid_vals(first_inp)
        out_vals = _grid_vals(first_out)

        # Input must have exactly 2 distinct values (bg=0 and cluster_val)
        if 0 not in in_vals or len(in_vals) != 2:
            return None
        cluster_val = next(v for v in in_vals if v != 0)

        # Output must add exactly 1 new value (fill_val)
        new_vals = out_vals - in_vals
        if len(new_vals) != 1:
            return None
        fill_val = next(iter(new_vals))

        # Verify consistency across all training pairs
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            out = pair["output"] if isinstance(pair, dict) else pair[1]
            if _grid_vals(inp) != in_vals:
                return None
            if _grid_vals(out) - in_vals != {fill_val}:
                return None

        # --- validate all training pairs ---
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _bbox_fill_solve(inp, cluster_val, fill_val)
            if pred is None or not _verify_exact(pred, exp):
                return None

        # --- apply to test inputs ---
        predictions = []
        for test_input in ctx.test_inputs:
            result = _bbox_fill_solve(test_input, cluster_val, fill_val)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )
