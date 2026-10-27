"""
advanced_solvers30.py — Four solvers for ARC-AGI-2

1. FrameSeedFillSolver       (task 444801d8) — fill frame interior + ceiling row from seed
2. StrayBlobFillFrameSolver  (task 465b7d93) — stray blob shape fills frame interior
3. ReflectAcrossColorTwoSolver (task 2bcee788) — reflect main-color across color-2 axis; bg→3
4. UniqueColorRingTwoSolver  (task 31aa019c) — ring of 2s around unique-once color cell
"""

from typing import List, Optional, Tuple
from collections import deque

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from src.solvers.registry import register_solver


def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    for r in range(len(pred)):
        if len(pred[r]) != len(expected[r]):
            return False
        for c in range(len(pred[r])):
            if pred[r][c] != expected[r][c]:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. FrameSeedFillSolver  (444801d8)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rule (verified on all 3 train + test):
#   - Find a rectangular frame of 1s with one gap (0 cell) in its top border.
#   - Seed cell: unique non-0, non-1 interior cell.  Seed color = seed value.
#   - Fill all interior cells with seed color.
#   - Fill the gap cell in the top border with seed color.
#   - Add a "ceiling" row above the top border (top_row-1), cols left..right,
#     all filled with seed color.
#   Frame detection: scan (bottom, left, right) combos — bottom row all 1s,
#   left/right cols all 1s, top row has 1s at left & right with exactly one
#   0 inside.

def _find_all_frames(grid: Grid) -> List[Tuple[int, int, int, int, int]]:
    """Return list of (top, bottom, left, right, gap_col) for all frames in grid."""
    H = len(grid)
    W = len(grid[0])
    frames = []
    seen: set = set()

    for bottom in range(H - 1, 0, -1):
        for left in range(W - 1):
            # Quick check: grid[bottom][left] must be 1
            if grid[bottom][left] != 1:
                continue
            for right in range(left + 2, W):
                # bottom row must be all 1s from left to right
                if not all(grid[bottom][c] == 1 for c in range(left, right + 1)):
                    continue
                # Scan upward for top row
                for top in range(bottom - 1, -1, -1):
                    # left col all 1s from top to bottom
                    if not all(grid[r][left] == 1 for r in range(top, bottom + 1)):
                        break
                    # right col all 1s
                    if not all(grid[r][right] == 1 for r in range(top, bottom + 1)):
                        break
                    # top row interior: exactly one 0 gap, rest 1s
                    zeros = [c for c in range(left + 1, right) if grid[top][c] == 0]
                    ones_count = sum(1 for c in range(left + 1, right) if grid[top][c] == 1)
                    if len(zeros) == 1 and ones_count == (right - left - 2):
                        gap_col = zeros[0]
                        key = (top, bottom, left, right)
                        if key not in seen:
                            seen.add(key)
                            frames.append((top, bottom, left, right, gap_col))
                        break  # found the top for this (bottom, left, right)
    return frames


def _apply_frame_fill(result: Grid, grid: Grid, frame: Tuple) -> bool:
    """Apply fill for a single frame. Returns False if no seed found."""
    top, bottom, left, right, gap_col = frame

    # Find seed: exactly one non-0 non-1 cell strictly inside the frame
    seed_color = None
    seed_count = 0
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            v = grid[r][c]
            if v != 0 and v != 1:
                seed_color = v
                seed_count += 1
    if seed_count != 1 or seed_color is None:
        return False

    H = len(grid)

    # Fill interior
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            result[r][c] = seed_color

    # Fill gap in top border
    result[top][gap_col] = seed_color

    # Add ceiling row (top_row - 1) if in bounds
    ceiling_row = top - 1
    if ceiling_row >= 0:
        for c in range(left, right + 1):
            result[ceiling_row][c] = seed_color

    return True


def _solve_frame_seed_fill(grid: Grid) -> Optional[Grid]:
    frames = _find_all_frames(grid)
    if not frames:
        return None

    result = _copy_grid(grid)
    for frame in frames:
        if not _apply_frame_fill(result, grid, frame):
            return None

    return result


@register_solver
class FrameSeedFillSolver(BaseSolver):
    """Fill frame interior + gap + ceiling row using seed color (444801d8)."""

    TARGET_TASK = "444801d8"

    @property
    def name(self) -> str:
        return "FrameSeedFillSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_frame_seed_fill(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_frame_seed_fill(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. StrayBlobFillFrameSolver  (465b7d93)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rule (verified on all 3 train + test):
#   Background=7, frame border=6 (closed rectangle of 6-cells).
#   Stray blob: connected non-7, non-6 cells (single color).
#   Fill frame interior based on blob's normalized bounding-box shape:
#     - Solid rectangle: flood-fill entire interior with blob color
#     - 2×2 L-shape (3 of 4 corners, 1 missing):
#         missing top-right    → fill left col + bottom row of interior
#         missing top-left     → fill right col + bottom row of interior
#         missing bottom-right → fill left col + top row of interior
#         missing bottom-left  → fill right col + top row of interior
#     - 1×1 single cell: fill single interior cell (center)
#   Remove blob from original position (replace with 7).

def _flood_fill_connected(grid: Grid, start_r: int, start_c: int,
                           match_colors: set) -> List[Tuple[int, int]]:
    """BFS flood fill from start, collecting cells whose color is in match_colors."""
    H, W = len(grid), len(grid[0])
    visited = set()
    q = deque()
    if grid[start_r][start_c] in match_colors:
        q.append((start_r, start_c))
        visited.add((start_r, start_c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if grid[nr][nc] in match_colors:
                    visited.add((nr, nc))
                    q.append((nr, nc))
    return list(visited)


def _solve_stray_blob_fill_frame(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])
    BG = 7
    BORDER = 6

    # Find frame: closed rectangle of 6s
    # Detect by finding top/bottom rows and left/right cols of 6s
    six_rows = [r for r in range(H) if all(grid[r][c] == BORDER for c in range(W))]
    six_cols = [c for c in range(W) if all(grid[r][c] == BORDER for r in range(H))]

    # Frame: top row, bottom row, left col, right col (partial — just border cells)
    # Find bounding rectangle of all 6-cells
    six_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == BORDER]
    if not six_cells:
        return None

    min_r = min(r for r, c in six_cells)
    max_r = max(r for r, c in six_cells)
    min_c = min(c for r, c in six_cells)
    max_c = max(c for r, c in six_cells)

    if min_r == max_r or min_c == max_c:
        return None

    # Verify frame: top row, bottom row, left col, right col all 6s
    for c in range(min_c, max_c + 1):
        if grid[min_r][c] != BORDER or grid[max_r][c] != BORDER:
            return None
    for r in range(min_r, max_r + 1):
        if grid[r][min_c] != BORDER or grid[r][max_c] != BORDER:
            return None

    # Interior bounds
    ir_top = min_r + 1
    ir_bot = max_r - 1
    ir_left = min_c + 1
    ir_right = max_c - 1

    if ir_top > ir_bot or ir_left > ir_right:
        return None

    # Find stray blob: all non-7, non-6 cells
    blob_cells = [(r, c) for r in range(H) for c in range(W)
                  if grid[r][c] != BG and grid[r][c] != BORDER]
    if not blob_cells:
        return None

    # All blob cells should be same color
    blob_colors = set(grid[r][c] for r, c in blob_cells)
    if len(blob_colors) != 1:
        return None
    blob_color = blob_colors.pop()

    # Compute normalized bounding box of blob
    br_min = min(r for r, c in blob_cells)
    br_max = max(r for r, c in blob_cells)
    bc_min = min(c for r, c in blob_cells)
    bc_max = max(c for r, c in blob_cells)

    blob_h = br_max - br_min + 1
    blob_w = bc_max - bc_min + 1

    # Normalize blob cells to (0,0)-origin
    blob_norm = set((r - br_min, c - bc_min) for r, c in blob_cells)
    total_cells = blob_h * blob_w
    blob_count = len(blob_cells)

    result = _copy_grid(grid)

    # Remove blob
    for r, c in blob_cells:
        result[r][c] = BG

    # Determine fill pattern
    int_h = ir_bot - ir_top + 1
    int_w = ir_right - ir_left + 1

    if blob_count == 1:
        # Single cell: fill one interior cell (center or just the first)
        # Place at center of interior
        cr = ir_top + int_h // 2
        cc = ir_left + int_w // 2
        result[cr][cc] = blob_color

    elif blob_count == total_cells:
        # Solid rectangle: fill entire interior
        for r in range(ir_top, ir_bot + 1):
            for c in range(ir_left, ir_right + 1):
                result[r][c] = blob_color

    elif blob_h == 2 and blob_w == 2 and blob_count == 3:
        # 2×2 L-shape: identify the missing corner
        corners = {(0, 0), (0, 1), (1, 0), (1, 1)}
        missing = corners - blob_norm
        if len(missing) != 1:
            return None
        (mr, mc) = missing.pop()

        if mr == 0 and mc == 1:
            # missing top-right → fill left col + bottom row
            for r in range(ir_top, ir_bot + 1):
                result[r][ir_left] = blob_color
            for c in range(ir_left, ir_right + 1):
                result[ir_bot][c] = blob_color
        elif mr == 0 and mc == 0:
            # missing top-left → fill right col + bottom row
            for r in range(ir_top, ir_bot + 1):
                result[r][ir_right] = blob_color
            for c in range(ir_left, ir_right + 1):
                result[ir_bot][c] = blob_color
        elif mr == 1 and mc == 1:
            # missing bottom-right → fill left col + top row
            for r in range(ir_top, ir_bot + 1):
                result[r][ir_left] = blob_color
            for c in range(ir_left, ir_right + 1):
                result[ir_top][c] = blob_color
        elif mr == 1 and mc == 0:
            # missing bottom-left → fill right col + top row
            for r in range(ir_top, ir_bot + 1):
                result[r][ir_right] = blob_color
            for c in range(ir_left, ir_right + 1):
                result[ir_top][c] = blob_color
        else:
            return None
    else:
        return None

    return result


@register_solver
class StrayBlobFillFrameSolver(BaseSolver):
    """Stray blob shape fills frame interior (465b7d93)."""

    TARGET_TASK = "465b7d93"

    @property
    def name(self) -> str:
        return "StrayBlobFillFrameSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_stray_blob_fill_frame(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_stray_blob_fill_frame(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ReflectAcrossColorTwoSolver  (2bcee788)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rule (verified on all 4 train):
#   1. Background 0 → 3 everywhere in output.
#   2. Find color-2 cells (axis marker) and main-color cells (only other non-0).
#   3. Determine axis direction by centroid comparison:
#      - |Δcol| > |Δrow|  →  horizontal (left/right) axis
#      - |Δrow| >= |Δcol| →  vertical (above/below) axis
#   4. Reflect each main-color cell across the axis:
#      - 2 LEFT of main:    c' = 2*max_2_col + 1 - c
#      - 2 RIGHT of main:   c' = 2*min_2_col - 1 - c
#      - 2 ABOVE main:      r' = 2*max_2_row + 1 - r
#      - 2 BELOW main:      r' = 2*min_2_row - 1 - r
#   5. Output = all 3s, place original main-color cells + reflected copies.
#      Color-2 cells are NOT placed in output.

def _solve_reflect_across_color_two(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])

    two_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == 2]
    if not two_cells:
        return None

    # Find main-color cells: non-0, non-2
    other_colors = set(grid[r][c] for r in range(H) for c in range(W)
                       if grid[r][c] != 0 and grid[r][c] != 2)
    if len(other_colors) != 1:
        return None
    main_color = other_colors.pop()

    main_cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == main_color]
    if not main_cells:
        return None

    # Centroids
    two_cr = sum(r for r, c in two_cells) / len(two_cells)
    two_cc = sum(c for r, c in two_cells) / len(two_cells)
    main_cr = sum(r for r, c in main_cells) / len(main_cells)
    main_cc = sum(c for r, c in main_cells) / len(main_cells)

    delta_r = two_cr - main_cr
    delta_c = two_cc - main_cc

    # Compute reflected cells
    reflected = []
    if abs(delta_c) > abs(delta_r):
        # Horizontal axis (left/right)
        if delta_c < 0:
            # 2 is LEFT of main: axis at max_2_col + 0.5
            max_2_col = max(c for r, c in two_cells)
            for r, c in main_cells:
                nc = 2 * max_2_col + 1 - c
                reflected.append((r, nc))
        else:
            # 2 is RIGHT of main: axis at min_2_col - 0.5
            min_2_col = min(c for r, c in two_cells)
            for r, c in main_cells:
                nc = 2 * min_2_col - 1 - c
                reflected.append((r, nc))
    else:
        # Vertical axis (above/below)
        if delta_r < 0:
            # 2 is ABOVE main: axis at max_2_row + 0.5
            max_2_row = max(r for r, c in two_cells)
            for r, c in main_cells:
                nr = 2 * max_2_row + 1 - r
                reflected.append((nr, c))
        else:
            # 2 is BELOW main: axis at min_2_row - 0.5
            min_2_row = min(r for r, c in two_cells)
            for r, c in main_cells:
                nr = 2 * min_2_row - 1 - r
                reflected.append((nr, c))

    # Build output: all 3s
    result = [[3] * W for _ in range(H)]

    # Place original main-color cells
    for r, c in main_cells:
        if 0 <= r < H and 0 <= c < W:
            result[r][c] = main_color

    # Place reflected cells
    for r, c in reflected:
        if 0 <= r < H and 0 <= c < W:
            result[r][c] = main_color

    return result


@register_solver
class ReflectAcrossColorTwoSolver(BaseSolver):
    """Reflect main-color across color-2 axis; background 0→3 (2bcee788)."""

    TARGET_TASK = "2bcee788"

    @property
    def name(self) -> str:
        return "ReflectAcrossColorTwoSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_reflect_across_color_two(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_reflect_across_color_two(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. UniqueColorRingTwoSolver  (31aa019c)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rule (verified on all 3 train):
#   1. Find the color that appears exactly once (excluding 0 and 2).
#   2. That cell's position is (r, c).
#   3. Output: all cells = 0.
#   4. Place a 3×3 ring of 2s centered at (r, c):
#      all cells in rows r-1..r+1, cols c-1..c+1 → 2, EXCEPT center (r,c)
#      which keeps its original color.
#   Boundary check: if ring goes out of bounds → return None.

def _solve_unique_color_ring_two(grid: Grid) -> Optional[Grid]:
    H = len(grid)
    W = len(grid[0])

    from collections import Counter
    # Count all non-0, non-2 colors
    freq: Counter = Counter()
    pos = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != 0 and v != 2:
                freq[v] += 1
                pos[v] = (r, c)

    # Find the unique-once color
    unique_colors = [v for v, cnt in freq.items() if cnt == 1]
    if len(unique_colors) != 1:
        return None
    target_color = unique_colors[0]
    r, c = pos[target_color]

    # Check ring fits in bounds
    if r - 1 < 0 or r + 1 >= H or c - 1 < 0 or c + 1 >= W:
        return None

    # Build output: all 0s
    result = [[0] * W for _ in range(H)]

    # Place 3×3 ring of 2s
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            result[r + dr][c + dc] = 2

    # Center keeps original color
    result[r][c] = target_color

    return result


@register_solver
class UniqueColorRingTwoSolver(BaseSolver):
    """3×3 ring of 2s around the unique-once color cell (31aa019c)."""

    TARGET_TASK = "31aa019c"

    @property
    def name(self) -> str:
        return "UniqueColorRingTwoSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = _solve_unique_color_ring_two(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_unique_color_ring_two(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
