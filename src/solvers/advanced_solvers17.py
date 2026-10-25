"""
advanced_solvers17.py — 4 pattern solvers
  1. KeyRotateTilePaintSolver    (task 103eff5b)
  2. SixFrameBorderFillSolver    (task 543a7ed5)
  3. QuadrantPaintBlockSolver    (task 639f5a19)
  4. SixBlockWaveExpandSolver    (task 7d419a02)
"""

from collections import deque
from typing import List, Tuple, Set

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from .registry import register_solver

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]

def _make_result(grids: List[Grid], ctx: TaskContext, solver_name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=solver_name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)

def _all_colors(grid: Grid) -> Set[int]:
    return {cell for row in grid for cell in row}

def _verify_exact(out: Grid, expected: Grid) -> bool:
    if len(out) != len(expected):
        return False
    for r, row in enumerate(out):
        if row != list(expected[r]):
            return False
    return True

def _bfs_components_color(grid: Grid, color: int) -> List[List[Tuple[int, int]]]:
    """Return list of 4-connected components for cells with the given color."""
    rows, cols = len(grid), len(grid[0])
    visited: Set[Tuple[int, int]] = set()
    components: List[List[Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == color and (r, c) not in visited:
                comp: List[Tuple[int, int]] = []
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

def _rotate_90_cw(grid: Grid) -> Grid:
    """Rotate grid 90° clockwise. new[c][rows-1-r] = old[r][c]."""
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            new_grid[c][rows - 1 - r] = grid[r][c]
    return new_grid


# ---------------------------------------------------------------------------
# 1. KeyRotateTilePaintSolver  (task 103eff5b)
# ---------------------------------------------------------------------------

def _apply_key_rotate_tile_paint(grid: Grid) -> Grid:
    """
    Key = bounding box of non-zero/non-8 cells (key_h × key_w).
    8-cells are scattered tile-blocks on a virtual key_h × key_w grid.
    Bounding box of all 8s: (er0,ec0)–(er1,ec1).
    After rotating key 90° CW (dims → key_w × key_h):
      tile_h = reg_h // rot_h,  tile_w = reg_w // rot_w
    Each 8-cell at (r,c): tile position tr=(r-er0)//tile_h, tc=(c-ec0)//tile_w
    → paint that cell rotated_key[tr][tc].
    """
    rows, cols = len(grid), len(grid[0])

    # Key cells (non-0, non-8)
    key_cells = [(r, c) for r in range(rows) for c in range(cols)
                 if grid[r][c] not in (0, 8)]
    if not key_cells:
        return _copy_grid(grid)
    kr0 = min(r for r, c in key_cells)
    kr1 = max(r for r, c in key_cells)
    kc0 = min(c for r, c in key_cells)
    kc1 = max(c for r, c in key_cells)

    # Build key grid (0 where empty inside bbox)
    key_grid = [[grid[r][c] for c in range(kc0, kc1 + 1)] for r in range(kr0, kr1 + 1)]

    # 8-cells bounding box (does NOT need to be a solid rectangle)
    eight_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    if not eight_cells:
        return _copy_grid(grid)
    er0 = min(r for r, c in eight_cells)
    er1 = max(r for r, c in eight_cells)
    ec0 = min(c for r, c in eight_cells)
    ec1 = max(c for r, c in eight_cells)
    reg_h = er1 - er0 + 1
    reg_w = ec1 - ec0 + 1

    # Rotate key 90° CW → dims key_w × key_h
    rotated = _rotate_90_cw(key_grid)
    rot_h = len(rotated)    # key_w
    rot_w = len(rotated[0]) # key_h

    if reg_h % rot_h != 0 or reg_w % rot_w != 0:
        return _copy_grid(grid)

    tile_h = reg_h // rot_h
    tile_w = reg_w // rot_w

    # Paint each 8-cell with rotated_key color for its tile position
    result = _copy_grid(grid)
    for r, c in eight_cells:
        tr = (r - er0) // tile_h
        tc = (c - ec0) // tile_w
        if 0 <= tr < rot_h and 0 <= tc < rot_w:
            result[r][c] = rotated[tr][tc]
    return result


@register_solver
class KeyRotateTilePaintSolver(BaseSolver):
    """
    Non-8/non-0 key grid + all-8 rectangular region.
    Rotate key 90° CW, paint each tile of the region with the corresponding key color.
    Confirmed on task 103eff5b.
    """

    @property
    def name(self) -> str:
        return "KeyRotateTilePaintSolver"

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
                key_colors = _all_colors(inp) - {0, 8}
                if not key_colors:
                    return False
                key_cells = [(r, c) for r in range(len(inp))
                             for c in range(len(inp[0])) if inp[r][c] not in (0, 8)]
                if not key_cells:
                    return False
                eight_cells = [(r, c) for r in range(len(inp))
                               for c in range(len(inp[0])) if inp[r][c] == 8]
                if not eight_cells:
                    return False
                out = _apply_key_rotate_tile_paint(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult:
        outputs = [_apply_key_rotate_tile_paint(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)


# ---------------------------------------------------------------------------
# 2. SixFrameBorderFillSolver  (task 543a7ed5)
# ---------------------------------------------------------------------------

def _apply_six_frame_border_fill(grid: Grid) -> Grid:
    """
    Background = 8.  Find connected components of 6s.
    Per 6-component bounding box (r0,c0)–(r1,c1):
      • Border ring (one cell outside bbox, clipped to grid) that are 8 → 3.
      • Interior cells inside bbox that are 8 (not 6) → 4.
    """
    rows, cols = len(grid), len(grid[0])
    result = _copy_grid(grid)
    comps = _bfs_components_color(grid, 6)

    for comp in comps:
        r0 = min(r for r, c in comp)
        r1 = max(r for r, c in comp)
        c0 = min(c for r, c in comp)
        c1 = max(c for r, c in comp)

        # Border ring: one step outside bbox
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue  # inside bbox — handled below
                if 0 <= r < rows and 0 <= c < cols and result[r][c] == 8:
                    result[r][c] = 3

        # Interior: inside bbox, not 6
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if result[r][c] == 8:
                    result[r][c] = 4

    return result


@register_solver
class SixFrameBorderFillSolver(BaseSolver):
    """
    Background 8; connected 6-components get a 3-border and 4-interior.
    Confirmed on task 543a7ed5.
    """

    @property
    def name(self) -> str:
        return "SixFrameBorderFillSolver"

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
                colors = _all_colors(inp)
                if 6 not in colors or 8 not in colors:
                    return False
                # Background should be 8 (majority or border cells are 8)
                comps_6 = _bfs_components_color(inp, 6)
                if not comps_6:
                    return False
                out = _apply_six_frame_border_fill(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult:
        outputs = [_apply_six_frame_border_fill(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)


# ---------------------------------------------------------------------------
# 3. QuadrantPaintBlockSolver  (task 639f5a19)
# ---------------------------------------------------------------------------

def _find_solid_8_blocks(grid: Grid) -> List[Tuple[int, int, int, int]]:
    """Return (r0, r1, c0, c1) for each solid all-8 rectangular region."""
    comps = _bfs_components_color(grid, 8)
    blocks = []
    for comp in comps:
        r0 = min(r for r, c in comp)
        r1 = max(r for r, c in comp)
        c0 = min(c for r, c in comp)
        c1 = max(c for r, c in comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if len(comp) == bh * bw:
            blocks.append((r0, r1, c0, c1))
    return blocks


def _apply_quadrant_paint(grid: Grid) -> Grid:
    """
    Solid all-8 rectangular block (bh×bw).
    Fixed zones: top_h=2, bot_h=2, bdr=2 (border width), half_h=bh//2.
    Colors: TL=6, TR=1, BL=2, BR=3, interior=4.
      lr < 2:               6 if lc < bw//2 else 1
      2 ≤ lr < half_h:     6 if lc<bdr else (1 if lc≥bw-bdr else 4)
      half_h ≤ lr < bh-2:  2 if lc<bdr else (3 if lc≥bw-bdr else 4)
      lr ≥ bh-2:            2 if lc < bw//2 else 3
    """
    rows, cols = len(grid), len(grid[0])
    result = _copy_grid(grid)
    blocks = _find_solid_8_blocks(grid)

    for r0, r1, c0, c1 in blocks:
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 4 or bw < 4:
            continue
        half_h = bh // 2
        bw2 = bw // 2
        bdr = 2

        for lr in range(bh):
            for lc in range(bw):
                if lr < 2:
                    color = 6 if lc < bw2 else 1
                elif lr < half_h:
                    if lc < bdr:
                        color = 6
                    elif lc >= bw - bdr:
                        color = 1
                    else:
                        color = 4
                elif lr < bh - 2:
                    if lc < bdr:
                        color = 2
                    elif lc >= bw - bdr:
                        color = 3
                    else:
                        color = 4
                else:  # lr >= bh - 2
                    color = 2 if lc < bw2 else 3
                result[r0 + lr][c0 + lc] = color

    return result


@register_solver
class QuadrantPaintBlockSolver(BaseSolver):
    """
    Solid rectangular 8-blocks split into 4 quadrant zones (colors 6,1,2,3) with interior=4.
    Confirmed on task 639f5a19.
    """

    @property
    def name(self) -> str:
        return "QuadrantPaintBlockSolver"

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
                # Input must have 8s on a 0 background (no other colors)
                if not (_all_colors(inp) <= {0, 8}):
                    return False
                blocks = _find_solid_8_blocks(inp)
                if not blocks:
                    return False
                out = _apply_quadrant_paint(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult:
        outputs = [_apply_quadrant_paint(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)


# ---------------------------------------------------------------------------
# 4. SixBlockWaveExpandSolver  (task 7d419a02)
# ---------------------------------------------------------------------------

def _apply_six_block_wave_expand(grid: Grid) -> Grid:
    """
    Grid has a 2×2 6-block and horizontal strips of 8s (separated by all-zero rows).
    Distance d = strip index distance from the strip containing the 6-block.
      d==0: no change.
      d>0:  stay window = cols [c0 - 2*(d-1), c1 + 2*(d-1)]
            8-cells inside window → stay 8
            8-cells outside window → become 4
            0-cells inside strip rows → stay 0
    """
    rows, cols = len(grid), len(grid[0])
    result = _copy_grid(grid)

    # Find 2×2 6-block bounding box columns
    six_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]
    if not six_cells:
        return result
    sc0 = min(c for r, c in six_cells)
    sc1 = max(c for r, c in six_cells)
    sr0 = min(r for r, c in six_cells)

    # Detect strips: maximal contiguous runs of rows where at least one cell is non-zero
    strips: List[Tuple[int, int]] = []
    i = 0
    while i < rows:
        if any(grid[i][c] != 0 for c in range(cols)):
            j = i
            while j < rows and any(grid[j][c] != 0 for c in range(cols)):
                j += 1
            strips.append((i, j - 1))
            i = j
        else:
            i += 1

    # Find the strip index that contains the 6-block
    six_strip_idx = None
    for idx, (rs, re) in enumerate(strips):
        if rs <= sr0 <= re:
            six_strip_idx = idx
            break
    if six_strip_idx is None:
        return result

    c0, c1 = sc0, sc1

    for strip_idx, (rs, re) in enumerate(strips):
        d = abs(strip_idx - six_strip_idx)
        if d == 0:
            continue  # 6-strip stays unchanged

        stay_lo = c0 - 2 * (d - 1)
        stay_hi = c1 + 2 * (d - 1)

        for r in range(rs, re + 1):
            for c in range(cols):
                if result[r][c] == 8:
                    if not (stay_lo <= c <= stay_hi):
                        result[r][c] = 4
                # 0-cells and 6-cells stay as-is

    return result


@register_solver
class SixBlockWaveExpandSolver(BaseSolver):
    """
    2×2 6-block + horizontal strips of 8s.
    Each strip at distance d from the 6-strip keeps an expanding stay window; 8s outside → 4.
    Confirmed on task 7d419a02.
    """

    @property
    def name(self) -> str:
        return "SixBlockWaveExpandSolver"

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
                colors = _all_colors(inp)
                if 6 not in colors or 8 not in colors:
                    return False
                six_cells = [(r, c) for r in range(len(inp))
                             for c in range(len(inp[0])) if inp[r][c] == 6]
                if len(six_cells) != 4:
                    return False
                out = _apply_six_block_wave_expand(inp)
                if not _verify_exact(out, expected):
                    return False
            return True
        except Exception:
            return False

    def solve(self, ctx: TaskContext) -> SolverResult:
        outputs = [_apply_six_block_wave_expand(t) for t in ctx.test_inputs]
        return _make_result(outputs, ctx, self.name)
