"""
advanced_solvers15.py
Confirmed solvers from pattern analysis of 743 unsolved training tasks.

Solvers:
  - ConcentricNestSolver     : confirmed match (5587a8d0)
  - Tile3xDiagonalSolver     : confirmed match (310f3251) — fixed modular wrap
  - FillFrameInteriorSolver  : fill closed rectangles via component propagation (09c534e7)
  - StampRepeatSolver        : N single-pixel markers -> tile blob N times (4852f2fa)
"""

from collections import Counter, deque
from typing import List, Optional, Tuple

from src.core.interfaces import (
    BaseSolver, TaskContext, SolverResult, Prediction, Grid
)
from .registry import register_solver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(ctx: TaskContext, preds: List[Grid]) -> SolverResult:
    predictions = [
        Prediction(attempt_1=p, attempt_2=p, solver_name=ctx.task_id)
        for p in preds
    ]
    return SolverResult(task_id=ctx.task_id, predictions=predictions,
                        solved=True, tier=1)


def _most_common_color(grid: Grid) -> int:
    flat = [v for row in grid for v in row]
    return Counter(flat).most_common(1)[0][0]


def _bfs_components(grid: Grid, target_color=None):
    """Return list of (color, list_of_(r,c)) for same-color connected components."""
    H = len(grid)
    W = len(grid[0]) if H else 0
    visited = [[False] * W for _ in range(H)]
    components = []
    for sr in range(H):
        for sc in range(W):
            v = grid[sr][sc]
            if visited[sr][sc]:
                continue
            if target_color is not None and v != target_color:
                continue
            if target_color is None and v == 0:
                continue
            queue = deque([(sr, sc)])
            visited[sr][sc] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                        if grid[nr][nc] == v:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
            components.append((v, cells))
    return components


# ---------------------------------------------------------------------------
# 1. ConcentricNestSolver
# ---------------------------------------------------------------------------

@register_solver
class ConcentricNestSolver(BaseSolver):
    """
    Multiple objects of different sizes/colors → output is (2n-1)×(2n-1) square
    where ring index = min(r,c,S-1-r,S-1-c) maps to object color by desc size.
    Confirmed: 5587a8d0
    """

    @property
    def name(self) -> str:
        return "ConcentricNestSolver"

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
        bg = _most_common_color(grid)
        H = len(grid); W = len(grid[0]) if H else 0
        visited = [[False] * W for _ in range(H)]
        objects = []
        for sr in range(H):
            for sc in range(W):
                v = grid[sr][sc]
                if visited[sr][sc] or v == bg:
                    continue
                queue = deque([(sr, sc)])
                visited[sr][sc] = True
                cells = [(sr, sc)]
                while queue:
                    r, c = queue.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                            if grid[nr][nc] == v:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                                cells.append((nr, nc))
                objects.append((len(cells), v))
        if not objects:
            return None
        n = len(objects)
        objects.sort(key=lambda x: -x[0])
        S = 2 * n - 1
        out = []
        for r in range(S):
            row = []
            for c in range(S):
                ring = min(r, c, S - 1 - r, S - 1 - c)
                row.append(objects[ring][1])
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# 2. Tile3xDiagonalSolver  (FIXED: modular wrap in base tile)
# ---------------------------------------------------------------------------

@register_solver
class Tile3xDiagonalSolver(BaseSolver):
    """
    Output is the input tiled 3×3.  For each non-zero cell (r,c) in the INPUT,
    place color-2 at ((r+dr)%H, (c+dc)%W) in every tile — but ONLY if the
    destination cell in the INPUT is 0.

    Tries all 4 diagonal offsets; new_color is always 2 (confirmed on 310f3251).
    """

    OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    @property
    def name(self) -> str:
        return "Tile3xDiagonalSolver"

    def _tile3(self, grid: Grid) -> Grid:
        return [list(row) * 3 for row in grid for _ in range(1)] * 3

    def _apply_diagonal(self, inp: Grid, dr: int, dc: int, new_color: int) -> Grid:
        """Compute 2-positions in BASE tile with modular wrap, then build tiled output."""
        H = len(inp); W = len(inp[0]) if H else 0
        # Find 2-cells in the base tile
        two_cells = set()
        for r in range(H):
            for c in range(W):
                if inp[r][c] != 0:
                    nr = (r + dr) % H
                    nc = (c + dc) % W
                    if inp[nr][nc] == 0:
                        two_cells.add((nr, nc))
        # Build tiled output
        result = [list(row) * 3 for row in inp] * 3
        # Apply 2-cells to all 9 tile copies
        for ti in range(3):
            for tj in range(3):
                for nr, nc in two_cells:
                    result[nr + ti * H][nc + tj * W] = new_color
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for dr, dc in self.OFFSETS:
            ok = True
            new_color_found = None
            for pair in ctx.train_pairs:
                inp = pair["input"]
                out = pair["output"]
                out_colors = set(v for row in out for v in row)
                inp_colors = set(v for row in inp for v in row)
                new_colors = out_colors - inp_colors
                if not new_colors:
                    ok = False; break
                nc_val = next(iter(new_colors))
                if new_color_found is None:
                    new_color_found = nc_val
                elif new_color_found != nc_val:
                    ok = False; break
                pred = self._apply_diagonal(inp, dr, dc, nc_val)
                if pred != out:
                    ok = False; break
            if ok and new_color_found is not None:
                preds = [self._apply_diagonal(t, dr, dc, new_color_found)
                         for t in ctx.test_inputs]
                return _make_result(ctx, preds)
        return None


# ---------------------------------------------------------------------------
# 3. FillFrameInteriorSolver  (REDESIGNED: per-component rectangle fill)
# ---------------------------------------------------------------------------

@register_solver
class FillFrameInteriorSolver(BaseSolver):
    """
    Finds all closed rectangular frames of 1-cells.  Locates non-1/non-0 seeds
    enclosed by the frame.  Propagates fill color through all rectangles in the
    same connected 1-component.  Interior cells (both 0 and 1) are replaced.
    Confirmed: 09c534e7
    """

    @property
    def name(self) -> str:
        return "FillFrameInteriorSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    # ---- helpers -----------------------------------------------------------

    def _exterior_mask(self, grid: Grid):
        """BFS from boundary through non-1 cells → exterior boolean mask."""
        H = len(grid); W = len(grid[0]) if H else 0
        ext = [[False] * W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                if (r == 0 or r == H - 1 or c == 0 or c == W - 1) and grid[r][c] != 1:
                    if not ext[r][c]:
                        ext[r][c] = True
                        q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not ext[nr][nc] and grid[nr][nc] != 1:
                    ext[nr][nc] = True
                    q.append((nr, nc))
        return ext

    def _find_1components(self, grid: Grid):
        """Return (comp_map, list_of_cell_lists) for 1-valued cells."""
        H = len(grid); W = len(grid[0]) if H else 0
        comp_map = [[-1] * W for _ in range(H)]
        comp_cells = []
        for sr in range(H):
            for sc in range(W):
                if grid[sr][sc] == 1 and comp_map[sr][sc] == -1:
                    cid = len(comp_cells)
                    comp_cells.append([])
                    q = deque([(sr, sc)])
                    comp_map[sr][sc] = cid
                    while q:
                        r, c = q.popleft()
                        comp_cells[cid].append((r, c))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < H and 0 <= nc < W
                                    and grid[nr][nc] == 1
                                    and comp_map[nr][nc] == -1):
                                comp_map[nr][nc] = cid
                                q.append((nr, nc))
        return comp_map, comp_cells

    def _find_rects_in_component(self, cells):
        """Find all valid closed rectangles within a set of cells."""
        cell_set = set(cells)
        # Index by row and column
        cols_of_row = {}
        rows_of_col = {}
        for r, c in cells:
            cols_of_row.setdefault(r, set()).add(c)
            rows_of_col.setdefault(c, set()).add(r)

        rects = []
        sorted_rows = sorted(cols_of_row)
        for i, r1 in enumerate(sorted_rows):
            for r2 in sorted_rows[i + 1:]:
                if r2 <= r1 + 1:
                    continue
                # Columns present in both rows
                common = sorted(cols_of_row.get(r1, set()) & cols_of_row.get(r2, set()))
                for j, c1 in enumerate(common):
                    for c2 in common[j + 1:]:
                        if c2 <= c1 + 1:
                            continue
                        # Check all top/bottom row cells present
                        if not all((r1, c) in cell_set for c in range(c1, c2 + 1)):
                            continue
                        if not all((r2, c) in cell_set for c in range(c1, c2 + 1)):
                            continue
                        # Check left/right column cells present
                        if not all((r, c1) in cell_set for r in range(r1, r2 + 1)):
                            continue
                        if not all((r, c2) in cell_set for r in range(r1, r2 + 1)):
                            continue
                        rects.append((r1, c1, r2, c2))
        return rects

    # ---- main predict ------------------------------------------------------

    def _predict(self, grid: Grid) -> Optional[Grid]:
        H = len(grid); W = len(grid[0]) if H else 0
        ext = self._exterior_mask(grid)

        # Enclosed non-zero non-1 seeds
        seeds = [(r, c, grid[r][c])
                 for r in range(H) for c in range(W)
                 if grid[r][c] not in (0, 1) and not ext[r][c]]
        if not seeds:
            return None

        comp_map, comp_cells = self._find_1components(grid)

        # Map each seed to its enclosing 1-component (all 4 scan directions agree)
        comp_fill = {}  # cid -> fill_color
        for r, c, color in seeds:
            adj = set()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                for dist in range(1, max(H, W)):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if grid[nr][nc] == 1:
                        adj.add(comp_map[nr][nc])
                        break
            for cid in adj:
                if cid < 0:
                    continue
                if cid not in comp_fill:
                    comp_fill[cid] = color
                elif comp_fill[cid] != color:
                    comp_fill[cid] = None  # conflict → skip

        # Fill rectangles for each component with a valid seed color
        out = [list(row) for row in grid]
        changed = False
        for cid, fill_color in comp_fill.items():
            if fill_color is None:
                continue
            cells = comp_cells[cid]
            # Skip very large components to avoid O(n^4) slow-down
            if len(cells) > 500:
                continue
            rects = self._find_rects_in_component(cells)
            for r1, c1, r2, c2 in rects:
                for r in range(r1 + 1, r2):
                    for c in range(c1 + 1, c2):
                        if out[r][c] != fill_color:
                            out[r][c] = fill_color
                            changed = True

        return out if changed else None


# ---------------------------------------------------------------------------
# 4. StampRepeatSolver  (FIXED: template = ALL template_color cells)
# ---------------------------------------------------------------------------

@register_solver
class StampRepeatSolver(BaseSolver):
    """
    One color (8) forms the multi-pixel template blob; another color (4) appears
    only as single-pixel markers (N of them).  Output = template tiled N times.

    Fix: template bounding box includes ALL cells of template_color (including
    any isolated single pixels of that color).  Template size is expanded to the
    max dimension, choosing the window that contains no marker cells.
    Confirmed: 4852f2fa
    """

    @property
    def name(self) -> str:
        return "StampRepeatSolver"

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
        H = len(grid); W = len(grid[0]) if H else 0
        comps = _bfs_components(grid)
        if not comps:
            return None

        multi_comps = [(c, cells) for c, cells in comps if len(cells) > 1]
        single_comps = [(c, cells) for c, cells in comps if len(cells) == 1]

        # Exactly one template color (the one with multi-pixel components)
        if len(multi_comps) != 1:
            return None
        template_color = multi_comps[0][0]

        # Marker = single-pixel cells of a DIFFERENT color
        marker_comps = [(c, cells) for c, cells in single_comps if c != template_color]
        if not marker_comps:
            return None
        marker_colors = set(c for c, _ in marker_comps)
        if len(marker_colors) != 1:
            return None  # ambiguous marker color

        n_markers = len(marker_comps)
        marker_color = next(iter(marker_colors))
        marker_cells_set = set(cells[0] for _, cells in marker_comps)

        # Template = ALL cells of template_color (multi + any isolated single pixels)
        all_tc = [(r, c) for color, cells in comps if color == template_color
                  for r, c in cells]
        rows_t = [r for r, c in all_tc]
        cols_t = [c for r, c in all_tc]
        r1, r2 = min(rows_t), max(rows_t)
        c1, c2 = min(cols_t), max(cols_t)
        th, tw = r2 - r1 + 1, c2 - c1 + 1
        max_dim = max(th, tw)

        # Find valid square window (max_dim × max_dim) containing all tc cells,
        # with no marker_color cells inside
        tmpl = None
        for row_off in range(max(0, H - max_dim + 1)):
            for col_off in range(max(0, c1 - (max_dim - tw)),
                                  min(W - max_dim, c1) + 1):
                # Window bounds
                wr1, wr2 = row_off, row_off + max_dim - 1
                wc1, wc2 = col_off, col_off + max_dim - 1
                # Must contain all tc cells
                if not all(wr1 <= r <= wr2 and wc1 <= c <= wc2 for r, c in all_tc):
                    continue
                # Must contain no marker cells
                if any(wr1 <= r <= wr2 and wc1 <= c <= wc2 for r, c in marker_cells_set):
                    continue
                # Extract template
                tmpl = [list(grid[r][wc1:wc2 + 1]) for r in range(wr1, wr2 + 1)]
                break
            if tmpl is not None:
                break

        if tmpl is None:
            # Fallback: use tight bounding box (non-square allowed)
            tmpl = [list(grid[r][c1:c2 + 1]) for r in range(r1, r2 + 1)]

        # Build output: tile template n_markers times in a row
        if n_markers == 0:
            return None
        th_t = len(tmpl)
        tw_t = len(tmpl[0]) if th_t else 0
        out = [tmpl[r][:] * n_markers for r in range(th_t)]
        return out
