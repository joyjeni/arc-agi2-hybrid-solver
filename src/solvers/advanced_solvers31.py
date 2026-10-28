"""
advanced_solvers31.py — Two solvers from Robina Mirbahar reference notebook

1. RayGuidedFaceRecolorSolver  (task 3d588dc9)
   — Locate gray-5 directional wedge; find largest black-0 connected component
     that horizontally overlaps with the wedge's row range; prune 1-cell-wide
     spikes (non-core cells → bg 7); recolor the face column facing the wedge
     to magenta 6.

2. PerimeterBFSCavitySolver    (task 00d62c1b)
   — BFS from every border 0-cell outward; any interior 0-cell never reached
     is a cavity → recolor it to yellow 4.
"""

from collections import Counter, deque

from src.core.interfaces import BaseSolver, Grid, Prediction, SolverResult, TaskContext
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


# ---------------------------------------------------------------------------
# Solver 1 — Ray-Guided Face Recoloring & Spike Pruning  (task 3d588dc9)
# ---------------------------------------------------------------------------

def _solve_3d588dc9(g: Grid) -> Grid | None:
    h = len(g)
    if h == 0:
        return None
    w = len(g[0])

    # Background = most-common color (expect 7=orange)
    flat = [v for row in g for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Find gray (5) wedge cells
    gray_cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] == 5]
    if not gray_cells:
        return None
    g_rmin = min(r for r, _ in gray_cells)
    g_rmax = max(r for r, _ in gray_cells)
    g_cmean = sum(c for _, c in gray_cells) / len(gray_cells)

    # BFS: find all black (0) connected components ≥6 cells
    vis = [[False] * w for _ in range(h)]
    best_comp: list[tuple[int, int]] | None = None
    best_overlap = -1

    for sr in range(h):
        for sc in range(w):
            if g[sr][sc] == 0 and not vis[sr][sc]:
                q: deque = deque([(sr, sc)])
                vis[sr][sc] = True
                cells: list[tuple[int, int]] = []
                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not vis[nr][nc] and g[nr][nc] == 0:
                            vis[nr][nc] = True
                            q.append((nr, nc))
                if len(cells) < 6:
                    continue
                rs = [r for r, _ in cells]
                overlap = max(0, min(max(rs), g_rmax) - max(min(rs), g_rmin) + 1)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_comp = cells

    if best_comp is None:
        return None

    out = _copy_grid(g)

    # Count how many component cells per row/col
    row_cnt: Counter = Counter(r for r, _ in best_comp)
    col_cnt: Counter = Counter(c for _, c in best_comp)

    # Core = rows with ≥3 cells AND cols with ≥3 cells
    core_rs = {r for r, n in row_cnt.items() if n >= 3}
    core_cs = {c for c, n in col_cnt.items() if n >= 3}

    # Prune spike cells (not fully in core)
    for r, c in best_comp:
        if r not in core_rs or c not in core_cs:
            out[r][c] = bg

    if not core_rs or not core_cs:
        return out

    # Face column: whichever side of core faces the wedge
    core_cs_sorted = sorted(core_cs)
    core_cs_mean = sum(core_cs_sorted) / len(core_cs_sorted)
    target_c = core_cs_sorted[0] if g_cmean < core_cs_mean else core_cs_sorted[-1]

    for r in sorted(core_rs):
        out[r][target_c] = 6

    return out


@register_solver
class RayGuidedFaceRecolorSolver(BaseSolver):
    """
    task 3d588dc9 — Gray-5 wedge points at black-0 object on orange-7 field.
    Prune 1-cell-wide spikes from the object, then recolor the face column
    nearest the wedge to magenta-6.
    """

    @property
    def name(self) -> str:
        return "RayGuidedFaceRecolorSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # Verify on all train pairs
        for pair in ctx.train_pairs:
            pred = _solve_3d588dc9(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        # Solve test inputs
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_3d588dc9(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ---------------------------------------------------------------------------
# Solver 2 — Perimeter BFS Cavity Enclosure  (task 00d62c1b)
# ---------------------------------------------------------------------------

def _solve_00d62c1b(g: Grid) -> Grid | None:
    h = len(g)
    if h == 0:
        return None
    w = len(g[0])

    vis = [[False] * w for _ in range(h)]
    q: deque = deque()

    # Seed BFS from all border 0-cells
    for r in range(h):
        for c in [0, w - 1]:
            if g[r][c] == 0 and not vis[r][c]:
                vis[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in [0, h - 1]:
            if g[r][c] == 0 and not vis[r][c]:
                vis[r][c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not vis[nr][nc] and g[nr][nc] == 0:
                vis[nr][nc] = True
                q.append((nr, nc))

    # Interior 0-cells not reached → cavity → color 4
    out = _copy_grid(g)
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0 and not vis[r][c]:
                out[r][c] = 4
    return out


@register_solver
class PerimeterBFSCavitySolver(BaseSolver):
    """
    task 00d62c1b — Flood-fill exterior background (0) from the grid perimeter
    via 4-connected BFS; any 0-cell not reached is an enclosed interior cavity
    → recolor to yellow 4.
    """

    @property
    def name(self) -> str:
        return "PerimeterBFSCavitySolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # Verify on all train pairs
        for pair in ctx.train_pairs:
            pred = _solve_00d62c1b(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None
        # Solve test inputs
        preds = []
        for test_input in ctx.test_inputs:
            grid = _solve_00d62c1b(test_input)
            if grid is None:
                return None
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ===========================================================================
# 3. ShapeRecolorByKeyColorSolver  —  task aabf363d
# ===========================================================================
#
# Rule:
#   Grid contains one main shape (all cells of one non-bg color) and a single
#   "key" cell at a corner of the grid (row 0 or last row, col 0 or last col).
#   Output: all main-shape cells → key color; key cell → bg (0).
#   Works regardless of which corner the key cell occupies.
#
# ===========================================================================

@register_solver
class ShapeRecolorByKeyColorSolver(BaseSolver):
    name = "ShapeRecolorByKeyColorSolver"

    @property
    def priority(self) -> int:
        return 11  # just after PaletteRemapInducerSolver (10)

    def _find_key_cell(self, grid):
        """Return (row, col, color) of isolated corner key cell, or None."""
        H, W = len(grid), len(grid[0])
        corners = [(0, 0), (0, W-1), (H-1, 0), (H-1, W-1)]
        from collections import Counter
        flat = [v for row in grid for v in row]
        bg = Counter(flat).most_common(1)[0][0]
        for r, c in corners:
            v = grid[r][c]
            if v == bg:
                continue
            # Count all cells of this color
            n = sum(1 for row in grid for cell in row if cell == v)
            if n == 1:
                return r, c, v
        return None

    def _solve_one(self, inp):
        H, W = len(inp), len(inp[0])
        from collections import Counter
        flat = [v for row in inp for v in row]
        bg = Counter(flat).most_common(1)[0][0]
        key = self._find_key_cell(inp)
        if key is None:
            return None
        kr, kc, key_color = key
        # Find main shape color: most frequent non-bg non-key color
        non_bg = [(v, cnt) for v, cnt in Counter(flat).items() if v != bg and v != key_color]
        if not non_bg:
            return None
        shape_color = max(non_bg, key=lambda x: x[1])[0]
        out = [row[:] for row in inp]
        for r in range(H):
            for c in range(W):
                if out[r][c] == shape_color:
                    out[r][c] = key_color
        out[kr][kc] = bg
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        # Verify all training pairs
        for p in ctx.train_pairs:
            pred = self._solve_one(p["input"])
            if pred is None or pred != p["output"]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._solve_one(inp)
            if g is None:
                return None
            preds.append(Prediction(attempt_1=g, attempt_2=g,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ===========================================================================
# 4. SeedAdjacentGroupRecolorSolver  —  task b7256dcd
# ===========================================================================
#
# Rule:
#   Background = bg (most-frequent color).
#   Main shape color = second most-frequent color.
#   "Seed" cells = isolated single cells that are neither bg nor main shape.
#   For each seed, find the 4-connected group of main-shape cells that is
#   4-adjacent (shares an edge) to the seed.  Recolor that group to seed color.
#   Seed cell itself becomes bg.
#   Main-shape groups with no adjacent seed remain unchanged.
#
# ===========================================================================

@register_solver
class SeedAdjacentGroupRecolorSolver(BaseSolver):
    name = "SeedAdjacentGroupRecolorSolver"

    @property
    def priority(self) -> int:
        return 79

    def _find_groups(self, grid, color):
        """Return list of 4-connected component cell-sets for a given color."""
        H, W = len(grid), len(grid[0])
        visited = set()
        groups = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color and (r, c) not in visited:
                    comp = set()
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited: continue
                        if cr < 0 or cr >= H or cc < 0 or cc >= W: continue
                        if grid[cr][cc] != color: continue
                        visited.add((cr, cc))
                        comp.add((cr, cc))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((cr+dr, cc+dc))
                    groups.append(frozenset(comp))
        return groups

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        from collections import Counter
        flat = [v for row in inp for v in row]
        cnt = Counter(flat)
        if len(cnt) < 3:
            return None
        bg = cnt.most_common(1)[0][0]
        shape_color = cnt.most_common(2)[1][0]

        # Find seed cells: isolated single non-bg non-shape cells
        seeds = []
        for r in range(H):
            for c in range(W):
                v = inp[r][c]
                if v == bg or v == shape_color:
                    continue
                # Must be isolated (no 4-neighbor of same color)
                neighbors_same = sum(
                    1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                    if 0 <= r+dr < H and 0 <= c+dc < W and inp[r+dr][c+dc] == v
                )
                if neighbors_same == 0:
                    seeds.append((r, c, v))

        if not seeds:
            return None

        # Get all shape groups
        groups = self._find_groups(inp, shape_color)

        # Build adjacency: seed → group
        seed_group_map = {}
        for sr, sc, sv in seeds:
            adj_cells = set()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = sr+dr, sc+dc
                if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] == shape_color:
                    adj_cells.add((nr, nc))
            # Find which group these adjacent cells belong to
            matched_groups = set()
            for g in groups:
                if g & adj_cells:
                    matched_groups.add(g)
            if len(matched_groups) > 1:
                return None  # ambiguous — seed touches multiple groups
            if len(matched_groups) == 1:
                seed_group_map[(sr, sc)] = (sv, next(iter(matched_groups)))
            else:
                # No adjacent shape group — seed just becomes bg
                seed_group_map[(sr, sc)] = (None, None)

        out = [row[:] for row in inp]
        for (sr, sc), (sv, grp) in seed_group_map.items():
            if grp is not None:
                for r, c in grp:
                    out[r][c] = sv
            out[sr][sc] = bg
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            pred = self._apply(p["input"])
            if pred is None or pred != p["output"]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp)
            if g is None:
                return None
            preds.append(Prediction(attempt_1=g, attempt_2=g,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)
