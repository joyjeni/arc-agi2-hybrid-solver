"""
advanced_solvers13.py — 4 new solvers from deep crop-task analysis

  1. SymmetricQuadrantCropSolver  (tasks 2013d3e2, 73182012)
  2. OddOneOutFillSolver          (task 358ba94e)
  3. MostClustersColorSolver      (task 39a8645d)
  4. MarkerNearestClusterSolver   (task 48d8fb45)
"""

from collections import Counter, deque

from src.core.interfaces import BaseSolver, Grid, Prediction, SolverResult, TaskContext

from .registry import register_solver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_of_nonzero(grid: Grid):
    H, W = len(grid), len(grid[0])
    r0, r1, c0, c1 = H, -1, W, -1
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                r0 = min(r0, r); r1 = max(r1, r)
                c0 = min(c0, c); c1 = max(c1, c)
    return (r0, c0, r1, c1) if r1 >= 0 else None


def _extract(grid: Grid, r0, c0, r1, c1) -> Grid:
    return [list(grid[r][c0:c1+1]) for r in range(r0, r1+1)]


def _components_4(grid: Grid, color: int):
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    comps = []
    for sr in range(H):
        for sc in range(W):
            if grid[sr][sc] == color and not visited[sr][sc]:
                q = deque([(sr, sc)]); visited[sr][sc] = True; cells = []
                while q:
                    r, c = q.popleft(); cells.append((r, c))
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True; q.append((nr, nc))
                comps.append(cells)
    return comps


def _components_8_color(grid: Grid, color: int):
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    comps = []
    for sr in range(H):
        for sc in range(W):
            if grid[sr][sc] == color and not visited[sr][sc]:
                q = deque([(sr, sc)]); visited[sr][sc] = True; cells = []
                while q:
                    r, c = q.popleft(); cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True; q.append((nr, nc))
                comps.append(cells)
    return comps


def _components_8_mask(grid: Grid, exclude=(0, 5)):
    """8-connected components ignoring cells with value in `exclude`."""
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    comps = []
    for sr in range(H):
        for sc in range(W):
            if grid[sr][sc] not in exclude and not visited[sr][sc]:
                q = deque([(sr, sc)]); visited[sr][sc] = True; cells = []
                while q:
                    r, c = q.popleft(); cells.append((r, c))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0: continue
                            nr, nc = r+dr, c+dc
                            if (0 <= nr < H and 0 <= nc < W
                                    and not visited[nr][nc]
                                    and grid[nr][nc] not in exclude):
                                visited[nr][nc] = True; q.append((nr, nc))
                comps.append(cells)
    return comps


def _cells_bbox(cells):
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _non_zero_colors(grid: Grid):
    seen = set()
    for row in grid:
        for v in row:
            if v != 0: seen.add(v)
    return list(seen)


def _make_result(ctx: TaskContext, preds: list[Grid]) -> SolverResult:
    predictions = [
        Prediction(attempt_1=p, attempt_2=p, solver_name=ctx.task_id)
        for p in preds
    ]
    return SolverResult(task_id=ctx.task_id, predictions=predictions,
                        solved=True, tier=1)


# ---------------------------------------------------------------------------
# 1. SymmetricQuadrantCropSolver
# ---------------------------------------------------------------------------

@register_solver
class SymmetricQuadrantCropSolver(BaseSolver):
    """
    Extract the top-left quadrant of the non-zero bounding box.
    Bounding box H×W → output is (H//2) × (W//2) top-left corner.
    Confirmed on: 2013d3e2, 73182012.
    """

    @property
    def name(self) -> str:
        return "SymmetricQuadrantCropSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    def _predict(self, grid: Grid) -> Grid | None:
        bb = _bbox_of_nonzero(grid)
        if bb is None: return None
        r0, c0, r1, c1 = bb
        H = r1 - r0 + 1; W = c1 - c0 + 1
        if H < 2 or W < 2: return None
        return _extract(grid, r0, c0, r0 + H//2 - 1, c0 + W//2 - 1)


# ---------------------------------------------------------------------------
# 2. OddOneOutFillSolver
# ---------------------------------------------------------------------------

@register_solver
class OddOneOutFillSolver(BaseSolver):
    """
    Among same-color same-size 4-connected blocks (≥3 in the group),
    the one with a unique hole count (differs from mode) is the answer.
    Return its bounding-box crop.
    Confirmed on: 358ba94e.
    """

    @property
    def name(self) -> str:
        return "OddOneOutFillSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    def _predict(self, grid: Grid) -> Grid | None:
        colors = _non_zero_colors(grid)
        for col in colors:
            comps = _components_4(grid, col)
            if len(comps) < 3: continue
            bboxes = [_cells_bbox(c) for c in comps]
            sizes = [(r1-r0+1, c1-c0+1) for r0, c0, r1, c1 in bboxes]
            size_groups: dict = {}
            for i, sz in enumerate(sizes):
                size_groups.setdefault(sz, []).append(i)
            for sz, idxs in size_groups.items():
                if len(idxs) < 3: continue
                holes = [sum(1 for r in range(bboxes[i][0], bboxes[i][2]+1)
                              for c in range(bboxes[i][1], bboxes[i][3]+1)
                              if grid[r][c] == 0)
                         for i in idxs]
                mode = Counter(holes).most_common(1)[0][0]
                unique = [j for j, h in enumerate(holes) if h != mode]
                if len(unique) != 1: continue
                r0, c0, r1, c1 = bboxes[idxs[unique[0]]]
                return _extract(grid, r0, c0, r1, c1)
        return None


# ---------------------------------------------------------------------------
# 3. MostClustersColorSolver
# ---------------------------------------------------------------------------

@register_solver
class MostClustersColorSolver(BaseSolver):
    """
    Find the color with the most 8-connected clusters (all same size).
    Return the top-left cluster's bounding-box crop.
    Confirmed on: 39a8645d.
    """

    @property
    def name(self) -> str:
        return "MostClustersColorSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    def _predict(self, grid: Grid) -> Grid | None:
        colors = _non_zero_colors(grid)
        best_n = 1  # require at least 2 clusters
        best_bboxes = None
        for col in colors:
            comps = _components_8_color(grid, col)
            if len(comps) <= best_n:
                continue
            bboxes = [_cells_bbox(c) for c in comps]
            # Generalized: no uniform-size constraint — just most clusters
            best_n = len(comps)
            best_bboxes = bboxes
        if best_bboxes is None:
            return None
        tl = min(best_bboxes, key=lambda b: (b[0], b[1]))
        r0, c0, r1, c1 = tl
        return _extract(grid, r0, c0, r1, c1)


# ---------------------------------------------------------------------------
# 4. MarkerNearestClusterSolver
# ---------------------------------------------------------------------------

@register_solver
class MarkerNearestClusterSolver(BaseSolver):
    """
    Color-5 is a single-cell marker. 8-connected clusters of all non-zero,
    non-5 cells; return the cluster whose bbox center has minimum Manhattan
    distance to the marker.
    Confirmed on: 48d8fb45.
    """

    @property
    def name(self) -> str:
        return "MarkerNearestClusterSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or pred != pair["output"]:
                return None
        preds = [self._predict(t) for t in ctx.test_inputs]
        if any(p is None for p in preds):
            return None
        return _make_result(ctx, preds)

    def _predict(self, grid: Grid) -> Grid | None:
        H, W = len(grid), len(grid[0])
        # Locate marker
        marker = None
        for r in range(H):
            for c in range(W):
                if grid[r][c] == 5:
                    marker = (r, c); break
            if marker: break
        if marker is None: return None
        mr, mc = marker
        clusters = _components_8_mask(grid, exclude=(0, 5))
        if not clusters: return None
        best_dist = float('inf'); best_bb = None
        for cells in clusters:
            r0, c0, r1, c1 = _cells_bbox(cells)
            dist = abs((r0+r1)/2 - mr) + abs((c0+c1)/2 - mc)
            if dist < best_dist:
                best_dist = dist; best_bb = (r0, c0, r1, c1)
        r0, c0, r1, c1 = best_bb
        return _extract(grid, r0, c0, r1, c1)
