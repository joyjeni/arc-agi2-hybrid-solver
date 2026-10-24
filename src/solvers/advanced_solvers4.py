"""
advanced_solvers4.py — Fourth batch of targeted solvers for ARC-AGI-2.

Targets identified in unsolved-task analysis (session 4):
  1.  KeepLargestComponentSolver      — keep only the largest same-color CC; erase rest
  2.  KeepLargestAnyColorSolver       — keep only the largest CC ignoring color
  3.  FloodFillByNeighborColorSolver  — fill each enclosed bg region w/ adjacent non-bg color
  4.  LRSymmetryCompleteSolver        — mirror non-bg cells horizontally into bg cells
  5.  TBSymmetryCompleteSolver        — mirror non-bg cells vertically into bg cells
  6.  LRSymmetryRemoveSolver          — remove non-bg cells that break LR symmetry
  7.  TBSymmetryRemoveSolver          — remove non-bg cells that break TB symmetry
  8.  ConcentricChebyshevRingSolver   — fill Chebyshev rings from sparse input markers
  9.  DiagonalXSpreadSolver           — spread non-bg cell color along all 4 diagonals
 10.  CrossLineFillSolver             — each non-bg cell fills its full row + full column
 11.  RowColSingleLineSolver          — each non-bg cell fills EITHER its row OR its col
 12.  RemoveMinorityColorSolver       — remove cells of the rarest non-bg color
 13.  ColorByRowIndexSolver           — recolor each row's non-bg cells to row's dominant color
 14.  ColorByColIndexSolver           — recolor each col's non-bg cells to col's dominant color
 15.  ExpandObjectBy1Solver           — grow each non-bg object by 1 pixel (4-connected)
"""

from __future__ import annotations
import copy
from collections import Counter, deque
from typing import Optional, List, Tuple, Set, Dict

from ..core.interfaces import BaseSolver, SolverResult, TaskContext, Prediction
from ..core.grid_codec import background_color
from .registry import register_solver

Grid = List[List[int]]


# ── shared helpers ─────────────────────────────────────────────────────────────

def _sh(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)

def _clone(g: Grid) -> Grid:
    return [r[:] for r in g]

def _bg(g: Grid) -> int:
    flat = [v for r in g for v in r]
    return Counter(flat).most_common(1)[0][0] if flat else 0

def _result(ctx: TaskContext, grids: List[Grid], name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

def _flood_exterior(g: Grid, bg: int) -> Set[Tuple[int, int]]:
    """BFS from all border-touching bg cells; returns the exterior bg set."""
    H, W = _sh(g)
    vis: Set[Tuple[int, int]] = set()
    q: deque = deque()
    for r in range(H):
        for c in range(W):
            if (r == 0 or r == H - 1 or c == 0 or c == W - 1) and g[r][c] == bg and (r, c) not in vis:
                vis.add((r, c))
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis and g[nr][nc] == bg:
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis

def _same_color_components(g: Grid, bg: int) -> List[Tuple[int, Set[Tuple[int, int]]]]:
    """Return list of (color, cells_set) for each 4-connected same-color component."""
    H, W = _sh(g)
    visited: Set[Tuple[int, int]] = set()
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg and (r, c) not in visited:
                color = g[r][c]
                comp: Set[Tuple[int, int]] = set()
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited or cr < 0 or cr >= H or cc < 0 or cc >= W:
                        continue
                    if g[cr][cc] != color:
                        continue
                    visited.add((cr, cc))
                    comp.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        stack.append((cr + dr, cc + dc))
                comps.append((color, comp))
    return comps

def _any_color_components(g: Grid, bg: int) -> List[Set[Tuple[int, int]]]:
    """Return list of cell sets for each 4-connected component (any non-bg)."""
    H, W = _sh(g)
    nonbg = {(r, c) for r in range(H) for c in range(W) if g[r][c] != bg}
    visited: Set[Tuple[int, int]] = set()
    comps = []
    for cell in nonbg:
        if cell not in visited:
            comp: Set[Tuple[int, int]] = set()
            stack = [cell]
            while stack:
                cur = stack.pop()
                if cur in visited or cur not in nonbg:
                    continue
                visited.add(cur)
                comp.add(cur)
                cr, cc = cur
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    stack.append((cr + dr, cc + dc))
            comps.append(comp)
    return comps

def _same_shape(a: Grid, b: Grid) -> bool:
    return len(a) == len(b) and (not a or len(a[0]) == len(b[0]))

def _grids_equal(a: Grid, b: Grid) -> bool:
    return a == b


# ══════════════════════════════════════════════════════════════════════════════
# 1. KeepLargestComponentSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class KeepLargestComponentSolver(BaseSolver):
    """Keep only the largest same-color connected component; set all others to bg."""

    @property
    def name(self) -> str:
        return "KeepLargestComponentSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        comps = _same_color_components(inp, bg)
        if not comps:
            return None
        # Group components by color
        by_color: Dict[int, List[Set[Tuple[int,int]]]] = {}
        for color, cells in comps:
            by_color.setdefault(color, []).append(cells)
        # Check: at least one color must have multiple components to remove
        if all(len(v) == 1 for v in by_color.values()):
            return None  # nothing to remove
        # For each color, keep only the largest component
        H, W = _sh(inp)
        out = _clone(inp)
        for color, comp_list in by_color.items():
            if len(comp_list) == 1:
                continue  # only one component — nothing to remove
            largest = max(comp_list, key=len)
            keep_cells = largest
            for cells in comp_list:
                if cells is not largest:
                    for (r, c) in cells:
                        out[r][c] = bg
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 2. KeepLargestAnyColorSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class KeepLargestAnyColorSolver(BaseSolver):
    """Keep only the largest connected component (ignoring color); set rest to bg."""

    @property
    def name(self) -> str:
        return "KeepLargestAnyColorSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        comps = _any_color_components(inp, bg)
        if len(comps) < 2:
            return None
        largest = max(comps, key=len)
        H, W = _sh(inp)
        out = _clone(inp)
        for r in range(H):
            for c in range(W):
                if inp[r][c] != bg and (r, c) not in largest:
                    out[r][c] = bg
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FloodFillByNeighborColorSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FloodFillByNeighborColorSolver(BaseSolver):
    """Fill each enclosed bg region with the color of its adjacent non-bg cells.
    
    If a region is adjacent to multiple non-bg colors, use the most common one.
    """

    @property
    def name(self) -> str:
        return "FloodFillByNeighborColorSolver"

    def _get_enclosed_regions(self, g: Grid, bg: int) -> List[Tuple[Set[Tuple[int, int]], int]]:
        """Return list of (enclosed_cells, fill_color) pairs."""
        H, W = _sh(g)
        exterior = _flood_exterior(g, bg)
        # Find all enclosed bg cells
        all_bg = {(r, c) for r in range(H) for c in range(W) if g[r][c] == bg}
        enclosed_all = all_bg - exterior
        if not enclosed_all:
            return []

        # Find connected components of enclosed bg
        visited: Set[Tuple[int, int]] = set()
        regions = []
        for cell in enclosed_all:
            if cell not in visited:
                comp: Set[Tuple[int, int]] = set()
                stack = [cell]
                while stack:
                    cur = stack.pop()
                    if cur in visited or cur not in enclosed_all:
                        continue
                    visited.add(cur)
                    comp.add(cur)
                    cr, cc = cur
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        stack.append((cr + dr, cc + dc))
                # Determine fill color: most common adjacent non-bg color
                adj_colors: Counter = Counter()
                for r, c in comp:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and g[nr][nc] != bg:
                            adj_colors[g[nr][nc]] += 1
                if not adj_colors:
                    continue
                fill_color = adj_colors.most_common(1)[0][0]
                regions.append((comp, fill_color))
        return regions

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        regions = self._get_enclosed_regions(inp, bg)
        if not regions:
            return None
        out = _clone(inp)
        for comp, fill_color in regions:
            for r, c in comp:
                out[r][c] = fill_color
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. LRSymmetryCompleteSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class LRSymmetryCompleteSolver(BaseSolver):
    """Complete LR symmetry: mirror non-bg cells horizontally into bg positions."""

    @property
    def name(self) -> str:
        return "LRSymmetryCompleteSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        out = _clone(inp)
        changed = False
        for r in range(H):
            for c in range(W):
                mirror_c = W - 1 - c
                v = inp[r][c]
                mv = inp[r][mirror_c]
                if v != bg and mv == bg:
                    out[r][mirror_c] = v
                    changed = True
                elif v == bg and mv != bg:
                    out[r][c] = mv
                    changed = True
        if not changed:
            return None
        # Verify result is LR symmetric
        for r in range(H):
            if out[r] != out[r][::-1]:
                return None
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TBSymmetryCompleteSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class TBSymmetryCompleteSolver(BaseSolver):
    """Complete TB symmetry: mirror non-bg cells vertically into bg positions."""

    @property
    def name(self) -> str:
        return "TBSymmetryCompleteSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        out = _clone(inp)
        changed = False
        for r in range(H):
            mirror_r = H - 1 - r
            for c in range(W):
                v = inp[r][c]
                mv = inp[mirror_r][c]
                if v != bg and mv == bg:
                    out[mirror_r][c] = v
                    changed = True
                elif v == bg and mv != bg:
                    out[r][c] = mv
                    changed = True
        if not changed:
            return None
        # Verify TB symmetric
        for r in range(H):
            if out[r] != out[H - 1 - r]:
                return None
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 6. LRSymmetryRemoveSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class LRSymmetryRemoveSolver(BaseSolver):
    """Remove non-bg cells that have no LR mirror partner; keep only symmetric pairs."""

    @property
    def name(self) -> str:
        return "LRSymmetryRemoveSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        out = _clone(inp)
        changed = False
        for r in range(H):
            for c in range(W):
                mirror_c = W - 1 - c
                v = inp[r][c]
                mv = inp[r][mirror_c]
                if v != bg and mv == bg:
                    # This cell has no mirror partner → remove it
                    out[r][c] = bg
                    changed = True
                elif v != bg and mv != bg and v != mv:
                    # Conflicting colors at mirror position — skip this task
                    return None
        if not changed:
            return None
        # Verify result is LR symmetric
        for r in range(H):
            if out[r] != out[r][::-1]:
                return None
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 7. TBSymmetryRemoveSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class TBSymmetryRemoveSolver(BaseSolver):
    """Remove non-bg cells that have no TB mirror partner; keep only symmetric pairs."""

    @property
    def name(self) -> str:
        return "TBSymmetryRemoveSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        out = _clone(inp)
        changed = False
        for r in range(H):
            mirror_r = H - 1 - r
            for c in range(W):
                v = inp[r][c]
                mv = inp[mirror_r][c]
                if v != bg and mv == bg:
                    out[r][c] = bg
                    changed = True
                elif v != bg and mv != bg and v != mv:
                    return None
        if not changed:
            return None
        for r in range(H):
            if out[r] != out[H - 1 - r]:
                return None
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 8. ConcentricChebyshevRingSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ConcentricChebyshevRingSolver(BaseSolver):
    """Scattered non-bg input cells define colors for Chebyshev rings from center.
    
    Input cell at Chebyshev distance d from grid center → output fills every cell
    at Chebyshev distance d with that color. Cells at distance d with no marker
    keep bg color.
    """

    @property
    def name(self) -> str:
        return "ConcentricChebyshevRingSolver"

    def _chebyshev_dist(self, r: int, c: int, cr: float, cc: float) -> int:
        return int(max(abs(r - cr), abs(c - cc)))

    def _learn_ring_map(self, inp: Grid, bg: int) -> Optional[Dict[int, int]]:
        H, W = _sh(inp)
        cr, cc = (H - 1) / 2.0, (W - 1) / 2.0
        ring_map: Dict[int, int] = {}
        nonbg_rows = []
        for r in range(H):
            for c in range(W):
                if inp[r][c] != bg:
                    nonbg_rows.append(r)
                    d = self._chebyshev_dist(r, c, cr, cc)
                    color = inp[r][c]
                    if d in ring_map and ring_map[d] != color:
                        return None  # conflicting colors for same ring
                    ring_map[d] = color
        if not ring_map:
            return None
        # Reject strip patterns: if all non-bg cells are confined to top or bottom
        # strip (row spread < 30% of height), this is not a ring task
        if nonbg_rows:
            row_spread = max(nonbg_rows) - min(nonbg_rows)
            if row_spread < H * 0.3:
                return None
        return ring_map

    def _apply(self, inp: Grid, ring_map: Dict[int, int], bg: int) -> Grid:
        H, W = _sh(inp)
        cr, cc = (H - 1) / 2.0, (W - 1) / 2.0
        out = _clone(inp)
        for r in range(H):
            for c in range(W):
                d = self._chebyshev_dist(r, c, cr, cc)
                out[r][c] = ring_map.get(d, bg)
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        ring_map = self._learn_ring_map(inp, bg)
        if ring_map is None:
            return False
        pred = self._apply(inp, ring_map, bg)
        return _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            ring_map = self._learn_ring_map(ti, bg)
            if ring_map is None:
                return None
            out = self._apply(ti, ring_map, bg)
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 9. DiagonalXSpreadSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class DiagonalXSpreadSolver(BaseSolver):
    """Each non-bg cell spreads its color along all 4 diagonals (X pattern).
    
    Output cell (r,c) = color of source cell if |r-sr|==|c-sc| for some source (sr,sc).
    When diagonals of different colors cross, skip (return None).
    """

    @property
    def name(self) -> str:
        return "DiagonalXSpreadSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        nonbg = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != bg]
        if not nonbg:
            return None

        out = [[bg] * W for _ in range(H)]
        # Place source cells first (they always appear in output)
        for sr, sc, color in nonbg:
            out[sr][sc] = color

        # Spread along diagonals
        for sr, sc, color in nonbg:
            for r in range(H):
                for c in range(W):
                    if abs(r - sr) == abs(c - sc) and (r, c) != (sr, sc):
                        if out[r][c] == bg:
                            out[r][c] = color
                        elif out[r][c] != color:
                            # Conflict — two source cells want to write different colors
                            # Keep the one that was there; don't fail unless this causes issues
                            pass
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 10. CrossLineFillSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class CrossLineFillSolver(BaseSolver):
    """Each non-bg cell fills its entire row AND column.
    
    At the intersection of two different-colored crosses, the color used
    is learned from training (typically a 'new' intersection color like 2).
    If no consistent intersection color can be learned, the first writer wins.
    """

    @property
    def name(self) -> str:
        return "CrossLineFillSolver"

    def _apply(self, inp: Grid, bg: int, intersect_color: Optional[int] = None) -> Optional[Grid]:
        H, W = _sh(inp)
        nonbg = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != bg]
        if not nonbg or len(nonbg) < 2:
            return None

        # Build row_color and col_color maps
        row_color: Dict[int, int] = {}
        col_color: Dict[int, int] = {}
        for sr, sc, color in nonbg:
            if sr in row_color and row_color[sr] != color:
                return None  # two different colors in same row
            if sc in col_color and col_color[sc] != color:
                return None  # two different colors in same col
            row_color[sr] = color
            col_color[sc] = color

        out = [[bg] * W for _ in range(H)]
        for r in range(H):
            rc = row_color.get(r)
            for c in range(W):
                cc = col_color.get(c)
                if rc is not None and cc is not None:
                    if rc == cc:
                        out[r][c] = rc
                    else:
                        # Intersection of two different colors
                        out[r][c] = intersect_color if intersect_color is not None else rc
                elif rc is not None:
                    out[r][c] = rc
                elif cc is not None:
                    out[r][c] = cc
                # else: stays bg
        return out

    def _learn_intersect_color(self, train: list, bg: int) -> Optional[int]:
        """Learn what color appears at cross-intersections of different-colored crosses."""
        colors_at_intersections: Counter = Counter()
        for p in train:
            inp, out = p['input'], p['output']
            H, W = _sh(inp)
            nonbg = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != bg]
            row_color = {r: col for r, c, col in nonbg}
            col_color = {c: col for r, c, col in nonbg}
            # Check conflicts
            rc_map: Dict[int, int] = {}
            cc_map: Dict[int, int] = {}
            valid = True
            for r, c, color in nonbg:
                if r in rc_map and rc_map[r] != color:
                    valid = False; break
                if c in cc_map and cc_map[c] != color:
                    valid = False; break
                rc_map[r] = color
                cc_map[c] = color
            if not valid:
                continue
            for r in range(H):
                rc = rc_map.get(r)
                for c in range(W):
                    cc = cc_map.get(c)
                    if rc is not None and cc is not None and rc != cc:
                        ov = out[r][c]
                        if ov != bg:
                            colors_at_intersections[ov] += 1
        if not colors_at_intersections:
            return None
        return colors_at_intersections.most_common(1)[0][0]

    def _verify(self, inp: Grid, out: Grid, bg: int, intersect_color: Optional[int]) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg, intersect_color)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None

        intersect_color = self._learn_intersect_color(train, bg)

        for p in train:
            if not self._verify(p['input'], p['output'], bg, intersect_color):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg, intersect_color)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 11. RowColSingleLineSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RowColSingleLineSolver(BaseSolver):
    """Each non-bg cell fills either its full row OR its full column (not both).
    
    Direction (row vs col) is learned per training pair: whichever assignment
    of 'row' or 'col' to each input cell matches the output is kept.
    """

    @property
    def name(self) -> str:
        return "RowColSingleLineSolver"

    def _apply_assignment(self, inp: Grid, bg: int,
                           row_cells: Set[Tuple[int, int]],
                           col_cells: Set[Tuple[int, int]]) -> Optional[Grid]:
        H, W = _sh(inp)
        out = [[bg] * W for _ in range(H)]
        # Fill rows
        for r, c in row_cells:
            color = inp[r][c]
            for cc in range(W):
                if out[r][cc] == bg:
                    out[r][cc] = color
                elif out[r][cc] != color:
                    # Another row cell already filled this row with different color
                    return None
        # Fill cols
        for r, c in col_cells:
            color = inp[r][c]
            for rr in range(H):
                if out[rr][c] == bg:
                    out[rr][c] = color
                elif out[rr][c] != color:
                    # Conflict
                    return None
        return out

    def _learn_directions(self, train: list, bg: int) -> Optional[Tuple[Set[int], Set[int]]]:
        """Return (row_colors, col_colors) that are consistent across all pairs."""
        # Each pair has a set of non-bg cells; try all assignments for first pair
        # then check consistency across other pairs
        if not train:
            return None

        p0_inp = train[0]['input']
        p0_out = train[0]['output']
        H0, W0 = _sh(p0_inp)
        nonbg0 = [(r, c, p0_inp[r][c]) for r in range(H0) for c in range(W0) if p0_inp[r][c] != bg]
        if not nonbg0:
            return None

        # Group by color: same color cells share same direction
        color_cells: Dict[int, List[Tuple[int, int]]] = {}
        for r, c, color in nonbg0:
            color_cells.setdefault(color, []).append((r, c))

        # Each color gets either 'row' or 'col' direction
        # Try to determine from first pair
        color_dirs: Dict[int, str] = {}
        for color, cells in color_cells.items():
            # Check what the output shows for these cells
            # If all their rows are fully filled with this color → 'row'
            # If all their cols are fully filled with this color → 'col'
            rows_filled = all(
                all(p0_out[r][cc] == color for cc in range(W0))
                for r, c in cells
            )
            cols_filled = all(
                all(p0_out[rr][c] == color for rr in range(H0))
                for r, c in cells
            )
            if rows_filled and not cols_filled:
                color_dirs[color] = 'row'
            elif cols_filled and not rows_filled:
                color_dirs[color] = 'col'
            elif rows_filled and cols_filled:
                # Both work (e.g., single cell at corner of grid) — prefer row
                color_dirs[color] = 'row'
            else:
                return None  # neither direction matches

        # Verify across all training pairs using the learned color→direction map
        for p in train:
            inp, out = p['input'], p['output']
            H, W = _sh(inp)
            nonbg = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != bg]
            row_cells_p = set()
            col_cells_p = set()
            for r, c, color in nonbg:
                if color not in color_dirs:
                    return None  # new color not seen in training
                if color_dirs[color] == 'row':
                    row_cells_p.add((r, c))
                else:
                    col_cells_p.add((r, c))
            pred = self._apply_assignment(inp, bg, row_cells_p, col_cells_p)
            if pred is None or not _grids_equal(pred, out):
                return None

        # Return consistent color→direction mapping
        row_colors = {c for c, d in color_dirs.items() if d == 'row'}
        col_colors = {c for c, d in color_dirs.items() if d == 'col'}
        return row_colors, col_colors

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None

        dirs = self._learn_directions(train, bg)
        if dirs is None:
            return None
        row_colors, col_colors = dirs

        results = []
        for ti in ctx.test_inputs:
            H, W = _sh(ti)
            nonbg = [(r, c, ti[r][c]) for r in range(H) for c in range(W) if ti[r][c] != bg]
            row_cells: Set[Tuple[int, int]] = set()
            col_cells: Set[Tuple[int, int]] = set()
            for r, c, color in nonbg:
                if color in row_colors:
                    row_cells.add((r, c))
                elif color in col_colors:
                    col_cells.add((r, c))
                else:
                    return None  # unseen color
            out = self._apply_assignment(ti, bg, row_cells, col_cells)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 12. RemoveMinorityColorSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RemoveMinorityColorSolver(BaseSolver):
    """Remove all cells of the least-frequent non-bg color (set to bg)."""

    @property
    def name(self) -> str:
        return "RemoveMinorityColorSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        nonbg_counts: Counter = Counter()
        for r in range(H):
            for c in range(W):
                if inp[r][c] != bg:
                    nonbg_counts[inp[r][c]] += 1
        if len(nonbg_counts) < 2:
            return None
        minority_color = nonbg_counts.most_common()[-1][0]
        out = _clone(inp)
        for r in range(H):
            for c in range(W):
                if inp[r][c] == minority_color:
                    out[r][c] = bg
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 13. ExpandObjectBy1Solver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ExpandObjectBy1Solver(BaseSolver):
    """Grow each non-bg object by exactly 1 pixel in all 4 cardinal directions."""

    @property
    def name(self) -> str:
        return "ExpandObjectBy1Solver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        out = _clone(inp)
        for r in range(H):
            for c in range(W):
                if inp[r][c] == bg:
                    # Check 4 neighbors
                    adj_colors = Counter()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and inp[nr][nc] != bg:
                            adj_colors[inp[nr][nc]] += 1
                    if adj_colors:
                        out[r][c] = adj_colors.most_common(1)[0][0]
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 14. ExpandObjectByNSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ExpandObjectByNSolver(BaseSolver):
    """Grow each non-bg object by N pixels (learned from training)."""

    @property
    def name(self) -> str:
        return "ExpandObjectByNSolver"

    def _expand_n(self, inp: Grid, bg: int, n: int) -> Grid:
        cur = inp
        for _ in range(n):
            H, W = _sh(cur)
            nxt = [r[:] for r in cur]
            for r in range(H):
                for c in range(W):
                    if cur[r][c] == bg:
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and cur[nr][nc] != bg:
                                nxt[r][c] = cur[nr][nc]
                                break
            cur = nxt
        return cur

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None

        # Learn N from first training pair
        p0 = train[0]
        inp0, out0 = p0['input'], p0['output']
        best_n = None
        for n in range(1, 10):
            pred = self._expand_n(inp0, bg, n)
            if _grids_equal(pred, out0):
                best_n = n
                break
        if best_n is None:
            return None

        # Verify on all pairs
        for p in train:
            pred = self._expand_n(p['input'], bg, best_n)
            if not _grids_equal(pred, p['output']):
                return None

        results = []
        for ti in ctx.test_inputs:
            results.append(self._expand_n(ti, bg, best_n))

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 15. RemoveSmallestComponentSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RemoveSmallestComponentSolver(BaseSolver):
    """Remove all cells of the smallest same-color connected component."""

    @property
    def name(self) -> str:
        return "RemoveSmallestComponentSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        comps = _same_color_components(inp, bg)
        if len(comps) < 2:
            return None
        smallest = min(comps, key=lambda x: len(x[1]))
        # Make sure there is a unique smallest
        sizes = sorted(len(c[1]) for c in comps)
        if sizes[0] == sizes[1]:
            return None  # tie → ambiguous
        H, W = _sh(inp)
        out = _clone(inp)
        for r, c in smallest[1]:
            out[r][c] = bg
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 16. RotationalSymmetryCompleteSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RotationalSymmetryCompleteSolver(BaseSolver):
    """Complete 180° rotational symmetry: mirror each non-bg cell to its 180° partner."""

    @property
    def name(self) -> str:
        return "RotationalSymmetryCompleteSolver"

    def _apply(self, inp: Grid, bg: int) -> Optional[Grid]:
        H, W = _sh(inp)
        out = _clone(inp)
        changed = False
        for r in range(H):
            for c in range(W):
                mr, mc = H - 1 - r, W - 1 - c
                v = inp[r][c]
                mv = inp[mr][mc]
                if v != bg and mv == bg:
                    out[mr][mc] = v
                    changed = True
                elif v == bg and mv != bg:
                    out[r][c] = mv
                    changed = True
                elif v != bg and mv != bg and v != mv:
                    return None  # conflict
        if not changed:
            return None
        # Verify 180° symmetric
        for r in range(H):
            for c in range(W):
                if out[r][c] != out[H - 1 - r][W - 1 - c]:
                    return None
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int) -> bool:
        if not _same_shape(inp, out):
            return False
        pred = self._apply(inp, bg)
        return pred is not None and _grids_equal(pred, out)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1:
            return None
        bg = bg_vals[0]

        for p in train:
            if not _same_shape(p['input'], p['output']):
                return None
            if not self._verify(p['input'], p['output'], bg):
                return None

        results = []
        for ti in ctx.test_inputs:
            out = self._apply(ti, bg)
            if out is None:
                return None
            results.append(out)

        if not results:
            return None
        return _result(ctx, results, self.name)
