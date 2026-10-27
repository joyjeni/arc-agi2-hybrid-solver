"""
advanced_solvers3.py — Third batch of targeted solvers for ARC-AGI-2.

Targeting fill-bg-single-color (76 tasks), recolor, and bar-extend buckets:
  1.  FillEnclosedRegionV2Solver    — flood exterior, fill interior bg cells
  2.  FillBoundingBoxBgSolver       — fill bg cells strictly inside non-bg bbox
  3.  FillBetweenSameColorRowSolver — fill bg between same-color endpoints in rows
  4.  FillBetweenSameColorColSolver — fill bg between same-color endpoints in cols
  5.  FillLargestBgComponentSolver  — fill the single largest connected bg region
  6.  FillAllBgSolver               — fill ALL bg cells with one learned color
  7.  SingleColorA2BSolver          — learn A→B recolor for one non-bg color
  8.  RemoveSingleColorSolver       — remove (→bg) a single unique non-bg color
  9.  ExtendVerticalBarsSolver      — extend non-bg cells vertically within their col span
 10.  LearnedColumnFillSolver       — learn which fixed columns always get filled
 11.  FillRowSpanBgSolver           — fill bg between leftmost/rightmost non-bg per row
 12.  FillColSpanBgSolver           — fill bg between topmost/bottommost non-bg per col
"""

from __future__ import annotations
import copy
from collections import Counter, deque
from typing import Optional, List, Tuple, Set

from ..core.interfaces import BaseSolver, SolverResult, TaskContext, Prediction
from ..core.grid_codec import background_color
from .registry import register_solver

Grid = List[List[int]]

# ── helpers ───────────────────────────────────────────────────────────────────

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
                vis.add((r, c)); q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis and g[nr][nc] == bg:
                vis.add((nr, nc)); q.append((nr, nc))
    return vis

def _learn_fill_color(inp: Grid, out: Grid, bg: int) -> Optional[int]:
    """
    Return fill_color if:
      - all changes are bg→fill_color (consistent single color)
      - no non-bg cell changed
    else return None.
    """
    if _sh(inp) != _sh(out): return None  # shape-change task → skip
    colors: Set[int] = set()
    for r in range(len(inp)):
        for c in range(len(inp[0])):
            iv, ov = inp[r][c], out[r][c]
            if iv == bg and ov != bg:
                colors.add(ov)
            elif iv != bg and iv != ov:
                return None          # non-bg cell changed → not a pure fill-bg task
    if len(colors) != 1:
        return None
    return list(colors)[0]

def _all_fill_colors(pairs: list, bg: int) -> Optional[int]:
    """Return fill color if consistent across all training pairs, else None."""
    fills = []
    for p in pairs:
        f = _learn_fill_color(p['input'], p['output'], bg)
        if f is None:
            return None
        fills.append(f)
    return fills[0] if fills and len(set(fills)) == 1 else None


# ══════════════════════════════════════════════════════════════════════════════
# 1. FillEnclosedRegionV2Solver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillEnclosedRegionV2Solver(BaseSolver):
    """Fill bg cells not reachable from the grid border (enclosed interior)."""

    @property
    def name(self) -> str:
        return "FillEnclosedRegionV2Solver"

    def _verify(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        """Check that output matches: interior bg→fill, everything else unchanged."""
        exterior = _flood_exterior(inp, bg)
        H, W = _sh(inp)
        for r in range(H):
            for c in range(W):
                iv, ov = inp[r][c], out[r][c]
                if iv == bg:
                    if (r, c) in exterior:
                        if ov != bg: return False   # exterior bg should stay bg
                    else:
                        if ov != fill: return False  # interior bg should become fill
                else:
                    if iv != ov: return False        # non-bg unchanged
        return True

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        # Verify the exact "enclosed fill" pattern holds for all pairs
        for p in train:
            if not self._verify(p['input'], p['output'], bg, fill):
                return None
            # Must have at least one interior bg cell
            ext = _flood_exterior(p['input'], bg)
            H, W = _sh(p['input'])
            interior = [(r, c) for r in range(H) for c in range(W)
                        if p['input'][r][c] == bg and (r, c) not in ext]
            if not interior:
                return None

        # Apply to test inputs
        results = []
        for ti in ctx.test_inputs:
            if _sh(ti) == (0, 0): continue
            ext = _flood_exterior(ti, bg)
            H, W = _sh(ti)
            out = _clone(ti)
            for r in range(H):
                for c in range(W):
                    if ti[r][c] == bg and (r, c) not in ext:
                        out[r][c] = fill
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FillBoundingBoxBgSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillBoundingBoxBgSolver(BaseSolver):
    """Fill all bg cells that fall strictly inside the bounding box of non-bg cells."""

    @property
    def name(self) -> str:
        return "FillBoundingBoxBgSolver"

    def _bbox(self, g: Grid, bg: int) -> Optional[Tuple[int, int, int, int]]:
        H, W = _sh(g)
        rows = [r for r in range(H) for c in range(W) if g[r][c] != bg]
        cols = [c for r in range(H) for c in range(W) if g[r][c] != bg]
        if not rows: return None
        return min(rows), max(rows), min(cols), max(cols)

    def _verify(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        bbox = self._bbox(inp, bg)
        if bbox is None: return False
        r0, r1, c0, c1 = bbox
        H, W = _sh(inp)
        for r in range(H):
            for c in range(W):
                iv, ov = inp[r][c], out[r][c]
                if iv == bg:
                    inside = r0 < r < r1 and c0 < c < c1
                    if inside and ov != fill: return False
                    if not inside and ov != bg: return False
                else:
                    if iv != ov: return False
        return True

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify(p['input'], p['output'], bg, fill):
                return None

        results = []
        for ti in ctx.test_inputs:
            bbox = self._bbox(ti, bg)
            if bbox is None: continue
            r0, r1, c0, c1 = bbox
            H, W = _sh(ti)
            out = _clone(ti)
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    if ti[r][c] == bg:
                        out[r][c] = fill
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FillBetweenSameColorRowSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillBetweenSameColorRowSolver(BaseSolver):
    """
    For each row: find the leftmost and rightmost cell of the same non-bg color;
    fill bg cells BETWEEN them with the learned fill color.
    """

    @property
    def name(self) -> str:
        return "FillBetweenSameColorRowSolver"

    def _apply(self, g: Grid, bg: int, fill: int) -> Grid:
        H, W = _sh(g)
        out = _clone(g)
        for r in range(H):
            row = g[r]
            # Find each non-bg color's leftmost and rightmost position
            positions: dict = {}
            for c in range(W):
                v = row[c]
                if v != bg:
                    if v not in positions:
                        positions[v] = [c, c]
                    else:
                        positions[v][1] = c
            for v, (l, ri) in positions.items():
                if l < ri:
                    for c in range(l + 1, ri):
                        if g[r][c] == bg:
                            out[r][c] = fill
        return out

    def _verify_pair(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        expected = self._apply(inp, bg, fill)
        return expected == out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify_pair(p['input'], p['output'], bg, fill):
                return None

        results = [self._apply(ti, bg, fill) for ti in ctx.test_inputs]
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FillBetweenSameColorColSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillBetweenSameColorColSolver(BaseSolver):
    """
    For each column: find the topmost and bottommost cell of the same non-bg color;
    fill bg cells BETWEEN them with the learned fill color.
    """

    @property
    def name(self) -> str:
        return "FillBetweenSameColorColSolver"

    def _apply(self, g: Grid, bg: int, fill: int) -> Grid:
        H, W = _sh(g)
        out = _clone(g)
        for c in range(W):
            positions: dict = {}
            for r in range(H):
                v = g[r][c]
                if v != bg:
                    if v not in positions:
                        positions[v] = [r, r]
                    else:
                        positions[v][1] = r
            for v, (t, b) in positions.items():
                if t < b:
                    for r in range(t + 1, b):
                        if g[r][c] == bg:
                            out[r][c] = fill
        return out

    def _verify_pair(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        return self._apply(inp, bg, fill) == out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify_pair(p['input'], p['output'], bg, fill):
                return None

        results = [self._apply(ti, bg, fill) for ti in ctx.test_inputs]
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 5. FillLargestBgComponentSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillLargestBgComponentSolver(BaseSolver):
    """Fill the single largest connected component of bg cells with learned fill color."""

    @property
    def name(self) -> str:
        return "FillLargestBgComponentSolver"

    def _largest_bg_comp(self, g: Grid, bg: int) -> List[Tuple[int, int]]:
        H, W = _sh(g)
        visited: Set[Tuple[int, int]] = set()
        best: List[Tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                if g[r][c] == bg and (r, c) not in visited:
                    # BFS
                    comp: List[Tuple[int, int]] = []
                    q = deque([(r, c)]); visited.add((r, c))
                    while q:
                        cr, cc = q.popleft(); comp.append((cr, cc))
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and g[nr][nc]==bg:
                                visited.add((nr,nc)); q.append((nr,nc))
                    if len(comp) > len(best):
                        best = comp
        return best

    def _verify(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        comp = set(self._largest_bg_comp(inp, bg))
        if not comp: return False
        H, W = _sh(inp)
        for r in range(H):
            for c in range(W):
                iv, ov = inp[r][c], out[r][c]
                if iv == bg:
                    if (r, c) in comp:
                        if ov != fill: return False
                    else:
                        if ov != bg: return False
                else:
                    if iv != ov: return False
        return True

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify(p['input'], p['output'], bg, fill):
                return None

        results = []
        for ti in ctx.test_inputs:
            comp = set(self._largest_bg_comp(ti, bg))
            out = _clone(ti)
            for r, c in comp:
                out[r][c] = fill
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 6. FillAllBgSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillAllBgSolver(BaseSolver):
    """
    Fill ALL bg cells with a single learned fill color.
    Fires when output has zero bg cells and all changes are bg→fill_color.
    """

    @property
    def name(self) -> str:
        return "FillAllBgSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        # Require that every bg cell is filled (no residual bg in output)
        for p in train:
            H, W = _sh(p['output'])
            for r in range(H):
                for c in range(W):
                    if p['output'][r][c] == bg: return None

        results = []
        for ti in ctx.test_inputs:
            H, W = _sh(ti)
            out = _clone(ti)
            for r in range(H):
                for c in range(W):
                    if ti[r][c] == bg:
                        out[r][c] = fill
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 7. SingleColorA2BSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class SingleColorA2BSolver(BaseSolver):
    """
    Learn a single non-bg A→B recolor: one color A changes to B, bg unchanged, 
    all other non-bg colors unchanged.
    """

    @property
    def name(self) -> str:
        return "SingleColorA2BSolver"

    def _infer_remap(self, inp: Grid, out: Grid, bg: int) -> Optional[Tuple[int, int]]:
        """Return (A, B) if exactly color A→B in this pair, else None."""
        changed: dict = {}
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                iv, ov = inp[r][c], out[r][c]
                if iv != ov:
                    if iv == bg: return None   # bg→something (wrong pattern)
                    if iv not in changed:
                        changed[iv] = ov
                    elif changed[iv] != ov:
                        return None             # inconsistent mapping
        if len(changed) != 1: return None
        A, B = list(changed.items())[0]
        # Verify ALL A cells in inp became B in out (complete remap, no partial)
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] == A and out[r][c] == A:
                    return None
        return (A, B)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        remaps = []
        for p in train:
            if _sh(p['input']) != _sh(p['output']): return None
            r = self._infer_remap(p['input'], p['output'], bg)
            if r is None: return None
            remaps.append(r)

        # All pairs must agree on same (A, B)
        if len(set(remaps)) != 1: return None
        A, B = remaps[0]

        results = []
        for ti in ctx.test_inputs:
            out = _clone(ti)
            H, W = _sh(ti)
            for r in range(H):
                for c in range(W):
                    if out[r][c] == A:
                        out[r][c] = B
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 8. RemoveSingleColorSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RemoveSingleColorSolver(BaseSolver):
    """
    Remove exactly one non-bg color: all cells of color X → bg, everything else unchanged.
    """

    @property
    def name(self) -> str:
        return "RemoveSingleColorSolver"

    def _infer_remove(self, inp: Grid, out: Grid, bg: int) -> Optional[int]:
        removed: Set[int] = set()
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                iv, ov = inp[r][c], out[r][c]
                if iv != ov:
                    if ov != bg: return None      # changed to non-bg
                    if iv == bg: return None       # bg changed
                    removed.add(iv)
                    if len(removed) > 1: return None
        if len(removed) != 1: return None
        X = list(removed)[0]
        # Verify ALL X cells in inp are bg in out (complete removal)
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] == X and out[r][c] == X:
                    return None
        return X

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        colors_to_remove = []
        for p in train:
            if _sh(p['input']) != _sh(p['output']): return None
            c = self._infer_remove(p['input'], p['output'], bg)
            if c is None: return None
            colors_to_remove.append(c)

        if len(set(colors_to_remove)) != 1: return None
        X = colors_to_remove[0]

        results = []
        for ti in ctx.test_inputs:
            out = _clone(ti)
            H, W = _sh(ti)
            for r in range(H):
                for c in range(W):
                    if out[r][c] == X:
                        out[r][c] = bg
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 9. ExtendVerticalBarsSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ExtendVerticalBarsSolver(BaseSolver):
    """
    For each column: find the topmost and bottommost non-bg cells.
    Fill ALL bg cells between them in that column with the same non-bg color (if uniform).
    If the column has multiple non-bg colors, use the learned fill color.
    """

    @property
    def name(self) -> str:
        return "ExtendVerticalBarsSolver"

    def _apply(self, g: Grid, bg: int, fill: int) -> Grid:
        H, W = _sh(g)
        out = _clone(g)
        for c in range(W):
            non_bg_rows = [r for r in range(H) if g[r][c] != bg]
            if len(non_bg_rows) < 2: continue
            top, bot = min(non_bg_rows), max(non_bg_rows)
            for r in range(top, bot + 1):
                if out[r][c] == bg:
                    out[r][c] = fill
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        return self._apply(inp, bg, fill) == out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify(p['input'], p['output'], bg, fill):
                return None

        results = [self._apply(ti, bg, fill) for ti in ctx.test_inputs]
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 10. LearnedColumnFillSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class LearnedColumnFillSolver(BaseSolver):
    """
    Learn which specific column indices always get fully/partially filled with fill color.
    The same set of columns must change across all training pairs.
    """

    @property
    def name(self) -> str:
        return "LearnedColumnFillSolver"

    def _changed_cols(self, inp: Grid, out: Grid, bg: int) -> Optional[Set[int]]:
        """Return set of column indices where at least one bg→fill change occurred."""
        cols: Set[int] = set()
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                iv, ov = inp[r][c], out[r][c]
                if iv == bg and ov != bg:
                    cols.add(c)
                elif iv != bg and iv != ov:
                    return None  # non-bg changed
        return cols if cols else None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        col_sets = []
        for p in train:
            cs = self._changed_cols(p['input'], p['output'], bg)
            if cs is None: return None
            col_sets.append(frozenset(cs))

        if len(set(col_sets)) != 1: return None
        target_cols = col_sets[0]
        if not target_cols: return None

        # Verify: in training, those cols were entirely filled (all bg→fill in those cols)
        for p in train:
            inp, out = p['input'], p['output']
            H, W = _sh(inp)
            for c in target_cols:
                if c >= W: return None
                for r in range(H):
                    if inp[r][c] == bg and out[r][c] != fill: return None
            # Verify no changes outside target cols
            for r in range(H):
                for c in range(W):
                    if c not in target_cols and inp[r][c] != out[r][c]:
                        return None

        results = []
        for ti in ctx.test_inputs:
            H, W = _sh(ti)
            out = _clone(ti)
            for c in target_cols:
                if c < W:
                    for r in range(H):
                        if ti[r][c] == bg:
                            out[r][c] = fill
            results.append(out)

        if not results: return None
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 11. FillRowSpanBgSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillRowSpanBgSolver(BaseSolver):
    """
    For each row: fill ALL bg cells between the leftmost and rightmost non-bg cell.
    (More permissive than FillBetweenSameColorRowSolver — uses any non-bg endpoints.)
    """

    @property
    def name(self) -> str:
        return "FillRowSpanBgSolver"

    def _apply(self, g: Grid, bg: int, fill: int) -> Grid:
        H, W = _sh(g)
        out = _clone(g)
        for r in range(H):
            non_bg = [c for c in range(W) if g[r][c] != bg]
            if len(non_bg) < 2: continue
            l, ri = min(non_bg), max(non_bg)
            for c in range(l + 1, ri):
                if out[r][c] == bg:
                    out[r][c] = fill
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        return self._apply(inp, bg, fill) == out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify(p['input'], p['output'], bg, fill):
                return None

        results = [self._apply(ti, bg, fill) for ti in ctx.test_inputs]
        return _result(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 12. FillColSpanBgSolver
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillColSpanBgSolver(BaseSolver):
    """
    For each column: fill ALL bg cells between the topmost and bottommost non-bg cell.
    """

    @property
    def name(self) -> str:
        return "FillColSpanBgSolver"

    def _apply(self, g: Grid, bg: int, fill: int) -> Grid:
        H, W = _sh(g)
        out = _clone(g)
        for c in range(W):
            non_bg = [r for r in range(H) if g[r][c] != bg]
            if len(non_bg) < 2: continue
            t, b = min(non_bg), max(non_bg)
            for r in range(t + 1, b):
                if out[r][c] == bg:
                    out[r][c] = fill
        return out

    def _verify(self, inp: Grid, out: Grid, bg: int, fill: int) -> bool:
        return self._apply(inp, bg, fill) == out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train: return None
        bg_vals = [_bg(p['input']) for p in train]
        if len(set(bg_vals)) != 1: return None
        bg = bg_vals[0]

        fill = _all_fill_colors(train, bg)
        if fill is None: return None

        for p in train:
            if not self._verify(p['input'], p['output'], bg, fill):
                return None

        results = [self._apply(ti, bg, fill) for ti in ctx.test_inputs]
        return _result(ctx, results, self.name)
