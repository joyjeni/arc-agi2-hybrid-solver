"""
advanced_solvers2.py — Second batch of targeted solvers for ARC-AGI-2.

Targeting newly analysed unsolved buckets:
  1. VoronoiColorFillSolver      — fill bg with nearest non-bg color
  2. ConnectEndpointsSolver      — fill between same-color pair endpoints
  3. NearestSolidLineRecolorSolver — recolor cells by nearest full row/col
  4. ExtendRayFromMarkerSolver    — extend ray from unique-color cell
  5. EnclosedFloodFillSolver2    — fill interior via flood-fill with leak detection
  6. RowPatternProjectSolver     — project row patterns onto marker rows
  7. MarkerColorsNearestCellSolver — edge markers recolor nearest target object cell
  8. ColumnPatternProjectSolver  — project col patterns onto marker cols
  9. ConnectSameColorColSolver   — fill col gaps between same-color cells with fill color
 10. TwoColorBandFillSolver      — detect h/v two-color bands and fill intersections
 11. ReflectAboutLineSolver      — reflect non-bg objects about detected symmetry line
 12. ObjectColorByNeighborSolver — recolor objects by the color of their nearest other object
 13. BackgroundFloodFillSolver   — flood-fill background from any seed color
 14. CopyRowColPatternToZerosSolver — copy non-zero row/col to zero rows/cols
 15. ShiftRowsColsByOffsetSolver — learn per-row/col shift offset and apply
"""

from __future__ import annotations
import copy
from collections import Counter, deque
from typing import Optional, List, Tuple

from ..core.interfaces import BaseSolver, SolverResult, TaskContext, Prediction
from ..core.grid_codec import background_color
from .registry import register_solver

Grid = List[List[int]]


# ── helpers ──────────────────────────────────────────────────────────────────

def _sh(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)

def _clone(g: Grid) -> Grid:
    return [r[:] for r in g]

def _bg(g: Grid) -> int:
    flat = [v for r in g for v in r]
    return Counter(flat).most_common(1)[0][0] if flat else 0

def _result(ctx: TaskContext, grids, solver_name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=solver_name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

def _bfs4(g: Grid, sr: int, sc: int, match_val: int) -> List[Tuple[int,int]]:
    H, W = _sh(g)
    visited = {(sr, sc)}
    q = deque([(sr, sc)])
    cells = []
    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and g[nr][nc]==match_val:
                visited.add((nr,nc)); q.append((nr,nc))
    return cells

def _connected_components(g: Grid, bg: int) -> List[List[Tuple[int,int]]]:
    H, W = _sh(g)
    visited = set()
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg and (r, c) not in visited:
                cells = _bfs4(g, r, c, g[r][c])
                visited.update(cells)
                comps.append(cells)
    return comps


# ── 1. VoronoiColorFillSolver ─────────────────────────────────────────────

@register_solver
class VoronoiColorFillSolver(BaseSolver):
    """
    Fill background cells with the color of their nearest non-background cell
    (Manhattan distance). Consistent across all training pairs → apply to test.
    """

    @property
    def name(self) -> str: return "VoronoiColorFillSolver"
    @property
    def priority(self) -> int: return 62

    def _apply(self, grid: Grid, bg: int) -> Grid:
        H, W = _sh(grid)
        # BFS multi-source from all non-bg cells
        dist = [[-1]*W for _ in range(H)]
        color_map = [[bg]*W for _ in range(H)]
        q = deque()
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg:
                    dist[r][c] = 0
                    color_map[r][c] = grid[r][c]
                    q.append((r, c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and dist[nr][nc]==-1:
                    dist[nr][nc] = dist[r][c]+1
                    color_map[nr][nc] = color_map[r][c]
                    q.append((nr,nc))
        out = _clone(grid)
        for r in range(H):
            for c in range(W):
                if grid[r][c] == bg:
                    out[r][c] = color_map[r][c]
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        # Detect consistent bg across training
        bgs = set()
        for p in ctx.train_pairs:
            inp, out = p["input"], p["output"]
            if not inp or not out: return None
            ih, iw = _sh(inp); oh, ow = _sh(out)
            if ih != oh or iw != ow: return None
            bgs.add(_bg(inp))
        if len(bgs) != 1: return None
        bg = bgs.pop()

        # Verify: applying voronoi to each training input matches output
        for p in ctx.train_pairs:
            pred = self._apply(p["input"], bg)
            if pred != p["output"]: return None

        preds = [self._apply(inp, bg) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 2. ConnectEndpointsSolver ─────────────────────────────────────────────

@register_solver
class ConnectEndpointsSolver(BaseSolver):
    """
    For each training pair:
      - find pairs of same-color non-bg cells that are collinear (same row or col)
      - fill the gap between them with a specific fill color
    Learn the (source_color, fill_color) mapping and apply consistently.
    """

    @property
    def name(self) -> str: return "ConnectEndpointsSolver"
    @property
    def priority(self) -> int: return 63

    def _connect_pairs(self, inp: Grid, bg: int, src_color: int, fill_color: int) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        # Find all cells of src_color
        src_cells = [(r,c) for r in range(H) for c in range(W) if inp[r][c]==src_color]
        # Group by row
        by_row = {}
        for r, c in src_cells:
            by_row.setdefault(r, []).append(c)
        for r, cols in by_row.items():
            cols.sort()
            for i in range(len(cols)-1):
                c1, c2 = cols[i], cols[i+1]
                # Fill between c1 and c2 if gap is all bg
                gap = all(inp[r][c] == bg for c in range(c1+1, c2))
                if gap:
                    for c in range(c1+1, c2):
                        out[r][c] = fill_color
        # Group by col
        by_col = {}
        for r, c in src_cells:
            by_col.setdefault(c, []).append(r)
        for c, rows in by_col.items():
            rows.sort()
            for i in range(len(rows)-1):
                r1, r2 = rows[i], rows[i+1]
                gap = all(inp[r][c] == bg for r in range(r1+1, r2))
                if gap:
                    for r in range(r1+1, r2):
                        out[r][c] = fill_color
        return out

    def _infer_rule(self, train_pairs):
        """Return (src_color, fill_color) or None."""
        bg_set = set(_bg(p["input"]) for p in train_pairs)
        if len(bg_set) != 1: return None
        bg = bg_set.pop()
        
        # Find all non-bg colors in inputs
        for p in train_pairs:
            inp, out = p["input"], p["output"]
            if not inp or not out: return None
            if _sh(inp) != _sh(out): return None
            H, W = _sh(inp)
            # Find bg cells in inp that became non-bg in out
            filled = [(r,c,out[r][c]) for r in range(H) for c in range(W)
                      if inp[r][c]==bg and out[r][c]!=bg]
            if not filled: return None
            fill_colors = set(oc for _,_,oc in filled)
            if len(fill_colors) != 1: return None
            fill_color = fill_colors.pop()
            
            # What source_color pairs are being connected?
            # Look for a non-bg, non-fill_color that has cells on same row/col
            inp_colors = set(inp[r][c] for r in range(H) for c in range(W) if inp[r][c]!=bg)
            inp_colors.discard(fill_color)
            if not inp_colors: return None
            
            for src_color in inp_colors:
                candidate = self._connect_pairs(inp, bg, src_color, fill_color)
                if candidate == out:
                    return (src_color, fill_color, bg)
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        rule = self._infer_rule(ctx.train_pairs)
        if rule is None: return None
        src_color, fill_color, bg = rule

        # Verify all pairs
        for p in ctx.train_pairs:
            pred = self._connect_pairs(p["input"], bg, src_color, fill_color)
            if pred != p["output"]: return None

        preds = [self._connect_pairs(inp, bg, src_color, fill_color) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 3. NearestSolidLineRecolorSolver ─────────────────────────────────────────

@register_solver
class NearestSolidLineRecolorSolver(BaseSolver):
    """
    Full rows (or cols) with uniform non-bg color act as "attractors."
    Non-bg, non-solid cells get recolored to the color of the nearest solid row/col.
    """

    @property
    def name(self) -> str: return "NearestSolidLineRecolorSolver"
    @property
    def priority(self) -> int: return 64

    def _solid_rows(self, grid: Grid) -> List[Tuple[int, int]]:
        """Returns list of (row_idx, color)."""
        H, W = _sh(grid)
        result = []
        for r in range(H):
            colors = set(grid[r])
            if len(colors) == 1 and list(colors)[0] != 0:
                result.append((r, list(colors)[0]))
        return result

    def _solid_cols(self, grid: Grid) -> List[Tuple[int, int]]:
        H, W = _sh(grid)
        result = []
        for c in range(W):
            colors = set(grid[r][c] for r in range(H))
            if len(colors) == 1 and list(colors)[0] != 0:
                result.append((c, list(colors)[0]))
        return result

    def _apply_row(self, grid: Grid, bg: int, solid_rows: List[Tuple[int,int]]) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        solid_row_set = {r for r, _ in solid_rows}
        for r in range(H):
            if r in solid_row_set: continue
            # Find nearest solid row
            best_dist, best_color = H+1, bg
            for sr, sc in solid_rows:
                d = abs(r - sr)
                if d < best_dist: best_dist, best_color = d, sc
            # Recolor non-bg cells in this row
            for c in range(W):
                if grid[r][c] != bg:
                    out[r][c] = best_color
        return out

    def _apply_col(self, grid: Grid, bg: int, solid_cols: List[Tuple[int,int]]) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        solid_col_set = {c for c, _ in solid_cols}
        for c in range(W):
            if c in solid_col_set: continue
            best_dist, best_color = W+1, bg
            for sc, col_c in solid_cols:
                d = abs(c - sc)
                if d < best_dist: best_dist, best_color = d, col_c
            for r in range(H):
                if grid[r][c] != bg:
                    out[r][c] = best_color
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            inp, out = p["input"], p["output"]
            if not inp or not out: return None
            ih, iw = _sh(inp); oh, ow = _sh(out)
            if ih != oh or iw != ow: return None

        # Try row-based
        bg = _bg(ctx.train_pairs[0]["input"])
        all_ok_row = True
        for p in ctx.train_pairs:
            solid = self._solid_rows(p["input"])
            if not solid: all_ok_row = False; break
            pred = self._apply_row(p["input"], bg, solid)
            if pred != p["output"]: all_ok_row = False; break
        if all_ok_row:
            preds = [self._apply_row(inp, bg, self._solid_rows(inp)) for inp in ctx.test_inputs]
            return _result(ctx, preds, self.name)

        # Try col-based
        all_ok_col = True
        for p in ctx.train_pairs:
            solid = self._solid_cols(p["input"])
            if not solid: all_ok_col = False; break
            pred = self._apply_col(p["input"], bg, solid)
            if pred != p["output"]: all_ok_col = False; break
        if all_ok_col:
            preds = [self._apply_col(inp, bg, self._solid_cols(inp)) for inp in ctx.test_inputs]
            return _result(ctx, preds, self.name)

        return None


# ── 4. ExtendRayFromMarkerSolver ─────────────────────────────────────────────

@register_solver
class ExtendRayFromMarkerSolver(BaseSolver):
    """
    Find a uniquely-colored "marker" cell. Extend a ray from it in 4/8 directions
    (chosen by learning from training) until hitting edge or another object.
    Fill the path with a learned fill color.
    """

    @property
    def name(self) -> str: return "ExtendRayFromMarkerSolver"
    @property
    def priority(self) -> int: return 65

    def _unique_color_cell(self, grid: Grid, bg: int) -> Optional[Tuple[int, int, int]]:
        """Find a cell whose color appears exactly once in grid, non-bg."""
        H, W = _sh(grid)
        color_count = Counter(grid[r][c] for r in range(H) for c in range(W))
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg and color_count[grid[r][c]] == 1:
                    return (r, c, grid[r][c])
        return None

    def _apply(self, grid: Grid, bg: int, direction: Tuple[int,int],
               fill_color: int, include_marker: bool) -> Grid:
        H, W = _sh(grid)
        marker = self._unique_color_cell(grid, bg)
        if marker is None: return None
        mr, mc, _ = marker
        out = _clone(grid)
        dr, dc = direction
        r, c = mr + dr, mc + dc
        while 0 <= r < H and 0 <= c < W:
            if grid[r][c] != bg: break
            out[r][c] = fill_color
            r += dr; c += dc
        return out

    def _infer_rule(self, train_pairs):
        """Return (direction, fill_color, include_marker) or None."""
        bg_set = set(_bg(p["input"]) for p in train_pairs)
        if len(bg_set) != 1: return None
        bg = bg_set.pop()

        DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        for direction in DIRECTIONS:
            for fill_color in range(10):
                if fill_color == bg: continue
                all_ok = True
                for p in train_pairs:
                    candidate = self._apply(p["input"], bg, direction, fill_color, False)
                    if candidate is None or candidate != p["output"]:
                        all_ok = False; break
                if all_ok: return (direction, fill_color, bg)
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            inp, out = p["input"], p["output"]
            if not inp or not out: return None
            if _sh(inp) != _sh(out): return None
        rule = self._infer_rule(ctx.train_pairs)
        if rule is None: return None
        direction, fill_color, bg = rule
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp, bg, direction, fill_color, False)
            if g is None: return None
            preds.append(g)
        return _result(ctx, preds, self.name)


# ── 5. EnclosedFloodFillSolver2 ──────────────────────────────────────────────

@register_solver
class EnclosedFloodFillSolver2(BaseSolver):
    """
    Find a marker color inside a closed boundary of another color.
    Flood-fill outward from the marker within the boundary.
    If the boundary has a gap, fill leaks through.
    """

    @property
    def name(self) -> str: return "EnclosedFloodFillSolver2"
    @property
    def priority(self) -> int: return 66

    def _apply(self, grid: Grid, boundary_color: int,
               fill_color: int, bg: int) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        # Find all bg cells that are NOT enclosed (flood from edges)
        outer = set()
        q = deque()
        for r in range(H):
            for c in [0, W-1]:
                if grid[r][c] == bg and (r,c) not in outer:
                    outer.add((r,c)); q.append((r,c))
        for c in range(W):
            for r in [0, H-1]:
                if grid[r][c] == bg and (r,c) not in outer:
                    outer.add((r,c)); q.append((r,c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and (nr,nc) not in outer and grid[nr][nc]==bg:
                    outer.add((nr,nc)); q.append((nr,nc))
        # Fill enclosed bg cells
        for r in range(H):
            for c in range(W):
                if grid[r][c] == bg and (r,c) not in outer:
                    out[r][c] = fill_color
        return out

    def _infer_rule(self, train_pairs):
        bg_set = set(_bg(p["input"]) for p in train_pairs)
        if len(bg_set) != 1: return None
        bg = bg_set.pop()

        for fill_color in range(10):
            if fill_color == bg: continue
            for boundary_color in range(10):
                if boundary_color == bg or boundary_color == fill_color: continue
                all_ok = True
                for p in train_pairs:
                    candidate = self._apply(p["input"], boundary_color, fill_color, bg)
                    if candidate != p["output"]: all_ok = False; break
                if all_ok: return (boundary_color, fill_color, bg)
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None
        rule = self._infer_rule(ctx.train_pairs)
        if rule is None: return None
        bc, fill_color, bg = rule
        preds = [self._apply(inp, bc, fill_color, bg) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 6. RowPatternProjectSolver ────────────────────────────────────────────────

@register_solver
class RowPatternProjectSolver(BaseSolver):
    """
    A reference row (the first non-bg non-zero row) defines a column pattern.
    Rows that have a specific marker at a specific column get that pattern applied
    (with the marker color replacing the pattern color).
    """

    @property
    def name(self) -> str: return "RowPatternProjectSolver"
    @property
    def priority(self) -> int: return 67

    def _find_reference_and_markers(self, inp: Grid, out: Grid, bg: int):
        H, W = _sh(inp)
        # Find rows where inp==out (unchanged rows) and rows that differ
        changed_rows = [r for r in range(H) if inp[r] != out[r]]
        if not changed_rows: return None

        # Check: for each changed row, which bg cells became non-bg in output?
        for ref_row in range(H):
            if any(inp[ref_row][c] != bg for c in range(W)):
                # ref_row is the pattern source
                ref_pattern = inp[ref_row]  # non-bg positions define the template
                ref_nonbg = [(c, ref_pattern[c]) for c in range(W) if ref_pattern[c] != bg]
                if not ref_nonbg: continue
                ref_cols = [c for c, _ in ref_nonbg]

                # For each changed row, verify the change matches ref_cols
                all_ok = True
                marker_col = None; marker_role = None
                for cr in changed_rows:
                    # What changed?
                    row_changes = [(c, inp[cr][c], out[cr][c]) for c in range(W) if inp[cr][c] != out[cr][c]]
                    if not row_changes: continue
                    # All changes should be: bg → some_color at ref_cols
                    change_cols = [c for c,_,_ in row_changes]
                    if sorted(change_cols) != sorted(ref_cols): all_ok = False; break
                    fill_color = out[cr][change_cols[0]]
                    if not all(out[cr][c] == fill_color for c in change_cols): all_ok = False; break
                    # The fill color should exist in that row's non-bg cells
                    row_nonbg = [(c, inp[cr][c]) for c in range(W) if inp[cr][c] != bg and c not in change_cols]
                    fill_cols_in_row = [c for c,v in row_nonbg if v == fill_color]
                    if not fill_cols_in_row: all_ok = False; break

                if all_ok:
                    return (ref_row, ref_cols)
        return None

    def _apply(self, inp: Grid, bg: int, ref_cols: list) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        # Find all non-ref rows that have exactly one non-bg run not at ref_cols
        for r in range(H):
            nonbg = [(c, inp[r][c]) for c in range(W) if inp[r][c] != bg]
            nonbg_at_ref = [c for c,v in nonbg if c in ref_cols]
            nonbg_not_ref = [(c,v) for c,v in nonbg if c not in ref_cols]
            if nonbg_not_ref and not nonbg_at_ref:
                # This row has a marker outside ref_cols → fill ref_cols with that color
                colors = set(v for c,v in nonbg_not_ref)
                if len(colors) == 1:
                    fill_color = list(colors)[0]
                    for c in ref_cols:
                        if inp[r][c] == bg:
                            out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None

        bg = _bg(ctx.train_pairs[0]["input"])
        # Infer ref_cols from first pair
        result = self._find_reference_and_markers(
            ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"], bg)
        if result is None: return None
        ref_row, ref_cols = result

        # Verify all pairs
        for p in ctx.train_pairs:
            pred = self._apply(p["input"], bg, ref_cols)
            if pred != p["output"]: return None

        preds = [self._apply(inp, bg, ref_cols) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 7. MarkerColorsNearestCellSolver ─────────────────────────────────────────

@register_solver
class MarkerColorsNearestCellSolver(BaseSolver):
    """
    Small unique-colored marker cells (typically at edges or far from the main object)
    recolor the nearest cell of a "target color" in their row or column.
    """

    @property
    def name(self) -> str: return "MarkerColorsNearestCellSolver"
    @property
    def priority(self) -> int: return 68

    def _apply(self, inp: Grid, bg: int, target_color: int) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        color_count = Counter(inp[r][c] for r in range(H) for c in range(W))
        # Markers: non-bg, non-target, count==1
        markers = [(r, c, inp[r][c]) for r in range(H) for c in range(W)
                   if inp[r][c] != bg and inp[r][c] != target_color
                   and color_count[inp[r][c]] == 1]
        if not markers: return None
        used = set()
        for mr, mc, mcolor in markers:
            # Find nearest target cell in same row or col
            best_dist, best_pos = H+W, None
            # same row
            for c in range(W):
                if inp[mr][c] == target_color:
                    d = abs(c - mc)
                    if d < best_dist: best_dist = d; best_pos = (mr, c)
            # same col
            for r in range(H):
                if inp[r][mc] == target_color:
                    d = abs(r - mr)
                    if d < best_dist: best_dist = d; best_pos = (r, mc)
            if best_pos and best_pos not in used:
                out[best_pos[0]][best_pos[1]] = mcolor
                used.add(best_pos)
        return out

    def _infer_target(self, train_pairs, bg):
        for target_color in range(10):
            if target_color == bg: continue
            all_ok = True
            for p in train_pairs:
                candidate = self._apply(p["input"], bg, target_color)
                if candidate is None or candidate != p["output"]:
                    all_ok = False; break
            if all_ok: return target_color
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None
        bg = _bg(ctx.train_pairs[0]["input"])
        target = self._infer_target(ctx.train_pairs, bg)
        if target is None: return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp, bg, target)
            if g is None: return None
            preds.append(g)
        return _result(ctx, preds, self.name)


# ── 8. ColumnPatternProjectSolver ────────────────────────────────────────────

@register_solver
class ColumnPatternProjectSolver(BaseSolver):
    """
    Like RowPatternProjectSolver but for columns:
    A reference column defines a row-position pattern.
    Columns with a single non-bg marker get that pattern projected onto them.
    """

    @property
    def name(self) -> str: return "ColumnPatternProjectSolver"
    @property
    def priority(self) -> int: return 69

    def _apply(self, inp: Grid, bg: int, ref_rows: list) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        for c in range(W):
            nonbg = [(r, inp[r][c]) for r in range(H) if inp[r][c] != bg]
            at_ref = [r for r,v in nonbg if r in ref_rows]
            not_ref = [(r,v) for r,v in nonbg if r not in ref_rows]
            if not_ref and not at_ref:
                colors = set(v for r,v in not_ref)
                if len(colors) == 1:
                    fill_color = list(colors)[0]
                    for r in ref_rows:
                        if inp[r][c] == bg:
                            out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None

        bg = _bg(ctx.train_pairs[0]["input"])
        # Infer ref_rows from first pair
        inp0, out0 = ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"]
        H, W = _sh(inp0)

        # Find ref_rows: rows that appear in fill positions
        changed_cols = [c for c in range(W) if any(inp0[r][c] != out0[r][c] for r in range(H))]
        if not changed_cols: return None
        ref_rows_set = set()
        for c in changed_cols:
            for r in range(H):
                if inp0[r][c] == bg and out0[r][c] != bg:
                    ref_rows_set.add(r)
        if not ref_rows_set: return None
        ref_rows = sorted(ref_rows_set)

        for p in ctx.train_pairs:
            pred = self._apply(p["input"], bg, ref_rows)
            if pred != p["output"]: return None

        preds = [self._apply(inp, bg, ref_rows) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 9. ConnectSameColorInColumnSolver ─────────────────────────────────────────

@register_solver
class ConnectSameColorInColumnSolver(BaseSolver):
    """
    For each column: if there are two same-color endpoints, fill the gap between
    them with a learned fill color. Works both row-wise and col-wise.
    Same as ConnectEndpointsSolver but also handles partially-noisy rows.
    """

    @property
    def name(self) -> str: return "ConnectSameColorInColumnSolver"
    @property
    def priority(self) -> int: return 70

    def _connect(self, inp: Grid, bg: int, src_color: int, fill_color: int,
                 axis: str) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        if axis == 'col':
            for c in range(W):
                rows_with_src = [r for r in range(H) if inp[r][c] == src_color]
                if len(rows_with_src) >= 2:
                    r1, r2 = rows_with_src[0], rows_with_src[-1]
                    for r in range(r1+1, r2):
                        if inp[r][c] == bg:
                            out[r][c] = fill_color
        else:  # row
            for r in range(H):
                cols_with_src = [c for c in range(W) if inp[r][c] == src_color]
                if len(cols_with_src) >= 2:
                    c1, c2 = cols_with_src[0], cols_with_src[-1]
                    for c in range(c1+1, c2):
                        if inp[r][c] == bg:
                            out[r][c] = fill_color
        return out

    def _infer(self, train_pairs):
        bg_set = set(_bg(p["input"]) for p in train_pairs)
        if len(bg_set) != 1: return None
        bg = bg_set.pop()
        for src_color in range(10):
            if src_color == bg: continue
            for fill_color in range(10):
                if fill_color == bg or fill_color == src_color: continue
                for axis in ['col', 'row']:
                    ok = True
                    for p in train_pairs:
                        c = self._connect(p["input"], bg, src_color, fill_color, axis)
                        if c != p["output"]: ok = False; break
                    if ok: return (src_color, fill_color, axis, bg)
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None
        rule = self._infer(ctx.train_pairs)
        if rule is None: return None
        src_color, fill_color, axis, bg = rule
        preds = [self._connect(inp, bg, src_color, fill_color, axis) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 10. TwoColorBandCrossSolver ───────────────────────────────────────────────

@register_solver
class TwoColorBandCrossSolver(BaseSolver):
    """
    Horizontal and vertical color bands exist.
    At intersection cells, assign a specific learned color.
    """

    @property
    def name(self) -> str: return "TwoColorBandCrossSolver"
    @property
    def priority(self) -> int: return 71

    def _apply(self, inp: Grid, bg: int, h_color: int, v_color: int, cross_color: int) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        h_rows = set(r for r in range(H) if any(inp[r][c] == h_color for c in range(W)))
        v_cols = set(c for c in range(W) if any(inp[r][c] == v_color for r in range(H)))
        for r in h_rows:
            for c in v_cols:
                out[r][c] = cross_color
        return out

    def _infer(self, train_pairs):
        bg_set = set(_bg(p["input"]) for p in train_pairs)
        if len(bg_set) != 1: return None
        bg = bg_set.pop()
        colors = list(range(10))
        for h_color in colors:
            if h_color == bg: continue
            for v_color in colors:
                if v_color == bg or v_color == h_color: continue
                for cross_color in colors:
                    if cross_color == bg: continue
                    ok = True
                    for p in train_pairs:
                        c = self._apply(p["input"], bg, h_color, v_color, cross_color)
                        if c != p["output"]: ok = False; break
                    if ok: return (h_color, v_color, cross_color, bg)
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None
        rule = self._infer(ctx.train_pairs)
        if rule is None: return None
        h_color, v_color, cross_color, bg = rule
        preds = [self._apply(inp, bg, h_color, v_color, cross_color) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 11. ReflectAboutCenterSolver ─────────────────────────────────────────────

@register_solver
class ReflectAboutCenterSolver(BaseSolver):
    """
    The output is the input reflected (h-flip, v-flip, or both) about the center.
    Handles cases where only part of the grid changes.
    """

    @property
    def name(self) -> str: return "ReflectAboutCenterSolver"
    @property
    def priority(self) -> int: return 72

    def _apply(self, inp: Grid, mode: str) -> Grid:
        H, W = _sh(inp)
        out = _clone(inp)
        if mode == 'hflip':
            out = [r[::-1] for r in inp]
        elif mode == 'vflip':
            out = inp[::-1]
        elif mode == 'both':
            out = [r[::-1] for r in inp[::-1]]
        elif mode == 'rot90':
            out = [[inp[H-1-c][r] for c in range(H)] for r in range(W)]
        elif mode == 'rot270':
            out = [[inp[c][W-1-r] for c in range(H)] for r in range(W)]
        return out

    def _check_mode(self, mode: str, train_pairs) -> bool:
        for p in train_pairs:
            if not p["input"] or not p["output"]: return False
            pred = self._apply(p["input"], mode)
            if pred != p["output"]: return False
        return True

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for mode in ['hflip', 'vflip', 'both', 'rot90', 'rot270']:
            if self._check_mode(mode, ctx.train_pairs):
                preds = [self._apply(inp, mode) for inp in ctx.test_inputs]
                return _result(ctx, preds, self.name)
        return None


# ── 12. ObjectColorByNeighborSolver ──────────────────────────────────────────

@register_solver
class ObjectColorByNeighborSolver(BaseSolver):
    """
    Each non-bg connected component gets recolored to the color of its
    nearest OTHER connected component.
    """

    @property
    def name(self) -> str: return "ObjectColorByNeighborSolver"
    @property
    def priority(self) -> int: return 73

    def _apply(self, inp: Grid, bg: int) -> Grid:
        comps = _connected_components(inp, bg)
        if len(comps) < 2: return None
        # Get color of each comp
        comp_colors = [inp[comps[i][0][0]][comps[i][0][1]] for i in range(len(comps))]
        # For each comp, find nearest OTHER comp by min cell distance
        out = _clone(inp)
        for i, comp in enumerate(comps):
            min_dist = 10**9; nearest_color = comp_colors[i]
            for j, other_comp in enumerate(comps):
                if i == j: continue
                for r1, c1 in comp:
                    for r2, c2 in other_comp:
                        d = abs(r1-r2) + abs(c1-c2)
                        if d < min_dist: min_dist = d; nearest_color = comp_colors[j]
            for r, c in comp:
                out[r][c] = nearest_color
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None
        bg = _bg(ctx.train_pairs[0]["input"])
        for p in ctx.train_pairs:
            pred = self._apply(p["input"], bg)
            if pred is None or pred != p["output"]: return None
        preds = []
        for inp in ctx.test_inputs:
            g = self._apply(inp, bg)
            if g is None: return None
            preds.append(g)
        return _result(ctx, preds, self.name)


# ── 13. CopyNonzeroRowToZeroRowsSolver ────────────────────────────────────────

@register_solver
class CopyNonzeroRowToZeroRowsSolver(BaseSolver):
    """
    Find a single non-all-zero row (or col) and copy it to every all-zero row (or col).
    """

    @property
    def name(self) -> str: return "CopyNonzeroRowToZeroRowsSolver"
    @property
    def priority(self) -> int: return 74

    def _apply_row(self, inp: Grid) -> Optional[Grid]:
        H, W = _sh(inp)
        nonzero_rows = [r for r in range(H) if any(v != 0 for v in inp[r])]
        zero_rows = [r for r in range(H) if all(v == 0 for v in inp[r])]
        if not nonzero_rows or not zero_rows: return None
        if len(nonzero_rows) != 1: return None
        template = inp[nonzero_rows[0]]
        out = _clone(inp)
        for r in zero_rows:
            out[r] = template[:]
        return out

    def _apply_col(self, inp: Grid) -> Optional[Grid]:
        H, W = _sh(inp)
        nonzero_cols = [c for c in range(W) if any(inp[r][c] != 0 for r in range(H))]
        zero_cols = [c for c in range(W) if all(inp[r][c] == 0 for r in range(H))]
        if not nonzero_cols or not zero_cols: return None
        if len(nonzero_cols) != 1: return None
        out = _clone(inp)
        for c in zero_cols:
            for r in range(H):
                out[r][c] = inp[r][nonzero_cols[0]]
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for axis, fn in [('row', self._apply_row), ('col', self._apply_col)]:
            ok = True
            for p in ctx.train_pairs:
                if not p["input"] or not p["output"]: ok=False; break
                if _sh(p["input"]) != _sh(p["output"]): ok=False; break
                pred = fn(p["input"])
                if pred is None or pred != p["output"]: ok=False; break
            if ok:
                preds = []
                for inp in ctx.test_inputs:
                    g = fn(inp)
                    if g is None: return None
                    preds.append(g)
                return _result(ctx, preds, self.name)
        return None


# ── 14. ShiftRowsColsByOffsetSolver ───────────────────────────────────────────

@register_solver
class ShiftRowsColsByOffsetSolver(BaseSolver):
    """
    Each row (or col) is cyclically shifted by a learned per-row offset.
    Infer offset from first training pair, verify on rest.
    """

    @property
    def name(self) -> str: return "ShiftRowsColsByOffsetSolver"
    @property
    def priority(self) -> int: return 75

    def _row_shift(self, inp: Grid, offsets: list) -> Grid:
        H, W = _sh(inp)
        if len(offsets) != H: return inp  # size mismatch — return unchanged
        out = []
        for r in range(H):
            shift = offsets[r] % W if W else 0
            out.append(inp[r][-shift:] + inp[r][:-shift] if shift else inp[r][:])
        return out

    def _col_shift(self, inp: Grid, offsets: list) -> Grid:
        H, W = _sh(inp)
        if len(offsets) != W: return inp
        out = _clone(inp)
        for c in range(W):
            col = [inp[r][c] for r in range(H)]
            shift = offsets[c] % H if H else 0
            shifted = col[-shift:] + col[:-shift] if shift else col[:]
            for r in range(H): out[r][c] = shifted[r]
        return out

    def _infer_row_offsets(self, inp: Grid, out: Grid) -> Optional[list]:
        H, W = _sh(inp)
        offsets = []
        for r in range(H):
            found = False
            for shift in range(W):
                shifted = inp[r][-shift:] + inp[r][:-shift] if shift else inp[r][:]
                if shifted == out[r]:
                    offsets.append(shift); found = True; break
            if not found: return None
        return offsets

    def _infer_col_offsets(self, inp: Grid, out: Grid) -> Optional[list]:
        H, W = _sh(inp)
        offsets = []
        for c in range(W):
            col_in = [inp[r][c] for r in range(H)]
            col_out = [out[r][c] for r in range(H)]
            found = False
            for shift in range(H):
                shifted = col_in[-shift:] + col_in[:-shift] if shift else col_in[:]
                if shifted == col_out:
                    offsets.append(shift); found = True; break
            if not found: return None
        return offsets

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        for p in ctx.train_pairs:
            if not p["input"] or not p["output"]: return None
            if _sh(p["input"]) != _sh(p["output"]): return None

        # Try row shifts
        offsets = self._infer_row_offsets(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if offsets:
            ok = all(self._row_shift(p["input"], offsets) == p["output"]
                     for p in ctx.train_pairs)
            if ok:
                preds = [self._row_shift(inp, offsets) for inp in ctx.test_inputs]
                return _result(ctx, preds, self.name)

        # Try col shifts
        offsets = self._infer_col_offsets(ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"])
        if offsets:
            ok = all(self._col_shift(p["input"], offsets) == p["output"]
                     for p in ctx.train_pairs)
            if ok:
                preds = [self._col_shift(inp, offsets) for inp in ctx.test_inputs]
                return _result(ctx, preds, self.name)

        return None


# ── 15. ColorRemapFromPaletteSolver ───────────────────────────────────────────

@register_solver
class ColorRemapFromPaletteSolver(BaseSolver):
    """
    Learn a complete color-to-color mapping from training pairs.
    Each input color maps to an output color consistently.
    Handles cases where multiple input colors map to the same output.
    """

    @property
    def name(self) -> str: return "ColorRemapFromPaletteSolver"
    @property
    def priority(self) -> int: return 76

    def _infer_map(self, train_pairs):
        cmap = {}
        for p in train_pairs:
            inp, out = p["input"], p["output"]
            if not inp or not out: return None
            ih, iw = _sh(inp); oh, ow = _sh(out)
            if ih != oh or iw != ow: return None
            for r in range(ih):
                for c in range(iw):
                    iv, ov = inp[r][c], out[r][c]
                    if iv not in cmap: cmap[iv] = ov
                    elif cmap[iv] != ov: return None
        return cmap

    def _apply(self, grid: Grid, cmap: dict) -> Grid:
        return [[cmap.get(v, v) for v in row] for row in grid]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs: return None
        cmap = self._infer_map(ctx.train_pairs)
        if cmap is None: return None
        # Must have at least one actual change
        if all(k == v for k, v in cmap.items()): return None
        # Verify
        for p in ctx.train_pairs:
            if self._apply(p["input"], cmap) != p["output"]: return None
        preds = []
        for inp in ctx.test_inputs:
            test_colors = {v for row in inp for v in row}
            if not test_colors.issubset(cmap.keys()):
                return None
            preds.append(self._apply(inp, cmap))
        return _result(ctx, preds, self.name)
