"""
advanced_solvers12.py — 5 new solvers derived from deep pattern analysis
  1. HSpanFillSolver         (task 22168020)
  2. FillRectFrameSolver     (task 44d8ac46)
  3. DiagonalExtensionSolver (task 6e19193c)
  4. ReflectAroundPivotSolver(task 4938f0c2)
  5. LPathConnectSolver      (task 2dd70a9a)
"""
from collections import deque

from src.core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext

from .registry import register_solver

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _copy_grid(g):
    return [list(row) for row in g]

def _grid_eq(a, b):
    return a == b or all(a[r][c] == b[r][c]
                         for r in range(len(a)) for c in range(len(a[r])))

def _flood_component(grid, sr, sc):
    """BFS flood fill — returns set of (r,c) with same color as (sr,sc)."""
    H, W = len(grid), len(grid[0])
    color = grid[sr][sc]
    visited = set()
    q = deque([(sr, sc)])
    visited.add((sr, sc))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and grid[nr][nc] == color:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited

def _connected_components(grid, color):
    """All 4-connected CCs of `color`; returns list of frozensets of (r,c)."""
    H, W = len(grid), len(grid[0])
    seen = set()
    ccs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == color and (r,c) not in seen:
                cc = _flood_component(grid, r, c)
                seen |= cc
                ccs.append(frozenset(cc))
    return ccs


# ---------------------------------------------------------------------------
# 1. HSpanFillSolver
# ---------------------------------------------------------------------------

@register_solver
class HSpanFillSolver(BaseSolver):
    """
    For every row, for each non-zero color C present in that row, find the
    leftmost and rightmost column that contain C. Fill every zero between
    them with C. Multiple colors in one row each span independently.
    """
    name = "HSpanFillSolver"

    def _apply(self, grid):
        g = _copy_grid(grid)
        H, W = len(g), len(g[0])
        for r in range(H):
            colors = {}
            for c in range(W):
                v = g[r][c]
                if v != 0:
                    if v not in colors:
                        colors[v] = [c, c]
                    else:
                        colors[v][0] = min(colors[v][0], c)
                        colors[v][1] = max(colors[v][1], c)
            for color, (lo, hi) in colors.items():
                for c in range(lo, hi+1):
                    if g[r][c] == 0:
                        g[r][c] = color
        return g

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # validate on every training pair
        for p in ctx.train_pairs:
            inp = p['input']
            expected = p['output']
            pred = self._apply(inp)
            if pred != expected:
                return None
        # apply to test
        preds = []
        for t_in in ctx.test_inputs:
            g = self._apply(t_in)
            preds.append(Prediction(attempt_1=g, attempt_2=t_in, solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 2. FillRectFrameSolver
# ---------------------------------------------------------------------------

@register_solver
class FillRectFrameSolver(BaseSolver):
    """
    Finds rectangular frame structures of border_color whose interior is all
    zeros AND whose interior dimensions are SQUARE (height == width), then
    fills those interiors with fill_color.
    Both colors are learned from training pairs.
    """
    name = "FillRectFrameSolver"

    def _find_frames(self, grid, border_color):
        """Return list of (r1,c1,r2,c2) closed rectangular frames with square interior."""
        H, W = len(grid), len(grid[0])
        frames = []
        for r1 in range(H):
            for r2 in range(r1+2, H):          # need at least 1-cell interior
                for c1 in range(W):
                    for c2 in range(c1+2, W):
                        ih = r2 - r1 - 1
                        iw = c2 - c1 - 1
                        # FIX: only square interiors
                        if ih != iw or ih == 0:
                            continue
                        # check top/bottom rows fully border_color
                        if not all(grid[r1][c] == border_color for c in range(c1, c2+1)):
                            continue
                        if not all(grid[r2][c] == border_color for c in range(c1, c2+1)):
                            continue
                        # check left/right cols fully border_color
                        if not all(grid[r][c1] == border_color for r in range(r1, r2+1)):
                            continue
                        if not all(grid[r][c2] == border_color for r in range(r1, r2+1)):
                            continue
                        # interior must be all zeros
                        interior_ok = all(grid[r][c] == 0
                                          for r in range(r1+1, r2)
                                          for c in range(c1+1, c2))
                        if interior_ok:
                            frames.append((r1, c1, r2, c2))
        return frames

    def _apply(self, grid, border_color, fill_color):
        g = _copy_grid(grid)
        for (r1, c1, r2, c2) in self._find_frames(grid, border_color):
            for r in range(r1+1, r2):
                for c in range(c1+1, c2):
                    g[r][c] = fill_color
        return g

    def _learn_colors(self, ctx):
        """
        Learn (border_color, fill_color) from training pairs.
        border_color = a color present in inputs that is unchanged.
        fill_color   = a color that appears in outputs but not in inputs (or
                       greatly increases).
        """
        border_candidates = None
        fill_candidates = None
        for p in ctx.train_pairs:
            inp, out = p['input'], p['output']
            H_in, W_in = len(inp), len(inp[0]) if inp else 0
            H_out, W_out = len(out), len(out[0]) if out else 0
            if H_in != H_out or W_in != W_out:
                return None, None
            H, W = H_in, W_in
            changed_from = set()
            changed_to = set()
            for r in range(H):
                for c in range(W):
                    if inp[r][c] != out[r][c]:
                        changed_from.add(inp[r][c])
                        changed_to.add(out[r][c])
            inp_colors = {inp[r][c] for r in range(H) for c in range(W)} - {0}
            bc = inp_colors - changed_from
            fc = changed_to - {0}
            if border_candidates is None:
                border_candidates = bc
                fill_candidates = fc if fc else None
            else:
                border_candidates &= bc
                if fc:
                    if fill_candidates is None:
                        fill_candidates = fc
                    else:
                        fill_candidates &= fc
        if not border_candidates or not fill_candidates:
            return None, None
        return min(border_candidates), min(fill_candidates)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        border_color, fill_color = self._learn_colors(ctx)
        if border_color is None:
            return None
        for p in ctx.train_pairs:
            expected = p['output']
            pred = self._apply(p['input'], border_color, fill_color)
            if pred != expected:
                return None
        preds = []
        for t_in in ctx.test_inputs:
            g = self._apply(t_in, border_color, fill_color)
            preds.append(Prediction(attempt_1=g, attempt_2=t_in, solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 3. DiagonalExtensionSolver
# ---------------------------------------------------------------------------

@register_solver
class DiagonalExtensionSolver(BaseSolver):
    """
    For each 3-cell L-shaped connected component of a non-zero color:
      - IC = inner corner (the cell adjacent to both arm tips)
      - direction = (A1.r+A2.r - 2*IC.r, A1.c+A2.c - 2*IC.c)  → ±1,±1
      - Extend diagonally starting two steps past IC (skip concave corner at IC+1*dir).
    """
    name = "DiagonalExtensionSolver"

    def _l_components(self, grid):
        H, W = len(grid), len(grid[0])
        results = []
        colors_present = {grid[r][c] for r in range(H) for c in range(W) if grid[r][c] != 0}
        for color in colors_present:
            for cc in _connected_components(grid, color):
                if len(cc) != 3:
                    continue
                cells = list(cc)
                def neighbor_count(cell):
                    r, c = cell
                    return sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                               if (r+dr, c+dc) in cc)
                nc_list = [(neighbor_count(cell), cell) for cell in cells]
                nc_list.sort(reverse=True)
                if nc_list[0][0] != 2:
                    continue
                IC = nc_list[0][1]
                A1, A2 = nc_list[1][1], nc_list[2][1]
                dr = A1[0] + A2[0] - 2*IC[0]
                dc = A1[1] + A2[1] - 2*IC[1]
                if abs(dr) != 1 or abs(dc) != 1:
                    continue
                results.append((color, IC, A1, A2, (dr, dc)))
        return results

    def _apply(self, grid):
        g = _copy_grid(grid)
        H, W = len(g), len(g[0])
        for (color, IC, A1, A2, (dr, dc)) in self._l_components(grid):
            r = IC[0] + 2*dr
            c = IC[1] + 2*dc
            while 0 <= r < H and 0 <= c < W:
                if g[r][c] == 0:
                    g[r][c] = color
                r += dr
                c += dc
        return g

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for p in ctx.train_pairs:
            pred = self._apply(p['input'])
            if pred != p['output']:
                return None
        preds = []
        for t_in in ctx.test_inputs:
            g = self._apply(t_in)
            preds.append(Prediction(attempt_1=g, attempt_2=t_in, solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 4. ReflectAroundPivotSolver
# ---------------------------------------------------------------------------

@register_solver
class ReflectAroundPivotSolver(BaseSolver):
    """
    Two non-zero colors: pivot (a compact block, never changes) and template
    (a pattern that gets reflected around the pivot center).
    Three reflections are added: H-flip, V-flip, both-flip.
    Colors are learned from training pairs.
    """
    name = "ReflectAroundPivotSolver"

    def _identify_roles(self, ctx):
        """
        pivot_color  = color whose cells are UNCHANGED across all training pairs.
        template_color = color that GAINS new cells across training pairs.
        """
        pivot_cand = None
        tmpl_cand = None
        for p in ctx.train_pairs:
            inp, out = p['input'], p['output']
            H_in, W_in = len(inp), len(inp[0]) if inp else 0
            H_out, W_out = len(out), len(out[0]) if out else 0
            if H_in != H_out or W_in != W_out:
                return None, None
            H, W = H_in, W_in
            inp_colors = {}
            out_colors = {}
            for r in range(H):
                for c in range(W):
                    v_in, v_out = inp[r][c], out[r][c]
                    if v_in != 0:
                        inp_colors.setdefault(v_in, set()).add((r,c))
                    if v_out != 0:
                        out_colors.setdefault(v_out, set()).add((r,c))
            unchanged = set()
            gained = set()
            for color in set(list(inp_colors.keys()) + list(out_colors.keys())):
                ic = inp_colors.get(color, set())
                oc = out_colors.get(color, set())
                if ic == oc:
                    unchanged.add(color)
                elif oc > ic:
                    gained.add(color)
            if pivot_cand is None:
                pivot_cand = unchanged - {0}
                tmpl_cand = (gained - {0}) if gained else None
            else:
                pivot_cand &= unchanged - {0}
                # FIX: skip no-change pairs (gained empty) when updating tmpl_cand
                if gained:
                    gc = gained - {0}
                    if tmpl_cand is None:
                        tmpl_cand = gc
                    else:
                        tmpl_cand &= gc
        if not pivot_cand or not tmpl_cand:
            return None, None
        return min(pivot_cand), min(tmpl_cand)

    def _pivot_center(self, grid, pivot_color):
        cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0]))
                 if grid[r][c] == pivot_color]
        if not cells:
            return None, None
        min_r = min(r for r,c in cells)
        max_r = max(r for r,c in cells)
        min_c = min(c for r,c in cells)
        max_c = max(c for r,c in cells)
        return (min_r + max_r) / 2.0, (min_c + max_c) / 2.0

    def _apply(self, grid, pivot_color, tmpl_color):
        g = _copy_grid(grid)
        H, W = len(g), len(g[0])
        cr, cc = self._pivot_center(grid, pivot_color)
        if cr is None:
            return g
        tmpl_cells = [(r, c) for r in range(H) for c in range(W)
                      if grid[r][c] == tmpl_color]
        for (r, c) in tmpl_cells:
            reflections = [
                (r,                  round(2*cc - c)),
                (round(2*cr - r),    c               ),
                (round(2*cr - r),    round(2*cc - c)),
            ]
            for (nr, nc) in reflections:
                if 0 <= nr < H and 0 <= nc < W and g[nr][nc] == 0:
                    g[nr][nc] = tmpl_color
        return g

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pivot_color, tmpl_color = self._identify_roles(ctx)
        if pivot_color is None:
            return None
        for p in ctx.train_pairs:
            pred = self._apply(p['input'], pivot_color, tmpl_color)
            if pred != p['output']:
                return None
        preds = []
        for t_in in ctx.test_inputs:
            g = self._apply(t_in, pivot_color, tmpl_color)
            preds.append(Prediction(attempt_1=g, attempt_2=t_in, solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# 5. LPathConnectSolver
# ---------------------------------------------------------------------------

@register_solver
class LPathConnectSolver(BaseSolver):
    """
    Three colors: obstacle (never overwritten), source (color A), target (color B).
    A source-colored L/U-shaped path connects the source segment to the target
    segment navigating around obstacles.

    Routing algorithm (3 phases):
      Phase 1 — extend source along its axis toward target until obstacle.
      Phase 2 — bridge perpendicularly from Phase-1 end to target's axis value.
      Phase 3 — extend along target's axis to reach the target segment.

    Direction priority: prefer direction toward target; break ties by the
    direction that allows the longest Phase-1 extension.
    """
    name = "LPathConnectSolver"

    # ------------------------------------------------------------------
    # color learning (unchanged from original — works correctly)
    # ------------------------------------------------------------------

    def _learn_colors(self, ctx):
        obstacle_cand = None
        source_cand = None
        for p in ctx.train_pairs:
            inp, out = p['input'], p['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None, None, None
            H, W = len(inp), len(inp[0])
            inp_cells, out_cells = {}, {}
            for r in range(H):
                for c in range(W):
                    if inp[r][c]: inp_cells.setdefault(inp[r][c], set()).add((r,c))
                    if out[r][c]: out_cells.setdefault(out[r][c], set()).add((r,c))
            unchanged, gained = set(), set()
            for color in set(list(inp_cells) + list(out_cells)):
                ic = inp_cells.get(color, set())
                oc = out_cells.get(color, set())
                if ic == oc:    unchanged.add(color)
                elif oc > ic:   gained.add(color)
            if obstacle_cand is None:
                obstacle_cand = unchanged - {0}
                source_cand   = gained    - {0}
            else:
                obstacle_cand &= unchanged - {0}
                source_cand   &= gained    - {0}
        if not source_cand:
            return None, None, None
        sc = min(source_cand)
        remaining = (obstacle_cand - source_cand) if obstacle_cand else set()
        if len(remaining) < 2:
            return None, None, None
        # disambiguate obstacle vs target by cell count in first training pair
        p0_inp = ctx.train_pairs[0]['input']
        H, W = len(p0_inp), len(p0_inp[0])
        counts = {}
        for r in range(H):
            for c in range(W):
                v = p0_inp[r][c]
                if v in remaining:
                    counts[v] = counts.get(v, 0) + 1
        if not counts:
            return None, None, None
        sorted_c = sorted(counts.items(), key=lambda x: -x[1])
        oc = sorted_c[0][0]
        tc_cands = remaining - {oc}
        if not tc_cands:
            return None, None, None
        return oc, sc, min(tc_cands)

    # ------------------------------------------------------------------
    # L/U routing
    # ------------------------------------------------------------------

    def _route_path(self, grid, src_cells, tgt_cells, obstacle_color):
        """
        Returns list of NEW (r,c) cells to paint with source color,
        or None if no valid route found.
        """
        H, W = len(grid), len(grid[0])
        tgt_set = set(tgt_cells)

        src_rs = [r for r,c in src_cells]
        src_cs = [c for r,c in src_cells]
        tgt_rs = [r for r,c in tgt_cells]
        tgt_cs = [c for r,c in tgt_cells]

        src_is_h = (len(set(src_rs)) == 1)
        (len(set(tgt_rs)) == 1)

        # center-of-mass for direction preference
        src_r_ctr = sum(src_rs) / len(src_rs)
        src_c_ctr = sum(src_cs) / len(src_cs)
        tgt_r_ctr = sum(tgt_rs) / len(tgt_rs)
        tgt_c_ctr = sum(tgt_cs) / len(tgt_cs)

        if src_is_h:
            primary_dirs = [(0, 1), (0, -1)]
            def tip(dr, dc):
                return (src_rs[0], max(src_cs) if dc > 0 else min(src_cs))
            def toward(dr, dc):
                return (dc > 0 and tgt_c_ctr > src_c_ctr) or (dc < 0 and tgt_c_ctr < src_c_ctr)
        else:
            primary_dirs = [(-1, 0), (1, 0)]
            def tip(dr, dc):
                return (min(src_rs) if dr < 0 else max(src_rs), src_cs[0])
            def toward(dr, dc):
                return (dr < 0 and tgt_r_ctr < src_r_ctr) or (dr > 0 and tgt_r_ctr > src_r_ctr)

        def phase1_len(dr, dc):
            tr, tc = tip(dr, dc)
            r, c = tr + dr, tc + dc
            n = 0
            while 0 <= r < H and 0 <= c < W \
                    and grid[r][c] != obstacle_color \
                    and (r, c) not in tgt_set:
                n += 1; r += dr; c += dc
            return n

        dirs_sorted = sorted(primary_dirs,
                             key=lambda d: (0 if toward(*d) else 1, -phase1_len(*d)))

        for (dr1, dc1) in dirs_sorted:
            tr, tc = tip(dr1, dc1)

            # ---- Phase 1: extend from tip ----
            path1 = []
            r, c = tr + dr1, tc + dc1
            while 0 <= r < H and 0 <= c < W:
                if grid[r][c] == obstacle_color:
                    break
                if (r, c) in tgt_set:
                    return path1          # direct hit
                path1.append((r, c))
                r += dr1; c += dc1
            if not path1:
                continue
            e1r, e1c = path1[-1]

            # ---- Phase 2: perpendicular bridge toward target axis ----
            if dr1 != 0:          # vertical Phase 1 → horizontal bridge
                # bridge to target's column
                tgt_ax = tgt_cs[0] if len(set(tgt_cs)) == 1 else \
                         (min(tgt_cs) if tgt_c_ctr < e1c else max(tgt_cs))
                if tgt_ax == e1c:
                    continue      # same column, bridge degenerate
                br2, bc2 = 0, (1 if tgt_ax > e1c else -1)
            else:                 # horizontal Phase 1 → vertical bridge
                # bridge to target's row
                tgt_ax = tgt_rs[0] if len(set(tgt_rs)) == 1 else \
                         (min(tgt_rs) if tgt_r_ctr < e1r else max(tgt_rs))
                if tgt_ax == e1r:
                    continue
                br2, bc2 = (1 if tgt_ax > e1r else -1), 0

            path2 = []
            r, c = e1r + br2, e1c + bc2
            blocked2 = False
            while 0 <= r < H and 0 <= c < W:
                if grid[r][c] == obstacle_color:
                    blocked2 = True; break
                if (r, c) in tgt_set:
                    return path1 + path2
                path2.append((r, c))
                if (br2 != 0 and r == tgt_ax) or (bc2 != 0 and c == tgt_ax):
                    break
                r += br2; c += bc2
            if blocked2 or not path2:
                continue
            e2r, e2c = path2[-1]

            # ---- Phase 3: extend from bridge end toward target ----
            if dr1 != 0:
                dr3, dc3 = dr1, 0     # same vertical direction as Phase 1
            else:
                dc3 = (-1 if tgt_c_ctr < e2c else 1)
                dr3 = 0

            path3 = []
            r, c = e2r + dr3, e2c + dc3
            blocked3 = False
            while 0 <= r < H and 0 <= c < W:
                if grid[r][c] == obstacle_color:
                    blocked3 = True; break
                if (r, c) in tgt_set:
                    break
                path3.append((r, c))
                r += dr3; c += dc3
            if blocked3:
                continue

            return path1 + path2 + path3

        return None   # no valid L/U route found

    # ------------------------------------------------------------------

    def _apply(self, grid, obstacle_color, source_color, target_color):
        g = _copy_grid(grid)
        H, W = len(g), len(g[0])
        src_cells = [(r,c) for r in range(H) for c in range(W) if grid[r][c] == source_color]
        tgt_cells = [(r,c) for r in range(H) for c in range(W) if grid[r][c] == target_color]
        if not src_cells or not tgt_cells:
            return g
        path = self._route_path(grid, src_cells, tgt_cells, obstacle_color)
        if path is None:
            return g
        for (r, c) in path:
            if g[r][c] == 0:
                g[r][c] = source_color
        return g

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        obstacle_color, source_color, target_color = self._learn_colors(ctx)
        if obstacle_color is None:
            return None
        for p in ctx.train_pairs:
            pred = self._apply(p['input'], obstacle_color, source_color, target_color)
            if pred != p['output']:
                return None
        preds = []
        for t_in in ctx.test_inputs:
            g = self._apply(t_in, obstacle_color, source_color, target_color)
            preds.append(Prediction(attempt_1=g, attempt_2=t_in, solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# Registry export
# ---------------------------------------------------------------------------

SOLVERS = [
    HSpanFillSolver,
    FillRectFrameSolver,
    DiagonalExtensionSolver,
    ReflectAroundPivotSolver,
    LPathConnectSolver,
]
