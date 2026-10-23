"""
generalized_solvers.py — Self-learning pattern induction engine for ARC-AGI-2.

Covers the 6 dominant unsolved buckets identified from the 864-task audit:
  1. NeighborhoodRuleInducerSolver   — context→color lookup (RECOLOR_SAME_SIZE)
  2. PaletteRemapInducerSolver       — systematic color remapping
  3. FloodFillFromSeedSolver         — flood fill from marker pixel
  4. FillEnclosedRegionsSolver       — fill interiors of closed borders
  5. RemoveOutliersSolver            — remove isolated / noise pixels
  6. ObjectRecolorBySizeSolver       — recolor connected regions by area rank
  7. ObjectRecolorByEnclosedSolver   — enclosed-region content determines recolor
  8. SymmetryCompletionInducerSolver — induce & complete grid symmetry
  9. SeparatorZoneFillInducerSolver  — divide by separator lines, fill zones
  10. ExtractLargestObjectSolver     — crop bounding-box of dominant object
  11. ColorMapFromIntersectionSolver — mark line-cross intersections
  12. NoiseFilterObjectsSolver       — keep largest connected component per color
  13. RegionCountOutputSolver        — count regions → single-cell output
  14. ObjectPropertyGridSolver       — grid of object properties (SHRINK)
  15. HoleCountRecolorSolver         — recolor objects by number of enclosed holes
  16. DiagonalFloodFillSolver        — 8-connected flood fill variant
  17. MirrorFillSolver               — reflect object across detected axis
  18. ColorFreqReplaceSolver         — replace least-frequent color with most-frequent
  19. CellNeighborColorSolver        — each cell gets color of majority neighbor
  20. FrameInteriorColorSolver       — fill frame interior with dominant frame color
"""
from __future__ import annotations

import collections
from collections import Counter, defaultdict, deque
from typing import Optional

from ..core.grid_codec import (
    Grid,
    background_color,
    color_histogram,
    copy_grid,
)
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver


# ── Shared utilities ──────────────────────────────────────────────────────────

def _shape(g: Grid):
    return len(g), len(g[0]) if g else 0

def _eq(a: Grid, b: Grid) -> bool:
    return a == b

def _bg(g: Grid) -> int:
    return background_color(g)

def _clone(g: Grid) -> Grid:
    return [list(r) for r in g]

def _unique(g: Grid) -> set:
    s = set()
    for r in g: s.update(r)
    return s

def _nonbg_cells(g: Grid, bg: int):
    H, W = _shape(g)
    return [(r, c) for r in range(H) for c in range(W) if g[r][c] != bg]

def _bfs(g: Grid, sr: int, sc: int, passable=None) -> set:
    H, W = _shape(g)
    color = g[sr][sc]
    if passable is None:
        passable = {color}
    visited = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and g[nr][nc] in passable:
                visited.add((nr,nc))
                q.append((nr,nc))
    return visited

def _connected_components(g: Grid, bg: int) -> list[set]:
    H, W = _shape(g)
    visited = set()
    comps = []
    for r in range(H):
        for c in range(W):
            if g[r][c] != bg and (r,c) not in visited:
                comp = _bfs(g, r, c, passable={g[r][c]})
                visited.update(comp)
                comps.append(comp)
    return comps

def _all_components_by_color(g: Grid) -> dict[int, list[set]]:
    H, W = _shape(g)
    by_color = defaultdict(list)
    visited = set()
    for r in range(H):
        for c in range(W):
            col = g[r][c]
            if (r,c) not in visited:
                comp = _bfs(g, r, c, passable={col})
                visited.update(comp)
                by_color[col].append(comp)
    return dict(by_color)

def _is_border_cell(r, c, H, W):
    return r == 0 or r == H-1 or c == 0 or c == W-1

def _interior_bg_cells(g: Grid, bg_val: int) -> set:
    """Cells that are background but NOT reachable from any border."""
    H, W = _shape(g)
    # BFS from all border bg cells
    visited = set()
    q = deque()
    for r in range(H):
        for c in range(W):
            if _is_border_cell(r, c, H, W) and g[r][c] == bg_val:
                if (r,c) not in visited:
                    visited.add((r,c))
                    q.append((r,c))
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and g[nr][nc] == bg_val:
                visited.add((nr,nc))
                q.append((nr,nc))
    interior = set()
    for r in range(H):
        for c in range(W):
            if g[r][c] == bg_val and (r,c) not in visited:
                interior.add((r,c))
    return interior

def _make_result(ctx: TaskContext, grid: Grid, name: str, conf: float = 1.0) -> SolverResult:
    preds = [Prediction(attempt_1=grid, attempt_2=grid, solver_name=name, confidence=conf)
             for _ in ctx.test_inputs]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

def _make_result_list(ctx: TaskContext, grids: list[Grid], name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=name, confidence=1.0)
             for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

def _get_neighborhood(g: Grid, r: int, c: int, radius: int = 1) -> tuple:
    H, W = _shape(g)
    cells = []
    for dr in range(-radius, radius+1):
        for dc in range(-radius, radius+1):
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                cells.append(g[nr][nc])
            else:
                cells.append(-1)
    return tuple(cells)


# ══════════════════════════════════════════════════════════════════════════════
# 1. NeighborhoodRuleInducerSolver
#    Learns: neighborhood_context → output_color lookup table from all training pairs.
#    Handles same-size I/O. Tries radius 1 then radius 2.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class NeighborhoodRuleInducerSolver(BaseSolver):
    """Strict version: neighborhood context -> output color.
    Requires:
    - All training pairs same-size I/O
    - Every neighborhood key maps to EXACTLY ONE output color across all pairs
    - Every test input key must appear in the learned rule (no unknowns)
    - The rule must change at least one cell (non-identity)
    - At least 2 training pairs required to avoid overfitting
    """
    name = "NeighborhoodRuleInducerSolver"
    priority = 20

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if len(pairs) < 2:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        for radius in (1,):  # radius=1 only; radius=2 too large key space
            # Build strict deterministic rule
            resolved: dict[tuple, int] = {}
            consistent = True
            for p in pairs:
                inp, out = p["input"], p["output"]
                H, W = _shape(inp)
                for r in range(H):
                    for c in range(W):
                        k = _get_neighborhood(inp, r, c, radius)
                        v = out[r][c]
                        if k in resolved:
                            if resolved[k] != v:
                                consistent = False
                                break
                        else:
                            resolved[k] = v
                    if not consistent:
                        break
                if not consistent:
                    break
            if not consistent:
                continue

            # Validate reconstruction on training pairs
            ok = True
            changed = False
            for p in pairs:
                inp, out = p["input"], p["output"]
                H, W = _shape(inp)
                for r in range(H):
                    for c in range(W):
                        k = _get_neighborhood(inp, r, c, radius)
                        expected = resolved.get(k)
                        if expected is None:
                            ok = False; break
                        if expected != out[r][c]:
                            ok = False; break
                        if expected != inp[r][c]:
                            changed = True
                    if not ok:
                        break
                if not ok:
                    break
            if not ok or not changed:
                continue

            # All test keys must be known
            results = []
            for test_in in ctx.test_inputs:
                H, W = _shape(test_in)
                pred = _clone(test_in)
                all_known = True
                for r in range(H):
                    for c in range(W):
                        k = _get_neighborhood(test_in, r, c, radius)
                        if k not in resolved:
                            all_known = False; break
                        pred[r][c] = resolved[k]
                    if not all_known:
                        break
                if not all_known:
                    return None
                results.append(pred)
            return _make_result_list(ctx, results, self.name)

        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. PaletteRemapInducerSolver
#    Learns a global per-color mapping from training pairs.
#    Only accepts if the mapping is consistent across all pairs AND non-trivial.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class PaletteRemapInducerSolver(BaseSolver):
    name = "PaletteRemapInducerSolver"
    priority = 10

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        mapping: dict[int, int] = {}
        for p in pairs:
            inp, out = p["input"], p["output"]
            if _shape(inp) != _shape(out):
                return None
            H, W = _shape(inp)
            for r in range(H):
                for c in range(W):
                    ci, co = inp[r][c], out[r][c]
                    if ci in mapping:
                        if mapping[ci] != co:
                            return None
                    else:
                        mapping[ci] = co

        if not mapping:
            return None
        # Must be non-trivial
        if all(k == v for k, v in mapping.items()):
            return None
        # Must not be identity-like with only bg→bg entries changed
        non_id = {k: v for k, v in mapping.items() if k != v}
        if not non_id:
            return None

        # Validate
        for p in pairs:
            inp, out = p["input"], p["output"]
            H, W = _shape(inp)
            pred = [[mapping.get(inp[r][c], inp[r][c]) for c in range(W)] for r in range(H)]
            if pred != out:
                return None

        results = []
        for test_in in ctx.test_inputs:
            H, W = _shape(test_in)
            pred = [[mapping.get(test_in[r][c], test_in[r][c]) for c in range(W)] for r in range(H)]
            results.append(pred)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FloodFillFromSeedSolver
#    Detects a seed cell that determines: which region to fill and with what color.
#    Tries: seed=rarest fg color, target=enclosed bg cells or connected component.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FloodFillFromSeedSolver(BaseSolver):
    name = "FloodFillFromSeedSolver"
    priority = 25

    def _learn_one(self, inp: Grid, out: Grid):
        """Returns (seed_color, fill_color, mode) or None."""
        if _shape(inp) != _shape(out):
            return None
        H, W = _shape(inp)
        # Find changed cells
        changed = [(r, c) for r in range(H) for c in range(W) if inp[r][c] != out[r][c]]
        if not changed:
            return None
        changed_set = set(changed)

        # New color in output at changed positions
        new_colors = Counter(out[r][c] for r, c in changed)
        orig_colors = Counter(inp[r][c] for r, c in changed)

        if len(new_colors) != 1:
            return None  # Changed cells must all become same color
        fill_color = list(new_colors.keys())[0]

        # All changed cells must come from same original color (or bg)
        if len(orig_colors) != 1:
            return None
        orig_color = list(orig_colors.keys())[0]

        # Find the seed: a non-bg cell NOT in changed_set, which is adjacent to changed region
        # The seed should be a color different from orig_color and fill_color
        bg = _bg(inp)

        # Approach 1: Changed cells are all connected → it's a flood fill
        # Check connectivity
        start = changed[0]
        visited = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (nr,nc) in changed_set and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    q.append((nr,nc))
        if len(visited) != len(changed_set):
            return None  # Not a connected region

        return (orig_color, fill_color)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        learned = []
        for p in pairs:
            r = self._learn_one(p["input"], p["output"])
            if r is None:
                return None
            learned.append(r)

        # Check consistency: same (orig_color, fill_color) across all pairs
        if len(set(learned)) != 1:
            return None
        orig_color, fill_color = learned[0]

        # Apply to test: find region of orig_color cells and flood-fill them with fill_color
        # But we need to know WHICH region to fill... check training for a "pointer"
        # Strategy: flood fill from every orig_color cell (entire connected component)
        results = []
        for test_in in ctx.test_inputs:
            H, W = _shape(test_in)
            result = _clone(test_in)
            # Find all cells of orig_color and flood fill each component
            visited = set()
            for r in range(H):
                for c in range(W):
                    if test_in[r][c] == orig_color and (r,c) not in visited:
                        comp = _bfs(test_in, r, c, passable={orig_color})
                        visited.update(comp)
                        for cr, cc in comp:
                            result[cr][cc] = fill_color

            # Validate: must be different from input
            if result == test_in:
                return None
            results.append(result)

        # Validate on training
        for p in pairs:
            inp, out = p["input"], p["output"]
            H, W = _shape(inp)
            pred = _clone(inp)
            visited = set()
            for r in range(H):
                for c in range(W):
                    if inp[r][c] == orig_color and (r,c) not in visited:
                        comp = _bfs(inp, r, c, passable={orig_color})
                        visited.update(comp)
                        for cr, cc in comp:
                            pred[cr][cc] = fill_color
            if pred != out:
                return None

        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FillEnclosedRegionsSolver
#    Fills background cells that are enclosed (not reachable from border) with
#    the dominant color of the surrounding border region.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FillEnclosedRegionsSolver(BaseSolver):
    name = "FillEnclosedRegionsSolver"
    priority = 22

    def _apply(self, g: Grid) -> Grid:
        H, W = _shape(g)
        bg = _bg(g)
        interior = _interior_bg_cells(g, bg)
        if not interior:
            return g
        # Group interior cells into connected components
        visited = set()
        result = _clone(g)
        for (r0, c0) in interior:
            if (r0, c0) in visited:
                continue
            # BFS within interior
            comp = set()
            q = deque([(r0,c0)])
            comp.add((r0,c0))
            while q:
                r, c = q.popleft()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr,nc) in interior and (nr,nc) not in comp:
                        comp.add((nr,nc))
                        q.append((nr,nc))
            visited.update(comp)
            # Find bordering colors (adjacent non-bg cells)
            border_colors = Counter()
            for (r, c) in comp:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and g[nr][nc] != bg and (nr,nc) not in interior:
                        border_colors[g[nr][nc]] += 1
            if border_colors:
                fill_col = border_colors.most_common(1)[0][0]
                for (r, c) in comp:
                    result[r][c] = fill_col
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # Same size I/O required
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Validate on training
        for p in pairs:
            pred = self._apply(p["input"])
            if pred != p["output"]:
                return None

        results = [self._apply(t) for t in ctx.test_inputs]
        if all(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 5. RemoveOutliersSolver
#    Removes isolated single-pixel (or small-cluster) foreground cells:
#    pixels whose color doesn't match any neighbor. Background fills in.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RemoveOutliersSolver(BaseSolver):
    name = "RemoveOutliersSolver"
    priority = 30

    def _apply(self, g: Grid, max_size: int) -> Grid:
        H, W = _shape(g)
        bg = _bg(g)
        comps = _connected_components(g, bg)
        result = _clone(g)
        for comp in comps:
            if len(comp) <= max_size:
                for (r, c) in comp:
                    result[r][c] = bg
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        for max_sz in (1, 2, 3):
            ok = True
            for p in pairs:
                pred = self._apply(p["input"], max_sz)
                if pred != p["output"]:
                    ok = False
                    break
                # Output must have fewer fg cells than input
                H, W = _shape(p["input"])
                inp_nz = sum(1 for r in range(H) for c in range(W) if p["input"][r][c] != _bg(p["input"]))
                out_nz = sum(1 for r in range(H) for c in range(W) if p["output"][r][c] != _bg(p["output"]))
                if out_nz >= inp_nz:
                    ok = False
                    break
            if ok:
                results = [self._apply(t, max_sz) for t in ctx.test_inputs]
                return _make_result_list(ctx, results, self.name)

        return None


# ══════════════════════════════════════════════════════════════════════════════
# 6. ObjectRecolorBySizeSolver
#    Recolors connected components based on their size rank.
#    Largest component → color A, second largest → color B, etc.
#    Learns the rank→color mapping from training pairs.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ObjectRecolorBySizeSolver(BaseSolver):
    name = "ObjectRecolorBySizeSolver"
    priority = 28

    def _extract_rank_map(self, inp: Grid, out: Grid):
        """Returns rank→output_color mapping, or None."""
        if _shape(inp) != _shape(out):
            return None
        bg = _bg(inp)
        comps = sorted(_connected_components(inp, bg), key=lambda c: len(c), reverse=True)
        if len(comps) < 2:
            return None
        rank_map = {}
        for rank, comp in enumerate(comps):
            # All cells of this component must have the same output color
            out_colors = set(out[r][c] for r, c in comp)
            if len(out_colors) != 1:
                return None
            rank_map[rank] = list(out_colors)[0]
        return rank_map

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn rank→color from first pair
        rank_map = self._extract_rank_map(pairs[0]["input"], pairs[0]["output"])
        if not rank_map:
            return None
        # Must actually recolor (not identity)
        if all(pairs[0]["input"][r][c] == pairs[0]["output"][r][c]
               for r in range(len(pairs[0]["input"])) for c in range(len(pairs[0]["input"][0]))):
            return None

        # Validate on all training pairs
        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = sorted(_connected_components(inp, bg), key=lambda c: len(c), reverse=True)
            pred = _clone(inp)
            for rank, comp in enumerate(comps):
                col = rank_map.get(rank)
                if col is None:
                    # Use last known rank color for overflow
                    col = rank_map.get(len(rank_map)-1, bg)
                for r, c in comp:
                    pred[r][c] = col
            if pred != out:
                return None

        results = []
        for test_in in ctx.test_inputs:
            bg = _bg(test_in)
            comps = sorted(_connected_components(test_in, bg), key=lambda c: len(c), reverse=True)
            pred = _clone(test_in)
            for rank, comp in enumerate(comps):
                col = rank_map.get(rank)
                if col is None:
                    col = rank_map.get(len(rank_map)-1, bg)
                for r, c in comp:
                    pred[r][c] = col
            results.append(pred)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 7. ObjectRecolorByEnclosedSolver
#    Objects that enclose interior holes get a different color than those that don't.
#    Learns: has_holes → output_color.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ObjectRecolorByEnclosedSolver(BaseSolver):
    name = "ObjectRecolorByEnclosedSolver"
    priority = 29

    def _has_holes(self, g: Grid, comp: set, bg: int) -> bool:
        """Does this component enclose any interior bg cells?"""
        H, W = _shape(g)
        # Temporarily mark comp as a barrier; check if any bg cells enclosed
        temp = _clone(g)
        temp_bg_reached = set()
        q = deque()
        visited = set()
        for r in range(H):
            for c in range(W):
                if _is_border_cell(r, c, H, W) and temp[r][c] == bg:
                    if (r,c) not in visited:
                        visited.add((r,c))
                        q.append((r,c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and temp[nr][nc] == bg:
                    visited.add((nr,nc))
                    q.append((nr,nc))
        # Any bg cell not in visited and adjacent to comp?
        comp_adj_bg = set()
        for (r, c) in comp:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and temp[nr][nc] == bg and (nr,nc) not in visited:
                    comp_adj_bg.add((nr,nc))
        return len(comp_adj_bg) > 0

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn: component has_holes → output_color
        hole_color = None
        no_hole_color = None
        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            for comp in comps:
                has_h = self._has_holes(inp, comp, bg)
                r0, c0 = next(iter(comp))
                out_cols = set(out[r][c] for r, c in comp)
                if len(out_cols) != 1:
                    return None
                out_col = list(out_cols)[0]
                if has_h:
                    if hole_color is None:
                        hole_color = out_col
                    elif hole_color != out_col:
                        return None
                else:
                    if no_hole_color is None:
                        no_hole_color = out_col
                    elif no_hole_color != out_col:
                        return None

        if hole_color is None or no_hole_color is None:
            return None
        if hole_color == no_hole_color:
            return None

        # Validate
        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            pred = _clone(inp)
            for comp in comps:
                has_h = self._has_holes(inp, comp, bg)
                col = hole_color if has_h else no_hole_color
                for r, c in comp:
                    pred[r][c] = col
            if pred != out:
                return None

        results = []
        for test_in in ctx.test_inputs:
            bg = _bg(test_in)
            comps = _connected_components(test_in, bg)
            pred = _clone(test_in)
            for comp in comps:
                has_h = self._has_holes(test_in, comp, bg)
                col = hole_color if has_h else no_hole_color
                for r, c in comp:
                    pred[r][c] = col
            results.append(pred)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 8. SymmetryCompletionInducerSolver
#    Detects which symmetry axis makes the output perfectly symmetric.
#    Tries: horizontal, vertical, diagonal, anti-diagonal.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class SymmetryCompletionInducerSolver(BaseSolver):
    name = "SymmetryCompletionInducerSolver"
    priority = 32

    def _complete_h(self, g: Grid) -> Grid:
        H, W = _shape(g)
        bg = _bg(g)
        res = _clone(g)
        for r in range(H):
            for c in range(W):
                mr = r
                mc = W - 1 - c
                if res[r][c] == bg and res[mr][mc] != bg:
                    res[r][c] = res[mr][mc]
                elif res[mr][mc] == bg and res[r][c] != bg:
                    res[mr][mc] = res[r][c]
        return res

    def _complete_v(self, g: Grid) -> Grid:
        H, W = _shape(g)
        bg = _bg(g)
        res = _clone(g)
        for r in range(H):
            for c in range(W):
                mr = H - 1 - r
                if res[r][c] == bg and res[mr][c] != bg:
                    res[r][c] = res[mr][c]
                elif res[mr][c] == bg and res[r][c] != bg:
                    res[mr][c] = res[r][c]
        return res

    def _complete_diag(self, g: Grid) -> Grid:
        H, W = _shape(g)
        if H != W:
            return g
        bg = _bg(g)
        res = _clone(g)
        for r in range(H):
            for c in range(W):
                if res[r][c] == bg and res[c][r] != bg:
                    res[r][c] = res[c][r]
                elif res[c][r] == bg and res[r][c] != bg:
                    res[c][r] = res[r][c]
        return res

    def _complete_antidiag(self, g: Grid) -> Grid:
        H, W = _shape(g)
        if H != W:
            return g
        bg = _bg(g)
        res = _clone(g)
        for r in range(H):
            for c in range(W):
                mr, mc = W-1-c, H-1-r
                if res[r][c] == bg and res[mr][mc] != bg:
                    res[r][c] = res[mr][mc]
                elif res[mr][mc] == bg and res[r][c] != bg:
                    res[mr][mc] = res[r][c]
        return res

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        for fn in (self._complete_h, self._complete_v, self._complete_diag, self._complete_antidiag):
            ok = True
            for p in pairs:
                pred = fn(p["input"])
                if pred != p["output"]:
                    ok = False
                    break
            if ok:
                results = [fn(t) for t in ctx.test_inputs]
                if any(r != t for r, t in zip(results, ctx.test_inputs)):
                    return _make_result_list(ctx, results, self.name)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 9. SeparatorZoneFillInducerSolver
#    Detects separator lines (full rows/columns of one color), divides grid into
#    zones, learns the fill-color rule for each zone from training examples.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class SeparatorZoneFillInducerSolver(BaseSolver):
    name = "SeparatorZoneFillInducerSolver"
    priority = 35

    def _find_separators(self, g: Grid):
        H, W = _shape(g)
        row_seps = [r for r in range(H) if len(set(g[r])) == 1]
        col_seps = [c for c in range(W) if len(set(g[r][c] for r in range(H))) == 1]
        return row_seps, col_seps

    def _get_zones(self, H: int, W: int, row_seps: list, col_seps: list):
        rs = sorted(set([-1] + row_seps + [H]))
        cs = sorted(set([-1] + col_seps + [W]))
        zones = []
        for i in range(len(rs)-1):
            for j in range(len(cs)-1):
                r0, r1 = rs[i]+1, rs[i+1]
                c0, c1 = cs[j]+1, cs[j+1]
                if r0 < r1 and c0 < c1:
                    zones.append((r0, r1, c0, c1))
        return zones

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn: position → fill color, from first pair
        p0 = pairs[0]
        row_seps, col_seps = self._find_separators(p0["input"])
        if not row_seps and not col_seps:
            return None

        H, W = _shape(p0["input"])
        zones = self._get_zones(H, W, row_seps, col_seps)
        if len(zones) < 2:
            return None

        # Learn zone→output_color from training
        zone_color_rule = {}  # zone_idx → output_color
        for p in pairs:
            inp, out = p["input"], p["output"]
            rs, cs = self._find_separators(inp)
            H2, W2 = _shape(inp)
            zones2 = self._get_zones(H2, W2, rs, cs)
            for zi, (r0, r1, c0, c1) in enumerate(zones2):
                zone_out_colors = Counter(out[r][c] for r in range(r0,r1) for c in range(c0,c1))
                if not zone_out_colors:
                    continue
                fill_col = zone_out_colors.most_common(1)[0][0]
                if zi in zone_color_rule:
                    if zone_color_rule[zi] != fill_col:
                        return None
                else:
                    zone_color_rule[zi] = fill_col

        if not zone_color_rule:
            return None

        # Validate
        for p in pairs:
            inp, out = p["input"], p["output"]
            rs, cs = self._find_separators(inp)
            H2, W2 = _shape(inp)
            zones2 = self._get_zones(H2, W2, rs, cs)
            pred = _clone(inp)
            for zi, (r0, r1, c0, c1) in enumerate(zones2):
                col = zone_color_rule.get(zi)
                if col is None:
                    return None
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        pred[r][c] = col
            if pred != out:
                return None

        results = []
        for test_in in ctx.test_inputs:
            rs, cs = self._find_separators(test_in)
            H2, W2 = _shape(test_in)
            zones2 = self._get_zones(H2, W2, rs, cs)
            pred = _clone(test_in)
            for zi, (r0, r1, c0, c1) in enumerate(zones2):
                col = zone_color_rule.get(zi)
                if col is not None:
                    for r in range(r0, r1):
                        for c in range(c0, c1):
                            pred[r][c] = col
            results.append(pred)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 10. ExtractLargestObjectSolver
#     Output = bounding box crop of the largest non-background connected component.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ExtractLargestObjectSolver(BaseSolver):
    name = "ExtractLargestObjectSolver"
    priority = 40

    def _apply(self, g: Grid) -> Optional[Grid]:
        bg = _bg(g)
        comps = _connected_components(g, bg)
        if not comps:
            return None
        largest = max(comps, key=lambda c: len(c))
        rows = [r for r, c in largest]
        cols = [c for r, c in largest]
        r0, r1 = min(rows), max(rows)+1
        c0, c1 = min(cols), max(cols)+1
        H, W = _shape(g)
        return [g[r][c0:c1] for r in range(r0, r1)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # Output must be smaller than input
        for p in pairs:
            ih, iw = _shape(p["input"])
            oh, ow = _shape(p["output"])
            if oh * ow >= ih * iw:
                return None

        for p in pairs:
            pred = self._apply(p["input"])
            if pred is None or pred != p["output"]:
                return None

        results = []
        for test_in in ctx.test_inputs:
            r = self._apply(test_in)
            if r is None:
                return None
            results.append(r)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 11. ColorMapFromIntersectionSolver
#     Detects lines of two colors; marks their intersections with a third color.
#     Generalizes CrossIntersectionFillSolver.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ColorMapFromIntersectionSolver(BaseSolver):
    name = "ColorMapFromIntersectionSolver"
    priority = 38

    def _find_lines(self, g: Grid):
        """Find colors that appear in full rows or full columns."""
        H, W = _shape(g)
        row_colors = {}
        for r in range(H):
            unique = set(g[r])
            if len(unique) == 1:
                row_colors[r] = list(unique)[0]
        col_colors = {}
        for c in range(W):
            unique = set(g[r][c] for r in range(H))
            if len(unique) == 1:
                col_colors[c] = list(unique)[0]
        return row_colors, col_colors

    def _apply(self, g: Grid, mark_color: int) -> Grid:
        row_colors, col_colors = self._find_lines(g)
        if not row_colors or not col_colors:
            return g
        result = _clone(g)
        for r, rc in row_colors.items():
            for c, cc in col_colors.items():
                if rc != cc:  # Different-colored lines intersect
                    result[r][c] = mark_color
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn the mark color
        mark_color = None
        for p in pairs:
            inp, out = p["input"], p["output"]
            changed = [(r, c, out[r][c]) for r in range(len(inp)) for c in range(len(inp[0]))
                       if inp[r][c] != out[r][c]]
            if not changed:
                return None
            colors = set(co for _, _, co in changed)
            if len(colors) != 1:
                return None
            mc = list(colors)[0]
            if mark_color is None:
                mark_color = mc
            elif mark_color != mc:
                return None

        if mark_color is None:
            return None

        for p in pairs:
            pred = self._apply(p["input"], mark_color)
            if pred != p["output"]:
                return None

        results = [self._apply(t, mark_color) for t in ctx.test_inputs]
        if any(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 12. NoiseFilterObjectsSolver
#     Keep only the connected components with size ≥ threshold (learned from training).
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class NoiseFilterObjectsSolver(BaseSolver):
    name = "NoiseFilterObjectsSolver"
    priority = 31

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn min_size threshold
        for min_size in range(2, 10):
            ok = True
            for p in pairs:
                inp, out = p["input"], p["output"]
                bg = _bg(inp)
                pred = _clone(inp)
                for comp in _connected_components(inp, bg):
                    if len(comp) < min_size:
                        for r, c in comp:
                            pred[r][c] = bg
                if pred != out:
                    ok = False
                    break
            if ok:
                results = []
                for test_in in ctx.test_inputs:
                    bg = _bg(test_in)
                    pred = _clone(test_in)
                    for comp in _connected_components(test_in, bg):
                        if len(comp) < min_size:
                            for r, c in comp:
                                pred[r][c] = bg
                    results.append(pred)
                if any(r == t for r, t in zip(results, ctx.test_inputs)):
                    continue
                return _make_result_list(ctx, results, self.name)

        return None


# ══════════════════════════════════════════════════════════════════════════════
# 13. RegionCountOutputSolver
#     Output = 1×1 grid whose value = number of distinct connected components.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RegionCountOutputSolver(BaseSolver):
    name = "RegionCountOutputSolver"
    priority = 45

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # Output must be 1×1
        for p in pairs:
            if _shape(p["output"]) != (1, 1):
                return None

        for p in pairs:
            inp = p["input"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            expected = p["output"][0][0]
            # Try different counting strategies
            if len(comps) == expected:
                continue
            # Try counting by non-bg unique colors
            n_colors = len([c for c in set(inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))) if c != bg])
            if n_colors == expected:
                continue
            return None

        # Determine which strategy works
        strategies = []
        for p in pairs:
            inp = p["input"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            expected = p["output"][0][0]
            if len(comps) == expected:
                strategies.append("n_comps")
            else:
                n_colors = len([c for c in set(inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))) if c != bg])
                if n_colors == expected:
                    strategies.append("n_colors")
                else:
                    return None

        if len(set(strategies)) != 1:
            return None
        strat = strategies[0]

        results = []
        for test_in in ctx.test_inputs:
            bg = _bg(test_in)
            if strat == "n_comps":
                comps = _connected_components(test_in, bg)
                val = len(comps)
            else:
                val = len([c for c in set(test_in[r][c] for r in range(len(test_in)) for c in range(len(test_in[0]))) if c != bg])
            results.append([[val]])
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 14. ObjectPropertyGridSolver
#     Large grid with multiple framed objects → small grid of per-object values.
#     Detects the "answer grid" pattern where each cell's value = property of object.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ObjectPropertyGridSolver(BaseSolver):
    name = "ObjectPropertyGridSolver"
    priority = 42

    def _find_frame_regions(self, g: Grid, frame_color: int):
        """Find interior regions of rectangles made of frame_color."""
        H, W = _shape(g)
        regions = []
        # Find all rectangles made of frame_color
        for r0 in range(H):
            for c0 in range(W):
                if g[r0][c0] != frame_color:
                    continue
                for r1 in range(r0+2, H):
                    for c1 in range(c0+2, W):
                        if g[r1][c1] != frame_color:
                            continue
                        # Check if top/bottom rows and left/right cols are all frame_color
                        top_ok = all(g[r0][c] == frame_color for c in range(c0, c1+1))
                        bot_ok = all(g[r1][c] == frame_color for c in range(c0, c1+1))
                        lft_ok = all(g[r][c0] == frame_color for r in range(r0, r1+1))
                        rgt_ok = all(g[r][c1] == frame_color for r in range(r0, r1+1))
                        if top_ok and bot_ok and lft_ok and rgt_ok:
                            interior = [(r,c) for r in range(r0+1,r1) for c in range(c0+1,c1)]
                            regions.append(interior)
        return regions

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Complex; skip for now if output has very different size
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            ih, iw = _shape(p["input"])
            oh, ow = _shape(p["output"])
            if oh * ow >= ih * iw * 0.5:
                return None
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 15. HoleCountRecolorSolver
#     Recolors objects by the number of enclosed holes.
#     0 holes → color A, 1 hole → color B, 2 holes → color C, etc.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class HoleCountRecolorSolver(BaseSolver):
    name = "HoleCountRecolorSolver"
    priority = 33

    def _count_holes(self, g: Grid, comp: set) -> int:
        """Count enclosed interior background regions of this component."""
        H, W = _shape(g)
        bg = _bg(g)
        # BFS from all border bg cells
        visited = set()
        q = deque()
        for r in range(H):
            for c in range(W):
                if _is_border_cell(r, c, H, W) and g[r][c] == bg:
                    if (r,c) not in visited:
                        visited.add((r,c))
                        q.append((r,c))
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and g[nr][nc] == bg:
                    visited.add((nr,nc))
                    q.append((nr,nc))
        # Count distinct interior bg components adjacent to comp
        interior_bg = set()
        for r, c in comp:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W and g[nr][nc] == bg and (nr,nc) not in visited:
                    interior_bg.add((nr,nc))
        if not interior_bg:
            return 0
        # Count connected components of interior bg
        visited2 = set()
        n_holes = 0
        for (r0, c0) in interior_bg:
            if (r0, c0) not in visited2:
                n_holes += 1
                q = deque([(r0,c0)])
                visited2.add((r0,c0))
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if (nr,nc) in interior_bg and (nr,nc) not in visited2:
                            visited2.add((nr,nc))
                            q.append((nr,nc))
        return n_holes

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        hole_to_color: dict[int, int] = {}
        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            if len(comps) < 2:
                return None
            for comp in comps:
                n = self._count_holes(inp, comp)
                out_cols = set(out[r][c] for r, c in comp)
                if len(out_cols) != 1:
                    return None
                oc = list(out_cols)[0]
                if n in hole_to_color:
                    if hole_to_color[n] != oc:
                        return None
                else:
                    hole_to_color[n] = oc

        if not hole_to_color or len(hole_to_color) < 2:
            return None

        # Validate
        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            pred = _clone(inp)
            for comp in comps:
                n = self._count_holes(inp, comp)
                col = hole_to_color.get(n)
                if col is None:
                    return None
                for r, c in comp:
                    pred[r][c] = col
            if pred != out:
                return None

        results = []
        for test_in in ctx.test_inputs:
            bg = _bg(test_in)
            comps = _connected_components(test_in, bg)
            pred = _clone(test_in)
            for comp in comps:
                n = self._count_holes(test_in, comp)
                col = hole_to_color.get(n)
                if col is None:
                    return None
                for r, c in comp:
                    pred[r][c] = col
            results.append(pred)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 16. DiagonalFloodFillSolver
#     Like FloodFillFromSeedSolver but uses 8-connectivity.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class DiagonalFloodFillSolver(BaseSolver):
    name = "DiagonalFloodFillSolver"
    priority = 26

    def _bfs8(self, g: Grid, sr: int, sc: int, color: int) -> set:
        H, W = _shape(g)
        visited = {(sr, sc)}
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr in [-1,0,1]:
                for dc in [-1,0,1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in visited and g[nr][nc] == color:
                        visited.add((nr,nc))
                        q.append((nr,nc))
        return visited

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        learned = []
        for p in pairs:
            inp, out = p["input"], p["output"]
            if _shape(inp) != _shape(out):
                return None
            H, W = _shape(inp)
            changed = [(r, c) for r in range(H) for c in range(W) if inp[r][c] != out[r][c]]
            if not changed:
                return None
            new_colors = Counter(out[r][c] for r, c in changed)
            orig_colors = Counter(inp[r][c] for r, c in changed)
            if len(new_colors) != 1 or len(orig_colors) != 1:
                return None
            fill_color = list(new_colors.keys())[0]
            orig_color = list(orig_colors.keys())[0]
            learned.append((orig_color, fill_color))

        if len(set(learned)) != 1:
            return None
        orig_color, fill_color = learned[0]

        def apply_fn(g):
            H, W = _shape(g)
            result = _clone(g)
            visited = set()
            for r in range(H):
                for c in range(W):
                    if g[r][c] == orig_color and (r,c) not in visited:
                        comp = self._bfs8(g, r, c, orig_color)
                        visited.update(comp)
                        for cr, cc in comp:
                            result[cr][cc] = fill_color
            return result

        for p in pairs:
            if apply_fn(p["input"]) != p["output"]:
                return None

        results = [apply_fn(t) for t in ctx.test_inputs]
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 17. MirrorFillSolver
#     Detects partial objects that should be reflected to fill out their mirror.
#     Tries h-flip, v-flip, and rot-180. Applies when one half is empty (bg).
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class MirrorFillSolver(BaseSolver):
    name = "MirrorFillSolver"
    priority = 36

    def _flip_h(self, g: Grid) -> Grid:
        return [row[::-1] for row in g]

    def _flip_v(self, g: Grid) -> Grid:
        return g[::-1]

    def _overlay_non_bg(self, base: Grid, overlay: Grid, bg: int) -> Grid:
        H, W = _shape(base)
        result = _clone(base)
        for r in range(H):
            for c in range(W):
                if result[r][c] == bg and overlay[r][c] != bg:
                    result[r][c] = overlay[r][c]
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        for fn in (self._flip_h, self._flip_v):
            ok = True
            for p in pairs:
                inp, out = p["input"], p["output"]
                bg = _bg(inp)
                pred = self._overlay_non_bg(inp, fn(inp), bg)
                if pred != out:
                    ok = False
                    break
            if ok:
                results = []
                for test_in in ctx.test_inputs:
                    bg = _bg(test_in)
                    pred = self._overlay_non_bg(test_in, fn(test_in), bg)
                    results.append(pred)
                if any(r != t for r, t in zip(results, ctx.test_inputs)):
                    return _make_result_list(ctx, results, self.name)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 18. ColorFreqReplaceSolver
#     Replace the rarest-colored cells with the most-frequent color.
#     Generalizable: learns rank→target from training pairs.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ColorFreqReplaceSolver(BaseSolver):
    name = "ColorFreqReplaceSolver"
    priority = 34

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn: which color gets replaced by which
        rule = {}  # src_color → dst_color
        for p in pairs:
            inp, out = p["input"], p["output"]
            H, W = _shape(inp)
            for r in range(H):
                for c in range(W):
                    if inp[r][c] != out[r][c]:
                        src, dst = inp[r][c], out[r][c]
                        if src in rule and rule[src] != dst:
                            return None
                        rule[src] = dst

        if not rule:
            return None

        def apply_fn(g):
            H, W = _shape(g)
            return [[rule.get(g[r][c], g[r][c]) for c in range(W)] for r in range(H)]

        for p in pairs:
            if apply_fn(p["input"]) != p["output"]:
                return None

        results = [apply_fn(t) for t in ctx.test_inputs]
        if all(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 19. CellNeighborMajorityColorSolver
#     Each cell takes the majority color among its 4-neighbors.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class CellNeighborMajorityColorSolver(BaseSolver):
    name = "CellNeighborMajorityColorSolver"
    priority = 37

    def _apply(self, g: Grid) -> Grid:
        H, W = _shape(g)
        result = _clone(g)
        for r in range(H):
            for c in range(W):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        neighbors.append(g[nr][nc])
                if neighbors:
                    majority = Counter(neighbors).most_common(1)[0][0]
                    result[r][c] = majority
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None
            if self._apply(p["input"]) != p["output"]:
                return None

        results = [self._apply(t) for t in ctx.test_inputs]
        if all(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 20. FrameInteriorColorSolver
#     Detects frames (rectangular borders) and fills their interior with a
#     learned color (could be the frame color, or a different specific color).
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class FrameInteriorColorSolver(BaseSolver):
    name = "FrameInteriorColorSolver"
    priority = 39

    def _find_frames(self, g: Grid):
        """Return list of (r0, c0, r1, c1, frame_color)."""
        H, W = _shape(g)
        frames = []
        for r0 in range(H-2):
            for c0 in range(W-2):
                for r1 in range(r0+2, H):
                    for c1 in range(c0+2, W):
                        fc = g[r0][c0]
                        if fc == 0:
                            continue
                        top = all(g[r0][c] == fc for c in range(c0, c1+1))
                        bot = all(g[r1][c] == fc for c in range(c0, c1+1))
                        lft = all(g[r][c0] == fc for r in range(r0, r1+1))
                        rgt = all(g[r][c1] == fc for r in range(r0, r1+1))
                        if top and bot and lft and rgt:
                            frames.append((r0, c0, r1, c1, fc))
        return frames

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        fill_color = None
        for p in pairs:
            inp, out = p["input"], p["output"]
            frames = self._find_frames(inp)
            if not frames:
                return None
            for r0, c0, r1, c1, fc in frames:
                interior_out = [out[r][c] for r in range(r0+1, r1) for c in range(c0+1, c1)]
                if not interior_out:
                    continue
                interior_colors = set(interior_out)
                if len(interior_colors) == 1:
                    ic = list(interior_colors)[0]
                    if fill_color is None:
                        fill_color = ic
                    elif fill_color != ic:
                        return None
                else:
                    return None

        if fill_color is None:
            return None

        def apply_fn(g):
            frames = self._find_frames(g)
            result = _clone(g)
            for r0, c0, r1, c1, fc in frames:
                for r in range(r0+1, r1):
                    for c in range(c0+1, c1):
                        result[r][c] = fill_color
            return result

        for p in pairs:
            if apply_fn(p["input"]) != p["output"]:
                return None

        results = [apply_fn(t) for t in ctx.test_inputs]
        if all(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 21. RowColPatternInducerSolver
#     Induces a per-row or per-column operation that transforms input.
#     Tries: row rotation, column rotation, row/col permutation.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class RowColPatternInducerSolver(BaseSolver):
    name = "RowColPatternInducerSolver"
    priority = 41

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Try: each row in output is the input row rotated by some consistent amount
        _H0, _W0 = _shape(pairs[0]["input"])

        def try_row_rotation():
            row_shifts = []
            for p in pairs:
                inp, out = p["input"], p["output"]
                pH, pW = _shape(inp)
                if (pH, pW) != (_H0, _W0):
                    return None
                shifts = []
                for r in range(pH):
                    if r >= len(out):
                        return None
                    found = False
                    for shift in range(pW):
                        rotated = inp[r][shift:] + inp[r][:shift]
                        if rotated == out[r]:
                            shifts.append(shift)
                            found = True
                            break
                    if not found:
                        return None
                row_shifts.append(shifts)
            if len(row_shifts) < 1:
                return None
            ref = row_shifts[0]
            for rs in row_shifts[1:]:
                if rs != ref:
                    return None
            return ref

        shifts = try_row_rotation()
        if shifts:
            def apply_fn(g):
                result = []
                for r in range(len(g)):
                    shift = shifts[r] if r < len(shifts) else 0
                    result.append(g[r][shift:] + g[r][:shift])
                return result

            for p in pairs:
                if apply_fn(p["input"]) != p["output"]:
                    return None

            results = [apply_fn(t) for t in ctx.test_inputs]
            if any(r != t for r, t in zip(results, ctx.test_inputs)):
                return _make_result_list(ctx, results, self.name)

        return None


# ══════════════════════════════════════════════════════════════════════════════
# 22. ColorByAreaSolver
#     Assigns each connected component a color based on its area (cell count).
#     Learns: area_value → color, or: area_rank → color.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ColorByAreaSolver(BaseSolver):
    name = "ColorByAreaSolver"
    priority = 27

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # Same-size I/O required
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn: area → output_color mapping
        area_to_color: dict[int, int] = {}
        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            for comp in comps:
                area = len(comp)
                out_cols = set(out[r][c] for r, c in comp)
                if len(out_cols) != 1:
                    return None
                oc = list(out_cols)[0]
                if area in area_to_color:
                    if area_to_color[area] != oc:
                        return None
                else:
                    area_to_color[area] = oc

        if not area_to_color or all(k == v for k, v in area_to_color.items()):
            return None

        def apply_fn(g):
            bg = _bg(g)
            comps = _connected_components(g, bg)
            pred = _clone(g)
            for comp in comps:
                area = len(comp)
                col = area_to_color.get(area)
                if col is None:
                    return None
                for r, c in comp:
                    pred[r][c] = col
            return pred

        for p in pairs:
            pred = apply_fn(p["input"])
            if pred is None or pred != p["output"]:
                return None

        results = []
        for test_in in ctx.test_inputs:
            r = apply_fn(test_in)
            if r is None:
                return None
            results.append(r)
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 23. PerimeterRecolorSolver
#     Recolors the border (perimeter) cells of each connected object.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class PerimeterRecolorSolver(BaseSolver):
    name = "PerimeterRecolorSolver"
    priority = 43

    def _get_perimeter(self, g: Grid, comp: set) -> set:
        H, W = _shape(g)
        perimeter = set()
        for (r, c) in comp:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if not (0 <= nr < H and 0 <= nc < W) or g[nr][nc] != g[r][c]:
                    perimeter.add((r, c))
                    break
        return perimeter

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn: interior stays or perimeter changes
        perim_color = None
        interior_preserves = None

        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)
            comps = _connected_components(inp, bg)
            for comp in comps:
                if len(comp) < 4:
                    continue
                perim = self._get_perimeter(inp, comp)
                interior = comp - perim
                if not interior:
                    continue
                # Check if perimeter cells all become same color
                perim_out_colors = set(out[r][c] for r, c in perim)
                if len(perim_out_colors) != 1:
                    return None
                pc = list(perim_out_colors)[0]
                if perim_color is None:
                    perim_color = pc
                elif perim_color != pc:
                    return None
                # Interior stays unchanged
                interior_same = all(out[r][c] == inp[r][c] for r, c in interior)
                if interior_preserves is None:
                    interior_preserves = interior_same
                elif interior_preserves != interior_same:
                    return None

        if perim_color is None:
            return None

        def apply_fn(g):
            bg = _bg(g)
            comps = _connected_components(g, bg)
            pred = _clone(g)
            for comp in comps:
                perim = self._get_perimeter(g, comp)
                for r, c in perim:
                    pred[r][c] = perim_color
            return pred

        for p in pairs:
            if apply_fn(p["input"]) != p["output"]:
                return None

        results = [apply_fn(t) for t in ctx.test_inputs]
        if all(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 24. UniformFillSolver
#     All non-bg cells become the same target color (learned from training).
#     Simple but handles a whole class of recoloring tasks.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class UniformFillSolver(BaseSolver):
    name = "UniformFillSolver"
    priority = 44

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        # Learn target color
        target_color = None
        for p in pairs:
            inp, out = p["input"], p["output"]
            H, W = _shape(inp)
            bg = _bg(inp)
            out_fg = set(out[r][c] for r in range(H) for c in range(W) if inp[r][c] != bg)
            if len(out_fg) != 1:
                return None
            tc = list(out_fg)[0]
            if target_color is None:
                target_color = tc
            elif target_color != tc:
                return None

        if target_color is None:
            return None

        def apply_fn(g):
            H, W = _shape(g)
            bg = _bg(g)
            return [[target_color if g[r][c] != bg else bg for c in range(W)] for r in range(H)]

        for p in pairs:
            if apply_fn(p["input"]) != p["output"]:
                return None

        results = [apply_fn(t) for t in ctx.test_inputs]
        if all(r == t for r, t in zip(results, ctx.test_inputs)):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 25. CropNonBackgroundSolver
#     SHRINK bucket: output = bounding box of all non-background pixels.
#     Background = most-frequent color in input.
#     Handles rectangular and irregular foreground objects.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class CropNonBackgroundSolver(BaseSolver):
    name = "CropNonBackgroundSolver"
    priority = 50

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        def _crop_bbox(g: Grid) -> Optional[Grid]:
            H, W = _shape(g)
            bg = _bg(g)
            rows = [r for r in range(H) if any(g[r][c] != bg for c in range(W))]
            cols = [c for c in range(W) if any(g[r][c] != bg for r in range(H))]
            if not rows or not cols:
                return None
            r0, r1 = rows[0], rows[-1]
            c0, c1 = cols[0], cols[-1]
            if r0 == 0 and r1 == H - 1 and c0 == 0 and c1 == W - 1:
                return None  # no actual shrink
            return [g[r][c0:c1+1] for r in range(r0, r1+1)]

        # Validate on all training pairs
        for p in pairs:
            expected = _crop_bbox(p["input"])
            if expected is None:
                return None
            if expected != p["output"]:
                return None

        results = [_crop_bbox(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 26. ExtractFrameInteriorSolver
#     SHRINK bucket: the input contains a solid rectangular frame drawn in a
#     single learned border color.  Output = interior content (cells inside
#     the frame, excluding the frame row/column themselves).
#     Handles any border color; learns it from training pairs.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ExtractFrameInteriorSolver(BaseSolver):
    name = "ExtractFrameInteriorSolver"
    priority = 51

    @staticmethod
    def _find_unique_frame(g: Grid) -> Optional[tuple]:
        """Find the unique frame: a color whose ALL occurrences form exactly
        the rectangular border of some axis-aligned rectangle.
        Returns (border_color, r0, c0, r1, c1) or None."""
        H, W = _shape(g)
        from collections import defaultdict
        color_cells: dict = defaultdict(list)
        for r in range(H):
            for c in range(W):
                v = g[r][c]
                if v != 0:
                    color_cells[v].append((r, c))

        for color, cells in color_cells.items():
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            # Build the expected border set
            expected: set = set()
            for cc in range(c0, c1 + 1):
                expected.add((r0, cc))
                expected.add((r1, cc))
            for rr in range(r0 + 1, r1):
                expected.add((rr, c0))
                expected.add((rr, c1))
            # All occurrences of this color must form exactly this border
            if set(cells) == expected:
                return (color, r0, c0, r1, c1)
        return None

    @staticmethod
    def _interior(g: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
        return [list(g[r][c0 + 1:c1]) for r in range(r0 + 1, r1)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        for p in pairs:
            inp, out = p["input"], p["output"]
            oh, ow = _shape(out)
            ih, iw = _shape(inp)
            if oh >= ih or ow >= iw:
                return None  # output must be strictly smaller
            frame = self._find_unique_frame(inp)
            if frame is None:
                return None
            _, r0, c0, r1, c1 = frame
            if self._interior(inp, r0, c0, r1, c1) != out:
                return None

        results = []
        for t in ctx.test_inputs:
            frame = self._find_unique_frame(t)
            if frame is None:
                return None
            _, r0, c0, r1, c1 = frame
            results.append(self._interior(t, r0, c0, r1, c1))

        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 27. ConnectedComponentRelabelSolver
#     NEW_COLORS_APPEAR bucket: connected components of a target color are
#     ranked by descending pixel count; each rank maps to a distinct output
#     color.  Learns the rank→color mapping from training pairs.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class ConnectedComponentRelabelSolver(BaseSolver):
    name = "ConnectedComponentRelabelSolver"
    priority = 52

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs or len(pairs) < 1:
            return None

        # All training pairs must have same shape in→out
        for p in pairs:
            if _shape(p["input"]) != _shape(p["output"]):
                return None

        rank_to_color: Optional[dict] = None
        target_color: Optional[int] = None

        for p in pairs:
            inp, out = p["input"], p["output"]
            bg = _bg(inp)

            # Try each color present in the input (excluding background)
            for tc in _unique(inp) - {bg}:
                comps = _connected_components(inp, bg)
                tc_comps = [c for c in comps if all(inp[r][cc] == tc for r, cc in c)]
                if len(tc_comps) < 2:
                    continue

                # Rank by size descending
                tc_comps.sort(key=lambda c: len(c), reverse=True)

                # Build rank→output_color mapping
                mapping: dict[int, int] = {}
                valid = True
                for rank, comp in enumerate(tc_comps):
                    out_colors = {out[r][c] for r, c in comp}
                    if len(out_colors) != 1:
                        valid = False
                        break
                    mapping[rank] = out_colors.pop()
                if not valid:
                    continue

                # Check target_color consistency
                if target_color is None:
                    target_color = tc
                elif target_color != tc:
                    continue

                # Check rank_to_color consistency
                if rank_to_color is None:
                    rank_to_color = mapping
                elif rank_to_color != mapping:
                    valid = False

                if valid and rank_to_color is not None and target_color == tc:
                    break  # found valid tc for this pair
            else:
                return None  # no valid tc found for this pair

        if rank_to_color is None or target_color is None:
            return None

        def apply_fn(g: Grid) -> Optional[Grid]:
            bg = _bg(g)
            comps = _connected_components(g, bg)
            tc_comps = [c for c in comps if all(g[r][cc] == target_color for r, cc in c)]
            if not tc_comps:
                return None
            tc_comps.sort(key=lambda c: len(c), reverse=True)
            pred = _clone(g)
            for rank, comp in enumerate(tc_comps):
                col = rank_to_color.get(rank)
                if col is None:
                    return None
                for r, c in comp:
                    pred[r][c] = col
            return pred

        # Validate on training pairs
        for p in pairs:
            if apply_fn(p["input"]) != p["output"]:
                return None

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ══════════════════════════════════════════════════════════════════════════════
# 28. CrossExpansionSolver
#     Pattern (task 0962bcdd): background color (most freq) fills the grid.
#     One or more "cross" structures exist: a center seed_color pixel surrounded
#     on all 4 orthogonal sides by the same arm_color (forming a + shape).
#     The output expands each cross into a 5×5 star template:
#       centre + diagonals (±1,±1) + double-diagonals (±2,±2) → seed_color
#       orthogonals (±1,0),(0,±1) + double-orthogonals (±2,0),(0,±2) → arm_color
#     All other cells remain background.
# ══════════════════════════════════════════════════════════════════════════════

@register_solver
class CrossExpansionSolver(BaseSolver):
    name = "CrossExpansionSolver"
    priority = 53

    # Star template: diagonal offsets → seed_color
    DIAG_OFFSETS = [(0, 0),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                    (-2, -2), (-2, 2), (2, -2), (2, 2)]
    # Orthogonal offsets → arm_color
    ORTH_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-2, 0), (2, 0), (0, -2), (0, 2)]

    @staticmethod
    def _background(g: Grid) -> int:
        """Most-frequent color."""
        counts: dict[int, int] = {}
        for row in g:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def _find_crosses(self, g: Grid, bg: int) -> list[tuple[int, int, int, int]]:
        """Return (r, c, seed_color, arm_color) for each valid cross centre."""
        H, W = _shape(g)
        result = []
        for r in range(H):
            for c in range(W):
                cell = g[r][c]
                if cell == bg:
                    continue
                # All 4 orthogonal neighbours must be in-bounds
                nbrs = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
                if not all(0 <= nr < H and 0 <= nc < W for nr, nc in nbrs):
                    continue
                arm_set = {g[nr][nc] for nr, nc in nbrs}
                if len(arm_set) == 1:
                    arm_color = arm_set.pop()
                    if arm_color != bg and arm_color != cell:
                        result.append((r, c, cell, arm_color))
        return result

    def _apply_star(self, g: Grid,
                    crosses: list[tuple[int,int,int,int]]) -> Grid:
        H, W = _shape(g)
        bg = self._background(g)
        # Start with all-background
        pred = [[bg]*W for _ in range(H)]
        for r, c, seed_color, arm_color in crosses:
            for dr, dc in self.DIAG_OFFSETS:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    pred[nr][nc] = seed_color
            for dr, dc in self.ORTH_OFFSETS:
                nr, nc = r+dr, c+dc
                if 0 <= nr < H and 0 <= nc < W:
                    pred[nr][nc] = arm_color
        return pred

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        for p in pairs:
            inp, out = p["input"], p["output"]
            if _shape(inp) != _shape(out):
                return None

            bg = self._background(inp)
            crosses = self._find_crosses(inp, bg)
            if not crosses:
                return None

            pred = self._apply_star(inp, crosses)
            if pred != out:
                return None

        def apply_fn(g: Grid) -> Optional[Grid]:
            bg = self._background(g)
            crosses = self._find_crosses(g, bg)
            if not crosses:
                return None
            return self._apply_star(g, crosses)

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


@register_solver
# ---------------------------------------------------------------------------
# LargestSolidRectColorSolver
# Pattern: input is a noisy grid; one non-bg color forms the largest axis-
#          aligned solid (all-same-color) rectangle.  The output is a constant
#          fill of that color.  Tie-breaking: larger total cell count wins.
# Targets: 3194b014, 445eab21 (and similar)
# ---------------------------------------------------------------------------
class LargestSolidRectColorSolver(BaseSolver):
    """Find the non-bg color whose largest solid rectangle is maximal → constant-fill output."""
    name = "LargestSolidRectColorSolver"
    priority = 54

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _largest_solid_rect(grid: Grid, color: int) -> int:
        """Maximal-rectangle-in-histogram for cells matching *color*."""
        rows, cols = _shape(grid)
        hist = [0] * cols
        best = 0
        for r in range(rows):
            for c in range(cols):
                hist[c] = (hist[c] + 1) if grid[r][c] == color else 0
            # max rect in this histogram row
            stack: list[tuple[int, int]] = []
            for c in range(cols + 1):
                h = hist[c] if c < cols else 0
                start = c
                while stack and stack[-1][1] > h:
                    sc, sh = stack.pop()
                    best = max(best, sh * (c - sc))
                    start = sc
                stack.append((start, h))
        return best

    def _predict(self, inp: Grid) -> Optional[int]:
        bg = _bg(inp)
        from collections import Counter as _Counter
        cnt = _Counter(c for row in inp for c in row)
        non_bg = {c: cnt[c] for c in cnt if c != bg}
        if not non_bg:
            return None
        areas = {c: self._largest_solid_rect(inp, c) for c in non_bg}
        max_area = max(areas.values())
        if max_area == 0:
            return None
        candidates = [c for c, a in areas.items() if a == max_area]
        return max(candidates, key=lambda c: cnt[c])   # tie-break: most cells

    # ------------------------------------------------------------------ solve
    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # All training output shapes must be identical (guards variable-output tasks)
        out_shapes = [_shape(p["output"]) for p in pairs]
        if len(set(out_shapes)) != 1:
            return None
        out_H, out_W = out_shapes[0]

        for p in pairs:
            inp, out = p["input"], p["output"]
            # Output must be constant fill (all same color)
            flat_out = set(c for row in out for c in row)
            if len(flat_out) != 1:
                return None
            expected = next(iter(flat_out))
            pred = self._predict(inp)
            if pred != expected:
                return None

        def apply_fn(g: Grid) -> Optional[Grid]:
            color = self._predict(g)
            if color is None:
                return None
            return [[color] * out_W for _ in range(out_H)]

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


@register_solver
# ---------------------------------------------------------------------------
# LargerFrameColorSolver
# Pattern: input contains exactly 2 non-bg colors, each forming a hollow
#          rectangular frame (border only).  Output = constant fill of the
#          color whose frame has the larger interior area.
# Targets: 445eab21
# ---------------------------------------------------------------------------
class LargerFrameColorSolver(BaseSolver):
    """Two hollow frames → output color of larger-interior frame."""
    name = "LargerFrameColorSolver"
    priority = 53

    @staticmethod
    def _frame_interior(grid: Grid, color: int) -> int:
        """Return interior area if *color* forms a perfect hollow rectangle, else -1."""
        rows, cols = _shape(grid)
        cells = [(r, c) for r in range(rows) for c in range(cols)
                 if grid[r][c] == color]
        if not cells:
            return -1
        rr = [p[0] for p in cells]
        cc = [p[1] for p in cells]
        min_r, max_r = min(rr), max(rr)
        min_c, max_c = min(cc), max(cc)
        h, w = max_r - min_r + 1, max_c - min_c + 1
        if h < 2 or w < 2:
            return -1
        # Build expected perimeter cell set
        perimeter: set[tuple[int, int]] = set()
        for c in range(min_c, max_c + 1):
            perimeter.add((min_r, c))
            perimeter.add((max_r, c))
        for r in range(min_r, max_r + 1):
            perimeter.add((r, min_c))
            perimeter.add((r, max_c))
        if set(cells) != perimeter:
            return -1
        return max(0, (h - 2) * (w - 2))

    def _predict(self, inp: Grid) -> Optional[int]:
        bg = _bg(inp)
        from collections import Counter as _Counter
        non_bg_colors = [c for c in set(v for row in inp for v in row) if c != bg]
        areas: dict[int, int] = {}
        for color in non_bg_colors:
            ia = self._frame_interior(inp, color)
            if ia < 0:
                return None          # not a clean frame → task doesn't match
            areas[color] = ia
        if len(areas) < 2:
            return None
        return max(areas, key=lambda c: areas[c])

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        for p in pairs:
            inp, out = p["input"], p["output"]
            flat_out = set(v for row in out for v in row)
            if len(flat_out) != 1:
                return None
            expected = next(iter(flat_out))
            pred = self._predict(inp)
            if pred is None or pred != expected:
                return None

        out_H, out_W = _shape(pairs[0]["output"])

        def apply_fn(g: Grid) -> Optional[Grid]:
            color = self._predict(g)
            if color is None:
                return None
            return [[color] * out_W for _ in range(out_H)]

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


@register_solver
# ---------------------------------------------------------------------------
# CountCellsToRowSolver
# Pattern: 3x3 (or small) input has N non-background cells all of same color.
#          Output = 1×N constant-fill row of that color.
# Targets: d631b094
# ---------------------------------------------------------------------------
class CountCellsToRowSolver(BaseSolver):
    """Count non-bg cells → output 1×N row of that color."""
    name = "CountCellsToRowSolver"
    priority = 52

    @staticmethod
    def _predict_with_bg(inp: Grid, bg: int) -> Optional[tuple[int, int]]:
        """Return (color, count) given explicit bg, or None."""
        non_bg = [inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))
                  if inp[r][c] != bg]
        if not non_bg:
            return None
        colors = set(non_bg)
        if len(colors) != 1:
            return None  # multiple non-bg colors → different task
        return (non_bg[0], len(non_bg))

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # Determine consistent background from training inputs
        bg = _bg(pairs[0]["input"])
        for p in pairs:
            if _bg(p["input"]) != bg:
                return None  # inconsistent bg across training

        for p in pairs:
            inp, out = p["input"], p["output"]
            result = self._predict_with_bg(inp, bg)
            if result is None:
                return None
            color, count = result
            # Output must be 1×count row of that color
            if len(out) != 1 or len(out[0]) != count:
                return None
            if any(v != color for v in out[0]):
                return None

        def apply_fn(g: Grid) -> Optional[Grid]:
            result = self._predict_with_bg(g, bg)
            if result is None:
                return None
            color, count = result
            return [[color] * count]

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ---------------------------------------------------------------------------
# ShapeLookupSolver — 27a28665 pattern
# 3×3 binary shape (non-bg=1, bg=0 as 9-tuple) → fixed output value from training
# ---------------------------------------------------------------------------
@register_solver
class ShapeLookupSolver(BaseSolver):
    """Map binary shape tuple from input grid → single output color (1×1 output).
    
    Pattern: each training pair has a small grid; compute bg, binarise all cells
    (non-bg→1, bg→0), build a tuple key, map to the single output cell value.
    Apply the same lookup to each test input.
    """
    name = "ShapeLookupSolver"
    priority = 55

    @staticmethod
    def _grid_shape_key(g: Grid, bg: int) -> tuple:
        return tuple(0 if v == bg else 1 for row in g for v in row)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # All outputs must be 1×1
        for p in pairs:
            if len(p["output"]) != 1 or len(p["output"][0]) != 1:
                return None

        # All inputs must have the same shape
        H0, W0 = len(pairs[0]["input"]), len(pairs[0]["input"][0])
        if any(len(p["input"]) != H0 or len(p["input"][0]) != W0 for p in pairs):
            return None

        # Build lookup from training pairs (per-pair bg detection)
        lookup: dict[tuple, int] = {}
        for p in pairs:
            bg = _bg(p["input"])
            key = self._grid_shape_key(p["input"], bg)
            val = p["output"][0][0]
            if key in lookup and lookup[key] != val:
                return None  # ambiguous mapping
            lookup[key] = val

        def apply_fn(g: Grid) -> Optional[Grid]:
            if len(g) != H0 or len(g[0]) != W0:
                return None
            bg = _bg(g)
            key = self._grid_shape_key(g, bg)
            if key not in lookup:
                return None
            return [[lookup[key]]]

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ---------------------------------------------------------------------------
# SingleComponentColorSolver — 1a2e2828 pattern
# Output = unique non-bg color that has exactly 1 connected component (4-conn)
# All other non-bg colors must have 2+ components. Output is 1×1.
# ---------------------------------------------------------------------------
@register_solver
class SingleComponentColorSolver(BaseSolver):
    """Find the unique non-bg color with exactly 1 connected component; output as 1×1."""
    name = "SingleComponentColorSolver"
    priority = 55

    @staticmethod
    def _count_components(g: Grid, color: int) -> int:
        rows, cols = len(g), len(g[0])
        visited = [[False]*cols for _ in range(rows)]
        count = 0
        for sr in range(rows):
            for sc in range(cols):
                if g[sr][sc] == color and not visited[sr][sc]:
                    count += 1
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if visited[r][c]:
                            continue
                        visited[r][c] = True
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] == color and not visited[nr][nc]:
                                stack.append((nr, nc))
        return count

    def _find_single_comp_color(self, g: Grid) -> Optional[int]:
        bg = _bg(g)
        rows, cols = len(g), len(g[0])
        colors = set(g[r][c] for r in range(rows) for c in range(cols)) - {bg}
        candidates = []
        for c in colors:
            if self._count_components(g, c) == 1:
                candidates.append(c)
        return candidates[0] if len(candidates) == 1 else None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None

        # All outputs must be 1×1
        for p in pairs:
            if len(p["output"]) != 1 or len(p["output"][0]) != 1:
                return None

        # Validate training pairs
        for p in pairs:
            found = self._find_single_comp_color(p["input"])
            if found is None:
                return None
            if found != p["output"][0][0]:
                return None

        def apply_fn(g: Grid) -> Optional[Grid]:
            c = self._find_single_comp_color(g)
            if c is None:
                return None
            return [[c]]

        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ---------------------------------------------------------------------------
# NoisyRegionColorSolver — de1cd16c pattern
# Grid = several solid-color regions + one "noise" color (many scattered single pixels)
# Output = the region that contains the most noise pixels (1×1 output)
# ---------------------------------------------------------------------------
@register_solver
class NoisyRegionColorSolver(BaseSolver):
    """Identify the solid region containing the most noise pixels; output its color (1×1).

    Pattern:
    - One color appears as many scattered single-pixel components (noise).
    - All other colors each form exactly 1 large connected component (solid region).
    - For each noise pixel, assign it to the surrounding solid-region color via neighbors.
    - Output = the solid region that received the most noise pixels.
    """
    name = "NoisyRegionColorSolver"
    priority = 55

    @staticmethod
    def _count_comps(g: Grid, color: int) -> int:
        rows, cols = len(g), len(g[0])
        vis = [[False]*cols for _ in range(rows)]
        comps = 0
        for sr in range(rows):
            for sc in range(cols):
                if g[sr][sc] == color and not vis[sr][sc]:
                    comps += 1
                    stack = [(sr, sc)]
                    while stack:
                        r, c = stack.pop()
                        if vis[r][c]: continue
                        vis[r][c] = True
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0<=nr<rows and 0<=nc<cols and g[nr][nc]==color and not vis[nr][nc]:
                                stack.append((nr, nc))
        return comps

    def _find_noise_region(self, g: Grid) -> Optional[int]:
        rows, cols = len(g), len(g[0])
        all_colors = set(v for row in g for v in row)
        if len(all_colors) < 3:
            return None
        comps = {c: self._count_comps(g, c) for c in all_colors}
        # noise color = most fragmented (highest component count, must be > 1)
        noise_color = max(all_colors, key=lambda c: comps[c])
        if comps[noise_color] <= 1:
            return None
        # all other colors must form exactly 1 component
        regions = [c for c in all_colors if c != noise_color and comps[c] == 1]
        if not regions:
            return None
        # assign each noise pixel to surrounding region via neighbor vote
        region_counts: dict[int, int] = {}
        for r in range(rows):
            for c in range(cols):
                if g[r][c] == noise_color:
                    nbrs = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<rows and 0<=nc<cols and g[nr][nc] != noise_color:
                            nbrs.append(g[nr][nc])
                    if nbrs:
                        region = max(set(nbrs), key=nbrs.count)
                        region_counts[region] = region_counts.get(region, 0) + 1
        if not region_counts:
            return None
        return max(region_counts, key=region_counts.get)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # All outputs must be 1×1
        for p in pairs:
            if len(p["output"]) != 1 or len(p["output"][0]) != 1:
                return None
        # Validate training pairs
        for p in pairs:
            found = self._find_noise_region(p["input"])
            if found is None or found != p["output"][0][0]:
                return None
        # Apply to test
        def apply_fn(g: Grid) -> Optional[Grid]:
            c = self._find_noise_region(g)
            return [[c]] if c is not None else None
        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# RarestColorBBoxCropSolver
# Rule: crop the input to the bounding box of the rarest (least-frequent)
#       non-background color.  Covers tasks where a small-area "target" object
#       defines the output region (e.g. 23b5c85d, c909285e, 0b148d64).
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class RarestColorBBoxCropSolver(BaseSolver):
    """
    Output = rectangular crop of input at the bounding box of the rarest
    non-background color.
    """
    name = "RarestColorBBoxCropSolver"
    priority = 56

    @staticmethod
    def _rarest_bbox(g: Grid) -> Optional[tuple]:
        rows, cols = len(g), len(g[0])
        flat = [v for row in g for v in row]
        counts: dict[int, int] = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        non_bg = [(c, ct) for c, ct in counts.items() if c != bg]
        if not non_bg:
            return None
        rarest_color = min(non_bg, key=lambda x: x[1])[0]
        rrows = [r for r in range(rows) if any(g[r][c] == rarest_color for c in range(cols))]
        rcols = [c for c in range(cols) if any(g[r][c] == rarest_color for r in range(rows))]
        if not rrows or not rcols:
            return None
        return rrows[0], rcols[0], rrows[-1], rcols[-1]

    def _validate(self, inp: Grid, out: Grid) -> bool:
        bbox = self._rarest_bbox(inp)
        if bbox is None:
            return False
        r0, c0, r1, c1 = bbox
        crop = [inp[r][c0:c1+1] for r in range(r0, r1+1)]
        return crop == out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if not self._validate(p["input"], p["output"]):
                return None
        def apply_fn(g: Grid) -> Optional[Grid]:
            bbox = self._rarest_bbox(g)
            if bbox is None:
                return None
            r0, c0, r1, c1 = bbox
            return [g[r][c0:c1+1] for r in range(r0, r1+1)]
        results = [apply_fn(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# ColorsFreqDescSolver
# Rule: output is a single-column grid listing all non-background colors
#       sorted by frequency descending.  Covers f8ff0b80 and similar tasks.
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class ColorsFreqDescSolver(BaseSolver):
    """
    Output = Nx1 column of all non-bg colors sorted by frequency descending.
    """
    name = "ColorsFreqDescSolver"
    priority = 55

    @staticmethod
    def _predict(g: Grid) -> Optional[Grid]:
        flat = [v for row in g for v in row]
        counts: dict[int, int] = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        non_bg = sorted([(c, ct) for c, ct in counts.items() if c != bg],
                        key=lambda x: -x[1])
        if not non_bg:
            return None
        return [[c] for c, _ in non_bg]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # All outputs must be 1 column
        for p in pairs:
            if any(len(row) != 1 for row in p["output"]):
                return None
        # Validate training pairs
        for p in pairs:
            pred = self._predict(p["input"])
            if pred != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# ColorsExclMaxFreqSolver
# Rule: same as ColorsFreqDescSolver but EXCLUDING the most frequent non-bg
#       color.  Covers f8b3ba0a: 4 non-bg colors → output lists the 3 less-
#       frequent ones in descending frequency order.
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class ColorsExclMaxFreqSolver(BaseSolver):
    """
    Output = Nx1 column of non-bg colors sorted by frequency descending,
    with the most-frequent non-bg color excluded.
    """
    name = "ColorsExclMaxFreqSolver"
    priority = 55

    @staticmethod
    def _predict(g: Grid) -> Optional[Grid]:
        flat = [v for row in g for v in row]
        counts: dict[int, int] = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        non_bg = sorted([(c, ct) for c, ct in counts.items() if c != bg],
                        key=lambda x: -x[1])
        if len(non_bg) < 2:
            return None
        # Skip first (most frequent)
        remaining = non_bg[1:]
        return [[c] for c, _ in remaining]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        # All outputs must be 1 column
        for p in pairs:
            if any(len(row) != 1 for row in p["output"]):
                return None
        # Validate training pairs
        for p in pairs:
            pred = self._predict(p["input"])
            if pred != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# TwoBlockConnectedSolver
# Rule: Two distinct non-overlapping 2×2 blocks of the same non-bg color
#       (block_color) exist; a second non-bg color (bridge_color) forms
#       scattered cells. BFS from block_A through bridge_color cells; if
#       block_B is reachable (a bridge cell is adjacent to block_B) →
#       output = [[bridge_color]], else output = [[bg]].
#       Generalizes over any color pair, not just 2 and 8.
# Targets: 239be575
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class TwoBlockConnectedSolver(BaseSolver):
    """
    Output = [[bridge_color]] if two 2×2 blocks are connected via bridge cells,
    else [[bg]].
    """
    name = "TwoBlockConnectedSolver"
    priority = 57

    @staticmethod
    def _find_2x2_blocks(g: Grid, color: int) -> list:
        rows, cols = len(g), len(g[0])
        blocks = []
        used: set = set()
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (g[r][c] == color and g[r+1][c] == color and
                        g[r][c+1] == color and g[r+1][c+1] == color):
                    cells = frozenset([(r,c),(r+1,c),(r,c+1),(r+1,c+1)])
                    if not cells & used:
                        blocks.append((r, c))
                        used |= cells
        return blocks

    @staticmethod
    def _bfs_connected(g: Grid, block_a_cells: list, bridge_color: int,
                       block_b_cells: set) -> bool:
        rows, cols = len(g), len(g[0])
        visited: set = set()
        queue = []
        # Seed: bridge_color neighbors of block_A
        for r, c in block_a_cells:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < rows and 0 <= nc < cols and
                        g[nr][nc] == bridge_color and (nr,nc) not in visited):
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        idx = 0
        while idx < len(queue):
            r, c = queue[idx]; idx += 1
            # Check adjacency to block_B
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                if (r+dr, c+dc) in block_b_cells:
                    return True
            # Expand bridge
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if (0 <= nr < rows and 0 <= nc < cols and
                        g[nr][nc] == bridge_color and (nr,nc) not in visited):
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        return False

    def _predict(self, g: Grid) -> Optional[Grid]:
        flat = [v for row in g for v in row]
        counts: dict = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        non_bg = [c for c in counts if c != bg]
        if len(non_bg) != 2:
            return None
        # Find which color forms exactly 2 non-overlapping 2×2 blocks
        block_color = bridge_color = None
        blocks = []
        for color in non_bg:
            found = self._find_2x2_blocks(g, color)
            if len(found) == 2:
                block_color = color
                blocks = found
                bridge_color = [c for c in non_bg if c != color][0]
                break
        if block_color is None:
            return None
        r0, c0 = blocks[0]
        r1, c1 = blocks[1]
        block_a = [(r0,c0),(r0+1,c0),(r0,c0+1),(r0+1,c0+1)]
        block_b = {(r1,c1),(r1+1,c1),(r1,c1+1),(r1+1,c1+1)}
        connected = self._bfs_connected(g, block_a, bridge_color, block_b)
        return [[bridge_color]] if connected else [[bg]]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if len(p["output"]) != 1 or len(p["output"][0]) != 1:
                return None
        for p in pairs:
            if self._predict(p["input"]) != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# CrossCenterArmModeSolver
# Rule: Find all interior plus/cross patterns where a non-bg center cell has
#       all 4 orthogonal neighbors of the same non-bg arm color (≠ center).
#       Collect one arm_color per cross; output = [[mode(arm_colors)]].
# Targets: 642d658d
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class CrossCenterArmModeSolver(BaseSolver):
    """
    Output = [[most-common arm color across all cross patterns]].
    """
    name = "CrossCenterArmModeSolver"
    priority = 57

    @staticmethod
    def _find_arm_colors(g: Grid) -> Optional[list]:
        rows, cols = len(g), len(g[0])
        flat = [v for row in g for v in row]
        counts: dict = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        arm_colors = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                center = g[r][c]
                if center == bg:
                    continue
                arms = [g[r-1][c], g[r+1][c], g[r][c-1], g[r][c+1]]
                # All 4 arms must be same non-bg color, different from center
                if len(set(arms)) == 1 and arms[0] != bg and arms[0] != center:
                    arm_colors.append(arms[0])
        return arm_colors if arm_colors else None

    @staticmethod
    def _mode(lst: list) -> int:
        counts: dict = {}
        for v in lst:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)

    def _predict(self, g: Grid) -> Optional[Grid]:
        arm_colors = self._find_arm_colors(g)
        if not arm_colors:
            return None
        return [[self._mode(arm_colors)]]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if len(p["output"]) != 1 or len(p["output"][0]) != 1:
                return None
        for p in pairs:
            if self._predict(p["input"]) != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# GridLineBandCountSolver
# Rule: Input has full-row and full-col "lines" of a non-bg line_color dividing
#       the grid into a rectangular array of bands. Count non-zero-width
#       horizontal bands (H) and vertical bands (W).
#       Output = H×W grid filled with bg.
# Targets: 7039b2d7
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class GridLineBandCountSolver(BaseSolver):
    """
    Output = H×W grid of bg where H=row-band count, W=col-band count.
    """
    name = "GridLineBandCountSolver"
    priority = 57

    @staticmethod
    def _analyze(g: Grid) -> Optional[tuple]:
        rows, cols = len(g), len(g[0])
        flat = [v for row in g for v in row]
        counts: dict = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        # Full-row colors
        full_row_colors: set = set()
        for r in range(rows):
            row_set = set(g[r][c] for c in range(cols))
            if len(row_set) == 1 and list(row_set)[0] != bg:
                full_row_colors.add(list(row_set)[0])
        # Full-col colors
        full_col_colors: set = set()
        for c in range(cols):
            col_set = set(g[r][c] for r in range(rows))
            if len(col_set) == 1 and list(col_set)[0] != bg:
                full_col_colors.add(list(col_set)[0])
        line_colors = full_row_colors & full_col_colors
        if not line_colors:
            return None
        line_color = next(iter(line_colors))
        line_rows = sorted(r for r in range(rows)
                           if all(g[r][c] == line_color for c in range(cols)))
        line_cols = sorted(c for c in range(cols)
                           if all(g[r][c] == line_color for r in range(rows)))
        # Count non-zero-width bands
        boundaries_r = [-1] + line_rows + [rows]
        H = sum(1 for i in range(len(boundaries_r)-1)
                if boundaries_r[i+1] - boundaries_r[i] > 1)
        boundaries_c = [-1] + line_cols + [cols]
        W = sum(1 for i in range(len(boundaries_c)-1)
                if boundaries_c[i+1] - boundaries_c[i] > 1)
        if H == 0 or W == 0:
            return None
        return H, W, bg

    def _predict(self, g: Grid) -> Optional[Grid]:
        result = self._analyze(g)
        if result is None:
            return None
        H, W, bg = result
        return [[bg] * W for _ in range(H)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if self._predict(p["input"]) != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# TipCountHalfSolver
# Rule: Count cells of the primary non-bg color with exactly one same-color
#       4-connected neighbor (tips). N = tips // 2.
#       Output = N×1 grid of bg (background color).
# Targets: e872b94a
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class TipCountHalfSolver(BaseSolver):
    """
    Output = Nx1 grid of bg where N = (tip_count) // 2.
    """
    name = "TipCountHalfSolver"
    priority = 56

    @staticmethod
    def _count_tips(g: Grid, target_color: int) -> int:
        rows, cols = len(g), len(g[0])
        tips = 0
        for r in range(rows):
            for c in range(cols):
                if g[r][c] != target_color:
                    continue
                same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                           if 0 <= r+dr < rows and 0 <= c+dc < cols
                           and g[r+dr][c+dc] == target_color)
                if same == 1:
                    tips += 1
        return tips

    def _predict(self, g: Grid) -> Optional[Grid]:
        flat = [v for row in g for v in row]
        counts: dict = {}
        for v in flat:
            counts[v] = counts.get(v, 0) + 1
        bg = max(counts, key=counts.get)
        non_bg = [c for c in counts if c != bg]
        if not non_bg:
            return None
        target_color = max(non_bg, key=lambda c: counts[c])
        tips = self._count_tips(g, target_color)
        if tips == 0 or tips % 2 != 0:
            return None
        # Each connected path has 2 tips; N = num_paths + 1
        N = tips // 2 + 1
        return [[bg] for _ in range(N)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        pairs = ctx.train_pairs
        if not pairs:
            return None
        for p in pairs:
            if any(len(row) != 1 for row in p["output"]):
                return None
        for p in pairs:
            if self._predict(p["input"]) != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ---------------------------------------------------------------------------
# LegendFrameNoiseSolver
# Tasks: ff2825db
# 10×10 grid; row-0 = legend [1,1,2,2,3,3,4,4,5,5]; rows 1-9 have outer
# rectangular frame (frame_color); interior (rows 2-8, cols 1-8) has scattered
# noise cells of ≥1 non-bg non-frame color.
# Rule:
#   1. Dominant noise color = noise color with most cells.
#   2. Output row 0: legend preserved.
#   3. Output outer frame → dominant color.
#   4. Output inner rectangular FRAME drawn at bounding-box of dominant noise cells.
#   5. All other interior cells = 0.
# ---------------------------------------------------------------------------
@register_solver
class LegendFrameNoiseSolver(BaseSolver):
    """Legend row + outer frame + interior noise → replace frame and draw inner bbox frame."""

    name = "LegendFrameNoiseSolver"

    def _is_legend_row(self, row: list) -> bool:
        return row == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    def _check_grid(self, inp: Grid) -> bool:
        if len(inp) != 10 or any(len(r) != 10 for r in inp):
            return False
        if not self._is_legend_row(inp[0]):
            return False
        fc = inp[1][0]
        if fc == 0:
            return False
        # Row 1 all fc
        if any(v != fc for v in inp[1]):
            return False
        # Row 9 all fc
        if any(v != fc for v in inp[9]):
            return False
        # Col 0 rows 1-9 all fc
        if any(inp[r][0] != fc for r in range(1, 10)):
            return False
        # Col 9 rows 1-9 all fc
        if any(inp[r][9] != fc for r in range(1, 10)):
            return False
        return True

    def _predict(self, inp: Grid) -> Optional[Grid]:
        if not self._check_grid(inp):
            return None
        fc = inp[1][0]
        # Collect noise cells in interior
        noise: dict = {}
        for r in range(2, 9):
            for c in range(1, 9):
                v = inp[r][c]
                if v != 0 and v != fc:
                    noise.setdefault(v, []).append((r, c))
        if not noise:
            return None
        dom = max(noise, key=lambda k: len(noise[k]))
        cells = noise[dom]
        min_r = min(p[0] for p in cells)
        max_r = max(p[0] for p in cells)
        min_c = min(p[1] for p in cells)
        max_c = max(p[1] for p in cells)

        out = [[0] * 10 for _ in range(10)]
        # Legend row preserved
        out[0] = list(inp[0])
        # Outer frame → dom
        for c in range(10):
            out[1][c] = dom
            out[9][c] = dom
        for r in range(1, 10):
            out[r][0] = dom
            out[r][9] = dom
        # Inner rectangular border at bbox
        for c in range(min_c, max_c + 1):
            out[min_r][c] = dom
            out[max_r][c] = dom
        for r in range(min_r, max_r + 1):
            out[r][min_c] = dom
            out[r][max_c] = dom
        return out

    def can_solve(self, ctx: TaskContext) -> bool:
        return all(self._check_grid(p["input"]) for p in ctx.train_pairs)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            if self._predict(p["input"]) != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)


# ---------------------------------------------------------------------------
# NestedFrameTopoSolver
# Tasks: 68bc2e87
# 18×19 grids with overlapping rectangular frames (bg=8).
# Rule: at each cell where two frame-boundary paths cross, the color shown
# is the "later" frame (drawn on top). Build a directed graph from these
# intersection wins and topological-sort to get the output frame order.
# Output: Nx1 list of frame colors in topo order (outer→inner).
# ---------------------------------------------------------------------------
@register_solver
class NestedFrameTopoSolver(BaseSolver):
    """Overlapping rect frames: topological sort by boundary intersection wins → Nx1 output."""

    name = "NestedFrameTopoSolver"

    def _get_bbox(self, grid, color):
        rows, cols = [], []
        for r, row in enumerate(grid):
            for c, v in enumerate(row):
                if v == color:
                    rows.append(r); cols.append(c)
        if not rows:
            return None
        return min(rows), max(rows), min(cols), max(cols)

    def _on_boundary(self, r, c, bbox) -> bool:
        r0, r1, c0, c1 = bbox
        return ((r == r0 or r == r1) and c0 <= c <= c1) or \
               ((c == c0 or c == c1) and r0 <= r <= r1)

    def _predict(self, inp: Grid) -> Optional[Grid]:
        bg = 8  # fixed bg for this task family
        colors = sorted(set(v for row in inp for v in row) - {bg})
        if len(colors) < 2:
            return None
        bboxes = {c: self._get_bbox(inp, c) for c in colors}
        # Build directed edges: loser → winner at each boundary intersection
        adj: dict = {c: [] for c in colors}
        in_deg: dict = {c: 0 for c in colors}
        seen_pairs: set = set()
        for i, ci in enumerate(colors):
            for cj in colors[i + 1:]:
                if (ci, cj) in seen_pairs:
                    continue
                bi, bj = bboxes[ci], bboxes[cj]
                if bi is None or bj is None:
                    continue
                for r in range(len(inp)):
                    for c in range(len(inp[0])):
                        if self._on_boundary(r, c, bi) and self._on_boundary(r, c, bj):
                            winner = inp[r][c]
                            if winner not in (ci, cj):
                                continue
                            loser = cj if winner == ci else ci
                            pair_key = (loser, winner)
                            if pair_key not in seen_pairs:
                                seen_pairs.add(pair_key)
                                adj[loser].append(winner)
                                in_deg[winner] = in_deg.get(winner, 0) + 1
                            break  # one intersection per pair is enough
        # Topological sort (Kahn's algorithm)
        from collections import deque
        q: deque = deque(c for c in colors if in_deg.get(c, 0) == 0)
        order: list = []
        while q:
            node = q.popleft()
            order.append(node)
            for nb in adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0:
                    q.append(nb)
        if len(order) != len(colors):
            return None  # cycle detected
        return [[c] for c in order]

    def can_solve(self, ctx: TaskContext) -> bool:
        # All outputs must be Nx1 and all inputs same size
        for p in ctx.train_pairs:
            if any(len(row) != 1 for row in p["output"]):
                return False
        return True

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            if self._predict(p["input"]) != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)

# ---------------------------------------------------------------------------
# DiagonalExtendEightSolver — targets task 22233c11
#
# Rule:
#   Color-3 cells form 4-connected components that appear in pairs arranged
#   diagonally (TL+BR, "main diagonal") or anti-diagonally (TR+BL).
#   Color 8 is placed at two "extension" block positions derived from each pair:
#
#   Sort all components by (min_row, min_col); pair consecutively (0+1),(2+3)...
#   G1 = first in pair (lower min_row).
#   h = G1 height = max_row-min_row+1,  w = G1 width = max_col-min_col+1
#
#   Main diagonal  (G1.min_col < G2.min_col):
#     8A block rows = [G1_min_r - h .. G1_max_r - h], cols = [G2_min_c + w .. G2_max_c + w]
#     8B block rows = [G2_min_r + h .. G2_max_r + h], cols = [G1_min_c - w .. G1_max_c - w]
#
#   Anti-diagonal  (G1.min_col >= G2.min_col):
#     8A block rows = [G1_min_r - h .. G1_max_r - h], cols = [G2_min_c - w .. G2_max_c - w]
#     8B block rows = [G2_min_r + h .. G2_max_r + h], cols = [G1_min_c + w .. G1_max_c + w]
#
#   All out-of-bounds cells are simply skipped (clipped).
# ---------------------------------------------------------------------------
@register_solver
class DiagonalExtendEightSolver(BaseSolver):
    """Color-3 group pairs → extend diagonally with color 8 blocks (task 22233c11)."""

    name = "DiagonalExtendEightSolver"

    # ------------------------------------------------------------------ helpers
    def _find_components(self, grid: Grid) -> list:
        """Return list of 4-conn components of color 3 (each a list of (r,c))."""
        R, C = len(grid), len(grid[0])
        visited = [[False] * C for _ in range(R)]
        comps = []
        for r in range(R):
            for c in range(C):
                if grid[r][c] == 3 and not visited[r][c]:
                    comp, queue = [], [(r, c)]
                    visited[r][c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        comp.append((cr, cc))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == 3:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                    comps.append(comp)
        return comps

    def _place_block(self, out: Grid, r0: int, r1: int, c0: int, c1: int) -> None:
        R, C = len(out), len(out[0])
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= r < R and 0 <= c < C:
                    out[r][c] = 8

    def _predict(self, inp: Grid) -> Optional[Grid]:
        comps = self._find_components(inp)
        if len(comps) == 0 or len(comps) % 2 != 0:
            return None

        comps.sort(key=lambda comp: (min(r for r, c in comp), min(c for r, c in comp)))

        out = [row[:] for row in inp]

        for i in range(0, len(comps), 2):
            G1, G2 = comps[i], comps[i + 1]

            g1r = [r for r, c in G1];  g1c = [c for r, c in G1]
            g2r = [r for r, c in G2];  g2c = [c for r, c in G2]

            g1_min_r, g1_max_r = min(g1r), max(g1r)
            g1_min_c, g1_max_c = min(g1c), max(g1c)
            g2_min_r, g2_max_r = min(g2r), max(g2r)
            g2_min_c, g2_max_c = min(g2c), max(g2c)

            h = g1_max_r - g1_min_r + 1
            w = g1_max_c - g1_min_c + 1

            if g1_min_c < g2_min_c:
                # Main diagonal: G1=TL, G2=BR
                # 8A: G1-shifted up, G2-shifted right
                self._place_block(out, g1_min_r - h, g1_max_r - h, g2_min_c + w, g2_max_c + w)
                # 8B: G2-shifted down, G1-shifted left
                self._place_block(out, g2_min_r + h, g2_max_r + h, g1_min_c - w, g1_max_c - w)
            else:
                # Anti-diagonal: G1=TR, G2=BL
                # 8A: G1-shifted up, G2-shifted left
                self._place_block(out, g1_min_r - h, g1_max_r - h, g2_min_c - w, g2_max_c - w)
                # 8B: G2-shifted down, G1-shifted right
                self._place_block(out, g2_min_r + h, g2_max_r + h, g1_min_c + w, g1_max_c + w)

        return out

    # ------------------------------------------------------------------ interface
    def can_solve(self, ctx: TaskContext) -> bool:
        for p in ctx.train_pairs:
            inp, out = p["input"], p["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return False
            if not any(inp[r][c] == 3 for r in range(len(inp)) for c in range(len(inp[0]))):
                return False
            if not any(out[r][c] == 8 for r in range(len(out)) for c in range(len(out[0]))):
                return False
        return True

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate on all training pairs first
        for p in ctx.train_pairs:
            pred = self._predict(p["input"])
            if pred is None or pred != p["output"]:
                return None
        results = [self._predict(t) for t in ctx.test_inputs]
        if any(r is None for r in results):
            return None
        return _make_result_list(ctx, results, self.name)
