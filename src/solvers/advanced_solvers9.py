"""advanced_solvers9.py — 5 solvers from pattern analysis of diag_sym, hollow_obj, fill_holes sub-categories.

Solvers:
1. ConcentricRingReversalSolver  — reverse the dist-based concentric ring colors (85c4e7cd)
2. CheckerboardLayerColorSolver  — color non-bg CCs by min-boundary-distance tier (8a371977)
3. RayToRectangleSolver          — isolated marker shoots line toward row/col-aligned solid rectangle (2c608aff)
4. ScatterNoiseRemovalSolver     — isolated single non-bg cells (no 8-conn neighbor) → bg (42a50994)
5. CheckerboardInteriorSolver    — solid rectangle: border rows stay, interior rows alternate fill/bg (3bdb4ada)
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core.grid_codec import background_color
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _connected_components_4(grid: np.ndarray, bg: int):
    """Return list of (color, set_of_(r,c)) for each 4-connected non-bg CC."""
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg and not visited[r, c]:
                color = grid[r, c]
                cells = set()
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                components.append((color, cells))
    return components


def _find_solid_rectangles(grid: np.ndarray, bg: int):
    """Find all solid filled rectangles (CCs that fill their bounding box).
    Returns list of (fill_color, rmin, rmax, cmin, cmax).
    """
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    rects = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != bg and not visited[r, c]:
                fc = grid[r, c]
                cells = set()
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    cells.add((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == fc:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                rows = [row for row, col in cells]
                cols = [col for row, col in cells]
                rmin, rmax = min(rows), max(rows)
                cmin, cmax = min(cols), max(cols)
                expected = (rmax - rmin + 1) * (cmax - cmin + 1)
                if len(cells) >= int(expected * 0.9):
                    rects.append((fc, rmin, rmax, cmin, cmax))
    return rects


# ---------------------------------------------------------------------------
# 1. ConcentricRingReversalSolver
# ---------------------------------------------------------------------------

@register_solver
class ConcentricRingReversalSolver(BaseSolver):
    """Reverse concentric ring color sequence using min-boundary-distance layers.

    Pattern (85c4e7cd):
    - Each cell's "layer" = min(r, H-1-r, c, W-1-c).
    - Collect one color per layer distance (must be uniform within each layer).
    - Reverse the color list and reassign.
    - Includes background cells as layer participants.
    """

    name = "ConcentricRingReversalSolver"

    def _get_layer_colors(self, grid: np.ndarray):
        """Return ordered list of colors [layer0_color, layer1_color, ...] or None if non-uniform."""
        H, W = grid.shape
        layer_colors = {}
        for r in range(H):
            for c in range(W):
                d = min(r, H-1-r, c, W-1-c)
                color = grid[r, c]
                if d not in layer_colors:
                    layer_colors[d] = color
                elif layer_colors[d] != color:
                    return None  # non-uniform layer
        num_layers = max(layer_colors.keys()) + 1
        return [layer_colors[i] for i in range(num_layers)]

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        H, W = inp.shape
        colors = self._get_layer_colors(inp)
        if colors is None:
            return inp.copy()
        rev = list(reversed(colors))
        out = np.zeros_like(inp)
        for r in range(H):
            for c in range(W):
                d = min(r, H-1-r, c, W-1-c)
                out[r, c] = rev[d]
        return out

    def _verify(self, inp: np.ndarray, expected: np.ndarray) -> bool:
        return np.array_equal(self._apply(inp), expected)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(np.array(pair['input']), np.array(pair['output'])):
                return None
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply(np.array(ti))
            preds.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 2. CheckerboardLayerColorSolver
# ---------------------------------------------------------------------------

@register_solver
class CheckerboardLayerColorSolver(BaseSolver):
    """Color non-bg CCs by their minimum distance to the grid boundary tier.

    Pattern (8a371977):
    - For each non-bg CC, compute min_dist = min over all its cells of min(r, H-1-r, c, W-1-c).
    - CCs are split into 2 groups by output color. The group with smaller average min_dist
      → outer_color; the other group → inner_color.
    - max_outer_dist = max(min_dist for outer CCs) learned from training; apply:
      min_dist <= max_outer_dist → outer_color, min_dist > max_outer_dist → inner_color.
    """

    name = "CheckerboardLayerColorSolver"

    def _learn_colors(self, inp: np.ndarray, out: np.ndarray):
        """Return (outer_color, inner_color, max_outer_dist) or None."""
        if inp.shape != out.shape:
            return None
        bg = background_color(inp)
        H, W = inp.shape
        comps = _connected_components_4(inp, bg)
        if not comps:
            return None
        color_to_dists: dict = {}
        for color, cells in comps:
            min_dist = min(min(r, H-1-r, c, W-1-c) for r, c in cells)
            sr, sc = next(iter(cells))
            oc = out[sr, sc]
            if oc not in color_to_dists:
                color_to_dists[oc] = []
            color_to_dists[oc].append(min_dist)
        # Must have exactly 2 distinct output colors (neither should be bg)
        out_colors = [c for c in color_to_dists if c != bg]
        if len(out_colors) != 2:
            return None
        # Outer = smaller average min_dist
        avg0 = sum(color_to_dists[out_colors[0]]) / len(color_to_dists[out_colors[0]])
        avg1 = sum(color_to_dists[out_colors[1]]) / len(color_to_dists[out_colors[1]])
        if avg0 < avg1:
            outer_color, inner_color = out_colors[0], out_colors[1]
        else:
            outer_color, inner_color = out_colors[1], out_colors[0]
        max_outer_dist = max(color_to_dists[outer_color])
        return outer_color, inner_color, max_outer_dist

    def _apply_with_colors(self, inp: np.ndarray, outer_color: int, inner_color: int,
                           max_outer_dist: int) -> np.ndarray:
        bg = background_color(inp)
        H, W = inp.shape
        out = np.full_like(inp, bg)
        comps = _connected_components_4(inp, bg)
        for color, cells in comps:
            min_dist = min(min(r, H-1-r, c, W-1-c) for r, c in cells)
            fill = outer_color if min_dist <= max_outer_dist else inner_color
            for (r, c) in cells:
                out[r, c] = fill
        return out

    def _verify(self, inp: np.ndarray, expected: np.ndarray) -> bool:
        result = self._learn_colors(inp, expected)
        if result is None:
            return False
        outer_color, inner_color, max_outer_dist = result
        return np.array_equal(self._apply_with_colors(inp, outer_color, inner_color, max_outer_dist), expected)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        outer_color = None
        inner_color = None
        max_outer_dist = 0
        learnable_pairs = []
        for pair in ctx.train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            result = self._learn_colors(inp, out)
            if result is None:
                continue  # skip pairs where CC structure can't determine 2 colors
            oc, ic, mod = result
            if outer_color is None:
                outer_color, inner_color = oc, ic
            elif outer_color != oc or inner_color != ic:
                return None  # inconsistent colors across pairs
            max_outer_dist = max(max_outer_dist, mod)
            learnable_pairs.append((inp, out))
        if outer_color is None:
            return None  # no learnable pairs at all
        # Verify only the learnable training pairs with combined max_outer_dist
        for inp, out in learnable_pairs:
            if not np.array_equal(self._apply_with_colors(inp, outer_color, inner_color, max_outer_dist), out):
                return None
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply_with_colors(np.array(ti), outer_color, inner_color, max_outer_dist)
            preds.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 3. RayToRectangleSolver
# ---------------------------------------------------------------------------

@register_solver
class RayToRectangleSolver(BaseSolver):
    """Isolated marker cells shoot a ray toward a row/col-aligned solid rectangle.

    Pattern (2c608aff):
    - Find the largest solid rectangle (largest CC).
    - Find isolated marker cells (single-cell CCs of a different color).
    - A marker is aligned if its row is within [rmin, rmax] OR col within [cmin, cmax].
    - Aligned markers: draw a line of marker_color from marker toward nearest rectangle face.
    - Non-aligned markers: unchanged.
    """

    name = "RayToRectangleSolver"

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        bg = background_color(inp)
        comps = _connected_components_4(inp, bg)
        if len(comps) < 2:
            return inp.copy()

        # Largest CC = rectangle
        comps_sorted = sorted(comps, key=lambda x: len(x[1]), reverse=True)
        rect_color, rect_cells = comps_sorted[0]
        rect_rows = [r for r, c in rect_cells]
        rect_cols = [c for r, c in rect_cells]
        rmin, rmax = min(rect_rows), max(rect_rows)
        cmin, cmax = min(rect_cols), max(rect_cols)

        out = inp.copy()
        for color, cells in comps_sorted[1:]:
            if len(cells) != 1:
                continue
            (mr, mc) = next(iter(cells))
            marker_color = color
            row_aligned = rmin <= mr <= rmax
            col_aligned = cmin <= mc <= cmax

            if not row_aligned and not col_aligned:
                continue

            if row_aligned:
                if mc < cmin:
                    for c in range(mc + 1, cmin):
                        out[mr, c] = marker_color
                elif mc > cmax:
                    for c in range(cmax + 1, mc):
                        out[mr, c] = marker_color
            elif col_aligned:
                if mr < rmin:
                    for r in range(mr + 1, rmin):
                        out[r, mc] = marker_color
                elif mr > rmax:
                    for r in range(rmax + 1, mr):
                        out[r, mc] = marker_color
        return out

    def _verify(self, inp: np.ndarray, expected: np.ndarray) -> bool:
        return np.array_equal(self._apply(inp), expected)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(np.array(pair['input']), np.array(pair['output'])):
                return None
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply(np.array(ti))
            preds.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 4. ScatterNoiseRemovalSolver
# ---------------------------------------------------------------------------

@register_solver
class ScatterNoiseRemovalSolver(BaseSolver):
    """Remove isolated noise: non-bg cells with NO 8-connected non-bg neighbor → bg.

    Pattern (42a50994):
    - Uses 8-connectivity (diagonal neighbors count).
    - A non-bg cell with at least one 8-connected non-bg neighbor → keep.
    - A non-bg cell with zero 8-connected non-bg neighbors → remove (set to bg).
    """

    name = "ScatterNoiseRemovalSolver"

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        bg = background_color(inp)
        H, W = inp.shape
        out = inp.copy()
        for r in range(H):
            for c in range(W):
                if inp[r, c] == bg:
                    continue
                has_neighbor = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != bg:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break
                if not has_neighbor:
                    out[r, c] = bg
        return out

    def _verify(self, inp: np.ndarray, expected: np.ndarray) -> bool:
        return np.array_equal(self._apply(inp), expected)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        has_isolated = False
        for pair in ctx.train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            if not self._verify(inp, out):
                return None
            # Confirm there's actually noise to remove
            bg = background_color(inp)
            H, W = inp.shape
            for r in range(H):
                for c in range(W):
                    if inp[r, c] != bg:
                        neighbor_found = False
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < H and 0 <= nc < W and inp[nr, nc] != bg:
                                    neighbor_found = True
                                    break
                            if neighbor_found:
                                break
                        if not neighbor_found:
                            has_isolated = True
        if not has_isolated:
            return None
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply(np.array(ti))
            preds.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 5. CheckerboardInteriorSolver
# ---------------------------------------------------------------------------

@register_solver
class CheckerboardInteriorSolver(BaseSolver):
    """Solid rectangle: border rows stay solid, interior rows alternate fill/bg from cmin.

    Pattern (3bdb4ada):
    - Find ALL solid filled rectangles in the grid.
    - For each: top and bottom rows (border rows) keep fill_color entirely.
    - Interior rows: alternate fill_color/bg starting with fill_color at column cmin.
      (No border-column concept — alternation covers the full width.)
    - Handles multiple rectangles per grid.
    """

    name = "CheckerboardInteriorSolver"

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        bg = background_color(inp)
        rects = _find_solid_rectangles(inp, bg)
        if not rects:
            return inp.copy()
        out = inp.copy()
        for fc, rmin, rmax, cmin, cmax in rects:
            for r in range(rmin, rmax + 1):
                is_border_row = (r == rmin or r == rmax)
                for c in range(cmin, cmax + 1):
                    if is_border_row:
                        out[r, c] = fc
                    else:
                        pos = c - cmin
                        out[r, c] = fc if pos % 2 == 0 else bg
        return out

    def _verify(self, inp: np.ndarray, expected: np.ndarray) -> bool:
        return np.array_equal(self._apply(inp), expected)

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        # Must have at least one interior row across all training rectangles
        has_interior = False
        for pair in ctx.train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])
            if not self._verify(inp, out):
                return None
            bg = background_color(inp)
            rects = _find_solid_rectangles(inp, bg)
            for fc, rmin, rmax, cmin, cmax in rects:
                if rmax - rmin >= 2:
                    has_interior = True
        if not has_interior:
            return None
        preds = []
        for ti in ctx.test_inputs:
            g = self._apply(np.array(ti))
            preds.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
