"""
Session 44 solvers — 15 high-coverage strategies targeting the dominant
unsolved-task patterns identified in the session-44 analysis.

Helper prefix: _44_  (avoids collision with helpers in other solver files)
"""

import numpy as np
from collections import defaultdict, deque
from itertools import product
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

# ──────────────────────────────────────────────────────────────────────
# Shared helpers (all prefixed _44_ )
# ──────────────────────────────────────────────────────────────────────

def _44_arr(grid):
    """Return grid as a numpy int array."""
    return np.array(grid, dtype=int)

def _44_lst(arr):
    """Return numpy array as a nested Python list (for Prediction)."""
    return arr.tolist()

def _44_bg(arr):
    """Most-frequent color in arr (background color)."""
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])

def _44_ccs(arr, bg):
    """
    Return list of connected components (4-connected) of non-bg pixels.
    Each CC is a list of (r, c) tuples.
    """
    rows, cols = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    ccs = []
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] != bg and not visited[r, c]:
                # BFS
                queue = deque([(r, c)])
                visited[r, c] = True
                cc = [(r, c)]
                while queue:
                    cr, cc_ = queue.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc_+dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and arr[nr,nc] != bg:
                            visited[nr,nc] = True
                            queue.append((nr,nc))
                            cc.append((nr,nc))
                ccs.append(cc)
    return ccs

def _44_consistent_cmap(pairs):
    """
    Build a per-pixel consistent color map from training pairs.
    Returns dict {src_color: tgt_color} if every src_color maps to exactly
    one tgt_color across all pairs, else returns None.
    """
    mapping = {}
    for inp, out in pairs:
        a_in  = _44_arr(inp)
        a_out = _44_arr(out)
        if a_in.shape != a_out.shape:
            return None
        for r in range(a_in.shape[0]):
            for c in range(a_in.shape[1]):
                s = int(a_in[r, c])
                t = int(a_out[r, c])
                if s in mapping:
                    if mapping[s] != t:
                        return None
                else:
                    mapping[s] = t
    return mapping if mapping else None

def _44_apply_cmap(arr, cmap):
    """Apply a color map dict to every pixel of arr."""
    out = arr.copy()
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = int(arr[r, c])
            if v in cmap:
                out[r, c] = cmap[v]
    return out

def _44_crop_bbox(arr, bg):
    """Crop arr to the tight bounding box of all non-bg pixels."""
    mask = arr != bg
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return arr
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return arr[r0:r1+1, c0:c1+1]

def _44_remove_bg_rows_cols(arr, bg):
    """Strip all entirely-bg rows and columns."""
    mask_r = np.any(arr != bg, axis=1)
    mask_c = np.any(arr != bg, axis=0)
    result = arr[mask_r, :]
    result = result[:, mask_c]
    return result if result.size > 0 else arr

def _44_nn_fill(arr, bg):
    """
    BFS Voronoi nearest-neighbor fill: every bg cell takes the color of
    the nearest non-bg cell (Manhattan-style BFS from all non-bg seeds).
    """
    out = arr.copy()
    rows, cols = arr.shape
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] != bg:
                queue.append((r, c))
    visited = (arr != bg).copy()
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                out[nr, nc] = out[r, c]
                queue.append((nr, nc))
    return out

def _44_gravity(arr, bg, direction):
    """
    Pull non-bg pixels toward `direction` ('down','up','left','right').
    Returns the resulting grid.
    """
    out = np.full_like(arr, bg)
    rows, cols = arr.shape
    if direction in ('down', 'up'):
        for c in range(cols):
            col_vals = [arr[r, c] for r in range(rows) if arr[r, c] != bg]
            if direction == 'down':
                start = rows - len(col_vals)
                for i, v in enumerate(col_vals):
                    out[start + i, c] = v
            else:
                for i, v in enumerate(col_vals):
                    out[i, c] = v
    else:
        for r in range(rows):
            row_vals = [arr[r, c] for c in range(cols) if arr[r, c] != bg]
            if direction == 'right':
                start = cols - len(row_vals)
                for i, v in enumerate(row_vals):
                    out[r, start + i] = v
            else:
                for i, v in enumerate(row_vals):
                    out[r, i] = v
    return out

# 7 geometric transforms (same set used in GeomPlusCmapSolver)
_44_GEOM = {
    'flipH':    lambda a: np.fliplr(a),
    'flipV':    lambda a: np.flipud(a),
    'rot90':    lambda a: np.rot90(a, 1),
    'rot180':   lambda a: np.rot90(a, 2),
    'rot270':   lambda a: np.rot90(a, 3),
    'transpT':  lambda a: a.T,
    'transpAT': lambda a: np.fliplr(a).T,
}


# ──────────────────────────────────────────────────────────────────────
# 1. ConsistentColorMapSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class ConsistentColorMapSolver(BaseSolver):
    """
    Learns a per-pixel color mapping that is consistent across all training
    pairs (same-size input/output required).  Covers recolor_only and
    single_fill_color patterns.
    """
    @property
    def name(self): return "ConsistentColorMapSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        cmap = _44_consistent_cmap(pairs)
        if cmap is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        # Check map is non-trivial (at least one color actually changes)
        if all(k == v for k, v in cmap.items()):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            if a.shape[0] == 0:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            p = _44_apply_cmap(a, cmap)
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 2. GeomPlusCmapSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class GeomPlusCmapSolver(BaseSolver):
    """
    Tries each of 7 geometric transforms on the input, then checks if the
    result + a consistent color map matches all training outputs.
    """
    @property
    def name(self): return "GeomPlusCmapSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        pairs = ctx.train_pairs
        for tname, tfn in _44_GEOM.items():
            # Build transformed pairs and check for consistent cmap
            tpairs = []
            ok = True
            for p in pairs:
                a_in  = _44_arr(p['input'])
                a_out = _44_arr(p['output'])
                ta = tfn(a_in)
                if ta.shape != a_out.shape:
                    ok = False; break
                tpairs.append((ta.tolist(), p['output']))
            if not ok:
                continue
            cmap = _44_consistent_cmap(tpairs)
            if cmap is None:
                # Try identity cmap (pure geom)
                cmap = {}
            # Verify on all training pairs
            valid = True
            for p in pairs:
                a_in  = _44_arr(p['input'])
                a_out = _44_arr(p['output'])
                pred = _44_apply_cmap(tfn(a_in), cmap)
                if pred.shape != a_out.shape or not np.array_equal(pred, a_out):
                    valid = False; break
            if not valid:
                continue
            # Apply to test inputs
            preds = []
            for ti in ctx.test_inputs:
                a = _44_arr(ti)
                pred = _44_apply_cmap(tfn(a), cmap)
                preds.append(Prediction(attempt_1=_44_lst(pred), attempt_2=_44_lst(pred),
                                        solver_name=self.name, confidence=1.0))
            return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
        return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 3. CropToBBoxSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class CropToBBoxSolver(BaseSolver):
    """
    Output = crop of input to tight bounding box of non-bg pixels.
    Targets shrink tasks where all non-bg content is spatially bounded.
    """
    @property
    def name(self): return "CropToBBoxSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            bg = _44_bg(a_in)
            cropped = _44_crop_bbox(a_in, bg)
            if not np.array_equal(cropped, a_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            p = _44_crop_bbox(a, bg)
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 4. RemoveBgRowsColsSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class RemoveBgRowsColsSolver(BaseSolver):
    """
    Output = input with all entirely-bg rows and columns stripped.
    Variant of crop that removes interior gaps too.
    """
    @property
    def name(self): return "RemoveBgRowsColsSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            bg = _44_bg(a_in)
            stripped = _44_remove_bg_rows_cols(a_in, bg)
            if not np.array_equal(stripped, a_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            p = _44_remove_bg_rows_cols(a, bg)
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 5. CropPlusCmapSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class CropPlusCmapSolver(BaseSolver):
    """
    Crop to bbox then apply a consistent color map.
    """
    @property
    def name(self): return "CropPlusCmapSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        pairs = ctx.train_pairs
        tpairs = []
        for p in pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            bg = _44_bg(a_in)
            cropped = _44_crop_bbox(a_in, bg)
            if cropped.shape != a_out.shape:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            tpairs.append((cropped.tolist(), p['output']))
        cmap = _44_consistent_cmap(tpairs)
        if cmap is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        # Verify
        for (cropped_list, out), p in zip(tpairs, pairs):
            pred = _44_apply_cmap(_44_arr(cropped_list), cmap)
            if not np.array_equal(pred, _44_arr(out)):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            p = _44_apply_cmap(_44_crop_bbox(a, bg), cmap)
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 6. NearestNeighborFillSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class NearestNeighborFillSolver(BaseSolver):
    """
    Output = Voronoi BFS fill of bg cells with color of nearest non-bg pixel.
    Targets the 15 nearest_neighbor fill_bg tasks.
    """
    @property
    def name(self): return "NearestNeighborFillSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            if a_in.shape != a_out.shape:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            bg = _44_bg(a_in)
            filled = _44_nn_fill(a_in, bg)
            if not np.array_equal(filled, a_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            p = _44_nn_fill(a, bg)
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 7. GravityFillSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class GravityFillSolver(BaseSolver):
    """
    Non-bg pixels fall toward one of 4 directions (down/up/left/right).
    Tries each direction; succeeds if one is consistent across all training pairs.
    """
    @property
    def name(self): return "GravityFillSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        for direction in ('down', 'up', 'left', 'right'):
            valid = True
            for p in ctx.train_pairs:
                a_in  = _44_arr(p['input'])
                a_out = _44_arr(p['output'])
                if a_in.shape != a_out.shape:
                    valid = False; break
                bg = _44_bg(a_in)
                g = _44_gravity(a_in, bg, direction)
                if not np.array_equal(g, a_out):
                    valid = False; break
            if valid:
                preds = []
                for ti in ctx.test_inputs:
                    a = _44_arr(ti)
                    bg = _44_bg(a)
                    p = _44_gravity(a, bg, direction)
                    preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                            solver_name=self.name, confidence=1.0))
                return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
        return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 8. RecolorByObjectSizeSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class RecolorByObjectSizeSolver(BaseSolver):
    """
    Recolor each connected component (non-bg) based on its pixel count.
    Learns size→output_color mapping from training pairs.
    """
    @property
    def name(self): return "RecolorByObjectSizeSolver"

    def _size_cmap(self, a_in, a_out, bg):
        """Build size→output_color from a single pair."""
        ccs = _44_ccs(a_in, bg)
        mapping = {}
        for cc in ccs:
            size = len(cc)
            # What color does the majority of these pixels become?
            colors = [int(a_out[r, c]) for r, c in cc]
            if not colors:
                continue
            col = max(set(colors), key=colors.count)
            if size in mapping and mapping[size] != col:
                return None
            mapping[size] = col
        return mapping

    def solve(self, ctx: TaskContext) -> SolverResult:
        size_map = {}
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            if a_in.shape != a_out.shape:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            bg = _44_bg(a_in)
            sm = self._size_cmap(a_in, a_out, bg)
            if sm is None:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            for k, v in sm.items():
                if k in size_map and size_map[k] != v:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
                size_map[k] = v
        if not size_map:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        # Verify on training pairs
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            bg = _44_bg(a_in)
            pred = np.full_like(a_in, bg)
            for cc in _44_ccs(a_in, bg):
                size = len(cc)
                col = size_map.get(size, bg)
                for r, c in cc:
                    pred[r, c] = col
            if not np.array_equal(pred, a_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            pred = np.full_like(a, bg)
            ccs = _44_ccs(a, bg)
            # Reject if any test object size is not in the learned map
            for cc in ccs:
                if len(cc) not in size_map:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            for cc in ccs:
                size = len(cc)
                col = size_map[size]
                for r, c in cc:
                    pred[r, c] = col
            preds.append(Prediction(attempt_1=_44_lst(pred), attempt_2=_44_lst(pred),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 9. KeepLargestObjectSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class KeepLargestObjectSolver(BaseSolver):
    """
    Output = input with all but the largest connected component set to bg.
    Targets remove_objects tasks where one dominant object is kept.
    """
    @property
    def name(self): return "KeepLargestObjectSolver"

    def _apply(self, a_in, bg):
        ccs = _44_ccs(a_in, bg)
        if not ccs:
            return a_in.copy()
        largest = max(ccs, key=len)
        out = np.full_like(a_in, bg)
        for r, c in largest:
            out[r, c] = a_in[r, c]
        return out

    def solve(self, ctx: TaskContext) -> SolverResult:
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            if a_in.shape != a_out.shape:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            bg = _44_bg(a_in)
            pred = self._apply(a_in, bg)
            if not np.array_equal(pred, a_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            p = self._apply(a, bg)
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 10. RemoveSmallObjectsSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class RemoveSmallObjectsSolver(BaseSolver):
    """
    Remove all CCs with pixel count <= threshold (tries thresholds 1..10).
    Covers remove_objects tasks where small noise/objects are stripped.
    """
    @property
    def name(self): return "RemoveSmallObjectsSolver"

    def _apply(self, a_in, bg, threshold):
        ccs = _44_ccs(a_in, bg)
        out = a_in.copy()
        for cc in ccs:
            if len(cc) <= threshold:
                for r, c in cc:
                    out[r, c] = bg
        return out

    def solve(self, ctx: TaskContext) -> SolverResult:
        for threshold in range(1, 11):
            valid = True
            for p in ctx.train_pairs:
                a_in  = _44_arr(p['input'])
                a_out = _44_arr(p['output'])
                if a_in.shape != a_out.shape:
                    valid = False; break
                bg = _44_bg(a_in)
                pred = self._apply(a_in, bg, threshold)
                if not np.array_equal(pred, a_out):
                    valid = False; break
            if valid:
                preds = []
                for ti in ctx.test_inputs:
                    a = _44_arr(ti)
                    bg = _44_bg(a)
                    p = self._apply(a, bg, threshold)
                    preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                            solver_name=self.name, confidence=1.0))
                return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
        return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 11. OutputConstantSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class OutputConstantSolver(BaseSolver):
    """
    If every training output is the same grid, predict that constant for test.
    Handles the rare but easy 'constant output' pattern.
    """
    @property
    def name(self): return "OutputConstantSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        outputs = [p['output'] for p in ctx.train_pairs]
        ref = outputs[0]
        for o in outputs[1:]:
            if o != ref:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for _ in ctx.test_inputs:
            preds.append(Prediction(attempt_1=ref, attempt_2=ref,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 12. TileDetectionSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class TileDetectionSolver(BaseSolver):
    """
    Output = input tiled N×M times.  Tries N,M in 1..5 (excluding 1×1).
    """
    @property
    def name(self): return "TileDetectionSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        for n in range(1, 6):
            for m in range(1, 6):
                if n == 1 and m == 1:
                    continue
                valid = True
                for p in ctx.train_pairs:
                    a_in  = _44_arr(p['input'])
                    a_out = _44_arr(p['output'])
                    tiled = np.tile(a_in, (n, m))
                    if not np.array_equal(tiled, a_out):
                        valid = False; break
                if valid:
                    preds = []
                    for ti in ctx.test_inputs:
                        a = _44_arr(ti)
                        p = np.tile(a, (n, m))
                        preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                                solver_name=self.name, confidence=1.0))
                    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
        return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 13. FillInteriorRegionSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class FillInteriorRegionSolver(BaseSolver):
    """
    Fill enclosed (non-border-reachable) bg regions with a learned fill color.
    Learns the fill color from training pairs.
    """
    @property
    def name(self): return "FillInteriorRegionSolver"

    def _enclosed_mask(self, arr, bg):
        """Return boolean mask of bg cells NOT reachable from any border cell."""
        rows, cols = arr.shape
        visited = np.zeros_like(arr, dtype=bool)
        queue = deque()
        for r in range(rows):
            for c in [0, cols-1]:
                if arr[r, c] == bg and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        for c in range(cols):
            for r in [0, rows-1]:
                if arr[r, c] == bg and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and arr[nr,nc] == bg:
                    visited[nr,nc] = True
                    queue.append((nr,nc))
        # Enclosed = bg but not border-reachable
        enclosed = (arr == bg) & ~visited
        return enclosed

    def solve(self, ctx: TaskContext) -> SolverResult:
        fill_color = None
        for p in ctx.train_pairs:
            a_in  = _44_arr(p['input'])
            a_out = _44_arr(p['output'])
            if a_in.shape != a_out.shape:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            bg = _44_bg(a_in)
            enclosed = self._enclosed_mask(a_in, bg)
            if not enclosed.any():
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            # All enclosed cells should have a single fill color in output
            fill_vals = a_out[enclosed]
            unique_fills = np.unique(fill_vals)
            if len(unique_fills) != 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            fc = int(unique_fills[0])
            if fc == bg:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            if fill_color is not None and fill_color != fc:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            fill_color = fc
            # Non-enclosed cells should be unchanged
            pred = a_in.copy()
            pred[enclosed] = fill_color
            if not np.array_equal(pred, a_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        if fill_color is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for ti in ctx.test_inputs:
            a = _44_arr(ti)
            bg = _44_bg(a)
            enclosed = self._enclosed_mask(a, bg)
            p = a.copy()
            p[enclosed] = fill_color
            preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 14. SymmetryFillSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class SymmetryFillSolver(BaseSolver):
    """
    Complete the grid to be symmetric.  Tries H-symmetry, V-symmetry, and
    both H+V together.  For each candidate, checks against training outputs.
    """
    @property
    def name(self): return "SymmetryFillSolver"

    def _fill_h(self, arr, bg):
        """Make left-right symmetric: bg cells take value from mirror position if non-bg."""
        out = arr.copy()
        cols = arr.shape[1]
        for r in range(arr.shape[0]):
            for c in range(cols):
                if out[r, c] == bg:
                    mc = cols - 1 - c
                    if arr[r, mc] != bg:
                        out[r, c] = arr[r, mc]
        return out

    def _fill_v(self, arr, bg):
        """Make top-bottom symmetric."""
        out = arr.copy()
        rows = arr.shape[0]
        for r in range(rows):
            for c in range(arr.shape[1]):
                if out[r, c] == bg:
                    mr = rows - 1 - r
                    if arr[mr, c] != bg:
                        out[r, c] = arr[mr, c]
        return out

    def solve(self, ctx: TaskContext) -> SolverResult:
        for mode in ('H', 'V', 'HV'):
            valid = True
            for p in ctx.train_pairs:
                a_in  = _44_arr(p['input'])
                a_out = _44_arr(p['output'])
                if a_in.shape != a_out.shape:
                    valid = False; break
                bg = _44_bg(a_in)
                if mode == 'H':
                    pred = self._fill_h(a_in, bg)
                elif mode == 'V':
                    pred = self._fill_v(a_in, bg)
                else:
                    pred = self._fill_h(self._fill_v(a_in, bg), bg)
                if not np.array_equal(pred, a_out):
                    valid = False; break
            if valid:
                preds = []
                for ti in ctx.test_inputs:
                    a = _44_arr(ti)
                    bg = _44_bg(a)
                    if mode == 'H':
                        p = self._fill_h(a, bg)
                    elif mode == 'V':
                        p = self._fill_v(a, bg)
                    else:
                        p = self._fill_h(self._fill_v(a, bg), bg)
                    preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                            solver_name=self.name, confidence=1.0))
                return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
        return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)


# ──────────────────────────────────────────────────────────────────────
# 15. SortRowsByColorSolver
# ──────────────────────────────────────────────────────────────────────
@register_solver
class SortRowsByColorSolver(BaseSolver):
    """
    Sort rows by their first non-bg color value (ascending or descending).
    Also tries sorting by dominant (most-frequent) color per row.
    """
    @property
    def name(self): return "SortRowsByColorSolver"

    def _row_key_first(self, row, bg):
        nz = [v for v in row if v != bg]
        return nz[0] if nz else 0

    def _row_key_dominant(self, row, bg):
        nz = [v for v in row if v != bg]
        if not nz: return 0
        return max(set(nz), key=nz.count)

    def solve(self, ctx: TaskContext) -> SolverResult:
        for key_fn in (self._row_key_first, self._row_key_dominant):
            for reverse in (False, True):
                valid = True
                for p in ctx.train_pairs:
                    a_in  = _44_arr(p['input'])
                    a_out = _44_arr(p['output'])
                    if a_in.shape != a_out.shape:
                        valid = False; break
                    bg = _44_bg(a_in)
                    rows = [a_in[r].tolist() for r in range(a_in.shape[0])]
                    sorted_rows = sorted(rows, key=lambda row: key_fn(row, bg), reverse=reverse)
                    pred = np.array(sorted_rows, dtype=int)
                    if not np.array_equal(pred, a_out):
                        valid = False; break
                if valid:
                    preds = []
                    for ti in ctx.test_inputs:
                        a = _44_arr(ti)
                        bg = _44_bg(a)
                        rows = [a[r].tolist() for r in range(a.shape[0])]
                        sorted_rows = sorted(rows, key=lambda row: key_fn(row, bg), reverse=reverse)
                        p = np.array(sorted_rows, dtype=int)
                        preds.append(Prediction(attempt_1=_44_lst(p), attempt_2=_44_lst(p),
                                                solver_name=self.name, confidence=1.0))
                    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
        return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
