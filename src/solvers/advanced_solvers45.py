"""Session 45 solvers — 20 new solvers targeting transform, color-select,
fill, subgrid, and object-isolation patterns in the remaining 552 unsolved tasks."""
from __future__ import annotations
import numpy as np
from collections import Counter, defaultdict
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

# ─── shared helpers ──────────────────────────────────────────────────────────
def _45_bg(grid) -> int:
    a = np.asarray(grid, dtype=int)
    vals, cnt = np.unique(a, return_counts=True)
    return int(vals[cnt.argmax()])

def _G(g):
    return np.asarray(g, dtype=int)

def _45_reach(arr, bg):
    """Return set of (r,c) bg-cells reachable from any border via 4-connectivity."""
    H, W = arr.shape
    vis = set()
    q = []
    for r in range(H):
        for c in (0, W - 1):
            if int(arr[r, c]) == bg and (r, c) not in vis:
                vis.add((r, c)); q.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if int(arr[r, c]) == bg and (r, c) not in vis:
                vis.add((r, c)); q.append((r, c))
    head = 0
    while head < len(q):
        r, c = q[head]; head += 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis and int(arr[nr, nc]) == bg:
                vis.add((nr, nc)); q.append((nr, nc))
    return vis

def _45_cc(arr, bg=None):
    """All 4-connected components excluding bg.
    Returns list of (color, frozenset_of_(r,c))."""
    H, W = arr.shape
    vis = np.zeros_like(arr, dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            v = int(arr[r, c])
            if vis[r, c] or (bg is not None and v == bg):
                continue
            stack, comp = [(r, c)], []
            vis[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not vis[nr, nc] and int(arr[nr, nc]) == v:
                        vis[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((v, frozenset(comp)))
    return comps


# ════════ SESSION 45 SOLVERS (+20) ════════

# ─── 1. Rotation 90° clockwise ───────────────────────────────────────────────
@register_solver
class Rot90CWSolver(BaseSolver):
    """Output = input rotated 90° clockwise."""
    @property
    def name(self): return "Rot90CWSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            r = np.rot90(i, 3)
            if r.shape != o.shape or not np.array_equal(r, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(np.rot90(_G(t), 3).tolist(),
                             np.rot90(_G(t), 3).tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 2. Rotation 90° counter-clockwise ──────────────────────────────────────
@register_solver
class Rot90CCWSolver(BaseSolver):
    """Output = input rotated 90° counter-clockwise."""
    @property
    def name(self): return "Rot90CCWSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            r = np.rot90(i, 1)
            if r.shape != o.shape or not np.array_equal(r, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(np.rot90(_G(t), 1).tolist(),
                             np.rot90(_G(t), 1).tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 3. Rotation 180° ────────────────────────────────────────────────────────
@register_solver
class Rot180Solver(BaseSolver):
    """Output = input rotated 180°."""
    @property
    def name(self): return "Rot180Solver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            r = np.rot90(i, 2)
            if r.shape != o.shape or not np.array_equal(r, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(np.rot90(_G(t), 2).tolist(),
                             np.rot90(_G(t), 2).tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 4. Flip horizontal (left-right) ────────────────────────────────────────
@register_solver
class FlipHorizSolver(BaseSolver):
    """Output = input flipped left-right."""
    @property
    def name(self): return "FlipHorizSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            f = np.fliplr(i)
            if f.shape != o.shape or not np.array_equal(f, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(np.fliplr(_G(t)).tolist(),
                             np.fliplr(_G(t)).tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 5. Flip vertical (up-down) ──────────────────────────────────────────────
@register_solver
class FlipVertSolver(BaseSolver):
    """Output = input flipped up-down."""
    @property
    def name(self): return "FlipVertSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            f = np.flipud(i)
            if f.shape != o.shape or not np.array_equal(f, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(np.flipud(_G(t)).tolist(),
                             np.flipud(_G(t)).tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 6. Transpose (main diagonal) ───────────────────────────────────────────
@register_solver
class TransposeSolver45(BaseSolver):
    """Output = transpose of input."""
    @property
    def name(self): return "TransposeSolver45"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            tr = i.T
            if tr.shape != o.shape or not np.array_equal(tr, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(_G(t).T.tolist(), _G(t).T.tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 7. Anti-diagonal transpose ─────────────────────────────────────────────
@register_solver
class AntiTransposeSolver(BaseSolver):
    """Output = anti-diagonal transpose (fliplr then transpose)."""
    @property
    def name(self): return "AntiTransposeSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            at = np.fliplr(i).T
            if at.shape != o.shape or not np.array_equal(at, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = [Prediction(np.fliplr(_G(t)).T.tolist(),
                             np.fliplr(_G(t)).T.tolist(), self.name, 1.0)
                 for t in ctx.test_inputs]
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 8. Keep single non-bg color (learned, consistent across pairs) ──────────
@register_solver
class KeepSingleNonBgColorSolver(BaseSolver):
    """Keep only cells of one consistent non-bg color; rest → bg.
    Validates that the kept color's mask is identical in input and output."""
    @property
    def name(self): return "KeepSingleNonBgColorSolver"
    def solve(self, ctx):
        kept_list = []
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            out_cs = set(int(v) for v in np.unique(o)) - {bg}
            if len(out_cs) != 1:
                return SolverResult(ctx.task_id, [], False, 2)
            kc = list(out_cs)[0]
            # mask must match exactly
            if not np.array_equal(o == kc, i == kc):
                return SolverResult(ctx.task_id, [], False, 2)
            kept_list.append(kc)
        if len(set(kept_list)) != 1:
            return SolverResult(ctx.task_id, [], False, 2)
        kc = kept_list[0]
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            bg = _45_bg(t)
            res = np.full_like(ia, bg)
            res[ia == kc] = kc
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 9. Value-based color replacement (color X → color Y globally) ───────────
@register_solver
class ValueColorReplaceSolver(BaseSolver):
    """Learn a consistent color-value→color-value map from training pairs and apply."""
    @property
    def name(self): return "ValueColorReplaceSolver"
    def solve(self, ctx):
        cmap: dict = {}
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            for v in np.unique(i):
                v = int(v)
                rows, cols = np.where(i == v)
                out_vals = set(int(o[r, c]) for r, c in zip(rows, cols))
                if len(out_vals) != 1:
                    return SolverResult(ctx.task_id, [], False, 2)
                ov = list(out_vals)[0]
                if v in cmap and cmap[v] != ov:
                    return SolverResult(ctx.task_id, [], False, 2)
                cmap[v] = ov
        if not cmap or all(k == v for k, v in cmap.items()):
            return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            res = ia.copy()
            for v, ov in cmap.items():
                res[ia == v] = ov
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 0.9))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 10. Keep most-common non-bg color ──────────────────────────────────────
@register_solver
class KeepMostCommonColorSolver(BaseSolver):
    """Keep only cells of the most-frequent non-bg color; rest → bg."""
    @property
    def name(self): return "KeepMostCommonColorSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            vs, cs = np.unique(i, return_counts=True)
            non_bg = [(int(v), int(c)) for v, c in zip(vs, cs) if int(v) != bg]
            if not non_bg:
                return SolverResult(ctx.task_id, [], False, 2)
            kc = max(non_bg, key=lambda x: x[1])[0]
            exp = np.full_like(i, bg)
            exp[i == kc] = kc
            if not np.array_equal(exp, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            bg = _45_bg(t)
            vs, cs = np.unique(ia, return_counts=True)
            non_bg = [(int(v), int(c)) for v, c in zip(vs, cs) if int(v) != bg]
            if not non_bg:
                preds.append(Prediction(ia.tolist(), ia.tolist(), self.name, 0.1)); continue
            kc = max(non_bg, key=lambda x: x[1])[0]
            res = np.full_like(ia, bg)
            res[ia == kc] = kc
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 0.9))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 11. Keep least-common non-bg color ─────────────────────────────────────
@register_solver
class KeepLeastCommonColorSolver(BaseSolver):
    """Keep only cells of the least-frequent non-bg color; rest → bg."""
    @property
    def name(self): return "KeepLeastCommonColorSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            vs, cs = np.unique(i, return_counts=True)
            non_bg = [(int(v), int(c)) for v, c in zip(vs, cs) if int(v) != bg]
            if not non_bg:
                return SolverResult(ctx.task_id, [], False, 2)
            kc = min(non_bg, key=lambda x: x[1])[0]
            exp = np.full_like(i, bg)
            exp[i == kc] = kc
            if not np.array_equal(exp, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            bg = _45_bg(t)
            vs, cs = np.unique(ia, return_counts=True)
            non_bg = [(int(v), int(c)) for v, c in zip(vs, cs) if int(v) != bg]
            if not non_bg:
                preds.append(Prediction(ia.tolist(), ia.tolist(), self.name, 0.1)); continue
            kc = min(non_bg, key=lambda x: x[1])[0]
            res = np.full_like(ia, bg)
            res[ia == kc] = kc
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 0.9))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 12. Remove exactly one non-bg color ────────────────────────────────────
@register_solver
class RemoveOneColorSolver(BaseSolver):
    """One specific non-bg color is removed (→ bg); all other cells unchanged."""
    @property
    def name(self): return "RemoveOneColorSolver"
    def solve(self, ctx):
        removed_list = []
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            in_cs = set(int(v) for v in np.unique(i)) - {bg}
            out_cs = set(int(v) for v in np.unique(o)) - {bg}
            removed = in_cs - out_cs
            if len(removed) != 1:
                return SolverResult(ctx.task_id, [], False, 2)
            rc = list(removed)[0]
            # all non-removed, non-bg cells must be identical in input and output
            for v in out_cs:
                if not np.array_equal(i == v, o == v):
                    return SolverResult(ctx.task_id, [], False, 2)
            # removed-color cells must become bg in output
            if not np.array_equal(o[i == rc], np.full(int((i == rc).sum()), bg)):
                return SolverResult(ctx.task_id, [], False, 2)
            removed_list.append(rc)
        if not removed_list or len(set(removed_list)) != 1:
            return SolverResult(ctx.task_id, [], False, 2)
        rc = removed_list[0]
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t).copy()
            bg = _45_bg(t)
            ia[ia == rc] = bg
            preds.append(Prediction(ia.tolist(), ia.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 13. Exact contiguous subgrid extraction ────────────────────────────────
@register_solver
class ExactSubgridSolver(BaseSolver):
    """Output is an exact contiguous subgrid of input at a consistent offset."""
    @property
    def name(self): return "ExactSubgridSolver"
    def _find(self, i, o):
        h, w = o.shape
        H, W = i.shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if np.array_equal(i[r0:r0 + h, c0:c0 + w], o):
                    return r0, c0
        return None
    def solve(self, ctx):
        slices = []
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape == o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            res = self._find(i, o)
            if res is None:
                return SolverResult(ctx.task_id, [], False, 2)
            slices.append((res[0], res[1], o.shape[0], o.shape[1]))
        r0, c0, h, w = slices[0]
        for s in slices[1:]:
            if s != (r0, c0, h, w):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            H, W = ia.shape
            ar0, ac0 = min(r0, H - 1), min(c0, W - 1)
            res = ia[ar0:min(ar0 + h, H), ac0:min(ac0 + w, W)]
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 0.9))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 14. Fill border-reachable bg cells with learned color ──────────────────
@register_solver
class FillBorderReachableV2Solver(BaseSolver):
    """Fill bg cells reachable from any grid border with a consistent learned color."""
    @property
    def name(self): return "FillBorderReachableV2Solver"
    def solve(self, ctx):
        fill_cs = []
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            reach = _45_reach(i, bg)
            if not reach:
                return SolverResult(ctx.task_id, [], False, 2)
            fcs = set()
            for r, c in reach:
                ov = int(o[r, c])
                if ov != bg:
                    fcs.add(ov)
            if len(fcs) != 1:
                return SolverResult(ctx.task_id, [], False, 2)
            fc = list(fcs)[0]
            # non-reachable cells must be unchanged
            for r in range(i.shape[0]):
                for c in range(i.shape[1]):
                    if (r, c) not in reach and int(i[r, c]) != int(o[r, c]):
                        return SolverResult(ctx.task_id, [], False, 2)
            fill_cs.append(fc)
        if not fill_cs or len(set(fill_cs)) != 1:
            return SolverResult(ctx.task_id, [], False, 2)
        fc = fill_cs[0]
        # GUARD: final full-output check on all training pairs
        for p in ctx.train_pairs:
            ia = _G(p["input"]).copy()
            bg = _45_bg(p["input"])
            for r, c in _45_reach(ia, bg):
                ia[r, c] = fc
            if ia.tolist() != p["output"]:
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t).copy()
            bg = _45_bg(t)
            for r, c in _45_reach(ia, bg):
                ia[r, c] = fc
            preds.append(Prediction(ia.tolist(), ia.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 15. Fill interior (enclosed) bg cells with learned color ───────────────
@register_solver
class FillInteriorBgSolver(BaseSolver):
    """Fill bg cells NOT reachable from any border (enclosed regions) with a learned color."""
    @property
    def name(self): return "FillInteriorBgSolver"
    def solve(self, ctx):
        fill_cs = []
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            reach = _45_reach(i, bg)
            interior = [(r, c) for r in range(i.shape[0]) for c in range(i.shape[1])
                        if int(i[r, c]) == bg and (r, c) not in reach]
            if not interior:
                return SolverResult(ctx.task_id, [], False, 2)
            fcs = set()
            for r, c in interior:
                ov = int(o[r, c])
                if ov != bg:
                    fcs.add(ov)
            if len(fcs) != 1:
                return SolverResult(ctx.task_id, [], False, 2)
            fc = list(fcs)[0]
            # GUARD: ALL interior cells must become exactly fc in output
            for r, c in interior:
                if int(o[r, c]) != fc:
                    return SolverResult(ctx.task_id, [], False, 2)
            # GUARD: non-interior cells must be unchanged
            interior_set = set(interior)
            for r in range(i.shape[0]):
                for c in range(i.shape[1]):
                    if (r, c) not in interior_set and int(i[r, c]) != int(o[r, c]):
                        return SolverResult(ctx.task_id, [], False, 2)
            fill_cs.append(fc)
        if not fill_cs or len(set(fill_cs)) != 1:
            return SolverResult(ctx.task_id, [], False, 2)
        fc = fill_cs[0]
        # GUARD: full reconstruction check on all training pairs
        for p in ctx.train_pairs:
            ia = _G(p['input']).copy()
            bg = _45_bg(p['input'])
            reach = _45_reach(ia, bg)
            for r in range(ia.shape[0]):
                for c in range(ia.shape[1]):
                    if int(ia[r, c]) == bg and (r, c) not in reach:
                        ia[r, c] = fc
            if ia.tolist() != p['output']:
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t).copy()
            bg = _45_bg(t)
            reach = _45_reach(ia, bg)
            for r in range(ia.shape[0]):
                for c in range(ia.shape[1]):
                    if int(ia[r, c]) == bg and (r, c) not in reach:
                        ia[r, c] = fc
            preds.append(Prediction(ia.tolist(), ia.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 16. Isolate the uniquely-sized connected component ─────────────────────
@register_solver
class IsolateUniqueSizedObjectSolver(BaseSolver):
    """Keep only the CC whose cell count is unique (not shared by any other CC)."""
    @property
    def name(self): return "IsolateUniqueSizedObjectSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            comps = _45_cc(i, bg)
            if len(comps) < 2:
                return SolverResult(ctx.task_id, [], False, 2)
            sc = Counter(len(c[1]) for c in comps)
            unique = [c for c in comps if sc[len(c[1])] == 1]
            if len(unique) != 1:
                return SolverResult(ctx.task_id, [], False, 2)
            exp = np.full_like(i, bg)
            col, pos = unique[0]
            for r2, c2 in pos:
                exp[r2, c2] = col
            if not np.array_equal(exp, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            bg = _45_bg(t)
            comps = _45_cc(ia, bg)
            sc = Counter(len(c[1]) for c in comps)
            unique = [c for c in comps if sc[len(c[1])] == 1]
            if len(unique) != 1:
                preds.append(Prediction(ia.tolist(), ia.tolist(), self.name, 0.1)); continue
            res = np.full_like(ia, bg)
            col, pos = unique[0]
            for r2, c2 in pos:
                res[r2, c2] = col
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 0.9))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 17. Object outline (border cells only) ─────────────────────────────────
@register_solver
class OutlineObjectSolver(BaseSolver):
    """Replace filled objects with just their border cells (adjacent to bg or grid edge)."""
    @property
    def name(self): return "OutlineObjectSolver"
    def _outline(self, arr, bg):
        H, W = arr.shape
        res = np.full_like(arr, bg)
        for r in range(H):
            for c in range(W):
                if int(arr[r, c]) == bg:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W) or int(arr[nr, nc]) == bg:
                        res[r, c] = arr[r, c]
                        break
        return res
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            bg = _45_bg(p['input'])
            if not np.array_equal(self._outline(i, bg), o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            bg = _45_bg(t)
            res = self._outline(ia, bg)
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 18. Reflect left half to right ─────────────────────────────────────────
@register_solver
class ReflectLeftHalfSolver(BaseSolver):
    """Output = left half of input mirrored right (horizontal symmetry)."""
    @property
    def name(self): return "ReflectLeftHalfSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            W = i.shape[1]
            if W < 2:
                return SolverResult(ctx.task_id, [], False, 2)
            half = W // 2
            left = i[:, :half]
            exp = np.hstack([left, np.fliplr(left)])
            if exp.shape != o.shape or not np.array_equal(exp, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            half = ia.shape[1] // 2
            left = ia[:, :half]
            res = np.hstack([left, np.fliplr(left)])
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 19. Reflect top half downward ──────────────────────────────────────────
@register_solver
class ReflectTopHalfSolver(BaseSolver):
    """Output = top half of input mirrored down (vertical symmetry)."""
    @property
    def name(self): return "ReflectTopHalfSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            H = i.shape[0]
            if H < 2:
                return SolverResult(ctx.task_id, [], False, 2)
            half = H // 2
            top = i[:half, :]
            exp = np.vstack([top, np.flipud(top)])
            if exp.shape != o.shape or not np.array_equal(exp, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            half = ia.shape[0] // 2
            top = ia[:half, :]
            res = np.vstack([top, np.flipud(top)])
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)


# ─── 20. Reflect right half leftward ────────────────────────────────────────
@register_solver
class ReflectRightHalfSolver(BaseSolver):
    """Output = right half of input mirrored left (horizontal symmetry from right)."""
    @property
    def name(self): return "ReflectRightHalfSolver"
    def solve(self, ctx):
        for p in ctx.train_pairs:
            i, o = _G(p['input']), _G(p['output'])
            if i.shape != o.shape:
                return SolverResult(ctx.task_id, [], False, 2)
            W = i.shape[1]
            if W < 2:
                return SolverResult(ctx.task_id, [], False, 2)
            half = W - W // 2
            right = i[:, W - half:]
            exp = np.hstack([np.fliplr(right), right])
            if exp.shape != o.shape or not np.array_equal(exp, o):
                return SolverResult(ctx.task_id, [], False, 2)
        preds = []
        for t in ctx.test_inputs:
            ia = _G(t)
            W = ia.shape[1]
            half = W - W // 2
            right = ia[:, W - half:]
            res = np.hstack([np.fliplr(right), right])
            preds.append(Prediction(res.tolist(), res.tolist(), self.name, 1.0))
        return SolverResult(ctx.task_id, preds, True, 2)
