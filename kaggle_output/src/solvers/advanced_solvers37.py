"""Session 40 solvers: ReflectFillSolver, ColorCanonicalLookupSolver, CoordinateCompressSolver."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
from collections import Counter
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver


def _most_common(grid):
    vals = [v for row in grid for v in row]
    return Counter(vals).most_common(1)[0][0]


def _dominant_nonzero(grid):
    vals = [v for row in grid for v in row if v != 0]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


@register_solver
class ReflectFillSolver(BaseSolver):
    """
    Task 60d73be6: detect exactly 1 uniform-non-bg row + 1 uniform-non-bg col forming cross.
    Find source quadrant (has nonzero content). Fill 3 empty quadrants via flips,
    aligned to cross-touching corner; bg-pad far sides, clip near sides if needed.
    """
    @property
    def name(self): return "ReflectFillSolver"

    def _apply(self, grid):
        g = np.array(grid)
        bg = _most_common(grid)
        H, W = g.shape

        uniform_rows = [r for r in range(H) if len(set(g[r])) == 1 and g[r, 0] != bg]
        uniform_cols = [c for c in range(W) if len(set(g[:, c])) == 1 and g[0, c] != bg]
        if len(uniform_rows) != 1 or len(uniform_cols) != 1:
            return None
        cr, cc = uniform_rows[0], uniform_cols[0]

        quads = {
            "TL": (0, cr, 0, cc),
            "TR": (0, cr, cc+1, W),
            "BL": (cr+1, H, 0, cc),
            "BR": (cr+1, H, cc+1, W),
        }

        def has_content(name):
            r0, r1, c0, c1 = quads[name]
            if r1 <= r0 or c1 <= c0:
                return False
            return np.any(g[r0:r1, c0:c1] != bg)

        content_quads = [n for n in quads if has_content(n)]
        if len(content_quads) != 1:
            return None
        src_name = content_quads[0]

        r0, r1, c0, c1 = quads[src_name]
        src = g[r0:r1, c0:c1].copy()
        out = g.copy()

        def fill_quad(tgt_name):
            tr0, tr1, tc0, tc1 = quads[tgt_name]
            if tr1 <= tr0 or tc1 <= tc0:
                return
            tH = tr1 - tr0
            tW = tc1 - tc0

            src_top  = src_name in ("TL", "TR")
            tgt_top  = tgt_name in ("TL", "TR")
            src_left = src_name in ("TL", "BL")
            tgt_left = tgt_name in ("TL", "BL")
            flip_ud  = (src_top != tgt_top)
            flip_lr  = (src_left != tgt_left)

            ref = src.copy()
            if flip_ud:
                ref = np.flipud(ref)
            if flip_lr:
                ref = np.fliplr(ref)

            rH, rW = ref.shape
            patch = np.full((tH, tW), bg, dtype=out.dtype)

            if tgt_name == "TL":
                pr0 = max(0, tH - rH); pr1 = tH
                pc0 = max(0, tW - rW); pc1 = tW
                rr0 = rH - (pr1 - pr0); rr1 = rH
                rc0 = rW - (pc1 - pc0); rc1 = rW
            elif tgt_name == "TR":
                pr0 = max(0, tH - rH); pr1 = tH
                pc0 = 0; pc1 = min(tW, rW)
                rr0 = rH - (pr1 - pr0); rr1 = rH
                rc0 = 0; rc1 = pc1 - pc0
            elif tgt_name == "BL":
                pr0 = 0; pr1 = min(tH, rH)
                pc0 = max(0, tW - rW); pc1 = tW
                rr0 = 0; rr1 = pr1 - pr0
                rc0 = rW - (pc1 - pc0); rc1 = rW
            else:  # BR
                pr0 = 0; pr1 = min(tH, rH)
                pc0 = 0; pc1 = min(tW, rW)
                rr0 = 0; rr1 = pr1 - pr0
                rc0 = 0; rc1 = pc1 - pc0

            if rr1 > rr0 and rc1 > rc0 and pr1 > pr0 and pc1 > pc0:
                patch[pr0:pr1, pc0:pc1] = ref[rr0:rr1, rc0:rc1]
            out[tr0:tr1, tc0:tc1] = patch

        for name in quads:
            if name != src_name:
                fill_quad(name)
        return out.tolist()

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._apply(pair["input"]) != pair["output"]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._apply(inp)
            if p is None:
                return None
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class ColorCanonicalLookupSolver(BaseSolver):
    """
    Tasks d4469b4b, 9110e3c5: build dominant_nonzero(train_input)→train_output map.
    Test: find dominant color, return stored canonical output.
    """
    @property
    def name(self): return "ColorCanonicalLookupSolver"

    def solve(self, ctx):
        train_map = {}
        for pair in ctx.train_pairs:
            dom = _dominant_nonzero(pair["input"])
            if dom is None:
                return None
            train_map[dom] = pair["output"]

        if not train_map:
            return None

        preds = []
        for inp in ctx.test_inputs:
            dom = _dominant_nonzero(inp)
            if dom not in train_map:
                return None
            p = train_map[dom]
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class CoordinateCompressSolver(BaseSolver):
    """
    Task bc1d5164: input always 5x7, output always 3x3.
    Active rows {0,1,4}→{0,1,2}; active cols {0,1,5,6}→{0,1,1,2}.
    Separator rows {2,3} and cols {2,3,4} are zero (skipped).
    """
    @property
    def name(self): return "CoordinateCompressSolver"

    def _compress(self, grid):
        g = np.array(grid)
        if g.shape != (5, 7):
            return None
        out = np.zeros((3, 3), dtype=g.dtype)
        # Row 2 is separator; rows {0,1,3,4} → {0,1,1,2}
        row_map = {0: 0, 1: 1, 3: 1, 4: 2}
        for r in range(5):
            if r not in row_map:
                continue
            out_r = row_map[r]
            for c in range(7):
                if c <= 1:
                    out_c = c
                elif c >= 5:
                    out_c = c - 4
                else:
                    continue
                val = int(g[r, c])
                if val != 0:
                    out[out_r, out_c] = val
        return out.tolist()

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._compress(pair["input"]) != pair["output"]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._compress(inp)
            if p is None:
                return None
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
