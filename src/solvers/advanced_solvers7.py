"""advanced_solvers7.py — Block downsample and related shrink solvers."""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..core.grid_codec import background_color
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver


# ── helpers ──────────────────────────────────────────────────────────────────
def _sh(g):  return (len(g), len(g[0])) if g else (0, 0)
def _bg(g):  return background_color(g)
def _clone(g): return [list(r) for r in g]
def _np(g):  return np.array(g, dtype=np.int32)
def _to_grid(a): return a.tolist()
def _grids_equal(a, b): return np.array_equal(_np(a), _np(b))
def _same_shape(a, b): return _sh(a) == _sh(b)

def _result(ctx, grids, name):
    preds = []
    for g in grids:
        preds.append(Prediction(attempt_1=g, attempt_2=g, solver_name=name))
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# BlockDownsampleSolver
# Pattern: Input is H*S × W*S, output is H × W.
# Each S×S block in input → majority non-background color → output cell.
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class BlockDownsampleSolver(BaseSolver):
    """Majority non-bg color per NxN block → downsampled output."""

    @property
    def name(self) -> str:
        return "BlockDownsampleSolver"

    def _get_scale(self, inp_shape, out_shape):
        ih, iw = inp_shape
        oh, ow = out_shape
        if oh == 0 or ow == 0:
            return None
        if ih % oh != 0 or iw % ow != 0:
            return None
        sh, sw = ih // oh, iw // ow
        if sh != sw or sh < 2:
            return None
        return sh

    def _apply(self, inp, scale):
        a = _np(inp)
        bg = _bg(inp)
        oh = a.shape[0] // scale
        ow = a.shape[1] // scale
        out = np.full((oh, ow), bg, dtype=np.int32)
        for r in range(oh):
            for c in range(ow):
                block = a[r*scale:(r+1)*scale, c*scale:(c+1)*scale].flatten()
                non_bg = [v for v in block if v != bg]
                if non_bg:
                    out[r, c] = Counter(non_bg).most_common(1)[0][0]
        return _to_grid(out)

    def solve(self, ctx: TaskContext):
        # Each pair can have its own scale (output size may be fixed while
        # input size varies, making scale vary too).
        train_scales = []
        train_out_shapes = []
        for p in ctx.train_pairs:
            s = self._get_scale(_sh(p['input']), _sh(p['output']))
            if s is None:
                return None
            train_scales.append(s)
            train_out_shapes.append(_sh(p['output']))

        if not train_scales:
            return None

        # Verify all train pairs with their per-pair scale
        for p, s in zip(ctx.train_pairs, train_scales):
            pred = self._apply(p['input'], s)
            if not _grids_equal(pred, p['output']):
                return None

        # Determine output shape for test inputs:
        # Strategy 1: all train outputs have the same shape → fixed output size
        unique_out_shapes = set(train_out_shapes)
        fixed_out_shape = next(iter(unique_out_shapes)) if len(unique_out_shapes) == 1 else None

        grids = []
        for ti in ctx.test_inputs:
            ti_sh = _sh(ti)
            if fixed_out_shape is not None:
                # infer scale from test input / fixed output
                s = self._get_scale(ti_sh, fixed_out_shape)
                if s is None:
                    return None
            else:
                # all train scales are the same → use that scale
                unique_scales = set(train_scales)
                if len(unique_scales) != 1:
                    return None
                s = next(iter(unique_scales))
            grids.append(self._apply(ti, s))

        return _result(ctx, grids, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# BlockDownsampleModeAllSolver
# Variant: majority of ALL colors (not just non-bg). Catches edge cases.
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class BlockDownsampleModeAllSolver(BaseSolver):
    """Majority of all colors (including bg) per NxN block → downsampled output."""

    @property
    def name(self) -> str:
        return "BlockDownsampleModeAllSolver"

    def _get_scale(self, inp_shape, out_shape):
        ih, iw = inp_shape
        oh, ow = out_shape
        if oh == 0 or ow == 0:
            return None
        if ih % oh != 0 or iw % ow != 0:
            return None
        sh, sw = ih // oh, iw // ow
        if sh != sw or sh < 2:
            return None
        return sh

    def _apply(self, inp, scale):
        a = _np(inp)
        oh = a.shape[0] // scale
        ow = a.shape[1] // scale
        out = np.zeros((oh, ow), dtype=np.int32)
        for r in range(oh):
            for c in range(ow):
                block = a[r*scale:(r+1)*scale, c*scale:(c+1)*scale].flatten()
                out[r, c] = Counter(block.tolist()).most_common(1)[0][0]
        return _to_grid(out)

    def solve(self, ctx: TaskContext):
        scale = None
        for p in ctx.train_pairs:
            ih, iw = _sh(p['input'])
            oh, ow = _sh(p['output'])
            if oh == 0 or ow == 0: return None
            if ih % oh != 0 or iw % ow != 0: return None
            sh, sw = ih // oh, iw // ow
            if sh != sw or sh < 2: return None
            if scale is None:
                scale = sh
            elif scale != sh:
                return None

        if scale is None:
            return None

        for p in ctx.train_pairs:
            pred = self._apply(p['input'], scale)
            if not _grids_equal(pred, p['output']):
                return None

        grids = [self._apply(ti, scale) for ti in ctx.test_inputs]
        return _result(ctx, grids, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# UniqueColorPerBlockSolver
# Each block contains exactly one non-bg color → output cell = that color.
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class UniqueColorPerBlockSolver(BaseSolver):
    """Each NxN block has exactly one distinct non-bg color → output cell."""

    @property
    def name(self) -> str:
        return "UniqueColorPerBlockSolver"

    def _get_scale(self, inp_shape, out_shape):
        ih, iw = inp_shape
        oh, ow = out_shape
        if oh == 0 or ow == 0: return None
        if ih % oh != 0 or iw % ow != 0: return None
        sh, sw = ih // oh, iw // ow
        if sh != sw or sh < 2: return None
        return sh

    def _apply(self, inp, scale):
        a = _np(inp)
        bg = _bg(inp)
        oh = a.shape[0] // scale
        ow = a.shape[1] // scale
        out = np.full((oh, ow), bg, dtype=np.int32)
        for r in range(oh):
            for c in range(ow):
                block = a[r*scale:(r+1)*scale, c*scale:(c+1)*scale].flatten()
                non_bg = set(v for v in block if v != bg)
                if len(non_bg) == 1:
                    out[r, c] = next(iter(non_bg))
                elif len(non_bg) > 1:
                    # fail this block — return None signal
                    return None
        return _to_grid(out)

    def solve(self, ctx: TaskContext):
        scale = None
        for p in ctx.train_pairs:
            s = self._get_scale(_sh(p['input']), _sh(p['output']))
            if s is None: return None
            if scale is None: scale = s
            elif scale != s: return None

        if scale is None: return None

        for p in ctx.train_pairs:
            pred = self._apply(p['input'], scale)
            if pred is None or not _grids_equal(pred, p['output']):
                return None

        grids = []
        for ti in ctx.test_inputs:
            g = self._apply(ti, scale)
            if g is None: return None
            grids.append(g)

        return _result(ctx, grids, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# RectangularBlockDownsampleSolver
# Like BlockDownsample but allows non-square scale (SH x SW blocks).
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class RectangularBlockDownsampleSolver(BaseSolver):
    """Majority non-bg per rectangular SH×SW block → downsampled cell."""

    @property
    def name(self) -> str:
        return "RectangularBlockDownsampleSolver"

    def _get_scale(self, inp_shape, out_shape):
        ih, iw = inp_shape
        oh, ow = out_shape
        if oh == 0 or ow == 0: return None
        if ih % oh != 0 or iw % ow != 0: return None
        sh, sw = ih // oh, iw // ow
        if sh < 2 and sw < 2: return None  # both 1 = no downsample
        if sh == sw: return None  # handled by BlockDownsampleSolver
        return sh, sw

    def _apply(self, inp, sh, sw):
        a = _np(inp)
        bg = _bg(inp)
        oh = a.shape[0] // sh
        ow = a.shape[1] // sw
        out = np.full((oh, ow), bg, dtype=np.int32)
        for r in range(oh):
            for c in range(ow):
                block = a[r*sh:(r+1)*sh, c*sw:(c+1)*sw].flatten()
                non_bg = [v for v in block if v != bg]
                if non_bg:
                    out[r, c] = Counter(non_bg).most_common(1)[0][0]
        return _to_grid(out)

    def solve(self, ctx: TaskContext):
        scale = None
        for p in ctx.train_pairs:
            s = self._get_scale(_sh(p['input']), _sh(p['output']))
            if s is None: return None
            if scale is None: scale = s
            elif scale != s: return None

        if scale is None: return None
        sh, sw = scale

        for p in ctx.train_pairs:
            pred = self._apply(p['input'], sh, sw)
            if not _grids_equal(pred, p['output']):
                return None

        grids = [self._apply(ti, sh, sw) for ti in ctx.test_inputs]
        return _result(ctx, grids, self.name)
