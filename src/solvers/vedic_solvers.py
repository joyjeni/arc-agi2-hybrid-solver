"""
Vedic & Chinese-inspired solvers — Tier-1 pattern solvers.

Naming follows traditional sutra / trigram concepts but implements
concrete DSA algorithms underneath, keeping primary logic obfuscated.
"""

from __future__ import annotations

from typing import Optional

from ..core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from ..core.grid_codec import (
    Grid, copy_grid, background_color, color_histogram,
    rotate90, flip_h, transpose,
)
from .registry import register_solver


def _mk(grid: Grid, name: str, conf: float = 1.0) -> Prediction:
    return Prediction(attempt_1=grid, attempt_2=grid, solver_name=name, confidence=conf)


def _res(tid: str, preds: list[Prediction]) -> SolverResult:
    return SolverResult(task_id=tid, predictions=preds, solved=True, tier=1)


# ── Anurupya: proportional scaling ─────────────────────────────────────────────

@register_solver
class AnurupyaScaleSolver(BaseSolver):
    """
    Anurupya (proportion): output is a scaled (upsampled) version of input.
    Learns integer scale factor k from training pairs.
    """

    @property
    def name(self) -> str:
        return "AnurupyaScaleSolver"

    @property
    def priority(self) -> int:
        return 20

    @staticmethod
    def _scale(grid: Grid, k: int) -> Grid:
        return [[grid[r][c] for c in range(len(grid[0])) for _ in range(k)]
                for r in range(len(grid)) for _ in range(k)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for k in range(2, 8):
            ok = True
            for pair in ctx.train_pairs:
                ih, iw = len(pair["input"]), len(pair["input"][0])
                oh, ow = len(pair["output"]), len(pair["output"][0])
                if oh != ih * k or ow != iw * k:
                    ok = False; break
                if self._scale(pair["input"], k) != pair["output"]:
                    ok = False; break
            if ok:
                preds = [_mk(self._scale(inp, k), self.name) for inp in ctx.test_inputs]
                return _res(ctx.task_id, preds)
        return None


# ── Paravartya: reflection/inversion ──────────────────────────────────────────

@register_solver
class ParavartyaSymmetrySolver(BaseSolver):
    """
    Paravartya (transpose/reverse): output completes an axis-symmetric pattern.
    Detects which axis (H/V/D) and fills in the missing half.
    """

    @property
    def name(self) -> str:
        return "ParavartyaSymmetrySolver"

    @property
    def priority(self) -> int:
        return 22

    @staticmethod
    def _h_sym(grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        result = copy_grid(grid)
        for r in range(h):
            for c in range(w):
                mirror = grid[r][w - 1 - c]
                if result[r][c] == 0 and mirror != 0:
                    result[r][c] = mirror
        return result

    @staticmethod
    def _v_sym(grid: Grid) -> Grid:
        h = len(grid)
        result = copy_grid(grid)
        for r in range(h):
            for c in range(len(grid[0])):
                mirror = grid[h - 1 - r][c]
                if result[r][c] == 0 and mirror != 0:
                    result[r][c] = mirror
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for fn in [self._h_sym, self._v_sym]:
            if all(fn(p["input"]) == p["output"] for p in ctx.train_pairs):
                preds = [_mk(fn(inp), self.name) for inp in ctx.test_inputs]
                return _res(ctx.task_id, preds)
        return None


# ── Vyashti-Samashti: individual→aggregate tiling ────────────────────────────

@register_solver
class VyashtiTileSolver(BaseSolver):
    """
    Vyashti-Samashti (individual→collective): output tiles the input
    in a 2×2 or 3×3 pattern.
    """

    @property
    def name(self) -> str:
        return "VyashtiTileSolver"

    @property
    def priority(self) -> int:
        return 25

    @staticmethod
    def _tile(grid: Grid, rows: int, cols: int) -> Grid:
        return [
            [grid[r % len(grid)][c % len(grid[0])]
             for c in range(len(grid[0]) * cols)]
            for r in range(len(grid) * rows)
        ]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for tr in [2, 3]:
            for tc in [2, 3]:
                ok = True
                for pair in ctx.train_pairs:
                    expected = self._tile(pair["input"], tr, tc)
                    if expected != pair["output"]:
                        ok = False; break
                if ok:
                    preds = [_mk(self._tile(inp, tr, tc), self.name) for inp in ctx.test_inputs]
                    return _res(ctx.task_id, preds)
        return None


# ── Lopana: crop / bounding-box extraction ────────────────────────────────────

@register_solver
class LopanaBBoxSolver(BaseSolver):
    """
    Lopana (removal): output is the bounding-box crop of the non-background region.
    """

    @property
    def name(self) -> str:
        return "LopanaBBoxSolver"

    @property
    def priority(self) -> int:
        return 18

    @staticmethod
    def _crop(grid: Grid) -> Grid:
        bg = background_color(grid)
        rows = [r for r, row in enumerate(grid) if any(c != bg for c in row)]
        cols = [c for c in range(len(grid[0])) if any(grid[r][c] != bg for r in range(len(grid)))]
        if not rows or not cols:
            return grid
        return [grid[r][cols[0]:cols[-1]+1] for r in rows]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if all(self._crop(p["input"]) == p["output"] for p in ctx.train_pairs):
            preds = [_mk(self._crop(inp), self.name) for inp in ctx.test_inputs]
            return _res(ctx.task_id, preds)
        return None


# ── Ba-Gua: 8-fold transform search ──────────────────────────────────────────

@register_solver
class BaGuaTransformSolver(BaseSolver):
    """
    Ba-Gua (8 trigrams → 8 dihedral transforms): exhaustively try all 8
    symmetry transforms and return the one consistent with training pairs.
    """

    @property
    def name(self) -> str:
        return "BaGuaTransformSolver"

    @property
    def priority(self) -> int:
        return 16

    def _transforms(self):
        """Return (name, fn) pairs for all 8 dihedral transforms."""
        return [
            ("rot0",   lambda g: g),
            ("rot90",  rotate90),
            ("rot180", lambda g: rotate90(rotate90(g))),
            ("rot270", lambda g: rotate90(rotate90(rotate90(g)))),
            ("flipH",  flip_h),
            ("flipH_rot90",  lambda g: flip_h(rotate90(g))),
            ("flipH_rot180", lambda g: flip_h(rotate90(rotate90(g)))),
            ("flipH_rot270", lambda g: flip_h(rotate90(rotate90(rotate90(g))))),
        ]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for tname, fn in self._transforms():
            if tname == "rot0":
                continue  # identity handled by IdentitySolver
            if all(fn(p["input"]) == p["output"] for p in ctx.train_pairs):
                preds = [_mk(fn(inp), f"{self.name}:{tname}") for inp in ctx.test_inputs]
                return _res(ctx.task_id, preds)
        return None


# ── Zhong-Li: gravity with direction learning ─────────────────────────────────

@register_solver
class ZhongLiDirectedGravitySolver(BaseSolver):
    """
    Zhong-Li (central force): gravity in learned direction (up/down/left/right).
    """

    @property
    def name(self) -> str:
        return "ZhongLiDirectedGravitySolver"

    @property
    def priority(self) -> int:
        return 28

    @staticmethod
    def _gravity(grid: Grid, direction: str) -> Grid:
        h, w = len(grid), len(grid[0])
        bg = background_color(grid)
        result = [[bg] * w for _ in range(h)]

        if direction == "down":
            for c in range(w):
                col = [grid[r][c] for r in range(h) if grid[r][c] != bg]
                for i, v in enumerate(reversed(col)):
                    result[h - 1 - i][c] = v
        elif direction == "up":
            for c in range(w):
                col = [grid[r][c] for r in range(h) if grid[r][c] != bg]
                for i, v in enumerate(col):
                    result[i][c] = v
        elif direction == "right":
            for r in range(h):
                row = [grid[r][c] for c in range(w) if grid[r][c] != bg]
                for i, v in enumerate(reversed(row)):
                    result[r][w - 1 - i] = v
        elif direction == "left":
            for r in range(h):
                row = [grid[r][c] for c in range(w) if grid[r][c] != bg]
                for i, v in enumerate(row):
                    result[r][i] = v
        return result

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for d in ("down", "up", "left", "right"):
            if all(self._gravity(p["input"], d) == p["output"] for p in ctx.train_pairs):
                preds = [_mk(self._gravity(inp, d), self.name) for inp in ctx.test_inputs]
                return _res(ctx.task_id, preds)
        return None


# ── Lo-Shu: magic-square / periodic fill ─────────────────────────────────────

@register_solver
class LoShuPeriodicFillSolver(BaseSolver):
    """
    Lo-Shu (magic square cycles): output fills background cells using a
    periodic tiling derived from the non-background pattern.
    """

    @property
    def name(self) -> str:
        return "LoShuPeriodicFillSolver"

    @property
    def priority(self) -> int:
        return 35

    @staticmethod
    def _fill(grid: Grid) -> Grid:
        h, w = len(grid), len(grid[0])
        bg = background_color(grid)
        # find tile: smallest repeating sub-grid
        for th in range(1, h + 1):
            for tw in range(1, w + 1):
                if h % th != 0 or w % tw != 0:
                    continue
                tile = [grid[r][:tw] for r in range(th)]
                ok = True
                for r in range(h):
                    for c in range(w):
                        if grid[r][c] != bg and grid[r][c] != tile[r % th][c % tw]:
                            ok = False; break
                    if not ok:
                        break
                if ok:
                    return [[tile[r % th][c % tw] for c in range(w)] for r in range(h)]
        return copy_grid(grid)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            if self._fill(pair["input"]) != pair["output"]:
                return None
        preds = [_mk(self._fill(inp), self.name) for inp in ctx.test_inputs]
        return _res(ctx.task_id, preds)
