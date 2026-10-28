"""advanced_solvers11.py — 5 targeted solvers from pattern analysis.

Patterns implemented:
1. HorizontalSeparatorStreakSolver  (task 8597cfd7)
2. TwoObjectTileSolver              (task 681b3aeb + 3 others)
3. KeepMostCommonRecolorFiveSolver  (task 9565186b)
4. SplitHalfBothZeroMark2Solver     (tasks fafffa47, 94f9d214)
5. ColSeparatorAndMark2Solver       (task 0520fde7)
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..core.grid_codec import background_color
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver


# ─────────────────────────────────────────────────────────────
# 1. HorizontalSeparatorStreakSolver
#    Full-row of a single non-bg color acts as separator.
#    Count each streak-color in rows ABOVE and BELOW the separator.
#    Winner = color with max(below_count − above_count).
#    Output = 2×2 grid filled with winner color.
# ─────────────────────────────────────────────────────────────
@register_solver
class HorizontalSeparatorStreakSolver(BaseSolver):
    name = "HorizontalSeparatorStreakSolver"
    tier = 1

    def _find_separator_row(self, inp: np.ndarray):
        """Return (row_index, sep_color) or None."""
        H, W = inp.shape
        bg = background_color(inp.tolist())
        for r in range(H):
            row = inp[r]
            unique = set(row.tolist())
            if len(unique) == 1:
                c = int(row[0])
                if c != bg:
                    return r, c
        return None

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        inp = np.asarray(inp)
        result = self._find_separator_row(inp)
        if result is None:
            raise ValueError("No separator row found")
        sep_row, sep_color = result
        bg = background_color(inp.tolist())

        above = inp[:sep_row]
        below = inp[sep_row + 1:]

        # count non-bg, non-sep colors
        def count_colors(region):
            cnt = Counter()
            for val in region.flatten():
                v = int(val)
                if v != bg and v != sep_color:
                    cnt[v] += 1
            return cnt

        above_cnt = count_colors(above)
        below_cnt = count_colors(below)

        all_colors = set(above_cnt.keys()) | set(below_cnt.keys())
        if not all_colors:
            raise ValueError("No streak colors found")

        winner = max(all_colors, key=lambda c: below_cnt.get(c, 0) - above_cnt.get(c, 0))
        return np.full((2, 2), winner, dtype=int)

    def _verify(self, inp: np.ndarray, out: np.ndarray) -> bool:
        try:
            pred = self._apply(inp)
            return pred.shape == np.asarray(out).shape and np.array_equal(pred, np.asarray(out))
        except Exception:
            return False

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not all(self._verify(np.asarray(p['input']), np.asarray(p['output'])) for p in pairs):
            return None
        results = []
        for ti in ctx.test_inputs:
            try:
                g = self._apply(np.asarray(ti))
                results.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
            except Exception:
                return None
        return SolverResult(task_id=ctx.task_id, predictions=results, solved=True, tier=self.tier)


# ─────────────────────────────────────────────────────────────
# 2. TwoObjectTileSolver
#    Input has exactly two non-bg connected components.
#    Their cell counts sum to 9 (i.e., they tile a 3×3 perfectly).
#    Brute-force find placement of both shapes inside 3×3 with
#    no overlap and full coverage.  Output = that 3×3 grid.
# ─────────────────────────────────────────────────────────────
def _connected_components(grid: np.ndarray, bg: int):
    """Return list of (color, set_of_(r,c)) for each connected component."""
    H, W = grid.shape
    visited = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            v = int(grid[r, c])
            if v == bg or visited[r, c]:
                continue
            # BFS
            cells = set()
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                if visited[rr, cc] or int(grid[rr, cc]) != v:
                    continue
                visited[rr, cc] = True
                cells.add((rr, cc))
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    stack.append((rr+dr, cc+dc))
            comps.append((v, cells))
    return comps


def _normalize_cells(cells):
    """Shift cells so top-left is (0,0)."""
    min_r = min(r for r, c in cells)
    min_c = min(c for r, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


@register_solver
class TwoObjectTileSolver(BaseSolver):
    name = "TwoObjectTileSolver"
    tier = 1

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        inp = np.asarray(inp)
        bg = background_color(inp.tolist())
        comps = _connected_components(inp, bg)

        # Must have exactly 2 non-bg components summing to 9 cells
        if len(comps) != 2:
            raise ValueError(f"Expected 2 components, got {len(comps)}")
        (c1, cells1), (c2, cells2) = comps
        if len(cells1) + len(cells2) != 9:
            raise ValueError(f"Cell counts {len(cells1)}+{len(cells2)} != 9")

        norm1 = _normalize_cells(cells1)
        norm2 = _normalize_cells(cells2)

        # Brute-force all offsets for shape1 and shape2 in 3×3
        for dr1 in range(3):
            for dc1 in range(3):
                placed1 = frozenset((r + dr1, c + dc1) for r, c in norm1)
                if any(r >= 3 or c >= 3 for r, c in placed1):
                    continue
                for dr2 in range(3):
                    for dc2 in range(3):
                        placed2 = frozenset((r + dr2, c + dc2) for r, c in norm2)
                        if any(r >= 3 or c >= 3 for r, c in placed2):
                            continue
                        if len(placed1 | placed2) == 9 and len(placed1 & placed2) == 0:
                            out = np.zeros((3, 3), dtype=int)
                            for r, c in placed1:
                                out[r, c] = c1
                            for r, c in placed2:
                                out[r, c] = c2
                            return out
        raise ValueError("No valid tiling found")

    def _verify(self, inp: np.ndarray, out: np.ndarray) -> bool:
        try:
            pred = self._apply(inp)
            return np.array_equal(pred, np.asarray(out))
        except Exception:
            return False

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not all(self._verify(np.asarray(p['input']), np.asarray(p['output'])) for p in pairs):
            return None
        results = []
        for ti in ctx.test_inputs:
            try:
                g = self._apply(np.asarray(ti))
                results.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
            except Exception:
                return None
        return SolverResult(task_id=ctx.task_id, predictions=results, solved=True, tier=self.tier)


# ─────────────────────────────────────────────────────────────
# 3. KeepMostCommonRecolorFiveSolver
#    Input has no background zeros (all cells non-zero).
#    Find the most-frequent color; replace all others with 5.
#    Output is same shape as input.
# ─────────────────────────────────────────────────────────────
@register_solver
class KeepMostCommonRecolorFiveSolver(BaseSolver):
    name = "KeepMostCommonRecolorFiveSolver"
    tier = 1

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        inp = np.asarray(inp)
        if np.any(inp == 0):
            raise ValueError("Input contains zeros")
        cnt = Counter(inp.flatten().tolist())
        most_common = cnt.most_common(1)[0][0]
        out = np.where(inp == most_common, inp, 5)
        return out.astype(int)

    def _verify(self, inp: np.ndarray, out: np.ndarray) -> bool:
        try:
            pred = self._apply(inp)
            return np.array_equal(pred, np.asarray(out))
        except Exception:
            return False

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not all(self._verify(np.asarray(p['input']), np.asarray(p['output'])) for p in pairs):
            return None
        results = []
        for ti in ctx.test_inputs:
            try:
                g = self._apply(np.asarray(ti))
                results.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
            except Exception:
                return None
        return SolverResult(task_id=ctx.task_id, predictions=results, solved=True, tier=self.tier)


# ─────────────────────────────────────────────────────────────
# 4. SplitHalfBothZeroMark2Solver
#    Input height must be even. top = inp[:H//2], bot = inp[H//2:].
#    Output = 2 where BOTH top and bot are 0, else 0.
#    Output shape = top.shape.
# ─────────────────────────────────────────────────────────────
@register_solver
class SplitHalfBothZeroMark2Solver(BaseSolver):
    name = "SplitHalfBothZeroMark2Solver"
    tier = 1

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        inp = np.asarray(inp)
        H, W = inp.shape
        if H % 2 != 0:
            raise ValueError("Height not even")
        top = inp[:H // 2]
        bot = inp[H // 2:]
        out = np.where((top == 0) & (bot == 0), 2, 0)
        return out.astype(int)

    def _verify(self, inp: np.ndarray, out: np.ndarray) -> bool:
        try:
            pred = self._apply(inp)
            return np.array_equal(pred, np.asarray(out))
        except Exception:
            return False

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not all(self._verify(np.asarray(p['input']), np.asarray(p['output'])) for p in pairs):
            return None
        results = []
        for ti in ctx.test_inputs:
            try:
                g = self._apply(np.asarray(ti))
                results.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
            except Exception:
                return None
        return SolverResult(task_id=ctx.task_id, predictions=results, solved=True, tier=self.tier)


# ─────────────────────────────────────────────────────────────
# 5. ColSeparatorAndMark2Solver
#    A full-column of a single non-bg color acts as separator.
#    left = inp[:, :sep_col], right = inp[:, sep_col+1:].
#    Output = 2 where BOTH halves are non-zero, else 0.
#    Output shape = left.shape (same as right.shape since widths equal).
# ─────────────────────────────────────────────────────────────
@register_solver
class ColSeparatorAndMark2Solver(BaseSolver):
    name = "ColSeparatorAndMark2Solver"
    tier = 1

    def _find_separator_col(self, inp: np.ndarray):
        """Return (col_index, sep_color) or None."""
        H, W = inp.shape
        bg = background_color(inp.tolist())
        for c in range(W):
            col = inp[:, c]
            unique = set(col.tolist())
            if len(unique) == 1:
                v = int(col[0])
                if v != bg:
                    return c, v
        return None

    def _apply(self, inp: np.ndarray) -> np.ndarray:
        inp = np.asarray(inp)
        result = self._find_separator_col(inp)
        if result is None:
            raise ValueError("No separator column found")
        sep_col, sep_color = result
        background_color(inp.tolist())

        left = inp[:, :sep_col]
        right = inp[:, sep_col + 1:]

        if left.shape != right.shape:
            raise ValueError(f"Left {left.shape} != right {right.shape}")

        out = np.where((left != 0) & (right != 0), 2, 0)
        return out.astype(int)

    def _verify(self, inp: np.ndarray, out: np.ndarray) -> bool:
        try:
            pred = self._apply(inp)
            return np.array_equal(pred, np.asarray(out))
        except Exception:
            return False

    def solve(self, ctx: TaskContext):
        pairs = ctx.train_pairs
        if not all(self._verify(np.asarray(p['input']), np.asarray(p['output'])) for p in pairs):
            return None
        results = []
        for ti in ctx.test_inputs:
            try:
                g = self._apply(np.asarray(ti))
                results.append(Prediction(attempt_1=g.tolist(), attempt_2=g.tolist(), solver_name=self.name))
            except Exception:
                return None
        return SolverResult(task_id=ctx.task_id, predictions=results, solved=True, tier=self.tier)
