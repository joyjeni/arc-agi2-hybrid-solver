"""
Advanced solvers batch 24.

Solvers:
  1. MetaGridCrossPropagateSolver  — task cbded52d
     Grid split by all-zero row/col dividers into equal sub-blocks.
     Propagate non-background values cross-wise when ≥2 blocks in same
     meta-row or meta-col share the same value at the same within-block position.

  2. AlternatingSpacedEchoSolver   — task 72207abc
     3-row grid; middle row has N non-zero seeds.  Echo formula:
       first_gap(rank r) = N*(N+1)//2 + r*N
       subsequent gaps increase by N² each step.

  3. LatinSquareCompletionSolver   — task 4cd1b7b2
     NxN grid with N distinct non-zero values; fill 0s to complete Latin square
     (each value appears exactly once per row and per column) via backtracking.

  4. VerticalBarSumSolver          — task 37ce87bb
     Background=7; k bottom-anchored vertical bars at odd columns with heights h_i.
     Output bar height = 5*k - sum(h_i), value=5, at column last_bar_col+2.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import List, Optional

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

Grid = List[List[int]]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _verify_exact(pred: Grid, expected: Grid) -> bool:
    return pred == expected


# ─────────────────────────────────────────────────────────────────────────────
# 1. MetaGridCrossPropagateSolver  (cbded52d)
# ─────────────────────────────────────────────────────────────────────────────

def _zero_rows(grid: Grid):
    return [r for r, row in enumerate(grid) if all(c == 0 for c in row)]


def _zero_cols(grid: Grid):
    if not grid:
        return []
    return [c for c in range(len(grid[0])) if all(grid[r][c] == 0 for r in range(len(grid)))]


def _decompose_meta_grid(grid: Grid):
    """Return (block_h, block_w, n_mr, n_mc, starts_r, starts_c) or None."""
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    zr = _zero_rows(grid)
    zc = _zero_cols(grid)
    if not zr or not zc:
        return None

    # Non-divider segment starts
    starts_r, prev = [], 0
    for r in zr:
        if r > prev:
            starts_r.append(prev)
        prev = r + 1
    if prev < rows:
        starts_r.append(prev)

    starts_c, prev = [], 0
    for c in zc:
        if c > prev:
            starts_c.append(prev)
        prev = c + 1
    if prev < cols:
        starts_c.append(prev)

    if not starts_r or not starts_c:
        return None

    # Block dims = distance from each block-start to the first following divider
    block_h = zr[0] - starts_r[0]
    block_w = zc[0] - starts_c[0]
    if block_h <= 0 or block_w <= 0:
        return None

    # Verify all segments have the same size
    seg_h = [
        (zr[i] - starts_r[i]) if i < len(zr) else (rows - starts_r[i])
        for i in range(len(starts_r))
    ]
    seg_w = [
        (zc[i] - starts_c[i]) if i < len(zc) else (cols - starts_c[i])
        for i in range(len(starts_c))
    ]
    if len(set(seg_h)) != 1 or len(set(seg_w)) != 1:
        return None

    return block_h, block_w, len(starts_r), len(starts_c), starts_r, starts_c


def _meta_background(grid: Grid) -> int:
    flat = [v for row in grid for v in row if v != 0]
    if not flat:
        return 0
    return Counter(flat).most_common(1)[0][0]


def _meta_propagate(grid: Grid) -> Optional[Grid]:
    result = _decompose_meta_grid(grid)
    if result is None:
        return None
    block_h, block_w, n_mr, n_mc, starts_r, starts_c = result
    bg = _meta_background(grid)

    out = copy.deepcopy(grid)

    for wr in range(block_h):
        for wc in range(block_w):
            # Collect non-bg values per meta-row and meta-col
            mr_vals: dict = {}
            mc_vals: dict = {}
            for mr in range(n_mr):
                for mc in range(n_mc):
                    gr = starts_r[mr] + wr
                    gc = starts_c[mc] + wc
                    v = grid[gr][gc]
                    if v != bg and v != 0:
                        mr_vals.setdefault(mr, []).append((mc, v))
                        mc_vals.setdefault(mc, []).append((mr, v))

            # Propagate along meta-rows
            for mr, entries in mr_vals.items():
                counts = Counter(val for _, val in entries)
                for val, cnt in counts.items():
                    if cnt >= 2:
                        for mc in range(n_mc):
                            out[starts_r[mr] + wr][starts_c[mc] + wc] = val

            # Propagate along meta-cols
            for mc, entries in mc_vals.items():
                counts = Counter(val for _, val in entries)
                for val, cnt in counts.items():
                    if cnt >= 2:
                        for mr in range(n_mr):
                            out[starts_r[mr] + wr][starts_c[mc] + wc] = val

    return out


@register_solver
class MetaGridCrossPropagateSolver(BaseSolver):
    """cbded52d: propagate shared values across meta-grid rows/cols."""

    @property
    def name(self) -> str:
        return "MetaGridCrossPropagateSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate on all training pairs
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _meta_propagate(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        # Apply to test inputs
        predictions = []
        for test_input in ctx.test_inputs:
            result = _meta_propagate(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. AlternatingSpacedEchoSolver  (72207abc)
# ─────────────────────────────────────────────────────────────────────────────

def _echo_solve(grid: Grid) -> Optional[Grid]:
    if len(grid) != 3:
        return None
    # Rows 0 and 2 must be all-zero
    if any(v != 0 for v in grid[0]) or any(v != 0 for v in grid[2]):
        return None

    mid = list(grid[1])
    width = len(mid)
    positions = sorted([(c, v) for c, v in enumerate(mid) if v != 0], key=lambda x: x[0])
    N = len(positions)
    if N == 0:
        return None

    step = N * N  # N²

    out_mid = list(mid)
    for rank, (init_col, val) in enumerate(positions):
        first_gap = N * (N + 1) // 2 + rank * N
        col = init_col
        k = 0
        while True:
            gap = first_gap + k * step
            col = col + gap
            if col >= width:
                break
            out_mid[col] = val
            k += 1

    return [list(grid[0]), out_mid, list(grid[2])]


@register_solver
class AlternatingSpacedEchoSolver(BaseSolver):
    """72207abc: 3-row grid; echo N seeds with step=N² spacing formula."""

    @property
    def name(self) -> str:
        return "AlternatingSpacedEchoSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _echo_solve(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _echo_solve(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. LatinSquareCompletionSolver  (4cd1b7b2)
# ─────────────────────────────────────────────────────────────────────────────

def _latin_solve(grid: Grid) -> Optional[Grid]:
    N = len(grid)
    if N == 0 or any(len(row) != N for row in grid):
        return None
    vals = sorted({v for row in grid for v in row if v != 0})
    if len(vals) != N:
        return None

    board = [list(row) for row in grid]

    def ok(r, c, v):
        if any(board[r][cc] == v for cc in range(N)):
            return False
        if any(board[rr][c] == v for rr in range(N)):
            return False
        return True

    def bt():
        for r in range(N):
            for c in range(N):
                if board[r][c] == 0:
                    for v in vals:
                        if ok(r, c, v):
                            board[r][c] = v
                            if bt():
                                return True
                            board[r][c] = 0
                    return False
        return True

    if not bt():
        return None
    return board


@register_solver
class LatinSquareCompletionSolver(BaseSolver):
    """4cd1b7b2: fill 0s to complete an NxN Latin square via backtracking."""

    @property
    def name(self) -> str:
        return "LatinSquareCompletionSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _latin_solve(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _latin_solve(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. VerticalBarSumSolver  (37ce87bb)
# ─────────────────────────────────────────────────────────────────────────────

_BAR_BG = 7
_BAR_OUT_VAL = 5


def _find_bar_cols(grid: Grid) -> list:
    rows, cols = len(grid), len(grid[0]) if grid else 0
    return [c for c in range(cols) if any(grid[r][c] != _BAR_BG for r in range(rows))]


def _bar_height(grid: Grid, col: int) -> int:
    rows = len(grid)
    h = 0
    for r in range(rows - 1, -1, -1):
        if grid[r][col] != _BAR_BG:
            h += 1
        else:
            break
    return h


def _barsum_solve(grid: Grid) -> Optional[Grid]:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    if rows < 2 or cols < 2:
        return None

    bar_cols = _find_bar_cols(grid)
    if not bar_cols:
        return None

    # Bars must sit at consecutive odd columns starting at 1
    expected_cols = list(range(1, 1 + 2 * len(bar_cols), 2))
    if bar_cols != expected_cols:
        return None

    k = len(bar_cols)
    heights = [_bar_height(grid, c) for c in bar_cols]
    if any(h == 0 for h in heights):
        return None

    out_h = 5 * k - sum(heights)
    out_col = bar_cols[-1] + 2

    if out_h <= 0 or out_h > rows or out_col >= cols:
        return None

    out = [[_BAR_BG] * cols for _ in range(rows)]
    # Copy input bars
    for bc, bh in zip(bar_cols, heights):
        for r in range(rows - bh, rows):
            out[r][bc] = grid[r][bc]
    # Output bar
    for r in range(rows - out_h, rows):
        out[r][out_col] = _BAR_OUT_VAL

    return out


@register_solver
class VerticalBarSumSolver(BaseSolver):
    """37ce87bb: background=7; output bar height = 5*k - sum(input_heights), value=5."""

    @property
    def name(self) -> str:
        return "VerticalBarSumSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _barsum_solve(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _barsum_solve(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name=self.name,
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Export (for direct use in benchmarks)
# ─────────────────────────────────────────────────────────────────────────────

ADVANCED_SOLVERS_24 = [
    MetaGridCrossPropagateSolver(),
    AlternatingSpacedEchoSolver(),
    LatinSquareCompletionSolver(),
    VerticalBarSumSolver(),
]
