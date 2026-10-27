"""
advanced_solvers28.py — Six solvers for ARC-AGI-2

1. DiagonalBounceSolver      (task: e179c5f4) — diagonal path bouncing off walls
2. PerimeterFillSolver       (task: fc754716) — fill border with single cell color
3. CrossArmsCondenserSolver  (task: ae3edfdc) — condense arms 1 step toward center (multi-center)
4. LPathFillSolver           (task: e5790162) — fill L-path chain; stop at wall, turn at marker
5. FractalStairSolver        (task: d06dbe63) — fractal stair pattern from anchor 8
6. SelfTilingSolver          (task: 8f2ea7aa) — self-referential 3x3 tiling to 9x9
"""

from collections import Counter
from typing import List, Optional, Tuple

from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction, Grid
from src.solvers.registry import register_solver


def _verify_exact(pred: Grid, expected: Grid) -> bool:
    if len(pred) != len(expected):
        return False
    for r in range(len(pred)):
        if list(pred[r]) != list(expected[r]):
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 1. DiagonalBounceSolver  (e179c5f4)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class DiagonalBounceSolver(BaseSolver):
    """
    Single non-zero cell at bottom row.
    Trace `rows` steps upward diagonally:
      - initial col direction = +1 if start_col < cols//2 else -1
      - flip direction when hitting left/right wall
    Path color = start_color; non-path = 8.
    """

    @property
    def name(self) -> str:
        return "DiagonalBounceSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])
        bg = 8

        nz = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] != 0]
        if len(nz) != 1:
            return None

        sr, sc = nz[0]
        start_color = inp[sr][sc]

        path = {(sr, sc)}
        r, c = sr, sc
        dc = 1 if sc < cols // 2 else -1

        for _ in range(rows):
            r -= 1
            c += dc
            if c < 0:
                c = -c
                dc = 1
            elif c >= cols:
                c = 2 * (cols - 1) - c
                dc = -1
            if r < 0:
                break
            path.add((r, c))

        out = [[bg] * cols for _ in range(rows)]
        for pr, pc in path:
            out[pr][pc] = start_color

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PerimeterFillSolver  (fc754716)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class PerimeterFillSolver(BaseSolver):
    """
    Single non-zero cell → fill all border cells with that color.
    Interior remains 0.
    """

    @property
    def name(self) -> str:
        return "PerimeterFillSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])

        nz = [(r, c, inp[r][c]) for r in range(rows) for c in range(cols) if inp[r][c] != 0]
        if len(nz) != 1:
            return None

        _, _, color = nz[0]
        out = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    out[r][c] = color

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CrossArmsCondenserSolver  (ae3edfdc)
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class CrossArmsCondenserSolver(BaseSolver):
    """
    Centers = any color appearing exactly once among non-zero cells.
    Multiple centers allowed (one per cross-arm cluster).
    Arms = non-zero, non-center cells on same row or col as any center.
    Each arm moves 1 step adjacent to its aligned center along that axis;
    original arm position erased.
    If an arm is aligned with multiple centers, picks the closest one.
    """

    @property
    def name(self) -> str:
        return "CrossArmsCondenserSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])

        color_count = Counter(
            inp[r][c] for r in range(rows) for c in range(cols) if inp[r][c] != 0
        )
        center_colors = {color for color, cnt in color_count.items() if cnt == 1}
        if not center_colors:
            return None

        # Map center color → position
        center_pos = {}
        for r in range(rows):
            for c in range(cols):
                v = inp[r][c]
                if v in center_colors:
                    center_pos[v] = (r, c)

        erase = []
        place = []

        for r in range(rows):
            for c in range(cols):
                v = inp[r][c]
                if v == 0 or v in center_colors:
                    continue

                # Find all centers this arm is aligned with (same row or col)
                aligned = []
                for cc_color, (cr, cc) in center_pos.items():
                    if r == cr:
                        aligned.append(('row', cr, cc, abs(c - cc)))
                    elif c == cc:
                        aligned.append(('col', cr, cc, abs(r - cr)))

                if not aligned:
                    continue

                # Pick closest center
                aligned.sort(key=lambda x: x[3])
                axis, cr, cc, _ = aligned[0]

                if axis == 'row':
                    direction = 1 if c > cc else -1
                    new_c = cc + direction
                    if 0 <= new_c < cols:
                        erase.append((r, c))
                        place.append((r, new_c, v))
                else:  # col
                    direction = 1 if r > cr else -1
                    new_r = cr + direction
                    if 0 <= new_r < rows:
                        erase.append((r, c))
                        place.append((new_r, c, v))

        out = [list(row) for row in inp]
        for r, c in erase:
            out[r][c] = 0
        for r, c, v in place:
            out[r][c] = v

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. LPathFillSolver  (e5790162)
# ─────────────────────────────────────────────────────────────────────────────
#
# Rules (confirmed from all 5 training pairs):
#   - Source = color-3 cell
#   - Markers = all other non-zero, non-3 cells
#   - Each leg fills cells in one direction (H or V) until:
#       * Hitting a MARKER (stop 1 before it) → turn 90° and continue
#       * Hitting GRID WALL (stop at last valid cell) → record candidate, DONE
#   - DFS tries all 4 start directions; alternates H/V after each turn.
#   - Training requires at least one candidate matches expected output.
# ─────────────────────────────────────────────────────────────────────────────

def _lpath_candidates(inp, rows, cols):
    """
    Deterministic L-path simulation using marker-color turn rule:
      - Source = color-3 cell (always at column 0); start direction H_RIGHT
      - H_RIGHT leg: fill rightward until marker or wall
          * Wall hit  -> DONE (terminate)
          * Marker 8  -> next direction = V_UP
          * Marker 6  -> next direction = V_DOWN
      - V_UP / V_DOWN leg: fill up/down until marker or wall
          * Wall hit  -> DONE (terminate)
          * Any marker -> next direction = H_RIGHT
      Returns list with a single result grid, or [] if source not found.
    """
    CHAIN_COLOR = 3

    src = None
    marker_map = {}  # (r, c) -> color
    for r in range(rows):
        for c in range(cols):
            v = inp[r][c]
            if v == CHAIN_COLOR:
                src = (r, c)
            elif v != 0:
                marker_map[(r, c)] = v

    if src is None:
        return []

    grid = [list(row) for row in inp]

    DIRS = {'H_RIGHT': (0, 1), 'V_UP': (-1, 0), 'V_DOWN': (1, 0)}

    r, c = src
    direction = 'H_RIGHT'

    for _ in range(50):  # max 50 legs; safety limit
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        cells_filled = []

        while True:
            if not (0 <= nr < rows and 0 <= nc < cols):
                # Hit grid wall — fill accumulated cells and terminate
                for fr, fc in cells_filled:
                    grid[fr][fc] = CHAIN_COLOR
                return [grid]
            if (nr, nc) in marker_map:
                # Hit marker — fill accumulated cells, then turn
                for fr, fc in cells_filled:
                    grid[fr][fc] = CHAIN_COLOR
                if cells_filled:
                    r, c = cells_filled[-1]
                # Determine next direction from marker color
                if direction == 'H_RIGHT':
                    direction = 'V_UP' if marker_map[(nr, nc)] == 8 else 'V_DOWN'
                else:
                    direction = 'H_RIGHT'
                break  # start new leg
            else:
                cells_filled.append((nr, nc))
                nr += dr
                nc += dc

    # Safety fallback: return whatever we have
    return [grid]


@register_solver
class LPathFillSolver(BaseSolver):
    """
    Source color-3 cell traces an L-shaped chain:
    - Each leg extends until hitting a marker (turns) or grid wall (stops).
    - Alternates H/V direction after each turn.
    Multiple candidates generated; all training pairs must match at least one.
    """

    @property
    def name(self) -> str:
        return "LPathFillSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            inp = pair["input"]
            rows, cols = len(inp), len(inp[0])
            candidates = _lpath_candidates(inp, rows, cols)
            expected = pair["output"]
            if not any(_verify_exact(cand, expected) for cand in candidates):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            rows, cols = len(test_inp), len(test_inp[0])
            candidates = _lpath_candidates(test_inp, rows, cols)
            if not candidates:
                return None
            a1 = candidates[0]
            a2 = candidates[1] if len(candidates) > 1 else candidates[0]
            predictions.append(Prediction(attempt_1=a1, attempt_2=a2,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FractalStairSolver  (d06dbe63)
# ─────────────────────────────────────────────────────────────────────────────
#
# Anchor = single cell with value 8.
# UP staircase (from anchor, going up row by row):
#   odd step k: single cell at (ar-k, cur_col)
#   even step k: 3-wide bar at (ar-k, cur_col..cur_col+2), then cur_col += 2
# DOWN staircase (from anchor, going down):
#   odd step k: single cell at (ar+k, cur_col)
#   even step k: 3-wide bar at (ar+k, cur_col-2..cur_col), then cur_col -= 2
# Background = 0; stair path cells = 5; anchor = 8.
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class FractalStairSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "FractalStairSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])

        anchor = None
        for r in range(rows):
            for c in range(cols):
                if inp[r][c] == 8:
                    if anchor is not None:
                        return None  # multiple 8s
                    anchor = (r, c)

        if anchor is None:
            return None

        ar, ac = anchor
        path = set()  # stair cells (value 5); anchor handled separately

        # UP direction
        cur_col = ac
        for step in range(1, rows * 2 + cols * 2):
            r = ar - step
            if r < 0:
                break
            if step % 2 == 1:  # odd: single cell
                if not (0 <= cur_col < cols):
                    break
                path.add((r, cur_col))
            else:  # even: 3-wide bar then cur_col += 2
                bar_cols = [cur_col + dc for dc in range(3) if 0 <= cur_col + dc < cols]
                if not bar_cols:
                    break
                for bc in bar_cols:
                    path.add((r, bc))
                cur_col += 2
                if cur_col >= cols:
                    break

        # DOWN direction
        cur_col = ac
        for step in range(1, rows * 2 + cols * 2):
            r = ar + step
            if r >= rows:
                break
            if step % 2 == 1:  # odd: single cell
                if not (0 <= cur_col < cols):
                    break
                path.add((r, cur_col))
            else:  # even: 3-wide bar (cur_col-2..cur_col) then cur_col -= 2
                bar_cols = [cur_col - 2 + dc for dc in range(3) if 0 <= cur_col - 2 + dc < cols]
                if not bar_cols:
                    break
                for bc in bar_cols:
                    path.add((r, bc))
                cur_col -= 2
                if cur_col < 0:
                    break

        # Build output: background=0, path=5, anchor=8
        out = [[0] * cols for _ in range(rows)]
        for pr, pc in path:
            out[pr][pc] = 5
        out[ar][ac] = 8

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. SelfTilingSolver  (8f2ea7aa)
# ─────────────────────────────────────────────────────────────────────────────
#
# Input is a 9x9 grid with a small 3x3 pattern in exactly ONE of the nine
# 3x3 blocks.
# For each non-zero cell (pr, pc) in that 3x3 pattern:
#   → tile the FULL 3x3 pattern into the 3x3 block at position (pr, pc) of the 9x9 output.
# Zero cells in pattern → corresponding 9x9 block stays all-zero.
# ─────────────────────────────────────────────────────────────────────────────

@register_solver
class SelfTilingSolver(BaseSolver):

    @property
    def name(self) -> str:
        return "SelfTilingSolver"

    def _apply(self, inp: Grid) -> Optional[Grid]:
        rows, cols = len(inp), len(inp[0])
        if rows != 9 or cols != 9:
            return None

        # Find which 3x3 block(s) contain non-zero cells
        block_has_nz = set()
        for r in range(9):
            for c in range(9):
                if inp[r][c] != 0:
                    block_has_nz.add((r // 3, c // 3))

        if len(block_has_nz) != 1:
            return None  # all non-zero cells must be in exactly one 3x3 block

        br, bc = block_has_nz.pop()

        # Extract the 3x3 pattern from that block
        pattern = [
            [inp[br * 3 + pr][bc * 3 + pc] for pc in range(3)]
            for pr in range(3)
        ]

        # Tile: for each non-zero cell (pr, pc) in pattern,
        # copy the full pattern to block (pr, pc) in the 9x9 output
        out = [[0] * 9 for _ in range(9)]
        for pr in range(3):
            for pc in range(3):
                if pattern[pr][pc] != 0:
                    for qr in range(3):
                        for qc in range(3):
                            out[pr * 3 + qr][pc * 3 + qc] = pattern[qr][qc]

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        train = ctx.train_pairs
        if not train:
            return None

        for pair in train:
            pred = self._apply(pair["input"])
            if pred is None or not _verify_exact(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._apply(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name))

        return SolverResult(task_id=ctx.task_id, predictions=predictions,
                            solved=True, tier=1)
