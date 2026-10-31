"""
advanced_solvers35.py — 6 new solvers targeting specific task patterns

Solvers:
  DiagonalXFromZeroSolver      — ea786f4a
  BoustrophedonColumnSolver     — cdecee7f
  FourCopiesTemplateSolver      — b1fc8b8e
  DiamondCompressionSolver      — ca8de6ea
  LShapeCornerAssemblySolver    — a61ba2ce
  NineBlockEdgeAssemblySolver   — a8c38be5 (best-effort, gated by train self-validation)
"""

from typing import Optional, List, Tuple
import numpy as np
from .registry import register_solver
from ..core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _grid_eq(a: List[List[int]], b: List[List[int]]) -> bool:
    return a == b


def _flip_v(g: List[List[int]]) -> List[List[int]]:
    return list(reversed(g))


def _flip_h(g: List[List[int]]) -> List[List[int]]:
    return [list(reversed(r)) for r in g]


def _rot90_ccw(g: List[List[int]]) -> List[List[int]]:
    """90° counter-clockwise rotation."""
    rows, cols = len(g), len(g[0])
    return [[g[r][c] for r in range(rows)] for c in range(cols - 1, -1, -1)]


def _rot90_cw(g: List[List[int]]) -> List[List[int]]:
    """90° clockwise rotation."""
    rows, cols = len(g), len(g[0])
    return [[g[rows - 1 - r][c] for r in range(rows)] for c in range(cols)]


def _sub(g: List[List[int]], r0: int, r1: int, c0: int, c1: int) -> List[List[int]]:
    """Extract sub-grid [r0:r1, c0:c1]."""
    return [row[c0:c1] for row in g[r0:r1]]


# ---------------------------------------------------------------------------
# 1. DiagonalXFromZeroSolver  (ea786f4a)
# ---------------------------------------------------------------------------
# N×N grid filled with one non-zero color C and exactly one 0 at (r0, c0).
# Output: out[r][c] = 0  if |r - r0| == |c - c0|  else C

@register_solver
class DiagonalXFromZeroSolver(BaseSolver):
    """
    Input: N×N grid, single background color C with exactly one 0 at (r0, c0).
    Output: draw two full diagonals crossing at (r0, c0) as 0; rest stays C.
    """

    @property
    def name(self) -> str:
        return "DiagonalXFromZeroSolver"

    def _predict(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        rows = len(inp)
        if rows == 0:
            return None
        cols = len(inp[0])
        if rows != cols:
            return None

        zeros = [(r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 0]
        if len(zeros) != 1:
            return None

        r0, c0 = zeros[0]
        # Background color: the unique non-zero value
        non_zero = {inp[r][c] for r in range(rows) for c in range(cols) if inp[r][c] != 0}
        if len(non_zero) != 1:
            return None
        C = next(iter(non_zero))

        out = [[C] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if abs(r - r0) == abs(c - c0):
                    out[r][c] = 0
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or not _grid_eq(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._predict(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 2. BoustrophedonColumnSolver  (cdecee7f)
# ---------------------------------------------------------------------------
# 10×10 input with ≤9 non-zero cells. Sort by col ASC (ties by row ASC).
# Fill 3×3 boustrophedon:
#   row 0  L→R : (0,0) (0,1) (0,2)
#   row 1  R→L : (1,2) (1,1) (1,0)
#   row 2  L→R : (2,0) (2,1) (2,2)

@register_solver
class BoustrophedonColumnSolver(BaseSolver):
    """
    Collect non-zero values from a 10×10 grid sorted by column (then row),
    and place them boustrophedon-style into a 3×3 output.
    """

    # Boustrophedon order: list of (out_row, out_col) for positions 0..8
    _ORDER = [
        (0, 0), (0, 1), (0, 2),   # row 0 L→R
        (1, 2), (1, 1), (1, 0),   # row 1 R→L
        (2, 0), (2, 1), (2, 2),   # row 2 L→R
    ]

    @property
    def name(self) -> str:
        return "BoustrophedonColumnSolver"

    def _predict(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        rows, cols = len(inp), len(inp[0]) if inp else 0
        if rows != 10 or cols != 10:
            return None

        # Collect non-zero cells sorted by col ASC, row ASC
        cells = sorted(
            [(inp[r][c], r, c) for r in range(rows) for c in range(cols) if inp[r][c] != 0],
            key=lambda x: (x[2], x[1])  # sort by col, then row
        )
        if len(cells) > 9:
            return None

        out = [[0] * 3 for _ in range(3)]
        for idx, (val, _, _) in enumerate(cells):
            or_, oc = self._ORDER[idx]
            out[or_][oc] = val
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or not _grid_eq(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._predict(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 3. FourCopiesTemplateSolver  (b1fc8b8e)
# ---------------------------------------------------------------------------
# 6×6 input with only 0s and 8s. The 8-cells are the union (OR semantics,
# overlapping allowed) of 4 placements of a 2×2 template T.
# Output: 5×5 — T at TL, TR, BL, BR corners with zero separator at index 2.

@register_solver
class FourCopiesTemplateSolver(BaseSolver):
    """
    Find a 2×2 template T such that 4 placements of T (OR/union semantics,
    overlapping allowed) cover exactly all 8-cells in a 6×6 input.
    Output: 5×5 with T in all 4 corners, zero row/col at index 2.
    """

    @property
    def name(self) -> str:
        return "FourCopiesTemplateSolver"

    def _find_template(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        """
        Enumerate all 15 non-empty 2×2 bit-patterns for T.
        For each T, find all valid placements (every T-8 offset lands on an
        input-8 cell).  Then use itertools.combinations(valid, 4) to find 4
        placements whose union of covered 8-cells exactly equals all input 8-cells.
        Handles both overlapping and non-overlapping cases.
        """
        from itertools import combinations
        rows, cols = len(inp), len(inp[0])
        eight_cells = frozenset(
            (r, c) for r in range(rows) for c in range(cols) if inp[r][c] == 8
        )
        if not eight_cells:
            return None

        for t_bits in range(1, 16):
            # Offsets within the 2×2 block that are marked 8 in this template
            T_offsets = frozenset(
                (dr, dc)
                for dr in range(2) for dc in range(2)
                if (t_bits >> (dr * 2 + dc)) & 1
            )
            # All valid top-left positions: every T-offset must land on an 8-cell
            valid = [
                (r0, c0)
                for r0 in range(rows - 1)
                for c0 in range(cols - 1)
                if all((r0 + dr, c0 + dc) in eight_cells for dr, dc in T_offsets)
            ]
            if len(valid) < 4:
                continue
            # Find 4 placements whose union equals eight_cells exactly
            for combo in combinations(valid, 4):
                covered = frozenset(
                    (r0 + dr, c0 + dc)
                    for r0, c0 in combo
                    for dr, dc in T_offsets
                )
                if covered == eight_cells:
                    T_grid = [[0] * 2 for _ in range(2)]
                    for dr, dc in T_offsets:
                        T_grid[dr][dc] = 8
                    return T_grid
        return None

    def _predict(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        if len(inp) != 6 or any(len(r) != 6 for r in inp):
            return None
        for row in inp:
            for v in row:
                if v not in (0, 8):
                    return None

        T = self._find_template(inp)
        if T is None:
            return None

        # Build 5×5 output: T at TL(0-1,0-1), TR(0-1,3-4), BL(3-4,0-1), BR(3-4,3-4)
        out = [[0] * 5 for _ in range(5)]
        for dr in range(2):
            for dc in range(2):
                out[dr][dc]         = T[dr][dc]  # TL
                out[dr][dc + 3]     = T[dr][dc]  # TR
                out[dr + 3][dc]     = T[dr][dc]  # BL
                out[dr + 3][dc + 3] = T[dr][dc]  # BR
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or not _grid_eq(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._predict(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 4. DiamondCompressionSolver  (ca8de6ea)
# ---------------------------------------------------------------------------
# 5×5 input with exactly 9 non-zero cells at fixed diamond positions →
# 3×3 output via fixed position mapping.

_DIAMOND_MAP = [
    ((0, 0), (0, 0)),
    ((1, 1), (0, 1)),
    ((0, 4), (0, 2)),
    ((3, 1), (1, 0)),
    ((2, 2), (1, 1)),
    ((1, 3), (1, 2)),
    ((4, 0), (2, 0)),
    ((3, 3), (2, 1)),
    ((4, 4), (2, 2)),
]
_DIAMOND_SRC = {src for src, _ in _DIAMOND_MAP}


@register_solver
class DiamondCompressionSolver(BaseSolver):
    """
    Map 9 specific cells from a 5×5 diamond pattern to a 3×3 output.
    Fixed cell mapping: (0,0)→(0,0), (1,1)→(0,1), (0,4)→(0,2),
    (3,1)→(1,0), (2,2)→(1,1), (1,3)→(1,2),
    (4,0)→(2,0), (3,3)→(2,1), (4,4)→(2,2).
    """

    @property
    def name(self) -> str:
        return "DiamondCompressionSolver"

    def _predict(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        if len(inp) != 5 or any(len(r) != 5 for r in inp):
            return None
        nz = {(r, c) for r in range(5) for c in range(5) if inp[r][c] != 0}
        if nz != _DIAMOND_SRC:
            return None

        out = [[0] * 3 for _ in range(3)]
        for (sr, sc), (dr, dc) in _DIAMOND_MAP:
            out[dr][dc] = inp[sr][sc]
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or not _grid_eq(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._predict(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 5. LShapeCornerAssemblySolver  (a61ba2ce)
# ---------------------------------------------------------------------------
# 13×13 with 4 L-shaped objects (3 cells each, different colors).
# Find the missing corner of each L-shape's 2×2 bbox, map to output quadrant.
# Output: 4×4 (2×2 blocks) — each 2×2 block is the colored L-shape.
#
# missing corner → output block position:
#   BR missing → TL block (rows 0-1, cols 0-1)
#   BL missing → TR block (rows 0-1, cols 2-3)
#   TR missing → BL block (rows 2-3, cols 0-1)
#   TL missing → BR block (rows 2-3, cols 2-3)

@register_solver
class LShapeCornerAssemblySolver(BaseSolver):
    """
    4 L-shaped 3-cell objects in a 13×13 grid. Each L fits in a 2×2 bbox with
    one corner empty. The missing corner determines which 2×2 quadrant of the
    4×4 output the L occupies.
    """

    @property
    def name(self) -> str:
        return "LShapeCornerAssemblySolver"

    def _extract_objects(self, inp: List[List[int]]):
        """Return list of (color, cells) for each distinct non-zero color."""
        from collections import defaultdict
        color_cells = defaultdict(list)
        for r, row in enumerate(inp):
            for c, v in enumerate(row):
                if v != 0:
                    color_cells[v].append((r, c))
        return list(color_cells.items())

    def _l_to_2x2(self, cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int, str]]:
        """
        Given 3 cells, find 2×2 bbox, identify missing corner.
        Returns (r0, c0, missing_corner) or None.
        missing_corner: 'TL', 'TR', 'BL', 'BR'
        """
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 != 1 or c1 - c0 != 1:
            return None
        cell_set = set(cells)
        corners = {'TL': (r0, c0), 'TR': (r0, c1), 'BL': (r1, c0), 'BR': (r1, c1)}
        missing = None
        for cname, pos in corners.items():
            if pos not in cell_set:
                missing = cname
                break
        if missing is None:
            return None
        return (r0, c0, missing)

    def _predict(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        objects = self._extract_objects(inp)
        if len(objects) != 4:
            return None

        corner_to_block = {
            'BR': (0, 0),
            'BL': (0, 2),
            'TR': (2, 0),
            'TL': (2, 2),
        }

        out = [[0] * 4 for _ in range(4)]

        for color, cells in objects:
            if len(cells) != 3:
                return None
            info = self._l_to_2x2(cells)
            if info is None:
                return None
            r0, c0, missing = info
            block_r, block_c = corner_to_block[missing]

            cell_set = set(cells)
            corners_pos = {
                'TL': (r0,     c0),
                'TR': (r0,     c0 + 1),
                'BL': (r0 + 1, c0),
                'BR': (r0 + 1, c0 + 1),
            }
            for cn, (cr, cc) in corners_pos.items():
                dr = cr - r0
                dc = cc - c0
                val = color if (cr, cc) in cell_set else 0
                out[block_r + dr][block_c + dc] = val

        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or not _grid_eq(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._predict(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)


# ---------------------------------------------------------------------------
# 6. NineBlockEdgeAssemblySolver  (a8c38be5)  — best-effort
# ---------------------------------------------------------------------------
# 9 input objects (8 colored + 1 all-5/blank) → 9×9 output with separator
# rows/cols at indices 2 and 6.
#
# Confirmed transforms per region (train pair 0):
#   TL ← color 3, flip_v, rows 0-1, cols 0-1
#   TC ← color 2, flip_v, rows 0-1, all cols (0-2)  → 2×3
#   TR ← color 4, no flip, rows 1-2, cols 1-2
#   CC ← blank obj, no transform, full 3×3
#   BL ← color 6, no flip, rows 0-1, cols 0-1
#   BR ← color 7, flip_v+flip_h, rows 1-2, cols 1-2
#   LC ← color 1, CCW 90°, cols 0-1  (3×2)
#   RC ← color 1, CW 90°,  cols 1-2  (3×2)
#   BC ← UNRESOLVED — handled via train-pair gating
#
# The solver extracts objects, applies transforms, assembles output, then
# validates against every training pair. Returns None if any pair fails.

def _extract_colored_objects_9b(inp: List[List[int]]) -> dict:
    """
    Extract 3×3 bounding-box objects by color from the input.
    Returns {color: 3x3_grid}.  Color 5 is treated as background/blank.
    """
    from collections import defaultdict
    color_cells = defaultdict(list)
    for r, row in enumerate(inp):
        for c, v in enumerate(row):
            color_cells[v].append((r, c))

    objects = {}
    for color, cells in color_cells.items():
        if color == 5:
            continue
        rs = [r for r, c in cells]
        cs_ = [c for r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs_), max(cs_)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h != 3 or w != 3:
            continue
        block = [[5] * 3 for _ in range(3)]
        for r, c in cells:
            block[r - r0][c - c0] = inp[r][c]
        objects[color] = block
    return objects


def _assemble_9x9(objs: dict) -> Optional[List[List[int]]]:
    """
    Assemble a 9×9 grid from extracted objects using confirmed transforms.
    Layout (separator rows/cols at 2 and 6):
      rows 0-1  : TL(cols 0-1)  TC(cols 3-5)  TR(cols 7-8)
      rows 3-5  : LC(cols 0-1)  CC(cols 3-5)  RC(cols 7-8)
      rows 7-8  : BL(cols 0-1)  BC(cols 3-5)  BR(cols 7-8)
    Separator rows: 2, 6 (all 5)
    Separator cols: 2, 6 (all 5)
    """
    out = [[5] * 9 for _ in range(9)]

    def _place(region: List[List[int]], row_start: int, col_start: int):
        for r, row in enumerate(region):
            for c, v in enumerate(row):
                out[row_start + r][col_start + c] = v

    # TL ← color 3, flip_v, rows 0-1, cols 0-1
    if 3 not in objs:
        return None
    tl = _sub(_flip_v(objs[3]), 0, 2, 0, 2)
    _place(tl, 0, 0)

    # TC ← color 2, flip_v, rows 0-1, all cols → 2×3 placed at rows 0-1, cols 3-5
    if 2 not in objs:
        return None
    tc = _sub(_flip_v(objs[2]), 0, 2, 0, 3)
    _place(tc, 0, 3)

    # TR ← color 4, no flip, rows 1-2, cols 1-2 → 2×2 placed at rows 0-1, cols 7-8
    if 4 not in objs:
        return None
    tr = _sub(objs[4], 1, 3, 1, 3)
    _place(tr, 0, 7)

    # CC ← blank object: all-5 3×3 at rows 3-5, cols 3-5 (already 5)

    # BL ← color 6, no flip, rows 0-1, cols 0-1 → 2×2 at rows 7-8, cols 0-1
    if 6 not in objs:
        return None
    bl = _sub(objs[6], 0, 2, 0, 2)
    _place(bl, 7, 0)

    # BR ← color 7, flip_v+flip_h, rows 1-2, cols 1-2 → 2×2 at rows 7-8, cols 7-8
    if 7 not in objs:
        return None
    br = _sub(_flip_h(_flip_v(objs[7])), 1, 3, 1, 3)
    _place(br, 7, 7)

    # LC ← color 1, CCW 90°, take cols 0-1 → 3×2 at rows 3-5, cols 0-1
    if 1 not in objs:
        return None
    lc_rotated = _rot90_ccw(objs[1])
    lc = _sub(lc_rotated, 0, 3, 0, 2)
    _place(lc, 3, 0)

    # RC ← color 1, CW 90°, take cols 1-2 → 3×2 at rows 3-5, cols 7-8
    rc_rotated = _rot90_cw(objs[1])
    rc = _sub(rc_rotated, 0, 3, 1, 3)
    _place(rc, 3, 7)

    # BC ← UNRESOLVED: leave as 5 — train validation will gate this

    return out


@register_solver
class NineBlockEdgeAssemblySolver(BaseSolver):
    """
    Best-effort solver for a8c38be5: assembles a 9×9 output from 9 input
    objects using confirmed sub-block + transform rules. BC region is left
    as all-5 (unresolved). Gated strictly by train-pair self-validation.
    """

    @property
    def name(self) -> str:
        return "NineBlockEdgeAssemblySolver"

    def _predict(self, inp: List[List[int]]) -> Optional[List[List[int]]]:
        objs = _extract_colored_objects_9b(inp)
        return _assemble_9x9(objs)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        for pair in ctx.train_pairs:
            pred = self._predict(pair["input"])
            if pred is None or not _grid_eq(pred, pair["output"]):
                return None

        predictions = []
        for test_inp in ctx.test_inputs:
            pred = self._predict(test_inp)
            if pred is None:
                return None
            predictions.append(Prediction(attempt_1=pred, attempt_2=pred,
                                           solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=predictions, solved=True, tier=2)
