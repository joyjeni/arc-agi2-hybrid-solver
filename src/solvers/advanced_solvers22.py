"""
advanced_solvers22.py — 4 pattern solvers
  BlockComplementSolver        (1c0d0a4b)
  FiveBoxDirectionSolver       (18286ef8)
  DiagonalTypeClassifierSolver (1d61978c)
  DiagonalWedgeFillSolver      (1f0c79e5)
"""
from __future__ import annotations

from collections import Counter

from src.core.interfaces import BaseSolver, Grid, Prediction, SolverResult, TaskContext

from .registry import register_solver

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _copy_grid(g: Grid) -> Grid:
    return [list(row) for row in g]


def _bg(grid: Grid) -> int:
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _verify_exact(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if list(ra) != list(rb):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. BlockComplementSolver  (1c0d0a4b)
# ---------------------------------------------------------------------------
# Grid divided by all-zero separator rows/cols into rectangular blocks.
# Within each block: foreground color (always 8) → 0; background (0) → 2.
# Separator rows/cols stay 0.
# ---------------------------------------------------------------------------

def _block_complement_transform(grid: Grid) -> Grid | None:
    R, C = len(grid), len(grid[0])

    sep_rows = set(r for r in range(R) if all(grid[r][c] == 0 for c in range(C)))
    sep_cols = set(c for c in range(C) if all(grid[r][c] == 0 for r in range(R)))

    fg_colors: set = set()
    for r in range(R):
        if r in sep_rows:
            continue
        for c in range(C):
            if c in sep_cols:
                continue
            if grid[r][c] != 0:
                fg_colors.add(grid[r][c])

    if len(fg_colors) != 1:
        return None
    fg = next(iter(fg_colors))

    out = _copy_grid(grid)
    for r in range(R):
        if r in sep_rows:
            continue
        for c in range(C):
            if c in sep_cols:
                continue
            out[r][c] = 0 if grid[r][c] == fg else 2
    return out


@register_solver
class BlockComplementSolver(BaseSolver):
    """Invert block contents: fg→0, bg→2; separator rows/cols stay 0."""

    @property
    def name(self) -> str:
        return "BlockComplementSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _block_complement_transform(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _block_complement_transform(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name="BlockComplementSolver"
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2
        )


# ---------------------------------------------------------------------------
# 2. FiveBoxDirectionSolver  (18286ef8)
# ---------------------------------------------------------------------------
# One 3x3 box: border=5, center=9.  One cell = 6 elsewhere.
# Direction = sign(section_row_6 - section_row_box,
#                  section_col_6 - section_col_box)
# Transform: center(9→5), edge-in-direction(5→9), 6-cell(6→9)
# ---------------------------------------------------------------------------

def _five_box_direction_transform(grid: Grid) -> Grid | None:
    R, C = len(grid), len(grid[0])

    sep_rows = sorted(r for r in range(R) if all(grid[r][c] == 0 for c in range(C)))
    sep_cols = sorted(c for c in range(C) if all(grid[r][c] == 0 for r in range(R)))

    cells_5 = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 5]
    cells_9 = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 9]
    cells_6 = [(r, c) for r in range(R) for c in range(C) if grid[r][c] == 6]

    if len(cells_9) != 1 or len(cells_6) != 1:
        return None

    r9, c9 = cells_9[0]

    # verify 3x3 box: 8 border cells are all 5
    box_5 = set()
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = r9 + dr, c9 + dc
            if nr < 0 or nr >= R or nc < 0 or nc >= C:
                return None
            if grid[nr][nc] != 5:
                return None
            box_5.add((nr, nc))

    # no stray 5s outside the box
    if any(cell not in box_5 for cell in cells_5):
        return None

    r6, c6 = cells_6[0]

    def sec_row(r):
        return sum(1 for s in sep_rows if r > s)

    def sec_col(c):
        return sum(1 for s in sep_cols if c > s)

    dr_dir = 0
    dc_dir = 0
    diff_r = sec_row(r6) - sec_row(r9)
    diff_c = sec_col(c6) - sec_col(c9)
    if diff_r != 0:
        dr_dir = 1 if diff_r > 0 else -1
    if diff_c != 0:
        dc_dir = 1 if diff_c > 0 else -1

    if dr_dir == 0 and dc_dir == 0:
        return None

    edge_r = r9 + dr_dir
    edge_c = c9 + dc_dir
    if edge_r < 0 or edge_r >= R or edge_c < 0 or edge_c >= C:
        return None
    if grid[edge_r][edge_c] != 5:
        return None

    out = _copy_grid(grid)
    out[r9][c9] = 5        # center: 9 → 5
    out[edge_r][edge_c] = 9  # edge in direction: 5 → 9
    out[r6][c6] = 9         # 6-cell → 9
    return out


@register_solver
class FiveBoxDirectionSolver(BaseSolver):
    """3x3 box of 5s with center-9; 6-cell encodes direction; swap accordingly."""

    @property
    def name(self) -> str:
        return "FiveBoxDirectionSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _five_box_direction_transform(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _five_box_direction_transform(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name="FiveBoxDirectionSolver"
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2
        )


# ---------------------------------------------------------------------------
# 3. DiagonalTypeClassifierSolver  (1d61978c)
# ---------------------------------------------------------------------------
# Grid of 7s with 5s on diagonal lines.
# Each 5: check main-diag neighbors (±1,±1) vs anti-diag neighbors (±1,∓1).
# pure-main → "main" group; pure-anti → "anti" group.
# Larger group → 8; smaller → 2.
# ---------------------------------------------------------------------------

def _diagonal_type_transform(grid: Grid) -> Grid | None:
    R, C = len(grid), len(grid[0])

    def is5(r, c):
        return 0 <= r < R and 0 <= c < C and grid[r][c] == 5

    main_cells = []
    anti_cells = []
    ambig_cells = []

    for r in range(R):
        for c in range(C):
            if grid[r][c] != 5:
                continue
            has_main = is5(r - 1, c - 1) or is5(r + 1, c + 1)
            has_anti = is5(r - 1, c + 1) or is5(r + 1, c - 1)
            if has_main and not has_anti:
                main_cells.append((r, c))
            elif has_anti and not has_main:
                anti_cells.append((r, c))
            else:
                ambig_cells.append((r, c))

    if not main_cells or not anti_cells:
        return None

    color_main = 8 if len(main_cells) >= len(anti_cells) else 2
    color_anti = 2 if color_main == 8 else 8

    main_set = set(main_cells)
    anti_set = set(anti_cells)

    out = _copy_grid(grid)
    for r, c in main_cells:
        out[r][c] = color_main
    for r, c in anti_cells:
        out[r][c] = color_anti
    for r, c in ambig_cells:
        main_nb = any(
            is5(r + dr, c + dc) and (r + dr, c + dc) in main_set
            for dr, dc in [(-1, -1), (1, 1)]
        )
        anti_nb = any(
            is5(r + dr, c + dc) and (r + dr, c + dc) in anti_set
            for dr, dc in [(-1, 1), (1, -1)]
        )
        if main_nb and not anti_nb:
            out[r][c] = color_main
        elif anti_nb and not main_nb:
            out[r][c] = color_anti

    return out


@register_solver
class DiagonalTypeClassifierSolver(BaseSolver):
    """Classify 5-cells as main- or anti-diagonal; color larger group 8, smaller 2."""

    @property
    def name(self) -> str:
        return "DiagonalTypeClassifierSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _diagonal_type_transform(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _diagonal_type_transform(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name="DiagonalTypeClassifierSolver"
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2
        )


# ---------------------------------------------------------------------------
# 4. DiagonalWedgeFillSolver  (1f0c79e5)
# ---------------------------------------------------------------------------
# 2×2 non-zero blocks: each has one color X and one or more 2-cells.
# Per 2-cell position in block:
#   TL/BR → main-diagonal band (r-c ∈ {K-1,K,K+1})
#   TR/BL → anti-diagonal band (r+c ∈ {K-1,K,K+1})
#   TL/TR → fill UP (rows ≤ r_block_top)
#   BL/BR → fill DOWN (rows ≥ r_block_bottom)
# Fill all qualifying cells with X; 2-cells become X.
# ---------------------------------------------------------------------------

def _find_2x2_blocks(grid: Grid):
    """Return list of (r0,c0,r1,c1) for 2×2 all-non-zero blocks."""
    R, C = len(grid), len(grid[0])
    visited: set = set()
    blocks = []
    for r in range(R - 1):
        for c in range(C - 1):
            if (r, c) in visited:
                continue
            vals = [grid[r][c], grid[r][c + 1], grid[r + 1][c], grid[r + 1][c + 1]]
            if all(v != 0 for v in vals):
                visited.add((r, c))
                blocks.append((r, c, r + 1, c + 1))
    return blocks


def _diagonal_wedge_fill_transform(grid: Grid) -> Grid | None:
    R, C = len(grid), len(grid[0])
    blocks = _find_2x2_blocks(grid)
    if not blocks:
        return None

    out = _copy_grid(grid)

    for r0, c0, r1, c1 in blocks:
        block_cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        colors = {grid[r][c] for r, c in block_cells}
        if 2 not in colors:
            return None
        x_colors = colors - {2}
        if len(x_colors) != 1:
            return None
        x_color = next(iter(x_colors))

        for r2, c2 in block_cells:
            if grid[r2][c2] != 2:
                continue

            is_top = (r2 == r0)
            is_left = (c2 == c0)

            if is_top and is_left:
                pos = "TL"
            elif is_top and not is_left:
                pos = "TR"
            elif not is_top and is_left:
                pos = "BL"
            else:
                pos = "BR"

            if pos in ("TL", "BR"):
                diag_type = "main"
                K = r2 - c2
            else:
                diag_type = "anti"
                K = r2 + c2

            if pos in ("TL", "TR"):
                direction = "UP"
                row_limit = r0
            else:
                direction = "DOWN"
                row_limit = r1

            for r in range(R):
                if direction == "UP" and r > row_limit:
                    continue
                if direction == "DOWN" and r < row_limit:
                    continue
                for c in range(C):
                    if diag_type == "main":
                        if (r - c) not in (K - 1, K, K + 1):
                            continue
                    else:
                        if (r + c) not in (K - 1, K, K + 1):
                            continue
                    out[r][c] = x_color

    return out


@register_solver
class DiagonalWedgeFillSolver(BaseSolver):
    """Fill diagonal wedge bands from 2×2 seed blocks containing color X and 2-cells."""

    @property
    def name(self) -> str:
        return "DiagonalWedgeFillSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            inp = pair["input"] if isinstance(pair, dict) else pair[0]
            exp = pair["output"] if isinstance(pair, dict) else pair[1]
            pred = _diagonal_wedge_fill_transform(inp)
            if pred is None or not _verify_exact(pred, exp):
                return None

        predictions = []
        for test_input in ctx.test_inputs:
            result = _diagonal_wedge_fill_transform(test_input)
            if result is None:
                return None
            predictions.append(Prediction(
                attempt_1=result,
                attempt_2=result,
                solver_name="DiagonalWedgeFillSolver"
            ))

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2
        )
