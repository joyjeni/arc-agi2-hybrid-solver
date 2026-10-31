"""
Session 47 Solvers — generalized pattern implementations
All helpers prefixed _47_ to avoid collisions.

Solvers (12 total):
  1.  NonBgCountCopiesSolver        — count non-bg cells N → place N copies of input tiled in output
  2.  NonBgCellToBlockSolver         — each non-bg cell → full input placed at that block position
  3.  SelfSimilarFractalSolver       — NxM → (N*N)x(M*M) fractal; sub-block = input if cell==fg else bg
  4.  BoundingBoxCropSolver          — output = tight bounding box of non-bg content
  5.  RotateFlipSolver               — detect and apply rot90/rot180/rot270/flipLR/flipUD
  6.  BijectiveColorRemapSolver      — bijective per-pair color substitution
  7.  SortRowsByNonBgCountSolver     — sort rows by ascending non-bg count
  8.  GridSeparatorFillSolver        — uniform-color separator rows/cols recolored by adjacent cell rule
  9.  TileSmallPatternSolver         — tile smallest repeating sub-pattern to fill output
 10.  GravitySolver                  — non-bg cells fall toward one edge (4 directions tested)
 11.  ColorByPositionRuleSolver      — output color at each cell = function of row/col index parity
 12.  MajorityColorFillSolver        — bg cells surrounded ≥N non-bg neighbors get filled
"""

from collections import Counter, deque
from itertools import product as iproduct
from math import isqrt, gcd
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

# ─── Session 47 Helpers ──────────────────────────────────────────────────────

def _47_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]

def _47_copy(grid):
    return [list(row) for row in grid]

def _47_eq(a, b):
    if len(a) != len(b): return False
    for r in range(len(a)):
        if len(a[r]) != len(b[r]) or a[r] != b[r]: return False
    return True

def _47_nonbg_cells(grid, bg):
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != bg]

def _47_bbox(cells):
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    return min(rs), min(cs), max(rs), max(cs)

def _47_rot90(grid):
    R, C = len(grid), len(grid[0])
    return [[grid[R-1-c][r] for r in range(R)] for c in range(C)]

def _47_flipLR(grid):
    return [list(reversed(row)) for row in grid]

def _47_flipUD(grid):
    return list(reversed([list(row) for row in grid]))

def _47_transpose(grid):
    R, C = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(R)] for c in range(C)]

def _47_bfs_ccs(grid, skip_color=None):
    R, C = len(grid), len(grid[0])
    visited = [[False]*C for _ in range(R)]
    result = []
    for r in range(R):
        for c in range(C):
            if visited[r][c]: continue
            color = grid[r][c]
            if skip_color is not None and color == skip_color:
                visited[r][c] = True; continue
            queue = deque([(r, c)])
            visited[r][c] = True
            cells = {(r, c)}
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = rr+dr, cc+dc
                    if 0<=nr<R and 0<=nc<C and not visited[nr][nc] and grid[nr][nc]==color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.add((nr, nc))
            result.append((color, frozenset(cells)))
    return result

def _47_factorizations(n):
    """Return all (rows, cols) pairs where rows*cols == n, rows<=cols."""
    result = []
    for i in range(1, isqrt(n)+1):
        if n % i == 0:
            result.append((i, n//i))
    return result

# ─── Solver 1: NonBgCountCopiesSolver ────────────────────────────────────────

@register_solver
class NonBgCountCopiesSolver(BaseSolver):
    """
    Count non-bg cells N in input → tile N copies of the input into output.
    Output shape = (H*nr) × (W*nc) where nr*nc == N.
    Factorization chosen to keep output as square as possible.
    """
    @property
    def name(self): return "NonBgCountCopiesSolver"
    @property
    def priority(self): return 45

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        # Validate rule across all training pairs
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            bg = _47_bg(inp)
            nbc = len(_47_nonbg_cells(inp, bg))
            if nbc < 1: return None
            iH, iW = len(inp), len(inp[0])
            oH, oW = len(out), len(out[0])
            # output must be multiple of input
            if oH % iH != 0 or oW % iW != 0: return None
            nr, nc = oH // iH, oW // iW
            if nr * nc != nbc: return None
            # verify tiled correctly
            expected = _47_tile_copies(inp, nr, nc)
            if not _47_eq(expected, out): return None

        # Apply to test inputs
        preds = []
        for test_inp in ctx.test_inputs:
            bg = _47_bg(test_inp)
            nbc = len(_47_nonbg_cells(test_inp, bg))
            if nbc < 1: return None
            facts = _47_factorizations(nbc)
            if not facts: return None
            # pick factorization closest to square
            nr, nc = min(facts, key=lambda x: abs(x[0]-x[1]))
            out = _47_tile_copies(test_inp, nr, nc)
            preds.append(Prediction(attempt_1=out, attempt_2=_47_tile_copies(test_inp, nc, nr),
                                    solver_name=self.name, confidence=0.85))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_tile_copies(inp, nr, nc):
    iH, iW = len(inp), len(inp[0])
    out = [[0]*(iW*nc) for _ in range(iH*nr)]
    for ri in range(nr):
        for ci in range(nc):
            for r in range(iH):
                for c in range(iW):
                    out[ri*iH+r][ci*iW+c] = inp[r][c]
    return out


# ─── Solver 2: NonBgCellToBlockSolver ────────────────────────────────────────

@register_solver
class NonBgCellToBlockSolver(BaseSolver):
    """
    For each non-bg cell at (r,c) in input → place the full input at block (r,c)
    in a (H*H)×(W*W) output (or H×W scaled by H×W).
    Background cells produce all-bg blocks.
    """
    @property
    def name(self): return "NonBgCellToBlockSolver"
    @property
    def priority(self): return 45

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            bg = _47_bg(inp)
            iH, iW = len(inp), len(inp[0])
            oH, oW = len(out), len(out[0])
            if oH != iH*iH or oW != iW*iW: return None
            expected = _47_cell_to_block(inp, bg)
            if not _47_eq(expected, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _47_bg(test_inp)
            out = _47_cell_to_block(test_inp, bg)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.88))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_cell_to_block(inp, bg):
    iH, iW = len(inp), len(inp[0])
    out = [[bg]*(iW*iW) for _ in range(iH*iH)]
    for r in range(iH):
        for c in range(iW):
            if inp[r][c] != bg:
                # place full input at block (r,c)
                for dr in range(iH):
                    for dc in range(iW):
                        out[r*iH+dr][c*iW+dc] = inp[dr][dc]
    return out


# ─── Solver 3: SelfSimilarFractalSolver ──────────────────────────────────────

@register_solver
class SelfSimilarFractalSolver(BaseSolver):
    """
    NxM input → (N*N)x(M*M) output (or (N*M)x(N*M)).
    Each (N×M) sub-block at position (r,c) in the macro grid:
      if input[r][c] == fg: sub-block = full input
      if input[r][c] == bg: sub-block = all-bg

    Generalizes to any 2-color input.
    """
    @property
    def name(self): return "SelfSimilarFractalSolver"
    @property
    def priority(self): return 46

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        # Detect: each pair must have exactly 2 colors and output = H*H x W*W
        ref_bg = None
        ref_fg = None
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            colors = set(c for row in inp for c in row)
            if len(colors) != 2: return None
            bg = _47_bg(inp)
            fg = (colors - {bg}).pop()
            iH, iW = len(inp), len(inp[0])
            oH, oW = len(out), len(out[0])
            if oH != iH*iH or oW != iW*iW: return None
            expected = _47_fractal(inp, bg, fg)
            if not _47_eq(expected, out): return None
            ref_bg, ref_fg = bg, fg

        preds = []
        for test_inp in ctx.test_inputs:
            colors = set(c for row in test_inp for c in row)
            if len(colors) != 2: return None
            bg = _47_bg(test_inp)
            fg = (colors - {bg}).pop()
            out = _47_fractal(test_inp, bg, fg)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.88))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_fractal(inp, bg, fg):
    iH, iW = len(inp), len(inp[0])
    out = [[bg]*(iW*iW) for _ in range(iH*iH)]
    for r in range(iH):
        for c in range(iW):
            if inp[r][c] == fg:
                for dr in range(iH):
                    for dc in range(iW):
                        out[r*iH+dr][c*iW+dc] = inp[dr][dc]
    return out


# ─── Solver 4: BoundingBoxCropSolver ─────────────────────────────────────────

@register_solver
class BoundingBoxCropSolver(BaseSolver):
    """Output = tight bounding box of all non-bg cells in the input."""
    @property
    def name(self): return "BoundingBoxCropSolver"
    @property
    def priority(self): return 44

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            bg = _47_bg(inp)
            nbc = _47_nonbg_cells(inp, bg)
            if not nbc: return None
            r0, c0, r1, c1 = _47_bbox(nbc)
            expected = [inp[r][c0:c1+1] for r in range(r0, r1+1)]
            if not _47_eq(expected, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _47_bg(test_inp)
            nbc = _47_nonbg_cells(test_inp, bg)
            if not nbc: return None
            r0, c0, r1, c1 = _47_bbox(nbc)
            out = [test_inp[r][c0:c1+1] for r in range(r0, r1+1)]
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.87))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─── Solver 5: RotateFlipSolver ──────────────────────────────────────────────

@register_solver
class RotateFlipSolver(BaseSolver):
    """
    Detect geometric transform (rot90/rot180/rot270/flipLR/flipUD/transpose/anti-transpose)
    consistently across all training pairs and apply to test inputs.
    """
    @property
    def name(self): return "RotateFlipSolver"
    @property
    def priority(self): return 43

    _TRANSFORMS = [
        ("rot90",    lambda g: _47_rot90(g)),
        ("rot180",   lambda g: _47_rot90(_47_rot90(g))),
        ("rot270",   lambda g: _47_rot90(_47_rot90(_47_rot90(g)))),
        ("flipLR",   lambda g: _47_flipLR(g)),
        ("flipUD",   lambda g: _47_flipUD(g)),
        ("transpose",      lambda g: _47_transpose(g)),
        ("anti_transpose", lambda g: _47_flipLR(_47_transpose(g))),
    ]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        best = None
        for name, fn in self._TRANSFORMS:
            ok = True
            for pair in pairs:
                try:
                    result = fn(pair["input"])
                except Exception:
                    ok = False; break
                if not _47_eq(result, pair["output"]):
                    ok = False; break
            if ok:
                best = fn; break

        if best is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = best(test_inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.92))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─── Solver 6: BijectiveColorRemapSolver ─────────────────────────────────────

@register_solver
class BijectiveColorRemapSolver(BaseSolver):
    """
    Learn a bijective color mapping from training pairs (same shape, different colors).
    Apply the consistent mapping to test inputs.
    """
    @property
    def name(self): return "BijectiveColorRemapSolver"
    @property
    def priority(self): return 42

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        # Learn mapping from first pair
        p0 = pairs[0]
        inp0, out0 = p0["input"], p0["output"]
        if len(inp0) != len(out0) or len(inp0[0]) != len(out0[0]): return None

        mapping = {}
        for r in range(len(inp0)):
            for c in range(len(inp0[0])):
                ci, co = inp0[r][c], out0[r][c]
                if ci in mapping:
                    if mapping[ci] != co: return None
                else:
                    mapping[ci] = co

        # Must be bijective (no two input colors map to same output)
        if len(set(mapping.values())) != len(mapping): return None
        # Must not be identity
        if all(k == v for k, v in mapping.items()): return None

        # Validate on remaining pairs
        for pair in pairs[1:]:
            inp, out = pair["input"], pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]): return None
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    ci, co = inp[r][c], out[r][c]
                    if ci not in mapping: return None
                    if mapping[ci] != co: return None

        preds = []
        for test_inp in ctx.test_inputs:
            R, C = len(test_inp), len(test_inp[0])
            out = [[mapping.get(test_inp[r][c], test_inp[r][c]) for c in range(C)] for r in range(R)]
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.90))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─── Solver 7: SortRowsByNonBgCountSolver ────────────────────────────────────

@register_solver
class SortRowsByNonBgCountSolver(BaseSolver):
    """Sort rows by ascending (or descending) non-bg cell count."""
    @property
    def name(self): return "SortRowsByNonBgCountSolver"
    @property
    def priority(self): return 50

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        direction = None
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]): return None
            bg = _47_bg(inp)
            asc = sorted(inp, key=lambda row: sum(1 for c in row if c != bg))
            desc = list(reversed(asc))
            if _47_eq(asc, out):
                d = "asc"
            elif _47_eq(desc, out):
                d = "desc"
            else:
                return None
            if direction is None:
                direction = d
            elif direction != d:
                return None

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _47_bg(test_inp)
            sorted_asc = sorted(test_inp, key=lambda row: sum(1 for c in row if c != bg))
            if direction == "asc":
                out = sorted_asc
            else:
                out = list(reversed(sorted_asc))
            preds.append(Prediction(attempt_1=out, attempt_2=list(reversed(out)),
                                    solver_name=self.name, confidence=0.85))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─── Solver 8: GridSeparatorFillSolver ───────────────────────────────────────

@register_solver
class GridSeparatorFillSolver(BaseSolver):
    """
    Grids divided by uniform-color separator rows/cols.
    In output the separators are recolored: each separator cell takes the color
    of the single non-separator color present in its adjacent section(s).
    Handles both row-only, col-only, and grid separators.
    """
    @property
    def name(self): return "GridSeparatorFillSolver"
    @property
    def priority(self): return 46

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]): return None
            result = _47_sep_fill(inp)
            if result is None: return None
            if not _47_eq(result, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            result = _47_sep_fill(test_inp)
            if result is None: return None
            preds.append(Prediction(attempt_1=result, attempt_2=result,
                                    solver_name=self.name, confidence=0.83))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_sep_fill(grid):
    R, C = len(grid), len(grid[0])
    bg = _47_bg(grid)

    # Identify separator rows (all same color, not the overall bg OR all one color)
    def is_sep_row(r):
        row = grid[r]
        return len(set(row)) == 1

    def is_sep_col(c):
        col = [grid[r][c] for r in range(R)]
        return len(set(col)) == 1

    sep_rows = [r for r in range(R) if is_sep_row(r)]
    sep_cols = [c for c in range(C) if is_sep_col(c)]
    if not sep_rows and not sep_cols: return None
    if len(sep_rows) == R or len(sep_cols) == C: return None

    # sep_color: color of separator rows/cols
    sep_color_r = grid[sep_rows[0]][0] if sep_rows else None
    sep_color_c = grid[0][sep_cols[0]] if sep_cols else None
    sep_color = sep_color_r if sep_color_r is not None else sep_color_c
    # seps must all be same color
    for r in sep_rows:
        if grid[r][0] != sep_color: return None
    for c in sep_cols:
        if grid[0][c] != sep_color: return None

    out = _47_copy(grid)

    # For each separator row, determine recolor from adjacent sections
    if sep_rows:
        non_sep_rows = [r for r in range(R) if r not in set(sep_rows)]
        for sr in sep_rows:
            # find the nearest non-sep rows above and below
            above = [r for r in non_sep_rows if r < sr]
            below = [r for r in non_sep_rows if r > sr]
            # collect colors from adjacent section
            colors = set()
            if above:
                last_above = max(above)
                for c in range(C):
                    if c not in set(sep_cols):
                        colors.add(grid[last_above][c])
            if below:
                first_below = min(below)
                for c in range(C):
                    if c not in set(sep_cols):
                        colors.add(grid[first_below][c])
            colors -= {sep_color, bg}
            if len(colors) == 1:
                fill = colors.pop()
                for c in range(C):
                    out[sr][c] = fill
            elif len(colors) == 0:
                pass  # leave as is
            else:
                # multiple colors — try dominant non-bg in adjacent row
                adj_colors = []
                if above:
                    adj_colors += [grid[max(above)][c] for c in range(C) if c not in set(sep_cols)]
                if below:
                    adj_colors += [grid[min(below)][c] for c in range(C) if c not in set(sep_cols)]
                adj_colors = [x for x in adj_colors if x not in (sep_color, bg)]
                if adj_colors:
                    fill = Counter(adj_colors).most_common(1)[0][0]
                    for c in range(C):
                        out[sr][c] = fill

    if sep_cols:
        non_sep_cols = [c for c in range(C) if c not in set(sep_cols)]
        for sc in sep_cols:
            left = [c for c in non_sep_cols if c < sc]
            right = [c for c in non_sep_cols if c > sc]
            colors = set()
            if left:
                last_left = max(left)
                for r in range(R):
                    if r not in set(sep_rows):
                        colors.add(grid[r][last_left])
            if right:
                first_right = min(right)
                for r in range(R):
                    if r not in set(sep_rows):
                        colors.add(grid[r][first_right])
            colors -= {sep_color, bg}
            if len(colors) == 1:
                fill = colors.pop()
                for r in range(R):
                    out[r][sc] = fill
            elif len(colors) == 0:
                pass
            else:
                adj_colors = []
                if left:
                    adj_colors += [grid[r][max(left)] for r in range(R) if r not in set(sep_rows)]
                if right:
                    adj_colors += [grid[r][min(right)] for r in range(R) if r not in set(sep_rows)]
                adj_colors = [x for x in adj_colors if x not in (sep_color, bg)]
                if adj_colors:
                    fill = Counter(adj_colors).most_common(1)[0][0]
                    for r in range(R):
                        out[r][sc] = fill

    if _47_eq(out, grid): return None  # no change made
    return out


# ─── Solver 8b: GridSeparatorColorByAdjacentCellSolver ───────────────────────

@register_solver
class GridSeparatorColorByAdjacentCellSolver(BaseSolver):
    """
    Variant: separator cell takes the color of the DIRECTLY adjacent non-sep cell
    (not whole section). Handles cases where each sep cell gets a unique color.
    """
    @property
    def name(self): return "GridSeparatorColorByAdjacentCellSolver"
    @property
    def priority(self): return 47

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]): return None
            result = _47_sep_fill_adj(inp)
            if result is None: return None
            if not _47_eq(result, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            result = _47_sep_fill_adj(test_inp)
            if result is None: return None
            preds.append(Prediction(attempt_1=result, attempt_2=result,
                                    solver_name=self.name, confidence=0.82))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_sep_fill_adj(grid):
    R, C = len(grid), len(grid[0])
    bg = _47_bg(grid)

    def is_sep_row(r):
        return len(set(grid[r])) == 1

    def is_sep_col(c):
        return len(set(grid[r][c] for r in range(R))) == 1

    sep_rows = set(r for r in range(R) if is_sep_row(r))
    sep_cols = set(c for c in range(C) if is_sep_col(c))
    if not sep_rows and not sep_cols: return None
    if len(sep_rows) == R or len(sep_cols) == C: return None

    sep_color = None
    for r in sep_rows:
        sep_color = grid[r][0]; break
    for c in sep_cols:
        sep_color = grid[0][c]; break

    out = _47_copy(grid)

    # Each sep cell → color from adjacent non-sep cell
    for r in range(R):
        for c in range(C):
            if r in sep_rows or c in sep_cols:
                # look at 4 neighbors, pick non-sep color
                neighbor_colors = []
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and nr not in sep_rows and nc not in sep_cols:
                        color = grid[nr][nc]
                        if color not in (sep_color, bg):
                            neighbor_colors.append(color)
                if len(set(neighbor_colors)) == 1:
                    out[r][c] = neighbor_colors[0]
                elif neighbor_colors:
                    out[r][c] = Counter(neighbor_colors).most_common(1)[0][0]

    if _47_eq(out, grid): return None
    return out


# ─── Solver 9: TileSmallPatternSolver ────────────────────────────────────────

@register_solver
class TileSmallPatternSolver(BaseSolver):
    """
    Output = input tiled to fill a larger output.
    Detects tile size from pattern, validates tiling is exact.
    Also handles: output = input with one axis tiled N times.
    """
    @property
    def name(self): return "TileSmallPatternSolver"
    @property
    def priority(self): return 44

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        tile_r, tile_c = None, None
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            iH, iW = len(inp), len(inp[0])
            oH, oW = len(out), len(out[0])
            if oH % iH != 0 or oW % iW != 0: return None
            tr, tc = oH // iH, oW // iW
            if tr == 1 and tc == 1: return None  # no tiling
            # Check if output is inp tiled tr×tc
            ok = True
            for ri in range(tr):
                for ci in range(tc):
                    for r in range(iH):
                        for c in range(iW):
                            if out[ri*iH+r][ci*iW+c] != inp[r][c]:
                                ok = False; break
                        if not ok: break
                    if not ok: break
                if not ok: break
            if not ok: return None
            if tile_r is None:
                tile_r, tile_c = tr, tc
            elif (tile_r, tile_c) != (tr, tc):
                return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _47_tile_copies(test_inp, tile_r, tile_c)
            alt = _47_tile_copies(test_inp, tile_c, tile_r) if tile_r != tile_c else out
            preds.append(Prediction(attempt_1=out, attempt_2=alt,
                                    solver_name=self.name, confidence=0.86))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─── Solver 10: GravitySolver ─────────────────────────────────────────────────

@register_solver
class GravitySolver(BaseSolver):
    """
    Non-bg cells fall toward one edge (down/up/left/right).
    Detects direction from training pairs.
    """
    @property
    def name(self): return "GravitySolver"
    @property
    def priority(self): return 48

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        direction = None
        for d in ("down", "up", "left", "right"):
            ok = True
            for pair in pairs:
                expected = _47_gravity(pair["input"], d)
                if not _47_eq(expected, pair["output"]):
                    ok = False; break
            if ok:
                direction = d; break

        if direction is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _47_gravity(test_inp, direction)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.87))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_gravity(grid, direction):
    R, C = len(grid), len(grid[0])
    bg = _47_bg(grid)
    out = [[bg]*C for _ in range(R)]

    if direction == "down":
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            non_bg = [x for x in col if x != bg]
            filled = [bg] * (R - len(non_bg)) + non_bg
            for r in range(R): out[r][c] = filled[r]
    elif direction == "up":
        for c in range(C):
            col = [grid[r][c] for r in range(R)]
            non_bg = [x for x in col if x != bg]
            filled = non_bg + [bg] * (R - len(non_bg))
            for r in range(R): out[r][c] = filled[r]
    elif direction == "right":
        for r in range(R):
            row = grid[r]
            non_bg = [x for x in row if x != bg]
            filled = [bg] * (C - len(non_bg)) + non_bg
            out[r] = filled
    elif direction == "left":
        for r in range(R):
            row = grid[r]
            non_bg = [x for x in row if x != bg]
            filled = non_bg + [bg] * (C - len(non_bg))
            out[r] = filled
    return out


# ─── Solver 11: ColorByPositionRuleSolver ────────────────────────────────────

@register_solver
class ColorByPositionRuleSolver(BaseSolver):
    """
    Each output cell color = f(r, c, input_colors...) based on position parity.
    Handles checkerboard, stripe, quadrant patterns.
    """
    @property
    def name(self): return "ColorByPositionRuleSolver"
    @property
    def priority(self): return 52

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        # Must be same shape input/output
        for pair in pairs:
            if len(pair["input"]) != len(pair["output"]): return None
            if len(pair["input"][0]) != len(pair["output"][0]): return None

        # Try checkerboard: out[r][c] = color_a if (r+c)%2==0 else color_b
        # Infer colors from first pair
        p0_in, p0_out = pairs[0]["input"], pairs[0]["output"]
        R, C = len(p0_out), len(p0_out[0])

        even_colors = set(p0_out[r][c] for r in range(R) for c in range(C) if (r+c)%2==0)
        odd_colors = set(p0_out[r][c] for r in range(R) for c in range(C) if (r+c)%2==1)

        if len(even_colors) == 1 and len(odd_colors) == 1:
            ca, cb = even_colors.pop(), odd_colors.pop()
            if ca == cb: return None
            # validate all pairs
            ok = True
            for pair in pairs:
                out = pair["output"]
                for r in range(len(out)):
                    for c in range(len(out[0])):
                        expected = ca if (r+c)%2==0 else cb
                        if out[r][c] != expected: ok=False; break
                    if not ok: break
                if not ok: break
            if ok:
                preds = []
                for test_inp in ctx.test_inputs:
                    R2, C2 = len(test_inp), len(test_inp[0])
                    out = [[ca if (r+c)%2==0 else cb for c in range(C2)] for r in range(R2)]
                    preds.append(Prediction(attempt_1=out, attempt_2=out,
                                            solver_name=self.name, confidence=0.84))
                return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

        return None


# ─── Solver 12: MajorityColorFillSolver ──────────────────────────────────────

@register_solver
class MajorityColorFillSolver(BaseSolver):
    """
    Bg cells with ≥ threshold non-bg neighbors get filled with the dominant neighbor color.
    Threshold determined from training pairs.
    """
    @property
    def name(self): return "MajorityColorFillSolver"
    @property
    def priority(self): return 53

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        threshold = None
        for t in (3, 2, 4, 1):
            ok = True
            for pair in pairs:
                result = _47_majority_fill(pair["input"], t)
                if not _47_eq(result, pair["output"]):
                    ok = False; break
            if ok:
                threshold = t; break

        if threshold is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _47_majority_fill(test_inp, threshold)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.80))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_majority_fill(grid, threshold):
    R, C = len(grid), len(grid[0])
    bg = _47_bg(grid)
    out = _47_copy(grid)
    changed = True
    while changed:
        changed = False
        for r in range(R):
            for c in range(C):
                if out[r][c] != bg: continue
                neighbors = []
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r+dr, c+dc
                    if 0<=nr<R and 0<=nc<C and out[nr][nc] != bg:
                        neighbors.append(out[nr][nc])
                if len(neighbors) >= threshold:
                    fill = Counter(neighbors).most_common(1)[0][0]
                    out[r][c] = fill
                    changed = True
    if _47_eq(out, grid): return grid
    return out


# ─── Solver 13: ScaleDownMajoritySolver ──────────────────────────────────────

@register_solver
class ScaleDownMajoritySolver(BaseSolver):
    """
    Input is larger than output by integer factor K.
    Each K×K block in input → single cell in output = majority color of block.
    """
    @property
    def name(self): return "ScaleDownMajoritySolver"
    @property
    def priority(self): return 44

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        scale = None
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            iH, iW = len(inp), len(inp[0])
            oH, oW = len(out), len(out[0])
            if iH % oH != 0 or iW % oW != 0: return None
            sr, sc = iH // oH, iW // oW
            if sr != sc: return None
            k = sr
            if k < 2: return None
            if scale is None: scale = k
            elif scale != k: return None
            expected = _47_scale_down(inp, k)
            if not _47_eq(expected, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            if scale is None: return None
            out = _47_scale_down(test_inp, scale)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.87))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_scale_down(grid, k):
    R, C = len(grid), len(grid[0])
    oR, oC = R // k, C // k
    out = []
    for r in range(oR):
        row = []
        for c in range(oC):
            block = [grid[r*k+dr][c*k+dc] for dr in range(k) for dc in range(k)]
            row.append(Counter(block).most_common(1)[0][0])
        out.append(row)
    return out


# ─── Solver 14: ScaleUpUniformSolver ─────────────────────────────────────────

@register_solver
class ScaleUpUniformSolver(BaseSolver):
    """
    Input scaled up by integer factor K: each cell → K×K block of same color.
    Detects K from output/input size ratio.
    """
    @property
    def name(self): return "ScaleUpUniformSolver"
    @property
    def priority(self): return 43

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        scale = None
        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            iH, iW = len(inp), len(inp[0])
            oH, oW = len(out), len(out[0])
            if oH % iH != 0 or oW % iW != 0: return None
            sr, sc = oH // iH, oW // iW
            if sr != sc: return None
            k = sr
            if k < 2: return None
            if scale is None: scale = k
            elif scale != k: return None
            expected = _47_scale_up(inp, k)
            if not _47_eq(expected, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            if scale is None: return None
            out = _47_scale_up(test_inp, scale)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.90))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_scale_up(grid, k):
    R, C = len(grid), len(grid[0])
    out = []
    for r in range(R):
        for _ in range(k):
            row = []
            for c in range(C):
                for _ in range(k):
                    row.append(grid[r][c])
            out.append(row)
    return out


# ─── Solver 15: UniqueColorPerRowColSolver ────────────────────────────────────

@register_solver
class UniqueColorPerRowColSolver(BaseSolver):
    """
    Output marks the cell at the intersection of (row with unique color) and
    (col with unique color). Or: output = single cell at unique-color intersection.
    """
    @property
    def name(self): return "UniqueColorPerRowColSolver"
    @property
    def priority(self): return 49

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            inp, out = pair["input"], pair["output"]
            result = _47_unique_intersect(inp)
            if result is None: return None
            if not _47_eq(result, out): return None

        preds = []
        for test_inp in ctx.test_inputs:
            result = _47_unique_intersect(test_inp)
            if result is None: return None
            preds.append(Prediction(attempt_1=result, attempt_2=result,
                                    solver_name=self.name, confidence=0.85))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_unique_intersect(grid):
    R, C = len(grid), len(grid[0])
    bg = _47_bg(grid)
    # find rows/cols where there's exactly 1 non-bg cell
    unique_rows = []
    for r in range(R):
        nbc = [(r, c) for c in range(C) if grid[r][c] != bg]
        if len(nbc) == 1:
            unique_rows.append(r)
    unique_cols = []
    for c in range(C):
        nbc = [(r, c) for r in range(R) if grid[r][c] != bg]
        if len(nbc) == 1:
            unique_cols.append(c)
    if len(unique_rows) == 1 and len(unique_cols) == 1:
        r, c = unique_rows[0], unique_cols[0]
        out = [[bg]*C for _ in range(R)]
        out[r][c] = grid[r][c]
        return out
    return None


# ─── Solver 16: PaintConnectedComponentsSolver ───────────────────────────────

@register_solver
class PaintConnectedComponentsSolver(BaseSolver):
    """
    Each connected component of non-bg cells gets painted a unique color
    determined by the component's rank (size, position, etc).
    """
    @property
    def name(self): return "PaintConnectedComponentsSolver"
    @property
    def priority(self): return 51

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        rank_method = None
        for method in ("size_desc", "size_asc", "top_left"):
            ok = True
            for pair in pairs:
                result = _47_paint_ccs(pair["input"], method)
                if result is None or not _47_eq(result, pair["output"]):
                    ok = False; break
            if ok:
                rank_method = method; break

        if rank_method is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _47_paint_ccs(test_inp, rank_method)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.82))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _47_paint_ccs(grid, method):
    R, C = len(grid), len(grid[0])
    bg = _47_bg(grid)
    ccs = _47_bfs_ccs(grid, skip_color=bg)
    if not ccs: return None
    n = len(ccs)

    if method == "size_desc":
        sorted_ccs = sorted(ccs, key=lambda x: -len(x[1]))
    elif method == "size_asc":
        sorted_ccs = sorted(ccs, key=lambda x: len(x[1]))
    elif method == "top_left":
        sorted_ccs = sorted(ccs, key=lambda x: (min(r for r,c in x[1]), min(c for r,c in x[1])))
    else:
        return None

    # Assign colors 1..n
    out = _47_copy(grid)
    for i, (orig_color, cells) in enumerate(sorted_ccs):
        new_color = i + 1
        for r, c in cells:
            out[r][c] = new_color
    return out
