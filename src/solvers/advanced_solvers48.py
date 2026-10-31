"""
Session 48 Solvers — fractal fix + error patches + new solvers
All helpers prefixed _48_ to avoid collisions.

Solvers:
  1.  FractalBlockSolver             — generalized fractal (nonzero/most_common/least_common rules)
  2.  ColorCountToOutputSolver       — output = single color determined by count relationship
  3.  CopyObjectToTemplateSolver     — copy the unique object onto each marker in output template
  4.  RemoveNoisePixelsSolver        — remove isolated single pixels (noise) from grid
  5.  RepeatPatternToFitSolver       — detect period, tile to fill output of given size
  6.  ColorByNeighborCountSolver     — recolor cells by number of distinct neighbors
  7.  DiagonalMirrorSolver           — detect and apply transpose/anti-transpose symmetry
  8.  FillRectangleOutlineSolver     — fill the outline of identified rectangle
  9.  ExtractPatternFromNoiseSolver  — majority vote in blocks to extract clean pattern
 10.  ObjectColorByCountSolver       — paint each object based on its cell count
"""

from collections import Counter, deque
from itertools import product as iproduct
from math import isqrt, gcd
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

# ─── Session 48 Helpers ──────────────────────────────────────────────────────

def _48_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]

def _48_copy(grid):
    return [list(row) for row in grid]

def _48_eq(a, b):
    if len(a) != len(b): return False
    for r in range(len(a)):
        if len(a[r]) != len(b[r]) or a[r] != b[r]: return False
    return True

def _48_bfs_ccs(grid, skip_color=None):
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

def _48_nonbg(grid, bg):
    return [(r,c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != bg]


# ─── Solver 1: FractalBlockSolver ────────────────────────────────────────────

@register_solver
class FractalBlockSolver(BaseSolver):
    """
    Generalized fractal: output = input scaled by k, where each sub-block at (r,c)
    is either a copy of the full input (if cell is 'active') or uniform empty_color.
    Tries four rules: nonzero, most_common, least_common, not_most_common.
    empty_color is detected from output blocks, not assumed to be most-common-input-color.
    """
    @property
    def name(self): return "FractalBlockSolver"
    @property
    def priority(self): return 95  # high priority

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        result = _48_fractal_solve(pairs)
        if result is None: return None
        rule_fn, k_list, ec_list = result

        preds = []
        for i, test_inp in enumerate(ctx.test_inputs):
            # Use k and empty_color from first training pair analysis
            iH, iW = len(test_inp), len(test_inp[0])
            # Re-derive k from training pair 0 if possible
            k = k_list[0]
            ec = ec_list[0]
            pred = _48_predict_fractal(test_inp, k, ec, rule_fn)
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.93))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_predict_fractal(inp, k, empty_color, rule_fn):
    iH, iW = len(inp), len(inp[0])
    out = [[empty_color] * (iW * k) for _ in range(iH * k)]
    for r in range(iH):
        for c in range(iW):
            if rule_fn(inp, r, c):
                for dr in range(iH):
                    for dc in range(iW):
                        out[r * k + dr][c * k + dc] = inp[dr][dc]
    return out


def _48_fractal_analyze_pair(pair):
    """Analyze one training pair to extract (k, empty_color, active_colors, inactive_colors)."""
    inp, out = pair['input'], pair['output']
    iH, iW = len(inp), len(inp[0])
    oH, oW = len(out), len(out[0])
    if oH % iH != 0 or oW % iW != 0: return None
    kr, kc = oH // iH, oW // iW
    if kr != kc or kr < 2: return None
    k = kr

    active_colors = set()
    inactive_colors = set()
    empty_color = None
    valid = True

    for r in range(iH):
        for c in range(iW):
            sub = [out[r * k + dr][c * k: c * k + k] for dr in range(k)]
            if sub == [list(inp[dr][cc] for cc in range(iW)) for dr in range(iH)]:
                # sub-block equals input — active
                active_colors.add(inp[r][c])
            elif len(set(x for row in sub for x in row)) == 1:
                # uniform block — inactive
                ec = sub[0][0]
                if empty_color is None:
                    empty_color = ec
                elif empty_color != ec:
                    valid = False; break
                inactive_colors.add(inp[r][c])
            else:
                valid = False; break
        if not valid: break

    if not valid: return None
    if not active_colors and not inactive_colors: return None
    if empty_color is None: empty_color = 0

    return {'k': k, 'empty': empty_color, 'active': active_colors, 'inactive': inactive_colors}


def _48_fractal_solve(pairs):
    """Try all four rule functions and return the one that validates on all pairs."""
    analyses = []
    for pair in pairs:
        a = _48_fractal_analyze_pair(pair)
        if a is None: return None
        analyses.append(a)

    # Ensure k is consistent
    ks = [a['k'] for a in analyses]
    if len(set(ks)) != 1: return None

    # Rule functions
    def rule_nonzero(inp, r, c): return inp[r][c] != 0
    def rule_most_common(inp, r, c):
        flat = [x for row in inp for x in row]
        mc = Counter(flat).most_common(1)[0][0]
        return inp[r][c] == mc
    def rule_least_common(inp, r, c):
        flat = [x for row in inp for x in row]
        lc = Counter(flat).most_common()[-1][0]
        return inp[r][c] == lc
    def rule_not_most_common(inp, r, c):
        flat = [x for row in inp for x in row]
        mc = Counter(flat).most_common(1)[0][0]
        return inp[r][c] != mc

    def validate_rule(rfn):
        for i, pair in enumerate(pairs):
            inp, out = pair['input'], pair['output']
            k = analyses[i]['k']
            ec = analyses[i]['empty']
            pred = _48_predict_fractal(inp, k, ec, rfn)
            if pred != out: return False
        return True

    for rfn in [rule_nonzero, rule_most_common, rule_least_common, rule_not_most_common]:
        if validate_rule(rfn):
            k_list = [a['k'] for a in analyses]
            ec_list = [a['empty'] for a in analyses]
            return rfn, k_list, ec_list

    return None


# ─── Solver 2: ColorCountToOutputSolver ──────────────────────────────────────

@register_solver
class ColorCountToOutputSolver(BaseSolver):
    """
    Output is a single cell or row/col whose value is determined by the count
    of a specific color in the input.
    Handles: output = single cell with color=count, or output = grid of count cells.
    """
    @property
    def name(self): return "ColorCountToOutputSolver"
    @property
    def priority(self): return 55

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None
        if len(pairs) < 2: return None

        rule = _48_detect_count_rule(pairs)
        if rule is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _48_apply_count_rule(test_inp, rule)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.80))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_detect_count_rule(pairs):
    """Try: output = grid filled with count-of-non-bg-color, using count as value."""
    # Attempt: output has specific shape/color based on count of some color in input
    analyses = []
    for pair in pairs:
        inp, out = pair['input'], pair['output']
        bg = _48_bg(inp)
        oH, oW = len(out), len(out[0])
        out_colors = set(c for row in out for c in row)

        # Case: single cell output
        if oH == 1 and oW == 1:
            val = out[0][0]
            # Does count of some color in input == val?
            color_counts = Counter(c for row in inp for c in row)
            match = None
            for col, cnt in color_counts.items():
                if cnt == val:
                    match = ('single', col, val)
                    break
            if match is None: return None
            analyses.append(match)
        else:
            return None

    if not analyses: return None
    # All pairs same type and target color consistent
    types = [a[0] for a in analyses]
    if len(set(types)) != 1: return None

    # For 'single': find consistent target color
    # Try: target_color = most non-bg color
    for pair_a, pair_b in zip(analyses, analyses[1:]):
        if pair_a[1] != pair_b[1]: return None

    return ('single_count', analyses[0][1])  # (rule_type, target_color)


def _48_apply_count_rule(inp, rule):
    if rule[0] == 'single_count':
        target_color = rule[1]
        cnt = sum(1 for row in inp for c in row if c == target_color)
        return [[cnt]]
    return None


# ─── Solver 3: RemoveNoisePixelsSolver ───────────────────────────────────────

@register_solver
class RemoveNoisePixelsSolver(BaseSolver):
    """
    Remove isolated single-pixel non-bg cells (no same-color neighbor).
    Output = input with all noise pixels set to bg.
    """
    @property
    def name(self): return "RemoveNoisePixelsSolver"
    @property
    def priority(self): return 56

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            expected = _48_remove_noise(pair['input'])
            if expected is None: return None
            if not _48_eq(expected, pair['output']): return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _48_remove_noise(test_inp)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.82))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_remove_noise(grid):
    R, C = len(grid), len(grid[0])
    bg = _48_bg(grid)
    out = _48_copy(grid)
    changed = False
    for r in range(R):
        for c in range(C):
            if grid[r][c] == bg: continue
            color = grid[r][c]
            has_neighbor = False
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == color:
                    has_neighbor = True; break
            if not has_neighbor:
                out[r][c] = bg
                changed = True
    if not changed: return None
    return out


# ─── Solver 4: RepeatPatternToFitSolver ──────────────────────────────────────

@register_solver
class RepeatPatternToFitSolver(BaseSolver):
    """
    Detect the minimal repeating tile in the input (GCD period in rows and cols)
    and tile it to fill the output dimensions.
    """
    @property
    def name(self): return "RepeatPatternToFitSolver"
    @property
    def priority(self): return 57

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        return None  # DISABLED: 0 correct, only producing wrong predictions
        pairs = ctx.train_pairs
        if not pairs: return None

        for pair in pairs:
            out = _48_tile_to_fit(pair['input'], len(pair['output']), len(pair['output'][0]))
            if out is None: return None
            if not _48_eq(out, pair['output']): return None

        preds = []
        for test_inp in ctx.test_inputs:
            # Output dims = input dims for test (unknown), default same size
            oH, oW = len(test_inp), len(test_inp[0])
            # Check if we can get hint from training output dims ratio
            if pairs:
                iH0, iW0 = len(pairs[0]['input']), len(pairs[0]['input'][0])
                oH0, oW0 = len(pairs[0]['output']), len(pairs[0]['output'][0])
                tH = len(test_inp) * oH0 // iH0 if iH0 > 0 else oH
                tW = len(test_inp[0]) * oW0 // iW0 if iW0 > 0 else oW
                out = _48_tile_to_fit(test_inp, tH, tW)
            else:
                out = _48_tile_to_fit(test_inp, oH, oW)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.80))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_find_period(seq):
    """Find minimum period of a sequence (list of lists or list of values)."""
    n = len(seq)
    for p in range(1, n+1):
        if n % p == 0:
            ok = True
            for i in range(n):
                if seq[i] != seq[i % p]:
                    ok = False; break
            if ok: return p
    return n


def _48_tile_to_fit(grid, tH, tW):
    R, C = len(grid), len(grid[0])
    # Find row period
    rp = _48_find_period(grid)
    # Find col period
    cols = [[grid[r][c] for r in range(R)] for c in range(C)]
    cp = _48_find_period(cols)

    tile_h = rp
    tile_w = cp
    if tile_h == R and tile_w == C: return None  # no period found → skip

    out = []
    for r in range(tH):
        row = [grid[r % tile_h][c % tile_w] for c in range(tW)]
        out.append(row)
    return out


# ─── Solver 5: DiagonalMirrorSolver ──────────────────────────────────────────

@register_solver
class DiagonalMirrorSolver(BaseSolver):
    """
    Apply transpose (reflect across main diagonal) or anti-transpose.
    """
    @property
    def name(self): return "DiagonalMirrorSolver"
    @property
    def priority(self): return 58

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        transform = None
        for tname, tfn in [("transpose", _48_transpose), ("anti_transpose", _48_anti_transpose)]:
            ok = True
            for pair in pairs:
                if not _48_eq(tfn(pair['input']), pair['output']):
                    ok = False; break
            if ok:
                transform = tfn; break

        if transform is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = transform(test_inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.88))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_transpose(grid):
    R, C = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(R)] for c in range(C)]


def _48_anti_transpose(grid):
    R, C = len(grid), len(grid[0])
    # Output is C x R: out[c][r] = grid[R-1-r][C-1-c]
    return [[grid[R-1-r][C-1-c] for r in range(R)] for c in range(C)]


# ─── Solver 6: ObjectColorByCountSolver ──────────────────────────────────────

@register_solver
class ObjectColorByCountSolver(BaseSolver):
    """
    Recolor each connected component (object) with a color determined by
    its cell count (e.g., count 3 → color 3).
    """
    @property
    def name(self): return "ObjectColorByCountSolver"
    @property
    def priority(self): return 59

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        rule = None
        for method in ("count_as_color", "rank_by_size_asc", "rank_by_size_desc"):
            ok = True
            for pair in pairs:
                out = _48_recolor_by_count(pair['input'], method)
                if out is None or not _48_eq(out, pair['output']):
                    ok = False; break
            if ok:
                rule = method; break

        if rule is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _48_recolor_by_count(test_inp, rule)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.82))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_recolor_by_count(grid, method):
    R, C = len(grid), len(grid[0])
    bg = _48_bg(grid)
    ccs = _48_bfs_ccs(grid, skip_color=bg)
    if not ccs: return None

    out = _48_copy(grid)
    if method == "count_as_color":
        for color, cells in ccs:
            cnt = len(cells)
            if cnt > 9: return None  # ARC colors 0-9 only
            for r, c in cells:
                out[r][c] = cnt
    elif method == "rank_by_size_asc":
        sorted_ccs = sorted(ccs, key=lambda x: len(x[1]))
        for i, (color, cells) in enumerate(sorted_ccs):
            nc = i + 1
            if nc > 9: return None
            for r, c in cells:
                out[r][c] = nc
    elif method == "rank_by_size_desc":
        sorted_ccs = sorted(ccs, key=lambda x: -len(x[1]))
        for i, (color, cells) in enumerate(sorted_ccs):
            nc = i + 1
            if nc > 9: return None
            for r, c in cells:
                out[r][c] = nc
    else:
        return None
    return out


# ─── Solver 7: FillInteriorHolesSolver ───────────────────────────────────────

@register_solver
class FillInteriorHolesSolver(BaseSolver):
    """
    Fill enclosed regions of bg color (holes) with a specific fill color.
    Holes = bg cells not reachable from any border cell.
    """
    @property
    def name(self): return "FillInteriorHolesSolver"
    @property
    def priority(self): return 60

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        fill_color = None
        for pair in pairs:
            fc, out = _48_fill_holes(pair['input'])
            if out is None: return None
            if not _48_eq(out, pair['output']): return None
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return None

        preds = []
        for test_inp in ctx.test_inputs:
            fc, out = _48_fill_holes(test_inp)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.85))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_fill_holes(grid):
    """Flood fill from border to find reachable bg cells; remaining bg = holes."""
    R, C = len(grid), len(grid[0])
    bg = _48_bg(grid)

    # Flood fill from border
    reachable = [[False]*C for _ in range(R)]
    queue = deque()
    for r in range(R):
        for c in [0, C-1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    for c in range(C):
        for r in [0, R-1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and not reachable[nr][nc] and grid[nr][nc]==bg:
                reachable[nr][nc] = True
                queue.append((nr, nc))

    # Find interior bg cells
    holes = [(r,c) for r in range(R) for c in range(C)
             if grid[r][c] == bg and not reachable[r][c]]
    if not holes: return None, None

    # Fill with the color of non-bg non-border neighbors (most common surrounding color)
    surrounding = []
    for r, c in holes:
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc] != bg:
                surrounding.append(grid[nr][nc])
    if not surrounding: return None, None

    fill_color = Counter(surrounding).most_common(1)[0][0]
    out = _48_copy(grid)
    for r, c in holes:
        out[r][c] = fill_color
    return fill_color, out


# ─── Solver 8: RowColSymmetrySolver ──────────────────────────────────────────

@register_solver
class RowColSymmetrySolver(BaseSolver):
    """
    Complete symmetric grid: detect which cells break symmetry and correct them
    using the majority rule (most consistent value for symmetric pairs).
    """
    @property
    def name(self): return "RowColSymmetrySolver"
    @property
    def priority(self): return 61

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        sym_type = None
        for stype in ("lr", "ud", "both"):
            ok = True
            for pair in pairs:
                out = _48_complete_symmetry(pair['input'], stype)
                if out is None or not _48_eq(out, pair['output']):
                    ok = False; break
            if ok:
                sym_type = stype; break

        if sym_type is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _48_complete_symmetry(test_inp, sym_type)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.84))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_complete_symmetry(grid, sym_type):
    R, C = len(grid), len(grid[0])
    out = _48_copy(grid)

    if sym_type in ("lr", "both"):
        for r in range(R):
            for c in range(C // 2):
                c2 = C - 1 - c
                if out[r][c] != out[r][c2]:
                    # Use non-zero value if one is zero, otherwise take first
                    if out[r][c] == 0:
                        out[r][c] = out[r][c2]
                    else:
                        out[r][c2] = out[r][c]

    if sym_type in ("ud", "both"):
        for r in range(R // 2):
            r2 = R - 1 - r
            for c in range(C):
                if out[r][c] != out[r2][c]:
                    if out[r][c] == 0:
                        out[r][c] = out[r2][c]
                    else:
                        out[r2][c] = out[r][c]

    if _48_eq(out, grid): return None  # no change → not this pattern
    return out


# ─── Solver 9: SelectUniqueShapeSolver ───────────────────────────────────────

@register_solver
class SelectUniqueShapeSolver(BaseSolver):
    """
    From input containing several similar objects and one unique one (different
    size, color, or shape), output = the unique object cropped to its bounding box.
    """
    @property
    def name(self): return "SelectUniqueShapeSolver"
    @property
    def priority(self): return 62

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        method = None
        for m in ("unique_color", "unique_size", "unique_shape"):
            ok = True
            for pair in pairs:
                out = _48_select_unique(pair['input'], m)
                if out is None or not _48_eq(out, pair['output']):
                    ok = False; break
            if ok:
                method = m; break

        if method is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _48_select_unique(test_inp, method)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.82))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_select_unique(grid, method):
    R, C = len(grid), len(grid[0])
    bg = _48_bg(grid)
    ccs = _48_bfs_ccs(grid, skip_color=bg)
    if len(ccs) < 2: return None

    if method == "unique_color":
        color_count = Counter(color for color, _ in ccs)
        unique_colors = [c for c, n in color_count.items() if n == 1]
        if len(unique_colors) != 1: return None
        target_cells = next(cells for color, cells in ccs if color == unique_colors[0])

    elif method == "unique_size":
        size_count = Counter(len(cells) for _, cells in ccs)
        unique_sizes = [s for s, n in size_count.items() if n == 1]
        if len(unique_sizes) != 1: return None
        target_cells = next(cells for _, cells in ccs if len(cells) == unique_sizes[0])

    elif method == "unique_shape":
        # Shape = normalized relative positions
        def normalize(cells):
            rs = [r for r, c in cells]; cs = [c for r, c in cells]
            mr, mc = min(rs), min(cs)
            return frozenset((r-mr, c-mc) for r, c in cells)
        shapes = [normalize(cells) for _, cells in ccs]
        shape_count = Counter(shapes)
        unique_shapes = [s for s, n in shape_count.items() if n == 1]
        if len(unique_shapes) != 1: return None
        idx = shapes.index(unique_shapes[0])
        target_cells = list(ccs)[idx][1]
    else:
        return None

    # Crop to bounding box
    rs = [r for r, c in target_cells]; cs = [c for r, c in target_cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    out = [[grid[r][c] for c in range(c0, c1+1)] for r in range(r0, r1+1)]
    return out


# ─── Solver 10: ColorByNeighborCountSolver ───────────────────────────────────

@register_solver
class ColorByNeighborCountSolver(BaseSolver):
    """
    Each non-bg cell is recolored based on the count of its non-bg neighbors.
    color = number of non-bg neighbors (0-8 for 8-connected, 0-4 for 4-connected).
    """
    @property
    def name(self): return "ColorByNeighborCountSolver"
    @property
    def priority(self): return 63

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        pairs = ctx.train_pairs
        if not pairs: return None

        connect = None
        for conn in (4, 8):
            ok = True
            for pair in pairs:
                out = _48_color_by_neighbor(pair['input'], conn)
                if out is None or not _48_eq(out, pair['output']):
                    ok = False; break
            if ok:
                connect = conn; break

        if connect is None: return None

        preds = []
        for test_inp in ctx.test_inputs:
            out = _48_color_by_neighbor(test_inp, connect)
            if out is None: return None
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=0.80))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


def _48_color_by_neighbor(grid, connect):
    R, C = len(grid), len(grid[0])
    bg = _48_bg(grid)
    dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
    dirs8 = dirs4 + [(-1,-1),(-1,1),(1,-1),(1,1)]
    dirs = dirs4 if connect == 4 else dirs8

    out = _48_copy(grid)
    changed = False
    for r in range(R):
        for c in range(C):
            if grid[r][c] == bg: continue
            cnt = sum(1 for dr,dc in dirs
                      if 0<=r+dr<R and 0<=c+dc<C and grid[r+dr][c+dc] != bg)
            out[r][c] = cnt
            if cnt != grid[r][c]: changed = True
    if not changed: return None
    return out
