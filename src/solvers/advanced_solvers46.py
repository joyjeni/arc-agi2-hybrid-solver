"""
Session 46 Solvers — confirmed-pattern implementations
All helpers prefixed _46_ to avoid name collisions.

Solvers:
  1. IdentityMatrixFromCCSolver       — n CCs → n×n identity matrix (d0f5fe59)
  2. ReplaceUniqueColorWithUniversalSolver — per-pair unique color → cross-pair universal (63613498)
  3. RemoveInteriorFillSolver         — enclosed non-bg cells → bg (9720b24f style)
  4. SwapTwoCellPositionsSolver       — exactly 2 changed cells swap colors (97c75046)
  5. AdjacentMoveToNeighborInRowSolver — isolated row cells move to be adjacent to nearest (494ef9d7)
  6. RemoveIsolatedSingleCellsSolver  — single-cell non-bg CCs → bg
  7. FillInteriorBgWithColorSolver    — enclosed bg cells → a specific fill color
"""

from collections import deque, Counter
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

# ─── Session 46 Helpers ──────────────────────────────────────────────────────

def _46_bg(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]

def _46_bfs_ccs(grid, skip_color=None):
    """Return list of (color, frozenset_of_cells) for all CCs, skipping skip_color."""
    R, C = len(grid), len(grid[0])
    visited = [[False] * C for _ in range(R)]
    result = []
    for r in range(R):
        for c in range(C):
            if visited[r][c]:
                continue
            color = grid[r][c]
            if skip_color is not None and color == skip_color:
                visited[r][c] = True
                continue
            queue = deque([(r, c)])
            visited[r][c] = True
            cells = {(r, c)}
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.add((nr, nc))
            result.append((color, frozenset(cells)))
    return result

def _46_copy(grid):
    return [list(row) for row in grid]

def _46_eq(a, b):
    if len(a) != len(b):
        return False
    for r in range(len(a)):
        if len(a[r]) != len(b[r]):
            return False
        for c in range(len(a[r])):
            if a[r][c] != b[r][c]:
                return False
    return True

def _46_can_reach_border(cells_set, grid, bg_color):
    """BFS from 'cells_set' through bg_color cells. Returns True if border reached."""
    R, C = len(grid), len(grid[0])
    visited = set(cells_set)
    queue = deque(cells_set)
    while queue:
        r, c = queue.popleft()
        if r == 0 or r == R - 1 or c == 0 or c == C - 1:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in visited and grid[nr][nc] == bg_color:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Solver 1: IdentityMatrixFromCCSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class IdentityMatrixFromCCSolver(BaseSolver):
    """
    Count n non-bg CCs in input → build n×n output grid:
      diagonal cells = fg_color, off-diagonal = out_bg_color.
    Confirmed for task d0f5fe59.
    """

    @property
    def name(self):
        return "IdentityMatrixFromCCSolver"

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if not pairs:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        fg_color = None
        out_bg_color = None

        for inp, out in pairs:
            bg = _46_bg(inp)
            ccs = _46_bfs_ccs(inp, skip_color=bg)
            n = len(ccs)
            if n < 2:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            # Output must be n×n
            if len(out) != n or any(len(row) != n for row in out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            o_bg = bg   # use input bg to avoid tie on small output grids
            # Diagonal cells must all be the same non-bg color
            diag_colors = {out[i][i] for i in range(n)}
            if len(diag_colors) != 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            diag_col = next(iter(diag_colors))
            if diag_col == o_bg:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            # Verify full grid
            ok = all(
                out[r][c] == (diag_col if r == c else o_bg)
                for r in range(n) for c in range(n)
            )
            if not ok:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            if fg_color is None:
                fg_color, out_bg_color = diag_col, o_bg
            elif fg_color != diag_col or out_bg_color != o_bg:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        # Apply to test
        preds = []
        for test_inp in ctx.test_inputs:
            bg = _46_bg(test_inp)
            ccs = _46_bfs_ccs(test_inp, skip_color=bg)
            n = len(ccs)
            if n < 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            grid = [[out_bg_color] * n for _ in range(n)]
            for i in range(n):
                grid[i][i] = fg_color
            preds.append(Prediction(attempt_1=grid, attempt_2=grid,
                                    solver_name=self.name, confidence=1.0))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 2: ReplaceUniqueColorWithUniversalSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class ReplaceUniqueColorWithUniversalSolver(BaseSolver):
    """
    Each training pair input has one 'unique' color that disappears in the output.
    Those cells are replaced by a single 'universal' color present in all pairs.
    Unique color differs per pair; replacement color is always the same.
    Confirmed for task 63613498 (colors 9, 6, 1 → 5).
    """

    @property
    def name(self):
        return "ReplaceUniqueColorWithUniversalSolver"

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if len(pairs) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        unique_colors = []
        target_colors = []

        for inp, out in pairs:
            in_colors = {c for row in inp for c in row}
            out_colors = {c for row in out for c in row}
            extra = in_colors - out_colors
            if len(extra) != 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            extra_color = next(iter(extra))
            # Find what color replaces extra_color
            replacements = set()
            for r in range(len(inp)):
                for c in range(len(inp[r])):
                    if inp[r][c] == extra_color:
                        replacements.add(out[r][c])
            if len(replacements) != 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            target = next(iter(replacements))
            unique_colors.append(extra_color)
            target_colors.append(target)

        # All targets must be the same universal color
        if len(set(target_colors)) != 1:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        target_color = target_colors[0]

        # All unique colors must be distinct per pair
        if len(set(unique_colors)) != len(unique_colors):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        # Full verification: replacing extra_color → target_color must reproduce output
        for (inp, out), extra_color in zip(pairs, unique_colors):
            pred = _46_copy(inp)
            for r in range(len(inp)):
                for c in range(len(inp[r])):
                    if pred[r][c] == extra_color:
                        pred[r][c] = target_color
            if not _46_eq(pred, out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        # Colors present in any training OUTPUT are "known" colors
        union_train_outputs = set()
        for _, out in pairs:
            union_train_outputs |= {c for row in out for c in row}

        # Apply to test
        preds = []
        for test_inp in ctx.test_inputs:
            test_colors = {c for row in test_inp for c in row}
            extra = test_colors - union_train_outputs
            if len(extra) != 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            extra_color = next(iter(extra))
            pred = _46_copy(test_inp)
            for r in range(len(test_inp)):
                for c in range(len(test_inp[r])):
                    if pred[r][c] == extra_color:
                        pred[r][c] = target_color
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.9))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 3: RemoveInteriorFillSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class RemoveInteriorFillSolver(BaseSolver):
    """
    Non-bg CCs whose adjacent bg cells cannot reach the grid border
    (i.e., they are enclosed inside a closed ring of another color) → set to bg.
    Confirmed for 9720b24f-style tasks.
    """

    @property
    def name(self):
        return "RemoveInteriorFillSolver"

    def _apply(self, grid, bg_color):
        R, C = len(grid), len(grid[0])
        ccs = _46_bfs_ccs(grid, skip_color=bg_color)
        result = _46_copy(grid)

        for color, cells in ccs:
            # Gather bg cells adjacent to this CC
            adj_bg = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == bg_color:
                        adj_bg.add((nr, nc))

            if not adj_bg:
                # Completely surrounded by non-bg → enclosed
                for r, c in cells:
                    result[r][c] = bg_color
                continue

            if not _46_can_reach_border(adj_bg, grid, bg_color):
                for r, c in cells:
                    result[r][c] = bg_color

        return result

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if not pairs:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        for inp, out in pairs:
            bg = _46_bg(inp)
            pred = self._apply(inp, bg)
            if not _46_eq(pred, out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _46_bg(test_inp)
            pred = self._apply(test_inp, bg)
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.9))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 4: SwapTwoCellPositionsSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class SwapTwoCellPositionsSolver(BaseSolver):
    """
    Exactly 2 cells differ between input and output, and they swap each other's
    color values (i.e. position swap of two distinct color objects).
    Confirmed for 97c75046.
    """

    @property
    def name(self):
        return "SwapTwoCellPositionsSolver"

    def _find_swap_colors(self, inp, out):
        """Returns (ca, cb) if exactly 2 cells changed and they swapped colors."""
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None  # different dimensions — not a cell-swap task
        R, C = len(inp), len(inp[0])
        changed = [(r, c) for r in range(R) for c in range(C) if inp[r][c] != out[r][c]]
        if len(changed) != 2:
            return None
        r1, c1 = changed[0]
        r2, c2 = changed[1]
        if inp[r1][c1] == out[r2][c2] and inp[r2][c2] == out[r1][c1]:
            return (inp[r1][c1], inp[r2][c2])
        return None

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if not pairs:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        swap_colors = None
        for inp, out in pairs:
            s = self._find_swap_colors(inp, out)
            if s is None:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            ca, cb = s
            if swap_colors is None:
                swap_colors = frozenset({ca, cb})
            elif swap_colors != frozenset({ca, cb}):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        if swap_colors is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        ca, cb = tuple(swap_colors)

        # Apply to test
        preds = []
        for test_inp in ctx.test_inputs:
            R, C = len(test_inp), len(test_inp[0])
            all_a = [(r, c) for r in range(R) for c in range(C) if test_inp[r][c] == ca]
            all_b = [(r, c) for r in range(R) for c in range(C) if test_inp[r][c] == cb]

            if not all_a or not all_b:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

            # If multiple cells of one color, find the nearest pair
            if len(all_a) == 1 and len(all_b) >= 1:
                pos_a = all_a[0]
                pos_b = min(all_b, key=lambda p: abs(p[0] - pos_a[0]) + abs(p[1] - pos_a[1]))
            elif len(all_b) == 1 and len(all_a) >= 1:
                pos_b = all_b[0]
                pos_a = min(all_a, key=lambda p: abs(p[0] - pos_b[0]) + abs(p[1] - pos_b[1]))
            elif len(all_a) == 1 and len(all_b) == 1:
                pos_a, pos_b = all_a[0], all_b[0]
            else:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

            pred = _46_copy(test_inp)
            pred[pos_a[0]][pos_a[1]] = cb
            pred[pos_b[0]][pos_b[1]] = ca
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.9))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 5: AdjacentMoveToNeighborInRowSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class AdjacentMoveToNeighborInRowSolver(BaseSolver):
    """
    Isolated non-bg cells in a row (with a gap to the nearest non-bg cell in
    the same row) move until they are adjacent (gap=0) to that neighbor.
    The cell closer to the right edge moves toward the left neighbor.
    Confirmed for 494ef9d7.
    """

    @property
    def name(self):
        return "AdjacentMoveToNeighborInRowSolver"

    def _apply(self, grid, bg_color):
        R, C = len(grid), len(grid[0])
        result = _46_copy(grid)

        for r in range(R):
            # Get sorted non-bg cells in this row
            nonbg = [(c, result[r][c]) for c in range(C) if result[r][c] != bg_color]
            if len(nonbg) < 2:
                continue

            nonbg.sort()
            changed = True
            # Iterative: keep moving until no more isolated gaps can be closed
            max_iter = C
            iteration = 0
            while changed and iteration < max_iter:
                changed = False
                iteration += 1
                nonbg = [(c, result[r][c]) for c in range(C) if result[r][c] != bg_color]
                nonbg.sort()
                for i in range(len(nonbg) - 1):
                    c1, col1 = nonbg[i]
                    c2, col2 = nonbg[i + 1]
                    gap = c2 - c1 - 1
                    if gap > 0:
                        # Move c2 one step toward c1
                        new_c2 = c2 - 1
                        if result[r][new_c2] == bg_color:
                            result[r][c2] = bg_color
                            result[r][new_c2] = col2
                            changed = True
                        break  # re-sort after each move

        return result

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if not pairs:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        # Must have at least one row change per pair
        any_change = False
        for inp, out in pairs:
            if not _46_eq(inp, out):
                any_change = True
            bg = _46_bg(inp)
            pred = self._apply(inp, bg)
            if not _46_eq(pred, out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        if not any_change:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _46_bg(test_inp)
            pred = self._apply(test_inp, bg)
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.9))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 6: RemoveIsolatedSingleCellsSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class RemoveIsolatedSingleCellsSolver(BaseSolver):
    """
    All non-bg CCs that consist of exactly 1 cell → set to bg.
    Everything else is unchanged.
    """

    @property
    def name(self):
        return "RemoveIsolatedSingleCellsSolver"

    def _apply(self, grid, bg_color):
        ccs = _46_bfs_ccs(grid, skip_color=bg_color)
        result = _46_copy(grid)
        for color, cells in ccs:
            if len(cells) == 1:
                r, c = next(iter(cells))
                result[r][c] = bg_color
        return result

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if not pairs:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        any_change = False
        for inp, out in pairs:
            bg = _46_bg(inp)
            pred = self._apply(inp, bg)
            if not _46_eq(pred, out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            if not _46_eq(inp, out):
                any_change = True

        if not any_change:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _46_bg(test_inp)
            pred = self._apply(test_inp, bg)
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.85))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ─────────────────────────────────────────────────────────────────────────────
# Solver 7: FillInteriorBgWithColorSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class FillInteriorBgWithColorSolver(BaseSolver):
    """
    Inverse of RemoveInteriorFillSolver.
    Bg cells that cannot reach the grid border (enclosed by non-bg rings)
    are filled with a specific color determined from training pairs.
    """

    @property
    def name(self):
        return "FillInteriorBgWithColorSolver"

    def _enclosed_bg_cells(self, grid, bg_color):
        """Return set of (r,c) bg cells that cannot reach the border through bg."""
        R, C = len(grid), len(grid[0])
        # BFS from border bg cells
        exterior = set()
        queue = deque()
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R - 1 or c == 0 or c == C - 1) and grid[r][c] == bg_color:
                    if (r, c) not in exterior:
                        exterior.add((r, c))
                        queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and (nr, nc) not in exterior and grid[nr][nc] == bg_color:
                    exterior.add((nr, nc))
                    queue.append((nr, nc))
        # All bg cells not in exterior = enclosed
        enclosed = {
            (r, c)
            for r in range(R) for c in range(C)
            if grid[r][c] == bg_color and (r, c) not in exterior
        }
        return enclosed

    def _apply(self, grid, bg_color, fill_color):
        result = _46_copy(grid)
        for r, c in self._enclosed_bg_cells(grid, bg_color):
            result[r][c] = fill_color
        return result

    def solve(self, ctx):
        pairs = [(p['input'], p['output']) for p in ctx.train_pairs]
        if not pairs:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        fill_color = None
        any_change = False

        for inp, out in pairs:
            bg = _46_bg(inp)
            enclosed = self._enclosed_bg_cells(inp, bg)
            if not enclosed:
                continue
            # Find what color the enclosed cells became
            fill_colors = {out[r][c] for r, c in enclosed}
            if len(fill_colors) != 1:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            fc = next(iter(fill_colors))
            if fc == bg:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            if fill_color is None:
                fill_color = fc
            elif fill_color != fc:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        if fill_color is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        # Full verification
        for inp, out in pairs:
            bg = _46_bg(inp)
            pred = self._apply(inp, bg, fill_color)
            if not _46_eq(pred, out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
            if not _46_eq(inp, out):
                any_change = True

        if not any_change:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)

        preds = []
        for test_inp in ctx.test_inputs:
            bg = _46_bg(test_inp)
            pred = self._apply(test_inp, bg, fill_color)
            preds.append(Prediction(attempt_1=pred, attempt_2=pred,
                                    solver_name=self.name, confidence=0.85))

        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
