"""
Session 42 solvers — 4 confirmed rules.
Confirmed targets: 913fb3ed, 508bd3b6, 56dc2b01, 575b1a71
"""
from collections import Counter, defaultdict
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver


def _bg(grid):
    flat = [v for row in grid for v in row]
    return Counter(flat).most_common(1)[0][0]


# ════════════════════════════════════════════════════════════════════
# S42-1  DotTo3x3BoxSolver  →  913fb3ed
# Each isolated dot → 3×3 box: center keeps dot color,
# ring cells get a training-learned border color per dot color.
# ════════════════════════════════════════════════════════════════════
@register_solver
class DotTo3x3BoxSolver(BaseSolver):
    @property
    def name(self): return "DotTo3x3BoxSolver"

    _RING = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _learn(self, train_pairs):
        bg_val = _bg(train_pairs[0]['input'])
        cmap = {}
        for pair in train_pairs:
            inp, out = pair['input'], pair['output']
            H, W = len(inp), len(inp[0])
            for r in range(H):
                for c in range(W):
                    k = inp[r][c]
                    if k == bg_val or k in cmap:
                        continue
                    ring_cols = []
                    for dr, dc in self._RING:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            v = out[nr][nc]
                            if v not in (bg_val, k):
                                ring_cols.append(v)
                    if ring_cols:
                        cmap[k] = Counter(ring_cols).most_common(1)[0][0]
        return cmap

    def _is_valid(self, ctx):
        for pair in ctx.train_pairs:
            inp, out = pair['input'], pair['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return False
            bg = _bg(inp)
            H, W = len(inp), len(inp[0])
            dots = [(r, c, inp[r][c]) for r in range(H) for c in range(W)
                    if inp[r][c] != bg]
            if not dots:
                return False
            # Each dot must keep its color at its position in output
            for r, c, v in dots:
                if out[r][c] != v:
                    return False
            # At least one dot must have a non-bg ring in output
            ok = False
            for r, c, v in dots:
                for dr, dc in self._RING:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W and out[nr][nc] not in (bg, v):
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                return False
        return True

    def _apply_dot(self, inp, cmap):
        from collections import Counter
        bg = _bg(inp)
        H, W = len(inp), len(inp[0])
        out = [row[:] for row in inp]
        for r in range(H):
            for c in range(W):
                k = inp[r][c]
                if k == bg:
                    continue
                bcol = cmap.get(k, k)
                for dr, dc in self._RING:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < H and 0 <= nc < W:
                        out[nr][nc] = bcol
        return out

    def solve(self, ctx):
        if not self._is_valid(ctx):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        cmap = self._learn(ctx.train_pairs)
        if not cmap:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        # GUARD: verify full transform on ALL training pairs
        for pair in ctx.train_pairs:
            pred = self._apply_dot(pair["input"], cmap)
            if pred != pair["output"]:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._apply_dot(test_inp, cmap)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ════════════════════════════════════════════════════════════════════
# S42-2  DiagonalBounceWallSolver  →  508bd3b6
# A diagonal "ball" travels toward a full-row/col "wall", bounces once
# (flips normal component) and draws its path in a trace color.
# ════════════════════════════════════════════════════════════════════
@register_solver
class DiagonalBounceWallSolver(BaseSolver):
    @property
    def name(self): return "DiagonalBounceWallSolver"

    def _get_walls(self, grid):
        H, W = len(grid), len(grid[0])
        walls = []
        for r in range(H):
            if len(set(grid[r])) == 1 and grid[r][0] != 0:
                walls.append(('row', r, grid[r][0]))
        for c in range(W):
            col = [grid[r][c] for r in range(H)]
            if len(set(col)) == 1 and col[0] != 0:
                walls.append(('col', c, col[0]))
        return walls

    def _get_ball(self, grid, wall_colors):
        """Return (sorted_cells, color) for the first diagonal run ≥2 cells."""
        bg = _bg(grid)
        H, W = len(grid), len(grid[0])
        by_color = defaultdict(list)
        for r in range(H):
            for c in range(W):
                v = grid[r][c]
                if v != bg and v not in wall_colors:
                    by_color[v].append((r, c))
        for color, cells in by_color.items():
            if len(cells) < 2:
                continue
            cs = sorted(cells)  # sort by row
            drs = [cs[i+1][0]-cs[i][0] for i in range(len(cs)-1)]
            dcs = [cs[i+1][1]-cs[i][1] for i in range(len(cs)-1)]
            if (all(d == 1 for d in drs) and
                    (all(d == 1 for d in dcs) or all(d == -1 for d in dcs))):
                return cs, color
        return None, None

    def _learn_trace_color(self, ctx):
        for pair in ctx.train_pairs:
            inp, out = pair['input'], pair['output']
            extra = ({v for row in out for v in row if v != 0} -
                     {v for row in inp for v in row if v != 0})
            if extra:
                return next(iter(extra))
        return None

    def _is_valid(self, ctx):
        for pair in ctx.train_pairs:
            inp, out = pair['input'], pair['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return False
            walls = self._get_walls(inp)
            if not walls:
                return False
            ball, _ = self._get_ball(inp, {w[2] for w in walls})
            if ball is None:
                return False
            extra = ({v for row in out for v in row if v != 0} -
                     {v for row in inp for v in row if v != 0})
            if not extra:
                return False
        return True

    def solve(self, ctx):
        if not self._is_valid(ctx):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        trace_color = self._learn_trace_color(ctx)
        if trace_color is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)

        preds = []
        for test_inp in ctx.test_inputs:
            H, W = len(test_inp), len(test_inp[0])
            walls = self._get_walls(test_inp)
            if not walls:
                continue
            wc_set = {w[2] for w in walls}
            ball, _ = self._get_ball(test_inp, wc_set)
            if ball is None:
                continue

            row_idxs = sorted({w[1] for w in walls if w[0] == 'row'})
            col_idxs = sorted({w[1] for w in walls if w[0] == 'col'})

            ball_rs = [x[0] for x in ball]
            ball_cs = [x[1] for x in ball]

            # Choose wall type: prefer the one the ball is heading toward
            use_row = bool(row_idxs)
            use_col = bool(col_idxs)
            if use_row and use_col:
                # Use whichever wall is not "between" the ball cells
                use_col = False  # default; row walls take priority

            if use_row:
                wall_type = 'row'
                wall_set = set(row_idxs)
                wmin = min(row_idxs)
                if wmin > max(ball_rs):        # wall below
                    leading  = max(ball, key=lambda x: x[0])
                    trailing = min(ball, key=lambda x: x[0])
                else:                           # wall above
                    leading  = min(ball, key=lambda x: x[0])
                    trailing = max(ball, key=lambda x: x[0])
            else:
                wall_type = 'col'
                wall_set = set(col_idxs)
                wmin = min(col_idxs)
                if wmin > max(ball_cs):        # wall right
                    leading  = max(ball, key=lambda x: x[1])
                    trailing = min(ball, key=lambda x: x[1])
                else:                           # wall left
                    leading  = min(ball, key=lambda x: x[1])
                    trailing = max(ball, key=lambda x: x[1])

            # Direction: trailing → leading
            dr = (1  if leading[0] > trailing[0] else
                  -1 if leading[0] < trailing[0] else 0)
            dc = (1  if leading[1] > trailing[1] else
                  -1 if leading[1] < trailing[1] else 0)

            out = [row[:] for row in test_inp]
            r, c = leading[0], leading[1]
            bounced = False
            for _ in range((H + W) * 3):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                hit = ((wall_type == 'row' and nr in wall_set) or
                       (wall_type == 'col' and nc in wall_set))
                if hit:
                    if bounced:
                        break
                    if wall_type == 'row':
                        dr = -dr
                    else:
                        dc = -dc
                    bounced = True
                    # direction flipped; don't move this step
                else:
                    out[nr][nc] = trace_color
                    r, c = nr, nc

            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ════════════════════════════════════════════════════════════════════
# S42-3  GravityShapeToWallSolver  →  56dc2b01
# Shape slides toward a full-row/col wall until adjacent.
# An 8-marker (full row/col of 8s) is placed one step behind the
# trailing edge of the moved shape.
# ════════════════════════════════════════════════════════════════════
@register_solver
class GravityShapeToWallSolver(BaseSolver):
    @property
    def name(self): return "GravityShapeToWallSolver"

    def _learn_wall_color(self, train_pairs):
        """Return the color of full row/col that stays in-place across all pairs."""
        candidates = None
        for pair in train_pairs:
            inp, out = pair['input'], pair['output']
            # Guard: output must match input dimensions
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return None
            bg = _bg(inp)
            H, W = len(inp), len(inp[0])
            pair_colors = set()
            for r in range(H):
                vals = set(inp[r])
                if len(vals) == 1:
                    v = next(iter(vals))
                    if v != 0 and v != bg and all(out[r][c] == v for c in range(W)):
                        pair_colors.add(v)
            for c in range(W):
                col_in = [inp[r][c] for r in range(H)]
                vals = set(col_in)
                if len(vals) == 1:
                    v = next(iter(vals))
                    if v != 0 and v != bg and all(out[r][c] == v for r in range(H)):
                        pair_colors.add(v)
            candidates = pair_colors if candidates is None else (candidates & pair_colors)
        return next(iter(candidates)) if candidates else None

    def _get_wall(self, grid, wall_color=None):
        """Return (wtype, sorted_idxs, wcolor) or (None, None, None)."""
        bg = _bg(grid)
        H, W = len(grid), len(grid[0])
        # Row walls
        rg = defaultdict(list)
        for r in range(H):
            vals = set(grid[r])
            if len(vals) == 1:
                v = next(iter(vals))
                if v != 0 and v != bg and (wall_color is None or v == wall_color):
                    rg[v].append(r)
        if rg:
            wc, idxs = max(rg.items(), key=lambda x: len(x[1]))
            return 'row', sorted(idxs), wc
        # Col walls
        cg = defaultdict(list)
        for c in range(W):
            col = [grid[r][c] for r in range(H)]
            vals = set(col)
            if len(vals) == 1:
                v = next(iter(vals))
                if v != 0 and v != bg and (wall_color is None or v == wall_color):
                    cg[v].append(c)
        if cg:
            wc, idxs = max(cg.items(), key=lambda x: len(x[1]))
            return 'col', sorted(idxs), wc
        return None, None, None

    def _is_valid(self, ctx):
        wcolor = self._learn_wall_color(ctx.train_pairs)
        if wcolor is None:
            return False
        for pair in ctx.train_pairs:
            inp, out = pair['input'], pair['output']
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                return False
            bg = _bg(inp)
            wt, widxs, wc = self._get_wall(inp, wall_color=wcolor)
            if wt is None:
                return False
            H, W = len(inp), len(inp[0])
            shape = [(r, c) for r in range(H) for c in range(W)
                     if inp[r][c] != bg and inp[r][c] != wc]
            if not shape:
                return False
            # Output must contain a full row or col of 8s
            has_8 = False
            if wt == 'row':
                for r in range(H):
                    if all(out[r][c] == 8 for c in range(W)):
                        has_8 = True
                        break
            else:
                for c in range(W):
                    if all(out[r][c] == 8 for r in range(H)):
                        has_8 = True
                        break
            if not has_8:
                return False
        return True

    def _apply_one(self, inp, wcolor):
        """Apply the gravity transform to one input grid. Returns output grid or None."""
        bg = _bg(inp)
        H, W = len(inp), len(inp[0])
        wt, widxs, wc = self._get_wall(inp, wall_color=wcolor)
        if wt is None:
            return None
        shape = [(r, c, inp[r][c]) for r in range(H) for c in range(W)
                 if inp[r][c] != bg and inp[r][c] != wc]
        if not shape:
            return None
        srs = [x[0] for x in shape]
        scs = [x[1] for x in shape]
        if wt == 'row':
            if max(widxs) < min(srs):
                target  = max(widxs) + 1
                shift_r = target - min(srs)
            else:
                target  = min(widxs) - 1
                shift_r = target - max(srs)
            new_shape = [(r + shift_r, c, v) for r, c, v in shape]
            new_rs = [x[0] for x in new_shape]
            marker_idx = (max(new_rs) + 1 if max(widxs) < min(srs)
                          else min(new_rs) - 1)
        else:
            if max(widxs) < min(scs):
                target  = max(widxs) + 1
                shift_c = target - min(scs)
            else:
                target  = min(widxs) - 1
                shift_c = target - max(scs)
            new_shape = [(r, c + shift_c, v) for r, c, v in shape]
            new_cs = [x[1] for x in new_shape]
            marker_idx = (max(new_cs) + 1 if max(widxs) < min(scs)
                          else min(new_cs) - 1)
        out = [[bg] * W for _ in range(H)]
        if wt == 'row':
            for wi in widxs:
                for c in range(W):
                    out[wi][c] = wc
        else:
            for wi in widxs:
                for r in range(H):
                    out[r][wi] = wc
        for r, c, v in new_shape:
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = v
        if wt == 'row' and 0 <= marker_idx < H:
            for c in range(W):
                out[marker_idx][c] = 8
        elif wt == 'col' and 0 <= marker_idx < W:
            for r in range(H):
                out[r][marker_idx] = 8
        return out

    def solve(self, ctx):
        if not self._is_valid(ctx):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        wcolor = self._learn_wall_color(ctx.train_pairs)
        # GUARD: end-to-end training verification
        for pair in ctx.train_pairs:
            pred = self._apply_one(pair['input'], wcolor)
            if pred is None or pred != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)

        preds = []
        for test_inp in ctx.test_inputs:
            out = self._apply_one(test_inp, wcolor)
            if out is None:
                continue
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


# ════════════════════════════════════════════════════════════════════
# S42-4  HolesToColumnColorsSolver  →  575b1a71
# Grid is bg-5 with 0-holes.  Distinct hole-columns sorted left-right
# are assigned colors by training-learned rank map.
# ════════════════════════════════════════════════════════════════════
@register_solver
class HolesToColumnColorsSolver(BaseSolver):
    @property
    def name(self): return "HolesToColumnColorsSolver"

    def _learn_rank_map(self, train_pairs):
        rmap = {}
        for pair in train_pairs:
            inp, out = pair['input'], pair['output']
            H, W = len(inp), len(inp[0])
            hole_cols = sorted({c for r in range(H) for c in range(W)
                                if inp[r][c] == 0})
            for rank, col in enumerate(hole_cols):
                if rank in rmap:
                    continue
                fills = [out[r][col] for r in range(H) if inp[r][col] == 0]
                if fills:
                    rmap[rank] = Counter(fills).most_common(1)[0][0]
        return rmap

    def _is_valid(self, ctx):
        for pair in ctx.train_pairs:
            inp = pair['input']
            if _bg(inp) != 5:
                return False
            H, W = len(inp), len(inp[0])
            # Input must contain only 0s and 5s
            for row in inp:
                for v in row:
                    if v not in (0, 5):
                        return False
            # Must have at least one hole
            if not any(inp[r][c] == 0
                       for r in range(H) for c in range(W)):
                return False
        return True

    def solve(self, ctx):
        if not self._is_valid(ctx):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        rmap = self._learn_rank_map(ctx.train_pairs)
        if not rmap:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        # GUARD: verify training predictions match ground truth exactly
        for pair in ctx.train_pairs:
            t_inp, t_out = pair['input'], pair['output']
            tH, tW = len(t_inp), len(t_inp[0])
            t_hole_cols = sorted({c for r in range(tH) for c in range(tW)
                                  if t_inp[r][c] == 0})
            t_pred = [row[:] for row in t_inp]
            for rank, col in enumerate(t_hole_cols):
                if rank not in rmap:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
                color = rmap[rank]
                for r in range(tH):
                    if t_inp[r][col] == 0:
                        t_pred[r][col] = color
            if t_pred != t_out:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
        preds = []
        for test_inp in ctx.test_inputs:
            H, W = len(test_inp), len(test_inp[0])
            hole_cols = sorted({c for r in range(H) for c in range(W)
                                if test_inp[r][c] == 0})
            # Reject if test has unknown rank
            if any(rank not in rmap for rank in range(len(hole_cols))):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=2)
            out = [row[:] for row in test_inp]
            for rank, col in enumerate(hole_cols):
                color = rmap[rank]
                for r in range(H):
                    if test_inp[r][col] == 0:
                        out[r][col] = color
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
