# advanced_solvers43.py — Session 43 (+4 solvers) [PATCHED: training validation added]
# Targets: e21a174a, 9ddd00f0, f45f5ca7, d968ffd4

from collections import Counter, defaultdict
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver


# ═══ Session-43 local helpers ════════════════════════════════════════════════

def _bg43(grid):
    flat = [c for row in grid for c in row]
    return Counter(flat).most_common(1)[0][0]


def _bfs_ccs43(grid, bg):
    """Same-color 4-connected components (non-bg). Returns [(color, [(r,c)...]), ...]."""
    H, W = len(grid), len(grid[0])
    vis = [[False] * W for _ in range(H)]
    ccs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and not vis[r][c]:
                col = grid[r][c]
                cells = []
                stk = [(r, c)]
                while stk:
                    rr, cc = stk.pop()
                    if rr < 0 or rr >= H or cc < 0 or cc >= W:
                        continue
                    if vis[rr][cc] or grid[rr][cc] != col:
                        continue
                    vis[rr][cc] = True
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        stk.append((rr + dr, cc + dc))
                ccs.append((col, cells))
    return ccs


def _group_runs(indices):
    """Group a sorted list of integers into contiguous runs: [(start, end), ...]."""
    if not indices:
        return []
    groups = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            groups.append((start, prev))
            start = prev = i
    groups.append((start, prev))
    return groups


# ═══ SESSION 43 SOLVERS (+4) ══════════════════════════════════════════════════

@register_solver
class LayerReversalSolver(BaseSolver):
    """Reverses the ORDER of color-blocks in non-bg rows using RLE segmentation.
    Target: e21a174a
    """

    @property
    def name(self):
        return "LayerReversalSolver"

    @staticmethod
    def _dominant_non_bg_color(row, bg):
        non_bg = [c for c in row if c != bg]
        if not non_bg:
            return bg
        return Counter(non_bg).most_common(1)[0][0]

    def _transform_one(self, inp):
        """Apply layer-reversal transform to a single grid."""
        bg = _bg43(inp)
        H = len(inp)
        segments = []
        cur_color, cur_rows = None, []
        for r in range(H):
            dom = self._dominant_non_bg_color(inp[r], bg)
            if dom == bg:
                if cur_rows:
                    segments.append((cur_color, cur_rows))
                    cur_color, cur_rows = None, []
            else:
                if dom == cur_color:
                    cur_rows.append(r)
                else:
                    if cur_rows:
                        segments.append((cur_color, cur_rows))
                    cur_color, cur_rows = dom, [r]
        if cur_rows:
            segments.append((cur_color, cur_rows))
        out = [list(row) for row in inp]
        if not segments:
            return out
        nb_rows = [r for _, rows in segments for r in rows]
        reversed_rows = [r for _, rows in reversed(segments) for r in rows]
        for nb_pos, src_r in zip(nb_rows, reversed_rows):
            out[nb_pos] = list(inp[src_r])
        return out

    def solve(self, ctx):
        # GUARD: verify transform on ALL training pairs
        for pair in ctx.train_pairs:
            pred = self._transform_one(pair["input"])
            if pred != pair["output"]:
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform_one(test_inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class GridBlockFillSolver(BaseSolver):
    """Fills empty blocks in a regular grid-of-blocks layout.
    Target: 9ddd00f0
    """

    @property
    def name(self):
        return "GridBlockFillSolver"

    def _transform_one(self, inp):
        """Apply the block-fill transform to a single grid."""
        H, W = len(inp), len(inp[0])
        bg = _bg43(inp)
        nz_rows = [r for r in range(H) if any(inp[r][c] != bg for c in range(W))]
        nz_cols = [c for c in range(W) if any(inp[r][c] != bg for r in range(H))]
        if not nz_rows or not nz_cols:
            return [list(row) for row in inp]
        row_groups = _group_runs(nz_rows)
        col_groups = _group_runs(nz_cols)
        block_h = Counter(e - s + 1 for s, e in row_groups).most_common(1)[0][0]
        block_w = Counter(e - s + 1 for s, e in col_groups).most_common(1)[0][0]
        period_h = block_h + 1
        period_w = block_w + 1
        first_r = row_groups[0][0]
        all_row_starts = []
        r = first_r
        while r < H:
            all_row_starts.append(r); r += period_h
        r = first_r - period_h
        while r >= 0:
            all_row_starts.insert(0, r); r -= period_h
        first_c = col_groups[0][0]
        all_col_starts = []
        c = first_c
        while c < W:
            all_col_starts.append(c); c += period_w
        c = first_c - period_w
        while c >= 0:
            all_col_starts.insert(0, c); c -= period_w
        block_row_ranges = [(rs, min(rs + block_h - 1, H - 1))
                            for rs in all_row_starts if 0 <= rs < H]
        block_col_ranges = [(cs, min(cs + block_w - 1, W - 1))
                            for cs in all_col_starts if 0 <= cs < W]
        nb_vals = [inp[r][c] for r in range(H) for c in range(W) if inp[r][c] != bg]
        if not nb_vals:
            return [list(row) for row in inp]
        fill_color = Counter(nb_vals).most_common(1)[0][0]
        out = [list(row) for row in inp]
        for gr, (br_s, br_e) in enumerate(block_row_ranges):
            for gc, (bc_s, bc_e) in enumerate(block_col_ranges):
                if any(inp[r][c] != bg
                       for r in range(br_s, br_e + 1)
                       for c in range(bc_s, bc_e + 1)):
                    continue
                hole_r = br_s + gr
                hole_c = bc_s + gc
                for r in range(br_s, br_e + 1):
                    for c in range(bc_s, bc_e + 1):
                        out[r][c] = bg if (r == hole_r and c == hole_c) else fill_color
        return out

    def solve(self, ctx):
        # GUARD: verify transform on ALL training pairs
        for pair in ctx.train_pairs:
            pred = self._transform_one(pair["input"])
            if pred != pair["output"]:
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform_one(test_inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class ColorToColumnSolver(BaseSolver):
    """Learns a color->column mapping from training pairs, then applies it.
    Target: f45f5ca7
    """

    @property
    def name(self):
        return "ColorToColumnSolver"

    def _learn(self, train_pairs):
        color_col = {}
        for pair in train_pairs:
            inp, out = pair["input"], pair["output"]
            bg_in = _bg43(inp)
            H_i = len(inp)
            W_o = len(out[0]) if out else 0
            for r in range(min(H_i, len(out))):
                v = inp[r][0]
                if v != bg_in:
                    for c in range(W_o):
                        if out[r][c] == v:
                            color_col[v] = c
                            break
        return color_col

    def _apply_one(self, inp, color_col):
        H, W = len(inp), len(inp[0])
        bg = _bg43(inp)
        out = [[bg] * W for _ in range(H)]
        for r in range(H):
            v = inp[r][0]
            if v != bg:
                if v in color_col and 0 <= color_col[v] < W:
                    out[r][color_col[v]] = v
                elif 0 <= v < W:
                    out[r][v] = v
        return out

    def solve(self, ctx):
        color_col = self._learn(ctx.train_pairs)
        if not color_col:
            return None
        # GUARD: verify mapping on ALL training pairs
        for pair in ctx.train_pairs:
            pred = self._apply_one(pair["input"], color_col)
            if pred != pair["output"]:
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._apply_one(test_inp, color_col)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class FillTowardCenterSolver(BaseSolver):
    """Fills the gap between two colored objects symmetrically toward center.
    Target: d968ffd4
    """

    @property
    def name(self):
        return "FillTowardCenterSolver"

    @staticmethod
    def _apply_fill_gaps(out, gap_s, gap_e, col_a, col_b, H, W, axis):
        gap_len = gap_e - gap_s + 1
        if gap_len <= 0:
            return
        mid = (gap_s + gap_e) // 2
        for idx in range(gap_s, gap_e + 1):
            if gap_len % 2 == 1:
                if idx < mid:
                    fill = col_a
                elif idx > mid:
                    fill = col_b
                else:
                    continue
            else:
                fill = col_a if idx <= mid else col_b
            if axis == "h":
                for r in range(H):
                    out[r][idx] = fill
            else:
                for c in range(W):
                    out[idx][c] = fill

    def _transform_one(self, inp):
        H, W = len(inp), len(inp[0])
        bg = _bg43(inp)
        ccs = _bfs_ccs43(inp, bg)
        out = [list(row) for row in inp]
        if len(ccs) < 2:
            return out
        ccs_s = sorted(ccs, key=lambda x: len(x[1]), reverse=True)
        col_a, cells_a = ccs_s[0]
        col_b, cells_b = ccs_s[1]
        max_col_a = max(c for r, c in cells_a)
        min_col_b = min(c for r, c in cells_b)
        max_col_b = max(c for r, c in cells_b)
        min_col_a = min(c for r, c in cells_a)
        max_row_a = max(r for r, c in cells_a)
        min_row_b = min(r for r, c in cells_b)
        max_row_b = max(r for r, c in cells_b)
        min_row_a = min(r for r, c in cells_a)
        if max_col_a < min_col_b:
            self._apply_fill_gaps(out, max_col_a + 1, min_col_b - 1,
                                   col_a, col_b, H, W, "h")
        elif max_col_b < min_col_a:
            self._apply_fill_gaps(out, max_col_b + 1, min_col_a - 1,
                                   col_b, col_a, H, W, "h")
        elif max_row_a < min_row_b:
            self._apply_fill_gaps(out, max_row_a + 1, min_row_b - 1,
                                   col_a, col_b, H, W, "v")
        elif max_row_b < min_row_a:
            self._apply_fill_gaps(out, max_row_b + 1, min_row_a - 1,
                                   col_b, col_a, H, W, "v")
        return out

    def solve(self, ctx):
        # GUARD: verify transform on ALL training pairs
        for pair in ctx.train_pairs:
            pred = self._transform_one(pair["input"])
            if pred != pair["output"]:
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform_one(test_inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                    solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
