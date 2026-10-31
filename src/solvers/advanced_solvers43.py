# advanced_solvers43.py — Session 43 (+4 solvers)
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
    Algorithm (RLE by dominant row color):
      1. For each row, compute dominant non-bg color (treat as bg if all-bg row).
      2. Group consecutive rows with the SAME dominant non-bg color → segments
         [(color, [row_indices]), ...].  bg-rows are skipped (not in any segment).
      3. nb_rows  = all non-bg row indices in original order (flattened segments).
      4. reversed_rows = row indices from reversed(segments), keeping internal
         order within each segment intact.
      5. For each (nb_pos, src_r) in zip(nb_rows, reversed_rows):
             out[nb_pos] = input[src_r]
    Target: e21a174a
    """

    @property
    def name(self):
        return "LayerReversalSolver"

    @staticmethod
    def _dominant_non_bg_color(row, bg):
        """Return the most common non-bg color in a row, or bg if the row is all-bg."""
        non_bg = [c for c in row if c != bg]
        if not non_bg:
            return bg
        return Counter(non_bg).most_common(1)[0][0]

    def solve(self, ctx):
        preds = []
        for test_inp in ctx.test_inputs:
            H, W = len(test_inp), len(test_inp[0])
            bg = _bg43(test_inp)

            # Step 1–2: RLE segmentation by dominant row color
            segments = []          # [(color, [row_indices])]
            cur_color, cur_rows = None, []

            for r in range(H):
                dom = self._dominant_non_bg_color(test_inp[r], bg)
                if dom == bg:
                    # bg row → flush current segment
                    if cur_rows:
                        segments.append((cur_color, cur_rows))
                        cur_color, cur_rows = None, []
                else:
                    if dom == cur_color:
                        cur_rows.append(r)
                    else:
                        # New color block starts
                        if cur_rows:
                            segments.append((cur_color, cur_rows))
                        cur_color, cur_rows = dom, [r]

            # Flush trailing segment
            if cur_rows:
                segments.append((cur_color, cur_rows))

            if not segments:
                out = [list(row) for row in test_inp]
                preds.append(Prediction(attempt_1=out, attempt_2=out,
                                         solver_name=self.name, confidence=0.0))
                continue

            # Step 3–4: flatten original and reversed order
            nb_rows       = [r for _, rows in segments for r in rows]
            reversed_rows = [r for _, rows in reversed(segments) for r in rows]

            # Step 5: apply block-reversal
            out = [list(row) for row in test_inp]
            for nb_pos, src_r in zip(nb_rows, reversed_rows):
                out[nb_pos] = list(test_inp[src_r])

            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                     solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class GridBlockFillSolver(BaseSolver):
    """Fills empty blocks in a regular grid-of-blocks layout.
    Each non-empty block has a single-cell hole at internal position (gr, gc)
    where (gr, gc) is the block's position in the block grid.
    Algorithm:
      1. Find nz_rows/nz_cols; group into content row/col runs.
      2. block_h/block_w = mode of run lengths; period = block_size + 1.
      3. Generate ALL block start positions (extend pattern from first seen block).
      4. fill_color = most common non-bg color.
      5. For each all-bg block at grid position (gr, gc): fill all cells with
         fill_color EXCEPT the hole at internal offset (gr, gc).
    Target: 9ddd00f0
    """

    @property
    def name(self):
        return "GridBlockFillSolver"

    def solve(self, ctx):
        preds = []
        for test_inp in ctx.test_inputs:
            H, W = len(test_inp), len(test_inp[0])
            bg = _bg43(test_inp)

            # Find non-bg rows/cols
            nz_rows = [r for r in range(H)
                       if any(test_inp[r][c] != bg for c in range(W))]
            nz_cols = [c for c in range(W)
                       if any(test_inp[r][c] != bg for r in range(H))]

            if not nz_rows or not nz_cols:
                out = [list(row) for row in test_inp]
                preds.append(Prediction(attempt_1=out, attempt_2=out,
                                         solver_name=self.name, confidence=0.0))
                continue

            row_groups = _group_runs(nz_rows)
            col_groups = _group_runs(nz_cols)

            block_h = Counter(e - s + 1 for s, e in row_groups).most_common(1)[0][0]
            block_w = Counter(e - s + 1 for s, e in col_groups).most_common(1)[0][0]
            period_h = block_h + 1
            period_w = block_w + 1

            # Generate all block row starts from the first observed group
            first_r = row_groups[0][0]
            all_row_starts = []
            r = first_r
            while r < H:
                all_row_starts.append(r)
                r += period_h
            r = first_r - period_h
            while r >= 0:
                all_row_starts.insert(0, r)
                r -= period_h

            first_c = col_groups[0][0]
            all_col_starts = []
            c = first_c
            while c < W:
                all_col_starts.append(c)
                c += period_w
            c = first_c - period_w
            while c >= 0:
                all_col_starts.insert(0, c)
                c -= period_w

            # Build valid block ranges
            block_row_ranges = [
                (rs, min(rs + block_h - 1, H - 1))
                for rs in all_row_starts if 0 <= rs < H
            ]
            block_col_ranges = [
                (cs, min(cs + block_w - 1, W - 1))
                for cs in all_col_starts if 0 <= cs < W
            ]

            # fill_color = most common non-bg color
            nb_vals = [test_inp[r][c] for r in range(H) for c in range(W)
                       if test_inp[r][c] != bg]
            if not nb_vals:
                out = [list(row) for row in test_inp]
                preds.append(Prediction(attempt_1=out, attempt_2=out,
                                         solver_name=self.name, confidence=0.0))
                continue
            fill_color = Counter(nb_vals).most_common(1)[0][0]

            out = [list(row) for row in test_inp]

            for gr, (br_s, br_e) in enumerate(block_row_ranges):
                for gc, (bc_s, bc_e) in enumerate(block_col_ranges):
                    # Only fill all-bg blocks
                    if any(test_inp[r][c] != bg
                           for r in range(br_s, br_e + 1)
                           for c in range(bc_s, bc_e + 1)):
                        continue
                    # Hole position: internal offset (gr, gc) within block
                    hole_r = br_s + gr
                    hole_c = bc_s + gc
                    for r in range(br_s, br_e + 1):
                        for c in range(bc_s, bc_e + 1):
                            if r == hole_r and c == hole_c:
                                out[r][c] = bg    # keep hole
                            else:
                                out[r][c] = fill_color

            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                     solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class ColorToColumnSolver(BaseSolver):
    """Learns a color→column mapping from training pairs, then applies it.
    Training: for each input row with a non-bg value v at col 0, find which
    output column c contains value v → learn map {v: c}.
    Test: for each non-bg col-0 value v, place v at (r, color_col[v]) in output.
    Target: f45f5ca7
    """

    @property
    def name(self):
        return "ColorToColumnSolver"

    def solve(self, ctx):
        # ── Learn color→column from training ──────────────────────────────
        color_col = {}
        for pair in ctx.train_pairs:
            inp, out = pair['input'], pair['output']
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

        preds = []
        for test_inp in ctx.test_inputs:
            H, W = len(test_inp), len(test_inp[0])
            bg = _bg43(test_inp)
            out = [[bg] * W for _ in range(H)]
            for r in range(H):
                v = test_inp[r][0]
                if v != bg:
                    if v in color_col and 0 <= color_col[v] < W:
                        out[r][color_col[v]] = v
                    elif 0 <= v < W:
                        # fallback: old identity mapping
                        out[r][v] = v
            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                     solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class FillTowardCenterSolver(BaseSolver):
    """Fills the gap between two colored objects symmetrically toward center.
    mid = (gap_start + gap_end) // 2.
    Odd gap: A fills [gap_s, mid-1], mid stays bg, B fills [mid+1, gap_e].
    Even gap: A fills [gap_s, mid], B fills [mid+1, gap_e].
    Handles both horizontal (h) and vertical (v) gap orientations.
    Target: d968ffd4
    """

    @property
    def name(self):
        return "FillTowardCenterSolver"

    @staticmethod
    def _apply_fill(out, gap_s, gap_e, col_a, col_b, H, W, axis):
        gap_len = gap_e - gap_s + 1
        if gap_len <= 0:
            return
        mid = (gap_s + gap_e) // 2
        for idx in range(gap_s, gap_e + 1):
            if gap_len % 2 == 1:   # odd: divider at mid stays bg
                if idx < mid:
                    fill = col_a
                elif idx > mid:
                    fill = col_b
                else:
                    continue        # mid stays bg
            else:                   # even: A claims mid, B claims mid+1
                fill = col_a if idx <= mid else col_b
            if axis == 'h':
                for r in range(H):
                    out[r][idx] = fill
            else:
                for c in range(W):
                    out[idx][c] = fill

    def solve(self, ctx):
        preds = []
        for test_inp in ctx.test_inputs:
            H, W = len(test_inp), len(test_inp[0])
            bg = _bg43(test_inp)
            ccs = _bfs_ccs43(test_inp, bg)
            out = [list(row) for row in test_inp]

            if len(ccs) < 2:
                preds.append(Prediction(attempt_1=out, attempt_2=out,
                                         solver_name=self.name, confidence=0.0))
                continue

            # Use two largest CCs
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

            if max_col_a < min_col_b:                              # A left, B right
                self._apply_fill(out, max_col_a + 1, min_col_b - 1,
                                 col_a, col_b, H, W, 'h')
            elif max_col_b < min_col_a:                            # B left, A right
                self._apply_fill(out, max_col_b + 1, min_col_a - 1,
                                 col_b, col_a, H, W, 'h')
            elif max_row_a < min_row_b:                            # A top, B bottom
                self._apply_fill(out, max_row_a + 1, min_row_b - 1,
                                 col_a, col_b, H, W, 'v')
            elif max_row_b < min_row_a:                            # B top, A bottom
                self._apply_fill(out, max_row_b + 1, min_row_a - 1,
                                 col_b, col_a, H, W, 'v')

            preds.append(Prediction(attempt_1=out, attempt_2=out,
                                     solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
