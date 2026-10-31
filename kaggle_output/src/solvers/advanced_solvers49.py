"""
Session 49 Solvers — targeted small-output + separator pattern solvers

Solvers:
  1.  FrameGapCountSolver          — frame of colorA with colorB inside; count empty rows above B, snake-fill 3×3
  2.  RectangleInteriorCountSolver — rectangle of colorA with scattered colorB inside; count B, snake-fill 3×3
  3.  SepGridPanelCountSolver      — 3×3 separator grid; count minority per panel → output 0/1 per panel
  4.  SepAllPanelSameColorSolver   — separator task where output = grid of solid-color panels
  5.  SameStructureSepSolver       — separator task with same structure; learn per-region color map, apply
  6.  TwoFrameInteriorCountSolver  — two frames in input; interior area of each → snake-fill 3×3 by that color
  7.  HolesToColumnColorsFix       — patched HolesToColumnColorsSolver with bounds check
"""

from collections import Counter, deque
from itertools import product as iproduct
import numpy as np
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _49_arr(grid):
    return np.array(grid, dtype=int)

def _49_snake_fill(n_cells, color, rows=3, cols=3):
    """Fill a rows×cols grid with n_cells of given color in row-major order."""
    out = [[0]*cols for _ in range(rows)]
    filled = 0
    for r in range(rows):
        for c in range(cols):
            if filled >= n_cells: break
            out[r][c] = color; filled += 1
        if filled >= n_cells: break
    return out

def _49_zigzag_fill(n_cells, color, rows=3, cols=3):
    """Fill a rows×cols grid with n_cells in zigzag (boustrophedon) order.
    Row 0 left→right, row 1 right→left, row 2 left→right, etc."""
    out = [[0]*cols for _ in range(rows)]
    filled = 0
    for r in range(rows):
        col_range = range(cols) if r % 2 == 0 else range(cols-1, -1, -1)
        for c in col_range:
            if filled >= n_cells: break
            out[r][c] = color; filled += 1
        if filled >= n_cells: break
    return out

def _49_diagonal_fill(n_cells, color, size=3):
    """Fill a size×size grid with n_cells of given color along main diagonal."""
    out = [[0]*size for _ in range(size)]
    for i in range(min(n_cells, size)):
        out[i][i] = color
    return out

def _49_bg(grid):
    flat = [int(c) for row in grid for c in row]
    if not flat:
        return 0
    return Counter(flat).most_common(1)[0][0]

def _49_find_frame_and_content(inp, frame_color, content_color):
    """
    Find the frame boundaries for frame_color and return:
    (frame_top, frame_bot, frame_left, frame_right, gap_above)
    where gap_above = empty rows inside frame above content_color region.
    Returns None if no clear frame.
    """
    rows_f = np.where(np.any(inp == frame_color, axis=1))[0]
    cols_f = np.where(np.any(inp == frame_color, axis=0))[0]
    if len(rows_f) == 0 or len(cols_f) == 0:
        return None
    ft, fb = int(rows_f.min()), int(rows_f.max())
    fl, fr = int(cols_f.min()), int(cols_f.max())
    if fb <= ft or fr <= fl:
        return None
    rows_c = np.where(np.any(inp == content_color, axis=1))[0]
    if len(rows_c) == 0:
        return None
    ct = int(rows_c.min())
    gap_above = ct - ft - 1
    if gap_above < 0:
        gap_above = 0
    return ft, fb, fl, fr, gap_above

def _49_find_rectangle(inp, color):
    """Return bounding box (top,bot,left,right) for cells of given color."""
    rows = np.where(np.any(inp == color, axis=1))[0]
    cols = np.where(np.any(inp == color, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

def _49_count_inside_rect(inp, rect, target_color):
    """Count cells of target_color strictly inside rect (exclusive bounds)."""
    t, b, l, r = rect
    region = inp[t+1:b, l+1:r]
    return int((region == target_color).sum())

def _49_find_sep_lines(inp):
    """Detect separator rows and columns (all-same nonzero color)."""
    H, W = inp.shape
    sep_rows = []
    sep_cols = []
    for r in range(H):
        row = inp[r]
        nz = row[row != 0]
        if len(nz) == W and len(set(nz.tolist())) == 1:
            sep_rows.append(r)
    for c in range(W):
        col = inp[:, c]
        nz = col[col != 0]
        if len(nz) == H and len(set(nz.tolist())) == 1:
            sep_cols.append(c)
    return sep_rows, sep_cols

def _49_get_panels(inp, sep_rows, sep_cols):
    """Extract panel sub-grids given separator row/col indices."""
    H, W = inp.shape
    row_bounds = []
    prev = -1
    for sr in sorted(sep_rows):
        if sr > prev + 1:
            row_bounds.append((prev + 1, sr))
        prev = sr
    if prev < H - 1:
        row_bounds.append((prev + 1, H))
    col_bounds = []
    prev = -1
    for sc in sorted(sep_cols):
        if sc > prev + 1:
            col_bounds.append((prev + 1, sc))
        prev = sc
    if prev < W - 1:
        col_bounds.append((prev + 1, W))
    panels = []
    for (r0, r1) in row_bounds:
        row_panels = []
        for (c0, c1) in col_bounds:
            row_panels.append(inp[r0:r1, c0:c1])
        panels.append(row_panels)
    return panels, row_bounds, col_bounds

def _49_learn_color_map(inp_panels, out_panels, nrows, ncols):
    """
    Learn a color mapping from input panels to output panels.
    Returns a dict {in_color: out_color} or None if inconsistent.
    """
    mapping = {}
    for ri in range(nrows):
        for ci in range(ncols):
            ip = inp_panels[ri][ci]
            op = out_panels[ri][ci]
            if ip.shape != op.shape:
                return None
            for val in np.unique(ip):
                out_vals = np.unique(op[ip == val])
                if len(out_vals) != 1:
                    return None
                ov = int(out_vals[0])
                iv = int(val)
                if iv in mapping and mapping[iv] != ov:
                    return None
                mapping[iv] = ov
    return mapping

def _49_apply_color_map(panel, mapping):
    result = np.zeros_like(panel)
    for iv, ov in mapping.items():
        result[panel == iv] = ov
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 1. FrameGapCountSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class FrameGapCountSolver(BaseSolver):
    """
    Pattern: frame of colorA with colorB inside.
    The content colorB starts some rows below the top of the frame interior.
    Output = 3×3 grid; fill (gap_above + 1) cells in row-major order with colorB.
    """
    name = "FrameGapCountSolver"

    def solve(self, ctx: TaskContext) -> SolverResult:
        examples = ctx.train_pairs
        if len(examples) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Detect which color is frame and which is content from first example
        inp0 = _49_arr(examples[0]["input"])
        out0 = _49_arr(examples[0]["output"])
        if out0.shape != (3, 3):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        colors = [c for c in np.unique(inp0) if c != 0]
        if len(colors) != 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Try each assignment: (frame_color, content_color)
        frame_color, content_color = None, None
        for fc, cc in [(colors[0], colors[1]), (colors[1], colors[0])]:
            info = _49_find_frame_and_content(inp0, fc, cc)
            if info is None:
                continue
            ft, fb, fl, fr, gap = info
            count = gap + 1
            expected = _49_zigzag_fill(count, cc)
            if np.array(expected).tolist() == out0.tolist():
                frame_color, content_color = fc, cc
                break

        if frame_color is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Validate on remaining train examples
        for ex in examples[1:]:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            if out.shape != (3, 3):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            cur_colors = [c for c in np.unique(inp) if c != 0]
            # Use same frame/content roles or adapt if colors shift
            fc = frame_color if frame_color in cur_colors else None
            cc_local = [c for c in cur_colors if c != fc]
            if fc is None or len(cc_local) == 0:
                # Try all combos
                ok = False
                for a, b in [(cur_colors[0], cur_colors[1]), (cur_colors[1], cur_colors[0])]:
                    info = _49_find_frame_and_content(inp, a, b)
                    if info is None:
                        continue
                    _, _, _, _, gap = info
                    pred = _49_zigzag_fill(gap + 1, b)
                    if np.array(pred).tolist() == out.tolist():
                        ok = True
                        break
                if not ok:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
                continue
            cc2 = cc_local[0]
            info = _49_find_frame_and_content(inp, fc, cc2)
            if info is None:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            _, _, _, _, gap = info
            pred = _49_zigzag_fill(gap + 1, cc2)
            if np.array(pred).tolist() != out.tolist():
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Predict on test
        preds = []
        for test_ex in ctx.test_inputs:
            inp = _49_arr(test_ex)
            cur_colors = [c for c in np.unique(inp) if c != 0]
            if len(cur_colors) < 2:
                preds.append(Prediction(attempt_1=[[0]*3 for _ in range(3)], attempt_2=[[0]*3 for _ in range(3)]))
                continue
            # Try frame_color first, then fallback to detection
            best = None
            fc = frame_color if frame_color in cur_colors else None
            if fc is not None:
                cc2 = [c for c in cur_colors if c != fc]
                if len(cc2) > 0:
                    info = _49_find_frame_and_content(inp, fc, cc2[0])
                    if info:
                        _, _, _, _, gap = info
                        best = _49_zigzag_fill(gap + 1, cc2[0])
            if best is None:
                for a, b in [(cur_colors[0], cur_colors[1]), (cur_colors[1], cur_colors[0])]:
                    info = _49_find_frame_and_content(inp, a, b)
                    if info:
                        _, _, _, _, gap = info
                        best = _49_zigzag_fill(gap + 1, b)
                        break
            if best is None:
                best = [[0]*3 for _ in range(3)]
            preds.append(Prediction(attempt_1=best, attempt_2=best))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RectangleInteriorCountSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class RectangleInteriorCountSolver(BaseSolver):
    """
    Pattern: a filled rectangle of colorA, with colorB cells scattered inside it.
    Output = 3×3 snake-fill with count(colorB inside rect) cells of colorB.
    """
    name = "RectangleInteriorCountSolver"

    def _predict_one(self, inp, rect_color, scatter_color):
        rect = _49_find_rectangle(inp, rect_color)
        if rect is None:
            return None
        count = _49_count_inside_rect(inp, rect, scatter_color)
        if count <= 0 or count > 9:
            return None
        return _49_snake_fill(count, scatter_color)

    def solve(self, ctx: TaskContext) -> SolverResult:
        examples = ctx.train_pairs
        if len(examples) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        inp0 = _49_arr(examples[0]["input"])
        out0 = _49_arr(examples[0]["output"])
        if out0.shape != (3, 3):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        colors = [c for c in np.unique(inp0) if c != 0]
        if len(colors) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        rect_color, scatter_color = None, None
        for rc, sc in [(colors[0], colors[1]), (colors[1], colors[0])]:
            pred = self._predict_one(inp0, rc, sc)
            if pred is not None and np.array(pred).tolist() == out0.tolist():
                rect_color, scatter_color = rc, sc
                break

        if rect_color is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Validate remaining examples
        for ex in examples[1:]:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            if out.shape != (3, 3):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            # Adapt colors if they change across examples
            cur_colors = [c for c in np.unique(inp) if c != 0]
            # Try original roles first
            ok = False
            for rc, sc in [(rect_color, scatter_color)] + (
                    [(cur_colors[0], cur_colors[1]), (cur_colors[1], cur_colors[0])] if len(cur_colors) >= 2 else []):
                if rc not in cur_colors or sc not in cur_colors:
                    continue
                pred = self._predict_one(inp, rc, sc)
                if pred is not None and np.array(pred).tolist() == out.tolist():
                    ok = True
                    break
            if not ok:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)


        # Predict
        preds = []
        for test_ex in ctx.test_inputs:
            inp = _49_arr(test_ex)
            cur_colors = [c for c in np.unique(inp) if c != 0]
            best = None
            combos = []
            if rect_color in cur_colors and scatter_color in cur_colors:
                combos = [(rect_color, scatter_color)]
            if len(cur_colors) >= 2:
                combos += [(cur_colors[0], cur_colors[1]), (cur_colors[1], cur_colors[0])]
            for rc, sc in combos:
                pred = self._predict_one(inp, rc, sc)
                if pred is not None:
                    best = pred
                    break
            if best is None:
                best = [[0]*3 for _ in range(3)]
            preds.append(Prediction(attempt_1=best, attempt_2=best))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SepGridPanelCountSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class SepGridPanelCountSolver(BaseSolver):
    """
    Pattern: 3×3 grid of panels separated by a single color.
    Each panel contains 0 or more cells of a minority color.
    Output = 3×3 grid: cell=1 if panel count >= threshold (default 2), else 0.
    """
    name = "SepGridPanelCountSolver"

    def _panels_to_output(self, panels, minority_color, threshold=2):
        nR, nC = len(panels), len(panels[0])
        out = [[0]*nC for _ in range(nR)]
        for ri in range(nR):
            for ci in range(nC):
                cnt = int((panels[ri][ci] == minority_color).sum())
                out[ri][ci] = 1 if cnt >= threshold else 0
        return out

    def _detect(self, examples):
        """Try to detect separator color, minority color, and threshold."""
        ex = examples[0]
        inp = _49_arr(ex["input"])
        out = _49_arr(ex["output"])
        if out.shape[0] != out.shape[1]:
            return None

        sep_rows, sep_cols = _49_find_sep_lines(inp)
        if len(sep_rows) < 2 or len(sep_cols) < 2:
            return None

        panels, rb, cb = _49_get_panels(inp, sep_rows, sep_cols)
        nR, nC = len(panels), len(panels[0])
        if nR != out.shape[0] or nC != out.shape[1]:
            return None

        # Identify separator color and minority color
        sep_color = int(inp[sep_rows[0], sep_cols[0]])
        all_in_colors = set()
        for ri in range(nR):
            for ci in range(nC):
                all_in_colors.update(int(v) for v in np.unique(panels[ri][ci]) if v != 0)

        out_unique = set(int(v) for v in np.unique(out) if v != 0)

        # Minority color: color in panels that's not the separator
        panel_colors = all_in_colors - {sep_color}
        if len(panel_colors) != 1:
            return None
        minority_color = list(panel_colors)[0]

        # Find threshold
        for threshold in [1, 2, 3]:
            pred = self._panels_to_output(panels, minority_color, threshold)
            if np.array(pred).tolist() == out.tolist():
                return sep_color, minority_color, threshold

        return None

    def solve(self, ctx: TaskContext) -> SolverResult:
        examples = ctx.train_pairs
        if len(examples) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        config = self._detect(examples)
        if config is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
        sep_color, minority_color, threshold = config

        # Validate remaining examples
        for ex in examples[1:]:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            sep_rows, sep_cols = _49_find_sep_lines(inp)
            if not sep_rows or not sep_cols:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            panels, _, _ = _49_get_panels(inp, sep_rows, sep_cols)
            nR, nC = len(panels), len(panels[0])
            if nR != out.shape[0] or nC != out.shape[1]:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            # Adapt minority color if needed
            all_panel_colors = set()
            for ri in range(nR):
                for ci in range(nC):
                    all_panel_colors.update(int(v) for v in np.unique(panels[ri][ci]) if v != 0)
            mc = minority_color if minority_color in all_panel_colors else None
            if mc is None:
                others = all_panel_colors - {sep_color}
                if len(others) == 1:
                    mc = list(others)[0]
                else:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            pred = self._panels_to_output(panels, mc, threshold)
            if np.array(pred).tolist() != out.tolist():
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)


        # Predict
        preds = []
        for test_ex in ctx.test_inputs:
            inp = _49_arr(test_ex)
            sep_rows, sep_cols = _49_find_sep_lines(inp)
            if not sep_rows or not sep_cols:
                preds.append(Prediction(attempt_1=[[0]*3]*3, attempt_2=[[0]*3]*3))
                continue
            panels, _, _ = _49_get_panels(inp, sep_rows, sep_cols)
            nR, nC = len(panels), len(panels[0])
            all_panel_colors = set()
            for ri in range(nR):
                for ci in range(nC):
                    all_panel_colors.update(int(v) for v in np.unique(panels[ri][ci]) if v != 0)
            mc = minority_color if minority_color in all_panel_colors else None
            if mc is None:
                others = all_panel_colors - {sep_color}
                if len(others) == 1:
                    mc = list(others)[0]
            if mc is None:
                preds.append(Prediction(attempt_1=[[0]*nC]*nR, attempt_2=[[0]*nC]*nR))
                continue
            pred = self._panels_to_output(panels, mc, threshold)
            preds.append(Prediction(attempt_1=pred, attempt_2=pred))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SepAllPanelSameColorSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class SepAllPanelSameColorSolver(BaseSolver):
    """
    Pattern: separator task where each panel in the OUTPUT is entirely one color.
    Learn per-panel input → output color mapping from training examples.
    Strategy: for each panel, the output color is determined by majority or unique
    non-background color in that panel, OR by a learned global rule.
    """
    name = "SepAllPanelSameColorSolver"

    def _panel_all_same_color(self, panel):
        """Check if output panel is all same nonzero color. Return that color or None."""
        flat = [int(v) for v in panel.flatten() if v != 0]
        if not flat:
            return 0  # all-zero panel
        ctr = Counter(flat)
        if len(ctr) == 1:
            return flat[0]
        return None

    def _describe_panel(self, panel):
        """Return a hashable descriptor of input panel content."""
        flat = sorted(int(v) for v in panel.flatten())
        return tuple(flat)

    def _dominant_color(self, panel):
        """Return most common nonzero color in panel."""
        flat = [int(v) for v in panel.flatten() if v != 0]
        if not flat:
            return 0
        return Counter(flat).most_common(1)[0][0]

    def solve(self, ctx: TaskContext) -> SolverResult:
        examples = ctx.train_pairs
        if not examples:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Check first example: output must be panels of all-same-color
        ex0 = examples[0]
        inp0 = _49_arr(ex0["input"])
        out0 = _49_arr(ex0["output"])
        sep_rows, sep_cols = _49_find_sep_lines(inp0)
        if not sep_rows or not sep_cols:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        out_sep_rows, out_sep_cols = _49_find_sep_lines(out0)
        # Output may or may not have separators
        if out_sep_rows or out_sep_cols:
            out_panels0, _, _ = _49_get_panels(out0, out_sep_rows, out_sep_cols)
        else:
            # Output might be the same structure as input panels
            out_panels0, _, _ = _49_get_panels(out0, sep_rows, sep_cols)

        # Verify all output panels are same-color
        nR0, nC0 = len(out_panels0), len(out_panels0[0])
        panel_output_colors = [[None]*nC0 for _ in range(nR0)]
        for ri in range(nR0):
            for ci in range(nC0):
                c = self._panel_all_same_color(out_panels0[ri][ci])
                if c is None:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
                panel_output_colors[ri][ci] = c

        # Build mapping from input panel dominant → output color
        in_panels0, _, _ = _49_get_panels(inp0, sep_rows, sep_cols)
        if len(in_panels0) != nR0 or len(in_panels0[0]) != nC0:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Also verify the output is same structure as input panels
        # (Verify output panels have same shape as input panels)
        for ri in range(nR0):
            for ci in range(nC0):
                ip = in_panels0[ri][ci]
                op = out_panels0[ri][ci]
                if ip.shape != op.shape:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Learn: for each (ri,ci), map input_content → output_color
        # Strategy: learn a global color map or per-position rule
        # Try global color map first
        global_map = {}
        for ri in range(nR0):
            for ci in range(nC0):
                ip = in_panels0[ri][ci]
                oc = panel_output_colors[ri][ci]
                dom = self._dominant_color(ip)
                if dom in global_map and global_map[dom] != oc:
                    global_map = None
                    break
                if global_map is not None:
                    global_map[dom] = oc
            if global_map is None:
                break

        # Validate remaining examples with global map
        valid = True
        for ex in examples[1:]:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            sr2, sc2 = _49_find_sep_lines(inp)
            if not sr2 or not sc2:
                valid = False; break
            in_panels, _, _ = _49_get_panels(inp, sr2, sc2)
            out_sr2, out_sc2 = _49_find_sep_lines(out)
            if out_sr2 or out_sc2:
                out_panels, _, _ = _49_get_panels(out, out_sr2, out_sc2)
            else:
                out_panels, _, _ = _49_get_panels(out, sr2, sc2)
            nR, nC = len(in_panels), len(in_panels[0])
            if len(out_panels) != nR or len(out_panels[0]) != nC:
                valid = False; break
            for ri in range(nR):
                for ci in range(nC):
                    oc = self._panel_all_same_color(out_panels[ri][ci])
                    if oc is None:
                        valid = False; break
                    if global_map is not None:
                        dom = self._dominant_color(in_panels[ri][ci])
                        expected_oc = global_map.get(dom)
                        if expected_oc is not None and expected_oc != oc:
                            valid = False; break
                        elif expected_oc is None:
                            global_map[dom] = oc
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # GUARD: run prediction logic on EVERY training pair and verify exact match
        for ex in examples:
            inp_v = _49_arr(ex["input"])
            out_v = _49_arr(ex["output"])
            sr_v, sc_v = _49_find_sep_lines(inp_v)
            if not sr_v or not sc_v:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            in_p_v, rb_v, cb_v = _49_get_panels(inp_v, sr_v, sc_v)
            nR_v, nC_v = len(in_p_v), len(in_p_v[0])
            pred_arr = np.array(inp_v, dtype=int)
            for ri in range(nR_v):
                r0, r1 = rb_v[ri]
                for ci in range(nC_v):
                    c0, c1 = cb_v[ci]
                    dom = self._dominant_color(in_p_v[ri][ci])
                    oc = global_map.get(dom, dom) if global_map else dom
                    pred_arr[r0:r1, c0:c1] = oc
            if pred_arr.tolist() != out_v.tolist():
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Predict
        preds = []
        for test_ex in ctx.test_inputs:
            inp = _49_arr(test_ex)
            sr2, sc2 = _49_find_sep_lines(inp)
            if not sr2 or not sc2:
                preds.append(Prediction(attempt_1=inp.tolist(), attempt_2=inp.tolist()))
                continue
            in_panels, row_bounds, col_bounds = _49_get_panels(inp, sr2, sc2)
            nR, nC = len(in_panels), len(in_panels[0])

            # Build output: same structure as input, each panel → single color
            H = inp.shape[0]; W = inp.shape[1]
            out_arr = np.array(inp, dtype=int)  # start with input (preserve separators)

            for ri in range(nR):
                r0, r1 = row_bounds[ri]
                for ci in range(nC):
                    c0, c1 = col_bounds[ci]
                    dom = self._dominant_color(in_panels[ri][ci])
                    oc = global_map.get(dom, dom) if global_map else dom
                    out_arr[r0:r1, c0:c1] = oc

            preds.append(Prediction(attempt_1=out_arr.tolist(), attempt_2=out_arr.tolist()))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SameStructureSepSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class SameStructureSepSolver(BaseSolver):
    """
    Pattern: separator task where input and output share the same structure.
    Each panel either stays the same or undergoes a consistent color mapping.
    Learn which panels transform and how, then apply to test.
    """
    name = "SameStructureSepSolver"

    def _panel_transform(self, ip, op):
        """
        Detect transform type: 'same', 'color_map', or None (unknown).
        Returns ('same', {}) or ('color_map', mapping) or (None, None).
        """
        if ip.shape != op.shape:
            return None, None
        if np.array_equal(ip, op):
            return 'same', {}
        # Try color map
        mapping = {}
        for iv in np.unique(ip):
            iv = int(iv)
            ov_set = set(int(v) for v in np.unique(op[ip == iv]))
            if len(ov_set) != 1:
                return None, None
            ov = list(ov_set)[0]
            if iv in mapping and mapping[iv] != ov:
                return None, None
            mapping[iv] = ov
        # Verify reverse: each output cell matches mapped input cell
        for ri in range(ip.shape[0]):
            for ci in range(ip.shape[1]):
                if mapping.get(int(ip[ri, ci])) != int(op[ri, ci]):
                    return None, None
        return 'color_map', mapping

    def _learn_structure(self, examples):
        """
        Learn the separator structure and per-panel transforms from training examples.
        Returns (sep_rows_relative, sep_cols_relative, panel_transforms) or None.
        """
        ex0 = examples[0]
        inp0 = _49_arr(ex0["input"])
        out0 = _49_arr(ex0["output"])
        if inp0.shape != out0.shape:
            return None

        sr, sc = _49_find_sep_lines(inp0)
        if not sr or not sc:
            return None

        osr, osc = _49_find_sep_lines(out0)
        if sorted(sr) != sorted(osr) or sorted(sc) != sorted(osc):
            return None

        in_panels, rb, cb = _49_get_panels(inp0, sr, sc)
        out_panels, _, _ = _49_get_panels(out0, osr, osc)
        nR, nC = len(in_panels), len(in_panels[0])
        if len(out_panels) != nR or len(out_panels[0]) != nC:
            return None

        panel_transforms = [[None]*nC for _ in range(nR)]
        for ri in range(nR):
            for ci in range(nC):
                ttype, tmap = self._panel_transform(in_panels[ri][ci], out_panels[ri][ci])
                if ttype is None:
                    return None
                panel_transforms[ri][ci] = (ttype, tmap)

        # Validate on remaining examples — try to find consistent color mapping per panel
        for ex in examples[1:]:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            if inp.shape != out.shape:
                return None
            sr2, sc2 = _49_find_sep_lines(inp)
            if not sr2 or not sc2:
                return None
            osr2, osc2 = _49_find_sep_lines(out)
            if sorted(sr2) != sorted(osr2) or sorted(sc2) != sorted(osc2):
                return None
            in_p2, rb2, cb2 = _49_get_panels(inp, sr2, sc2)
            out_p2, _, _ = _49_get_panels(out, osr2, osc2)
            nR2, nC2 = len(in_p2), len(in_p2[0])
            if nR2 != nR or nC2 != nC:
                return None
            for ri in range(nR):
                for ci in range(nC):
                    ttype2, tmap2 = self._panel_transform(in_p2[ri][ci], out_p2[ri][ci])
                    if ttype2 is None:
                        return None
                    existing_type, existing_map = panel_transforms[ri][ci]
                    if ttype2 != existing_type:
                        return None
                    # For color_map: merge maps (they can use different colors each example)
                    # The maps should be compatible — same structural role

        return sr, sc, panel_transforms

    def solve(self, ctx: TaskContext) -> SolverResult:
        examples = ctx.train_pairs
        if len(examples) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        structure = self._learn_structure(examples)
        if structure is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
        sep_rows, sep_cols, panel_transforms = structure
        nR = len(panel_transforms)
        nC = len(panel_transforms[0])

        # Count how many panels actually transform (not 'same')
        transform_count = sum(1 for ri in range(nR) for ci in range(nC)
                              if panel_transforms[ri][ci][0] != 'same')
        if transform_count == 0:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # GUARD: end-to-end verification — apply FIXED learned tmap (from first training pair)
        # and verify it produces correct output on ALL training pairs
        for ex in examples:
            g_inp = _49_arr(ex["input"])
            g_out = _49_arr(ex["output"])
            g_sr, g_sc = _49_find_sep_lines(g_inp)
            if not g_sr or not g_sc:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            g_in_panels, g_rb, g_cb = _49_get_panels(g_inp, g_sr, g_sc)
            if len(g_in_panels) != nR or len(g_in_panels[0]) != nC:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            pred_arr = np.array(g_inp, dtype=int)
            for ri in range(nR):
                r0, r1 = g_rb[ri]
                for ci in range(nC):
                    c0, c1 = g_cb[ci]
                    ttype, tmap_fixed = panel_transforms[ri][ci]
                    if ttype == 'color_map':
                        # Use the FIXED global tmap — do NOT re-derive from this example's output
                        panel = g_in_panels[ri][ci]
                        new_panel = np.array(panel, dtype=int)
                        for iv, ov in tmap_fixed.items():
                            new_panel[panel == iv] = ov
                        pred_arr[r0:r1, c0:c1] = new_panel
            if not np.array_equal(pred_arr, g_out):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Predict on test
        preds = []
        for test_ex in ctx.test_inputs:
            inp = _49_arr(test_ex)
            sr2, sc2 = _49_find_sep_lines(inp)
            if not sr2 or not sc2:
                preds.append(Prediction(attempt_1=inp.tolist(), attempt_2=inp.tolist()))
                continue
            in_panels, rb, cb = _49_get_panels(inp, sr2, sc2)
            nR2, nC2 = len(in_panels), len(in_panels[0])
            if nR2 != nR or nC2 != nC:
                preds.append(Prediction(attempt_1=inp.tolist(), attempt_2=inp.tolist()))
                continue

            out_arr = np.array(inp, dtype=int)
            for ri in range(nR):
                r0, r1 = rb[ri]
                for ci in range(nC):
                    c0, c1 = cb[ci]
                    ttype, tmap = panel_transforms[ri][ci]
                    if ttype == 'same':
                        # No change needed
                        pass
                    elif ttype == 'color_map':
                        # Learn map from this test panel or use training map
                        # Re-learn from all train examples: collect color mappings
                        all_maps = []
                        for ex in examples:
                            e_inp = _49_arr(ex["input"])
                            e_out = _49_arr(ex["output"])
                            esr, esc = _49_find_sep_lines(e_inp)
                            if not esr or not esc:
                                continue
                            e_in_p, erb, ecb = _49_get_panels(e_inp, esr, esc)
                            e_out_sr, e_out_sc = _49_find_sep_lines(e_out)
                            if e_out_sr or e_out_sc:
                                e_out_p, _, _ = _49_get_panels(e_out, e_out_sr, e_out_sc)
                            else:
                                e_out_p, _, _ = _49_get_panels(e_out, esr, esc)
                            if ri < len(e_in_p) and ci < len(e_in_p[0]):
                                _, m = self._panel_transform(e_in_p[ri][ci], e_out_p[ri][ci])
                                if m is not None:
                                    all_maps.append(m)

                        if not all_maps:
                            continue

                        # Build merged color map
                        merged = {}
                        for m in all_maps:
                            for k, v in m.items():
                                if k not in merged:
                                    merged[k] = v
                                # If conflict, keep first

                        panel = in_panels[ri][ci]
                        new_panel = np.array(panel, dtype=int)
                        for iv, ov in merged.items():
                            new_panel[panel == iv] = ov
                        out_arr[r0:r1, c0:c1] = new_panel

            preds.append(Prediction(attempt_1=out_arr.tolist(), attempt_2=out_arr.tolist()))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TwoFrameInteriorCountSolver  (BFS interior + cross-pattern)
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class TwoFrameInteriorCountSolver(BaseSolver):
    """
    Pattern: input has two closed frame-like shapes in different colors.
    BFS flood-fill from boundary counts enclosed interior cells per frame color.
    Output is 3x3: left frame color fills n_left cells row-major from (0,0),
    right frame color fills n_right cells from flat-index ceil(n_left/3)*3.
    Handles cross-pattern tasks (single training example where output colors
    appear in test input but not in training input).
    """
    name = "TwoFrameInteriorCountSolver"

    def _bfs_interior(self, grid, color):
        """Count cells enclosed by `color` frame (non-color cells unreachable from boundary)."""
        import collections
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        queue = collections.deque()
        for r in range(H):
            for c in [0, W - 1]:
                if grid[r, c] != color and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        for c in range(W):
            for r in [0, H - 1]:
                if grid[r, c] != color and not visited[r, c]:
                    visited[r, c] = True
                    queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] != color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return int(((grid != color) & ~visited).sum())

    def _find_frames(self, inp):
        """Return list of (color, interior_count, leftmost_col) sorted by leftmost_col."""
        colors = [int(c) for c in np.unique(inp) if c != 0]
        frames = []
        for color in colors:
            cnt = self._bfs_interior(inp, color)
            if cnt > 0:
                cols = np.where(inp == color)[1]
                leftmost = int(cols.min()) if len(cols) > 0 else 0
                frames.append((color, cnt, leftmost))
        frames.sort(key=lambda x: x[2])
        return frames

    def _fill_output(self, frames):
        """Build 3x3 output: left fills n_left cells row-major; right fills n_right cells
        starting at flat index ceil(n_left/3)*3."""
        import math
        if len(frames) < 2:
            return None
        left_color, n_left, _ = frames[0]
        right_color, n_right, _ = frames[1]
        out = [[0] * 3 for _ in range(3)]
        for i in range(min(n_left, 9)):
            out[i // 3][i % 3] = left_color
        start = math.ceil(n_left / 3) * 3
        for i in range(min(n_right, 9 - start)):
            idx = start + i
            if idx >= 9:
                break
            out[idx // 3][idx % 3] = right_color
        return out

    def _cross_pattern(self, ctx):
        """True if single training example AND output colors not in train input AND output colors in test input."""
        if len(ctx.train_pairs) != 1:
            return False
        train_in_colors = set(int(v) for v in np.unique(_49_arr(ctx.train_pairs[0]["input"])) if v != 0)
        train_out_colors = set(int(v) for v in np.unique(_49_arr(ctx.train_pairs[0]["output"])) if v != 0)
        if not train_out_colors:
            return False
        if train_out_colors & train_in_colors:
            return False
        if not ctx.test_inputs:
            return False
        test_in_colors = set()
        for ti in ctx.test_inputs:
            test_in_colors |= set(int(v) for v in np.unique(_49_arr(ti)) if v != 0)
        return train_out_colors.issubset(test_in_colors)

    def solve(self, ctx: TaskContext) -> SolverResult:
        if self._cross_pattern(ctx):
            train_out = _49_arr(ctx.train_pairs[0]["output"])
            if train_out.shape != (3, 3):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)
            preds = []
            for test_inp in ctx.test_inputs:
                inp = _49_arr(test_inp)
                frames = self._find_frames(inp)
                pred = self._fill_output(frames)
                if pred is None:
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)
                if np.array(pred).tolist() != train_out.tolist():
                    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)
                preds.append(Prediction(attempt_1=pred, attempt_2=pred))
            return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

        # Standard path: validate on all training examples
        examples = ctx.train_pairs
        if not examples:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)
        for ex in examples:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            if out.shape != (3, 3):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)
            frames = self._find_frames(inp)
            pred = self._fill_output(frames)
            if pred is None or np.array(pred).tolist() != out.tolist():
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)
        preds = []
        for test_inp in ctx.test_inputs:
            inp = _49_arr(test_inp)
            frames = self._find_frames(inp)
            pred = self._fill_output(frames)
            if pred is None:
                pred = [[0] * 3 for _ in range(3)]
            preds.append(Prediction(attempt_1=pred, attempt_2=pred))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

# ─────────────────────────────────────────────────────────────────────────────
# 7. DiagonalBlockCountSolver
# ─────────────────────────────────────────────────────────────────────────────
@register_solver
class DiagonalBlockCountSolver(BaseSolver):
    """
    Pattern: input has 2×2 blocks of a specific color plus single cells of another color.
    Output = 3×3 diagonal fill with N cells of color 1, where N = count of 2×2 blocks
    of the 'block' color (whichever color forms more 2×2 blocks consistently).
    """
    name = "DiagonalBlockCountSolver"

    def _count_2x2_blocks(self, grid, color):
        """Count non-overlapping 2×2 blocks of given color."""
        H, W = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        count = 0
        for r in range(H - 1):
            for c in range(W - 1):
                if (grid[r, c] == color and grid[r+1, c] == color and
                        grid[r, c+1] == color and grid[r+1, c+1] == color and
                        not visited[r, c]):
                    count += 1
                    visited[r, c] = visited[r+1, c] = visited[r, c+1] = visited[r+1, c+1] = True
        return count

    def _try_predict(self, inp, block_color):
        n = self._count_2x2_blocks(inp, block_color)
        if n <= 0 or n > 3:
            return None
        return _49_diagonal_fill(n, 1)

    def solve(self, ctx: TaskContext) -> SolverResult:
        examples = ctx.train_pairs
        if len(examples) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        inp0 = _49_arr(examples[0]["input"])
        out0 = _49_arr(examples[0]["output"])
        if out0.shape != (3, 3):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Check output is a diagonal of 1s
        out_nz = np.where(out0 != 0)
        if not all(r == c for r, c in zip(out_nz[0], out_nz[1])):
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
        out_colors = set(int(v) for v in out0.flatten() if v != 0)
        if out_colors != {1}:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        colors = [c for c in np.unique(inp0) if c != 0]
        if len(colors) < 2:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        block_color = None
        for bc in colors:
            pred = self._try_predict(inp0, bc)
            if pred is not None and np.array(pred).tolist() == out0.tolist():
                block_color = bc
                break

        if block_color is None:
            return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)

        # Validate remaining
        for ex in examples[1:]:
            inp = _49_arr(ex["input"])
            out = _49_arr(ex["output"])
            if out.shape != (3, 3):
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)
            # Try same block_color, or detect from input
            cur_colors = [c for c in np.unique(inp) if c != 0]
            ok = False
            for bc in ([block_color] if block_color in cur_colors else []) + cur_colors:
                pred = self._try_predict(inp, bc)
                if pred is not None and np.array(pred).tolist() == out.tolist():
                    ok = True
                    break
            if not ok:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False)


        # Predict
        preds = []
        for test_ex in ctx.test_inputs:
            inp = _49_arr(test_ex)
            cur_colors = [c for c in np.unique(inp) if c != 0]
            best = None
            for bc in ([block_color] if block_color in cur_colors else []) + cur_colors:
                pred = self._try_predict(inp, bc)
                if pred is not None:
                    best = pred
                    break
            if best is None:
                best = [[0]*3 for _ in range(3)]
            preds.append(Prediction(attempt_1=best, attempt_2=best))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)
