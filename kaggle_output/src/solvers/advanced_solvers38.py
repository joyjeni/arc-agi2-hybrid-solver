"""
Session 41 solvers — 10 confirmed rules.
Each solver verifies against train pairs before committing to test output.
"""
import copy
from collections import Counter, defaultdict
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver


def _bg(grid):
    flat = [v for row in grid for v in row]
    return Counter(flat).most_common(1)[0][0]


def _bfs_ccs(grid, color):
    H, W = len(grid), len(grid[0])
    visited = [[False]*W for _ in range(H)]
    ccs = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == color and not visited[r][c]:
                queue = [(r, c)]
                visited[r][c] = True
                cells = [(r, c)]
                while queue:
                    cr, cc = queue.pop()
                    for nr, nc in ((cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)):
                        if 0<=nr<H and 0<=nc<W and grid[nr][nc]==color and not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                            cells.append((nr, nc))
                ccs.append(cells)
    return ccs


def _bbox(cells):
    rows = [r for r,c in cells]
    cols = [c for r,c in cells]
    return min(rows), max(rows), min(cols), max(cols)


# ─────────────────────────────────────────────────────────
# 1. BBoxFillSolver  (56ff96f3)
# ─────────────────────────────────────────────────────────
@register_solver
class BBoxFillSolver(BaseSolver):
    @property
    def name(self): return "BBoxFillSolver"

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        bg = _bg(inp)
        out = [row[:] for row in inp]
        colors = set(v for row in inp for v in row if v != bg)
        for c in colors:
            rows = [r for r in range(H) for cc in range(W) if inp[r][cc] == c]
            cols = [cc for r in range(H) for cc in range(W) if inp[r][cc] == c]
            if not rows: continue
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            for r in range(r0, r1+1):
                for cc in range(c0, c1+1):
                    out[r][cc] = c
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._apply(pair['input']) != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 2. TriangularCCSortSolver  (2753e76c)
# ─────────────────────────────────────────────────────────
@register_solver
class TriangularCCSortSolver(BaseSolver):
    @property
    def name(self): return "TriangularCCSortSolver"

    def _apply(self, inp):
        bg = _bg(inp)
        colors = list(set(v for row in inp for v in row if v != bg))
        cc_counts = {c: len(_bfs_ccs(inp, c)) for c in colors}
        sorted_colors = sorted(colors, key=lambda c: -cc_counts[c])
        H = len(sorted_colors)
        W = max(cc_counts.values())
        out = [[bg]*W for _ in range(H)]
        for i, c in enumerate(sorted_colors):
            n = cc_counts[c]
            for j in range(n):
                out[i][W - n + j] = c
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._apply(pair['input']) != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 3. ConnectDotsHVSolver  (ded97339)
# ─────────────────────────────────────────────────────────
@register_solver
class ConnectDotsHVSolver(BaseSolver):
    @property
    def name(self): return "ConnectDotsHVSolver"

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        bg = _bg(inp)
        out = [row[:] for row in inp]
        color_pos = defaultdict(list)
        for r in range(H):
            for c in range(W):
                if inp[r][c] != bg:
                    color_pos[inp[r][c]].append((r, c))
        for color, positions in color_pos.items():
            row_groups = defaultdict(list)
            col_groups = defaultdict(list)
            for r, c in positions:
                row_groups[r].append(c)
                col_groups[c].append(r)
            for r, cs in row_groups.items():
                if len(cs) >= 2:
                    for c in range(min(cs), max(cs)+1):
                        out[r][c] = color
            for c, rs in col_groups.items():
                if len(rs) >= 2:
                    for r in range(min(rs), max(rs)+1):
                        out[r][c] = color
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._apply(pair['input']) != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 4. CountDotsInRect3x3Solver  (c8b7cc0f)
# ─────────────────────────────────────────────────────────
@register_solver
class CountDotsInRect3x3Solver(BaseSolver):
    @property
    def name(self): return "CountDotsInRect3x3Solver"

    def _find_rect_colors(self, inp):
        H, W = len(inp), len(inp[0])
        bg = _bg(inp)
        colors = [c for c in set(v for row in inp for v in row) if c != bg]
        for rect_c in colors:
            for mark_c in colors:
                if rect_c == mark_c: continue
                rows = [r for r in range(H) for c in range(W) if inp[r][c] == rect_c]
                cols = [c for r in range(H) for c in range(W) if inp[r][c] == rect_c]
                if not rows: continue
                r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
                if r1-r0 < 2 or c1-c0 < 2: continue
                # Check forms a proper border
                border = (all(inp[r0][c] == rect_c for c in range(c0, c1+1)) and
                          all(inp[r1][c] == rect_c for c in range(c0, c1+1)) and
                          all(inp[r][c0] == rect_c for r in range(r0, r1+1)) and
                          all(inp[r][c1] == rect_c for r in range(r0, r1+1)))
                if border:
                    return rect_c, mark_c, r0, r1, c0, c1
        return None, None, None, None, None, None

    def _apply(self, inp):
        rect_c, mark_c, r0, r1, c0, c1 = self._find_rect_colors(inp)
        if rect_c is None: return None
        N = sum(1 for r in range(r0+1, r1) for c in range(c0+1, c1) if inp[r][c] == mark_c)
        out = [[0]*3 for _ in range(3)]
        for i in range(min(N, 9)):
            out[i//3][i%3] = mark_c
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out is None or out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            if out:
                preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 5. QuadrantInnerFillSolver  (77fdfe62)
# FIX: iterate ALL colors (not just non-bg) to find divider.
# The divider color (e.g. 1) may be the most common color in the grid,
# so _bg() would return it and exclude it from the search incorrectly.
# ─────────────────────────────────────────────────────────
@register_solver
class QuadrantInnerFillSolver(BaseSolver):
    @property
    def name(self): return "QuadrantInnerFillSolver"

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        # Search ALL distinct colors for the divider (don't exclude bg)
        all_colors = list(set(v for row in inp for v in row))
        for div_c in all_colors:
            row_counts = [sum(1 for c in range(W) if inp[r][c] == div_c) for r in range(H)]
            col_counts = [sum(1 for r in range(H) if inp[r][c] == div_c) for c in range(W)]
            full_rows = [r for r, cnt in enumerate(row_counts) if cnt == W]
            full_cols = [c for c, cnt in enumerate(col_counts) if cnt == H]
            if len(full_rows) >= 2 and len(full_cols) >= 2:
                r0, r1 = sorted(full_rows)[0], sorted(full_rows)[-1]
                c0, c1 = sorted(full_cols)[0], sorted(full_cols)[-1]
                ir0, ir1 = r0+1, r1-1
                ic0, ic1 = c0+1, c1-1
                if ir0 > ir1 or ic0 > ic1: continue
                inner_H = ir1 - ir0 + 1
                inner_W = ic1 - ic0 + 1
                # Corner colors: single non-div cell in each corner quadrant
                def corner_color(rslice, cslice):
                    vals = [inp[r][c] for r in rslice for c in cslice if inp[r][c] != div_c]
                    # Return first non-zero non-div, else first non-div
                    nz = [v for v in vals if v != 0]
                    return nz[0] if nz else (vals[0] if vals else 0)
                tl = corner_color(range(0, r0), range(0, c0))
                tr = corner_color(range(0, r0), range(c1+1, W))
                bl = corner_color(range(r1+1, H), range(0, c0))
                br = corner_color(range(r1+1, H), range(c1+1, W))
                # Marker color: non-div non-0 inside inner area
                marker_vals = set(inp[r][c] for r in range(ir0, ir1+1) for c in range(ic0, ic1+1)
                                  if inp[r][c] != div_c and inp[r][c] != 0)
                if not marker_vals: continue
                marker_c = list(marker_vals)[0]
                mid_r = inner_H / 2
                mid_c = inner_W / 2
                out = [[0]*inner_W for _ in range(inner_H)]
                for r in range(inner_H):
                    for c in range(inner_W):
                        if inp[ir0+r][ic0+c] == marker_c:
                            if r < mid_r and c < mid_c:     out[r][c] = tl
                            elif r < mid_r and c >= mid_c:  out[r][c] = tr
                            elif r >= mid_r and c < mid_c:  out[r][c] = bl
                            else:                            out[r][c] = br
                return out
        return None

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out is None or out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            if out:
                preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 6. BlobQuadrant2x2Solver  (19bb5feb)
# FIX: use 0 as background always.
# The large rectangle fill (e.g. color 8) is more common than the outer bg (0),
# so _bg() would return 8 and incorrectly treat 0 as the base rectangle.
# ─────────────────────────────────────────────────────────
@register_solver
class BlobQuadrant2x2Solver(BaseSolver):
    @property
    def name(self): return "BlobQuadrant2x2Solver"

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        bg = 0  # always treat 0 as background
        color_cnt = Counter(v for row in inp for v in row if v != bg)
        if not color_cnt: return None
        # base_c = large rectangle fill = most common non-0 color
        base_c = color_cnt.most_common(1)[0][0]
        base_cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == base_c]
        if not base_cells: return None
        br0, br1, bc0, bc1 = _bbox(base_cells)
        # Blob colors: other non-0 colors with >= 4 cells inside rectangle
        blob_colors = {}
        for oc in color_cnt:
            if oc == base_c: continue
            cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == oc]
            if len(cells) >= 4:
                rr0, rr1, cc0, cc1 = _bbox(cells)
                cr = (rr0 + rr1) / 2
                cc = (cc0 + cc1) / 2
                blob_colors[oc] = (cr, cc)
        if not blob_colors: return None
        mid_r = (br0 + br1) / 2
        mid_c = (bc0 + bc1) / 2
        out = [[0]*2 for _ in range(2)]
        for oc, (cr, cc) in blob_colors.items():
            qr = 0 if cr < mid_r else 1
            qc = 0 if cc < mid_c else 1
            out[qr][qc] = oc
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out is None or out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            if out:
                preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 7. NestedShapeInFrameSolver  (bf699163)
# ─────────────────────────────────────────────────────────
@register_solver
class NestedShapeInFrameSolver(BaseSolver):
    @property
    def name(self): return "NestedShapeInFrameSolver"

    def _apply(self, inp):
        H, W = len(inp), len(inp[0])
        bg = _bg(inp)
        colors = list(set(v for row in inp for v in row if v != bg))
        if not colors: return None
        bboxes = {}
        for c in colors:
            cells = [(r, cc) for r in range(H) for cc in range(W) if inp[r][cc] == c]
            bboxes[c] = _bbox(cells)
        def area(b): return (b[1]-b[0]+1) * (b[3]-b[2]+1)
        frame_candidates = [(area(bboxes[c]), c) for c in colors
                            if (bboxes[c][1]-bboxes[c][0]) > 3 or (bboxes[c][3]-bboxes[c][2]) > 3]
        if not frame_candidates: return None
        frame_c = max(frame_candidates, key=lambda x: x[0])[1]
        fb = bboxes[frame_c]
        for c in colors:
            if c == frame_c: continue
            cb = bboxes[c]
            if cb[1]-cb[0]+1 == 3 and cb[3]-cb[2]+1 == 3:
                if cb[0] >= fb[0] and cb[1] <= fb[1] and cb[2] >= fb[2] and cb[3] <= fb[3]:
                    return [inp[cb[0]+r][cb[2]:cb[2]+3] for r in range(3)]
        return None

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out is None or out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            if out:
                preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 8. MostFilledPatternSolver  (a87f7484)
# FIX: use 0 as background always.
# When patterns contain a heavily-filled color (e.g. 8 = 8 cells),
# _bg() may return that color, inverting the fill-count ranking.
# ─────────────────────────────────────────────────────────
@register_solver
class MostFilledPatternSolver(BaseSolver):
    @property
    def name(self): return "MostFilledPatternSolver"

    def _split_patterns(self, inp):
        H, W = len(inp), len(inp[0])
        patterns = []
        if H == 3 and W % 3 == 0:
            for i in range(W // 3):
                pat = [inp[r][i*3:(i+1)*3] for r in range(3)]
                patterns.append(pat)
        elif W == 3 and H % 3 == 0:
            for i in range(H // 3):
                pat = inp[i*3:(i+1)*3]
                patterns.append([row[:] for row in pat])
        return patterns

    def _apply(self, inp):
        bg = 0  # always treat 0 as background
        patterns = self._split_patterns(inp)
        if not patterns: return None
        best = max(patterns, key=lambda p: sum(1 for row in p for v in row if v != bg))
        return best

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out is None or out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            if out:
                preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 9. CountBlobsToCheckerSolver  (ff28f65a)
# Count 2×2 blobs → N → fill first N positions of checkerboard sequence with 1
# Sequence: (0,0),(0,2),(1,1),(2,0),(2,2)
# ─────────────────────────────────────────────────────────
_CHECKER_SEQ = [(0,0),(0,2),(1,1),(2,0),(2,2)]

@register_solver
class CountBlobsToCheckerSolver(BaseSolver):
    @property
    def name(self): return "CountBlobsToCheckerSolver"

    def _count_blobs(self, inp):
        bg = _bg(inp)
        H, W = len(inp), len(inp[0])
        visited = set()
        count = 0
        for sr in range(H):
            for sc in range(W):
                if inp[sr][sc] != bg and (sr, sc) not in visited:
                    queue = [(sr, sc)]
                    visited.add((sr, sc))
                    while queue:
                        cr, cc = queue.pop()
                        for nr, nc in ((cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)):
                            if 0<=nr<H and 0<=nc<W and inp[nr][nc]!=bg and (nr,nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    count += 1
        return count

    def _apply(self, inp):
        N = self._count_blobs(inp)
        out = [[0]*3 for _ in range(3)]
        for i in range(min(N, len(_CHECKER_SEQ))):
            r, c = _CHECKER_SEQ[i]
            out[r][c] = 1
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)


# ─────────────────────────────────────────────────────────
# 10. BracketFillCountSolver  (b0c4d837)
# Bracket frame of one color (walls+bottom), fill color inside from bottom up.
# N = number of empty rows (no fill) inside bracket → fill first N cells
# of spiral order in 3×3 with fill color.
# Spiral: (0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(1,1)
# FIX: require that no frame_c cells appear inside the inner region —
# prevents the fill color from being misidentified as the frame.
# ─────────────────────────────────────────────────────────
_SPIRAL_SEQ = [(0,0),(0,1),(0,2),(1,2),(2,2),(2,1),(2,0),(1,0),(1,1)]

@register_solver
class BracketFillCountSolver(BaseSolver):
    @property
    def name(self): return "BracketFillCountSolver"

    def _find_bracket(self, inp):
        H, W = len(inp), len(inp[0])
        bg = _bg(inp)
        colors = list(set(v for row in inp for v in row if v != bg))
        if len(colors) < 2: return None
        for frame_c in colors:
            cells = [(r, c) for r in range(H) for c in range(W) if inp[r][c] == frame_c]
            if not cells: continue
            fr0, fr1, fc0, fc1 = _bbox(cells)
            # Must form U-shape: bottom row + left col + right col all frame_c
            if not all(inp[fr1][c] == frame_c for c in range(fc0, fc1+1)): continue
            if not all(inp[r][fc0] == frame_c for r in range(fr0, fr1+1)): continue
            if not all(inp[r][fc1] == frame_c for r in range(fr0, fr1+1)): continue
            inner_rows = list(range(fr0, fr1))
            inner_cols = list(range(fc0+1, fc1))
            if not inner_rows or not inner_cols: continue
            # FIX: frame_c must NOT appear inside the inner region
            if any(inp[r][c] == frame_c for r in inner_rows for c in inner_cols): continue
            # Identify fill color = non-bg non-frame color inside inner region
            fill_vals = set(inp[r][c] for r in inner_rows for c in inner_cols
                            if inp[r][c] != bg and inp[r][c] != frame_c)
            if not fill_vals: continue
            fill_c = list(fill_vals)[0]
            # N = number of inner rows with NO fill cells
            N = sum(1 for r in inner_rows
                    if not any(inp[r][c] == fill_c for c in inner_cols))
            return N, fill_c
        return None

    def _apply(self, inp):
        result = self._find_bracket(inp)
        if result is None: return None
        N, fill_c = result
        out = [[0]*3 for _ in range(3)]
        for i in range(min(N, len(_SPIRAL_SEQ))):
            r, c = _SPIRAL_SEQ[i]
            out[r][c] = fill_c
        return out

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            out = self._apply(pair['input'])
            if out is None or out != pair['output']:
                return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=3)
        preds = []
        for inp in ctx.test_inputs:
            out = self._apply(inp)
            if out:
                preds.append(Prediction(attempt_1=out, attempt_2=out, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=bool(preds), tier=2)
