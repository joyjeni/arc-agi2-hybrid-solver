"""Session 39 solvers — 6 new solver classes, +7 training tasks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from collections import Counter
from src.core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction
from src.solvers.registry import register_solver


@register_solver
class RemoveProtrusionsSolver(BaseSolver):
    """7f4411dc — remove non-bg cells with <=1 non-bg 4-connected neighbor."""
    @property
    def name(self): return "RemoveProtrusionsSolver"

    def _bg(self, grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def _remove(self, grid):
        bg = self._bg(grid)
        H, W = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        changed = True
        while changed:
            changed = False
            for r in range(H):
                for c in range(W):
                    if result[r][c] == bg:
                        continue
                    nb = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                             if 0 <= r+dr < H and 0 <= c+dc < W and result[r+dr][c+dc] != bg)
                    if nb <= 1:
                        result[r][c] = bg
                        changed = True
        return result

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._remove(pair["input"]) != pair["output"]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._remove(inp)
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class RowColFillByColorSolver(BaseSolver):
    """178fcbfb — each non-bg pixel fills full row or col; color->axis learned from training."""
    @property
    def name(self): return "RowColFillByColorSolver"

    def _bg(self, grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def _learn_color_axis(self, inp, out):
        bg = self._bg(inp)
        H, W = len(inp), len(inp[0])
        pixels = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != bg]
        if not pixels:
            return None
        color_axis = {}
        for (pr, pc, pcolor) in pixels:
            row_count = sum(1 for c in range(W) if inp[pr][c] == bg and out[pr][c] == pcolor)
            col_count = sum(1 for r in range(H) if inp[r][pc] == bg and out[r][pc] == pcolor)
            if row_count == 0 and col_count == 0:
                return None
            axis = "row" if row_count >= col_count else "col"
            if pcolor in color_axis and color_axis[pcolor] != axis:
                return None
            color_axis[pcolor] = axis
        return color_axis

    def _predict(self, inp, color_axis):
        bg = self._bg(inp)
        H, W = len(inp), len(inp[0])
        pixels = [(r, c, inp[r][c]) for r in range(H) for c in range(W) if inp[r][c] != bg]
        if not pixels:
            return None
        result = [row[:] for row in inp]
        for (pr, pc, pcolor) in pixels:
            if color_axis.get(pcolor) == "col":
                for r in range(H):
                    result[r][pc] = pcolor
        for (pr, pc, pcolor) in pixels:
            if color_axis.get(pcolor) == "row":
                for c in range(W):
                    result[pr][c] = pcolor
        return result

    def solve(self, ctx):
        color_axis = None
        for pair in ctx.train_pairs:
            ca = self._learn_color_axis(pair["input"], pair["output"])
            if ca is None:
                return None
            if color_axis is None:
                color_axis = ca
            elif ca != color_axis:
                return None
        for pair in ctx.train_pairs:
            if self._predict(pair["input"], color_axis) != pair["output"]:
                return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._predict(inp, color_axis)
            if p is None:
                return None
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class CenterLinesRectSolver(BaseSolver):
    """41e4d17e — hollow rects: draw full-grid center row+col for each rect with learned new_color."""
    @property
    def name(self): return "CenterLinesRectSolver"

    def _bg(self, grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def _find_hollow_rects(self, grid, bg):
        H, W = len(grid), len(grid[0])
        visited = [[False]*W for _ in range(H)]
        rects = []
        for sr in range(H):
            for sc in range(W):
                if grid[sr][sc] != bg and not visited[sr][sc]:
                    color = grid[sr][sc]
                    comp, queue = [], [(sr, sc)]
                    visited[sr][sc] = True
                    while queue:
                        r, c = queue.pop(0)
                        comp.append((r, c))
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                    rs = [r for r,c in comp]; cs = [c for r,c in comp]
                    top, bot, lft, rgt = min(rs), max(rs), min(cs), max(cs)
                    border_set = set()
                    for c in range(lft, rgt+1):
                        border_set.add((top, c)); border_set.add((bot, c))
                    for r in range(top+1, bot):
                        border_set.add((r, lft)); border_set.add((r, rgt))
                    if set(comp) == border_set and (bot-top) >= 2 and (rgt-lft) >= 2:
                        rects.append((top, bot, lft, rgt, color))
        return rects

    def _apply(self, grid, rects, new_color):
        bg = self._bg(grid)
        H, W = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        for (top, bot, lft, rgt, _) in rects:
            cr, cc = (top+bot)//2, (lft+rgt)//2
            for c in range(W):
                if result[cr][c] == bg: result[cr][c] = new_color
            for r in range(H):
                if result[r][cc] == bg: result[r][cc] = new_color
        return result

    def _learn_new_color(self, inp, out, rects):
        bg = self._bg(inp)
        H, W = len(inp), len(inp[0])
        colors = set()
        for (top, bot, lft, rgt, _) in rects:
            cr, cc = (top+bot)//2, (lft+rgt)//2
            for c in range(W):
                if inp[cr][c] == bg and out[cr][c] != bg: colors.add(out[cr][c])
            for r in range(H):
                if inp[r][cc] == bg and out[r][cc] != bg: colors.add(out[r][cc])
        return colors.pop() if len(colors) == 1 else None

    def solve(self, ctx):
        new_color = None
        for pair in ctx.train_pairs:
            bg = self._bg(pair["input"])
            rects = self._find_hollow_rects(pair["input"], bg)
            if not rects: return None
            nc = self._learn_new_color(pair["input"], pair["output"], rects)
            if nc is None: return None
            if new_color is None: new_color = nc
            elif new_color != nc: return None
        for pair in ctx.train_pairs:
            bg = self._bg(pair["input"])
            rects = self._find_hollow_rects(pair["input"], bg)
            if self._apply(pair["input"], rects, new_color) != pair["output"]: return None
        preds = []
        for inp in ctx.test_inputs:
            bg = self._bg(inp)
            rects = self._find_hollow_rects(inp, bg)
            if not rects: return None
            p = self._apply(inp, rects, new_color)
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class CornerExtensionSolver(BaseSolver):
    """705a3229 — each non-bg pixel extends toward nearest edge using its OWN color."""
    @property
    def name(self): return "CornerExtensionSolver"

    def _bg(self, grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def _apply(self, grid):
        bg = self._bg(grid)
        H, W = len(grid), len(grid[0])
        result = [row[:] for row in grid]
        pixels = [(r, c, grid[r][c]) for r in range(H) for c in range(W) if grid[r][c] != bg]
        for (r, c, color) in pixels:
            if r * 2 < H:
                for rr in range(0, r):
                    if result[rr][c] == bg: result[rr][c] = color
            else:
                for rr in range(r+1, H):
                    if result[rr][c] == bg: result[rr][c] = color
            if c * 2 < W:
                for cc in range(0, c):
                    if result[r][cc] == bg: result[r][cc] = color
            else:
                for cc in range(c+1, W):
                    if result[r][cc] == bg: result[r][cc] = color
        return result

    def solve(self, ctx):
        for pair in ctx.train_pairs:
            if self._apply(pair["input"]) != pair["output"]: return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._apply(inp)
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class BboxBorderFrom3PointsSolver(BaseSolver):
    """e7639916 — exactly 3 non-bg cells -> draw hollow bounding box border with learned new_color."""
    @property
    def name(self): return "BboxBorderFrom3PointsSolver"

    def _bg(self, grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def _apply(self, grid, new_color):
        bg = self._bg(grid)
        H, W = len(grid), len(grid[0])
        cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
        if len(cells) != 3: return None
        rs = [r for r,c in cells]; cs = [c for r,c in cells]
        top, bot, lft, rgt = min(rs), max(rs), min(cs), max(cs)
        if top == bot or lft == rgt: return None
        result = [row[:] for row in grid]
        for c in range(lft, rgt+1):
            if result[top][c] == bg: result[top][c] = new_color
            if result[bot][c] == bg: result[bot][c] = new_color
        for r in range(top+1, bot):
            if result[r][lft] == bg: result[r][lft] = new_color
            if result[r][rgt] == bg: result[r][rgt] = new_color
        return result

    def _learn_color(self, inp, out):
        bg = self._bg(inp)
        colors = set()
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] == bg and out[r][c] != bg: colors.add(out[r][c])
        return colors.pop() if len(colors) == 1 else None

    def solve(self, ctx):
        new_color = None
        for pair in ctx.train_pairs:
            nc = self._learn_color(pair["input"], pair["output"])
            if nc is None: return None
            if new_color is None: new_color = nc
            elif new_color != nc: return None
        for pair in ctx.train_pairs:
            if self._apply(pair["input"], new_color) != pair["output"]: return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._apply(inp, new_color)
            if p is None: return None
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)


@register_solver
class LShapeConnectorSolver(BaseSolver):
    """a2fd1cf0, d4a91cb9 — 2 diff-color cells; learn color->row/col role; draw L-shape connector."""
    @property
    def name(self): return "LShapeConnectorSolver"

    def _bg(self, grid):
        return Counter(c for row in grid for c in row).most_common(1)[0][0]

    def _cells(self, grid):
        bg = self._bg(grid)
        return [(r, c, grid[r][c]) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != bg]

    def _try_order(self, inp, row_cell, col_cell, new_color):
        bg = self._bg(inp)
        rr, rc, _ = row_cell
        cr, cc, _ = col_cell
        result = [row[:] for row in inp]
        c_start, c_end = min(rc, cc), max(rc, cc)
        for c in range(c_start, c_end+1):
            if result[rr][c] == bg: result[rr][c] = new_color
        r_start, r_end = min(cr, rr), max(cr, rr)
        for r in range(r_start, r_end+1):
            if result[r][cc] == bg: result[r][cc] = new_color
        return result

    def _learn_schema(self, inp, out):
        bg = self._bg(inp)
        cells = self._cells(inp)
        if len(cells) != 2: return None
        (r0,c0,col0), (r1,c1,col1) = cells
        if col0 == col1: return None
        new_colors = set()
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                if inp[r][c] == bg and out[r][c] != bg: new_colors.add(out[r][c])
        if len(new_colors) != 1: return None
        nc = new_colors.pop()
        resA = self._try_order(inp, (r0,c0,col0), (r1,c1,col1), nc)
        if resA == out: return {"row_color": col0, "col_color": col1, "new_color": nc}
        resB = self._try_order(inp, (r1,c1,col1), (r0,c0,col0), nc)
        if resB == out: return {"row_color": col1, "col_color": col0, "new_color": nc}
        return None

    def _predict(self, inp, schema):
        bg = self._bg(inp)
        cells = self._cells(inp)
        if len(cells) != 2: return None
        row_cell = next(((r,c,col) for r,c,col in cells if col == schema["row_color"]), None)
        col_cell = next(((r,c,col) for r,c,col in cells if col == schema["col_color"]), None)
        if row_cell is None or col_cell is None: return None
        return self._try_order(inp, row_cell, col_cell, schema["new_color"])

    def solve(self, ctx):
        schema = None
        for pair in ctx.train_pairs:
            s = self._learn_schema(pair["input"], pair["output"])
            if s is None: return None
            if schema is None: schema = s
            elif s != schema: return None
        for pair in ctx.train_pairs:
            if self._predict(pair["input"], schema) != pair["output"]: return None
        preds = []
        for inp in ctx.test_inputs:
            p = self._predict(inp, schema)
            if p is None: return None
            preds.append(Prediction(attempt_1=p, attempt_2=p, solver_name=self.name, confidence=1.0))
        return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=2)
