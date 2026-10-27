"""
advanced_solvers8.py — Targeted solvers for confirmed unsolved patterns.

Solvers:
  1. TileRepeatByInputSizeSolver   — reps = input dimensions (ccd554ac +1)
  2. OddQuadrantOutSolver          — zero-row/col divider, odd quadrant out (88a62173 +1)
  3. SinglePixelColorIndicatorSolver — single pixel recolors the main shape (aabf363d +1)
  4. SingleColorFixedRecolorSolver — same mask, one non-bg color → fixed target color per task
  5. CropMainObjectSolver          — crop = bounding box of the largest non-bg connected component
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Optional

from ..core.grid_codec import Grid, background_color
from ..core.interfaces import BaseSolver, SolverResult, TaskContext, Prediction
from .registry import register_solver


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _sh(g: Grid):
    return len(g), (len(g[0]) if g else 0)

def _bg(g: Grid) -> int:
    return background_color(g)

def _clone(g: Grid) -> Grid:
    return [list(r) for r in g]

def _same_shape(a: Grid, b: Grid) -> bool:
    return _sh(a) == _sh(b)

def _result(ctx: TaskContext, grids, solver_name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=solver_name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

def _bfs4(g: Grid, sr: int, sc: int) -> set:
    H, W = _sh(g)
    col = g[sr][sc]
    vis = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in vis and g[nr][nc] == col:
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis

def _all_ccs(g: Grid, bg: int) -> list:
    """Return list of (color, frozenset_of_cells) for all non-bg CCs."""
    H, W = _sh(g)
    vis = set()
    ccs = []
    for r in range(H):
        for c in range(W):
            if (r, c) not in vis and g[r][c] != bg:
                cc = _bfs4(g, r, c)
                vis.update(cc)
                ccs.append((g[r][c], cc))
    return ccs


# ── 1. TileRepeatByInputSizeSolver ─────────────────────────────────────────────

@register_solver
class TileRepeatByInputSizeSolver(BaseSolver):
    """
    Output = input tiled (ih × iw) times (reps equal to the input dimensions).
    No flipping. Each train pair may have a different input size; the rule is
    reps_r = ih, reps_c = iw for each pair independently.
    Covers: ccd554ac (2×2→4×4, 3×3→9×9, 4×4→16×16, test 5×5→25×25).
    """

    @property
    def name(self) -> str:
        return "TileRepeatByInputSizeSolver"

    @property
    def priority(self) -> int:
        return 66  # higher than TileWithFlipSolver (63)

    def _tile(self, grid: Grid, rr: int, rc: int) -> Grid:
        ih, iw = _sh(grid)
        out = []
        for ri in range(rr):
            for r in range(ih):
                row = []
                for ci in range(rc):
                    row.extend(grid[r])
                out.append(row)
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            if p["output"] is None:
                return None
            ih, iw = _sh(p["input"])
            oh, ow = _sh(p["output"])
            # Strict: output must be exactly ih*ih rows and iw*iw cols
            if oh != ih * ih or ow != iw * iw:
                return None
            expected = self._tile(p["input"], ih, iw)
            if expected != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            ih, iw = _sh(inp)
            preds.append(self._tile(inp, ih, iw))
        return _result(ctx, preds, self.name)


# ── 2. OddQuadrantOutSolver ────────────────────────────────────────────────────

@register_solver
class OddQuadrantOutSolver(BaseSolver):
    """
    Grid has at least one all-background row AND at least one all-background col
    that divide it into rectangular blocks. Exactly 4 blocks with equal size;
    3 are identical and 1 differs. Output is the differing (odd) block.
    Covers: 88a62173.
    """

    @property
    def name(self) -> str:
        return "OddQuadrantOutSolver"

    @property
    def priority(self) -> int:
        return 67

    def _divider_rows(self, grid: Grid, bg: int) -> list:
        H, W = _sh(grid)
        return [r for r in range(H) if all(grid[r][c] == bg for c in range(W))]

    def _divider_cols(self, grid: Grid, bg: int) -> list:
        H, W = _sh(grid)
        return [c for c in range(W) if all(grid[r][c] == bg for r in range(H))]

    def _extract_blocks(self, grid: Grid, div_rows: list, div_cols: list) -> list:
        """Extract rectangular blocks between dividers. Returns list of (block_grid,)."""
        H, W = _sh(grid)
        row_ranges = []
        prev = 0
        for dr in sorted(div_rows):
            if dr > prev:
                row_ranges.append((prev, dr))
            prev = dr + 1
        if prev < H:
            row_ranges.append((prev, H))

        col_ranges = []
        prev = 0
        for dc in sorted(div_cols):
            if dc > prev:
                col_ranges.append((prev, dc))
            prev = dc + 1
        if prev < W:
            col_ranges.append((prev, W))

        blocks = []
        for r0, r1 in row_ranges:
            for c0, c1 in col_ranges:
                blk = [grid[r][c0:c1] for r in range(r0, r1)]
                blocks.append(blk)
        return blocks

    def _find_odd(self, blocks: list) -> Optional[list]:
        """Among 4 equal-sized blocks, return the one that differs from the other 3."""
        if len(blocks) != 4:
            return None
        # Convert to hashable
        def _key(b): return tuple(tuple(r) for r in b)
        keys = [_key(b) for b in blocks]
        cnt = Counter(keys)
        # Expect: one majority key (3 times) and one minority key (1 time)
        if len(cnt) != 2:
            return None
        majority_count = max(cnt.values())
        minority_count = min(cnt.values())
        if majority_count != 3 or minority_count != 1:
            return None
        odd_key = [k for k, v in cnt.items() if v == 1][0]
        idx = keys.index(odd_key)
        return blocks[idx]

    def _apply(self, grid: Grid) -> Optional[list]:
        bg = _bg(grid)
        div_rows = self._divider_rows(grid, bg)
        div_cols = self._divider_cols(grid, bg)
        if not div_rows or not div_cols:
            return None
        blocks = self._extract_blocks(grid, div_rows, div_cols)
        if len(blocks) != 4:
            return None
        # All blocks must be same size
        sizes = [_sh(b) for b in blocks]
        if len(set(sizes)) != 1:
            return None
        return self._find_odd(blocks)

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            if p["output"] is None:
                return None
            pred = self._apply(p["input"])
            if pred is None:
                return None
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            pred = self._apply(inp)
            if pred is None:
                return None
            preds.append(pred)
        return _result(ctx, preds, self.name)


# ── 3. SinglePixelColorIndicatorSolver ────────────────────────────────────────

@register_solver
class SinglePixelColorIndicatorSolver(BaseSolver):
    """
    Input has exactly ONE large connected non-bg shape and exactly ONE isolated
    single-pixel of a DIFFERENT color. Rule: recolor the shape to the pixel's
    color; the pixel position becomes background.
    Covers: aabf363d.
    """

    @property
    def name(self) -> str:
        return "SinglePixelColorIndicatorSolver"

    @property
    def priority(self) -> int:
        return 68

    def _parse(self, grid: Grid):
        """Return (shape_color, pixel_color, pixel_pos) or None."""
        bg = _bg(grid)
        ccs = _all_ccs(grid, bg)
        if not ccs:
            return None
        pixels = [(col, cc) for col, cc in ccs if len(cc) == 1]
        shapes = [(col, cc) for col, cc in ccs if len(cc) > 1]
        if len(pixels) != 1 or len(shapes) != 1:
            return None
        pixel_color, p_cc = pixels[0]
        shape_color, _ = shapes[0]
        if pixel_color == shape_color:
            return None
        pixel_pos = next(iter(p_cc))
        return shape_color, pixel_color, pixel_pos

    def _apply(self, grid: Grid, shape_color: int, pixel_color: int, pixel_pos: tuple) -> Grid:
        bg = _bg(grid)
        out = _clone(grid)
        H, W = _sh(out)
        for r in range(H):
            for c in range(W):
                if out[r][c] == shape_color:
                    out[r][c] = pixel_color
        pr, pc = pixel_pos
        out[pr][pc] = bg
        return out

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            if p["output"] is None:
                return None
            if not _same_shape(p["input"], p["output"]):
                return None
            info = self._parse(p["input"])
            if info is None:
                return None
            shape_color, pixel_color, pixel_pos = info
            pred = self._apply(p["input"], shape_color, pixel_color, pixel_pos)
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            info = self._parse(inp)
            if info is None:
                return None
            shape_color, pixel_color, pixel_pos = info
            preds.append(self._apply(inp, shape_color, pixel_color, pixel_pos))
        return _result(ctx, preds, self.name)


# ── 4. SingleColorFixedRecolorSolver ──────────────────────────────────────────

@register_solver
class SingleColorFixedRecolorSolver(BaseSolver):
    """
    Input has bg + exactly ONE non-bg color A.
    Output has bg + exactly ONE non-bg color B (same mask positions, B≠A).
    Learn: color A → color B consistently across ALL train pairs.
    All train pairs must share the SAME (A, B) mapping.
    Apply: test input's non-bg color → B.
    Targets: single-color same-shape recolor tasks (~28 tasks).
    """

    @property
    def name(self) -> str:
        return "SingleColorFixedRecolorSolver"

    @property
    def priority(self) -> int:
        return 69

    def _one_nonbg_color(self, grid: Grid, bg: int) -> Optional[int]:
        """Return the single non-bg color in grid, or None if not exactly one."""
        colors = {grid[r][c] for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != bg}
        if len(colors) == 1:
            return next(iter(colors))
        return None

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None

        # Learn fixed mapping A → B from train pairs
        mapping_a = None  # input color
        mapping_b = None  # output color

        for p in ctx.train_pairs:
            if p["output"] is None:
                return None
            if not _same_shape(p["input"], p["output"]):
                return None
            bg = _bg(p["input"])
            a = self._one_nonbg_color(p["input"], bg)
            b = self._one_nonbg_color(p["output"], _bg(p["output"]))
            if a is None or b is None:
                return None
            if a == b:
                return None  # colors must differ (otherwise identity)

            if mapping_a is None:
                mapping_a, mapping_b = a, b
            else:
                if a != mapping_a or b != mapping_b:
                    return None  # inconsistent mapping

        if mapping_a is None or mapping_b is None:
            return None

        # Verify: mask must be identical between input and output in each pair
        for p in ctx.train_pairs:
            bg = _bg(p["input"])
            H, W = _sh(p["input"])
            for r in range(H):
                for c in range(W):
                    iv = p["input"][r][c]
                    ov = p["output"][r][c]
                    expected_o = mapping_b if iv == mapping_a else bg
                    if ov != expected_o:
                        return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            out = _clone(inp)
            H, W = _sh(inp)
            for r in range(H):
                for c in range(W):
                    if out[r][c] == mapping_a:
                        out[r][c] = mapping_b
            preds.append(out)
        return _result(ctx, preds, self.name)


# ── 5. CropMainObjectSolver ────────────────────────────────────────────────────

@register_solver
class CropMainObjectSolver(BaseSolver):
    """
    Output = bounding box crop of the largest non-bg connected component in input.
    The output size and position vary per pair (content-driven), but the rule is
    consistent: always the LARGEST CC's bounding box.
    Targets: subset of the 30 confirmed crop tasks.
    """

    @property
    def name(self) -> str:
        return "CropMainObjectSolver"

    @property
    def priority(self) -> int:
        return 70

    def _largest_cc_bbox_crop(self, grid: Grid) -> Optional[Grid]:
        bg = _bg(grid)
        ccs = _all_ccs(grid, bg)
        if not ccs:
            return None
        # Largest CC by area
        _, largest = max(ccs, key=lambda x: len(x[1]))
        rows = [r for r, c in largest]
        cols = [c for r, c in largest]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        if not ctx.train_pairs:
            return None
        for p in ctx.train_pairs:
            if p["output"] is None:
                return None
            pred = self._largest_cc_bbox_crop(p["input"])
            if pred is None:
                return None
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            pred = self._largest_cc_bbox_crop(inp)
            if pred is None:
                return None
            preds.append(pred)
        return _result(ctx, preds, self.name)
