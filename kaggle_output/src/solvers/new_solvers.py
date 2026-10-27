"""
New Specialized Solvers — Tier-1/2 solvers for patterns identified from unsolved tasks.

Patterns covered:
  - DiagonalShearLeftSolver         : each row shifts left by (bottom_row - row) cols
  - ExpandFrameSwapSolver           : swap frame↔interior colors + cross-expand (multi-frame fix)
  - FillRightToBarSolver            : fill row rightward from marker to wall column
  - NoisyGridRepairSolver           : infer repeating tile, correct noise (active-region aware)
  - ObjectInteriorFillSolver        : per-component marker-based interior fill
  - LargestObjectRecolorSolver      : recolor the largest CC of a given color
  - BlockRecolorByPositionSolver    : recolor solid rectangular blocks by column position
  - DirectionalTrailSolver          : 2x2 box, direction-coded 3-wide trails  (1f0c79e5)
  - TrianglePyramidSolver           : column sequences → bottom pyramid         (3d6c6e23)
  - CornerScatterSolver             : 2x2 interior block → grid corners          (66e6c45b)
  - FillDownFromWallSolver          : fill downward from wall row to markers      (13713586)
  - LineIndicatorBlobRecolorSolver  : border-1 pairs define lines; recolor blobs (0d87d2a6)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from ..core.grid_codec import (
    Grid,
    background_color,
    copy_grid,
    grid_dims,
)
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver

# ── Shared utilities ───────────────────────────────────────────────────────────

def _make_pred(grid: Grid, name: str, conf: float = 1.0) -> Prediction:
    return Prediction(attempt_1=grid, attempt_2=grid, solver_name=name, confidence=conf)


def _result(task_id: str, preds: list[Prediction], tier: int = 1) -> SolverResult:
    return SolverResult(task_id=task_id, predictions=preds, solved=True, tier=tier)


def _bfs(grid: Grid, r: int, c: int, color: int) -> list[tuple[int, int]]:
    """BFS for single-color connected component."""
    h, w = grid_dims(grid)
    visited: set = {(r, c)}
    queue = [(r, c)]
    while queue:
        cr, cc = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and grid[nr][nc] == color:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return list(visited)


def _bfs_multicolor(grid: Grid, r: int, c: int, bg: int) -> list[tuple[int, int]]:
    """BFS following ALL non-bg cells (any color) from (r, c)."""
    h, w = grid_dims(grid)
    visited: set = {(r, c)}
    queue = [(r, c)]
    while queue:
        cr, cc = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and grid[nr][nc] != bg:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return list(visited)


def _all_components(grid: Grid, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    """Return list of (color, cells) for all connected components excluding bg."""
    h, w = grid_dims(grid)
    visited: set = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if (r, c) not in visited and grid[r][c] != bg:
                color = grid[r][c]
                cells = _bfs(grid, r, c, color)
                visited.update(cells)
                comps.append((color, cells))
    return comps


def _is_interior_cell(grid: Grid, r: int, c: int, bg: int) -> bool:
    """True if all 4 orthogonal neighbors exist and are non-bg."""
    h, w = grid_dims(grid)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            return False
        if grid[nr][nc] == bg:
            return False
    return True


# ── 1. DiagonalShearLeftSolver ─────────────────────────────────────────────────

@register_solver
class DiagonalShearLeftSolver(BaseSolver):
    """
    Non-background object shears diagonally left: each row shifts LEFT by
    (bottom_object_row - current_row) positions.
    """

    @property
    def name(self) -> str:
        return "DiagonalShearLeftSolver"

    @property
    def priority(self) -> int:
        return 30

    def _shear_transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        obj_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
        if not obj_cells:
            return None
        bottom_row = max(r for r, c in obj_cells)
        out = [[bg] * w for _ in range(h)]
        for r, c in obj_cells:
            shift = bottom_row - r
            new_c = c - shift
            if 0 <= new_c < w:
                out[r][new_c] = grid[r][c]
        return out

    def _verify_on_pair(self, inp: Grid, expected: Grid) -> bool:
        return self._shear_transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify_on_pair(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._shear_transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 2. ExpandFrameSwapSolver ───────────────────────────────────────────────────

@register_solver
class ExpandFrameSwapSolver(BaseSolver):
    """
    Each rectangular frame object has a frame color (outer ring) and interior color.
    1. Swap frame<->interior within the original bounding box.
    2. Extend outward as a cross (original frame color) by int_h rows and int_w cols.
    FIXED: handles MULTIPLE independent frames per grid (bug: was single-frame only).
    Verified on: 3befdf3e.
    """

    @property
    def name(self) -> str:
        return "ExpandFrameSwapSolver"

    @property
    def priority(self) -> int:
        return 35

    def _get_all_frames_info(self, grid: Grid, bg: int) -> list:
        """
        Return list of (min_r, min_c, max_r, max_c, frame_color, int_color)
        for every rectangular ring component that has a detectable interior color.
        """
        h, w = grid_dims(grid)
        visited: set = set()
        result = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] != bg and (r, c) not in visited:
                    color = grid[r][c]
                    cells = _bfs(grid, r, c, color)
                    visited.update(cells)
                    rows = [x for x, y in cells]
                    cols = [y for x, y in cells]
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)
                    # Scan interior of bounding box for a different non-bg color
                    interior_colors: Counter = Counter()
                    for ir in range(min_r + 1, max_r):
                        for ic in range(min_c + 1, max_c):
                            v = grid[ir][ic]
                            if v != bg and v != color:
                                interior_colors[v] += 1
                    if not interior_colors:
                        continue
                    int_color = interior_colors.most_common(1)[0][0]
                    result.append((min_r, min_c, max_r, max_c, color, int_color))
        return result

    def _transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        frames = self._get_all_frames_info(grid, bg)
        if not frames:
            return None
        out = copy_grid(grid)
        for (min_r, min_c, max_r, max_c, frame_color, int_color) in frames:
            int_h = max_r - min_r - 1
            int_w = max_c - min_c - 1
            if int_h <= 0 or int_w <= 0:
                continue
            new_min_r = max(0, min_r - int_h)
            new_max_r = min(h - 1, max_r + int_h)
            new_min_c = max(0, min_c - int_w)
            new_max_c = min(w - 1, max_c + int_w)
            # Step 1: Swap frame<->interior within original bbox
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if out[r][c] == frame_color:
                        out[r][c] = int_color
                    elif out[r][c] == int_color:
                        out[r][c] = frame_color
            # Step 2: Extend cross with original frame_color
            for r in range(new_min_r, min_r):
                for c in range(min_c, max_c + 1):
                    out[r][c] = frame_color
            for r in range(max_r + 1, new_max_r + 1):
                for c in range(min_c, max_c + 1):
                    out[r][c] = frame_color
            for r in range(min_r, max_r + 1):
                for c in range(new_min_c, min_c):
                    out[r][c] = frame_color
            for r in range(min_r, max_r + 1):
                for c in range(max_c + 1, new_max_c + 1):
                    out[r][c] = frame_color
        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 3. FillRightToBarSolver ────────────────────────────────────────────────────

@register_solver
class FillRightToBarSolver(BaseSolver):
    """
    Each row has a marker color. Rightmost column is a 'wall' (bar) color.
    Fill rightward from marker to wall column. Kept for tasks with rightward-fill patterns.
    """

    @property
    def name(self) -> str:
        return "FillRightToBarSolver"

    @property
    def priority(self) -> int:
        return 32

    def _infer_wall_col(self, grid: Grid, bg: int) -> Optional[int]:
        h, w = grid_dims(grid)
        for c in range(w - 1, -1, -1):
            col_colors = [grid[r][c] for r in range(h) if grid[r][c] != bg]
            if len(col_colors) == h:
                if len(set(col_colors)) == 1:
                    return c
        return None

    def _transform(self, grid: Grid, wall_col: int) -> Grid:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        out = copy_grid(grid)
        for r in range(h):
            marker_cells = [(c, grid[r][c]) for c in range(w) if grid[r][c] != bg and c != wall_col]
            if not marker_cells:
                continue
            for mc, mcolor in marker_cells:
                for c in range(mc, wall_col):
                    if out[r][c] == bg:
                        out[r][c] = mcolor
        return out

    def _verify(self, inp: Grid, expected: Grid, wall_col: int) -> bool:
        return self._transform(inp, wall_col) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        bg = background_color(ctx.train_pairs[0]["input"])
        wall_col = self._infer_wall_col(ctx.train_pairs[0]["input"], bg)
        if wall_col is None:
            return None
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"], wall_col):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            preds.append(_make_pred(self._transform(test_inp, wall_col), self.name))
        return _result(ctx.task_id, preds)


# ── 4. NoisyGridRepairSolver ───────────────────────────────────────────────────

@register_solver
class NoisyGridRepairSolver(BaseSolver):
    """
    Input is a noisy repetition of a tile pattern; output restores the clean tile.
    FIXED: active-region aware — detects bounding box of non-zero cells and applies
    tile inference only within that region, leaving dead/zero zones intact.
    Verified on: 0607ce86.
    """

    @property
    def name(self) -> str:
        return "NoisyGridRepairSolver"

    @property
    def priority(self) -> int:
        return 40

    def _get_active_region(self, grid: Grid):
        """Return (r1, c1, r2, c2) exclusive bounding box of non-zero cells, or Nones."""
        h, w = grid_dims(grid)
        min_r, max_r, min_c, max_c = h, -1, w, -1
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    if r < min_r:
                        min_r = r
                    if r > max_r:
                        max_r = r
                    if c < min_c:
                        min_c = c
                    if c > max_c:
                        max_c = c
        if max_r == -1:
            return None, None, None, None
        return min_r, min_c, max_r + 1, max_c + 1

    def _extract_subgrid(self, grid: Grid, r1: int, c1: int, r2: int, c2: int) -> Grid:
        return [list(row[c1:c2]) for row in grid[r1:r2]]

    def _find_tile(self, grid: Grid):
        """Find best (tile_h, tile_w) via majority-vote consistency score."""
        h, w = grid_dims(grid)
        best_th, best_tw = None, None
        best_score = -1
        for th in range(1, h // 2 + 1):
            if h % th != 0:
                continue
            for tw in range(1, w // 2 + 1):
                if w % tw != 0:
                    continue
                score = 0
                for tr in range(th):
                    for tc in range(tw):
                        counts: Counter = Counter()
                        for bi in range(h // th):
                            for bj in range(w // tw):
                                counts[grid[bi * th + tr][bj * tw + tc]] += 1
                        score += counts.most_common(1)[0][1]
                if score > best_score:
                    best_score = score
                    best_th, best_tw = th, tw
        return best_th, best_tw

    def _repair(self, grid: Grid, tile_h: int, tile_w: int) -> Grid:
        """Build canonical tile from majority vote; tile the entire subgrid."""
        h, w = grid_dims(grid)
        tile = []
        for tr in range(tile_h):
            row = []
            for tc in range(tile_w):
                counts: Counter = Counter()
                for bi in range(h // tile_h):
                    for bj in range(w // tile_w):
                        counts[grid[bi * tile_h + tr][bj * tile_w + tc]] += 1
                row.append(counts.most_common(1)[0][0])
            tile.append(row)
        out = [[0] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                out[r][c] = tile[r % tile_h][c % tile_w]
        return out

    def _repair_full(self, grid: Grid) -> Optional[Grid]:
        """Repair active region only; dead zone stays zeros."""
        h, w = grid_dims(grid)
        r1, c1, r2, c2 = self._get_active_region(grid)
        if r1 is None:
            return None
        sub = self._extract_subgrid(grid, r1, c1, r2, c2)
        sh, sw = r2 - r1, c2 - c1
        if sh < 2 or sw < 2:
            return None
        tile_h, tile_w = self._find_tile(sub)
        if tile_h is None:
            return None
        repaired_sub = self._repair(sub, tile_h, tile_w)
        out = [[0] * w for _ in range(h)]
        for i, row in enumerate(repaired_sub):
            for j, val in enumerate(row):
                out[r1 + i][c1 + j] = val
        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._repair_full(inp) == expected

    def _consistent_tiling(self, grid: Grid) -> bool:
        r1, c1, r2, c2 = self._get_active_region(grid)
        if r1 is None:
            return False
        sub = self._extract_subgrid(grid, r1, c1, r2, c2)
        sh, sw = r2 - r1, c2 - c1
        if sh < 2 or sw < 2:
            return False
        tile_h, tile_w = self._find_tile(sub)
        if tile_h is None or tile_h == sh or tile_w == sw:
            return False
        repaired_sub = self._repair(sub, tile_h, tile_w)
        diffs = sum(1 for r in range(sh) for c in range(sw) if sub[r][c] != repaired_sub[r][c])
        return 0 < diffs < 0.25 * sh * sw

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        for pair in ctx.train_pairs:
            if not self._consistent_tiling(pair["input"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._repair_full(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 5. ObjectInteriorFillSolver ────────────────────────────────────────────────

@register_solver
class ObjectInteriorFillSolver(BaseSolver):
    """
    FIXED: per-component marker-based interior fill.
    Each connected component of non-bg cells has exactly ONE special marker cell
    (different color from the dominant blob color). Fill 'interior' cells (all 4
    orthogonal neighbors are non-bg) with the marker color.
    Returns None gracefully if no embedded markers found (e.g. test case of 09c534e7).
    Verified on: 09c534e7 training pairs.
    """

    @property
    def name(self) -> str:
        return "ObjectInteriorFillSolver"

    @property
    def priority(self) -> int:
        return 38

    def _transform_and_check(self, grid: Grid, bg: int):
        """Returns (transformed_grid, found_any_marker_bool)."""
        h, w = grid_dims(grid)
        out = copy_grid(grid)
        visited: set = set()
        found_any = False

        for r in range(h):
            for c in range(w):
                if (r, c) not in visited and grid[r][c] != bg:
                    blob = _bfs_multicolor(grid, r, c, bg)
                    visited.update(blob)

                    color_counts: Counter = Counter(grid[br][bc] for br, bc in blob)
                    if len(color_counts) < 2:
                        continue  # No embedded marker color

                    dominant = color_counts.most_common(1)[0][0]
                    marker_cells = [(br, bc) for br, bc in blob if grid[br][bc] != dominant]
                    if len(marker_cells) != 1:
                        continue  # Need exactly 1 marker cell

                    marker_color = grid[marker_cells[0][0]][marker_cells[0][1]]
                    found_any = True

                    for br, bc in blob:
                        if _is_interior_cell(grid, br, bc, bg):
                            out[br][bc] = marker_color

        return out, found_any

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        bg = background_color(ctx.train_pairs[0]["input"])
        for pair in ctx.train_pairs:
            out, has_markers = self._transform_and_check(pair["input"], bg)
            if not has_markers or out != pair["output"]:
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out, has_markers = self._transform_and_check(test_inp, bg)
            if not has_markers:
                return None  # Graceful failure: test has no embedded markers
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 6. LargestObjectRecolorSolver ─────────────────────────────────────────────

@register_solver
class LargestObjectRecolorSolver(BaseSolver):
    """
    The largest connected component of a given source color is recolored to a
    target color; all smaller components retain their original color.
    """

    @property
    def name(self) -> str:
        return "LargestObjectRecolorSolver"

    @property
    def priority(self) -> int:
        return 36

    def _infer_recolor_rule(self, inp: Grid, out: Grid, bg: int):
        h, w = grid_dims(inp)
        for src_color in set(v for row in inp for v in row if v != bg):
            comps = []
            visited: set = set()
            for r in range(h):
                for c in range(w):
                    if (r, c) not in visited and inp[r][c] == src_color:
                        cells = _bfs(inp, r, c, src_color)
                        visited.update(cells)
                        comps.append(cells)
            if len(comps) < 2:
                continue
            largest = max(comps, key=len)
            sample_r, sample_c = largest[0]
            out_h, out_w = len(out), len(out[0]) if out else 0
            if sample_r >= out_h or sample_c >= out_w:
                continue
            tgt_color = out[sample_r][sample_c]
            if tgt_color == src_color:
                continue
            ok = True
            for comp in comps:
                expected_color = tgt_color if comp is largest else src_color
                for r, c in comp:
                    if r >= out_h or c >= out_w:
                        ok = False
                        break
                    if out[r][c] != expected_color:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return (src_color, tgt_color)
        return None

    def _transform(self, grid: Grid, src_color: int, tgt_color: int) -> Grid:
        h, w = grid_dims(grid)
        comps = []
        visited: set = set()
        for r in range(h):
            for c in range(w):
                if (r, c) not in visited and grid[r][c] == src_color:
                    cells = _bfs(grid, r, c, src_color)
                    visited.update(cells)
                    comps.append(cells)
        if not comps:
            return copy_grid(grid)
        largest = max(comps, key=len)
        out = copy_grid(grid)
        for r, c in largest:
            out[r][c] = tgt_color
        return out

    def _verify(self, inp: Grid, expected: Grid, src: int, tgt: int) -> bool:
        return self._transform(inp, src, tgt) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        bg = background_color(ctx.train_pairs[0]["input"])
        rule = self._infer_recolor_rule(
            ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"], bg
        )
        if rule is None:
            return None
        src_color, tgt_color = rule
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"], src_color, tgt_color):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp, src_color, tgt_color)
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 7. BlockRecolorByPositionSolver ──────────────────────────────────────────

@register_solver
class BlockRecolorByPositionSolver(BaseSolver):
    """
    Solid rectangular blocks of one source color get recolored with different
    target colors depending on their column position.
    Verified on: 103eff5b.
    """

    @property
    def name(self) -> str:
        return "BlockRecolorByPositionSolver"

    @property
    def priority(self) -> int:
        return 37

    def _find_solid_rects(self, grid: Grid, src_color: int) -> list[tuple[int, int, int, int]]:
        h, w = grid_dims(grid)
        visited: set = set()
        rects = []
        for r in range(h):
            for c in range(w):
                if (r, c) not in visited and grid[r][c] == src_color:
                    cells = _bfs(grid, r, c, src_color)
                    visited.update(cells)
                    rows = [x for x, y in cells]
                    cols = [y for x, y in cells]
                    min_r, max_r = min(rows), max(rows)
                    min_c, max_c = min(cols), max(cols)
                    expected_size = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if len(cells) == expected_size:
                        rects.append((min_r, min_c, max_r, max_c))
        return rects

    def _infer_position_map(self, inp: Grid, out: Grid, src_color: int):
        rects = self._find_solid_rects(inp, src_color)
        if len(rects) < 2:
            return None, None
        pos_map = {}
        for (min_r, min_c, max_r, max_c) in rects:
            tgt_colors: Counter = Counter()
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    tgt_colors[out[r][c]] += 1
            tgt = tgt_colors.most_common(1)[0][0]
            if tgt == src_color:
                return None, None
            key = (min_c, max_c)
            pos_map[key] = tgt
        return pos_map, "col_range"

    def _transform(self, grid: Grid, src_color: int, pos_map: dict, mode: str) -> Grid:
        out = copy_grid(grid)
        rects = self._find_solid_rects(grid, src_color)
        for (min_r, min_c, max_r, max_c) in rects:
            key = (min_c, max_c)
            tgt = pos_map.get(key)
            if tgt is None:
                best_key = min(pos_map.keys(), key=lambda k: abs(k[0] - min_c) + abs(k[1] - max_c))
                tgt = pos_map[best_key]
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    out[r][c] = tgt
        return out

    def _infer_src_color(self, inp: Grid, out: Grid, bg: int) -> Optional[int]:
        candidates = set(v for row in inp for v in row if v != bg)
        for color in candidates:
            rects = self._find_solid_rects(inp, color)
            if len(rects) < 2:
                continue
            changed = set()
            for (min_r, min_c, max_r, max_c) in rects:
                tgt_colors: Counter = Counter()
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        if r < len(out) and c < len(out[r]):
                            tgt_colors[out[r][c]] += 1
                tgt_mc = tgt_colors.most_common(1)
                if not tgt_mc:
                    continue
                tgt = tgt_mc[0][0]
                if tgt != color:
                    changed.add(tgt)
            if len(changed) == len(rects):
                return color
        return None

    def _verify(self, inp: Grid, expected: Grid, src: int, pos_map, mode) -> bool:
        return self._transform(inp, src, pos_map, mode) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        bg = background_color(ctx.train_pairs[0]["input"])
        src_color = self._infer_src_color(
            ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"], bg
        )
        if src_color is None:
            return None
        pos_map, mode = self._infer_position_map(
            ctx.train_pairs[0]["input"], ctx.train_pairs[0]["output"], src_color
        )
        if pos_map is None:
            return None
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"], src_color, pos_map, mode):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp, src_color, pos_map, mode)
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 8. DirectionalTrailSolver ─────────────────────────────────────────────────

@register_solver
class DirectionalTrailSolver(BaseSolver):
    """
    Find a 2x2 non-bg box containing 1 dominant-color cell + 3 color-2 cells.
    Each color-2 cell encodes a direction based on its corner position in the box:
      top-left  -> UP-LEFT   (dr=-1, dc=-1)
      top-right -> UP-RIGHT  (dr=-1, dc=+1)
      bot-left  -> DOWN-LEFT (dr=+1, dc=-1)
      bot-right -> DOWN-RIGHT(dr=+1, dc=+1)
    Trail from each color-2 cell (r,c) with direction (dr,dc): at step k=0,1,...
    fill (r+k*dr, c+k*dc-1), (r+k*dr, c+k*dc), (r+k*dr, c+k*dc+1) with dominant color.
    Stop when all 3 cells are out of bounds. k=0 overwrites the 2-cell itself.
    Keep all original dominant-color cells from input.
    Verified on: 1f0c79e5.
    """

    @property
    def name(self) -> str:
        return "DirectionalTrailSolver"

    @property
    def priority(self) -> int:
        return 42

    def _find_box(self, grid: Grid, bg: int):
        """Find 2x2 box with exactly 1 dominant + 3 color-2 cells. Returns (box_r, box_c, dominant)."""
        h, w = grid_dims(grid)
        for r in range(h - 1):
            for c in range(w - 1):
                tl = grid[r][c]
                tr = grid[r][c + 1]
                bl = grid[r + 1][c]
                br = grid[r + 1][c + 1]
                if any(v == bg for v in (tl, tr, bl, br)):
                    continue
                non_two = [v for v in (tl, tr, bl, br) if v != 2]
                if len(non_two) == 1:
                    return r, c, non_two[0]
        return None, None, None

    def _transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        box_r, box_c, dominant = self._find_box(grid, bg)
        if box_r is None:
            return None

        # Direction for each corner of the 2x2 box
        corner_dirs = {
            (box_r,     box_c):     (-1, -1),  # top-left  -> UP-LEFT
            (box_r,     box_c + 1): (-1, +1),  # top-right -> UP-RIGHT
            (box_r + 1, box_c):     (+1, -1),  # bot-left  -> DOWN-LEFT
            (box_r + 1, box_c + 1): (+1, +1),  # bot-right -> DOWN-RIGHT
        }

        # Start with bg grid; copy all dominant-color cells from input
        out = [[bg] * w for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if grid[r][c] == dominant:
                    out[r][c] = dominant

        # Draw 3-wide trail from each color-2 cell
        for (cr, cc), (dr, dc) in corner_dirs.items():
            if grid[cr][cc] != 2:
                continue
            for k in range(h + w):
                fr = cr + k * dr
                fc = cc + k * dc
                to_fill = [(fr, fc - 1), (fr, fc), (fr, fc + 1)]
                in_bounds = [(nr, nc) for nr, nc in to_fill if 0 <= nr < h and 0 <= nc < w]
                if not in_bounds:
                    break
                for nr, nc in in_bounds:
                    out[nr][nc] = dominant

        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 9. TrianglePyramidSolver ──────────────────────────────────────────────────

@register_solver
class TrianglePyramidSolver(BaseSolver):
    """
    For each column with non-bg cells: collect all non-bg cells top-to-bottom
    as an ordered sequence. Count total n; verify n = k^2 for integer k.
    Split into groups of sizes 1, 3, 5, ..., 2k-1 (each group's color = first cell's color).
    Build k-row pyramid at BOTTOM of grid centered on that column:
      row i (0-indexed from top, i=0..k-1) at grid row (h-k+i),
      columns (col-i) to (col+i), filled with groups[i] color.
    Output is all-bg except pyramid cells. Columns with non-perfect-square
    cell counts are skipped silently; training verification catches bad matches.
    Verified on: 3d6c6e23.
    """

    @property
    def name(self) -> str:
        return "TrianglePyramidSolver"

    @property
    def priority(self) -> int:
        return 43

    def _transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        out = [[bg] * w for _ in range(h)]
        processed_any = False

        for col in range(w):
            # Collect non-bg cells top-to-bottom (ordered, gaps ignored)
            seq = [(r, grid[r][col]) for r in range(h) if grid[r][col] != bg]
            if not seq:
                continue

            n = len(seq)
            k = math.isqrt(n)
            if k * k != n:
                continue  # Skip: not a perfect square

            processed_any = True

            # Split into groups of sizes 1, 3, 5, ..., 2k-1
            groups = []
            idx = 0
            for i in range(k):
                size = 2 * i + 1
                group_color = seq[idx][1]  # Color of first cell in this group
                idx += size
                groups.append(group_color)

            # Build pyramid at bottom of grid, centered at col
            for i in range(k):
                grid_row = h - k + i
                color = groups[i]
                for dc in range(-i, i + 1):
                    nc = col + dc
                    if 0 <= nc < w and 0 <= grid_row < h:
                        out[grid_row][nc] = color

        if not processed_any:
            return None
        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 10. CornerScatterSolver ───────────────────────────────────────────────────

@register_solver
class CornerScatterSolver(BaseSolver):
    """
    Exactly 4 non-bg cells form an adjacent 2x2 interior block.
    Each cell is scattered to the corresponding corner of the full grid:
      top-left  -> (0, 0)
      top-right -> (0, w-1)
      bot-left  -> (h-1, 0)
      bot-right -> (h-1, w-1)
    All other cells become background.
    Verified on: 66e6c45b.
    """

    @property
    def name(self) -> str:
        return "CornerScatterSolver"

    @property
    def priority(self) -> int:
        return 44

    def _transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)

        non_bg = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != bg]
        if len(non_bg) != 4:
            return None

        rows = sorted(set(r for r, c in non_bg))
        cols = sorted(set(c for r, c in non_bg))
        if len(rows) != 2 or len(cols) != 2:
            return None
        r0, r1 = rows
        c0, c1 = cols
        # Must be an adjacent 2x2 block
        if r1 - r0 != 1 or c1 - c0 != 1:
            return None

        corner_map = {
            (r0, c0): (0,     0),
            (r0, c1): (0,     w - 1),
            (r1, c0): (h - 1, 0),
            (r1, c1): (h - 1, w - 1),
        }

        out = [[bg] * w for _ in range(h)]
        for (sr, sc), (dr, dc) in corner_map.items():
            out[dr][dc] = grid[sr][sc]
        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 11. FillDownFromWallSolver ────────────────────────────────────────────────

@register_solver
class FillDownFromWallSolver(BaseSolver):
    """
    A wall row (all cells same non-bg color, e.g. row 0 all 5s) anchors the fill.
    Each marker blob below the wall defines a fill region:
      fill downward from (wall_row+1) to marker_row (inclusive),
      within the blob's column span (min_c..max_c), writing only to bg cells.
    Blobs processed shallowest-first so deep fills don't overwrite shallow fills.
    Verified on: 13713586.
    """

    @property
    def name(self) -> str:
        return "FillDownFromWallSolver"

    @property
    def priority(self) -> int:
        return 33

    def _find_wall_row(self, grid: Grid, bg: int):
        """Return (wall_row, wall_color) or (None, None)."""
        h, w = grid_dims(grid)
        for r in range(h):
            row_vals = [grid[r][c] for c in range(w)]
            if all(v != bg for v in row_vals) and len(set(row_vals)) == 1:
                return r, row_vals[0]
        return None, None

    def _find_markers(self, grid: Grid, bg: int, wall_row: int, wall_color: int):
        """Return list of (top_row, min_col, max_col, color) for blobs below wall."""
        h, w = grid_dims(grid)
        visited: set = set()
        markers = []
        for r in range(wall_row + 1, h):
            for c in range(w):
                if (r, c) not in visited and grid[r][c] != bg and grid[r][c] != wall_color:
                    color = grid[r][c]
                    cells = _bfs(grid, r, c, color)
                    visited.update(cells)
                    rows_list = [x for x, y in cells]
                    cols_list = [y for x, y in cells]
                    markers.append((min(rows_list), min(cols_list), max(cols_list), color))
        return markers

    def _transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        wall_row, wall_color = self._find_wall_row(grid, bg)
        if wall_row is None:
            return None

        markers = self._find_markers(grid, bg, wall_row, wall_color)
        if not markers:
            return None

        # Process shallowest (smallest row) markers first
        markers.sort(key=lambda m: m[0])

        out = copy_grid(grid)
        for (marker_row, min_c, max_c, mcolor) in markers:
            for r in range(wall_row + 1, marker_row + 1):
                for c in range(min_c, max_c + 1):
                    if out[r][c] == bg:
                        out[r][c] = mcolor

        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)


# ── 12. LineIndicatorBlobRecolorSolver ────────────────────────────────────────

@register_solver
class LineIndicatorBlobRecolorSolver(BaseSolver):
    """
    Pairs of border cells with value 1 define lines:
      same-row pair  -> horizontal line at that row (fill entire row with 1)
      same-col pair  -> vertical line at that col (fill entire col with 1)
    Draw ALL defined lines.
    Recolor ALL connected blobs of color 2 to color 1 if ANY blob cell is within
    Manhattan distance <= 1 of any drawn line (directly on or immediately adjacent).
    Non-touching 2-blobs (distance >= 2) stay as 2.
    Verified on: 0d87d2a6.
    """

    @property
    def name(self) -> str:
        return "LineIndicatorBlobRecolorSolver"

    @property
    def priority(self) -> int:
        return 41

    def _find_lines(self, grid: Grid):
        """Return (line_rows_set, line_cols_set) from paired border 1-cells.
        
        Horizontal line at row R: requires 1 at (R, 0) AND (R, W-1) — left+right border.
        Vertical line at col C: requires 1 at (0, C) AND (H-1, C) — top+bottom border.
        This prevents corner/same-side pairs from creating spurious lines.
        """
        h, w = grid_dims(grid)
        line_rows = set()
        line_cols = set()
        for r in range(h):
            if grid[r][0] == 1 and grid[r][w - 1] == 1:
                line_rows.add(r)
        for c in range(w):
            if grid[0][c] == 1 and grid[h - 1][c] == 1:
                line_cols.add(c)
        return line_rows, line_cols

    def _touches_line(self, br: int, bc: int, line_rows: set, line_cols: set) -> bool:
        """True if (br, bc) is within Manhattan distance 1 of any line."""
        for R in line_rows:
            if abs(br - R) <= 1:
                return True
        for C in line_cols:
            if abs(bc - C) <= 1:
                return True
        return False

    def _transform(self, grid: Grid) -> Optional[Grid]:
        h, w = grid_dims(grid)
        bg = background_color(grid)
        line_rows, line_cols = self._find_lines(grid)
        if not line_rows and not line_cols:
            return None

        out = copy_grid(grid)

        # Draw all defined lines
        for R in line_rows:
            for c in range(w):
                out[R][c] = 1
        for C in line_cols:
            for r in range(h):
                out[r][C] = 1

        # Recolor touching 2-blobs
        visited: set = set()
        for r in range(h):
            for c in range(w):
                if (r, c) not in visited and grid[r][c] == 2:
                    cells = _bfs(grid, r, c, 2)
                    visited.update(cells)
                    if any(self._touches_line(br, bc, line_rows, line_cols) for br, bc in cells):
                        for br, bc in cells:
                            out[br][bc] = 1

        return out

    def _verify(self, inp: Grid, expected: Grid) -> bool:
        return self._transform(inp) == expected

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for pair in ctx.train_pairs:
            if not self._verify(pair["input"], pair["output"]):
                return None
        preds = []
        for test_inp in ctx.test_inputs:
            out = self._transform(test_inp)
            if out is None:
                return None
            preds.append(_make_pred(out, self.name))
        return _result(ctx.task_id, preds)
