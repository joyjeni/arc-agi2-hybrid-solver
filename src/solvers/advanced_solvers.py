"""
advanced_solvers.py — Specialized pattern-learning solvers for ARC-AGI-2.

New solvers targeting 8 unsolved buckets:
  1. ParameterizedColorSwapSolver  — anchor-driven per-pair color mapping (f76d97a5)
  2. RowPeriodicFillSolver          — learn output period from non-zero rows
  3. ZeroFillByPeriodicTileSolver   — fill zeros using detected tile pattern
  4. TileWithFlipSolver             — tile + optional mirror
  5. ObjectRecolorByShapeSolver     — recolor CCs by shape-type classification
  6. SymmetryCompleteSolver         — complete partial h/v/rot180 symmetry
  7. SelectSubgridByMarkerSolver    — select one sub-block indicated by marker
  8. GridRowColSortSolver           — sort rows or cols by a derived key
  9. RepeatRowColPatternSolver      — repeat a detected tile row/col pattern
 10. ColorCountToOutputSolver       — count color frequencies → output grid
 11. MosaicSolver                   — fill grid from mosaic tile rule
 12. GravityAllDirectionsSolver     — gravity towards any of 4 borders
 13. FillEnclosedHolesSolver        — fill all holes enclosed by fg color
 14. PerPairObjectMoverSolver       — learn per-pair object displacement
 15. ZeroRunFillSolver              — fill zero-runs by surrounding colors
"""
from __future__ import annotations

from collections import Counter, deque

from ..core.grid_codec import Grid, background_color
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .registry import register_solver

# ── Shared utilities ────────────────────────────────────────────────────────

def _sh(g: Grid):
    return len(g), (len(g[0]) if g else 0)

def _clone(g: Grid) -> Grid:
    return [list(r) for r in g]

def _bg(g: Grid) -> int:
    return background_color(g)

def _unique_nonbg(g: Grid, bg: int) -> set:
    return {v for r in g for v in r if v != bg}

def _result(ctx: TaskContext, grids, solver_name: str) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name=solver_name) for g in grids]
    return SolverResult(task_id=ctx.task_id, predictions=preds, solved=True, tier=1)

def _fail(ctx: TaskContext) -> SolverResult:
    return SolverResult(task_id=ctx.task_id, predictions=[], solved=False, tier=1)

def _bfs4(g: Grid, sr: int, sc: int) -> set:
    H, W = _sh(g)
    col = g[sr][sc]
    vis = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<H and 0<=nc<W and (nr,nc) not in vis and g[nr][nc]==col:
                vis.add((nr,nc)); q.append((nr,nc))
    return vis

def _connected_components(g: Grid, exclude_color: int = -1) -> list:
    H, W = _sh(g)
    vis = set()
    ccs = []
    for r in range(H):
        for c in range(W):
            if (r,c) not in vis and g[r][c] != exclude_color:
                cc = _bfs4(g, r, c)
                vis.update(cc)
                ccs.append((g[r][c], cc))
    return ccs


# ── 1. ParameterizedColorSwapSolver ─────────────────────────────────────────

@register_solver
class ParameterizedColorSwapSolver(BaseSolver):
    """
    Handles tasks where the color mapping is parameterized per pair.
    Pattern: one color (anchor) appears in ALL inputs and outputs unchanged;
    for each pair, there is one unique secondary color that maps to 0 (or bg),
    and anchor maps to the secondary color.
    Meta-rule is learned from training; test color is inferred from test input.
    Covers: f76d97a5 and similar parameterized remapping tasks.
    """

    @property
    def name(self) -> str:
        return "ParameterizedColorSwapSolver"

    @property
    def priority(self) -> int:
        return 60

    def _infer_anchor(self, train_pairs):
        """Find the color present in ALL training inputs."""
        if not train_pairs:
            return None
        sets = [set(v for r in p["input"] for v in r) for p in train_pairs]
        common = sets[0]
        for s in sets[1:]:
            common = common & s
        common.discard(0)
        return list(common) if common else None

    def _check_consistent_rule(self, train_pairs, anchor: int):
        """
        For each pair: anchor → X, X → 0, rest unchanged.
        Return the meta-rule description if consistent.
        """
        rules = []
        for p in train_pairs:
            inp, out = p["input"], p["output"]
            if inp is None or out is None:
                return None
            H, W = _sh(inp)
            if _sh(out) != (H, W):
                return None
            # Find which color X gets replaced (X → 0)
            inp_colors = {v for r in inp for v in r if v != 0}
            out_colors = {v for r in out for v in r if v != 0}
            lost = inp_colors - out_colors  # colors that disappeared
            out_colors - inp_colors  # colors that appeared
            # anchor should stay or transform
            if not lost:
                return None
            # Simple check: verify cell-by-cell consistency
            color_map = {}
            ok = True
            for r in range(H):
                for c in range(W):
                    iv, ov = inp[r][c], out[r][c]
                    if iv in color_map:
                        if color_map[iv] != ov:
                            ok = False; break
                    else:
                        color_map[iv] = ov
                if not ok:
                    break
            if not ok:
                return None
            rules.append(color_map)
        return rules

    def _find_per_pair_secondary(self, rules, anchor: int):
        """
        Each rule maps anchor → secondary, secondary → 0.
        Check this pattern holds for all rules.
        """
        anchor_targets = []
        for rule in rules:
            if anchor not in rule:
                return None
            target = rule[anchor]
            if target == anchor:
                return None
            anchor_targets.append(target)
        # The anchor always maps to a per-pair unique color
        # and that color maps to 0
        for i, rule in enumerate(rules):
            tgt = anchor_targets[i]
            if tgt not in rule or rule[tgt] != 0:
                return None
        return anchor_targets  # list of per-pair secondary colors

    def _apply_rule(self, grid: Grid, anchor: int, secondary: int) -> Grid:
        """Apply: anchor→secondary, secondary→0, rest unchanged."""
        H, W = _sh(grid)
        out = _clone(grid)
        for r in range(H):
            for c in range(W):
                v = grid[r][c]
                if v == anchor:
                    out[r][c] = secondary
                elif v == secondary:
                    out[r][c] = 0
        return out

    def _find_test_secondary(self, test_inp: Grid, anchor: int, training_secondaries: list) -> int | None:
        """Find the secondary color in test input (any color not 0, not anchor that appears in test)."""
        test_colors = {v for r in test_inp for v in r if v not in (0, anchor)}
        # Must be exactly one unknown color that matches training pattern
        candidates = test_colors - set(training_secondaries)
        if not candidates:
            # Fall back: any non-anchor non-zero color
            candidates = test_colors
        if len(candidates) == 1:
            return list(candidates)[0]
        # Try all training secondaries that appear in test
        in_test = [s for s in training_secondaries if s in test_colors]
        if len(in_test) == 1:
            return in_test[0]
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs or len(train_pairs) < 2:
            return None

        anchors = self._infer_anchor(train_pairs)
        if not anchors:
            return None

        for anchor in anchors:
            rules = self._check_consistent_rule(train_pairs, anchor)
            if not rules:
                continue
            secondaries = self._find_per_pair_secondary(rules, anchor)
            if not secondaries:
                continue

            # Verify all training pairs
            ok = True
            for i, p in enumerate(train_pairs):
                pred = self._apply_rule(p["input"], anchor, secondaries[i])
                if pred != p["output"]:
                    ok = False; break
            if not ok:
                continue

            # Apply to test
            preds = []
            for test_inp in ctx.test_inputs:
                sec = self._find_test_secondary(test_inp, anchor, secondaries)
                if sec is None:
                    ok = False; break
                preds.append(self._apply_rule(test_inp, anchor, sec))

            if ok and preds:
                return _result(ctx, preds, self.name)

        return None


# ── 2. RowPeriodicFillSolver ─────────────────────────────────────────────────

@register_solver
class RowPeriodicFillSolver(BaseSolver):
    """
    For tasks where the output rows follow a periodic pattern.
    Strategy: learn the periodic template from training outputs,
    then apply it to test inputs (where input has zeros where output has values).
    """

    @property
    def name(self) -> str:
        return "RowPeriodicFillSolver"

    @property
    def priority(self) -> int:
        return 61

    def _detect_row_period(self, grid: Grid) -> list | None:
        """Return the minimal repeating row sequence, or None."""
        H = len(grid)
        for period in range(1, H // 2 + 1):
            if H % period == 0:
                template = grid[:period]
                if all(grid[r] == template[r % period] for r in range(H)):
                    return template
        return None

    def _detect_col_period(self, row: list) -> list | None:
        """Return minimal repeating element sequence in a row."""
        W = len(row)
        for period in range(1, W // 2 + 1):
            if W % period == 0:
                tmpl = row[:period]
                if all(row[c] == tmpl[c % period] for c in range(W)):
                    return tmpl
        return None

    def _infer_template(self, train_pairs):
        """Find a consistent output row-periodic template."""
        templates = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            t = self._detect_row_period(p["output"])
            if t is None:
                return None
            templates.append(t)
        # All templates must be identical
        if len(set(tuple(tuple(r) for r in t) for t in templates)) == 1:
            return templates[0]
        return None

    def _apply_template(self, grid: Grid, template: list) -> Grid:
        H, W = _sh(grid)
        period = len(template)
        tmpl_w = len(template[0]) if template else 0
        out = _clone(grid)
        for r in range(H):
            for c in range(W):
                if grid[r][c] == 0 and c < tmpl_w:
                    out[r][c] = template[r % period][c]
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        template = self._infer_template(train_pairs)
        if template is None:
            return None

        # Verify: applying template to each training input gives training output
        for p in train_pairs:
            pred = self._apply_template(p["input"], template)
            if pred != p["output"]:
                return None

        preds = [self._apply_template(inp, template) for inp in ctx.test_inputs]
        return _result(ctx, preds, self.name)


# ── 3. ZeroFillByPeriodicTileSolver ──────────────────────────────────────────

@register_solver
class ZeroFillByPeriodicTileSolver(BaseSolver):
    """
    Detects a tile pattern in the non-zero part of the input,
    then fills zeros by extending the tile.
    """

    @property
    def name(self) -> str:
        return "ZeroFillByPeriodicTileSolver"

    @property
    def priority(self) -> int:
        return 62

    def _find_tile(self, grid: Grid, bg: int) -> Grid | None:
        """Find the smallest tile that tiles the non-zero region."""
        H, W = _sh(grid)
        # Collect non-bg cells
        non_bg = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
        if not non_bg:
            return None
        min_r = min(r for r,c in non_bg)
        min_c = min(c for r,c in non_bg)
        max_r = max(r for r,c in non_bg)
        max_c = max(c for r,c in non_bg)

        # Try tile sizes
        patch_h = max_r - min_r + 1
        patch_w = max_c - min_c + 1

        for th in range(1, patch_h + 1):
            for tw in range(1, patch_w + 1):
                tile = [[bg] * tw for _ in range(th)]
                valid = True
                for r, c in non_bg:
                    tr, tc = (r - min_r) % th, (c - min_c) % tw
                    v = grid[r][c]
                    if tile[tr][tc] == bg:
                        tile[tr][tc] = v
                    elif tile[tr][tc] != v:
                        valid = False; break
                if valid:
                    return tile
        return None

    def _apply_tile(self, grid: Grid, tile: Grid, bg: int) -> Grid:
        H, W = _sh(grid)
        th, tw = _sh(tile)
        out = _clone(grid)
        for r in range(H):
            for c in range(W):
                if grid[r][c] == bg:
                    tile_val = tile[r % th][c % tw]
                    if tile_val != bg:
                        out[r][c] = tile_val
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        tiles = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            H, W = _sh(p["input"])
            Ho, Wo = _sh(p["output"])
            if H != Ho or W != Wo:
                return None
            bg = _bg(p["input"])
            tile = self._find_tile(p["input"], bg)
            if tile is None:
                return None
            tiles.append(tile)

        # All tiles must be identical
        if len(set(tuple(tuple(r) for r in t) for t in tiles)) != 1:
            return None

        tile = tiles[0]
        # Verify
        for p in train_pairs:
            bg = _bg(p["input"])
            pred = self._apply_tile(p["input"], tile, bg)
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            preds.append(self._apply_tile(inp, tile, bg))
        return _result(ctx, preds, self.name)


# ── 4. TileWithFlipSolver ─────────────────────────────────────────────────────

@register_solver
class TileWithFlipSolver(BaseSolver):
    """
    Input is a small tile; output is a tiled version with alternating
    h-flip, v-flip, or both.
    Pattern: output = tile repeated NxM times with alternating mirroring.
    """

    @property
    def name(self) -> str:
        return "TileWithFlipSolver"

    @property
    def priority(self) -> int:
        return 63

    def _hflip(self, tile):
        return [list(reversed(r)) for r in tile]

    def _vflip(self, tile):
        return list(reversed([list(r) for r in tile]))

    def _build_tiled(self, tile, reps_r, reps_c, flip_h, flip_v):
        th, tw = len(tile), len(tile[0])
        H, W = th * reps_r, tw * reps_c
        out = [[0]*W for _ in range(H)]
        for tr in range(reps_r):
            for tc in range(reps_c):
                t = tile
                if flip_h and tc % 2 == 1:
                    t = self._hflip(t)
                if flip_v and tr % 2 == 1:
                    t = self._vflip(t)
                if flip_h and flip_v and tr % 2 == 1 and tc % 2 == 1:
                    t = tile  # both flipped = original
                for r in range(th):
                    for c in range(tw):
                        out[tr*th+r][tc*tw+c] = t[r][c]
        return out

    def _try_tile_and_flip(self, inp, out):
        ih, iw = _sh(inp)
        oh, ow = _sh(out)
        if oh < ih or ow < iw:
            return None
        if oh % ih != 0 or ow % iw != 0:
            return None
        rr, rc = oh // ih, ow // iw
        # Try all flip combinations
        for fh in (False, True):
            for fv in (False, True):
                tiled = self._build_tiled(inp, rr, rc, fh, fv)
                if tiled == out:
                    return (rr, rc, fh, fv)
        return None

    def _infer_reps(self, inp, train_pairs):
        """Infer reps for test when training reps vary.
        Tries: num_distinct_colors, input_H, input_W."""
        ih, iw = _sh(inp)
        n_colors = len({v for row in inp for v in row})
        for rr_cand in [n_colors, ih, iw]:
            ok = True
            for p in train_pairs:
                pi_h, pi_w = _sh(p["input"])
                po_h, po_w = _sh(p["output"])
                p_n_colors = len({v for row in p["input"] for v in row})
                p_rr_h = pi_h
                if rr_cand == n_colors:
                    expected_rr = p_n_colors
                elif rr_cand == ih:
                    expected_rr = p_rr_h
                else:
                    expected_rr = pi_w
                if po_h != pi_h * expected_rr or po_w != pi_w * expected_rr:
                    ok = False; break
            if ok:
                # compute actual rr/rc for this test input
                if rr_cand == n_colors:
                    return n_colors, n_colors
                elif rr_cand == ih:
                    return ih, ih
                else:
                    return iw, iw
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        params = None
        train_reps = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            result = self._try_tile_and_flip(p["input"], p["output"])
            if result is None:
                return None
            rr, rc, fh, fv = result
            train_reps.append((rr, rc))
            if params is None:
                params = (fh, fv)
            elif params != (fh, fv):
                return None

        fh, fv = params
        reps_fixed = len(set(train_reps)) == 1  # all same reps across training

        preds = []
        for inp in ctx.test_inputs:
            if reps_fixed:
                rr, rc = train_reps[0]
            else:
                inferred = self._infer_reps(inp, train_pairs)
                if inferred is None:
                    return None
                rr, rc = inferred
            preds.append(self._build_tiled(inp, rr, rc, fh, fv))

        return _result(ctx, preds, self.name)



# ── 5. ObjectRecolorByShapeSolver ─────────────────────────────────────────────

@register_solver
class ObjectRecolorByShapeSolver(BaseSolver):
    """
    Detect connected components (objects), classify by shape features
    (bounding-box aspect ratio, is-square, size rank, etc.),
    then recolor each class consistently.
    """

    @property
    def name(self) -> str:
        return "ObjectRecolorByShapeSolver"

    @property
    def priority(self) -> int:
        return 64

    def _shape_key(self, cells: set) -> tuple:
        rows = [r for r,c in cells]
        cols = [c for r,c in cells]
        h = max(rows)-min(rows)+1
        w = max(cols)-min(cols)+1
        area = len(cells)
        is_rect = (area == h*w)
        is_sq = is_rect and (h == w)
        return (h, w, area, is_rect, is_sq)

    def _classify_objects(self, grid: Grid, bg: int):
        H, W = _sh(grid)
        vis = set()
        objs = []
        for r in range(H):
            for c in range(W):
                if (r,c) not in vis and grid[r][c] != bg:
                    cc = _bfs4(grid, r, c)
                    vis.update(cc)
                    objs.append((grid[r][c], cc, self._shape_key(cc)))
        return objs

    def _infer_shape_to_color_map(self, inp: Grid, out: Grid, bg: int):
        """For each object in inp, find what color it becomes in out."""
        objs = self._classify_objects(inp, bg)
        shape_map = {}
        for src_color, cells, sk in objs:
            # Sample output color at the cells
            out_colors = Counter(out[r][c] for r,c in cells if r < len(out) and c < len(out[r]))
            if not out_colors:
                return None
            tgt = out_colors.most_common(1)[0][0]
            if sk in shape_map and shape_map[sk] != tgt:
                return None
            shape_map[sk] = tgt
        return shape_map

    def _apply_shape_map(self, grid: Grid, shape_map: dict, bg: int) -> Grid | None:
        objs = self._classify_objects(grid, bg)
        out = _clone(grid)
        for src_color, cells, sk in objs:
            tgt = shape_map.get(sk)
            if tgt is None:
                return None  # unknown shape — don't guess
            for r, c in cells:
                out[r][c] = tgt
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        shape_maps = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            H, W = _sh(p["input"])
            if _sh(p["output"]) != (H, W):
                return None
            bg = _bg(p["input"])
            sm = self._infer_shape_to_color_map(p["input"], p["output"], bg)
            if sm is None:
                return None
            shape_maps.append(sm)

        # Merge shape maps (must be consistent)
        merged = {}
        for sm in shape_maps:
            for k, v in sm.items():
                if k in merged and merged[k] != v:
                    return None
                merged[k] = v

        if not merged:
            return None

        # Verify
        for p in train_pairs:
            bg = _bg(p["input"])
            pred = self._apply_shape_map(p["input"], merged, bg)
            if pred is None or pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            pred = self._apply_shape_map(inp, merged, _bg(inp))
            if pred is None:
                return None
            preds.append(pred)
        return _result(ctx, preds, self.name)


# ── 6. SymmetryCompleteSolver ─────────────────────────────────────────────────

@register_solver
class SymmetryCompleteSolver(BaseSolver):
    """
    Input is partially filled; output completes it to be symmetric.
    Supports: h-symmetry, v-symmetry, rot180, rot90, diagonal.
    Non-zero cells in input are preserved; zeros are filled from symmetry.
    """

    @property
    def name(self) -> str:
        return "SymmetryCompleteSolver"

    @property
    def priority(self) -> int:
        return 65

    def _complete_sym(self, grid: Grid, sym: str) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        bg = _bg(grid)
        changed = True
        iters = 0
        while changed and iters < 10:
            changed = False
            iters += 1
            for r in range(H):
                for c in range(W):
                    v = out[r][c]
                    mr, mc = None, None
                    if sym == "h":   mr, mc = r, W-1-c
                    elif sym == "v": mr, mc = H-1-r, c
                    elif sym == "r180": mr, mc = H-1-r, W-1-c
                    elif sym == "diag" and H==W: mr, mc = c, r
                    elif sym == "adiag" and H==W: mr, mc = W-1-c, H-1-r
                    else: continue
                    mv = out[mr][mc]
                    if v == bg and mv != bg:
                        out[r][c] = mv; changed = True
                    elif mv == bg and v != bg:
                        out[mr][mc] = v; changed = True
        return out

    def _check_sym_consistent(self, train_pairs, sym: str) -> bool:
        for p in train_pairs:
            if p["output"] is None:
                return False
            pred = self._complete_sym(p["input"], sym)
            if pred != p["output"]:
                return False
        return True

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        # Only accept same-size
        for p in train_pairs:
            if p["output"] is None:
                return None
            if _sh(p["input"]) != _sh(p["output"]):
                return None

        for sym in ("h", "v", "r180", "diag", "adiag"):
            if self._check_sym_consistent(train_pairs, sym):
                preds = [self._complete_sym(inp, sym) for inp in ctx.test_inputs]
                return _result(ctx, preds, self.name)

        return None


# ── 7. SelectSubgridByMarkerSolver ────────────────────────────────────────────

@register_solver
class SelectSubgridByMarkerSolver(BaseSolver):
    """
    Input has multiple equal-size sub-blocks arranged in a grid.
    One sub-block is "selected" by a marker color in the border/edge.
    Output is the selected sub-block.
    """

    @property
    def name(self) -> str:
        return "SelectSubgridByMarkerSolver"

    @property
    def priority(self) -> int:
        return 66

    def _find_separator_lines(self, grid: Grid):
        """Find rows/cols that are all same color (separator lines)."""
        H, W = _sh(grid)
        sep_rows = []
        sep_cols = []
        for r in range(H):
            if len(set(grid[r])) == 1:
                sep_rows.append(r)
        for c in range(W):
            col = [grid[r][c] for r in range(H)]
            if len(set(col)) == 1:
                sep_cols.append(c)
        return sep_rows, sep_cols

    def _split_by_separators(self, grid: Grid, sep_rows, sep_cols):
        """Split grid into rectangular blocks at separator positions."""
        H, W = _sh(grid)
        row_bounds = []
        prev = 0
        for sr in sep_rows:
            if sr > prev:
                row_bounds.append((prev, sr))
            prev = sr + 1
        if prev < H:
            row_bounds.append((prev, H))

        col_bounds = []
        prev = 0
        for sc in sep_cols:
            if sc > prev:
                col_bounds.append((prev, sc))
            prev = sc + 1
        if prev < W:
            col_bounds.append((prev, W))

        blocks = []
        for (r0, r1) in row_bounds:
            for (c0, c1) in col_bounds:
                block = [grid[r][c0:c1] for r in range(r0, r1)]
                blocks.append(((r0, c0), block))
        return blocks

    def _select_block_index(self, grid: Grid, blocks: list, sep_rows, sep_cols):
        """Find which block index is indicated by a unique marker."""
        H, W = _sh(grid)
        # Find the separator color
        sep_colors = set()
        for sr in sep_rows:
            sep_colors.add(grid[sr][0])
        for sc in sep_cols:
            sep_colors.add(grid[0][sc])

        # Find marker: a unique non-bg, non-sep color in one of the blocks
        block_nonbg = []
        bg = _bg(grid)
        for i, ((r0,c0), blk) in enumerate(blocks):
            colors = {v for r in blk for v in r if v != bg and v not in sep_colors}
            block_nonbg.append(colors)

        # Find block with unique non-bg color
        all_colors = []
        for colors in block_nonbg:
            all_colors.extend(colors)
        color_count = Counter(all_colors)

        for i, colors in enumerate(block_nonbg):
            unique = [c for c in colors if color_count[c] == 1]
            if unique:
                return i
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        # Learn: which block index produces the output for each training pair
        learned_selectors = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            sep_r, sep_c = self._find_separator_lines(p["input"])
            if not sep_r and not sep_c:
                return None
            blocks = self._split_by_separators(p["input"], sep_r, sep_c)
            if len(blocks) < 2:
                return None
            # Find which block matches the output
            matched_idx = None
            for i, ((r0,c0), blk) in enumerate(blocks):
                if blk == p["output"]:
                    matched_idx = i
                    break
            if matched_idx is None:
                return None
            # Verify this is the unique-marker block
            marker_idx = self._select_block_index(p["input"], blocks, sep_r, sep_c)
            if marker_idx != matched_idx:
                return None
            learned_selectors.append(matched_idx)

        # Apply to test inputs
        preds = []
        for inp in ctx.test_inputs:
            sep_r, sep_c = self._find_separator_lines(inp)
            blocks = self._split_by_separators(inp, sep_r, sep_c)
            if len(blocks) < 2:
                return None
            idx = self._select_block_index(inp, blocks, sep_r, sep_c)
            if idx is None:
                return None
            preds.append(blocks[idx][1])

        return _result(ctx, preds, self.name)


# ── 8. GridRowColSortSolver ────────────────────────────────────────────────────

@register_solver
class GridRowColSortSolver(BaseSolver):
    """
    Output = input rows (or cols) sorted by some ordering
    (e.g., by number of non-zero cells, by dominant color, by sum).
    """

    @property
    def name(self) -> str:
        return "GridRowColSortSolver"

    @property
    def priority(self) -> int:
        return 67

    def _row_key(self, row, mode):
        if mode == "nonzero_count":
            return sum(1 for v in row if v != 0)
        elif mode == "sum":
            return sum(row)
        elif mode == "max_color":
            c = Counter(v for v in row if v != 0)
            return c.most_common(1)[0][0] if c else 0
        elif mode == "first_nonzero":
            for v in row:
                if v != 0: return v
            return 0
        return 0

    def _try_sort_rows(self, inp, out, mode, reverse):
        sorted_rows = sorted(inp, key=lambda r: self._row_key(r, mode), reverse=reverse)
        return sorted_rows == out

    def _try_sort_cols(self, inp, out, mode, reverse):
        H, W = _sh(inp)
        cols = [[inp[r][c] for r in range(H)] for c in range(W)]
        sorted_cols = sorted(cols, key=lambda col: self._row_key(col, mode), reverse=reverse)
        reconstructed = [[sorted_cols[c][r] for c in range(W)] for r in range(H)]
        return reconstructed == out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None

        modes = ["nonzero_count", "sum", "max_color", "first_nonzero"]

        for mode in modes:
            for reverse in (False, True):
                for axis in ("rows", "cols"):
                    ok = True
                    for p in train_pairs:
                        if axis == "rows":
                            if not self._try_sort_rows(p["input"], p["output"], mode, reverse):
                                ok = False; break
                        else:
                            if not self._try_sort_cols(p["input"], p["output"], mode, reverse):
                                ok = False; break
                    if ok:
                        preds = []
                        for inp in ctx.test_inputs:
                            H, W = _sh(inp)
                            if axis == "rows":
                                preds.append(sorted(inp, key=lambda r: self._row_key(r, mode), reverse=reverse))
                            else:
                                cols = [[inp[r][c] for r in range(H)] for c in range(W)]
                                sc = sorted(cols, key=lambda col: self._row_key(col, mode), reverse=reverse)
                                preds.append([[sc[c][r] for c in range(W)] for r in range(H)])
                        return _result(ctx, preds, self.name)

        return None


# ── 9. RepeatRowColPatternSolver ──────────────────────────────────────────────

@register_solver
class RepeatRowColPatternSolver(BaseSolver):
    """
    Output = a row/column pattern from input repeated to fill the grid.
    E.g., a single non-zero row in input is repeated to fill all rows.
    """

    @property
    def name(self) -> str:
        return "RepeatRowColPatternSolver"

    @property
    def priority(self) -> int:
        return 68

    def _find_template_row(self, grid: Grid, bg: int):
        """Find single non-bg row to use as template."""
        candidates = [row for row in grid if any(v != bg for v in row)]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _find_template_col(self, grid: Grid, bg: int):
        H, W = _sh(grid)
        cols = [[grid[r][c] for r in range(H)] for c in range(W)]
        candidates = [col for col in cols if any(v != bg for v in col)]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _apply_row_template(self, grid: Grid, template, bg: int) -> Grid:
        H = len(grid)
        return [list(template) for _ in range(H)]

    def _apply_col_template(self, grid: Grid, template, bg: int) -> Grid:
        H, W = _sh(grid)
        return [[template[r] for _ in range(W)] for r in range(H)]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None

        # Try row template
        templates = []
        for p in train_pairs:
            bg = _bg(p["input"])
            tmpl = self._find_template_row(p["input"], bg)
            if tmpl is None:
                break
            pred = self._apply_row_template(p["input"], tmpl, bg)
            if pred != p["output"]:
                break
            templates.append(tmpl)
        else:
            if templates:
                preds = []
                for inp in ctx.test_inputs:
                    bg = _bg(inp)
                    tmpl = self._find_template_row(inp, bg)
                    if tmpl is None:
                        return None
                    preds.append(self._apply_row_template(inp, tmpl, bg))
                return _result(ctx, preds, self.name)

        # Try col template
        for p in train_pairs:
            bg = _bg(p["input"])
            tmpl = self._find_template_col(p["input"], bg)
            if tmpl is None:
                return None
            pred = self._apply_col_template(p["input"], tmpl, bg)
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            tmpl = self._find_template_col(inp, bg)
            if tmpl is None:
                return None
            preds.append(self._apply_col_template(inp, tmpl, bg))
        return _result(ctx, preds, self.name)


# ── 10. ColorCountToOutputSolver ─────────────────────────────────────────────

@register_solver
class ColorCountToOutputSolver(BaseSolver):
    """
    Output = small grid encoding color counts from input.
    E.g., count of each color → single row or column.
    """

    @property
    def name(self) -> str:
        return "ColorCountToOutputSolver"

    @property
    def priority(self) -> int:
        return 69

    def _count_colors(self, grid: Grid) -> dict:
        return Counter(v for r in grid for v in r)

    def _encode_counts(self, grid: Grid, mode: str) -> Grid | None:
        counts = self._count_colors(grid)
        bg = _bg(grid)
        nonbg = {k: v for k, v in counts.items() if k != bg}
        if not nonbg:
            return None

        sorted(nonbg.keys())
        sorted_by_count = sorted(nonbg.items(), key=lambda x: -x[1])

        if mode == "single_cell_dominant":
            # Output = 1x1 grid with most frequent non-bg color
            return [[sorted_by_count[0][0]]]

        elif mode == "row_of_colors_by_count":
            # Output = 1 row, each cell = color, sorted by count desc
            return [[c for c, n in sorted_by_count]]

        elif mode == "col_of_colors_by_count":
            return [[[c] for c, n in sorted_by_count]]

        elif mode == "single_cell_count":
            # Output = 1x1 with count of non-bg cells
            total = sum(nonbg.values())
            return [[total % 10]]  # mod 10 as color

        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None:
                return None

        for mode in ("single_cell_dominant", "row_of_colors_by_count"):
            ok = True
            for p in train_pairs:
                pred = self._encode_counts(p["input"], mode)
                if pred != p["output"]:
                    ok = False; break
            if ok:
                preds = [self._encode_counts(inp, mode) for inp in ctx.test_inputs]
                if all(p is not None for p in preds):
                    return _result(ctx, preds, self.name)

        return None


# ── 11. GravityAllDirectionsSolver ────────────────────────────────────────────

@register_solver
class GravityAllDirectionsSolver(BaseSolver):
    """
    Objects fall towards one of the 4 borders (up/down/left/right).
    Each object moves as far as possible in the direction.
    Also handles "gravity to bg" (compaction).
    """

    @property
    def name(self) -> str:
        return "GravityAllDirectionsSolver"

    @property
    def priority(self) -> int:
        return 70

    def _gravity(self, grid: Grid, direction: str, bg: int) -> Grid:
        H, W = _sh(grid)
        out = [[bg]*W for _ in range(H)]

        if direction == "down":
            for c in range(W):
                col = [grid[r][c] for r in range(H)]
                non_bg = [v for v in col if v != bg]
                new_col = [bg]*(H-len(non_bg)) + non_bg
                for r in range(H): out[r][c] = new_col[r]
        elif direction == "up":
            for c in range(W):
                col = [grid[r][c] for r in range(H)]
                non_bg = [v for v in col if v != bg]
                new_col = non_bg + [bg]*(H-len(non_bg))
                for r in range(H): out[r][c] = new_col[r]
        elif direction == "right":
            for r in range(H):
                row = grid[r]
                non_bg = [v for v in row if v != bg]
                new_row = [bg]*(W-len(non_bg)) + non_bg
                out[r] = new_row
        elif direction == "left":
            for r in range(H):
                row = grid[r]
                non_bg = [v for v in row if v != bg]
                new_row = non_bg + [bg]*(W-len(non_bg))
                out[r] = new_row
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None

        for direction in ("down", "up", "left", "right"):
            ok = True
            for p in train_pairs:
                bg = _bg(p["input"])
                pred = self._gravity(p["input"], direction, bg)
                if pred != p["output"]:
                    ok = False; break
            if ok:
                preds = []
                for inp in ctx.test_inputs:
                    bg = _bg(inp)
                    preds.append(self._gravity(inp, direction, bg))
                return _result(ctx, preds, self.name)

        return None


# ── 12. FillEnclosedHolesSolver ──────────────────────────────────────────────

@register_solver
class FillEnclosedHolesSolver(BaseSolver):
    """
    Fill all bg-color holes that are fully enclosed by non-bg color.
    Fill color is learned from training pairs.
    """

    @property
    def name(self) -> str:
        return "FillEnclosedHolesSolver"

    @property
    def priority(self) -> int:
        return 71

    def _get_enclosed_bg(self, grid: Grid, bg: int) -> set:
        H, W = _sh(grid)
        # BFS from border to find reachable bg cells
        reachable = set()
        q = deque()
        for r in range(H):
            for c in [0, W-1]:
                if grid[r][c] == bg and (r,c) not in reachable:
                    reachable.add((r,c)); q.append((r,c))
        for c in range(W):
            for r in [0, H-1]:
                if grid[r][c] == bg and (r,c) not in reachable:
                    reachable.add((r,c)); q.append((r,c))
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and (nr,nc) not in reachable and grid[nr][nc] == bg:
                    reachable.add((nr,nc)); q.append((nr,nc))
        # Enclosed = bg cells not reachable from border
        all_bg = {(r,c) for r in range(H) for c in range(W) if grid[r][c] == bg}
        return all_bg - reachable

    def _infer_fill_color(self, inp: Grid, out: Grid, bg: int) -> int | None:
        enclosed = self._get_enclosed_bg(inp, bg)
        if not enclosed:
            return None
        # What color fills the enclosed cells?
        colors = Counter(out[r][c] for r,c in enclosed if r < len(out) and c < len(out[r]))
        if not colors:
            return None
        tgt = colors.most_common(1)[0][0]
        return tgt

    def _apply_fill(self, grid: Grid, bg: int, fill_color: int) -> Grid:
        out = _clone(grid)
        enclosed = self._get_enclosed_bg(grid, bg)
        for r, c in enclosed:
            out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        fill_colors = []
        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None
            bg = _bg(p["input"])
            fc = self._infer_fill_color(p["input"], p["output"], bg)
            if fc is None:
                return None
            fill_colors.append(fc)

        if len(set(fill_colors)) != 1:
            return None

        fill_color = fill_colors[0]

        # Verify
        for p in train_pairs:
            bg = _bg(p["input"])
            pred = self._apply_fill(p["input"], bg, fill_color)
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            preds.append(self._apply_fill(inp, bg, fill_color))
        return _result(ctx, preds, self.name)


# ── 13. PerPairObjectMoverSolver ─────────────────────────────────────────────

@register_solver
class PerPairObjectMoverSolver(BaseSolver):
    """
    All non-bg objects move by the same displacement (dr, dc) per pair.
    The displacement is learned from training and applied to test.
    If each pair has same displacement: global; if varies: learn from test context.
    """

    @property
    def name(self) -> str:
        return "PerPairObjectMoverSolver"

    @property
    def priority(self) -> int:
        return 72

    def _find_displacement(self, inp: Grid, out: Grid, bg: int) -> tuple | None:
        H, W = _sh(inp)
        if _sh(out) != (H, W):
            return None
        # Find a non-bg cell and check its displacement in out
        disps = set()
        for r in range(H):
            for c in range(W):
                if inp[r][c] != bg:
                    v = inp[r][c]
                    # Find where v appears in out (not in inp position)
                    for dr in range(-H, H):
                        for dc in range(-W, W):
                            nr, nc = r+dr, c+dc
                            if 0<=nr<H and 0<=nc<W and out[nr][nc] == v:
                                disps.add((dr, dc))
        if len(disps) == 1:
            dr, dc = list(disps)[0]
            # Verify: applying this displacement to all non-bg cells gives out
            pred = [[bg]*W for _ in range(H)]
            ok = True
            for r in range(H):
                for c in range(W):
                    if inp[r][c] != bg:
                        nr, nc = r+dr, c+dc
                        if not (0<=nr<H and 0<=nc<W):
                            ok = False; break
                        pred[nr][nc] = inp[r][c]
                if not ok: break
            if ok and pred == out:
                return (dr, dc)
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        disps = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            bg = _bg(p["input"])
            d = self._find_displacement(p["input"], p["output"], bg)
            if d is None:
                return None
            disps.append(d)

        if len(set(disps)) != 1:
            return None

        dr, dc = disps[0]

        preds = []
        for inp in ctx.test_inputs:
            H, W = _sh(inp)
            bg = _bg(inp)
            out = [[bg]*W for _ in range(H)]
            ok = True
            for r in range(H):
                for c in range(W):
                    if inp[r][c] != bg:
                        nr, nc = r+dr, c+dc
                        if not (0<=nr<H and 0<=nc<W):
                            ok = False; break
                        out[nr][nc] = inp[r][c]
                if not ok: break
            if not ok:
                return None
            preds.append(out)

        return _result(ctx, preds, self.name)


# ── 14. ZeroRunFillSolver ──────────────────────────────────────────────────────

@register_solver
class ZeroRunFillSolver(BaseSolver):
    """
    Fills runs of zeros between non-zero values.
    Strategy: each zero-run takes the value of the adjacent non-zero cell
    (left/right/up/down or nearest).
    """

    @property
    def name(self) -> str:
        return "ZeroRunFillSolver"

    @property
    def priority(self) -> int:
        return 73

    def _fill_rows_nearest(self, grid: Grid, mode: str) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        for r in range(H):
            if mode == "left":
                last = 0
                for c in range(W):
                    if grid[r][c] != 0:
                        last = grid[r][c]
                    else:
                        out[r][c] = last
            elif mode == "right":
                last = 0
                for c in range(W-1, -1, -1):
                    if grid[r][c] != 0:
                        last = grid[r][c]
                    else:
                        out[r][c] = last
            elif mode == "both":
                # Fill from both sides (nearest non-zero)
                left = [0]*W
                last = 0
                for c in range(W):
                    if grid[r][c] != 0: last = grid[r][c]
                    left[c] = last
                right = [0]*W
                last = 0
                for c in range(W-1, -1, -1):
                    if grid[r][c] != 0: last = grid[r][c]
                    right[c] = last
                for c in range(W):
                    if grid[r][c] == 0:
                        out[r][c] = left[c] if left[c] != 0 else right[c]
        return out

    def _fill_cols_nearest(self, grid: Grid, mode: str) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        for c in range(W):
            if mode == "down":
                last = 0
                for r in range(H):
                    if grid[r][c] != 0: last = grid[r][c]
                    elif last != 0: out[r][c] = last
            elif mode == "up":
                last = 0
                for r in range(H-1, -1, -1):
                    if grid[r][c] != 0: last = grid[r][c]
                    elif last != 0: out[r][c] = last
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None

        for mode_fn, modes in [
            (self._fill_rows_nearest, ["left", "right", "both"]),
            (self._fill_cols_nearest, ["down", "up"])
        ]:
            for mode in modes:
                ok = True
                for p in train_pairs:
                    pred = mode_fn(p["input"], mode)
                    if pred != p["output"]:
                        ok = False; break
                if ok:
                    preds = [mode_fn(inp, mode) for inp in ctx.test_inputs]
                    return _result(ctx, preds, self.name)

        return None


# ── 15. ColorBandSolver ────────────────────────────────────────────────────────

@register_solver
class ColorBandSolver(BaseSolver):
    """
    Input has colored horizontal/vertical bands; output assigns cells
    to the band they belong to. Or input has two intersecting colored
    lines and output colors the intersection.
    """

    @property
    def name(self) -> str:
        return "ColorBandSolver"

    @property
    def priority(self) -> int:
        return 74

    def _find_bands(self, grid: Grid):
        H, W = _sh(grid)
        bg = _bg(grid)
        # Find rows/cols that are fully non-bg
        h_bands = [(r, grid[r][0]) for r in range(H) if all(v == grid[r][0] != bg for v in grid[r])]
        v_bands = [(c, grid[0][c]) for c in range(W) if all(grid[r][c] == grid[0][c] != bg for r in range(H))]
        return h_bands, v_bands

    def _fill_intersection(self, grid: Grid, h_bands, v_bands, fill_color: int) -> Grid:
        out = _clone(grid)
        h_rows = {r for r, _ in h_bands}
        v_cols = {c for c, _ in v_bands}
        for r in h_rows:
            for c in v_cols:
                out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None

        fill_colors = []
        for p in train_pairs:
            h_bands, v_bands = self._find_bands(p["input"])
            if not h_bands or not v_bands:
                return None
            # Find fill color: what color appears at intersections in output?
            h_rows = {r for r, _ in h_bands}
            v_cols = {c for c, _ in v_bands}
            bg = _bg(p["input"])
            intersection_colors = Counter(
                p["output"][r][c] for r in h_rows for c in v_cols
                if r < len(p["output"]) and c < len(p["output"][r])
            )
            if not intersection_colors:
                return None
            fc = intersection_colors.most_common(1)[0][0]
            if fc == bg:
                return None
            fill_colors.append(fc)

        if len(set(fill_colors)) != 1:
            return None

        fill_color = fill_colors[0]

        # Verify
        for p in train_pairs:
            h_bands, v_bands = self._find_bands(p["input"])
            pred = self._fill_intersection(p["input"], h_bands, v_bands, fill_color)
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            h_bands, v_bands = self._find_bands(inp)
            if not h_bands or not v_bands:
                return None
            preds.append(self._fill_intersection(inp, h_bands, v_bands, fill_color))
        return _result(ctx, preds, self.name)


# ── 16. PatternCompletionBySeedSolver ──────────────────────────────────────────

@register_solver
class PatternCompletionBySeedSolver(BaseSolver):
    """
    Input has a "seed" pattern (non-zero cells) and output completes it
    by repeating the seed pattern from the topmost-leftmost non-zero cell.
    Works by finding the tile shape from the spacing of seed cells.
    """

    @property
    def name(self) -> str:
        return "PatternCompletionBySeedSolver"

    @property
    def priority(self) -> int:
        return 75

    def _find_seed_offset(self, grid: Grid, bg: int):
        """Find min row and col of non-bg cells."""
        cells = [(r,c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != bg]
        if not cells:
            return None, None
        return min(r for r,c in cells), min(c for r,c in cells)

    def _extract_seed(self, grid: Grid, bg: int):
        """Extract the minimal bounding box containing all non-bg cells."""
        cells = [(r,c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] != bg]
        if not cells:
            return None
        r0 = min(r for r,c in cells)
        r1 = max(r for r,c in cells)+1
        c0 = min(c for r,c in cells)
        c1 = max(c for r,c in cells)+1
        return [grid[r][c0:c1] for r in range(r0, r1)]

    def _tile_with_seed(self, grid: Grid, seed: Grid, bg: int) -> Grid:
        H, W = _sh(grid)
        sh, sw = _sh(seed)
        if sh == 0 or sw == 0:
            return _clone(grid)
        out = _clone(grid)
        # Find offset of first non-bg cell in input
        r0, c0 = self._find_seed_offset(grid, bg)
        if r0 is None:
            return _clone(grid)
        # Fill all positions that align with the seed
        for r in range(H):
            for c in range(W):
                if grid[r][c] == bg:
                    sr = (r - r0) % sh
                    sc = (c - c0) % sw
                    tile_val = seed[sr][sc]
                    if tile_val != bg:
                        out[r][c] = tile_val
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None

        seeds = []
        for p in train_pairs:
            bg = _bg(p["input"])
            seed = self._extract_seed(p["output"], bg)
            if seed is None:
                return None
            seeds.append(seed)

        # Seeds must be consistent
        if len(set(tuple(tuple(r) for r in s) for s in seeds)) != 1:
            return None

        seed = seeds[0]

        # Verify
        for p in train_pairs:
            bg = _bg(p["input"])
            pred = self._tile_with_seed(p["input"], seed, bg)
            if pred != p["output"]:
                return None

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            preds.append(self._tile_with_seed(inp, seed, bg))
        return _result(ctx, preds, self.name)


# ── 17. ObjectCountOutputSolver ───────────────────────────────────────────────

@register_solver
class ObjectCountOutputSolver(BaseSolver):
    """
    Output encodes the count of objects (connected components) in input.
    E.g., input has 3 distinct objects → output is 3x3 or 1x3, etc.
    """

    @property
    def name(self) -> str:
        return "ObjectCountOutputSolver"

    @property
    def priority(self) -> int:
        return 76

    def _count_objects(self, grid: Grid) -> int:
        bg = _bg(grid)
        H, W = _sh(grid)
        vis = set()
        count = 0
        for r in range(H):
            for c in range(W):
                if (r,c) not in vis and grid[r][c] != bg:
                    cc = _bfs4(grid, r, c)
                    vis.update(cc)
                    count += 1
        return count

    def _count_to_grid(self, n: int, template_out: Grid) -> Grid:
        """Build output grid from count n, matching output template shape."""
        if not template_out:
            return None
        H, W = _sh(template_out)
        # Fill with n using a specific value from the template
        vals = [v for r in template_out for v in r if v != 0]
        fill_val = Counter(vals).most_common(1)[0][0] if vals else 1
        _bg(template_out)

        # If output is a row of n filled cells
        if H == 1 and W == n:
            return [[fill_val]*n]
        if H == n and W == 1:
            return [[fill_val] for _ in range(n)]
        if H == n and W == n:
            return [[fill_val]*n for _ in range(n)]
        # n x something
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        counts = []
        for p in train_pairs:
            if p["output"] is None:
                return None
            cnt = self._count_objects(p["input"])
            counts.append(cnt)

        # Check if count determines output shape/content
        for p, cnt in zip(train_pairs, counts):
            out = p["output"]
            oh, ow = _sh(out)
            if oh != cnt and ow != cnt and oh*ow != cnt:
                return None

        # Guard: fill color must be CONSISTENT across all training outputs
        # (otherwise we cannot infer test fill color from count alone)
        fill_vals = set()
        for p in train_pairs:
            vals = [v for row in p["output"] for v in row if v != 0]
            if vals:
                fill_vals.add(Counter(vals).most_common(1)[0][0])
        if len(fill_vals) != 1:
            return None

        # Verify exact reconstruction
        for p, cnt in zip(train_pairs, counts):
            pred = self._count_to_grid(cnt, p["output"])
            if pred != p["output"]:
                return None

        preds = []
        for i, inp in enumerate(ctx.test_inputs):
            cnt = self._count_objects(inp)
            pred = self._count_to_grid(cnt, train_pairs[0]["output"])
            if pred is None:
                return None
            preds.append(pred)

        return _result(ctx, preds, self.name)


# ── 18. LinesFromObjectsSolver ────────────────────────────────────────────────

@register_solver
class LinesFromObjectsSolver(BaseSolver):
    """
    Each non-bg cell shoots a ray in a fixed direction until hitting another
    non-bg cell or the border. The ray is drawn in the grid.
    """

    @property
    def name(self) -> str:
        return "LinesFromObjectsSolver"

    @property
    def priority(self) -> int:
        return 77

    def _shoot_rays(self, grid: Grid, direction: str, bg: int, ray_color: int) -> Grid:
        H, W = _sh(grid)
        out = _clone(grid)
        dr_map = {"up": (-1,0), "down": (1,0), "left": (0,-1), "right": (0,1)}
        dr, dc = dr_map[direction]
        for r in range(H):
            for c in range(W):
                if grid[r][c] != bg:
                    # Shoot ray from this cell
                    nr, nc = r+dr, c+dc
                    while 0<=nr<H and 0<=nc<W and grid[nr][nc] == bg:
                        out[nr][nc] = ray_color
                        nr += dr; nc += dc
        return out

    def _infer_params(self, inp: Grid, out: Grid, bg: int):
        """Find direction and ray color."""
        # Find which cells were added (were bg in inp, non-bg in out)
        H, W = _sh(inp)
        added = [(r, c, out[r][c]) for r in range(H) for c in range(W)
                 if r < len(out) and c < len(out[r]) and inp[r][c] == bg and out[r][c] != bg]
        if not added:
            return None
        # Ray color = most common added color
        ray_color = Counter(v for _,_,v in added).most_common(1)[0][0]
        # Try each direction
        for direction in ("up", "down", "left", "right"):
            pred = self._shoot_rays(inp, direction, bg, ray_color)
            if pred == out:
                return (direction, ray_color)
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        params = None
        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None
            bg = _bg(p["input"])
            pr = self._infer_params(p["input"], p["output"], bg)
            if pr is None:
                return None
            if params is None:
                params = pr
            elif params != pr:
                return None

        direction, ray_color = params
        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            preds.append(self._shoot_rays(inp, direction, bg, ray_color))
        return _result(ctx, preds, self.name)


# ── 19. SizeBasedOutputSolver ─────────────────────────────────────────────────

@register_solver
class SizeBasedOutputSolver(BaseSolver):
    """
    Output encodes the sizes (areas) of objects as a small grid.
    E.g., three objects of sizes 2, 4, 6 → output row [2, 4, 6].
    """

    @property
    def name(self) -> str:
        return "SizeBasedOutputSolver"

    @property
    def priority(self) -> int:
        return 78

    def _get_object_sizes(self, grid: Grid) -> list:
        bg = _bg(grid)
        H, W = _sh(grid)
        vis = set()
        sizes = []
        for r in range(H):
            for c in range(W):
                if (r,c) not in vis and grid[r][c] != bg:
                    cc = _bfs4(grid, r, c)
                    vis.update(cc)
                    sizes.append(len(cc))
        return sorted(sizes)

    def _sizes_to_grid(self, sizes: list, out_shape: tuple) -> Grid | None:
        oh, ow = out_shape
        if oh == 1 and ow == len(sizes):
            return [sizes]
        if oh == len(sizes) and ow == 1:
            return [[s] for s in sizes]
        return None

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        for p in train_pairs:
            if p["output"] is None:
                return None
            sizes = self._get_object_sizes(p["input"])
            pred = self._sizes_to_grid(sizes, _sh(p["output"]))
            if pred != p["output"]:
                return None

        preds = []
        for i, inp in enumerate(ctx.test_inputs):
            sizes = self._get_object_sizes(inp)
            out_shape = _sh(train_pairs[0]["output"])
            pred = self._sizes_to_grid(sizes, out_shape)
            if pred is None:
                return None
            preds.append(pred)

        return _result(ctx, preds, self.name)


# ── 20. RectangleFillSolver ───────────────────────────────────────────────────

@register_solver
class RectangleFillSolver(BaseSolver):
    """
    Input has a partial rectangle outline; output fills the interior.
    Or: input marks two corners; output draws the filled rectangle.
    """

    @property
    def name(self) -> str:
        return "RectangleFillSolver"

    @property
    def priority(self) -> int:
        return 79

    def _find_rectangle_outline(self, grid: Grid, bg: int):
        """Find non-bg rectangle outline and return its bounds."""
        H, W = _sh(grid)
        non_bg = [(r,c) for r in range(H) for c in range(W) if grid[r][c] != bg]
        if not non_bg:
            return None
        r0 = min(r for r,c in non_bg)
        r1 = max(r for r,c in non_bg)
        c0 = min(c for r,c in non_bg)
        c1 = max(c for r,c in non_bg)
        if r1 == r0 or c1 == c0:
            return None
        # Verify outline: only border cells are non-bg
        expected = set()
        for r in range(r0, r1+1):
            expected.add((r, c0)); expected.add((r, c1))
        for c in range(c0, c1+1):
            expected.add((r0, c)); expected.add((r1, c))
        if set(non_bg) == expected:
            return (r0, r1, c0, c1)
        return None

    def _fill_rect(self, grid: Grid, r0, r1, c0, c1, fill_color: int, bg: int) -> Grid:
        out = _clone(grid)
        for r in range(r0+1, r1):
            for c in range(c0+1, c1):
                out[r][c] = fill_color
        return out

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        train_pairs = ctx.train_pairs
        if not train_pairs:
            return None

        fill_colors = []
        for p in train_pairs:
            if p["output"] is None or _sh(p["input"]) != _sh(p["output"]):
                return None
            bg = _bg(p["input"])
            bounds = self._find_rectangle_outline(p["input"], bg)
            if bounds is None:
                return None
            r0, r1, c0, c1 = bounds
            # Find fill color from output
            interior = [p["output"][r][c] for r in range(r0+1, r1) for c in range(c0+1, c1)
                        if r < len(p["output"]) and c < len(p["output"][r])]
            if not interior:
                return None
            fc = Counter(interior).most_common(1)[0][0]
            if fc == bg:
                return None
            fill_colors.append(fc)
            # Verify
            pred = self._fill_rect(p["input"], r0, r1, c0, c1, fc, bg)
            if pred != p["output"]:
                return None

        if len(set(fill_colors)) != 1:
            return None

        fill_color = fill_colors[0]

        preds = []
        for inp in ctx.test_inputs:
            bg = _bg(inp)
            bounds = self._find_rectangle_outline(inp, bg)
            if bounds is None:
                return None
            r0, r1, c0, c1 = bounds
            preds.append(self._fill_rect(inp, r0, r1, c0, c1, fill_color, bg))
        return _result(ctx, preds, self.name)
