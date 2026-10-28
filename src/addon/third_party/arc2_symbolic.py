# ─────────────────────────────────────────────────────────────────────────────
# Third-party code — NOT written by the repository author.
#
# Source : "EDA Gemini 4 Argon ARC-AGI-2 Solver Suite" by Robina Mirbahar (Kaggle),
#          https://www.kaggle.com/code/robinamirbahar/eda-gemini-4-argon-arc-agi-2-solver-suite
#          (cell `%%writefile arc2_symbolic.py`, notebook version 23)
# License: Apache License 2.0 (default license for public Kaggle notebooks).
# Changes: none to the code; this header added. Excluded from ruff linting.
# Used by : src/addon/verified.py (only `Arc2SymbolicSolver.solve_verified_geometry`).
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Grid = Tuple[Tuple[int, ...], ...]

def to_arr(grid) -> np.ndarray:
    return np.asarray(grid, dtype=np.int8)

def to_grid(arr: np.ndarray) -> Grid:
    arr = np.asarray(arr, dtype=np.int8)
    return tuple(tuple(int(x) for x in row) for row in arr.tolist())

def valid_grid(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    return arr.ndim == 2 and 1 <= arr.shape[0] <= 30 and 1 <= arr.shape[1] <= 30

def bg_color(a: np.ndarray) -> int:
    vals, cnt = np.unique(a, return_counts=True)
    return int(vals[int(np.argmax(cnt))])

def bbox_non_bg(a: np.ndarray, bg: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
    if bg is None:
        bg = bg_color(a)
    ys, xs = np.where(a != bg)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

def crop_bbox(a: np.ndarray, bg: Optional[int] = None) -> np.ndarray:
    bb = bbox_non_bg(a, bg)
    if bb is None:
        return a.copy()
    y0, y1, x0, x1 = bb
    return a[y0:y1, x0:x1].copy()

def trim_uniform_border(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    while out.shape[0] > 1 and len(set(out[0].tolist())) == 1:
        out = out[1:]
    while out.shape[0] > 1 and len(set(out[-1].tolist())) == 1:
        out = out[:-1]
    while out.shape[1] > 1 and len(set(out[:, 0].tolist())) == 1:
        out = out[:, 1:]
    while out.shape[1] > 1 and len(set(out[:, -1].tolist())) == 1:
        out = out[:, :-1]
    return out.copy()

def connected_components(a: np.ndarray, include_bg: bool = False):
    h, w = a.shape
    bg = bg_color(a)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if seen[y, x] or (not include_bg and int(a[y, x]) == bg):
                continue
            col = int(a[y, x])
            q = deque([(y, x)])
            seen[y, x] = True
            cells = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(a[ny, nx]) == col:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            comps.append(
                {
                    "color": col,
                    "cells": cells,
                    "area": len(cells),
                    "bbox": (min(ys), max(ys) + 1, min(xs), max(xs) + 1),
                    "height": max(ys) - min(ys) + 1,
                    "width": max(xs) - min(xs) + 1,
                }
            )
    return comps

def crop_component(a: np.ndarray, comp: dict, mask_bg: Optional[int] = None) -> np.ndarray:
    y0, y1, x0, x1 = comp["bbox"]
    sub = a[y0:y1, x0:x1].copy()
    if mask_bg is not None:
        out = np.full_like(sub, int(mask_bg))
        for y, x in comp["cells"]:
            out[y - y0, x - x0] = a[y, x]
        return out
    return sub

def transforms() -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    return [\
        ("id", lambda a: a.copy()),\
        ("rot90", lambda a: np.rot90(a, 1).copy()),\
        ("rot180", lambda a: np.rot90(a, 2).copy()),\
        ("rot270", lambda a: np.rot90(a, 3).copy()),\
        ("flipud", lambda a: np.flipud(a).copy()),\
        ("fliplr", lambda a: np.fliplr(a).copy()),\
        ("transpose", lambda a: a.T.copy()),\
        ("anti_transpose", lambda a: np.fliplr(np.flipud(a)).T.copy()),\
    ]

def tile_to_shape(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    reps = (int(np.ceil(shape[0] / a.shape[0])), int(np.ceil(shape[1] / a.shape[1])))
    return np.tile(a, reps)[: shape[0], : shape[1]].copy()

def shifted_tile_to_shape(a: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = a.shape
    out = np.zeros(shape, dtype=np.int8)
    for y in range(shape[0]):
        shift = y // h
        for x in range(shape[1]):
            out[y, x] = a[y % h, (x + shift) % w]
    return out

def infer_color_map(srcs: Sequence[np.ndarray], dsts: Sequence[np.ndarray]) -> Optional[Dict[int, int]]:
    mapping: Dict[int, int] = {}
    for s, d in zip(srcs, dsts):
        if s.shape != d.shape:
            return None
        for x, y in zip(s.ravel().tolist(), d.ravel().tolist()):
            x = int(x)
            y = int(y)
            if x in mapping and mapping[x] != y:
                return None
            mapping[x] = y
    return mapping

def apply_color_map(a: np.ndarray, mapping: Dict[int, int], default_keep: bool = True) -> np.ndarray:
    out = a.copy()
    for c in range(10):
        if c in mapping:
            out[a == c] = int(mapping[c])
        elif not default_keep:
            out[a == c] = 0
    return out

def remap_by_counts(src: np.ndarray, dst: np.ndarray) -> Optional[Dict[int, int]]:
    """Fallback color map by rank of color frequencies."""
    sc = Counter(src.ravel().tolist())
    dc = Counter(dst.ravel().tolist())
    if len(sc) != len(dc):
        return None
    s_order = [c for c, _ in sorted(sc.items(), key=lambda kv: (-kv[1], kv[0]))]
    d_order = [c for c, _ in sorted(dc.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {int(s): int(d) for s, d in zip(s_order, d_order)}

@dataclass
class Candidate:
    name: str
    outputs: List[np.ndarray]
    score: float

class Arc2SymbolicSolver:
    def __init__(self):
        self.max_candidates = 64

    def solve_task(self, task: dict) -> List[List[np.ndarray]]:
        tests = [to_arr(x["input"]) for x in task["test"]]
        by_test = [[] for _ in tests]
        for cand in self._exact_programs(task):
            for i, out in enumerate(cand.outputs):
                self._add(by_test[i], out, cand.score, cand.name)
        for cand in self._heuristic_programs(task):
            for i, out in enumerate(cand.outputs):
                self._add(by_test[i], out, cand.score, cand.name)
        answers = []
        for i, inp in enumerate(tests):
            if not by_test[i]:
                by_test[i].append((0.0, "fallback_input", inp.copy()))
            ordered = []
            seen = set()
            for _score, _name, out in sorted(by_test[i], key=lambda x: (-x[0], x[1])):
                if not valid_grid(out):
                    continue
                key = to_grid(out)
                if key not in seen:
                    seen.add(key)
                    ordered.append(out.astype(np.int8, copy=False))
                if len(ordered) == 2:
                    break
            while len(ordered) < 2:
                ordered.append(ordered[0].copy())
            answers.append(ordered[:2])
        return answers

    def _add(self, bucket, out: np.ndarray, score: float, name: str):
        if valid_grid(out):
            bucket.append((float(score), name, out.astype(np.int8, copy=False)))

    def _pairs(self, task: dict):
        return [(to_arr(p["input"]), to_arr(p["output"])) for p in task["train"]]

    def _exact_programs(self, task: dict) -> Iterable[Candidate]:
        pairs = self._pairs(task)
        tests = [to_arr(x["input"]) for x in task["test"]]
        yield from self._fit_border_ray_obstacle(task)
        yield from self._fit_masked_reflection(task)
        yield from self._fit_constant_output(pairs, tests)
        yield from self._fit_unary_transform(pairs, tests)
        yield from self._fit_crop_transform(pairs, tests)
        yield from self._fit_scale_transform(pairs, tests)
        yield from self._fit_periodic_tile(pairs, tests)
        yield from self._fit_shifted_tile(pairs, tests)
        yield from self._fit_masked_self_tile(pairs, tests)
        yield from self._fit_separator_overlay(pairs, tests)
        yield from self._fit_component_selector(pairs, tests)
        yield from self._fit_gridline_decomposition(pairs, tests)

    def _heuristic_programs(self, task: dict) -> Iterable[Candidate]:
        pairs = self._pairs(task)
        tests = [to_arr(x["input"]) for x in task["test"]]
        for idx, t in enumerate(tests):
            yield Candidate("input", [t.copy() if j == idx else tests[j].copy() for j in range(len(tests))], 0.05)
            yield Candidate("crop_non_bg", [crop_bbox(x) for x in tests], 0.10)
            yield Candidate("trim_border", [trim_uniform_border(x) for x in tests], 0.08)
        yield from self._nearest_shape_output(pairs, tests)

    def _simulate_border_rays(self, source, background, ray_color, obstacle_colors):
        grid = source.copy()
        height, width = grid.shape
        active = {
            col
            for col, value in enumerate(source[0])
            if int(value) == int(ray_color)
        }
        if not active:
            return None

        collisions = 0
        for row in range(1, height):
            runs = []
            col = 0
            while col < width:
                if int(source[row, col]) not in obstacle_colors:
                    col += 1
                    continue
                left = col
                while (
                    col + 1 < width
                    and int(source[row, col + 1]) in obstacle_colors
                ):
                    col += 1
                runs.append((left, col))
                col += 1

            blocked = set()
            spawned = set()
            for left, right in runs:
                hit = {
                    col
                    for col in active
                    if left <= col <= right
                }
                if not hit:
                    continue
                collisions += 1
                blocked.update(hit)

                for bypass_col in range(
                    max(0, left - 1),
                    min(width - 1, right + 1) + 1,
                ):
                    if int(grid[row - 1, bypass_col]) == int(background):
                        grid[row - 1, bypass_col] = ray_color

                for bypass_col in (left - 1, right + 1):
                    if 0 <= bypass_col < width:
                        spawned.add(bypass_col)
                        if int(grid[row, bypass_col]) == int(background):
                            grid[row, bypass_col] = ray_color

            active = (active - blocked) | spawned
            for active_col in active:
                if int(grid[row, active_col]) == int(background):
                    grid[row, active_col] = ray_color

        if collisions == 0:
            return None
        return grid

    def _fit_border_ray_obstacle(self, task):
        pairs = self._pairs(task)
        tests = [to_arr(item["input"]) for item in task["test"]]
        if not pairs or any(source.shape != target.shape for source, target in pairs):
            return

        transitions = {
            (int(source_value), int(target_value))
            for source, target in pairs
            for source_value, target_value in zip(source.ravel(), target.ravel())
            if int(source_value) != int(target_value)
        }
        if len(transitions) != 1:
            return
        background, ray_color = next(iter(transitions))
        if background == ray_color:
            return

        for source, target in pairs:
            allowed_change = (
                (source == background)
                & (target == ray_color)
            )
            if np.any((source != target) & ~allowed_change):
                return

        all_colors = {
            int(value)
            for source, _target in pairs
            for value in np.unique(source)
        }
        obstacle_colors = all_colors - {background, ray_color}
        if not obstacle_colors:
            return

        for rotation in range(4):
            valid = True
            for source, target in pairs:
                normalized = np.rot90(source, rotation)
                prediction = self._simulate_border_rays(
                    normalized,
                    background,
                    ray_color,
                    obstacle_colors,
                )
                if prediction is None:
                    valid = False
                    break
                prediction = np.rot90(prediction, -rotation)
                if not np.array_equal(prediction, target):
                    valid = False
                    break
            if not valid:
                continue

            outputs = []
            for test in tests:
                prediction = self._simulate_border_rays(
                    np.rot90(test, rotation),
                    background,
                    ray_color,
                    obstacle_colors,
                )
                if prediction is None:
                    valid = False
                    break
                outputs.append(np.rot90(prediction, -rotation))
            if valid:
                yield Candidate(
                    f"verified_border_rays:r{rotation}",
                    outputs,
                    1.20,
                )

    def _solid_color_rectangle(self, source, color):
        rows, cols = np.where(source == color)
        if len(rows) == 0:
            return None
        top = int(rows.min())
        bottom = int(rows.max()) + 1
        left = int(cols.min())
        right = int(cols.max()) + 1
        if (bottom - top) * (right - left) != len(rows):
            return None
        if not np.all(source[top:bottom, left:right] == color):
            return None
        return top, bottom, left, right

    def _recover_masked_reflection(self, source, mask_color):
        rectangle = self._solid_color_rectangle(source, mask_color)
        if rectangle is None:
            return None
        top, bottom, left, right = rectangle
        output_height = bottom - top
        output_width = right - left
        height, width = source.shape
        non_mask_count = int(np.count_nonzero(source != mask_color))
        minimum_support = max(20, (non_mask_count + 1) // 2)

        reflection_maps = []
        for orientation, size in (("horizontal", height), ("vertical", width)):
            for doubled_axis in range(2 * size - 1):
                matching = 0
                support = 0
                if orientation == "horizontal":
                    for row in range(height):
                        reflected_row = doubled_axis - row
                        if not 0 <= reflected_row < height:
                            continue
                        valid = (
                            (source[row, :] != mask_color)
                            & (source[reflected_row, :] != mask_color)
                        )
                        support += int(np.count_nonzero(valid))
                        matching += int(
                            np.count_nonzero(
                                (source[row, :] == source[reflected_row, :])
                                & valid
                            )
                        )
                else:
                    for col in range(width):
                        reflected_col = doubled_axis - col
                        if not 0 <= reflected_col < width:
                            continue
                        valid = (
                            (source[:, col] != mask_color)
                            & (source[:, reflected_col] != mask_color)
                        )
                        support += int(np.count_nonzero(valid))
                        matching += int(
                            np.count_nonzero(
                                (source[:, col] == source[:, reflected_col])
                                & valid
                            )
                        )
                if support >= minimum_support and matching == support:
                    reflection_maps.append((orientation, doubled_axis))

        if not reflection_maps:
            return None

        partial = np.full(
            (output_height, output_width),
            -1,
            dtype=np.int16,
        )
        for out_row in range(output_height):
            for out_col in range(output_width):
                row = top + out_row
                col = left + out_col
                reflected_values = []
                for orientation, doubled_axis in reflection_maps:
                    if orientation == "horizontal":
                        source_row = doubled_axis - row
                        source_col = col
                    else:
                        source_row = row
                        source_col = doubled_axis - col
                    if (
                        0 <= source_row < height
                        and 0 <= source_col < width
                        and int(source[source_row, source_col]) != int(mask_color)
                    ):
                        reflected_values.append(
                            int(source[source_row, source_col])
                        )
                if reflected_values:
                    if len(set(reflected_values)) != 1:
                        return None
                    partial[out_row, out_col] = reflected_values[0]

        known = partial >= 0
        if int(np.count_nonzero(known)) < max(
            4,
            (output_height * output_width + 2) // 3,
        ):
            return None

        candidates = {}
        source_shapes = {
            (output_height, output_width),
            (output_width, output_height),
        }
        for patch_height, patch_width in source_shapes:
            if patch_height > height or patch_width > width:
                continue
            for row in range(height - patch_height + 1):
                for col in range(width - patch_width + 1):
                    patch = source[\
                        row:row + patch_height,\
                        col:col + patch_width,\
                    ]
                    if np.any(patch == mask_color):
                        continue
                    variants = (
                        patch,
                        np.rot90(patch, 1),
                        np.rot90(patch, 2),
                        np.rot90(patch, 3),
                        np.flipud(patch),
                        np.fliplr(patch),
                        patch.T,
                        np.flipud(np.fliplr(patch)).T,
                    )
                    for variant in variants:
                        if variant.shape != partial.shape:
                            continue
                        if not np.array_equal(variant[known], partial[known]):
                            continue
                        key = to_grid(variant)
                        candidates[key] = variant.copy()

        if len(candidates) != 1:
            return None
        return next(iter(candidates.values()))

    def _fit_masked_reflection(self, task):
        pairs = self._pairs(task)
        tests = [to_arr(item["input"]) for item in task["test"]]
        if not pairs:
            return

        shared_mask_colors = None
        for source, target in pairs:
            output_colors = {int(value) for value in np.unique(target)}
            pair_mask_colors = set()
            for value in np.unique(source):
                color = int(value)
                if color in output_colors:
                    continue
                rectangle = self._solid_color_rectangle(source, color)
                if rectangle is None:
                    continue
                top, bottom, left, right = rectangle
                if (bottom - top, right - left) == target.shape:
                    pair_mask_colors.add(color)
            if shared_mask_colors is None:
                shared_mask_colors = pair_mask_colors
            else:
                shared_mask_colors &= pair_mask_colors
        if not shared_mask_colors:
            return

        for mask_color in sorted(shared_mask_colors):
            if not all(
                (
                    prediction := self._recover_masked_reflection(
                        source,
                        mask_color,
                    )
                ) is not None
                and np.array_equal(prediction, target)
                for source, target in pairs
            ):
                continue

            outputs = []
            valid = True
            for test in tests:
                prediction = self._recover_masked_reflection(
                    test,
                    mask_color,
                )
                if prediction is None:
                    valid = False
                    break
                outputs.append(prediction)
            if valid:
                yield Candidate(
                    f"verified_masked_reflection:c{mask_color}",
                    outputs,
                    1.20,
                )

    def solve_verified_geometry(self, task):
        programs = (
            list(self._fit_border_ray_obstacle(task))
            + list(self._fit_masked_reflection(task))
        )
        if not programs:
            return None

        reference = programs[0].outputs
        for program in programs[1:]:
            if len(program.outputs) != len(reference):
                return None
            if any(
                not np.array_equal(left, right)
                for left, right in zip(program.outputs, reference)
            ):
                return None
        names = ",".join(program.name for program in programs)
        return names, reference

    def _fit_constant_output(self, pairs, tests):
        outs = [to_grid(o) for _, o in pairs]
        if len(set(outs)) == 1:
            out = to_arr(outs[0])
            yield Candidate("constant_output", [out.copy() for _ in tests], 0.96)

    def _fit_unary_transform(self, pairs, tests):
        for name, fn in transforms():
            srcs, dsts = [], []
            ok = True
            for x, y in pairs:
                z = fn(x)
                if z.shape != y.shape:
                    ok = False
                    break
                srcs.append(z)
                dsts.append(y)
            if ok:
                cmap = infer_color_map(srcs, dsts)
                if cmap is not None:
                    yield Candidate("unary:" + name, [apply_color_map(fn(t), cmap) for t in tests], 1.00)
            if all(fn(x).shape == y.shape for x, y in pairs):
                maps = [remap_by_counts(fn(x), y) for x, y in pairs]
                if all(m is not None for m in maps) and len({tuple(sorted(m.items())) for m in maps}) == 1:
                    cmap = maps[0]
                    yield Candidate("unary_countmap:" + name, [apply_color_map(fn(t), cmap) for t in tests], 0.74)

    def _fit_crop_transform(self, pairs, tests):
        crop_fns = [\
            ("crop_bg", lambda a: crop_bbox(a, bg_color(a))),\
            ("crop_zero", lambda a: crop_bbox(a, 0)),\
            ("trim_uniform_border", trim_uniform_border),\
        ]
        for crop_name, crop_fn in crop_fns:
            for tname, tfn in transforms():
                srcs, dsts = [], []
                ok = True
                for x, y in pairs:
                    try:
                        z = tfn(crop_fn(x))
                    except Exception:
                        ok = False
                        break
                    if z.shape != y.shape:
                        ok = False
                        break
                    srcs.append(z)
                    dsts.append(y)
                if not ok:
                    continue
                cmap = infer_color_map(srcs, dsts)
                if cmap is not None:
                    outs = [apply_color_map(tfn(crop_fn(t)), cmap) for t in tests]
                    yield Candidate(f"crop:{crop_name}:{tname}", outs, 0.98)

    def _fit_scale_transform(self, pairs, tests):
        for tname, tfn in transforms():
            factors = []
            srcs, dsts = [], []
            ok = True
            for x, y in pairs:
                z = tfn(x)
                if y.shape[0] % z.shape[0] or y.shape[1] % z.shape[1]:
                    ok = False
                    break
                fy, fx = y.shape[0] // z.shape[0], y.shape[1] // z.shape[1]
                if fy < 1 or fx < 1 or fy > 10 or fx > 10:
                    ok = False
                    break
                zz = np.repeat(np.repeat(z, fy, axis=0), fx, axis=1)
                if zz.shape != y.shape:
                    ok = False
                    break
                factors.append((fy, fx))
                srcs.append(zz)
                dsts.append(y)
            if ok and len(set(factors)) == 1:
                cmap = infer_color_map(srcs, dsts)
                if cmap is not None:
                    fy, fx = factors[0]
                    outs = []
                    for t in tests:
                        z = tfn(t)
                        z = np.repeat(np.repeat(z, fy, axis=0), fx, axis=1)
                        outs.append(apply_color_map(z, cmap))
                    yield Candidate(f"scale:{tname}:{fy}x{fx}", outs, 0.99)

    def _fit_periodic_tile(self, pairs, tests):
        for tname, tfn in transforms():
            for mode in ("factor", "constant"):
                srcs, dsts, shapes, factors = [], [], [], []
                ok = True
                for x, y in pairs:
                    z = tfn(x)
                    if mode == "factor":
                        if y.shape[0] % z.shape[0] or y.shape[1] % z.shape[1]:
                            ok = False
                            break
                        factors.append((y.shape[0] // z.shape[0], y.shape[1] // z.shape[1]))
                        target_shape = y.shape
                    else:
                        target_shape = y.shape
                        shapes.append(target_shape)
                    zz = tile_to_shape(z, target_shape)
                    srcs.append(zz)
                    dsts.append(y)
                if not ok:
                    continue
                if mode == "factor" and len(set(factors)) != 1:
                    continue
                if mode == "constant" and len(set(shapes)) != 1:
                    continue
                cmap = infer_color_map(srcs, dsts)
                if cmap is None:
                    continue
                outs = []
                for t in tests:
                    z = tfn(t)
                    if mode == "factor":
                        fy, fx = factors[0]
                        target_shape = (z.shape[0] * fy, z.shape[1] * fx)
                    else:
                        target_shape = shapes[0]
                    outs.append(apply_color_map(tile_to_shape(z, target_shape), cmap))
                yield Candidate(f"periodic_tile:{mode}:{tname}", outs, 0.97)

    def _fit_shifted_tile(self, pairs, tests):
        factors = []
        srcs, dsts = [], []
        ok = True
        for x, y in pairs:
            if y.shape[0] % x.shape[0] or y.shape[1] % x.shape[1]:
                ok = False
                break
            fy, fx = y.shape[0] // x.shape[0], y.shape[1] // x.shape[1]
            factors.append((fy, fx))
            srcs.append(shifted_tile_to_shape(x, y.shape))
            dsts.append(y)
        if ok and len(set(factors)) == 1:
            cmap = infer_color_map(srcs, dsts)
            if cmap is not None:
                fy, fx = factors[0]
                outs = [apply_color_map(shifted_tile_to_shape(t, (t.shape[0] * fy, t.shape[1] * fx)), cmap)\
                        for t in tests]
                yield Candidate(f"shifted_tile:{fy}x{fx}", outs, 0.985)

    def _masked_self_tile(self, a: np.ndarray, bg: int) -> np.ndarray:
        h, w = a.shape
        out = np.full((h * h, w * w), int(bg), dtype=np.int8)
        for y in range(h):
            for x in range(w):
                if int(a[y, x]) != int(bg):
                    out[y * h:(y + 1) * h, x * w:(x + 1) * w] = a
        return out

    def _fit_masked_self_tile(self, pairs, tests):
        for bg_mode in ("zero", "major"):
            srcs, dsts = [], []
            ok = True
            for x, y in pairs:
                bg = 0 if bg_mode == "zero" else bg_color(x)
                z = self._masked_self_tile(x, bg)
                if z.shape != y.shape:
                    ok = False
                    break
                srcs.append(z)
                dsts.append(y)
            if not ok:
                continue
            cmap = infer_color_map(srcs, dsts)
            if cmap is None:
                continue
            outs = []
            for t in tests:
                bg = 0 if bg_mode == "zero" else bg_color(t)
                outs.append(apply_color_map(self._masked_self_tile(t, bg), cmap))
            yield Candidate(f"masked_self_tile:{bg_mode}", outs, 0.99)

    def _split_by_separator(self, a: np.ndarray):
        h, w = a.shape
        for x in range(1, w - 1):
            if len(set(a[:, x].tolist())) == 1 and x == w - x - 1:
                yield "v", int(a[0, x]), a[:, :x], a[:, x + 1:]
        for y in range(1, h - 1):
            if len(set(a[y, :].tolist())) == 1 and y == h - y - 1:
                yield "h", int(a[y, 0]), a[:y, :], a[y + 1:, :]

    def _overlay_panels(self, left: np.ndarray, right: np.ndarray, mode: str) -> Optional[np.ndarray]:
        if left.shape != right.shape:
            return None
        vals = set(left.ravel().tolist()) | set(right.ravel().tolist())
        bg = 0 if 0 in vals else bg_color(np.concatenate([left.reshape(-1), right.reshape(-1)]))
        out = np.full(left.shape, int(bg), dtype=np.int8)
        if mode == "intersection_same":
            mask = (left == right) & (left != bg)
            out[mask] = left[mask]
        elif mode == "intersection_nonbg":
            mask = (left != bg) & (right != bg)
            out[mask] = left[mask]
        elif mode == "left_minus_right":
            mask = (left != bg) & (right == bg)
            out[mask] = left[mask]
        elif mode == "right_minus_left":
            mask = (right != bg) & (left == bg)
            out[mask] = right[mask]
        elif mode == "xor":
            mask = (left != bg) ^ (right != bg)
            out[mask] = np.where(left[mask] != bg, left[mask], right[mask])
        elif mode == "binary_intersection_same":
            mask = (left == right) & (left != bg)
            out[mask] = 1
        elif mode == "binary_intersection_nonbg":
            mask = (left != bg) & (right != bg)
            out[mask] = 1
        elif mode == "binary_left_minus_right":
            mask = (left != bg) & (right == bg)
            out[mask] = 1
        elif mode == "binary_right_minus_left":
            mask = (right != bg) & (left == bg)
            out[mask] = 1
        elif mode == "binary_xor":
            mask = (left != bg) ^ (right != bg)
            out[mask] = 1
        else:
            return None
        return out

    def _fit_separator_overlay(self, pairs, tests):
        modes = [\
            "intersection_same", "intersection_nonbg", "left_minus_right", "right_minus_left", "xor",\
            "binary_intersection_same", "binary_intersection_nonbg", "binary_left_minus_right",\
            "binary_right_minus_left", "binary_xor",\
        ]
        for mode in modes:
            for side_order in ("normal", "swap"):
                srcs, dsts = [], []
                ok = True
                orientations = []
                for x, y in pairs:
                    splits = list(self._split_by_separator(x))
                    matched = False
                    for orient, _sep, a, b in splits:
                        left, right = (a, b) if side_order == "normal" else (b, a)
                        z = self._overlay_panels(left, right, mode)
                        if z is not None and z.shape == y.shape:
                            srcs.append(z)
                            dsts.append(y)
                            orientations.append(orient)
                            matched = True
                            break
                    if not matched:
                        ok = False
                        break
                if not ok or len(set(orientations)) > 1:
                    continue
                cmap = infer_color_map(srcs, dsts)
                if cmap is None:
                    continue
                outs = []
                for t in tests:
                    splits = [s for s in self._split_by_separator(t) if s[0] == orientations[0]]
                    if not splits:
                        outs.append(t.copy())
                        continue
                    _orient, _sep, a, b = splits[0]
                    left, right = (a, b) if side_order == "normal" else (b, a)
                    z = self._overlay_panels(left, right, mode)
                    outs.append(apply_color_map(z if z is not None else t, cmap))
                yield Candidate(f"separator:{mode}:{side_order}", outs, 0.94)

    def _fit_component_selector(self, pairs, tests):
        selectors = [\
            ("largest", lambda comps: max(comps, key=lambda c: c["area"]) if comps else None),\
            ("smallest", lambda comps: min(comps, key=lambda c: c["area"]) if comps else None),\
            ("topmost", lambda comps: min(comps, key=lambda c: (c["bbox"][0], c["bbox"][2], -c["area"])) if comps else None),\
            ("leftmost", lambda comps: min(comps, key=lambda c: (c["bbox"][2], c["bbox"][0], -c["area"])) if comps else None),\
        ]
        for sname, selector in selectors:
            for mask in (None, 0):
                for tname, tfn in transforms():
                    srcs, dsts = [], []
                    ok = True
                    for x, y in pairs:
                        comp = selector(connected_components(x))
                        if comp is None:
                            ok = False
                            break
                        z = tfn(crop_component(x, comp, mask_bg=mask))
                        if z.shape != y.shape:
                            ok = False
                            break
                        srcs.append(z)
                        dsts.append(y)
                    if not ok:
                        continue
                    cmap = infer_color_map(srcs, dsts)
                    if cmap is None:
                        continue
                    outs = []
                    for t in tests:
                        comp = selector(connected_components(t))
                        if comp is None:
                            outs.append(t.copy())
                        else:
                            outs.append(apply_color_map(tfn(crop_component(t, comp, mask_bg=mask)), cmap))
                    yield Candidate(f"component:{sname}:{mask}:{tname}", outs, 0.93)

    def _fit_gridline_decomposition(self, pairs, tests):
        for mode in ("rows", "cols"):
            for reducer_name, reducer in (
                ("remove_uniform", self._remove_uniform_lines),
                ("keep_uniform", self._keep_uniform_lines),
            ):
                srcs, dsts = [], []
                ok = True
                for x, y in pairs:
                    z = reducer(x, mode)
                    if z is None or z.shape != y.shape:
                        ok = False
                        break
                    srcs.append(z)
                    dsts.append(y)
                if not ok:
                    continue
                cmap = infer_color_map(srcs, dsts)
                if cmap is None:
                    continue
                outs = []
                for t in tests:
                    z = reducer(t, mode)
                    outs.append(apply_color_map(z if z is not None else t, cmap))
                yield Candidate(f"gridline:{mode}:{reducer_name}", outs, 0.86)

    def _remove_uniform_lines(self, a: np.ndarray, mode: str) -> Optional[np.ndarray]:
        if mode == "rows":
            keep = [i for i in range(a.shape[0]) if len(set(a[i].tolist())) > 1]
            return a[keep, :].copy() if keep else None
        keep = [j for j in range(a.shape[1]) if len(set(a[:, j].tolist())) > 1]
        return a[:, keep].copy() if keep else None

    def _keep_uniform_lines(self, a: np.ndarray, mode: str) -> Optional[np.ndarray]:
        if mode == "rows":
            keep = [i for i in range(a.shape[0]) if len(set(a[i].tolist())) == 1]
            return a[keep, :].copy() if keep else None
        keep = [j for j in range(a.shape[1]) if len(set(a[:, j].tolist())) == 1]
        return a[:, keep].copy() if keep else None

    def _nearest_shape_output(self, pairs, tests):
        outs = [y for _, y in pairs]
        for t in tests:
            best = min(outs, key=lambda y: abs(y.shape[0] - t.shape[0]) + abs(y.shape[1] - t.shape[1]))
            yield Candidate("nearest_train_output", [best.copy() for _ in tests], 0.01)

def solve_challenges(challenges: dict) -> dict:
    solver = Arc2SymbolicSolver()
    submission = {}
    for task_id, task in challenges.items():
        solved = solver.solve_task(task)
        submission[task_id] = []
        for attempts in solved:
            submission[task_id].append(
                {
                    "attempt_1": np.asarray(attempts[0], dtype=int).tolist(),
                    "attempt_2": np.asarray(attempts[1], dtype=int).tolist(),
                }
            )
    return submission

def score_submission(challenges: dict, solutions: dict, submission: dict) -> Tuple[int, int, List[str]]:
    correct = 0
    total = 0
    solved_ids = []
    for task_id, answers in solutions.items():
        task_ok = False
        for i, truth in enumerate(answers):
            total += 1
            pred = submission.get(task_id, [{}])[i]
            t = to_grid(truth)
            if to_grid(pred.get("attempt_1", [[0]])) == t or to_grid(pred.get("attempt_2", [[0]])) == t:
                correct += 1
                task_ok = True
        if task_ok:
            solved_ids.append(task_id)
    return correct, total, solved_ids

def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
