"""General rule families for the verified layer.

Each family is a *source*: ``(train_pairs, test_inputs) -> [grid per test input] | None``.
A family fits its few parameters on the training pairs only, then applies them. It is
written for a whole class of ARC tasks, never for one task id, and is accepted for a task
only after ``verified.verify`` (train-exact + leave-one-out).

Families
  symmetry   fill a masked region from the grid's own mirror / rotational symmetries
  periodic   fill a masked region from a 2-D repeating tile
  object     output is one object of the input, chosen by a property learned from training
  partition  grid is split into equal cells; output is one cell, or a boolean overlay of cells
"""

from __future__ import annotations

from collections import Counter

import numpy as np

Grid = list[list[int]]


# ── shared helpers ────────────────────────────────────────────────────────────

def _arr(g) -> np.ndarray:
    return np.asarray(g, dtype=int)


def _bg(a: np.ndarray) -> int:
    vals, counts = np.unique(a, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask: np.ndarray):
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    if len(rows) == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def _components(a: np.ndarray, bg: int, diag: bool, multicolor: bool) -> list[dict]:
    """Connected components of non-background cells."""
    h, w = a.shape
    seen = np.zeros_like(a, dtype=bool)
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diag:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or a[r, c] == bg:
                continue
            col = a[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and a[ny, nx] != bg \
                            and (multicolor or a[ny, nx] == col):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            mask = np.zeros_like(a, dtype=bool)
            for y, x in cells:
                mask[y, x] = True
            comps.append({"mask": mask, "bbox": _bbox(mask)})
    return comps


def _ok(g: np.ndarray | None) -> bool:
    return g is not None and g.ndim == 2 and 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30


# ── family: symmetry completion ───────────────────────────────────────────────

def _symmetry_maps(h: int, w: int):
    """Candidate symmetries as (name, function (r, c) -> (r', c'))."""
    maps = []
    for a in range(1, 2 * w - 2):          # mirror left-right about column a/2
        maps.append(("lr", a, lambda r, c, a=a: (r, a - c)))
    for b in range(1, 2 * h - 2):          # mirror up-down about row b/2
        maps.append(("ud", b, lambda r, c, b=b: (b - r, c)))
    for d in range(-(h - 2), w - 1):       # transpose about the diagonal c - r = d
        maps.append(("tr", d, lambda r, c, d=d: (c - d, r + d)))
    for e in range(1, h + w - 2):          # anti-transpose about r + c = e
        maps.append(("at", e, lambda r, c, e=e: (e - c, e - r)))
    for dy in range(-(h // 2), h // 2 + 1):  # translations (repeating motifs)
        for dx in range(-(w // 2), w // 2 + 1):
            if dy or dx:
                maps.append(("sh", (dy, dx), lambda r, c, dy=dy, dx=dx: (r + dy, c + dx)))
    return maps


def _fill_by_symmetry(a: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = a.shape
    known = ~mask
    n_known = int(known.sum())
    if n_known == 0 or mask.sum() == 0:
        return None
    rr, cc = np.nonzero(known)
    valid = []
    for _kind, _p, f in _symmetry_maps(h, w):
        r2, c2 = f(rr, cc)
        inside = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        if inside.sum() < max(12, n_known // 3):
            continue
        r2i, c2i = r2[inside], c2[inside]
        both = known[r2i, c2i]
        if both.sum() < max(12, n_known // 4):
            continue
        if np.array_equal(a[rr[inside][both], cc[inside][both]], a[r2i[both], c2i[both]]):
            valid.append(f)
    if not valid:
        return None
    out = a.copy()
    filled = known.copy()
    changed = True
    while changed and not filled.all():
        changed = False
        mr, mc = np.nonzero(~filled)
        for f in valid:
            r2, c2 = f(mr, mc)
            inside = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            idx = np.nonzero(inside)[0]
            ok = idx[filled[r2[idx], c2[idx]]]
            if len(ok):
                out[mr[ok], mc[ok]] = out[r2[ok], c2[ok]]
                filled[mr[ok], mc[ok]] = True
                changed = True
            mr, mc = np.nonzero(~filled)
            if len(mr) == 0:
                break
    return out if filled.all() else None


def _mask_colour(train) -> int | None:
    """The single colour that is present in every input and absent from every same-shape output
    (the 'unknown' marker), or None."""
    common = None
    for p in train:
        i, o = _arr(p["input"]), _arr(p["output"])
        cand = set(np.unique(i).tolist())
        if i.shape == o.shape:
            cand -= set(np.unique(o).tolist())
        common = cand if common is None else common & cand
    return next(iter(common)) if common and len(common) == 1 else None


def _completion_family(train, test_inputs, filler):
    c = _mask_colour(train)
    if c is None:
        return None
    first_in, first_out = _arr(train[0]["input"]), _arr(train[0]["output"])
    crop = first_in.shape != first_out.shape  # output is just the masked rectangle
    preds = []
    for g in test_inputs:
        a = _arr(g)
        mask = a == c
        full = filler(a, mask)
        if full is None:
            return None
        if crop:
            bb = _bbox(mask)
            if bb is None:
                return None
            r0, r1, c0, c1 = bb
            full = full[r0:r1, c0:c1]
        if not _ok(full):
            return None
        preds.append(full.tolist())
    return preds


def symmetry(train, test_inputs):
    return _completion_family(train, test_inputs, _fill_by_symmetry)


# ── family: periodic pattern completion ───────────────────────────────────────

def _lattice_classes(h: int, w: int, py: int, px: int, sh: int) -> np.ndarray:
    """Class id of each cell for the lattice generated by (0, px) and (py, sh)."""
    r = np.arange(h)[:, None]
    c = np.arange(w)[None, :]
    col = (c - sh * (r // py)) % px
    return (r % py) * px + col


def _fill_by_period(a: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Fill masked cells from the smallest 2-D lattice tiling consistent with known cells.
    Covers plain repeats (shift 0) and staggered / diagonal repeats (shift > 0)."""
    h, w = a.shape
    known = ~mask
    if mask.sum() == 0 or known.sum() == 0:
        return None
    candidates = sorted(
        ((py * px, py, px, sh) for py in range(1, h + 1) for px in range(1, w + 1)
         for sh in range(px) if not (py == h and px == w)),
    )
    flat_known = known.ravel()
    flat_a = a.ravel()
    for _size, py, px, sh in candidates:
        if py * px > known.sum() // 2:
            break  # a tile this large is not evidenced by the known cells
        cls = _lattice_classes(h, w, py, px, sh).ravel()
        kc, kv = cls[flat_known], flat_a[flat_known]
        n = py * px
        lo = np.full(n, 10)
        hi = np.full(n, -1)
        np.minimum.at(lo, kc, kv)
        np.maximum.at(hi, kc, kv)
        if (hi < 0).any() or (lo != hi).any():
            continue  # a class with no evidence, or with conflicting colours
        return lo[cls].reshape(h, w)
    return None


def periodic(train, test_inputs):
    return _completion_family(train, test_inputs, _fill_by_period)


# ── family: object selection ──────────────────────────────────────────────────

def _features(a: np.ndarray, comps: list[dict], bg: int) -> list[dict]:
    shapes = []
    colours = []
    for o in comps:
        r0, r1, c0, c1 = o["bbox"]
        sub = np.where(o["mask"][r0:r1, c0:c1], a[r0:r1, c0:c1], -1)
        shapes.append(sub.tobytes() + bytes(sub.shape))
        colours.append(frozenset(np.unique(a[o["mask"]]).tolist()))
    shape_n, colour_n = Counter(shapes), Counter(colours)
    feats = []
    h, w = a.shape
    for o, s, col in zip(comps, shapes, colours):
        r0, r1, c0, c1 = o["bbox"]
        area = int(o["mask"].sum())
        feats.append({
            "area": area,
            "bbox_area": (r1 - r0) * (c1 - c0),
            "height": r1 - r0,
            "width": c1 - c0,
            "n_colours": len(col),
            "shape_rarity": -shape_n[s],
            "colour_rarity": -colour_n[col],
            "density": area / ((r1 - r0) * (c1 - c0)),
            "top": -r0,
            "left": -c0,
            "border": int(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
            "symmetric": int(_is_symmetric(a, o)),
        })
    return feats


def _is_symmetric(a: np.ndarray, o: dict) -> bool:
    r0, r1, c0, c1 = o["bbox"]
    sub_ = np.where(o["mask"][r0:r1, c0:c1], a[r0:r1, c0:c1], -1)
    return bool(np.array_equal(sub_, sub_[:, ::-1]) or np.array_equal(sub_, sub_[::-1]))


def _crops(a: np.ndarray, o: dict, bg: int) -> dict[str, np.ndarray]:
    r0, r1, c0, c1 = o["bbox"]
    raw = a[r0:r1, c0:c1]
    masked = np.where(o["mask"][r0:r1, c0:c1], raw, bg)
    return {"raw": raw, "masked": masked}


def _pick(feats: list[dict], key: str, sign: int) -> int | None:
    vals = [sign * f[key] for f in feats]
    best = max(vals)
    idx = [i for i, v in enumerate(vals) if v == best]
    return idx[0] if len(idx) == 1 else None


def _pick_obj(a: np.ndarray, comps: list[dict], feats: list[dict], key: str, sign: int,
              bg: int, mode: str) -> np.ndarray | None:
    """Crop of the object with the extreme property. Ties are allowed only when every
    tied object gives the same crop (e.g. two copies of one shape)."""
    vals = [sign * f[key] for f in feats]
    best = max(vals)
    crops = [_crops(a, comps[i], bg)[mode] for i, v in enumerate(vals) if v == best]
    first = crops[0]
    return first if all(np.array_equal(c, first) for c in crops[1:]) else None


def _consensus(answers: list[list[Grid]]) -> list[Grid] | None:
    """Return the answer only if every rule that fitted the training pairs agrees on it."""
    if not answers:
        return None
    first = answers[0]
    return first if all(a == first for a in answers[1:]) else None


def object_select(train, test_inputs):
    configs = [(d, m) for d in (True, False) for m in (True, False)]
    keys = ["area", "bbox_area", "height", "width", "n_colours", "shape_rarity",
            "colour_rarity", "density", "top", "left", "border", "symmetric"]
    answers: list[list[Grid]] = []
    for diag, multi in configs:
        per_pair = []
        for p in train:
            a, o = _arr(p["input"]), _arr(p["output"])
            bg = _bg(a)
            comps = _components(a, bg, diag, multi)
            if not 2 <= len(comps) <= 40:
                break
            per_pair.append((a, o, bg, comps, _features(a, comps, bg)))
        else:
            for mode in ("masked", "raw"):
                for key in keys:
                    for sign in (1, -1):
                        good = True
                        for a, o, bg, comps, feats in per_pair:
                            got = _pick_obj(a, comps, feats, key, sign, bg, mode)
                            if got is None or not np.array_equal(got, o):
                                good = False
                                break
                        if not good:
                            continue
                        preds = []
                        for g in test_inputs:
                            a = _arr(g)
                            bg = _bg(a)
                            comps = _components(a, bg, diag, multi)
                            got = (_pick_obj(a, comps, _features(a, comps, bg), key, sign, bg, mode)
                                   if comps else None)
                            if got is None:
                                return None  # a fitted rule cannot decide: abstain
                            preds.append(got.tolist())
                        answers.append(preds)
    return _consensus(answers)


# ── family: grid partition ────────────────────────────────────────────────────

def _cells(a: np.ndarray, shape: tuple[int, int]) -> list[np.ndarray] | None:
    """Split `a` into equal cells of `shape`, using separator lines if present."""
    h, w = a.shape
    ch, cw = shape
    # separator lines: full rows/cols of one colour
    for sep in np.unique(a).tolist():
        rows = [r for r in range(h) if (a[r] == sep).all()]
        cols = [c for c in range(w) if (a[:, c] == sep).all()]
        if not rows and not cols:
            continue
        rcuts = [-1] + rows + [h]
        ccuts = [-1] + cols + [w]
        cells = []
        for r0, r1 in zip(rcuts, rcuts[1:]):
            for c0, c1 in zip(ccuts, ccuts[1:]):
                if r1 - r0 - 1 > 0 and c1 - c0 - 1 > 0:
                    cells.append(a[r0 + 1:r1, c0 + 1:c1])
        if len(cells) >= 2 and all(c.shape == (ch, cw) for c in cells):
            return cells
    # no separators: exact tiling
    if h % ch == 0 and w % cw == 0 and (h // ch) * (w // cw) >= 2:
        return [a[r:r + ch, c:c + cw] for r in range(0, h, ch) for c in range(0, w, cw)]
    return None


_BOOL_OPS = {
    "and": lambda m: m.all(0),
    "or": lambda m: m.any(0),
    "xor": lambda m: m.sum(0) == 1,
    "nor": lambda m: ~m.any(0),
    "a_not_b": lambda m: m[0] & ~m[1:].any(0),
}


def _cell_shape(train) -> tuple[int, int] | None:
    shapes = {_arr(p["output"]).shape for p in train}
    return shapes.pop() if len(shapes) == 1 else None


def partition(train, test_inputs):
    shape = _cell_shape(train)
    if shape is None:
        return None
    data = []
    for p in train:
        a, o = _arr(p["input"]), _arr(p["output"])
        cells = _cells(a, shape)
        if cells is None:
            return None
        data.append((a, o, cells))
    test_cells = []
    for g in test_inputs:
        cells = _cells(_arr(g), shape)
        if cells is None:
            return None
        test_cells.append(cells)

    # (1) output is one cell, chosen by a property
    def props(cells):
        flat = [c.tobytes() for c in cells]
        n = Counter(flat)
        bgs = [_bg(np.concatenate([c.ravel() for c in cells]))] * len(cells)
        return [{
            "unique": -n[f],
            "fg": int((c != b).sum()),
            "n_colours": len(np.unique(c)),
        } for c, f, b in zip(cells, flat, bgs)]

    answers: list[list[Grid]] = []
    for key in ("unique", "fg", "n_colours"):
        for sign in (1, -1):
            if all((i := _pick(props(cells), key, sign)) is not None and np.array_equal(cells[i], o)
                   for _a, o, cells in data):
                out = []
                for cells in test_cells:
                    i = _pick(props(cells), key, sign)
                    if i is None:
                        return None
                    out.append(cells[i].tolist())
                answers.append(out)
    if answers:
        return _consensus(answers)  # a cell-selection rule fitted: don't also try overlays

    # (2) boolean overlay of all cells: a cell is "set" where it differs from a learned
    #     background colour; output = on-colour where the op holds, else off-colour.
    n_cells = {len(cells) for _a, _o, cells in data}
    if len(n_cells) != 1:
        return None
    bg_candidates = set.intersection(*[set(np.unique(np.stack(cells)).tolist())
                                       for _a, _o, cells in data])
    for bg in sorted(bg_candidates):
        for op in _BOOL_OPS.values():
            on_c = off_c = None
            good = True
            for _a, o, cells in data:
                m = op(np.stack([c != bg for c in cells]))
                for val, which in ((o[m], "on"), (o[~m], "off")):
                    if val.size == 0:
                        continue
                    if (val != val[0]).any():
                        good = False
                        break
                    v = int(val[0])
                    if which == "on":
                        on_c = v if on_c is None else on_c
                        good &= on_c == v
                    else:
                        off_c = v if off_c is None else off_c
                        good &= off_c == v
                if not good:
                    break
            if not good or on_c is None or off_c is None:
                continue
            out = []
            for cells in test_cells:
                if len(cells) not in n_cells:
                    return None
                m = op(np.stack([c != bg for c in cells]))
                out.append(np.where(m, on_c, off_c).tolist())
            answers.append(out)
    return _consensus(answers)



# ── family: motif recolouring ─────────────────────────────────────────────────

def _shape_key(cells: list[tuple[int, int]]) -> tuple:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def motif_recolour(train, test_inputs):
    """Every copy of a learned shape drawn in colour `a` is repainted in colour `b`."""
    src = dst = None
    shapes: set[tuple] = set()
    for p in train:
        a, o = _arr(p["input"]), _arr(p["output"])
        if a.shape != o.shape:
            return None
        d = a != o
        if not d.any():
            continue
        ins, outs = np.unique(a[d]), np.unique(o[d])
        if len(ins) != 1 or len(outs) != 1:
            return None
        if src is None:
            src, dst = int(ins[0]), int(outs[0])
        elif (src, dst) != (int(ins[0]), int(outs[0])):
            return None
        mask = np.where(d, 1, 0)
        for comp in _components(mask, 0, diag=False, multicolor=False):
            shapes.add(_shape_key(list(zip(*np.nonzero(comp["mask"])))))
    if src is None or not shapes or len(shapes) > 3:
        return None
    if any(len(sh) < 2 for sh in shapes):
        return None  # single cells are not a motif

    def paint(a: np.ndarray) -> np.ndarray:
        out = a.copy()
        h, w = a.shape
        for sh in shapes:
            sr = max(r for r, _ in sh) + 1
            sc = max(c for _, c in sh) + 1
            rr = np.array([r for r, _ in sh])
            cc = np.array([c for _, c in sh])
            for y in range(h - sr + 1):
                for x in range(w - sc + 1):
                    if (a[rr + y, cc + x] == src).all():
                        out[rr + y, cc + x] = dst
        return out

    return [paint(_arr(g)).tolist() for g in test_inputs]


# ── family: local neighbourhood rule ──────────────────────────────────────────

def _patches(a: np.ndarray, kind: str, bg: int) -> np.ndarray:
    """One hashable key per cell describing its 3x3 neighbourhood."""
    h, w = a.shape
    pad = np.pad(a, 1, constant_values=-1)
    keys = np.empty((h, w), dtype=object)
    for r in range(h):
        for c in range(w):
            blk = pad[r:r + 3, c:c + 3]
            if kind == "exact":
                keys[r, c] = blk.tobytes()
            elif kind == "cross":
                keys[r, c] = (blk[0, 1], blk[1, 0], blk[1, 1], blk[1, 2], blk[2, 1])
            else:  # "shape": centre colour + which neighbours are filled / empty / outside
                pattern = tuple(np.where(blk == -1, 2, (blk != bg).astype(int)).ravel().tolist())
                keys[r, c] = (int(blk[1, 1]), pattern)
    return keys


def local_rule(train, test_inputs):
    """Output cell = learned function of the input's 3x3 neighbourhood. Answers only when every
    test neighbourhood was seen in training."""
    if any(_arr(p["input"]).shape != _arr(p["output"]).shape for p in train):
        return None
    answers: list[list[Grid]] = []
    for kind in ("exact", "cross", "shape"):
        bg = None
        if kind == "shape":
            bgs = {_bg(_arr(p["input"])) for p in train}
            if len(bgs) != 1:
                continue
            bg = bgs.pop()
        table: dict = {}
        good = True
        for p in train:
            a, o = _arr(p["input"]), _arr(p["output"])
            keys = _patches(a, kind, bg)
            for k, v in zip(keys.ravel(), o.ravel().tolist()):
                if table.setdefault(k, v) != v:
                    good = False
                    break
            if not good:
                break
        if not good:
            continue
        out = []
        for g in test_inputs:
            a = _arr(g)
            if kind == "shape" and _bg(a) != bg:
                good = False
                break
            keys = _patches(a, kind, bg)
            if any(k not in table for k in keys.ravel()):
                good = False
                break
            out.append(np.array([table[k] for k in keys.ravel()]).reshape(a.shape).tolist())
        if good:
            answers.append(out)
    return _consensus(answers)



FAMILIES = [
    ("family:symmetry", symmetry),
    ("family:periodic", periodic),
    ("family:object", object_select),
    ("family:partition", partition),
    ("family:motif", motif_recolour),
    ("family:local", local_rule),
]
