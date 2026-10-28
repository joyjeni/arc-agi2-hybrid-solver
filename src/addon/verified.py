"""Verified symbolic layer — adds answers to an NVARC submission only when they are proven.

A candidate rule is accepted for a task only if it passes two checks:

1. **Train-exact**: applied to every training input, it reproduces every training output.
2. **Leave-one-out (LOO)**: re-fitted on all training pairs but one, it predicts the
   held-out pair's output exactly — for every pair. A rule that only "fits" because it
   saw all the examples fails here; a rule that captures the task's logic passes.

Sources, tried in order (first verified answer wins):
  * ``registry``  — the 399 hand-written solvers in ``src/solvers``
  * ``geometry``  — Arc2SymbolicSolver.solve_verified_geometry (third_party, Apache-2.0)
  * ``search``    — small generic program search: dihedral transforms, crop, tile,
                    scale, mirror, plus a learned colour map
  * ``families``  — general rule families (src/addon/families.py): symmetry and periodic
                    completion, object selection, grid partition

CLI (writes JSON: {task_id: [{"grid": [[...]] | null, "source": str | null}, ...]}):

    python -m src.addon.verified CHALLENGES.json OUT.json [--budget-s 600]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

Grid = list[list[int]]
Pair = dict[str, Grid]
# A source maps (train_pairs, test_inputs) -> one predicted grid per test input, or None.
Source = Callable[[list[Pair], list[Grid]], list[Grid] | None]


# ── helpers ───────────────────────────────────────────────────────────────────

def _plain(grid) -> Grid | None:
    """Return grid as a rectangular list of int lists in 0..9, or None if invalid."""
    try:
        g = [[int(v) for v in row] for row in grid]
    except (TypeError, ValueError):
        return None
    if not g or not g[0] or any(len(r) != len(g[0]) for r in g):
        return None
    if len(g) > 30 or len(g[0]) > 30:
        return None
    if any(not 0 <= v <= 9 for r in g for v in r):
        return None
    return g


def _call_with_timeout(fn: Callable, timeout_s: float, *args):
    """Run fn(*args) in a daemon thread; return its result, or None on timeout/error."""
    box: list = [None]

    def run():
        try:
            box[0] = fn(*args)
        except Exception:  # a failing candidate is simply "no answer"
            box[0] = None

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    return None if t.is_alive() else box[0]


def verify(source: Source, train: list[Pair], timeout_s: float = 2.0) -> bool:
    """True if `source` is train-exact and passes leave-one-out on `train`."""
    if len(train) < 2:
        return False  # LOO needs at least two pairs
    ins = [p["input"] for p in train]
    outs = [p["output"] for p in train]

    preds = _call_with_timeout(source, timeout_s, train, ins)
    if not preds or len(preds) != len(train):
        return False
    if any(_plain(p) != o for p, o in zip(preds, outs)):
        return False

    for k in range(len(train)):
        rest = train[:k] + train[k + 1:]
        pk = _call_with_timeout(source, timeout_s, rest, [ins[k]])
        if not pk or _plain(pk[0]) != outs[k]:
            return False
    return True


# ── source 1: the hand-written registry solvers ───────────────────────────────

def registry_sources() -> list[tuple[str, Source]]:
    from ..core.interfaces import TaskContext
    from ..solvers import REGISTRY

    solvers = sorted(REGISTRY.all_solvers(), key=lambda s: s.priority)
    out: list[tuple[str, Source]] = []
    for solver in solvers:
        if solver.name == "IdentitySolver":
            continue

        def src(train, test_inputs, _s=solver):
            res = _s.solve(TaskContext(task_id="_", train_pairs=train, test_inputs=test_inputs))
            if not res or len(res.predictions) != len(test_inputs):
                return None
            return [p.attempt_1 for p in res.predictions]

        out.append((f"registry:{solver.name}", src))
    return out


def _dihedral():
    """The 7 non-identity rotations/flips as (name, forward, inverse) on numpy grids."""
    return [
        ("rot90", lambda g: np.rot90(g, 1), lambda g: np.rot90(g, -1)),
        ("rot180", lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 2)),
        ("rot270", lambda g: np.rot90(g, 3), lambda g: np.rot90(g, -3)),
        ("flipud", np.flipud, np.flipud),
        ("fliplr", np.fliplr, np.fliplr),
        ("transpose", lambda g: g.T, lambda g: g.T),
        ("antitranspose", lambda g: np.rot90(g, 2).T, lambda g: np.rot90(g.T, 2)),
    ]


def oriented(name: str, src: Source) -> list[tuple[str, Source]]:
    """The same source applied to rotated/flipped copies of the task, answers turned back.
    Lets a rule written for one orientation cover the other seven."""
    out: list[tuple[str, Source]] = []
    for tname, fwd, inv in _dihedral():
        def wrapped(train, test_inputs, _f=fwd, _i=inv, _s=src):
            t_train = [{"input": _f(np.asarray(p["input"])).tolist(),
                        "output": _f(np.asarray(p["output"])).tolist()} for p in train]
            t_tests = [_f(np.asarray(g)).tolist() for g in test_inputs]
            preds = _s(t_train, t_tests)
            if not preds:
                return None
            return [None if p is None else _i(np.asarray(p)).tolist() for p in preds]
        out.append((f"{name}@{tname}", wrapped))
    return out


# ── source 2: verified geometry (third party) ─────────────────────────────────

def geometry_source() -> list[tuple[str, Source]]:
    try:
        from .third_party.arc2_symbolic import Arc2SymbolicSolver
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("geometry source unavailable: %s", exc)
        return []
    solver = Arc2SymbolicSolver()

    def src(train, test_inputs):
        task = {"train": train, "test": [{"input": g} for g in test_inputs]}
        res = solver.solve_verified_geometry(task)  # -> (program_names, [output per test]) | None
        if res is None:
            return None
        _names, outputs = res
        return [np.asarray(g).tolist() for g in outputs]

    return [("geometry:Arc2SymbolicSolver", src)]


# ── source 3: generic program search ──────────────────────────────────────────

def _crop(g: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(g, return_counts=True)
    bg = vals[np.argmax(counts)]
    rows = np.where((g != bg).any(1))[0]
    cols = np.where((g != bg).any(0))[0]
    if len(rows) == 0:
        return g
    return g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


_PRIMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "flipud": np.flipud,
    "fliplr": np.fliplr,
    "transpose": lambda g: g.T,
    "crop": _crop,
    "tile2": lambda g: np.tile(g, (2, 2)),
    "tile3": lambda g: np.tile(g, (3, 3)),
    "mirror_h": lambda g: np.hstack([g, np.fliplr(g)]),
    "mirror_v": lambda g: np.vstack([g, np.flipud(g)]),
    "mirror_4": lambda g: np.vstack([np.hstack([g, np.fliplr(g)]),
                                     np.flipud(np.hstack([g, np.fliplr(g)]))]),
    "up2": lambda g: np.kron(g, np.ones((2, 2), dtype=g.dtype)),
    "up3": lambda g: np.kron(g, np.ones((3, 3), dtype=g.dtype)),
    "top_half": lambda g: g[: g.shape[0] // 2],
    "left_half": lambda g: g[:, : g.shape[1] // 2],
}
_PROGRAMS = [(a,) for a in _PRIMS] + [(a, b) for a in _PRIMS for b in _PRIMS if a != b]


def _run_program(prog: tuple[str, ...], g: np.ndarray) -> np.ndarray | None:
    try:
        for name in prog:
            g = _PRIMS[name](g)
        return g if g.size and g.shape[0] <= 30 and g.shape[1] <= 30 else None
    except Exception:
        return None


def _fit_colour_map(preds: list[np.ndarray], outs: list[np.ndarray]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for p, o in zip(preds, outs):
        for a, b in zip(p.ravel().tolist(), o.ravel().tolist()):
            if mapping.setdefault(a, b) != b:
                return None
    return mapping


def _search(train: list[Pair], test_inputs: list[Grid]) -> list[Grid] | None:
    ins = [np.array(p["input"]) for p in train]
    outs = [np.array(p["output"]) for p in train]
    for prog in _PROGRAMS:
        preds = [_run_program(prog, g) for g in ins]
        if any(p is None or p.shape != o.shape for p, o in zip(preds, outs)):
            continue
        tests = [_run_program(prog, np.array(g)) for g in test_inputs]
        if any(t is None for t in tests):
            continue
        if all(np.array_equal(p, o) for p, o in zip(preds, outs)):
            return [t.tolist() for t in tests]
        mapping = _fit_colour_map(preds, outs)
        if mapping is None:
            continue
        # every test colour must have been seen in training, else the map is a guess
        if any(int(v) not in mapping for t in tests for v in np.unique(t)):
            continue
        return [[[mapping[int(v)] for v in row] for row in t.tolist()] for t in tests]
    return None


def search_source() -> list[tuple[str, Source]]:
    return [("search:program", _search)]


# ── driver ────────────────────────────────────────────────────────────────────

def family_sources() -> list[tuple[str, Source]]:
    from .families import FAMILIES
    return list(FAMILIES)


def registry_oriented_sources() -> list[tuple[str, Source]]:
    out: list[tuple[str, Source]] = []
    for name, src in registry_sources():
        out += oriented(name, src)
    return out


DEFAULT_SOURCES = ("registry", "geometry", "search", "families")
# Optional (not default until its precision is confirmed): "registry_oriented"


def all_sources(include: tuple[str, ...] = DEFAULT_SOURCES):
    sources: list[tuple[str, Source]] = []
    if "registry" in include:
        sources += registry_sources()
    if "geometry" in include:
        sources += geometry_source()
    if "search" in include:
        sources += search_source()
    if "families" in include:
        sources += family_sources()
    if "registry_oriented" in include:
        sources += registry_oriented_sources()
    return sources


def solve_task(task: dict, sources, timeout_s: float = 2.0) -> tuple[list[Grid | None], str | None]:
    """Return (one verified grid or None per test input, name of the source used)."""
    train = task["train"]
    test_inputs = [t["input"] for t in task["test"]]
    for name, src in sources:
        if not verify(src, train, timeout_s):
            continue
        preds = _call_with_timeout(src, timeout_s, train, test_inputs)
        if not preds or len(preds) != len(test_inputs):
            continue
        grids = [_plain(p) if p is not None else None for p in preds]
        if any(g is not None for g in grids):  # a source may abstain on some test inputs
            return grids, name
    return [None] * len(test_inputs), None


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def solve_challenges(challenges: dict, budget_s: float = 600.0, timeout_s: float = 2.0,
                     include: tuple[str, ...] = DEFAULT_SOURCES,
                     checkpoint: Path | None = None, every: int = 10) -> dict:
    """Verified answers for every task, stopping cleanly when the time budget runs out.

    If `checkpoint` is given, partial results are written there every `every` tasks,
    so a caller can use them even if this process is stopped early.
    """
    sources = all_sources(include)
    deadline = time.monotonic() + budget_s
    out: dict = {}
    for i, (tid, task) in enumerate(challenges.items(), 1):
        n = len(task["test"])
        if time.monotonic() > deadline:
            out[tid] = [{"grid": None, "source": None}] * n
            continue
        grids, name = solve_task(task, sources, timeout_s)
        out[tid] = [{"grid": g, "source": name if g is not None else None} for g in grids]
        if checkpoint is not None and i % every == 0:
            _write_atomic(checkpoint, out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("challenges")
    ap.add_argument("out")
    ap.add_argument("--budget-s", type=float, default=600.0)
    ap.add_argument("--timeout-s", type=float, default=2.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    t0 = time.monotonic()
    chal = json.loads(Path(a.challenges).read_text())
    out = Path(a.out)
    res = solve_challenges(chal, budget_s=a.budget_s, timeout_s=a.timeout_s, checkpoint=out)
    _write_atomic(out, res)
    Path(str(out) + ".done").write_text("ok")
    n = sum(1 for v in res.values() for e in v if e["grid"] is not None)
    print(f"[verified] {n} verified test outputs over {len(res)} tasks in {time.monotonic() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
