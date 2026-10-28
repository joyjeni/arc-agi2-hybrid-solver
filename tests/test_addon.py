"""Tests for the verified symbolic add-on (src/addon)."""

import json

from src.addon.merge import merge, merge_files
from src.addon.verified import _search, solve_task, verify

A = [[1, 2], [3, 4]]
B = [[5, 5], [5, 5]]
C = [[0, 1], [1, 0]]


def _sub(a1, a2):
    return {"t": [{"attempt_1": a1, "attempt_2": a2}]}


def test_merge_keeps_submission_when_verified_agrees():
    new, stats = merge(_sub(A, B), {"t": [{"grid": B, "source": "x"}]})
    assert new == _sub(A, B)
    assert stats["agreed"] == 1 and stats["added"] == 0


def test_merge_replaces_only_attempt_2():
    new, stats = merge(_sub(A, B), {"t": [{"grid": C, "source": "x"}]})
    assert new["t"][0] == {"attempt_1": A, "attempt_2": C}
    assert stats["added"] == 1


def test_merge_skips_mismatched_test_count_and_none():
    sub = _sub(A, B)
    new, stats = merge(sub, {"t": [{"grid": C}, {"grid": C}], "u": [{"grid": C}]})
    assert new == sub
    new, _ = merge(sub, {"t": [{"grid": None, "source": None}]})
    assert new == sub


def test_merge_rejects_invalid_grid():
    sub = _sub(A, B)
    new, stats = merge(sub, {"t": [{"grid": [[1, 2], [3]], "source": "x"}]})
    assert new == sub and stats["skipped"] == 1


def test_merge_files_is_fail_safe(tmp_path):
    sub_p = tmp_path / "submission.json"
    sub_p.write_text(json.dumps(_sub(A, B)))
    bad = tmp_path / "verified.json"
    bad.write_text("{not json")
    stats = merge_files(sub_p, bad)
    assert "error" in stats
    assert json.loads(sub_p.read_text()) == _sub(A, B)
    assert "error" in merge_files(tmp_path / "missing.json", bad)


def test_verify_requires_leave_one_out():
    # Rule "flip left-right" generalises: passes train-exact and LOO.
    train = [{"input": [[1, 0]], "output": [[0, 1]]}, {"input": [[2, 3]], "output": [[3, 2]]}]
    assert verify(_search, train)

    # A source that memorises the training outputs is train-exact but fails LOO.
    lookup = {json.dumps(p["input"]): p["output"] for p in train}

    def memoriser(tr, test_inputs):
        known = {json.dumps(p["input"]): p["output"] for p in tr}
        return [known.get(json.dumps(g), lookup[json.dumps(g)]) if json.dumps(g) in known else [[9]]
                for g in test_inputs]

    assert not verify(memoriser, train)


def test_solve_task_returns_verified_answer():
    task = {
        "train": [{"input": [[1, 0]], "output": [[0, 1]]}, {"input": [[2, 3]], "output": [[3, 2]]}],
        "test": [{"input": [[4, 5]]}],
    }
    grids, name = solve_task(task, [("search:program", _search)])
    assert grids == [[[5, 4]]] and name == "search:program"


# ── rule families ─────────────────────────────────────────────────────────────

def _mirror_task(grid, hole):
    """Left-right symmetric grid with a rectangle of colour 9 punched in it."""
    import numpy as np
    full = np.array(grid)
    masked = full.copy()
    r0, r1, c0, c1 = hole
    masked[r0:r1, c0:c1] = 9
    return masked.tolist(), full.tolist()


def test_symmetry_family_fills_masked_region():
    from src.addon.families import symmetry
    base = [[1, 2, 3, 3, 2, 1], [4, 5, 6, 6, 5, 4], [7, 1, 2, 2, 1, 7], [3, 4, 5, 5, 4, 3]]
    base2 = [[2, 3, 1, 1, 3, 2], [5, 6, 4, 4, 6, 5], [1, 7, 2, 2, 7, 1], [4, 3, 5, 5, 3, 4]]
    i1, o1 = _mirror_task(base, (0, 2, 0, 2))
    i2, o2 = _mirror_task(base2, (2, 4, 4, 6))
    ti, to = _mirror_task(base, (1, 3, 3, 5))
    train = [{"input": i1, "output": o1}, {"input": i2, "output": o2}]
    assert verify(symmetry, train)
    assert symmetry(train, [ti]) == [to]


def test_periodic_family_fills_staggered_pattern():
    from src.addon.families import periodic

    def stagger(h, w, hole):
        g = [[(c + r) % 3 + 1 for c in range(w)] for r in range(h)]  # diagonal repeat
        full = [row[:] for row in g]
        r0, r1, c0, c1 = hole
        for r in range(r0, r1):
            for c in range(c0, c1):
                g[r][c] = 0
        return g, full

    i1, o1 = stagger(6, 6, (1, 3, 1, 3))
    i2, o2 = stagger(7, 5, (4, 6, 0, 2))
    ti, to = stagger(6, 7, (2, 4, 3, 6))
    train = [{"input": i1, "output": o1}, {"input": i2, "output": o2}]
    assert verify(periodic, train)
    assert periodic(train, [ti]) == [to]


def test_object_family_abstains_when_fitted_rules_disagree():
    from src.addon.families import _consensus
    assert _consensus([[[[1]]], [[[1]]]]) == [[[1]]]
    assert _consensus([[[[1]]], [[[2]]]]) is None
    assert _consensus([]) is None


def test_partition_family_boolean_overlay():
    from src.addon.families import partition
    # two 2x2 halves split by a column of 5; output marks cells set in both halves (AND)
    def pair(a, b):
        inp = [ra + [5] + rb for ra, rb in zip(a, b)]
        out = [[3 if x and y else 0 for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
        return {"input": inp, "output": out}
    # 3x3 cells, mostly background, four examples: enough evidence for one rule only
    train = [pair([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 1, 0], [0, 0, 0], [0, 0, 1]]),
             pair([[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, 1], [1, 0, 0], [1, 0, 0]]),
             pair([[1, 1, 0], [0, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1], [0, 1, 0]]),
             pair([[0, 0, 0], [1, 0, 1], [0, 0, 1]], [[0, 0, 0], [1, 1, 0], [0, 0, 1]])]
    test = pair([[1, 0, 1], [0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    assert verify(partition, train)
    assert partition(train, [test["input"]]) == [test["output"]]


def test_partition_family_abstains_on_ambiguous_evidence():
    """Tiny cells where several (background, operation) rules fit: the family must abstain."""
    from src.addon.families import partition
    train = [{"input": [[1, 0, 5, 1, 1], [1, 1, 5, 0, 1]], "output": [[3, 0], [0, 3]]},
             {"input": [[0, 1, 5, 1, 1], [1, 0, 5, 1, 1]], "output": [[0, 3], [3, 0]]}]
    assert not verify(partition, train)


def test_motif_family_recolours_every_copy_of_learned_shape():
    from src.addon.families import motif_recolour

    def grid(h, w, plus_at, noise):
        g = [[2] * w for _ in range(h)]
        out = [[2] * w for _ in range(h)]
        for r, c in noise:
            g[r][c] = out[r][c] = 4
        for r, c in plus_at:
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)):
                g[r + dr][c + dc] = 4
                out[r + dr][c + dc] = 8
        return {"input": g, "output": out}

    train = [grid(8, 8, [(0, 0)], [(6, 6)]), grid(8, 9, [(4, 5)], [(0, 7), (7, 0)]),
             grid(9, 8, [(1, 4), (5, 0)], [(8, 7)])]
    test = grid(9, 9, [(2, 2), (6, 6)], [(0, 0), (8, 0)])
    assert verify(motif_recolour, train)
    assert motif_recolour(train, [test["input"]]) == [test["output"]]
