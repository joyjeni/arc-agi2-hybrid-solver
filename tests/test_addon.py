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
