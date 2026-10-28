"""Tests — submission format validation."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.interfaces import Prediction, SolverResult, TaskContext
from src.core.submission import build_submission, score_submission, validate_submission

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tid: str, n_test: int = 1) -> TaskContext:
    test_inputs = [[[0, 1], [2, 3]]] * n_test
    train_pairs = [{"input": [[0, 1]], "output": [[1, 0]]}]
    return TaskContext(task_id=tid, train_pairs=train_pairs, test_inputs=test_inputs)


def _make_result(tid: str, grids: list) -> SolverResult:
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name="test") for g in grids]
    return SolverResult(task_id=tid, predictions=preds, solved=True, tier=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubmissionFormat(unittest.TestCase):

    def setUp(self) -> None:
        """One default context per test; task_id matches the test name."""
        self.ctx = _make_ctx("default")

    # -- Structure ----------------------------------------------------------

    def test_single_test_input(self) -> None:
        ctx = _make_ctx("task_a")
        res = _make_result("task_a", [[[1, 0], [3, 2]]])
        sub = build_submission({"task_a": ctx}, {"task_a": res})
        self.assertIn("task_a", sub)
        self.assertEqual(len(sub["task_a"]), 1)
        self.assertIn("attempt_1", sub["task_a"][0])
        self.assertIn("attempt_2", sub["task_a"][0])
        self.assertIsInstance(sub["task_a"][0]["attempt_1"], list)

    def test_multi_test_inputs(self) -> None:
        ctx = _make_ctx("task_b", n_test=2)
        res = _make_result("task_b", [[[1]], [[2]]])
        sub = build_submission({"task_b": ctx}, {"task_b": res})
        self.assertEqual(len(sub["task_b"]), 2)

    def test_fallback_copies_input(self) -> None:
        ctx = _make_ctx("task_c")
        sub = build_submission({"task_c": ctx}, {}, fallback_input=True)
        self.assertEqual(sub["task_c"][0]["attempt_1"], ctx.test_inputs[0])

    # -- Validation ---------------------------------------------------------

    def test_validate_passes(self) -> None:
        ctx = _make_ctx("task_d")
        res = _make_result("task_d", [[[0, 1]]])
        sub = build_submission({"task_d": ctx}, {"task_d": res})
        errors = validate_submission(sub, {"task_d": ctx})
        self.assertEqual(errors, [])

    def test_validate_detects_missing_attempt(self) -> None:
        ctx = _make_ctx("task_e")
        sub = {"task_e": [{"attempt_1": [[0]]}]}   # missing attempt_2
        errors = validate_submission(sub, {"task_e": ctx})
        self.assertTrue(any("attempt_2" in e for e in errors))

    def test_validate_detects_out_of_range_cell(self) -> None:
        ctx = _make_ctx("task_f")
        sub = {"task_f": [{"attempt_1": [[99]], "attempt_2": [[0]]}]}
        errors = validate_submission(sub, {"task_f": ctx})
        self.assertTrue(len(errors) > 0)

    # -- Scoring ------------------------------------------------------------

    def test_score_correct(self) -> None:
        grid = [[1, 2], [3, 4]]
        sub = {"task_g": [{"attempt_1": grid, "attempt_2": grid}]}
        solutions = {"task_g": [grid]}
        self.assertEqual(score_submission(sub, solutions), 1.0)

    def test_score_wrong(self) -> None:
        grid_wrong = [[9, 9]]
        grid_gt = [[1, 2], [3, 4]]
        sub = {"task_h": [{"attempt_1": grid_wrong, "attempt_2": grid_wrong}]}
        solutions = {"task_h": [grid_gt]}
        self.assertEqual(score_submission(sub, solutions), 0.0)

    # -- JSON round-trip ----------------------------------------------------

    def test_json_roundtrip(self) -> None:
        """Ensure submission survives json.dumps / json.loads cycle."""
        ctx = _make_ctx("task_i")
        res = _make_result("task_i", [[[3, 4], [5, 6]]])
        sub = build_submission({"task_i": ctx}, {"task_i": res})
        raw = json.dumps(sub)
        loaded = json.loads(raw)
        self.assertEqual(sub, loaded)


class TestGridCodec(unittest.TestCase):

    def test_encode_decode_numeric(self) -> None:
        from src.core.grid_codec import grid_to_text, text_to_grid
        g = [[0, 1, 2], [3, 4, 5]]
        self.assertEqual(g, text_to_grid(grid_to_text(g, "numeric"), "numeric"))

    def test_encode_decode_symbol(self) -> None:
        from src.core.grid_codec import grid_to_text, text_to_grid
        g = [[0, 1], [2, 3]]
        self.assertEqual(g, text_to_grid(grid_to_text(g, "symbol"), "symbol"))


class TestAugmentation(unittest.TestCase):

    def test_eight_views_invertible(self) -> None:
        from src.core.augmentation import eight_views, inverse_view
        g = [[1, 2, 3], [4, 5, 6]]
        views = eight_views(g)
        self.assertEqual(len(views), 8)
        for i, v in enumerate(views):
            recovered = inverse_view(i, v)
            self.assertEqual(recovered, g, f"View {i} not invertible")


if __name__ == "__main__":
    unittest.main()


def test_build_submission_converts_numpy_ints_to_plain_ints():
    """A solver returning numpy values must not break json.dumps of the submission."""
    import json

    import numpy as np

    from src.core.interfaces import Prediction, SolverResult, TaskContext
    from src.core.submission import build_submission

    ctx = TaskContext(task_id="t", train_pairs=[], test_inputs=[[[0, 1]]])
    grid = np.array([[1, 2], [3, 4]], dtype=np.int64)
    res = SolverResult(task_id="t", predictions=[Prediction(attempt_1=grid, attempt_2=grid.tolist())])
    sub = build_submission({"t": ctx}, {"t": res})
    assert sub["t"][0]["attempt_1"] == [[1, 2], [3, 4]]
    assert all(type(v) is int for row in sub["t"][0]["attempt_1"] for v in row)
    json.dumps(sub)  # must not raise
