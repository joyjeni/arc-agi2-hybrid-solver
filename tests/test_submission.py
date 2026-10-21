"""Tests — submission format validation."""

import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.submission import build_submission, validate_submission, score_submission
from src.core.interfaces import TaskContext, SolverResult, Prediction


def _make_ctx(tid, n_test=1):
    test_inputs = [[[0, 1], [2, 3]]] * n_test
    train_pairs = [{"input": [[0, 1]], "output": [[1, 0]]}]
    return TaskContext(task_id=tid, train_pairs=train_pairs, test_inputs=test_inputs)


def _make_result(tid, grids):
    preds = [Prediction(attempt_1=g, attempt_2=g, solver_name="test") for g in grids]
    return SolverResult(task_id=tid, predictions=preds, solved=True, tier=1)


class TestSubmissionFormat(unittest.TestCase):

    def test_single_test_input(self):
        ctx = _make_ctx("task_a", n_test=1)
        res = _make_result("task_a", [[[1, 0], [3, 2]]])
        sub = build_submission({"task_a": ctx}, {"task_a": res})
        self.assertIn("task_a", sub)
        self.assertEqual(len(sub["task_a"]), 1)
        self.assertIn("attempt_1", sub["task_a"][0])
        self.assertIn("attempt_2", sub["task_a"][0])
        self.assertIsInstance(sub["task_a"][0]["attempt_1"], list)

    def test_multi_test_inputs(self):
        ctx = _make_ctx("task_b", n_test=2)
        res = _make_result("task_b", [[[1]], [[2]]])
        sub = build_submission({"task_b": ctx}, {"task_b": res})
        self.assertEqual(len(sub["task_b"]), 2)

    def test_fallback_copies_input(self):
        ctx = _make_ctx("task_c", n_test=1)
        sub = build_submission({"task_c": ctx}, {}, fallback_input=True)
        self.assertEqual(sub["task_c"][0]["attempt_1"], ctx.test_inputs[0])

    def test_validate_passes(self):
        ctx = _make_ctx("task_d", n_test=1)
        res = _make_result("task_d", [[[0, 1]]])
        sub = build_submission({"task_d": ctx}, {"task_d": res})
        errors = validate_submission(sub, {"task_d": ctx})
        self.assertEqual(errors, [])

    def test_validate_detects_missing_attempt(self):
        ctx = _make_ctx("task_e")
        sub = {"task_e": [{"attempt_1": [[0]]}]}   # missing attempt_2
        errors = validate_submission(sub, {"task_e": ctx})
        self.assertTrue(any("attempt_2" in e for e in errors))

    def test_validate_detects_out_of_range_cell(self):
        ctx = _make_ctx("task_f")
        sub = {"task_f": [{"attempt_1": [[99]], "attempt_2": [[0]]}]}
        errors = validate_submission(sub, {"task_f": ctx})
        self.assertTrue(len(errors) > 0)

    def test_score_correct(self):
        ctx = _make_ctx("task_g")
        grid = [[1, 2], [3, 4]]
        sub = {"task_g": [{"attempt_1": grid, "attempt_2": grid}]}
        solutions = {"task_g": [grid]}
        self.assertEqual(score_submission(sub, solutions), 1.0)

    def test_score_wrong(self):
        ctx = _make_ctx("task_h")
        grid_wrong = [[9, 9]]
        grid_gt = [[1, 2], [3, 4]]
        sub = {"task_h": [{"attempt_1": grid_wrong, "attempt_2": grid_wrong}]}
        solutions = {"task_h": [grid_gt]}
        self.assertEqual(score_submission(sub, solutions), 0.0)

    def test_json_roundtrip(self):
        """Ensure submission survives json.dumps / json.loads cycle."""
        ctx = _make_ctx("task_i")
        res = _make_result("task_i", [[[3, 4], [5, 6]]])
        sub = build_submission({"task_i": ctx}, {"task_i": res})
        raw = json.dumps(sub)
        loaded = json.loads(raw)
        self.assertEqual(sub, loaded)


class TestGridCodec(unittest.TestCase):

    def test_encode_decode_numeric(self):
        from src.core.grid_codec import grid_to_text, text_to_grid
        g = [[0, 1, 2], [3, 4, 5]]
        text = grid_to_text(g, "numeric")
        decoded = text_to_grid(text, "numeric")
        self.assertEqual(g, decoded)

    def test_encode_decode_symbol(self):
        from src.core.grid_codec import grid_to_text, text_to_grid
        g = [[0, 1], [2, 3]]
        text = grid_to_text(g, "symbol")
        decoded = text_to_grid(text, "symbol")
        self.assertEqual(g, decoded)


class TestAugmentation(unittest.TestCase):

    def test_eight_views_invertible(self):
        from src.core.augmentation import eight_views, inverse_view
        g = [[1, 2, 3], [4, 5, 6]]
        views = eight_views(g)
        self.assertEqual(len(views), 8)
        for i, v in enumerate(views):
            recovered = inverse_view(i, v)
            self.assertEqual(recovered, g, f"View {i} not invertible")


if __name__ == "__main__":
    unittest.main()
