"""
advanced_solvers34.py — ScaleUpSolver
Rule: input is 3x3; scale factor k = number of distinct non-zero colors in input.
Each pixel is expanded to a k×k block in the output.
Scale factor is learned per training pair (must be consistent: k == len(distinct non-zero values)).
"""

from typing import Optional, List
from .registry import register_solver
from ..core.interfaces import BaseSolver, TaskContext, SolverResult, Prediction


def _count_nonzero_colors(grid: List[List[int]]) -> int:
    """Count distinct non-zero values in a grid."""
    vals = set()
    for row in grid:
        for v in row:
            if v != 0:
                vals.add(v)
    return len(vals)


def _scale_grid(grid: List[List[int]], k: int) -> List[List[int]]:
    """Scale each pixel to a k×k block."""
    result = []
    for row in grid:
        scaled_rows = [[] for _ in range(k)]
        for v in row:
            for r in range(k):
                scaled_rows[r].extend([v] * k)
        result.extend(scaled_rows)
    return result


@register_solver
class ScaleUpSolver(BaseSolver):
    """
    Solves tasks where a 3×3 input is scaled up by a factor equal to the
    number of distinct non-zero colors in that input.
    Scale factor is variable per pair — learned from the output size.
    """

    @property
    def name(self) -> str:
        return "ScaleUpSolver"

    def solve(self, ctx: TaskContext) -> Optional[SolverResult]:
        # Validate all training pairs
        for pair in ctx.train_pairs:
            inp = pair["input"]
            out = pair["output"]

            # Input must be 3×3
            if len(inp) != 3 or any(len(r) != 3 for r in inp):
                return None

            k = _count_nonzero_colors(inp)

            # k=0 means all-zero input — skip (no meaningful scale)
            if k == 0:
                return None

            # Output must be k*3 × k*3
            expected_rows = k * 3
            expected_cols = k * 3
            if len(out) != expected_rows or any(len(r) != expected_cols for r in out):
                return None

            # Verify pixel expansion
            predicted = _scale_grid(inp, k)
            if predicted != out:
                return None

        # All train pairs validated — predict test outputs
        predictions = []
        for test_inp in ctx.test_inputs:
            if len(test_inp) != 3 or any(len(r) != 3 for r in test_inp):
                return None

            k = _count_nonzero_colors(test_inp)
            if k == 0:
                return None

            result_grid = _scale_grid(test_inp, k)
            predictions.append(
                Prediction(
                    attempt_1=result_grid,
                    attempt_2=result_grid,
                    solver_name=self.name,
                    confidence=1.0,
                )
            )

        if not predictions:
            return None

        return SolverResult(
            task_id=ctx.task_id,
            predictions=predictions,
            solved=True,
            tier=2,
        )
