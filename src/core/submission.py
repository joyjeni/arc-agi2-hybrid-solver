"""Submission builder — writes competition-format submission.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .interfaces import Grid, Prediction, SolverResult, TaskContext

logger = logging.getLogger(__name__)


def build_submission(
    tasks: dict[str, TaskContext],
    results: dict[str, SolverResult],
    fallback_input: bool = True,
) -> dict:
    """
    Build a dict ready for json.dump matching the competition format:

        {
          "task_id": [
            {"attempt_1": [[...]], "attempt_2": [[...]]},
            ...                  # one entry per test input
          ]
        }

    Args:
        tasks:          {task_id: TaskContext}
        results:        {task_id: SolverResult}
        fallback_input: if True, copy input grid when no prediction exists.
    """
    submission: dict = {}

    for task_id, ctx in tasks.items():
        result = results.get(task_id)
        task_preds: list[dict] = []

        for i, test_input in enumerate(ctx.test_inputs):
            if result and i < len(result.predictions):
                pred = result.predictions[i]
                a1 = pred.attempt_1
                a2 = pred.attempt_2
            elif fallback_input:
                a1 = test_input
                a2 = test_input
            else:
                # empty 1×1 black
                a1 = [[0]]
                a2 = [[0]]

            task_preds.append({"attempt_1": a1, "attempt_2": a2})

        submission[task_id] = task_preds

    return submission


def validate_submission(sub: dict, tasks: dict[str, TaskContext]) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []
    for task_id, task_preds in sub.items():
        ctx = tasks.get(task_id)
        if ctx is None:
            errors.append(f"{task_id}: unknown task")
            continue
        n_test = len(ctx.test_inputs)
        if len(task_preds) != n_test:
            errors.append(
                f"{task_id}: expected {n_test} predictions, got {len(task_preds)}"
            )
            continue
        for j, pred in enumerate(task_preds):
            for key in ("attempt_1", "attempt_2"):
                if key not in pred:
                    errors.append(f"{task_id}[{j}]: missing {key}")
                    continue
                grid = pred[key]
                if not isinstance(grid, list) or not grid:
                    errors.append(f"{task_id}[{j}].{key}: not a list")
                    continue
                if not all(isinstance(r, list) for r in grid):
                    errors.append(f"{task_id}[{j}].{key}: rows not lists")
                    continue
                w0 = len(grid[0])
                for r in grid:
                    if len(r) != w0:
                        errors.append(f"{task_id}[{j}].{key}: non-rectangular")
                    for c in r:
                        if not (0 <= c <= 9):
                            errors.append(
                                f"{task_id}[{j}].{key}: cell {c} out of range"
                            )
    return errors


def write_submission(
    sub: dict,
    path: str | Path = "submission.json",
    validate: bool = True,
    tasks: Optional[dict] = None,
) -> None:
    """Write submission.json, optionally validating first."""
    if validate and tasks:
        errs = validate_submission(sub, tasks)
        if errs:
            logger.warning("Submission validation errors:\n  %s", "\n  ".join(errs[:20]))
    Path(path).write_text(json.dumps(sub), encoding="utf-8")
    logger.info("Wrote %s (%d tasks)", path, len(sub))


def score_submission(sub: dict, solutions: dict) -> float:
    """
    Compute exact-match accuracy against ground truth solutions.

    solutions format:
        { "task_id": [ [[grid]], [[grid]], ... ] }
        or
        { "task_id": [ [grid], ... ] }   (single-attempt ground truth)
    """
    total = solved = 0
    for task_id, gt_list in solutions.items():
        preds = sub.get(task_id, [])
        for i, gt in enumerate(gt_list):
            total += 1
            if i >= len(preds):
                continue
            # gt may be wrapped in an extra list
            if isinstance(gt[0][0], list):
                gt_grid = gt[0]
            else:
                gt_grid = gt
            pred = preds[i]
            if pred["attempt_1"] == gt_grid or pred["attempt_2"] == gt_grid:
                solved += 1
    return solved / total if total else 0.0
