"""Core interfaces and abstract base classes."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

Grid = list[list[int]]
Pair = dict[str, Grid]


@dataclass
class TaskContext:
    """Immutable context for one ARC task."""
    task_id: str
    train_pairs: list[Pair]
    test_inputs: list[Grid]
    # Stage-3 hints: best-guess grids from Qwen3-4B TTT, one per test input.
    # Set by run_stage3(); Stage-2 solvers use these as attempt_2 fallbacks.
    stage3_hints: list[Grid] | None = None


@dataclass
class Prediction:
    """Two-attempt prediction for one test input."""
    attempt_1: Grid
    attempt_2: Grid
    solver_name: str = "unknown"
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverResult:
    """Full result for one task (may have multiple test inputs)."""
    task_id: str
    predictions: list[Prediction]      # one per test input
    solved: bool = False
    tier: int = 0                      # 1=DSA, 2=ProgramSynth, 3=LLM


class BaseSolver(ABC):
    """Strategy interface — every solver must implement this contract."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def priority(self) -> int:
        """Lower = runs first. Override to control execution order."""
        return 100

    @abstractmethod
    def solve(self, ctx: TaskContext) -> SolverResult | None:
        """
        Attempt to solve the task.
        Returns SolverResult if confident, else None.
        """
        ...

    def _timed_solve(self, ctx: TaskContext) -> SolverResult | None:
        t0 = time.perf_counter()
        result = self.solve(ctx)
        elapsed = (time.perf_counter() - t0) * 1000
        if result:
            for p in result.predictions:
                p.elapsed_ms = elapsed
        return result


class EnsembleSolver(BaseSolver):
    """Composite: merges results from multiple child solvers."""

    def __init__(self, children: list[BaseSolver]):
        self._children = sorted(children, key=lambda s: s.priority)

    @property
    def name(self) -> str:
        return "EnsembleSolver"

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        for solver in self._children:
            try:
                result = solver._timed_solve(ctx)
                if result and result.solved:
                    return result
            except Exception:
                pass
        return None
