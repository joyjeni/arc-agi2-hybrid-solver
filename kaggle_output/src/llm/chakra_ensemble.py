"""
Chakra Ensemble Solver — Tier-3.

Runs multiple LLM prompt templates in parallel (thread pool),
collects grids, votes by exact-grid majority, returns top-2 as attempt_1/attempt_2.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.grid_codec import Grid
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext
from .engine import QwenEngine, _parse_grid_from_response
from .prompts import ALL_CHAKRA_TEMPLATES

logger = logging.getLogger(__name__)


class ChakraEnsembleSolver(BaseSolver):
    """
    Tier-3 direct-prediction solver.

    Uses ALL_CHAKRA_TEMPLATES (8 prompts) with ThreadPoolExecutor,
    votes on returned grids per test input.
    """

    def __init__(
        self,
        engine: QwenEngine,
        n_templates: int = 4,
        per_task_timeout: float = 300.0,
        max_workers: int = 4,
    ):
        self._engine = engine
        self._n_templates = min(n_templates, len(ALL_CHAKRA_TEMPLATES))
        self._timeout = per_task_timeout
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        return "ChakraEnsembleSolver"

    @property
    def priority(self) -> int:
        return 90

    def _predict_single(self, train_pairs: list[dict], test_input: Grid) -> Grid | None:
        """Run n_templates prompts concurrently and vote."""
        templates = ALL_CHAKRA_TEMPLATES[:self._n_templates]
        # Skip program-synthesis template (handled by ProgramSynthesisSolver)
        templates = [t for t in templates if t.__name__ != "chakra_4_prog_synth"]

        votes: dict[str, int] = {}
        grids: dict[str, Grid] = {}

        def run_template(fn):
            try:
                prompt = fn(train_pairs, test_input)
                resp = self._engine.generate(prompt, max_tokens=512, temperature=0.0)
                grid = _parse_grid_from_response(resp)
                return grid
            except Exception as e:
                logger.debug("Template %s failed: %s", fn.__name__, e)
                return None

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(run_template, fn): fn.__name__ for fn in templates}
            for fut in as_completed(futures, timeout=self._timeout):
                grid = fut.result()
                if grid is None:
                    continue
                import json
                key = json.dumps(grid, separators=(",", ":"))
                votes[key] = votes.get(key, 0) + 1
                grids[key] = grid

        if not votes:
            return None
        best = sorted(votes, key=votes.__getitem__, reverse=True)
        return grids[best[0]]

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        preds: list[Prediction] = []
        for test_inp in ctx.test_inputs:
            g1 = self._predict_single(ctx.train_pairs, test_inp)
            if g1 is None:
                return None
            # Second attempt: slightly higher temperature
            try:
                from .prompts import chakra_3_cot
                resp2 = self._engine.generate(
                    chakra_3_cot(ctx.train_pairs, test_inp),
                    max_tokens=512, temperature=0.3,
                )
                g2 = _parse_grid_from_response(resp2) or g1
            except Exception:
                g2 = g1

            preds.append(Prediction(
                attempt_1=g1, attempt_2=g2,
                solver_name=self.name,
                confidence=0.6,
            ))

        return SolverResult(
            task_id=ctx.task_id, predictions=preds,
            solved=True, tier=3,
        )
