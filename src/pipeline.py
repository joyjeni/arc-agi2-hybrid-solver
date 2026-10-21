"""
Main pipeline orchestrator — runs all 3 tiers in sequence.

Tier 1: DSA / Vedic deterministic solvers    (~30–90 s, covers ~5–10%)
Tier 2: Program Synthesis via LLM           (~15–45 min, covers ~20–30%)
Tier 3: Chakra Ensemble direct prediction   (remaining budget, covers ~10–20%)

Total target: 35–45% on ARC-AGI-2 evaluation set.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .core.interfaces import SolverResult, TaskContext
from .core.submission import build_submission, score_submission, write_submission
from .solvers import REGISTRY
from .solvers.registry import SolverRegistry

logger = logging.getLogger(__name__)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_challenges(path: str | Path) -> dict[str, TaskContext]:
    """Load arc-agi test/eval challenges JSON → dict[task_id, TaskContext]."""
    data = json.loads(Path(path).read_text())
    tasks: dict[str, TaskContext] = {}
    for tid, task in data.items():
        tasks[tid] = TaskContext(
            task_id=tid,
            train_pairs=task["train"],
            test_inputs=[t["input"] for t in task["test"]],
        )
    return tasks


def load_solutions(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ── Tier-1 DSA runner ────────────────────────────────────────────────────────

def run_tier1(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    registry: SolverRegistry,
) -> dict[str, SolverResult]:
    """Run all registered deterministic solvers."""
    solvers = [s for s in registry.all_solvers() if s.name != "IdentitySolver"]
    solvers.sort(key=lambda s: s.priority)

    results: dict[str, SolverResult] = {}
    pending = {tid: ctx for tid, ctx in tasks.items() if tid not in already_solved}
    logger.info("Tier-1: %d solvers, %d tasks", len(solvers), len(pending))
    t0 = time.perf_counter()

    for tid, ctx in pending.items():
        for solver in solvers:
            try:
                res = solver._timed_solve(ctx)
                if res and res.solved:
                    results[tid] = res
                    logger.debug("T1 solved %s via %s", tid, solver.name)
                    break
            except Exception as e:
                logger.debug("T1 %s failed on %s: %s", solver.name, tid, e)

    elapsed = time.perf_counter() - t0
    logger.info("Tier-1 done in %.1fs: %d/%d solved", elapsed, len(results), len(pending))
    return results


# ── Tier-2 Program Synthesis runner ─────────────────────────────────────────

def run_tier2(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    llm_engine,
    time_budget_s: float = 2400.0,
) -> dict[str, SolverResult]:
    """Run program synthesis on unsolved tasks."""
    from .llm.program_synth import ProgramSynthesisSolver
    solver = ProgramSynthesisSolver(llm_engine)

    pending = {tid: ctx for tid, ctx in tasks.items() if tid not in already_solved}
    logger.info("Tier-2: %d tasks, budget=%.0fs", len(pending), time_budget_s)

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    for tid, ctx in pending.items():
        if time.perf_counter() - t0 > time_budget_s:
            logger.info("Tier-2: budget exhausted after %d tasks", len(results))
            break
        try:
            res = solver._timed_solve(ctx)
            if res and res.solved:
                results[tid] = res
                logger.info("T2 solved %s", tid)
        except Exception as e:
            logger.debug("T2 failed on %s: %s", tid, e)

    elapsed = time.perf_counter() - t0
    logger.info("Tier-2 done in %.1fs: %d/%d solved", elapsed, len(results), len(pending))
    return results


# ── Tier-3 Chakra Ensemble runner ─────────────────────────────────────────

def run_tier3(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    llm_engine,
    time_budget_s: float = 18000.0,
    n_templates: int = 4,
) -> dict[str, SolverResult]:
    """Run Chakra ensemble direct prediction on remaining tasks."""
    from .llm.chakra_ensemble import ChakraEnsembleSolver
    solver = ChakraEnsembleSolver(llm_engine, n_templates=n_templates)

    pending = {tid: ctx for tid, ctx in tasks.items() if tid not in already_solved}
    logger.info("Tier-3: %d tasks, budget=%.0fs", len(pending), time_budget_s)

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    for tid, ctx in pending.items():
        if time.perf_counter() - t0 > time_budget_s:
            logger.info("Tier-3: budget exhausted")
            break
        try:
            res = solver._timed_solve(ctx)
            if res and res.solved:
                results[tid] = res
        except Exception as e:
            logger.debug("T3 failed on %s: %s", tid, e)

    elapsed = time.perf_counter() - t0
    logger.info("Tier-3 done in %.1fs: %d/%d solved", elapsed, len(results), len(pending))
    return results


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    challenges_path: str,
    output_path: str = "submission.json",
    solutions_path: str | None = None,
    model_path: str | None = None,
    server_url: str | None = None,
    skip_tier2: bool = False,
    skip_tier3: bool = False,
    time_limit_s: float = 43200.0,   # 12 h Kaggle limit
) -> dict:
    """
    End-to-end pipeline.

    Returns the submission dict.
    """
    from .llm.engine import build_engine

    wall_start = time.perf_counter()
    margin = 900.0  # 15 min safety margin for file writing

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Loading challenges from %s", challenges_path)
    tasks = load_challenges(challenges_path)
    logger.info("Loaded %d tasks", len(tasks))

    all_results: dict[str, SolverResult] = {}

    # ── Tier 1: DSA ────────────────────────────────────────────────
    t1 = run_tier1(tasks, set(), REGISTRY)
    all_results.update(t1)
    solved = set(all_results.keys())
    logger.info("After T1: %d/%d solved (%.1f%%)", len(solved), len(tasks), 100*len(solved)/len(tasks))

    def remaining_s():
        return time_limit_s - (time.perf_counter() - wall_start) - margin

    # ── Tier 2: Program Synthesis ──────────────────────────────────
    if not skip_tier2 and remaining_s() > 60:
        llm = build_engine(model_path=model_path, server_url=server_url)
        budget2 = min(remaining_s() * 0.35, 2400.0)
        t2 = run_tier2(tasks, solved, llm, time_budget_s=budget2)
        all_results.update(t2)
        solved = set(all_results.keys())
        logger.info("After T2: %d/%d solved (%.1f%%)", len(solved), len(tasks), 100*len(solved)/len(tasks))
    else:
        llm = None

    # ── Tier 3: Chakra Ensemble ────────────────────────────────────
    if not skip_tier3 and remaining_s() > 60:
        if llm is None:
            llm = build_engine(model_path=model_path, server_url=server_url)
        budget3 = remaining_s()
        t3 = run_tier3(tasks, solved, llm, time_budget_s=budget3)
        all_results.update(t3)
        solved = set(all_results.keys())
        logger.info("After T3: %d/%d solved (%.1f%%)", len(solved), len(tasks), 100*len(solved)/len(tasks))

    # ── Build & write submission ───────────────────────────────────
    submission = build_submission(tasks, all_results, fallback_input=True)
    write_submission(submission, output_path, validate=True, tasks=tasks)

    # ── Optional scoring ──────────────────────────────────────────
    if solutions_path:
        solutions = load_solutions(solutions_path)
        acc = score_submission(submission, solutions)
        logger.info("Score vs ground truth: %.2f%%", acc * 100)

    total_elapsed = time.perf_counter() - wall_start
    logger.info("Pipeline done in %.1fs total", total_elapsed)
    return submission
