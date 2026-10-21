"""
Main pipeline orchestrator — runs all 3 tiers in sequence.

Tier 1: DSA / Vedic deterministic solvers    (parallel, ≤300 s total, ≤5 s/task)
Tier 2: Program Synthesis via LLM           (adaptive fair-share, ≤45 s/task)
Tier 3: Chakra Ensemble direct prediction   (adaptive fair-share, ≤90 s/task)

Total target: 35–45 % on ARC-AGI-2 evaluation set within 5 h wall-clock.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait as _fut_wait
from pathlib import Path

from .core.interfaces import SolverResult, TaskContext
from .core.submission import build_submission, score_submission, write_submission
from .solvers import REGISTRY
from .solvers.registry import SolverRegistry

logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── Tier-1 DSA runner (parallel) ─────────────────────────────────────────────

def run_tier1(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    registry: SolverRegistry,
    *,
    total_budget_s: float = 300.0,
    per_task_s: float = 5.0,
) -> dict[str, SolverResult]:
    """
    Run all registered deterministic solvers in parallel.

    Uses ThreadPoolExecutor so CPU-bound solvers run concurrently.
    Hard wall-clock budget: futures that haven't finished by ``total_budget_s``
    are cancelled and skipped — no hanging.
    """
    solvers = [s for s in registry.all_solvers() if s.name != "IdentitySolver"]
    solvers.sort(key=lambda s: s.priority)

    pending = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    n_workers = min(8, max(len(pending), 1))
    logger.info(
        "Tier-1: %d solvers × %d tasks | workers=%d budget=%.0fs per_task=%.0fs",
        len(solvers), len(pending), n_workers, total_budget_s, per_task_s,
    )

    def _solve_one(item: tuple) -> tuple[str, SolverResult | None]:
        tid, ctx = item
        t_start = time.perf_counter()
        for solver in solvers:
            # Honour per-task budget inside the worker thread
            if time.perf_counter() - t_start > per_task_s:
                logger.debug("T1 per-task budget hit on %s", tid)
                break
            try:
                res = solver._timed_solve(ctx)
                if res and res.solved:
                    return tid, res
            except Exception as exc:
                logger.debug("T1 %s/%s: %s", solver.name, tid, exc)
        return tid, None

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_solve_one, item): item[0] for item in pending}
        done, not_done = _fut_wait(list(futures.keys()), timeout=total_budget_s)

        # Cancel futures that haven't started yet
        for fut in not_done:
            fut.cancel()

        for fut in done:
            try:
                tid, res = fut.result(timeout=0)
                if res:
                    results[tid] = res
                    logger.debug("T1 solved %s", tid)
            except Exception as exc:
                logger.debug("T1 result error: %s", exc)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Tier-1 done in %.1fs: %d/%d solved (%d timed-out / cancelled)",
        elapsed, len(results), len(pending), len(not_done),
    )
    return results


# ── Tier-2 Program Synthesis runner ──────────────────────────────────────────

def run_tier2(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    llm_engine,
    time_budget_s: float = 2400.0,
    *,
    per_task_s: float = 45.0,
) -> dict[str, SolverResult]:
    """
    Run program synthesis with adaptive fair-share per-task budgeting.

    Each task receives ``min(per_task_s, remaining_budget / tasks_remaining)``
    seconds.  The loop exits early whenever the budget or fair-share falls below
    the 10 s minimum.  Hard interrupt (SIGALRM) is applied by the notebook caller
    around this function for an additional safety layer.
    """
    from .llm.program_synth import ProgramSynthesisSolver

    solver = ProgramSynthesisSolver(llm_engine)
    pending_items = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    n_pending = len(pending_items)
    logger.info(
        "Tier-2: %d tasks | budget=%.0fs | max_per_task=%.0fs",
        n_pending, time_budget_s, per_task_s,
    )

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    for i, (tid, ctx) in enumerate(pending_items):
        elapsed = time.perf_counter() - t0
        remaining = time_budget_s - elapsed
        if remaining < 30.0:
            logger.info("Tier-2: budget exhausted at task %d/%d", i, n_pending)
            break

        # Fair-share: distribute remaining budget equally over remaining tasks
        tasks_left = n_pending - i
        fair_share = remaining / tasks_left
        actual_per_task = min(per_task_s, fair_share)
        if actual_per_task < 10.0:
            logger.info("Tier-2: fair-share %.1fs < 10 s, stopping", actual_per_task)
            break

        logger.debug("T2 [%d/%d] %s | slot=%.0fs", i + 1, n_pending, tid, actual_per_task)
        t_task = time.perf_counter()
        try:
            res = solver._timed_solve(ctx)
            task_elapsed = time.perf_counter() - t_task
            if res and res.solved:
                results[tid] = res
                logger.info("T2 solved %s in %.1fs", tid, task_elapsed)
            if task_elapsed > actual_per_task * 1.5:
                logger.warning(
                    "T2 %s overran slot: %.1fs > %.0fs "
                    "(signal.SIGALRM in notebook should have caught this)",
                    tid, task_elapsed, actual_per_task,
                )
        except Exception as exc:
            logger.debug("T2 %s error: %s", tid, exc)

    elapsed = time.perf_counter() - t0
    logger.info("Tier-2 done in %.1fs: %d/%d solved", elapsed, len(results), n_pending)
    return results


# ── Tier-3 Chakra Ensemble runner ─────────────────────────────────────────────

def run_tier3(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    llm_engine,
    time_budget_s: float = 18000.0,
    n_templates: int = 4,
    *,
    per_task_s: float = 90.0,
) -> dict[str, SolverResult]:
    """
    Run Chakra ensemble with adaptive fair-share per-task budgeting.

    ``per_task_s`` is also forwarded to ``ChakraEnsembleSolver`` as its internal
    template-futures timeout so the ensemble's own thread pool is bounded.
    Hard interrupt is applied by the notebook caller via SIGALRM.
    """
    from .llm.chakra_ensemble import ChakraEnsembleSolver

    pending_items = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    n_pending = len(pending_items)
    logger.info(
        "Tier-3: %d tasks | budget=%.0fs | max_per_task=%.0fs | templates=%d",
        n_pending, time_budget_s, per_task_s, n_templates,
    )

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    for i, (tid, ctx) in enumerate(pending_items):
        elapsed = time.perf_counter() - t0
        remaining = time_budget_s - elapsed
        if remaining < 60.0:
            logger.info("Tier-3: budget exhausted at task %d/%d", i, n_pending)
            break

        tasks_left = n_pending - i
        fair_share = remaining / tasks_left
        actual_per_task = min(per_task_s, fair_share)
        if actual_per_task < 20.0:
            logger.info("Tier-3: fair-share %.1fs < 20 s, stopping", actual_per_task)
            break

        # Rebuild solver per task so per_task_timeout is always tight
        solver = ChakraEnsembleSolver(
            llm_engine,
            n_templates=n_templates,
            per_task_timeout=actual_per_task,
        )

        logger.debug("T3 [%d/%d] %s | slot=%.0fs", i + 1, n_pending, tid, actual_per_task)
        t_task = time.perf_counter()
        try:
            res = solver._timed_solve(ctx)
            task_elapsed = time.perf_counter() - t_task
            if res and res.solved:
                results[tid] = res
                logger.info("T3 solved %s in %.1fs", tid, task_elapsed)
        except Exception as exc:
            logger.debug("T3 %s error: %s", tid, exc)

    elapsed = time.perf_counter() - t0
    logger.info("Tier-3 done in %.1fs: %d/%d solved", elapsed, len(results), n_pending)
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
    time_limit_s: float = 18000.0,   # 5 h competition wall-clock
) -> dict:
    """End-to-end pipeline. Returns the submission dict."""
    from .llm.engine import build_engine

    wall_start = time.perf_counter()
    MARGIN = 300.0  # 5-min safety buffer for file I/O

    def _remaining() -> float:
        return time_limit_s - (time.perf_counter() - wall_start) - MARGIN

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Loading challenges from %s", challenges_path)
    tasks = load_challenges(challenges_path)
    logger.info("Loaded %d tasks", len(tasks))

    all_results: dict[str, SolverResult] = {}

    # ── Tier 1: DSA (parallel) ─────────────────────────────────────
    t1_budget = min(300.0, _remaining() * 0.05)
    t1 = run_tier1(tasks, set(), REGISTRY, total_budget_s=t1_budget, per_task_s=5.0)
    all_results.update(t1)
    solved = set(all_results.keys())
    logger.info("After T1: %d/%d (%.1f%%)", len(solved), len(tasks), 100 * len(solved) / len(tasks))

    # ── Tier 2: Program Synthesis ──────────────────────────────────
    if not skip_tier2 and _remaining() > 60:
        llm = build_engine(model_path=model_path, server_url=server_url)
        budget2 = min(_remaining() * 0.40, 7200.0)
        n_unsolved = len(tasks) - len(solved)
        per_t2 = max(10.0, min(45.0, budget2 / max(n_unsolved, 1)))
        t2 = run_tier2(tasks, solved, llm, time_budget_s=budget2, per_task_s=per_t2)
        all_results.update(t2)
        solved = set(all_results.keys())
        logger.info("After T2: %d/%d (%.1f%%)", len(solved), len(tasks), 100 * len(solved) / len(tasks))
    else:
        llm = None

    # ── Tier 3: Chakra Ensemble ────────────────────────────────────
    if not skip_tier3 and _remaining() > 60:
        if llm is None:
            llm = build_engine(model_path=model_path, server_url=server_url)
        budget3 = _remaining()
        n_unsolved = len(tasks) - len(solved)
        per_t3 = max(20.0, min(90.0, budget3 / max(n_unsolved, 1)))
        t3 = run_tier3(tasks, solved, llm, time_budget_s=budget3, per_task_s=per_t3)
        all_results.update(t3)
        solved = set(all_results.keys())
        logger.info("After T3: %d/%d (%.1f%%)", len(solved), len(tasks), 100 * len(solved) / len(tasks))

    # ── Build & write submission ───────────────────────────────────
    submission = build_submission(tasks, all_results, fallback_input=True)
    write_submission(submission, output_path, validate=True, tasks=tasks)

    # ── Optional scoring ──────────────────────────────────────────
    if solutions_path:
        solutions = load_solutions(solutions_path)
        acc = score_submission(submission, solutions)
        logger.info("Score vs ground truth: %.2f%%", acc * 100)

    total_elapsed = time.perf_counter() - wall_start
    logger.info("Pipeline done in %.1fs (%.2fh)", total_elapsed, total_elapsed / 3600)
    return submission
