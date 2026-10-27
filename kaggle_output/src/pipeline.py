"""
Main pipeline orchestrator — three stages in reversed priority order.

NEW ORDER (Stage 3 → Stage 2 → Stage 1):
  Stage 3 │ Qwen3-4B LLM + Test-Time Training  (per-task TTT, biggest budget)
  Stage 2 │ Pattern Induction Engine            (all registered solvers, parallel)
  Stage 1 │ Deterministic Rule Bank             (DSA / Vedic solvers, fast fallback)

Stage-3 predictions are stored in TaskContext.stage3_hints so Stage-2 and
Stage-1 solvers can use them as a fallback attempt_2 when their own match
produces only one confident answer.

Budget allocation (5 h wall-clock = 18 000 s):
  Stage 3  60 % → max 10 800 s
  Stage 2  25 % → max  4 500 s
  Stage 1  15 % → max    300 s (always fast/parallel)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _fut_wait
from pathlib import Path

from .core.interfaces import SolverResult, TaskContext
from .core.submission import build_submission, score_submission, write_submission
from .solvers import REGISTRY
from .solvers.registry import SolverRegistry

logger = logging.getLogger(__name__)

# ── Dynamic per-task time budget ──────────────────────────────────────────────

class TaskBudgetManager:
    """
    Thread-safe rolling-average per-task time budget.

    Budget = max(MIN_FLOOR_S, min(MAX_CAP_S, 2 × rolling_avg)).

    Only tasks that ran for at least MIN_RECORD_S are included in the
    average — instant no-match skips (elapsed ≈ 0) are ignored so the
    budget does not collapse toward zero.

    Usage inside _solve_one::

        t_task = time.perf_counter()
        for solver in solvers:
            elapsed = time.perf_counter() - t_task
            if budget_mgr.should_skip(elapsed):
                break
            res = solver._timed_solve(ctx, timeout_s=per_task_s)
            ...
        budget_mgr.record(time.perf_counter() - t_task)
    """

    MIN_FLOOR_S:   float = 2.0    # never below 2 s per task
    MAX_CAP_S:     float = 30.0   # never above 30 s per task
    MIN_RECORD_S:  float = 0.05   # (unused — only solved tasks are recorded)

    def __init__(self, max_cap_s: float = 30.0) -> None:
        self.MAX_CAP_S = max_cap_s
        self._lock = threading.Lock()
        self._times: list[float] = []
        self._skipped: int = 0

    # ------------------------------------------------------------------ #
    def record(self, elapsed_s: float) -> None:
        """Register a SOLVED task's wall time."""
        with self._lock:
            self._times.append(max(elapsed_s, 1e-6))

    def record_skip(self) -> None:
        with self._lock:
            self._skipped += 1

    def per_task_budget(self) -> float:
        """2 × rolling average of meaningful tasks, clamped to [MIN_FLOOR_S, MAX_CAP_S]."""
        with self._lock:
            if not self._times:
                return self.MAX_CAP_S
            avg = sum(self._times) / len(self._times)
        return max(self.MIN_FLOOR_S, min(self.MAX_CAP_S, avg * 2.0))

    def should_skip(self, elapsed_s: float) -> bool:
        """True when this task has already exceeded its dynamic budget."""
        return elapsed_s > self.per_task_budget()

    def remaining(self, elapsed_s: float) -> float:
        """Seconds left in budget for the current task (>= MIN_FLOOR_S)."""
        return max(self.MIN_FLOOR_S, self.per_task_budget() - elapsed_s)

    def stats(self) -> dict:
        with self._lock:
            n = len(self._times)
            avg = sum(self._times) / n if n else 0.0
            skipped = self._skipped
        return {
            "completed": n,
            "avg_s": round(avg, 3),
            "budget_s": round(self.per_task_budget(), 3),
            "skipped": skipped,
        }


# ── Solver-tier classification ────────────────────────────────────────────────
# Stage 1 = deterministic (dsa_solvers, vedic_solvers)
# Stage 2 = pattern induction (all other registered solvers)
# Stage 3 = LLM (handled separately, not from registry)

_STAGE1_MODULES = frozenset({"dsa_solvers", "vedic_solvers"})


def _solver_stage(solver) -> int:
    """Return 1 (deterministic) or 2 (pattern induction) for registered solvers."""
    mod = solver.__class__.__module__.split(".")[-1]
    return 1 if mod in _STAGE1_MODULES else 2


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


# ── Stage 3: Qwen3-4B TTT ─────────────────────────────────────────────────────

def run_stage3(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    llm_engine,                         # QwenEngine (for Chakra/ProgramSynth)
    time_budget_s: float = 10800.0,
    *,
    ttt_per_task_s: float = 120.0,
    skip_ttt: bool = False,
    skip_chakra: bool = False,
    skip_progsynth: bool = False,
    n_templates: int = 4,
) -> dict[str, SolverResult]:
    """
    Stage 3: LLM + Test-Time Training pipeline.

    Sub-stages (in order, budget shared):
      3a  Qwen3-4B TTT solver   (new — per-task gradient update)
      3b  Chakra Ensemble        (existing LLM, n_templates prompts)
      3c  Program Synthesis      (existing LLM, code-gen approach)

    Each sub-stage only touches tasks not yet solved by earlier sub-stages.
    Stage-3 hints (TTT best guesses) are written into ctx.stage3_hints for
    downstream stages to use as fallback attempt_2.
    """
    from .llm.qwen3_ttt_solver import Qwen3TTTSolver
    from .llm.chakra_ensemble import ChakraEnsembleSolver
    from .llm.program_synth import ProgramSynthesisSolver

    pending = {tid: ctx for tid, ctx in tasks.items() if tid not in already_solved}
    n_pending = len(pending)
    logger.info(
        "Stage-3: %d tasks | budget=%.0fs | ttt_per_task=%.0fs",
        n_pending, time_budget_s, ttt_per_task_s,
    )

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    def _remaining() -> float:
        return time_budget_s - (time.perf_counter() - t0)

    # ── 3a: Qwen3-4B TTT ──────────────────────────────────────────
    if not skip_ttt and _remaining() > 30:
        ttt_solver = Qwen3TTTSolver()
        solved_3a = set()
        for tid, ctx in list(pending.items()):
            if _remaining() < 30:
                break
            if _remaining() / max(1, n_pending - len(solved_3a)) < 10:
                break
            try:
                res = ttt_solver._timed_solve(ctx)
                if res and res.solved:
                    results[tid] = res
                    solved_3a.add(tid)
                    logger.debug("Stage-3a TTT solved %s", tid)
            except Exception as exc:
                logger.debug("Stage-3a TTT %s: %s", tid, exc)
        logger.info(
            "Stage-3a TTT: %d/%d solved (%.1fs elapsed)",
            len(solved_3a), n_pending, time.perf_counter() - t0,
        )

    solved_so_far = already_solved | set(results)

    # ── 3b: Chakra Ensemble ────────────────────────────────────────
    if not skip_chakra and _remaining() > 60:
        unsolved_chakra = [(tid, ctx) for tid, ctx in pending.items()
                           if tid not in solved_so_far]
        chakra_budget = min(_remaining() * 0.55, 7200.0)
        per_chakra = max(20.0, min(90.0, chakra_budget / max(len(unsolved_chakra), 1)))
        for tid, ctx in unsolved_chakra:
            if _remaining() < 60:
                break
            solver = ChakraEnsembleSolver(
                llm_engine, n_templates=n_templates, per_task_timeout=per_chakra
            )
            try:
                res = solver._timed_solve(ctx)
                if res and res.solved:
                    results[tid] = res
                    solved_so_far.add(tid)
                    logger.debug("Stage-3b Chakra solved %s", tid)
            except Exception as exc:
                logger.debug("Stage-3b Chakra %s: %s", tid, exc)
        logger.info(
            "Stage-3b Chakra: total Stage-3 now %d/%d (%.1fs elapsed)",
            len(results), n_pending, time.perf_counter() - t0,
        )

    # ── 3c: Program Synthesis ──────────────────────────────────────
    if not skip_progsynth and _remaining() > 30:
        unsolved_ps = [(tid, ctx) for tid, ctx in pending.items()
                       if tid not in solved_so_far]
        ps_budget = _remaining()
        per_ps = max(10.0, min(45.0, ps_budget / max(len(unsolved_ps), 1)))
        ps_solver = ProgramSynthesisSolver(llm_engine)
        for tid, ctx in unsolved_ps:
            if _remaining() < 30:
                break
            try:
                res = ps_solver._timed_solve(ctx)
                if res and res.solved:
                    results[tid] = res
                    solved_so_far.add(tid)
                    logger.debug("Stage-3c ProgramSynth solved %s", tid)
            except Exception as exc:
                logger.debug("Stage-3c PS %s: %s", tid, exc)
        logger.info(
            "Stage-3c PS: total Stage-3 now %d/%d (%.1fs elapsed)",
            len(results), n_pending, time.perf_counter() - t0,
        )

    elapsed = time.perf_counter() - t0
    logger.info("Stage-3 done in %.1fs: %d/%d solved", elapsed, len(results), n_pending)
    return results


# ── Stage 2: Pattern Induction (registered solvers) ──────────────────────────

def run_stage2(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    registry: SolverRegistry,
    *,
    total_budget_s: float = 4500.0,
    per_task_s: float = 8.0,
) -> dict[str, SolverResult]:
    """
    Stage 2: Pattern Induction — all registered non-deterministic solvers.

    Runs in parallel (ThreadPoolExecutor).  Tasks that have stage3_hints
    available use those as fallback attempt_2 when the solver produces only
    one grid. This is how Stage 2 "learns from Stage 3".
    """
    # Select pattern-induction solvers (exclude DSA/Vedic Stage-1 modules)
    all_solvers = [s for s in registry.all_solvers() if s.name != "IdentitySolver"]
    solvers = sorted(
        [s for s in all_solvers if _solver_stage(s) == 2],
        key=lambda s: s.priority,
    )

    pending = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    n_workers = min(8, max(len(pending), 1))
    logger.info(
        "Stage-2: %d pattern solvers × %d tasks | workers=%d budget=%.0fs",
        len(solvers), len(pending), n_workers, total_budget_s,
    )

    budget_mgr = TaskBudgetManager(max_cap_s=min(per_task_s, 30.0))

    def _solve_one(item):
        tid, ctx = item
        t_start = time.perf_counter()
        task_budget = budget_mgr.per_task_budget()
        for solver in solvers:
            elapsed = time.perf_counter() - t_start
            if elapsed > per_task_s:
                logger.debug("S2 hard-cap skip %s (%.2fs > %.2fs)", tid, elapsed, per_task_s)
                break
            if budget_mgr.should_skip(elapsed):
                logger.debug(
                    "S2 dyn-skip %s (%.2fs > budget %.2fs)", tid, elapsed, task_budget
                )
                budget_mgr.record_skip()
                break
            try:
                # Use static per_task_s as hard solver timeout to catch hangs.
                # Dynamic budget only controls the inter-iteration skip check above.
                res = solver._timed_solve(ctx, timeout_s=per_task_s)
                if res and res.solved:
                    # If Stage-3 produced hints, use them as attempt_2 fallback
                    if ctx.stage3_hints:
                        for i, pred in enumerate(res.predictions):
                            if (i < len(ctx.stage3_hints)
                                    and ctx.stage3_hints[i] is not None):
                                pred.attempt_2 = ctx.stage3_hints[i]
                    # Record only solved task times so the budget tracks
                    # real solve complexity, not failed-attempt noise.
                    budget_mgr.record(time.perf_counter() - t_start)
                    return tid, res
            except Exception as exc:
                logger.debug("S2 %s/%s: %s", solver.name, tid, exc)
        return tid, None

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_solve_one, item): item[0] for item in pending}
        done, not_done = _fut_wait(list(futures.keys()), timeout=total_budget_s)
        for fut in not_done:
            fut.cancel()
        for fut in done:
            try:
                tid, res = fut.result(timeout=0)
                if res:
                    results[tid] = res
            except Exception as exc:
                logger.debug("S2 result error: %s", exc)

    elapsed = time.perf_counter() - t0
    bstats = budget_mgr.stats()
    logger.info(
        "Stage-2 done in %.1fs: %d/%d solved (%d timed-out) | "
        "avg_task=%.3fs budget=%.3fs skipped=%d",
        elapsed, len(results), len(pending), len(not_done),
        bstats["avg_s"], bstats["budget_s"], bstats["skipped"],
    )
    return results


# ── Stage 1: Deterministic Rule Bank (DSA / Vedic) ───────────────────────────

def run_stage1(
    tasks: dict[str, TaskContext],
    already_solved: set[str],
    registry: SolverRegistry,
    *,
    total_budget_s: float = 300.0,
    per_task_s: float = 5.0,
) -> dict[str, SolverResult]:
    """
    Stage 1: Deterministic Rule Bank — DSA/Vedic solvers, fast parallel fallback.

    Runs last. These solvers are exact-match: either they fire or they don't.
    Stage-3 hints are injected as attempt_2 for tasks the deterministic solver
    partially matches (confidence > 0 but attempt_2 unavailable).
    """
    all_solvers = [s for s in registry.all_solvers() if s.name != "IdentitySolver"]
    solvers = sorted(
        [s for s in all_solvers if _solver_stage(s) == 1],
        key=lambda s: s.priority,
    )

    pending = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    n_workers = min(8, max(len(pending), 1))
    logger.info(
        "Stage-1: %d deterministic solvers × %d tasks | workers=%d budget=%.0fs",
        len(solvers), len(pending), n_workers, total_budget_s,
    )

    budget_mgr = TaskBudgetManager(max_cap_s=min(per_task_s, 30.0))

    def _solve_one(item):
        tid, ctx = item
        t_start = time.perf_counter()
        task_budget = budget_mgr.per_task_budget()
        for solver in solvers:
            elapsed = time.perf_counter() - t_start
            if elapsed > per_task_s:
                logger.debug("S1 hard-cap skip %s (%.2fs > %.2fs)", tid, elapsed, per_task_s)
                break
            if budget_mgr.should_skip(elapsed):
                logger.debug(
                    "S1 dyn-skip %s (%.2fs > budget %.2fs)", tid, elapsed, task_budget
                )
                budget_mgr.record_skip()
                break
            try:
                # Static per_task_s as hard solver timeout (catch hangs).
                res = solver._timed_solve(ctx, timeout_s=per_task_s)
                if res and res.solved:
                    # Inject Stage-3 hints as attempt_2 when present
                    if ctx.stage3_hints:
                        for i, pred in enumerate(res.predictions):
                            if (i < len(ctx.stage3_hints)
                                    and ctx.stage3_hints[i] is not None):
                                pred.attempt_2 = ctx.stage3_hints[i]
                    budget_mgr.record(time.perf_counter() - t_start)
                    return tid, res
            except Exception as exc:
                logger.debug("S1 %s/%s: %s", solver.name, tid, exc)
        return tid, None

    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_solve_one, item): item[0] for item in pending}
        done, not_done = _fut_wait(list(futures.keys()), timeout=total_budget_s)
        for fut in not_done:
            fut.cancel()
        for fut in done:
            try:
                tid, res = fut.result(timeout=0)
                if res:
                    results[tid] = res
            except Exception as exc:
                logger.debug("S1 result error: %s", exc)

    elapsed = time.perf_counter() - t0
    bstats = budget_mgr.stats()
    logger.info(
        "Stage-1 done in %.1fs: %d/%d solved (%d timed-out) | "
        "avg_task=%.3fs budget=%.3fs skipped=%d",
        elapsed, len(results), len(pending), len(not_done),
        bstats["avg_s"], bstats["budget_s"], bstats["skipped"],
    )
    return results


# ── Backward-compat aliases (benchmark runner uses run_tier1) ─────────────────

def run_tier1(tasks, already_solved, registry, *, total_budget_s=300.0, per_task_s=5.0):
    """Alias: run both Stage-2 and Stage-1 registered solvers sequentially."""
    r2 = run_stage2(tasks, already_solved, registry,
                    total_budget_s=total_budget_s * 0.70, per_task_s=per_task_s)
    solved = already_solved | set(r2)
    r1 = run_stage1(tasks, solved, registry,
                    total_budget_s=total_budget_s * 0.30, per_task_s=per_task_s)
    r2.update(r1)
    return r2


def run_tier2(tasks, already_solved, llm_engine, time_budget_s=2400.0, *, per_task_s=45.0):
    """Alias: program synthesis portion of Stage 3."""
    from .llm.program_synth import ProgramSynthesisSolver
    solver = ProgramSynthesisSolver(llm_engine)
    pending = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()
    for i, (tid, ctx) in enumerate(pending):
        if time_budget_s - (time.perf_counter() - t0) < 30:
            break
        try:
            res = solver._timed_solve(ctx)
            if res and res.solved:
                results[tid] = res
        except Exception:
            pass
    return results


def run_tier3(tasks, already_solved, llm_engine, time_budget_s=18000.0,
              n_templates=4, *, per_task_s=90.0):
    """Alias: Chakra ensemble portion of Stage 3."""
    from .llm.chakra_ensemble import ChakraEnsembleSolver
    pending = [(tid, ctx) for tid, ctx in tasks.items() if tid not in already_solved]
    results: dict[str, SolverResult] = {}
    t0 = time.perf_counter()
    for tid, ctx in pending:
        if time_budget_s - (time.perf_counter() - t0) < 60:
            break
        solver = ChakraEnsembleSolver(
            llm_engine, n_templates=n_templates, per_task_timeout=per_task_s
        )
        try:
            res = solver._timed_solve(ctx)
            if res and res.solved:
                results[tid] = res
        except Exception:
            pass
    return results


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    challenges_path: str,
    output_path: str = "submission.json",
    solutions_path: str | None = None,
    model_path: str | None = None,
    server_url: str | None = None,
    skip_stage3: bool = False,
    skip_stage2: bool = False,
    skip_stage1: bool = False,
    # Legacy kwargs (kept for backward compat with notebook callers)
    skip_tier2: bool = False,
    skip_tier3: bool = False,
    time_limit_s: float = 18000.0,   # 5 h competition wall-clock
) -> dict:
    """
    End-to-end pipeline (Stage 3 → Stage 2 → Stage 1).

    Stage 3  Qwen3-4B TTT + Chakra + ProgramSynth  60 % budget
    Stage 2  All registered pattern solvers         25 % budget
    Stage 1  DSA / Vedic deterministic              15 % budget (≤300 s)
    """
    from .llm.engine import build_engine

    wall_start = time.perf_counter()
    MARGIN = 300.0   # 5-min safety buffer

    def _remaining() -> float:
        return max(0.0, time_limit_s - (time.perf_counter() - wall_start) - MARGIN)

    # ── Load data ──────────────────────────────────────────────────
    logger.info("Loading challenges from %s", challenges_path)
    tasks = load_challenges(challenges_path)
    logger.info("Loaded %d tasks", len(tasks))

    all_results: dict[str, SolverResult] = {}

    # ── Stage 3: LLM + TTT ────────────────────────────────────────
    # skip_tier2/tier3 flags (legacy) also suppress Stage 3 sub-stages
    _skip_s3 = skip_stage3 or (skip_tier2 and skip_tier3)
    if not _skip_s3 and _remaining() > 60:
        llm = build_engine(model_path=model_path, server_url=server_url)
        s3_budget = min(_remaining() * 0.60, 10800.0)
        s3 = run_stage3(
            tasks,
            set(all_results),
            llm,
            time_budget_s=s3_budget,
            skip_ttt=False,
            skip_chakra=skip_tier3,
            skip_progsynth=skip_tier2,
        )
        all_results.update(s3)
        solved = set(all_results)
        logger.info(
            "After Stage-3: %d/%d (%.1f%%)",
            len(solved), len(tasks), 100 * len(solved) / len(tasks),
        )
    else:
        llm = None

    # ── Stage 2: Pattern Induction ────────────────────────────────
    if not skip_stage2 and _remaining() > 30:
        s2_budget = min(_remaining() * 0.70, 4500.0)   # 70% of what's left
        s2 = run_stage2(
            tasks,
            set(all_results),
            REGISTRY,
            total_budget_s=s2_budget,
            per_task_s=8.0,
        )
        all_results.update(s2)
        solved = set(all_results)
        logger.info(
            "After Stage-2: %d/%d (%.1f%%)",
            len(solved), len(tasks), 100 * len(solved) / len(tasks),
        )

    # ── Stage 1: Deterministic fallback ───────────────────────────
    if not skip_stage1 and _remaining() > 10:
        s1_budget = min(_remaining(), 300.0)
        s1 = run_stage1(
            tasks,
            set(all_results),
            REGISTRY,
            total_budget_s=s1_budget,
            per_task_s=5.0,
        )
        all_results.update(s1)
        solved = set(all_results)
        logger.info(
            "After Stage-1: %d/%d (%.1f%%)",
            len(solved), len(tasks), 100 * len(solved) / len(tasks),
        )

    # ── Build & write submission ───────────────────────────────────
    submission = build_submission(tasks, all_results, fallback_input=True)
    write_submission(submission, output_path, validate=True, tasks=tasks)

    # ── Optional scoring ──────────────────────────────────────────
    if solutions_path:
        solutions = load_solutions(solutions_path)
        acc = score_submission(submission, solutions)
        logger.info("Score vs ground truth: %.2f%%", acc * 100)

    total_elapsed = time.perf_counter() - wall_start
    logger.info(
        "Pipeline done in %.1fs (%.2fh)", total_elapsed, total_elapsed / 3600
    )
    return submission
