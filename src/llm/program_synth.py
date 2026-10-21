"""
Program Synthesis solver — Tier-2.

Prompts the LLM to write a `solve(grid)` Python function, executes it
in a sandboxed subprocess, validates against training pairs, then applies
to test inputs.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import textwrap

from ..core.grid_codec import Grid
from ..core.interfaces import BaseSolver, Prediction, SolverResult, TaskContext

logger = logging.getLogger(__name__)

# Maximum execution time for synthesized code (seconds)
EXEC_TIMEOUT = 5
MAX_CODE_RETRIES = 3


def _extract_code(text: str) -> str | None:
    """Extract Python code from LLM output (markdown fenced or raw)."""
    # Try ```python ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try to find a def solve(...) block
    m = re.search(r"(def solve\(.*?)(?=\nRESULT:|\Z)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _extract_result_json(text: str) -> Grid | None:
    """Extract the RESULT: [...] line from LLM output."""
    m = re.search(r"RESULT:\s*(\[.*?\])\s*$", text, re.DOTALL | re.MULTILINE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


def _run_code_sandboxed(code: str, test_grid: Grid, timeout: int = EXEC_TIMEOUT) -> Grid | None:
    """Run synthesized `solve(grid)` in a subprocess and return the result."""
    script = textwrap.dedent(f"""
import json, sys
{code}
test_input = {json.dumps(test_grid)}
try:
    result = solve(test_input)
    print(json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout.strip())
        if isinstance(result, list):
            return result
        return None
    except Exception:
        return None
    finally:
        import os
        try:
            os.unlink(fname)
        except Exception:
            pass


def validate_code_on_pairs(code: str, train_pairs: list[dict]) -> bool:
    """Return True only if synthesized code passes all training pairs."""
    for pair in train_pairs:
        result = _run_code_sandboxed(code, pair["input"])
        if result != pair["output"]:
            return False
    return True


class ProgramSynthesisSolver(BaseSolver):
    """
    Tier-2 solver using LLM-generated Python code.
    Uses an injected LLMEngine — no direct model weights here.
    """

    def __init__(self, llm_engine, max_retries: int = MAX_CODE_RETRIES):
        self._llm = llm_engine
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return "ProgramSynthesisSolver"

    @property
    def priority(self) -> int:
        return 50  # runs after Tier-1 DSA solvers

    def solve(self, ctx: TaskContext) -> SolverResult | None:
        from .prompts import chakra_4_prog_synth

        for attempt in range(self._max_retries):
            # Generate one prompt per test input (re-use same code for all)
            prompt = chakra_4_prog_synth(ctx.train_pairs, ctx.test_inputs[0])
            response = self._llm.generate(prompt, max_tokens=512, temperature=0.1 * attempt)

            code = _extract_code(response)
            if code is None:
                logger.debug("ProgramSynth: no code extracted (attempt %d)", attempt)
                continue

            if not validate_code_on_pairs(code, ctx.train_pairs):
                logger.debug("ProgramSynth: code failed validation (attempt %d)", attempt)
                continue

            # Code is valid — apply to all test inputs
            preds: list[Prediction] = []
            all_ok = True
            for test_inp in ctx.test_inputs:
                result = _run_code_sandboxed(code, test_inp)
                if result is None:
                    all_ok = False
                    break
                preds.append(Prediction(
                    attempt_1=result,
                    attempt_2=result,
                    solver_name=self.name,
                    confidence=0.9,
                    metadata={"code_attempt": attempt},
                ))
            if all_ok:
                return SolverResult(
                    task_id=ctx.task_id, predictions=preds,
                    solved=True, tier=2,
                )
        return None
