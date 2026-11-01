#!/usr/bin/env python3
"""
patch_trm.py — Apply TRM + signal.alarm integration to jeni_llm_fast.py
Patches:
  1. Add io + signal imports
  2. Insert TRM constants, PythonREPL, helpers, recursive_trm_loop
  3. Add _panel_start hook in worker panel loop
  4. Add TRM Strategy 5 + signal.alarm guard before hard fallback
"""
from pathlib import Path

TARGET = Path("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py")
src = TARGET.read_text()

# ─────────────────────────────────────────────────────────────────────────────
# Patch 1: Add io + signal imports
# ─────────────────────────────────────────────────────────────────────────────
OLD1 = "import random\nimport torch\nimport torch.multiprocessing as mp"
NEW1 = "import io\nimport random\nimport signal\nimport torch\nimport torch.multiprocessing as mp"
assert OLD1 in src, "Patch 1 anchor not found"
src = src.replace(OLD1, NEW1, 1)
print("Patch 1 OK: io + signal imports added")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 2: Insert full TRM section before "# ── System prompts"
# ─────────────────────────────────────────────────────────────────────────────
TRM_SECTION = '''
# ══════════════════════════════════════════════════════════════════════════════
# TRM  —  Test-time Recursive Modification  (Python DSL code-generation stage)
# ══════════════════════════════════════════════════════════════════════════════
#
# Pipeline:
#   1. Model generates a Python `transform(grid)` function using numpy.
#   2. PythonREPL executes it against every training pair inside the container.
#   3. If any pair fails, structured execution feedback is fed back to the model.
#   4. Repeats up to max_recursion_depth times until all pairs pass.
#   5. Up to 2 successful program variants → attempt_1 + attempt_2.
#
# Strategy 5 (last resort): triggered only when all prior strategies + D4
# transform-retry have failed to produce valid predictions AND sufficient
# wall-clock budget remains (_PANEL_BUDGET_SEC).
# ══════════════════════════════════════════════════════════════════════════════

_PANEL_BUDGET_SEC = 180      # hard wall-clock budget per panel (incl. TRM stage)


class TimeoutException(Exception):
    """Raised by SIGALRM handler when per-panel time budget is exceeded."""


def _timeout_handler(signum, frame):  # noqa: ANN001
    raise TimeoutException("Panel time budget exceeded")


class PythonREPL:
    """
    Lightweight isolated execution sandbox for TRM code validation.
    Executes model-generated `transform(grid: list) -> list` functions against
    ARC training pairs to check correctness before using the output.
    """

    def __init__(self):
        self.globals = {"np": __import__("numpy")}
        try:
            self.globals["scipy"] = __import__("scipy")
        except ImportError:
            pass

    def execute_code(self, code_str: str, input_grid: list):
        """
        Execute generated transform(grid) on input_grid.
        Returns the result as a nested Python list, or None on any error.
        """
        exec_ctx = f"{code_str}\\n\\n_result_grid = transform({input_grid!r})"
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            local_vars: dict = {}
            exec(exec_ctx, self.globals, local_vars)  # noqa: S102
            result = local_vars.get("_result_grid", None)
            if hasattr(result, "tolist"):   # normalise numpy → list
                result = result.tolist()
            return result
        except Exception:  # noqa: BLE001
            return None
        finally:
            sys.stdout = old_stdout


def _trm_format_prompt(
    task_data: dict,
    current_code: "str | None",
    feedback: "str | None",
) -> str:
    """Build a stateful model prompt for TRM code generation / refinement."""
    prompt = (
        "<|user|>\\nYou are a high-level recursive program synthesizer solving an ARC visual puzzle.\\n"
        "Write a Python function named `transform(grid: list[list[int]]) -> list[list[int]]` using numpy.\\n\\n"
    )
    for i, pair in enumerate(task_data["train"]):
        prompt += f"Example {i+1} Input:\\n{json.dumps(pair[\'input\'])}\\n"
        prompt += f"Example {i+1} Expected Output:\\n{json.dumps(pair[\'output\'])}\\n\\n"
    prompt += f"Test Target Input:\\n{json.dumps(task_data[\'test\'][\'input\'])}\\n\\n"
    if current_code and feedback:
        prompt += (
            f"### RECURSIVE REFINEMENT STEP ###\\n"
            f"Your previous code version failed verification:\\n```python\\n{current_code}\\n```\\n"
            f"Execution Feedback: {feedback}\\n"
            f"Please correct the function logic and output the entire rewritten function.\\n"
        )
    prompt += "Provide the python code block wrapped inside ```python and ```.\\n<|assistant|>\\n"
    return prompt


def _trm_extract_code(model_output: str) -> str:
    """Strip markdown fences to return the raw Python string block."""
    if "```python" in model_output:
        return model_output.split("```python")[1].split("```")[0].strip()
    if "```" in model_output:
        return model_output.split("```")[1].split("```")[0].strip()
    return model_output.strip()


def recursive_trm_loop(
    task_id: str,
    task_data: dict,
    model,
    tokenizer,
    max_recursion_depth: int = 4,
) -> dict:
    """
    TRM Core Engine — iterates code generation and REPL validation.

    Args:
        task_id:             Logging label only.
        task_data:           {"train": [{"input":..,"output":..},...],
                              "test":  {"input": grid}}
        model / tokenizer:   Already-loaded Qwen3-4B on the worker GPU.
        max_recursion_depth: Max generate → validate → refine cycles.

    Returns:
        {"attempt_1": grid, "attempt_2": grid}
    """
    repl = PythonREPL()
    current_code: "str | None" = None
    feedback_message: "str | None" = None
    fallback = task_data["test"]["input"]
    successful_programs: list = []

    for depth in range(max_recursion_depth):
        # 1. Build context-aware prompt (includes failure feedback when depth > 0)
        prompt = _trm_format_prompt(task_data, current_code, feedback_message)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=(depth > 0),
                temperature=0.2 if depth == 0 else 0.6,
                top_p=0.95 if depth > 0 else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        current_code = _trm_extract_code(generated_text)

        # 2. Validate against all training pairs in REPL
        all_passed = True
        failed_idx = -1
        observed_out = None

        for idx, pair in enumerate(task_data["train"]):
            test_res = repl.execute_code(current_code, pair["input"])
            if test_res is None or test_res != pair["output"]:
                all_passed = False
                failed_idx = idx
                observed_out = test_res
                break

        # 3. Evaluate recursion state
        if all_passed:
            final_out = repl.execute_code(current_code, task_data["test"]["input"])
            if final_out is not None:
                successful_programs.append(final_out)
                if len(successful_programs) >= 2:
                    break  # got both attempt slots — done
            # Ask for an alternate logic path to diversify attempt_2
            feedback_message = (
                "Generates valid solution. "
                "Provide an alternative transformation logic path using a different algorithm."
            )
        else:
            if observed_out is None:
                feedback_message = (
                    f"SyntaxError or execution crash on Example {failed_idx + 1}."
                )
            else:
                feedback_message = (
                    f"Logic mismatch on Example {failed_idx + 1}: "
                    f"generated output shape/values did not match expected."
                )

        print(
            f"[TRM] {task_id} depth={depth} all_passed={all_passed} "
            f"programs={len(successful_programs)}",
            flush=True,
        )

    # 4. Build submission candidates
    attempt_1 = successful_programs[0] if len(successful_programs) >= 1 else fallback
    attempt_2 = successful_programs[1] if len(successful_programs) >= 2 else attempt_1
    return {"attempt_1": attempt_1, "attempt_2": attempt_2}

'''

ANCHOR2 = "\n# ── System prompts ─────────────────────────────────────────────────────────────"
assert ANCHOR2 in src, "Patch 2 anchor not found"
src = src.replace(ANCHOR2, TRM_SECTION + ANCHOR2, 1)
print("Patch 2 OK: TRM section inserted")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 3: Add _panel_start timing hook after "attempt_2: Optional[list] = None"
# ─────────────────────────────────────────────────────────────────────────────
OLD3 = (
    "                attempt_2: Optional[list] = None\n"
    "\n"
    "                # ── A/B override: env var can force a single style ───────────────"
)
NEW3 = (
    "                attempt_2: Optional[list] = None\n"
    "                _panel_start = time.time()  # TRM budget: track elapsed time\n"
    "\n"
    "                # ── A/B override: env var can force a single style ───────────────"
)
assert OLD3 in src, "Patch 3 anchor not found"
src = src.replace(OLD3, NEW3, 1)
print("Patch 3 OK: _panel_start hook added")

# ─────────────────────────────────────────────────────────────────────────────
# Patch 4: Insert TRM Strategy 5 + signal.alarm guard before hard fallback
# ─────────────────────────────────────────────────────────────────────────────
OLD4 = (
    "                # ── Hard fallback: test input ─────────────────────────────────\n"
    "                if attempt_1 is None:\n"
    "                    attempt_1 = inp\n"
    "                if attempt_2 is None:\n"
    "                    attempt_2 = attempt_1"
)
NEW4 = (
    "                # ── Strategy 5: TRM code-generation (last resort) ─────────────\n"
    "                # Triggered only when prior strategies + transform-retry both\n"
    "                # failed AND we still have budget left within _PANEL_BUDGET_SEC.\n"
    "                _elapsed = time.time() - _panel_start\n"
    "                _budget_left = _PANEL_BUDGET_SEC - int(_elapsed)\n"
    "                if (\n"
    "                    (\n"
    "                        not _is_valid_result(attempt_1, inp)\n"
    "                        or not _is_valid_result(attempt_2, inp)\n"
    "                    )\n"
    "                    and _budget_left > 30\n"
    "                ):\n"
    "                    signal.signal(signal.SIGALRM, _timeout_handler)\n"
    "                    signal.alarm(max(30, _budget_left))\n"
    "                    try:\n"
    "                        _trm_data = {\"train\": task.get(\"train\", []), \"test\": panel}\n"
    "                        _trm_res = recursive_trm_loop(\n"
    "                            task_id, _trm_data, model, tokenizer, max_recursion_depth=4\n"
    "                        )\n"
    "                        if not _is_valid_result(attempt_1, inp):\n"
    "                            _cand = _trm_res[\"attempt_1\"]\n"
    "                            if _is_valid_result(_cand, inp):\n"
    "                                attempt_1 = _cand\n"
    "                                print(f\"[GPU{rank}] {task_id} TRM -> attempt_1 OK\", flush=True)\n"
    "                        if not _is_valid_result(attempt_2, inp):\n"
    "                            _cand2 = _trm_res[\"attempt_2\"]\n"
    "                            if _is_valid_result(_cand2, inp):\n"
    "                                attempt_2 = _cand2\n"
    "                                print(f\"[GPU{rank}] {task_id} TRM -> attempt_2 OK\", flush=True)\n"
    "                    except (TimeoutException, Exception) as _trm_exc:\n"
    "                        print(\n"
    "                            f\"[GPU{rank}] {task_id} TRM err ({type(_trm_exc).__name__}): \"\n"
    "                            f\"{_trm_exc}\",\n"
    "                            flush=True,\n"
    "                        )\n"
    "                    finally:\n"
    "                        signal.alarm(0)  # always cancel alarm\n"
    "\n"
    "                # ── Hard fallback: test input ─────────────────────────────────\n"
    "                if attempt_1 is None:\n"
    "                    attempt_1 = inp\n"
    "                if attempt_2 is None:\n"
    "                    attempt_2 = attempt_1"
)
assert OLD4 in src, "Patch 4 anchor not found"
src = src.replace(OLD4, NEW4, 1)
print("Patch 4 OK: TRM Strategy 5 + signal.alarm inserted")

# ─────────────────────────────────────────────────────────────────────────────
# Write result
# ─────────────────────────────────────────────────────────────────────────────
TARGET.write_text(src)
lines = src.count("\n") + 1
print(f"\nPatched file written: {TARGET} ({lines} lines)")
