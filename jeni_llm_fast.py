#!/usr/bin/env python3
"""
jeni_llm_fast.py  —  LLM fast-inference pass for ARC-AGI-2
=========================================================
Uses up to 4x L4 GPUs (or however many CUDA devices are visible).
Each worker process loads Qwen3-4B on its own GPU and processes a slice of tasks.

Output: /kaggle/working/jeni_llm_preds.json
  Format: {task_id: [{attempt_1: [[int,...]], attempt_2: [[int,...]]}, ...]}

Design (matches SFT training format):
  • attempt_1 ← greedy NVARC-format (T=0, compact multi-turn, no system msg)
  • attempt_2 ← compact+CoT (T=0.7, thinking mode, 512 tok reason)
  • fallback   ← compact prompt (T=0.4) when either failed
  • hard fallback ← copy of test input when all parsing fails

Merge priority in the notebook's merge cell:
  1. NVARC submission.json  → attempt_1   (highest trust)
  2. DSA  dsa_solved.json   → attempt_2   (deterministic)
  3. LLM  jeni_llm_preds.json → attempt_2 diversity (fill gaps)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

import io
import random
import signal
import torch
import torch.multiprocessing as mp


# ── Color map ─────────────────────────────────────────────────────────────────
COLOR_NAMES = {
    0: "black", 1: "blue",  2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "magenta", 7: "orange", 8: "azure", 9: "maroon",
}

# ── Grid utilities ─────────────────────────────────────────────────────────────

def grid_to_digits(grid: list) -> str:
    """Space-separated rows: '0 1 2\\n3 4 5'"""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_to_compact(grid: list) -> str:
    """No-space rows: '012\\n345'"""
    return "\n".join("".join(str(c) for c in row) for row in grid)


def grid_to_colored(grid: list) -> str:
    """'black blue\\nred green' label rows"""
    return "\n".join(
        " ".join(COLOR_NAMES.get(c, str(c)) for c in row) for row in grid
    )



def convert_grid_to_string(grid) -> str:
    """NVARC/SFT format: rows of compact digits separated by newlines (no spaces).
    Matches the exact format the qwen3_4b_grids15_sft139 model was SFT-trained on.
    """
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()

def parse_grid(text: str, ref_shape: Optional[tuple] = None) -> Optional[list]:
    """
    Parse LLM output into a 2-D integer grid.
    Strips <think>...</think> blocks and markdown fences first.
    Returns None if no valid grid can be extracted.
    """
    # Strip thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    text = re.sub(r"```[^\n]*\n?", "", text)
    text = text.strip()

    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    rows: list[list[int]] = []
    for line in lines:
        # Space-separated single digits
        parts = line.split()
        if parts and all(p.isdigit() and len(p) == 1 for p in parts):
            rows.append([int(p) for p in parts])
            continue
        # Compact: no spaces, all digits
        compact = line.replace(" ", "")
        if compact and all(c.isdigit() for c in compact):
            rows.append([int(c) for c in compact])
            continue
        # Comma-separated
        parts_c = [p.strip() for p in line.split(",")]
        if parts_c and all(p.isdigit() and len(p) == 1 for p in parts_c):
            rows.append([int(p) for p in parts_c])
            continue
        # Bracket / mixed: extract all single digits on this line
        digits = re.findall(r"\b\d\b", line)
        if digits and len(digits) >= 2:
            rows.append([int(d) for d in digits])
            continue

    if not rows:
        return None

    # Use most-common row length
    most_common_len = Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == most_common_len]

    # Validate values 0-9
    for row in rows:
        if any(v < 0 or v > 9 for v in row):
            return None

    # Clip / pad to ref_shape
    if ref_shape is not None:
        h, w = ref_shape
        rows = rows[:h]
        rows = [r[:w] + [0] * max(0, w - len(r)) for r in rows]
        while len(rows) < h:
            rows.append([0] * w)

    return rows if rows else None




# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL-TRANSFORM RETRY HELPERS
# If the ensemble fails to produce a valid grid, the worker retries inference
# on spatially-transformed versions of the puzzle and inverse-transforms back.
# Uses a subset of D4 (rot90 × {1,2,3} + fliplr) — 4 cheap retries per task.
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as _np_aug  # alias to avoid collision

_TRANSFORM_OPS = [
    ("rot90_1",   lambda g: _np_aug.rot90(_np_aug.array(g), 1).tolist(),
                  lambda g: _np_aug.rot90(_np_aug.array(g), -1).tolist()),
    ("rot90_2",   lambda g: _np_aug.rot90(_np_aug.array(g), 2).tolist(),
                  lambda g: _np_aug.rot90(_np_aug.array(g), 2).tolist()),
    ("rot90_3",   lambda g: _np_aug.rot90(_np_aug.array(g), 3).tolist(),
                  lambda g: _np_aug.rot90(_np_aug.array(g), -3).tolist()),
    ("fliplr",    lambda g: _np_aug.fliplr(_np_aug.array(g)).tolist(),
                  lambda g: _np_aug.fliplr(_np_aug.array(g)).tolist()),
]


def _transform_task_view(task_view: dict, fwd_fn) -> dict:
    """Apply a spatial transform to all train + test grids in a task view."""
    def _t(grid):
        try:
            return fwd_fn(grid)
        except Exception:
            return grid
    new_train = [
        {"input": _t(p["input"]), "output": _t(p["output"])}
        for p in task_view.get("train", [])
    ]
    new_test = [{"input": _t(p["input"])} for p in task_view.get("test", [])]
    return {"train": new_train, "test": new_test}


def _is_valid_result(grid, test_input) -> bool:
    """Return True if grid looks like a real prediction (not a trivial copy)."""
    if grid is None:
        return False
    if grid == test_input:          # identical to input → likely a failure
        return False
    flat = [c for row in grid for c in row]
    if len(flat) == 0:
        return False
    if all(v == 0 for v in flat):   # all-zero → failure
        return False
    return True

def _pixel_majority_vote_fast(candidates: list):
    """Return the candidate grid that appears most often (pixel-level majority vote).
    Inline copy for jeni_llm_fast.py — does not import from trm_orchestrator.
    Falls back to first valid candidate if no majority; None if list is empty.
    """
    if not candidates:
        return None
    valid = [g for g in candidates if g is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    counts: Counter = Counter()
    for g in valid:
        counts[tuple(tuple(row) for row in g)] += 1
    best_key, _ = counts.most_common(1)[0]
    return [list(row) for row in best_key]


def _demo_consistency_score_fast(grid: list, train_pairs: list) -> float:
    """Return [0, 1]: fraction of training outputs whose shape matches grid.
    Inline copy for jeni_llm_fast.py — does not import from trm_orchestrator.
    """
    if grid is None or not train_pairs:
        return 0.0
    g_rows, g_cols = len(grid), (len(grid[0]) if grid else 0)
    return sum(
        1 for p in train_pairs
        if p.get("output")
        and len(p["output"]) == g_rows
        and len(p["output"][0]) == g_cols
    ) / max(len(train_pairs), 1)



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



class ARCDSL:
    """Static ARC Domain-Specific Language primitives exposed to REPL sandbox."""
    @staticmethod
    def to_np(grid):
        return np.array(grid, dtype=int)

    @staticmethod
    def flood_fill(grid, start_r, start_c, fill_color):
        from scipy import ndimage as _nd
        arr = np.array(grid, dtype=int)
        H, W = arr.shape
        if not (0 <= start_r < H and 0 <= start_c < W):
            return arr.tolist()
        target_color = arr[start_r, start_c]
        if target_color == fill_color:
            return arr.tolist()
        mask = (arr == target_color)
        labels, _ = _nd.label(mask)
        if labels[start_r, start_c] > 0:
            arr[labels == labels[start_r, start_c]] = fill_color
        return arr.tolist()

    @staticmethod
    def get_objects(grid, connectivity=8):
        from scipy import ndimage as _nd
        arr = np.array(grid, dtype=int)
        objects = []
        for color in np.unique(arr):
            if color == 0:
                continue
            labels, num = _nd.label(arr == color)
            for lab in range(1, num + 1):
                rc = np.argwhere(labels == lab)
                min_r, min_c = rc.min(axis=0)
                max_r, max_c = rc.max(axis=0)
                objects.append({
                    "color": int(color),
                    "bounds": [int(min_r), int(max_r), int(min_c), int(max_c)]
                })
        return objects

    @staticmethod
    def crop_to_content(grid, bg_color=0):
        arr = np.array(grid, dtype=int)
        non_bg = np.argwhere(arr != bg_color)
        if len(non_bg) == 0:
            return [[bg_color]]
        min_r, min_c = non_bg.min(axis=0)
        max_r, max_c = non_bg.max(axis=0)
        return arr[min_r:max_r+1, min_c:max_c+1].tolist()


class PythonREPL:
    """
    Lightweight isolated execution sandbox for TRM code validation.
    Executes model-generated `transform(grid: list) -> list` functions against
    ARC training pairs to check correctness before using the output.
    """

    def __init__(self):
        import numpy as _np
        try:
            from scipy import ndimage as _ndimage
        except ImportError:
            _ndimage = None
        def _get_connected_components(grid):
            """4-connected component labelling (Duck Harness segmentation helper)."""
            import numpy as __np
            g = __np.array(grid)
            rows, cols = g.shape
            visited = __np.zeros_like(g, dtype=bool)
            label = 0
            labels = __np.full_like(g, -1)
            for r in range(rows):
                for c in range(cols):
                    if not visited[r, c]:
                        color = g[r, c]
                        stack = [(r, c)]
                        visited[r, c] = True
                        cells = []
                        while stack:
                            cr, cc = stack.pop()
                            cells.append((cr, cc))
                            labels[cr, cc] = label
                            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                                nr, nc = cr+dr, cc+dc
                                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and g[nr,nc]==color:
                                    visited[nr,nc] = True
                                    stack.append((nr,nc))
                        label += 1
            return labels.tolist()

        self.globals = {
            "np": _np,
            "ndimage": _ndimage,
            "flood_fill": ARCDSL.flood_fill,
            "get_objects": ARCDSL.get_objects,
            "crop_to_content": ARCDSL.crop_to_content,
            "get_connected_components": _get_connected_components,
        }

    def execute_code(self, code_str: str, input_grid: list):
        """
        Execute generated transform(grid) on input_grid.
        Returns the result as a nested Python list, or None on any error.
        """
        exec_ctx = f"{code_str}\n\n_result_grid = transform({input_grid!r})"
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
        "<|user|>\nYou are a high-level recursive program synthesizer solving an ARC visual puzzle.\n"
        "Write a Python function named `transform(grid: list[list[int]]) -> list[list[int]]` using numpy.\n\n"
        "Solving strategies (apply the most relevant one):\n"
        "• Vedic Anurupyena (proportionality): detect ratio-based scaling / tiling rules.\n"
        "• Vedic Sunyam Samyasamuccaye (symmetry): find mirror / palindrome / rotation structure.\n"
        "• Vedic Vilokanam (direct observation): spot the simplest single-rule transform.\n"
        "• Japanese Kaizen (iterative steps): look for a rule applied cell-by-cell or row-by-row.\n"
        "• Japanese Kanban (pull-based): each output cell is determined by its colored neighbor.\n"
        "• Chinese Nine-Palace (九宫格): map 3×3 block spatial relationships and symmetry axes.\n"
        "• Nature-inspired: detect flood-fill propagation, cellular automata, or fractal self-similarity.\n\n"
    )
    for i, pair in enumerate(task_data["train"]):
        prompt += f"Example {i+1} Input:\n{json.dumps(pair['input'])}\n"
        prompt += f"Example {i+1} Expected Output:\n{json.dumps(pair['output'])}\n\n"
    prompt += f"Test Target Input:\n{json.dumps(task_data['test']['input'])}\n\n"
    if current_code and feedback:
        prompt += (
            f"### RECURSIVE REFINEMENT STEP ###\n"
            f"Your previous code version failed verification:\n```python\n{current_code}\n```\n"
            f"Execution Feedback: {feedback}\n"
            f"Please correct the function logic and output the entire rewritten function.\n"
        )
    prompt += "Provide the python code block wrapped inside ```python and ```.\n<|assistant|>\n"
    return prompt


def _trm_extract_code(model_output: str) -> str:
    """Strip markdown fences to return the raw Python string block."""
    if "```python" in model_output:
        return model_output.split("```python")[1].split("```")[0].strip()
    if "```" in model_output:
        return model_output.split("```")[1].split("```")[0].strip()
    return model_output.strip()



def verify_program_robustness(repl, code_str, train_pairs):
    """
    Validate a generated transform() against all training pairs.
    Also tests D4 rotational symmetry to reject brittle programs.
    Returns (passed: bool, reason: str).
    """
    for idx, pair in enumerate(train_pairs):
        base_res = repl.execute_code(code_str, pair["input"])
        if base_res is None or base_res != pair["output"]:
            return False, f"Fails base pair verification on step {idx + 1}"
        try:
            rot_in  = np.rot90(np.array(pair["input"]),  1).tolist()
            rot_out = np.rot90(np.array(pair["output"]), 1).tolist()
            if repl.execute_code(code_str, rot_in) != rot_out:
                return False, f"Fails D4 rotational symmetry on pair {idx + 1}"
        except Exception:
            return False, "Symmetry validation parse fault"
    return True, "OK"


def check_demo_success(repl: "PythonREPL", code_str: str, train_pairs: list) -> int:
    """
    Count how many training pairs the current code solves correctly.
    Used for partial-pass diagnostics and retry detection (Duck Harness pattern).
    Returns number of pairs solved (0 … len(train_pairs)).
    """
    count = 0
    for pair in train_pairs:
        try:
            out = repl.execute_code(code_str, pair["input"])
            if out is not None and out == pair["output"]:
                count += 1
        except Exception:  # noqa: BLE001
            pass
    return count


def _extract_world_model(text: str) -> "str | None":
    """
    Extract a world-model note from LLM output (Duck Harness / Tufa Labs pattern).
    Looks for a line starting with "World model:" and returns its text payload.
    Returns None if no world-model line is present.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("world model:"):
            payload = stripped[len("world model:"):].strip()
            return payload if payload else None
    return None


def recursive_trm_loop(
    task_id: str,
    task_data: dict,
    model,
    tokenizer,
    max_recursion_depth: int = 4,
) -> dict:
    """
    TRM Core Engine — iterates code generation and REPL validation.
    Duck Harness enhancements (Session 60):
      • world_model state carry-forward across depths (Tufa Labs / Franzen)
      • context eviction when input token count exceeds _TRM_MAX_INPUT_TOKENS
      • per-depth try/except isolation — one crash yields, not fails
      • wall-clock depth budget (_TRM_DEPTH_BUDGET_SEC) — yield, don't hang
      • check_demo_success() for partial pass count logging
      • max_new_tokens raised to 1024 for richer code generation

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
    world_model: "str | None" = None       # Duck Harness: persistent world model note

    for depth in range(max_recursion_depth):
        _depth_t0 = time.time()
        try:
            # 1. Build context-aware prompt (includes failure feedback when depth > 0)
            _fb = feedback_message or ""
            if world_model:
                _fb = "World model: " + world_model + "\n" + _fb
            prompt = _trm_format_prompt(task_data, current_code, _fb or None)

            # 2. Tokenise + context eviction (Duck Harness: guard OOM)
            raw_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            if raw_ids.shape[-1] > _TRM_MAX_INPUT_TOKENS:
                # Keep first 1024 (system/examples head) + last 27000 (recent context)
                head = raw_ids[:, :1024]
                tail = raw_ids[:, -(27000):]
                raw_ids = torch.cat([head, tail], dim=1)
            input_len = raw_ids.shape[-1]

            # 3. Generate
            with torch.no_grad():
                outputs = model.generate(
                    raw_ids,
                    max_new_tokens=1024,
                    do_sample=(depth > 0),
                    temperature=0.2 if depth == 0 else 0.6,
                    top_p=0.95 if depth > 0 else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            # Decode only newly generated tokens (shape-safe)
            generated_text = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

            # 4. Extract world model note (Tufa Labs pattern)
            wm_update = _extract_world_model(generated_text)
            if wm_update:
                world_model = wm_update

            current_code = _trm_extract_code(generated_text)

            # 5. Partial count diagnostics (Duck Harness check_demo_success)
            partial_count = check_demo_success(repl, current_code, task_data["train"])

            # 6. Full validation: all pairs + D4 symmetry
            all_passed, verify_msg = verify_program_robustness(
                repl, current_code, task_data["train"]
            )

            # 7. Evaluate recursion state
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
                feedback_message = (
                    f"Program verification failed: {verify_msg}. "
                    f"partial={partial_count}/{len(task_data['train'])} pairs passed. "
                    "Fix the transform logic."
                )

            print(
                f"[TRM] {task_id} depth={depth} all_passed={all_passed} "
                f"partial={partial_count}/{len(task_data['train'])} "
                f"programs={len(successful_programs)} "
                f"elapsed={time.time()-_depth_t0:.1f}s",
                flush=True,
            )

        except Exception as _exc:  # noqa: BLE001  — per-depth isolation (Franzen yield)
            print(
                f"[TRM] {task_id} depth={depth} EXCEPTION (yielding): {_exc}",
                flush=True,
            )
            feedback_message = f"Runtime exception at depth {depth}: {_exc}. Rewrite from scratch."

        # 8. Wall-clock depth budget (yield on timeout, don't hang)
        if time.time() - _depth_t0 > _TRM_DEPTH_BUDGET_SEC:
            print(
                f"[TRM] {task_id} depth={depth} budget exhausted "
                f"({_TRM_DEPTH_BUDGET_SEC}s), yielding.",
                flush=True,
            )
            break

    # 9. Build submission candidates
    attempt_1 = successful_programs[0] if len(successful_programs) >= 1 else fallback
    attempt_2 = successful_programs[1] if len(successful_programs) >= 2 else attempt_1
    return {"attempt_1": attempt_1, "attempt_2": attempt_2}


# ── TRM budget constants (Duck Harness) ───────────────────────────────────────
_TRM_MAX_INPUT_TOKENS: int = 28_000   # evict context above this to prevent OOM
_TRM_DEPTH_BUDGET_SEC: float = 90.0   # wall-clock per depth; yield instead of hang

# ── System prompts ─────────────────────────────────────────────────────────────

_SYS_DIRECT = (
    "You are an expert at ARC-AGI puzzles. "
    "Study the input/output training examples, find the transformation rule, "
    "and apply it to the test input. "
    "Output ONLY the answer grid: rows of space-separated single digits. "
    "No explanation, no code, no markdown."
)

_SYS_COT = (
    "You are an expert at ARC-AGI puzzles. "
    "Study the input/output training examples and find the transformation rule. "
    "Reason step by step inside <think>...</think> tags. "
    "After </think>, output ONLY the final answer grid: "
    "rows of space-separated single digits."
)

_SYS_COMPACT = (
    "You solve ARC-AGI puzzles. "
    "Find the rule from the examples and apply it to the test input. "
    "Output ONLY the answer grid: each row on its own line, digits with no spaces."
)

_SYS_COLORED = (
    "You are an ARC-AGI expert. "
    "0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=magenta 7=orange 8=azure 9=maroon. "
    "Study the examples and output ONLY the answer: "
    "rows of space-separated single digits."
)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def make_prompt(task_view: dict, style: str) -> tuple[str, str]:
    """
    Build (system_msg, user_msg) for one test panel.
    task_view has 'train' list and 'test' list with exactly one panel.
    """
    train = task_view.get("train", [])
    test_panels = task_view.get("test", [])
    test_input = test_panels[0]["input"] if test_panels else [[0]]

    if style == "direct":
        sys_msg = _SYS_DIRECT
        parts = ["Training examples:"]
        for i, ex in enumerate(train):
            parts.append(f"\nExample {i+1} input:\n{grid_to_digits(ex['input'])}")
            parts.append(f"Example {i+1} output:\n{grid_to_digits(ex['output'])}")
        parts.append(f"\nTest input:\n{grid_to_digits(test_input)}")
        parts.append("\nOutput:")

    elif style == "cot":
        sys_msg = _SYS_COT
        parts = ["Study these ARC examples:"]
        for i, ex in enumerate(train):
            parts.append(f"\nExample {i+1} input:\n{grid_to_digits(ex['input'])}")
            parts.append(f"Example {i+1} output:\n{grid_to_digits(ex['output'])}")
        parts.append(f"\nTest input:\n{grid_to_digits(test_input)}")
        parts.append("\nFind the pattern and give the answer grid.")

    elif style == "compact":
        sys_msg = _SYS_COMPACT
        parts = ["Examples (compact, no spaces):"]
        for i, ex in enumerate(train):
            parts.append(f"\nEx{i+1} in:\n{grid_to_compact(ex['input'])}")
            parts.append(f"Ex{i+1} out:\n{grid_to_compact(ex['output'])}")
        parts.append(f"\nTest:\n{grid_to_compact(test_input)}")
        parts.append("\nAnswer:")

    elif style == "colored":
        sys_msg = _SYS_COLORED
        parts = ["Examples with color names:"]
        for i, ex in enumerate(train):
            parts.append(f"\nExample {i+1} input:\n{grid_to_colored(ex['input'])}")
            parts.append(f"Example {i+1} output:\n{grid_to_digits(ex['output'])}")
        parts.append(f"\nTest input:\n{grid_to_colored(test_input)}")
        parts.append("\nOutput (digits only):")


    elif style == "nvarc":
        # Exact SFT training format: multi-turn, no system msg, compact grids
        # Each train pair = one user turn (input) + one assistant turn (output)
        # Returns ("", messages_list) — caller must handle list case
        messages = []
        for ex in train:
            messages.append({"role": "user",      "content": convert_grid_to_string(ex["input"])})
            messages.append({"role": "assistant",  "content": convert_grid_to_string(ex["output"])})
        messages.append({"role": "user", "content": convert_grid_to_string(test_input)})
        return ("__NVARC_MULTITURN__", messages)

    else:
        return make_prompt(task_view, "nvarc")

    return sys_msg, "\n".join(parts)


# ── Per-GPU worker ─────────────────────────────────────────────────────────────

def _apply_chat(tokenizer, messages: list, enable_thinking: bool = False):
    """Apply chat template, gracefully ignoring unknown kwargs."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )


def worker_inference(
    rank: int,
    task_items: list,
    model_path: str,
    worker_out_file: str,
    deadline: float = 0.0,
) -> None:
    """
    Spawned child process: loads model on GPU `rank`, processes task_items,
    writes JSON results to worker_out_file.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    results: dict = {}
    random.seed(42 + rank)  # reproducible per-worker inference

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        print(f"[GPU{rank}] Loading model …", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )
        model.eval()
        print(
            f"[GPU{rank}] Ready — {len(task_items)} tasks to process.", flush=True
        )

        t0 = time.time()
        for step, (task_id, task) in enumerate(task_items):
            test_panels = task.get("test", [])
            if not test_panels:
                results[task_id] = []
                continue

            task_preds = []
            for panel in test_panels:
                inp = panel.get("input", [[0]])
                ref_shape = (len(inp), len(inp[0])) if inp and inp[0] else None
                task_view = {"train": task.get("train", []), "test": [panel]}

                attempt_1: Optional[list] = None
                attempt_2: Optional[list] = None
                _panel_start = time.time()  # TRM budget: track elapsed time

                # ── A/B override: env var can force a single style ───────────────
                _FORCE_STYLE = os.getenv("ARC_LLM_FORCE_STYLE", "").strip().lower()

                # ── Strategy 1: QwenFormatter SFT format → attempt_1 (greedy) ──────
                # Model SFT-trained on QwenFormatter format — bypass apply_chat_template.
                try:
                    from arc_loader import QwenFormatter as _QF1  # noqa: PLC0415
                    _qf1 = _QF1(tokenizer)
                    _prompt1 = _qf1.fmt_train(task_view["train"]) + _qf1.fmt_query([panel])
                    ids = tokenizer(_prompt1, return_tensors="pt").input_ids.to("cuda:0")
                    with torch.no_grad():
                        out = model.generate(
                            ids,
                            max_new_tokens=32 * 31,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
                    attempt_1 = parse_grid(gen, None)
                except Exception as exc:
                    print(f"[GPU{rank}] {task_id} direct err: {exc}", flush=True)

                # ── Strategy 2: QwenFormatter sampled T=0.8 → attempt_2 diversity ─
                try:
                    from arc_loader import QwenFormatter as _QF2  # noqa: PLC0415
                    _qf2 = _QF2(tokenizer)
                    _prompt2 = _qf2.fmt_train(task_view["train"]) + _qf2.fmt_query([panel])
                    ids2 = tokenizer(_prompt2, return_tensors="pt").input_ids.to("cuda:0")
                    with torch.no_grad():
                        out2 = model.generate(
                            ids2,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.95,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    gen2 = tokenizer.decode(out2[0][ids2.shape[-1]:], skip_special_tokens=True)
                    attempt_2 = parse_grid(gen2, None)
                except Exception as exc:
                    print(f"[GPU{rank}] {task_id} nvarc_s2 err: {exc}", flush=True)


                # ── Strategy 2b: 3-temperature ensemble + consistency voting ──────
                # Runs T=0.1 / T=0.4 / T=0.7 when attempt_1 is already valid,
                # then picks the majority-voted grid as attempt_1 and the
                # runner-up (highest demo-consistency score) as attempt_2.
                # Budget guard: skip if < 20 s remain in panel budget.
                if _is_valid_result(attempt_1, inp) and                         (time.time() - _panel_start) < (_PANEL_BUDGET_SEC - 20):
                    try:
                        from arc_loader import QwenFormatter as _QFE  # noqa: PLC0415
                        _qfe = _QFE(tokenizer)
                        _prompt_e = _qfe.fmt_train(task_view["train"]) + _qfe.fmt_query([panel])
                        _ids_e = tokenizer(_prompt_e, return_tensors="pt").input_ids.to("cuda:0")
                        _ens_cands = [attempt_1]  # seed with greedy result
                        for _et in (0.1, 0.4, 0.7):
                            _et_elapsed = time.time() - _panel_start
                            if _et_elapsed >= (_PANEL_BUDGET_SEC - 15):
                                break  # hard budget guard
                            with torch.no_grad():
                                _et_out = model.generate(
                                    _ids_e,
                                    max_new_tokens=32 * 31,
                                    do_sample=True,
                                    temperature=_et,
                                    top_p=0.95,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                            _et_gen  = tokenizer.decode(
                                _et_out[0][_ids_e.shape[-1]:], skip_special_tokens=True
                            )
                            _et_grid = parse_grid(_et_gen, None)
                            if _et_grid is not None:
                                _ens_cands.append(_et_grid)
                        # Vote for best attempt_1; rank by demo-consistency for attempt_2
                        if len(_ens_cands) >= 2:
                            _voted = _pixel_majority_vote_fast(_ens_cands)
                            _ranked = sorted(
                                [g for g in _ens_cands if g is not None],
                                key=lambda g: _demo_consistency_score_fast(
                                    g, task_view["train"]
                                ),
                                reverse=True,
                            )
                            if _voted is not None:
                                attempt_1 = _voted
                            if len(_ranked) >= 2:
                                attempt_2 = _ranked[1]  # runner-up → diversity
                            elif _ranked:
                                attempt_2 = _ranked[0]
                            print(
                                f"[GPU{rank}] {task_id} ens3({len(_ens_cands)} cands) "
                                f"T=0.1/0.4/0.7 → voted a1, ranked a2",
                                flush=True,
                            )
                    except Exception as _ens_exc:
                        print(f"[GPU{rank}] {task_id} ens3 err: {_ens_exc}", flush=True)

                # ── Strategy 3: compact T=0.4 (fallback if either failed) ────
                if attempt_1 is None or attempt_2 is None:
                    try:
                        sys_m3, usr_m3 = make_prompt(task_view, "compact")
                        msgs3 = [
                            {"role": "system",  "content": sys_m3},
                            {"role": "user",    "content": usr_m3},
                        ]
                        ids3 = _apply_chat(tokenizer, msgs3, enable_thinking=False).to("cuda:0")
                        with torch.no_grad():
                            out3 = model.generate(
                                ids3,
                                max_new_tokens=32 * 31,
                                do_sample=True,
                                temperature=0.4,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        gen3 = tokenizer.decode(out3[0][ids3.shape[-1]:], skip_special_tokens=True)
                        fb = parse_grid(gen3, None)
                        if attempt_1 is None:
                            attempt_1 = fb
                        if attempt_2 is None:
                            attempt_2 = fb
                    except Exception as exc:
                        print(f"[GPU{rank}] {task_id} compact err: {exc}", flush=True)

                # ── Strategy 4: colored T=0.4 (second fallback) ─────────────
                if attempt_1 is None or attempt_2 is None:
                    try:
                        sys_m4, usr_m4 = make_prompt(task_view, "colored")
                        msgs4 = [
                            {"role": "system",  "content": sys_m4},
                            {"role": "user",    "content": usr_m4},
                        ]
                        ids4 = _apply_chat(tokenizer, msgs4, enable_thinking=False).to("cuda:0")
                        with torch.no_grad():
                            out4 = model.generate(
                                ids4,
                                max_new_tokens=32 * 31,
                                do_sample=True,
                                temperature=0.4,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        gen4 = tokenizer.decode(out4[0][ids4.shape[-1]:], skip_special_tokens=True)
                        fb4 = parse_grid(gen4, None)
                        if attempt_1 is None:
                            attempt_1 = fb4
                        if attempt_2 is None:
                            attempt_2 = fb4
                    except Exception as exc:
                        print(f"[GPU{rank}] {task_id} colored err: {exc}", flush=True)

                # ── FORCE_STYLE override (A/B benchmark mode) ────────────────────
                if _FORCE_STYLE and _FORCE_STYLE != "nvarc":
                    # Re-run with the forced style and overwrite both attempts
                    try:
                        _fs_result = make_prompt(task_view, _FORCE_STYLE)
                        if isinstance(_fs_result, tuple) and _fs_result[0] == "__NVARC_MULTITURN__":
                            _fs_msgs = _fs_result[1]
                        else:
                            _fs_sys, _fs_usr = _fs_result
                            _fs_msgs = [{"role": "system", "content": _fs_sys},
                                        {"role": "user",   "content": _fs_usr}]
                        _fs_ids = _apply_chat(tokenizer, _fs_msgs, enable_thinking=False).to("cuda:0")
                        with torch.no_grad():
                            _fs_out = model.generate(_fs_ids, max_new_tokens=32 * 31,
                                do_sample=False, temperature=None, top_p=None,
                                pad_token_id=tokenizer.eos_token_id)
                        _fs_gen  = tokenizer.decode(_fs_out[0][_fs_ids.shape[-1]:], skip_special_tokens=True)
                        _fs_grid = parse_grid(_fs_gen, None)
                        attempt_1 = _fs_grid
                        attempt_2 = _fs_grid
                    except Exception as _fs_exc:
                        print(f"[GPU{rank}] {task_id} force_style err: {_fs_exc}", flush=True)


                # ── Transform-retry fallback (if task fails) ──────────────────
                # If both attempts are still invalid (None or trivial copy of
                # test input), try running inference on spatially-transformed
                # versions and inverse-transform the result back.
                if not _is_valid_result(attempt_1, inp) or not _is_valid_result(attempt_2, inp):
                    for _tname, _fwd, _inv in _TRANSFORM_OPS:
                        if _is_valid_result(attempt_1, inp) and _is_valid_result(attempt_2, inp):
                            break
                        try:
                            _tv_aug = _transform_task_view(task_view, _fwd)
                            _aug_inp = _tv_aug["test"][0]["input"]
                            _aug_shape = (len(_aug_inp), len(_aug_inp[0])) if _aug_inp and _aug_inp[0] else ref_shape
                            _aug_ret = make_prompt(_tv_aug, "nvarc")
                            _aug_msgs = _aug_ret[1] if _aug_ret[0] == "__NVARC_MULTITURN__" else [
                                {"role": "system", "content": _aug_ret[0]},
                                {"role": "user",   "content": _aug_ret[1]},
                            ]
                            _aug_ids = _apply_chat(tokenizer, _aug_msgs, enable_thinking=False).to("cuda:0")
                            with torch.no_grad():
                                _aug_out = model.generate(
                                    _aug_ids,
                                    max_new_tokens=32 * 31,
                                    do_sample=False,
                                    temperature=None,
                                    top_p=None,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                            _aug_gen  = tokenizer.decode(_aug_out[0][_aug_ids.shape[-1]:], skip_special_tokens=True)
                            _aug_grid = parse_grid(_aug_gen, None)
                            if _aug_grid is not None:
                                _restored = _inv(_aug_grid)
                                if not _is_valid_result(attempt_1, inp):
                                    attempt_1 = _restored
                                elif not _is_valid_result(attempt_2, inp):
                                    attempt_2 = _restored
                        except Exception as _tex:
                            print(f"[GPU{rank}] {task_id} transform {_tname} err: {_tex}", flush=True)
                            continue

                # ── Strategy 5: TRM code-generation (last resort) ─────────────
                # Triggered only when prior strategies + transform-retry both
                # failed AND we still have budget left within _PANEL_BUDGET_SEC.
                _elapsed = time.time() - _panel_start
                _budget_left = _PANEL_BUDGET_SEC - int(_elapsed)
                if (
                    (
                        not _is_valid_result(attempt_1, inp)
                        or not _is_valid_result(attempt_2, inp)
                    )
                    and _budget_left > 30
                ):
                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(max(30, _budget_left))
                    try:
                        _trm_data = {"train": task.get("train", []), "test": panel}
                        _trm_res = recursive_trm_loop(
                            task_id, _trm_data, model, tokenizer, max_recursion_depth=4
                        )
                        if not _is_valid_result(attempt_1, inp):
                            _cand = _trm_res["attempt_1"]
                            if _is_valid_result(_cand, inp):
                                attempt_1 = _cand
                                print(f"[GPU{rank}] {task_id} TRM -> attempt_1 OK", flush=True)
                        if not _is_valid_result(attempt_2, inp):
                            _cand2 = _trm_res["attempt_2"]
                            if _is_valid_result(_cand2, inp):
                                attempt_2 = _cand2
                                print(f"[GPU{rank}] {task_id} TRM -> attempt_2 OK", flush=True)
                    except (TimeoutException, Exception) as _trm_exc:
                        print(
                            f"[GPU{rank}] {task_id} TRM err ({type(_trm_exc).__name__}): "
                            f"{_trm_exc}",
                            flush=True,
                        )
                    finally:
                        signal.alarm(0)  # always cancel alarm

                # ── Hard fallback: test input ─────────────────────────────────
                if attempt_1 is None:
                    attempt_1 = inp
                if attempt_2 is None:
                    attempt_2 = attempt_1

                task_preds.append(
                    {"attempt_1": attempt_1, "attempt_2": attempt_2}
                )

            results[task_id] = task_preds

            # Incremental save: crash recovery — write after every task
            try:
                with open(worker_out_file, "w") as _cf:
                    json.dump(results, _cf)
            except Exception:
                pass
            # Hard deadline guard
            if deadline > 0.0 and time.time() > deadline:
                print(f"[GPU{rank}] Deadline reached after {step+1}/{len(task_items)} tasks.", flush=True)
                break

            if (step + 1) % 10 == 0 or (step + 1) == len(task_items):
                elapsed = time.time() - t0
                rate = (step + 1) / max(elapsed, 1e-9)
                eta = (len(task_items) - step - 1) / rate
                print(
                    f"[GPU{rank}] {step+1}/{len(task_items)} | "
                    f"{elapsed:.0f}s | ETA {eta:.0f}s",
                    flush=True,
                )

        print(f"[GPU{rank}] Done — {len(results)} predictions.", flush=True)

    except Exception as exc:  # noqa: BLE001
        print(f"[GPU{rank}] Worker crashed: {exc}", flush=True)
        traceback.print_exc()

    Path(worker_out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(worker_out_file, "w") as fh:
        json.dump(results, fh)


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_llm_fast_pass(
    tasks: dict,
    model_path: str,
    output_path: str,
    n_workers: int = 4,
    deadline: float = 0.0,
) -> int:
    """
    Distribute tasks across n_workers GPU processes.
    Returns count of task predictions saved.
    """
    task_items = list(tasks.items())
    n = len(task_items)
    print(
        f"[llm-fast] {n} tasks | {n_workers} GPU workers | model={model_path}",
        flush=True,
    )

    # Even distribution: round-robin
    chunks: list[list] = [[] for _ in range(n_workers)]
    for i, item in enumerate(task_items):
        chunks[i % n_workers].append(item)

    worker_files = [
        f"/kaggle/working/jeni_llm_worker_{r}.json" for r in range(n_workers)
    ]

    ctx = mp.get_context("spawn")
    processes: list[tuple[int, mp.Process]] = []
    for rank in range(n_workers):
        if not chunks[rank]:
            continue
        p = ctx.Process(
            target=worker_inference,
            args=(rank, chunks[rank], model_path, worker_files[rank], deadline),
            daemon=False,
        )
        p.start()
        processes.append((rank, p))

    for rank, p in processes:
        p.join()
        print(
            f"[llm-fast] GPU{rank} joined (exit={p.exitcode})", flush=True
        )

    # Aggregate results
    all_results: dict = {}
    for rank in range(n_workers):
        wf = worker_files[rank]
        if Path(wf).exists():
            try:
                with open(wf) as fh:
                    all_results.update(json.load(fh))
                print(f"[llm-fast] Loaded worker {rank}: {wf}", flush=True)
            except Exception as exc:
                print(f"[llm-fast] Could not load {wf}: {exc}", flush=True)
        else:
            print(f"[llm-fast] Worker {rank} output missing: {wf}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(all_results, fh)

    print(
        f"[llm-fast] Saved {len(all_results)} task predictions → {output_path}",
        flush=True,
    )
    return len(all_results)


# ── Model discovery ────────────────────────────────────────────────────────────

# Hard-coded Kaggle model mount paths for sorokin/qwen3_4b_grids15_sft139
# These mirror the NVARC arc_solver.py discovery order exactly.
_NVARC_MODEL_PATHS = [
    "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
]


def discover_model_path() -> str:
    """
    Find the sorokin/qwen3_4b_grids15_sft139 model directory.
    Priority:
      1. ARC_MODEL_PATH env var (set by Cell 8 env dict)
      2. Hard-coded Kaggle mount paths (mirrors NVARC arc_solver.py)
      3. Glob scan /kaggle/input for any config.json with qwen/arc in path
    """
    env_path = os.getenv("ARC_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        print(f"[llm-fast] Model via ARC_MODEL_PATH: {env_path}", flush=True)
        return env_path

    # Hard-coded paths first (fastest, most reliable)
    for p in _NVARC_MODEL_PATHS:
        if Path(p).is_dir() and Path(p, "config.json").exists():
            print(f"[llm-fast] Found model at {p}", flush=True)
            return p

    # Fallback: scan /kaggle/input
    search_root = Path("/kaggle/input")
    if not search_root.exists():
        return ""

    candidates: list[Path] = []
    for entry in sorted(search_root.rglob("config.json")):
        try:
            cfg = json.loads(entry.read_text())
        except Exception:
            continue
        if "vocab_size" in cfg or "model_type" in cfg:
            candidates.append(entry.parent)

    for c in candidates:
        if "qwen" in str(c).lower() or "arc" in str(c).lower():
            print(f"[llm-fast] Found model (scan) at {c}", flush=True)
            return str(c)
    if candidates:
        print(f"[llm-fast] Using fallback model at {candidates[0]}", flush=True)
        return str(candidates[0])
    return ""


# ── CLI entry ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM fast-inference pass for ARC-AGI-2")
    parser.add_argument("--model-path",     default=None,
                        help="Path to Qwen3-4B model dir. Defaults to ARC_MODEL_PATH env.")
    parser.add_argument("--challenge-file", default=None,
                        help="Path to ARC challenge JSON. Auto-discovered if omitted.")
    parser.add_argument("--output",         default="/kaggle/working/jeni_llm_preds.json",
                        help="Output path for LLM predictions.")
    parser.add_argument("--n-workers",      type=int, default=4,
                        help="Number of GPU workers (default 4).")
    parser.add_argument("--end-time",       type=float, default=None,
                        help="Unix timestamp deadline. Script exits early if < 30 min remain.")
    args = parser.parse_args()

    # ── Time-budget check ─────────────────────────────────────────────────────
    if args.end_time is not None:
        remaining = args.end_time - time.time()
        if remaining < 1800:  # < 30 min
            print(
                f"[llm-fast] Only {remaining:.0f}s remaining — skipping LLM pass.",
                flush=True,
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as fh:
                json.dump({}, fh)
            return

    # ── Model path ────────────────────────────────────────────────────────────
    model_path = args.model_path or discover_model_path()
    if not model_path or not Path(model_path).exists():
        print("[llm-fast] ERROR: model not found. Set --model-path or ARC_MODEL_PATH.",
              flush=True)
        sys.exit(1)

    # ── Challenge file ────────────────────────────────────────────────────────
    challenge_file = args.challenge_file
    if not challenge_file:
        for d in [
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
            "/kaggle/input/arc-prize-2026-arc-agi-2",
        ]:
            is_rerun = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
            fname = (
                "arc-agi_test_challenges.json"
                if is_rerun
                else "arc-agi_evaluation_challenges.json"
            )
            candidate = Path(d) / fname
            if candidate.exists():
                challenge_file = str(candidate)
                break

    if not challenge_file or not Path(challenge_file).exists():
        print(f"[llm-fast] ERROR: challenge file not found: {challenge_file}", flush=True)
        sys.exit(1)

    print(f"[llm-fast] Challenge file: {challenge_file}", flush=True)
    with open(challenge_file) as fh:
        tasks = json.load(fh)

    # ── GPU count ─────────────────────────────────────────────────────────────
    n_visible = torch.cuda.device_count()
    n_workers = min(args.n_workers, max(1, n_visible))
    print(f"[llm-fast] CUDA devices visible: {n_visible}, using {n_workers} workers",
          flush=True)

    t0 = time.time()
    _deadline = (args.end_time - 300) if args.end_time is not None else 0.0
    count = run_llm_fast_pass(tasks, model_path, args.output, n_workers=n_workers, deadline=_deadline)
    elapsed = time.time() - t0
    print(f"[llm-fast] Finished: {count} tasks in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
