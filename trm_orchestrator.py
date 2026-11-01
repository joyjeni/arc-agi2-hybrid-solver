#!/usr/bin/env python3
"""
trm_orchestrator.py  —  Standalone TRM + LLM inference orchestrator for ARC-AGI-2
====================================================================================
Session 60 — Duck Harness edition.

Architecture:
  • ARCDSL         — advanced grid primitives (flood_fill, get_objects, upscale, etc.)
  • PythonREPL     — isolated exec sandbox with get_connected_components pre-loaded
  • recursive_trm_loop — Duck Harness TRM:
      - world_model state carry-forward across depths (Tufa Labs / Franzen)
      - context eviction at _TRM_MAX_INPUT_TOKENS (prevent OOM)
      - per-depth try/except isolation (yield, not fail)
      - wall-clock depth budget _TRM_DEPTH_BUDGET_SEC (Franzen yield-on-timeout)
      - check_demo_success() partial diagnostics
  • worker_process — per-GPU worker running 5-strategy inference pipeline:
      Strategy 1 — QwenFormatter SFT greedy (attempt_1)
      Strategy 2 — QwenFormatter sampled T=0.8 (attempt_2 diversity)
      Strategy 3 — compact T=0.4 fallback
      Strategy 4 — colored T=0.4 second fallback
      Strategy 5 — TRM code generation (last resort)
      D4 spatial-transform retry between S4 and S5
  • main()         — CLI entry with argparse + GPU auto-detection

Output: /kaggle/working/trm_preds.json
  Format: {task_id: [{attempt_1: [[int,...]], attempt_2: [[int,...]]}, ...]}

Self-contained: does not require arc_loader.py at runtime (QwenFormatter imported
with try/except; inline fallback used when unavailable).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import signal
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy import ndimage

# Ensure CUDA-safe multiprocessing
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# COLOR MAP
# ══════════════════════════════════════════════════════════════════════════════

COLOR_NAMES = {
    0: "black", 1: "blue",    2: "red",    3: "green",  4: "yellow",
    5: "grey",  6: "magenta", 7: "orange", 8: "azure",  9: "maroon",
}


# ══════════════════════════════════════════════════════════════════════════════
# ADVANCED FUZZY TOPOLOGY & ARC PRIMITIVE DSL
# ══════════════════════════════════════════════════════════════════════════════

class ARCDSL:
    """High-level grid primitives exposed to the TRM REPL sandbox."""

    @staticmethod
    def to_np(grid: Any) -> np.ndarray:
        return np.array(grid, dtype=int)

    @staticmethod
    def flood_fill(grid: Any, start_r: int, start_c: int, fill_color: int) -> list:
        arr = ARCDSL.to_np(grid)
        H, W = arr.shape
        if not (0 <= start_r < H and 0 <= start_c < W):
            return arr.tolist()
        target_color = int(arr[start_r, start_c])
        if target_color == fill_color:
            return arr.tolist()
        mask = (arr == target_color)
        labels, _ = ndimage.label(mask)
        match_label = int(labels[start_r, start_c])
        if match_label > 0:
            arr[labels == match_label] = fill_color
        return arr.tolist()

    @staticmethod
    def get_objects(
        grid: Any, connectivity: int = 8, exclude_bg: int = 0
    ) -> List[Dict[str, Any]]:
        arr = ARCDSL.to_np(grid)
        objects: list = []
        struct = ndimage.generate_binary_structure(
            2, 1 if connectivity == 4 else 2
        )
        for color in np.unique(arr):
            if color == exclude_bg:
                continue
            mask = arr == color
            labels, num = ndimage.label(mask, structure=struct)
            for lab in range(1, num + 1):
                rc = np.argwhere(labels == lab)
                if len(rc) == 0:
                    continue
                min_r, min_c = rc.min(axis=0)
                max_r, max_c = rc.max(axis=0)
                objects.append({
                    "color":    int(color),
                    "coords":   rc.tolist(),
                    "centroid": [float(np.mean(rc[:, 0])), float(np.mean(rc[:, 1]))],
                    "shape":    [int(max_r - min_r + 1), int(max_c - min_c + 1)],
                    "bounds":   [int(min_r), int(max_r), int(min_c), int(max_c)],
                    "mask":     arr[min_r:max_r+1, min_c:max_c+1].tolist(),
                })
        return objects

    @staticmethod
    def crop_to_content(grid: Any, bg_color: int = 0) -> list:
        arr = ARCDSL.to_np(grid)
        non_bg = np.argwhere(arr != bg_color)
        if len(non_bg) == 0:
            return [[bg_color]]
        min_r, min_c = non_bg.min(axis=0)
        max_r, max_c = non_bg.max(axis=0)
        return arr[min_r:max_r+1, min_c:max_c+1].tolist()

    @staticmethod
    def upscale_nearest(grid: Any, scale_factor: int) -> list:
        arr = ARCDSL.to_np(grid)
        return np.repeat(
            np.repeat(arr, scale_factor, axis=0), scale_factor, axis=1
        ).tolist()


def _make_get_connected_components():
    """
    Build a 4-connected component labelling function for the REPL sandbox.
    Duck Harness segmentation helper (Session 60).
    Returned as a closure so it has no external deps inside the REPL.
    """
    def get_connected_components(grid):
        """Label 4-connected regions of same color. Returns label grid (list[list[int]])."""
        import numpy as __np
        g = __np.array(grid, dtype=int)
        rows, cols = g.shape
        visited = __np.zeros_like(g, dtype=bool)
        labels = __np.full_like(g, -1, dtype=int)
        label = 0
        for r in range(rows):
            for c in range(cols):
                if not visited[r, c]:
                    color = g[r, c]
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        cr, cc = stack.pop()
                        labels[cr, cc] = label
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and not visited[nr, nc]
                                and g[nr, nc] == color
                            ):
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                    label += 1
        return labels.tolist()
    return get_connected_components


# Global REPL namespace shared by all PythonREPL instances
REPL_GLOBALS: dict = {
    "np":                     np,
    "ndimage":                ndimage,
    "flood_fill":             ARCDSL.flood_fill,
    "get_objects":            ARCDSL.get_objects,
    "crop_to_content":        ARCDSL.crop_to_content,
    "upscale_nearest":        ARCDSL.upscale_nearest,
    "get_connected_components": _make_get_connected_components(),
}


class PythonREPL:
    """
    Lightweight isolated execution sandbox for TRM code validation.
    Executes model-generated `transform(grid)` functions against ARC
    training pairs to check correctness before using the output.
    """

    def __init__(self):
        self.globals = dict(REPL_GLOBALS)

    def execute_code(self, code_str: str, input_grid: list) -> Optional[list]:
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


# ══════════════════════════════════════════════════════════════════════════════
# TRM HELPERS — TIME-AWARE RECURSIVE SYNTHESIS WITH INVARIANCE
# ══════════════════════════════════════════════════════════════════════════════

class TimeoutException(Exception):
    """Raised by SIGALRM handler when per-panel time budget is exceeded."""


def _timeout_handler(signum, frame):  # noqa: ANN001
    raise TimeoutException("Task thread calculation window closed.")


def verify_program_robustness(
    repl: PythonREPL, code_str: str, train_pairs: list
) -> Tuple[bool, str]:
    """
    Validate a generated transform() against all training pairs.
    Also tests D4 rotational invariance to reject brittle programs.
    Returns (passed: bool, reason: str).
    """
    for idx, pair in enumerate(train_pairs):
        base_res = repl.execute_code(code_str, pair["input"])
        if base_res is None or base_res != pair["output"]:
            return (
                False,
                f"Fails structural target reconstruction mapping on training instance {idx+1}",
            )
        try:
            rot_in  = np.rot90(np.array(pair["input"]),  1).tolist()
            rot_out = np.rot90(np.array(pair["output"]), 1).tolist()
            rot_res = repl.execute_code(code_str, rot_in)
            if rot_res is None or rot_res != rot_out:
                return (
                    False,
                    f"Fails rotational invariance test on instance {idx+1} "
                    "(logic is too grid-locked)",
                )
        except Exception:
            return False, "Symmetry interpolation transformation error"
    return True, "Passed complete robustness validation matrix"


def check_demo_success(
    repl: PythonREPL, code_str: str, train_pairs: list
) -> int:
    """
    Count how many training pairs the current code solves correctly.
    Duck Harness partial-pass diagnostics and retry detection (Session 60).
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


def _extract_world_model(text: str) -> Optional[str]:
    """
    Extract a world-model note from LLM output (Duck Harness / Tufa Labs pattern).
    Looks for a line starting with "World model:" and returns its text payload.
    Returns None if no such line is present.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("world model:"):
            payload = stripped[len("world model:"):].strip()
            return payload if payload else None
    return None


# ── TRM budget constants (Duck Harness, Session 60) ──────────────────────────
_TRM_MAX_INPUT_TOKENS: int   = 28_000  # evict context above this → prevent OOM
_TRM_DEPTH_BUDGET_SEC: float = 90.0    # wall-clock per depth → yield, not hang
_PANEL_BUDGET_SEC:     int   = 180     # hard wall-clock budget per panel (incl. TRM)


def recursive_trm_loop(
    task_id: str,
    task_data: dict,
    model,
    tokenizer,
    time_allowance: int = _PANEL_BUDGET_SEC,
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
        time_allowance:      Soft wall-clock limit for the entire TRM loop (s).
        max_recursion_depth: Max generate → validate → refine cycles.

    Returns:
        {"attempt_1": grid, "attempt_2": grid}
    """
    repl = PythonREPL()
    current_code: Optional[str] = None
    feedback_message: Optional[str] = None
    fallback: list = task_data["test"]["input"]
    successful_programs: list = []
    world_model: Optional[str] = None      # Duck Harness: persistent world model
    loop_start = time.time()

    for depth in range(max_recursion_depth):
        # Outer soft wall: respect caller's time_allowance
        if time.time() - loop_start > time_allowance - 15:
            print(
                f"[TRM] {task_id} depth={depth} soft wall reached — exiting loop.",
                flush=True,
            )
            break

        _depth_t0 = time.time()

        try:
            # 1. Build prompt with optional world model prefix
            _fb = feedback_message or ""
            if world_model:
                _fb = "World model: " + world_model + "\n" + _fb

            # SFT structural layout — ARCDSL primitives declared in system preamble
            prompt = (
                "<|im_start|>user\nYou are an automated computer science logic engineer "
                "synthesizing Python code for ARC puzzles.\n"
                "Write a Python function `transform(grid: list[list[int]]) -> list[list[int]]` "
                "utilizing numpy.\n\n"
                "You have access to the following built-in high-level abstraction primitives "
                "natively:\n"
                " - `flood_fill(grid, start_r, start_c, fill_color)`\n"
                " - `get_objects(grid, connectivity=8)` -> "
                "Returns List of Dict metadata tags per distinct object shape profile\n"
                " - `crop_to_content(grid, bg_color=0)`\n"
                " - `upscale_nearest(grid, scale_factor)`\n"
                " - `get_connected_components(grid)` -> 4-connected label grid\n\n"
                "Solving strategies (apply the most relevant):\n"
                "• Vedic Anurupyena (proportionality): detect ratio-based scaling / tiling.\n"
                "• Vedic Sunyam Samyasamuccaye (symmetry): find mirror / rotation structure.\n"
                "• Vedic Vilokanam (direct observation): spot the simplest single-rule transform.\n"
                "• Japanese Kaizen (iterative): rule applied cell-by-cell or row-by-row.\n"
                "• Japanese Kanban (pull-based): each output cell determined by colored neighbor.\n"
                "• Chinese Nine-Palace (九宫格): map 3×3 block spatial relationships.\n"
                "• Nature-inspired: flood-fill propagation, cellular automata, fractal self-similarity.\n\n"
            )
            for i, pair in enumerate(task_data["train"]):
                prompt += (
                    f"Example {i+1} Input: {json.dumps(pair['input'])}\n"
                    f"Example {i+1} Expected: {json.dumps(pair['output'])}\n\n"
                )
            prompt += f"Test Target Input: {json.dumps(task_data['test']['input'])}\n\n"
            if current_code and _fb:
                prompt += (
                    f"### RECURSIVE REFINEMENT STEP ###\n"
                    f"Your previous code variant failed validation tests:\n"
                    f"```python\n{current_code}\n```\n"
                    f"Feedback: {_fb}\n"
                    f"Rewrite the entire algorithm to correct the issue.\n"
                )
            prompt += (
                "Output the pure Python code block enclosed in ```python and ```.\n"
                "<|im_end|>\n<|im_start|>assistant\n"
            )

            # 2. Tokenise + context eviction (Duck Harness OOM guard)
            raw_ids = tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(model.device)
            if raw_ids.shape[-1] > _TRM_MAX_INPUT_TOKENS:
                # Keep head 1024 (system/examples) + tail 27 000 (recent context)
                head = raw_ids[:, :1024]
                tail = raw_ids[:, -27000:]
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
            gen_text = tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

            # 4. Extract world model note (Tufa Labs / Duck Harness)
            wm_update = _extract_world_model(gen_text)
            if wm_update:
                world_model = wm_update

            # 5. Extract code block
            if "```python" in gen_text:
                _parts_py = gen_text.split("```python")
                current_code = (_parts_py[1].split("```")[0] if len(_parts_py) > 1 else gen_text).strip()
            elif "```" in gen_text:
                _parts_g = gen_text.split("```")
                current_code = (_parts_g[1] if len(_parts_g) > 1 else gen_text).strip()
            else:
                current_code = gen_text.strip()

            # 6. Partial count diagnostics (Duck Harness check_demo_success)
            partial_count = check_demo_success(
                repl, current_code, task_data["train"]
            )

            # 7. Full validation: all pairs + D4 symmetry
            all_passed, diagnostics = verify_program_robustness(
                repl, current_code, task_data["train"]
            )

            # 8. Evaluate recursion state
            if all_passed:
                final_out = repl.execute_code(
                    current_code, task_data["test"]["input"]
                )
                if final_out is not None:
                    successful_programs.append(final_out)
                    if len(successful_programs) >= 2:
                        break  # both attempt slots filled — done
                feedback_message = (
                    "Generates valid solution. "
                    "Output an alternative topological transformation logic path "
                    "using a different algorithm."
                )
            else:
                feedback_message = (
                    f"{diagnostics}. "
                    f"partial={partial_count}/{len(task_data['train'])} pairs passed. "
                    "Fix the transform logic."
                )

            print(
                f"[TRM] {task_id} depth={depth} passed={all_passed} "
                f"partial={partial_count}/{len(task_data['train'])} "
                f"programs={len(successful_programs)} "
                f"elapsed={time.time()-_depth_t0:.1f}s",
                flush=True,
            )

        except Exception as _exc:  # noqa: BLE001 — per-depth isolation (Franzen yield)
            print(
                f"[TRM] {task_id} depth={depth} EXCEPTION (yielding): {_exc}",
                flush=True,
            )
            feedback_message = (
                f"Runtime exception at depth {depth}: {_exc}. "
                "Rewrite from scratch."
            )

        # 9. Wall-clock depth budget (yield on timeout — Franzen pattern)
        if time.time() - _depth_t0 > _TRM_DEPTH_BUDGET_SEC:
            print(
                f"[TRM] {task_id} depth={depth} budget exhausted "
                f"({_TRM_DEPTH_BUDGET_SEC}s), yielding.",
                flush=True,
            )
            break

    # 10. Build submission candidates
    attempt_1 = successful_programs[0] if len(successful_programs) >= 1 else fallback
    attempt_2 = (
        successful_programs[1] if len(successful_programs) >= 2 else attempt_1
    )
    return {"attempt_1": attempt_1, "attempt_2": attempt_2}


# ══════════════════════════════════════════════════════════════════════════════
# GRID UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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
    """
    NVARC/SFT format: compact digits separated by newlines (no spaces).
    Matches the exact format the qwen3_4b_grids15_sft139 model was SFT-trained on.
    """
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)


def parse_grid(text: str, ref_shape: Optional[tuple] = None) -> Optional[list]:
    """
    Parse LLM output into a 2-D integer grid.
    Strips <think>...</think> blocks and markdown fences first.
    Returns None if no valid grid can be extracted.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```[^\n]*\n?", "", text).strip()

    rows: list = []
    for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
        parts = line.split()
        if parts and all(p.isdigit() and len(p) == 1 for p in parts):
            rows.append([int(p) for p in parts])
            continue
        compact = line.replace(" ", "")
        if compact and all(c.isdigit() for c in compact):
            rows.append([int(c) for c in compact])
            continue
        parts_c = [p.strip() for p in line.split(",")]
        if parts_c and all(p.isdigit() and len(p) == 1 for p in parts_c):
            rows.append([int(p) for p in parts_c])
            continue
        digits = re.findall(r"\b\d\b", line)
        if len(digits) >= 2:
            rows.append([int(d) for d in digits])

    if not rows:
        return None

    # Clip / pad to ref_shape
    if ref_shape is not None:
        h, w = ref_shape
        rows = rows[:h]
        rows = [r[:w] + [0] * max(0, w - len(r)) for r in rows]
        while len(rows) < h:
            rows.append([0] * w)

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL-TRANSFORM RETRY HELPERS (D4 subset)
# ══════════════════════════════════════════════════════════════════════════════

import numpy as _np_aug  # alias to avoid collision with outer np

_TRANSFORM_OPS = [
    ("rot90_1",  lambda g: _np_aug.rot90(_np_aug.array(g),  1).tolist(),
                 lambda g: _np_aug.rot90(_np_aug.array(g), -1).tolist()),
    ("rot90_2",  lambda g: _np_aug.rot90(_np_aug.array(g),  2).tolist(),
                 lambda g: _np_aug.rot90(_np_aug.array(g),  2).tolist()),
    ("rot90_3",  lambda g: _np_aug.rot90(_np_aug.array(g),  3).tolist(),
                 lambda g: _np_aug.rot90(_np_aug.array(g), -3).tolist()),
    ("fliplr",   lambda g: _np_aug.fliplr(_np_aug.array(g)).tolist(),
                 lambda g: _np_aug.fliplr(_np_aug.array(g)).tolist()),
    ("flipud",   lambda g: _np_aug.flipud(_np_aug.array(g)).tolist(),
                 lambda g: _np_aug.flipud(_np_aug.array(g)).tolist()),
    ("transpose", lambda g: _np_aug.array(g).T.tolist(),
                  lambda g: _np_aug.array(g).T.tolist()),
]


def _transform_task_view(task_view: dict, fwd_fn) -> dict:
    """Apply a spatial transform to all train + test grids in a task view."""
    def _t(grid):
        try:
            return fwd_fn(grid)
        except Exception:
            return grid
    return {
        "train": [{"input": _t(p["input"]), "output": _t(p["output"])}
                  for p in task_view.get("train", [])],
        "test":  [{"input": _t(p["input"])} for p in task_view.get("test", [])],
    }


def _is_valid_result(grid, test_input) -> bool:
    """Return True if grid looks like a real prediction (not a trivial copy)."""
    if grid is None:
        return False
    if grid == test_input:          # identical to input → likely a failure
        return False
    flat = [c for row in grid for c in row]
    if not flat or all(v == 0 for v in flat):
        return False
    return True


def _pixel_majority_vote(candidates: list):
    """Return the candidate grid appearing most often (pixel-level majority vote).
    Falls back to first valid candidate if no majority, or None if empty.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    counts: "Counter" = Counter()
    for g in candidates:
        if g is not None:
            key = tuple(tuple(row) for row in g)
            counts[key] += 1
    if not counts:
        return candidates[0]
    best_key, _ = counts.most_common(1)[0]
    return [list(row) for row in best_key]


def _demo_consistency_score(grid: list, train_pairs: list) -> float:
    """Return [0, 1] score: fraction of training outputs whose shape matches grid.
    Used as a TbV heuristic to prefer correctly-shaped predictions.
    """
    if grid is None or not train_pairs:
        return 0.0
    g_rows = len(grid)
    g_cols = len(grid[0]) if grid else 0
    matches = sum(
        1 for p in train_pairs
        if p.get("output")
        and len(p["output"]) == g_rows
        and len(p["output"][0]) == g_cols
    )
    return matches / max(len(train_pairs), 1)


class ARCLogitsProcessor:
    """Zero-overhead logit mask: only ARC digit tokens (0-9, newline, space, EOS) survive.
    Inline definition — requires only torch (no transformers at module level).
    Compatible with HuggingFace LogitsProcessorList via duck typing.
    """
    def __init__(self, allowed_token_ids: list):
        self._allowed = allowed_token_ids
        self._mask = None

    def __call__(self, input_ids, scores):
        V = scores.shape[-1]
        if self._mask is None or self._mask.shape[-1] != V:
            m = torch.full((V,), float("-inf"), dtype=scores.dtype, device=scores.device)
            for tid in self._allowed:
                if 0 <= tid < V:
                    m[tid] = 0.0
            self._mask = m
        return scores + self._mask.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS + PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

_SYS_DIRECT = (
    "You are an expert at ARC-AGI puzzles. "
    "Study the input/output training examples, find the transformation rule, "
    "and apply it to the test input. "
    "Output ONLY the answer grid: rows of space-separated single digits. "
    "No explanation, no code, no markdown."
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


def make_prompt(task_view: dict, style: str):
    """
    Build (system_msg, user_msg) or ('__NVARC_MULTITURN__', messages) for one panel.
    task_view has 'train' list and 'test' list with exactly one panel.
    """
    train      = task_view.get("train", [])
    test_panels = task_view.get("test", [])
    test_input  = test_panels[0]["input"] if test_panels else [[0]]

    if style == "direct":
        parts = ["Training examples:"]
        for i, ex in enumerate(train):
            parts += [
                f"\nExample {i+1} input:\n{grid_to_digits(ex['input'])}",
                f"Example {i+1} output:\n{grid_to_digits(ex['output'])}",
            ]
        parts.append(f"\nTest input:\n{grid_to_digits(test_input)}\n\nOutput:")
        return _SYS_DIRECT, "\n".join(parts)

    if style == "compact":
        parts = ["Examples (compact, no spaces):"]
        for i, ex in enumerate(train):
            parts += [
                f"\nEx{i+1} in:\n{grid_to_compact(ex['input'])}",
                f"Ex{i+1} out:\n{grid_to_compact(ex['output'])}",
            ]
        parts.append(f"\nTest:\n{grid_to_compact(test_input)}\n\nAnswer:")
        return _SYS_COMPACT, "\n".join(parts)

    if style == "colored":
        parts = ["Examples with color names:"]
        for i, ex in enumerate(train):
            parts += [
                f"\nExample {i+1} input:\n{grid_to_colored(ex['input'])}",
                f"Example {i+1} output:\n{grid_to_digits(ex['output'])}",
            ]
        parts.append(
            f"\nTest input:\n{grid_to_colored(test_input)}\n\nOutput (digits only):"
        )
        return _SYS_COLORED, "\n".join(parts)

    # Default: nvarc SFT multi-turn format
    messages = []
    for ex in train:
        messages.append({"role": "user",      "content": convert_grid_to_string(ex["input"])})
        messages.append({"role": "assistant", "content": convert_grid_to_string(ex["output"])})
    messages.append({"role": "user", "content": convert_grid_to_string(test_input)})
    return "__NVARC_MULTITURN__", messages


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


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL WORKER AND INFERENCE PLUMBING
# ══════════════════════════════════════════════════════════════════════════════

def worker_process(
    rank: int,
    chunk_tasks: list,
    model_path: str,
    output_dir: str,
    time_per_task: int,
) -> None:
    """
    Isolated process running exclusively on CUDA device `rank`.

    5-strategy inference pipeline per panel:
      S1 — QwenFormatter SFT greedy        → attempt_1
      S2 — QwenFormatter sampled T=0.8     → attempt_2 diversity
      S3 — compact T=0.4                   → fallback
      S4 — colored T=0.4                   → second fallback
      D4 spatial-transform retry           → between S4 and TRM
      S5 — TRM code generation (last resort)

    Args:
        rank:        CUDA device index.
        chunk_tasks: List of (task_id, task_data) tuples for this worker.
        model_path:  Path to Qwen3-4B model directory.
        output_dir:  Directory where the worker JSON result is written.
        time_per_task: Soft per-task time budget in seconds (used for TRM alarm).
    """
    os.environ["CUDA_VISIBLE_DEVICES"]   = str(rank)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"]         = "1"
    os.environ["TRANSFORMERS_OFFLINE"]   = "1"

    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList  # noqa: PLC0415

    print(f"[GPU {rank}] Initializing model loader footprint…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # ── ARCLogitsProcessor: compute allowed token IDs and build logits mask ──
    try:
        _arc_ids: list = []
        for _ch in "0123456789":
            for _tok in tokenizer.encode(_ch, add_special_tokens=False):
                _arc_ids.append(_tok)
        for _extra in ("\n", " "):
            for _tok in tokenizer.encode(_extra, add_special_tokens=False):
                _arc_ids.append(_tok)
        if tokenizer.eos_token_id is not None:
            _arc_ids.append(tokenizer.eos_token_id)
        _arc_ids = list(set(_arc_ids))
        _arc_lp_list = LogitsProcessorList([ARCLogitsProcessor(_arc_ids)])
        print(
            f"[GPU {rank}] ARCLogitsProcessor ready ({len(_arc_ids)} allowed tokens).",
            flush=True,
        )
    except Exception as _lp_exc:
        _arc_lp_list = LogitsProcessorList([])
        print(
            f"[GPU {rank}] ARCLogitsProcessor disabled ({_lp_exc}) — no token masking.",
            flush=True,
        )

    print(
        f"[GPU {rank}] Model operational. Executing {len(chunk_tasks)} task assignments…",
        flush=True,
    )

    worker_preds: dict = {}
    worker_out = Path(output_dir) / f"trm_worker_{rank}.json"
    signal.signal(signal.SIGALRM, _timeout_handler)
    t0 = time.time()

    for step, (task_id, task_data) in enumerate(chunk_tasks):
        test_panels = task_data.get("test", [])
        if not test_panels:
            worker_preds[task_id] = []
            continue

        task_preds = []
        for panel in test_panels:
            inp = panel.get("input", [[0]])
            ref_shape = (
                (len(inp), len(inp[0])) if inp and inp[0] else None
            )
            task_view   = {"train": task_data.get("train", []), "test": [panel]}
            _panel_start = time.time()

            attempt_1: Optional[list] = None
            attempt_2: Optional[list] = None
            _candidates: list = []  # collects valid grids for majority vote

            # ── Strategy 1: QwenFormatter SFT greedy → attempt_1 ────────────
            try:
                try:
                    from arc_loader import QwenFormatter as _QF  # noqa: PLC0415
                except ImportError:
                    _QF = None  # type: ignore[assignment]

                if _QF is not None:
                    _qf = _QF(tokenizer)
                    _p1 = _qf.fmt_train(task_view["train"]) + _qf.fmt_query([panel])
                else:
                    _, _msgs1 = make_prompt(task_view, "nvarc")
                    _p1 = None
                    _msgs1_use = _msgs1

                if _p1 is not None:
                    _ids1 = tokenizer(_p1, return_tensors="pt").input_ids.to("cuda:0")
                else:
                    _ids1 = _apply_chat(
                        tokenizer, _msgs1_use, enable_thinking=False
                    ).to("cuda:0")

                with torch.no_grad():
                    _out1 = model.generate(
                        _ids1,
                        max_new_tokens=32 * 31,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=_arc_lp_list,
                    )
                _gen1   = tokenizer.decode(_out1[0][_ids1.shape[-1]:], skip_special_tokens=True)
                attempt_1 = parse_grid(_gen1, ref_shape)
                if _is_valid_result(attempt_1, inp):
                    _candidates.append(attempt_1)
            except Exception as exc:
                print(f"[GPU {rank}] {task_id} S1 err: {exc}", flush=True)

            # ── Strategy 2: QwenFormatter sampled T=0.8 → attempt_2 ─────────
            try:
                try:
                    from arc_loader import QwenFormatter as _QF2  # noqa: PLC0415
                except ImportError:
                    _QF2 = None  # type: ignore[assignment]

                if _QF2 is not None:
                    _qf2 = _QF2(tokenizer)
                    _p2  = _qf2.fmt_train(task_view["train"]) + _qf2.fmt_query([panel])
                    _ids2 = tokenizer(_p2, return_tensors="pt").input_ids.to("cuda:0")
                else:
                    _, _msgs2 = make_prompt(task_view, "nvarc")
                    _ids2 = _apply_chat(
                        tokenizer, _msgs2, enable_thinking=False
                    ).to("cuda:0")

                with torch.no_grad():
                    _out2 = model.generate(
                        _ids2,
                        max_new_tokens=32 * 31,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        logits_processor=_arc_lp_list,
                    )
                _gen2   = tokenizer.decode(_out2[0][_ids2.shape[-1]:], skip_special_tokens=True)
                attempt_2 = parse_grid(_gen2, ref_shape)
                if _is_valid_result(attempt_2, inp):
                    _candidates.append(attempt_2)
            except Exception as exc:
                print(f"[GPU {rank}] {task_id} S2 err: {exc}", flush=True)

            # ── Strategy 3: compact T=0.4 (fallback) ────────────────────────
            if attempt_1 is None or attempt_2 is None:
                try:
                    _sys3, _usr3 = make_prompt(task_view, "compact")
                    _msgs3 = [
                        {"role": "system", "content": _sys3},
                        {"role": "user",   "content": _usr3},
                    ]
                    _ids3 = _apply_chat(
                        tokenizer, _msgs3, enable_thinking=False
                    ).to("cuda:0")
                    with torch.no_grad():
                        _out3 = model.generate(
                            _ids3,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=0.4,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                            logits_processor=_arc_lp_list,
                        )
                    _gen3 = tokenizer.decode(_out3[0][_ids3.shape[-1]:], skip_special_tokens=True)
                    _fb3  = parse_grid(_gen3, ref_shape)
                    if _is_valid_result(_fb3, inp):
                        _candidates.append(_fb3)
                    if attempt_1 is None:
                        attempt_1 = _fb3
                    if attempt_2 is None:
                        attempt_2 = _fb3
                except Exception as exc:
                    print(f"[GPU {rank}] {task_id} S3 err: {exc}", flush=True)

            # ── Strategy 4: colored T=0.4 (second fallback) ─────────────────
            if attempt_1 is None or attempt_2 is None:
                try:
                    _sys4, _usr4 = make_prompt(task_view, "colored")
                    _msgs4 = [
                        {"role": "system", "content": _sys4},
                        {"role": "user",   "content": _usr4},
                    ]
                    _ids4 = _apply_chat(
                        tokenizer, _msgs4, enable_thinking=False
                    ).to("cuda:0")
                    with torch.no_grad():
                        _out4 = model.generate(
                            _ids4,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=0.4,
                            top_p=0.9,
                            pad_token_id=tokenizer.eos_token_id,
                            logits_processor=_arc_lp_list,
                        )
                    _gen4 = tokenizer.decode(_out4[0][_ids4.shape[-1]:], skip_special_tokens=True)
                    _fb4  = parse_grid(_gen4, ref_shape)
                    if _is_valid_result(_fb4, inp):
                        _candidates.append(_fb4)
                    if attempt_1 is None:
                        attempt_1 = _fb4
                    if attempt_2 is None:
                        attempt_2 = _fb4
                except Exception as exc:
                    print(f"[GPU {rank}] {task_id} S4 err: {exc}", flush=True)

            # ── D4 spatial-transform retry ───────────────────────────────────
            if not _is_valid_result(attempt_1, inp) or not _is_valid_result(attempt_2, inp):
                for _tname, _fwd, _inv in _TRANSFORM_OPS:
                    if (
                        _is_valid_result(attempt_1, inp)
                        and _is_valid_result(attempt_2, inp)
                    ):
                        break
                    try:
                        _tv_aug   = _transform_task_view(task_view, _fwd)
                        _aug_inp  = _tv_aug["test"][0]["input"]
                        _, _aug_msgs = make_prompt(_tv_aug, "nvarc")
                        _aug_ids  = _apply_chat(
                            tokenizer, _aug_msgs, enable_thinking=False
                        ).to("cuda:0")
                        with torch.no_grad():
                            _aug_out = model.generate(
                                _aug_ids,
                                max_new_tokens=32 * 31,
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                pad_token_id=tokenizer.eos_token_id,
                                logits_processor=_arc_lp_list,
                            )
                        _aug_gen  = tokenizer.decode(
                            _aug_out[0][_aug_ids.shape[-1]:], skip_special_tokens=True
                        )
                        _aug_shape = (len(_aug_inp), len(_aug_inp[0])) if _aug_inp and _aug_inp[0] else ref_shape
                        _aug_grid = parse_grid(_aug_gen, _aug_shape)
                        if _aug_grid is not None:
                            _restored = _inv(_aug_grid)
                            if _is_valid_result(_restored, inp):
                                _candidates.append(_restored)
                            if not _is_valid_result(attempt_1, inp):
                                attempt_1 = _restored
                            elif not _is_valid_result(attempt_2, inp):
                                attempt_2 = _restored
                    except Exception as _tex:
                        print(
                            f"[GPU {rank}] {task_id} D4 {_tname} err: {_tex}",
                            flush=True,
                        )

            # ── Majority vote + demo consistency (TbV) selection ─────────────
            if len(_candidates) >= 2:
                _mv = _pixel_majority_vote(_candidates)
                if _mv is not None and _is_valid_result(_mv, inp):
                    _train_pairs = task_data.get("train", [])
                    _scored = sorted(
                        [c for c in _candidates if c is not None],
                        key=lambda g: _demo_consistency_score(g, _train_pairs),
                        reverse=True,
                    )
                    _best_consistent = _scored[0] if _scored else _mv
                    attempt_1 = _mv  # majority-voted best
                    if _is_valid_result(_best_consistent, inp) and _best_consistent != _mv:
                        attempt_2 = _best_consistent  # shape-consistent alternative
                    print(
                        f"[GPU {rank}] {task_id} MV({len(_candidates)} cands) "
                        f"a1_ok={_is_valid_result(attempt_1, inp)} "
                        f"a2_ok={_is_valid_result(attempt_2, inp)}",
                        flush=True,
                    )

            # ── Strategy 5: TRM code generation (last resort) ────────────────
            _elapsed      = time.time() - _panel_start
            _budget_left  = _PANEL_BUDGET_SEC - int(_elapsed)
            if (
                (
                    not _is_valid_result(attempt_1, inp)
                    or not _is_valid_result(attempt_2, inp)
                )
                and _budget_left > 30
            ):
                signal.alarm(max(30, min(_budget_left, time_per_task + 10)))
                try:
                    _trm_data = {
                        "train": task_data.get("train", []),
                        "test":  panel,
                    }
                    _trm_res = recursive_trm_loop(
                        task_id,
                        _trm_data,
                        model,
                        tokenizer,
                        time_allowance=_budget_left,
                        max_recursion_depth=4,
                    )
                    if not _is_valid_result(attempt_1, inp):
                        _c1 = _trm_res["attempt_1"]
                        if _is_valid_result(_c1, inp):
                            attempt_1 = _c1
                            print(
                                f"[GPU {rank}] {task_id} TRM → attempt_1 OK",
                                flush=True,
                            )
                    if not _is_valid_result(attempt_2, inp):
                        _c2 = _trm_res["attempt_2"]
                        if _is_valid_result(_c2, inp):
                            attempt_2 = _c2
                            print(
                                f"[GPU {rank}] {task_id} TRM → attempt_2 OK",
                                flush=True,
                            )
                except (TimeoutException, Exception) as _trm_exc:
                    print(
                        f"[GPU {rank}] {task_id} TRM err "
                        f"({type(_trm_exc).__name__}): {_trm_exc}",
                        flush=True,
                    )
                finally:
                    signal.alarm(0)  # always cancel alarm

            # ── Diversity guard: force attempt_2 ≠ attempt_1 ─────────────────
            if (
                _is_valid_result(attempt_1, inp)
                and _is_valid_result(attempt_2, inp)
                and attempt_1 == attempt_2
            ):
                try:
                    _sys_dv, _usr_dv = make_prompt(task_view, "colored")
                    _msgs_dv = [
                        {"role": "system", "content": _sys_dv},
                        {"role": "user",   "content": _usr_dv},
                    ]
                    _ids_dv = _apply_chat(
                        tokenizer, _msgs_dv, enable_thinking=False
                    ).to("cuda:0")
                    with torch.no_grad():
                        _out_dv = model.generate(
                            _ids_dv,
                            max_new_tokens=32 * 31,
                            do_sample=True,
                            temperature=1.2,
                            top_p=0.95,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    _gen_dv = tokenizer.decode(
                        _out_dv[0][_ids_dv.shape[-1]:], skip_special_tokens=True
                    )
                    _dv_grid = parse_grid(_gen_dv, ref_shape)
                    if _is_valid_result(_dv_grid, inp) and _dv_grid != attempt_1:
                        attempt_2 = _dv_grid
                        print(
                            f"[GPU {rank}] {task_id} diversity guard → attempt_2 diversified",
                            flush=True,
                        )
                except Exception as _dv_exc:
                    print(
                        f"[GPU {rank}] {task_id} diversity guard err: {_dv_exc}",
                        flush=True,
                    )

            # ── Hard fallback: echo test input ───────────────────────────────
            if attempt_1 is None:
                attempt_1 = inp
            if attempt_2 is None:
                attempt_2 = attempt_1

            task_preds.append({"attempt_1": attempt_1, "attempt_2": attempt_2})

        worker_preds[task_id] = task_preds

        # Incremental save — crash recovery
        try:
            worker_out.parent.mkdir(parents=True, exist_ok=True)
            worker_out.write_text(json.dumps(worker_preds))
        except Exception:
            pass

        if (step + 1) % 10 == 0 or (step + 1) == len(chunk_tasks):
            elapsed = time.time() - t0
            rate    = (step + 1) / max(elapsed, 1e-9)
            eta     = (len(chunk_tasks) - step - 1) / max(rate, 1e-9)
            print(
                f"[GPU {rank}] {step+1}/{len(chunk_tasks)} | "
                f"{elapsed:.0f}s elapsed | ETA {eta:.0f}s",
                flush=True,
            )

    # Final write
    worker_out.parent.mkdir(parents=True, exist_ok=True)
    worker_out.write_text(json.dumps(worker_preds))
    print(f"[GPU {rank}] Done — {len(worker_preds)} predictions → {worker_out}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_trm_pass(
    tasks: dict,
    model_path: str,
    output_path: str,
    output_dir: str = "/kaggle/working",
    n_workers: int = 4,
    time_per_task: int = 60,
    deadline: float = 0.0,
) -> int:
    """
    Distribute tasks across n_workers GPU processes.
    Returns count of task predictions saved.
    """
    task_items = list(tasks.items())
    n = len(task_items)
    print(
        f"[trm] {n} tasks | {n_workers} GPU workers | model={model_path}",
        flush=True,
    )

    # Round-robin distribution
    chunks: list = [[] for _ in range(n_workers)]
    for i, item in enumerate(task_items):
        chunks[i % n_workers].append(item)

    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(n_workers):
        if not chunks[rank]:
            continue
        p = ctx.Process(
            target=worker_process,
            args=(rank, chunks[rank], model_path, output_dir, time_per_task),
            daemon=False,
        )
        p.start()
        processes.append((rank, p))

    for rank, p in processes:
        if deadline > 0.0:
            timeout = max(1.0, deadline - time.time())
            p.join(timeout=timeout)
            if p.is_alive():
                print(f"[trm] GPU{rank} deadline exceeded — terminating.", flush=True)
                p.terminate()
                p.join(timeout=30)
        else:
            p.join()
        print(f"[trm] GPU{rank} joined (exit={p.exitcode})", flush=True)

    # Aggregate results
    all_results: dict = {}
    for rank in range(n_workers):
        wf = Path(output_dir) / f"trm_worker_{rank}.json"
        if wf.exists():
            try:
                all_results.update(json.loads(wf.read_text()))
                print(f"[trm] Loaded worker {rank}: {wf}", flush=True)
            except Exception as exc:
                print(f"[trm] Could not load {wf}: {exc}", flush=True)
        else:
            print(f"[trm] Worker {rank} output missing: {wf}", flush=True)

    # ── Two-pass retry: re-run tasks where attempt_1 echoes the test input ──
    try:
        _echo_tasks = {
            tid: tasks[tid]
            for tid, preds in all_results.items()
            if tid in tasks
            and preds
            and preds[0].get("attempt_1") == tasks[tid].get("test", [{}])[0].get("input")
        }
        if _echo_tasks:
            print(
                f"[trm] Two-pass retry: {len(_echo_tasks)} echo tasks → re-running.",
                flush=True,
            )
            _retry_dir = str(Path(output_dir) / "trm_retry")
            Path(_retry_dir).mkdir(parents=True, exist_ok=True)
            _chunks2: list = [[] for _ in range(n_workers)]
            for _i2, _item2 in enumerate(_echo_tasks.items()):
                _chunks2[_i2 % n_workers].append(_item2)
            _ctx2 = mp.get_context("spawn")
            _procs2 = []
            for _rk2 in range(n_workers):
                if not _chunks2[_rk2]:
                    continue
                _p2 = _ctx2.Process(
                    target=worker_process,
                    args=(_rk2, _chunks2[_rk2], model_path, _retry_dir,
                          max(time_per_task, 90)),
                    daemon=False,
                )
                _p2.start()
                _procs2.append((_rk2, _p2))
            for _rk2, _p2 in _procs2:
                if deadline > 0.0:
                    _t2 = max(1.0, deadline - time.time())
                    _p2.join(timeout=_t2)
                    if _p2.is_alive():
                        _p2.terminate()
                        _p2.join(timeout=30)
                else:
                    _p2.join()
            for _rk2 in range(n_workers):
                _rf2 = Path(_retry_dir) / f"trm_worker_{_rk2}.json"
                if _rf2.exists():
                    try:
                        _rr = json.loads(_rf2.read_text())
                        for _tid2, _rp in _rr.items():
                            _orig_inp = tasks.get(_tid2, {}).get("test", [{}])[0].get("input")
                            if _rp and _rp[0].get("attempt_1") != _orig_inp:
                                all_results[_tid2] = _rp
                                print(f"[trm] Retry improved {_tid2}.", flush=True)
                    except Exception as _re2:
                        print(f"[trm] Retry load err rank {_rk2}: {_re2}", flush=True)
        else:
            print("[trm] Two-pass retry: no echo tasks — skipping.", flush=True)
    except Exception as _tp_exc:
        print(f"[trm] Two-pass retry error: {_tp_exc}", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(all_results))
    print(
        f"[trm] Saved {len(all_results)} task predictions → {output_path}",
        flush=True,
    )
    return len(all_results)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

_NVARC_MODEL_PATHS = [
    "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
]

_CODER_FALLBACK_PATHS: list = [
    "/kaggle/input/qwen2.5-coder-7b-instruct/transformers/default/1",
    "/kaggle/input/models/qwen/qwen2.5-coder-7b-instruct/transformers/default/1",
]


def discover_model_path() -> str:
    """
    Find the sorokin/qwen3_4b_grids15_sft139 model directory.
    Priority:
      1. ARC_MODEL_PATH env var
      2. Hard-coded Kaggle mount paths
      3. Glob scan /kaggle/input for config.json with qwen/arc in path
    """
    env_path = os.getenv("ARC_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        print(f"[trm] Model via ARC_MODEL_PATH: {env_path}", flush=True)
        return env_path

    for p in _NVARC_MODEL_PATHS:
        if Path(p).is_dir() and Path(p, "config.json").exists():
            print(f"[trm] Found model at {p}", flush=True)
            return p

    search_root = Path("/kaggle/input")
    if not search_root.exists():
        return ""

    candidates: list = []
    for entry in sorted(search_root.rglob("config.json")):
        try:
            cfg = json.loads(entry.read_text())
        except Exception:
            continue
        if "vocab_size" in cfg or "model_type" in cfg:
            candidates.append(entry.parent)

    for c in candidates:
        path_str = str(c).lower()
        if "qwen" in path_str or "arc" in path_str:
            print(f"[trm] Found model (scan) at {c}", flush=True)
            return str(c)

    if candidates:
        print(f"[trm] Using fallback model at {candidates[0]}", flush=True)
        return str(candidates[0])

    return ""


def discover_competition_models() -> tuple:
    """
    Scan /kaggle/input dynamically for both the Coder and SFT model directories.

    Priority per model:
      1. Glob scan /kaggle/input rglob("config.json") — validates model metadata.
      2. Hardcoded fallback paths (_CODER_FALLBACK_PATHS / _NVARC_MODEL_PATHS).
      3. Env var ARC_MODEL_PATH used as SFT override when set.

    Returns:
        (coder_path, sft_path) — either may be "" if not found.
    """
    coder_path: str = ""
    sft_path:   str = ""

    # Env var override for SFT (highest priority)
    _env = os.getenv("ARC_MODEL_PATH", "")
    if _env and Path(_env).exists():
        sft_path = _env
        print(f"[trm] SFT via ARC_MODEL_PATH: {sft_path}", flush=True)

    search_root = Path("/kaggle/input")
    if search_root.exists():
        for _cfg_file in sorted(search_root.rglob("config.json")):
            _parent = str(_cfg_file.parent).lower()
            try:
                _cfg = json.loads(_cfg_file.read_text())
            except Exception:
                continue
            # Only accept files that look like a real model config
            if "model_type" not in _cfg and "vocab_size" not in _cfg:
                continue
            if not coder_path and (
                "coder" in _parent or "qwen2.5-coder" in _parent
            ):
                coder_path = str(_cfg_file.parent)
                print(f"[trm] Coder model found: {coder_path}", flush=True)
            if not sft_path and (
                "grids15" in _parent or "sorokin" in _parent
            ):
                sft_path = str(_cfg_file.parent)
                print(f"[trm] SFT model found:   {sft_path}", flush=True)
            if coder_path and sft_path:
                break

    # Fallback: hardcoded coder paths
    if not coder_path:
        for _p in _CODER_FALLBACK_PATHS:
            if Path(_p).is_dir() and Path(_p, "config.json").exists():
                coder_path = _p
                print(f"[trm] Coder model (fallback): {coder_path}", flush=True)
                break

    # Fallback: hardcoded SFT paths
    if not sft_path:
        for _p in _NVARC_MODEL_PATHS:
            if Path(_p).is_dir() and Path(_p, "config.json").exists():
                sft_path = _p
                print(f"[trm] SFT model (fallback):   {sft_path}", flush=True)
                break

    # Last resort: full scan (any valid model config)
    if not sft_path and search_root.exists():
        for _cfg_file in sorted(search_root.rglob("config.json")):
            try:
                _cfg = json.loads(_cfg_file.read_text())
            except Exception:
                continue
            if "vocab_size" in _cfg or "model_type" in _cfg:
                sft_path = str(_cfg_file.parent)
                print(f"[trm] SFT model (last-resort scan): {sft_path}", flush=True)
                break

    if not coder_path:
        print(f"[trm] WARNING: Coder model path not found.", flush=True)
    if not sft_path:
        print(f"[trm] WARNING: SFT model path not found.", flush=True)

    return coder_path, sft_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRM orchestrator for ARC-AGI-2 (Session 60 Duck Harness edition)"
    )
    parser.add_argument("--model-path",     default=None,
                        help="Path to Qwen3-4B model dir. Defaults to ARC_MODEL_PATH env.")
    parser.add_argument("--challenge-file", default=None,
                        help="Path to ARC challenge JSON. Auto-discovered if omitted.")
    parser.add_argument("--output",         default="/kaggle/working/trm_preds.json",
                        help="Output path for TRM predictions.")
    parser.add_argument("--output-dir",     default="/kaggle/working",
                        help="Directory for per-worker intermediate JSON files.")
    parser.add_argument("--n-workers",      type=int, default=4,
                        help="Number of GPU workers (default 4 for 4xL4).")
    parser.add_argument("--time-per-task",  type=int, default=60,
                        help="Soft per-task time budget for TRM (seconds, default 60).")
    parser.add_argument("--end-time",       type=float, default=None,
                        help="Unix timestamp hard deadline. Script exits early if < 30 min remain.")
    args = parser.parse_args()

    # Time-budget check
    if args.end_time is not None:
        remaining = args.end_time - time.time()
        if remaining < 1800:  # < 30 min
            print(
                f"[trm] Only {remaining:.0f}s remaining — skipping TRM pass.",
                flush=True,
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text("{}")
            return

    # Model path — use discover_competition_models() so both models are resolved
    # in one scan pass; trm_orchestrator uses only the SFT path.
    _disc_coder, _disc_sft = discover_competition_models()
    model_path = args.model_path or _disc_sft
    if not model_path or not Path(model_path).exists():
        print("[trm] ERROR: model not found. Set --model-path or ARC_MODEL_PATH.", flush=True)
        sys.exit(1)

    # Challenge file
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
        print(f"[trm] ERROR: challenge file not found: {challenge_file}", flush=True)
        sys.exit(1)

    print(f"[trm] Challenge file: {challenge_file}", flush=True)
    tasks = json.loads(Path(challenge_file).read_text())

    # GPU count
    n_visible = torch.cuda.device_count()
    n_workers = min(args.n_workers, max(1, n_visible))
    print(
        f"[trm] CUDA devices visible: {n_visible}, using {n_workers} workers",
        flush=True,
    )

    _deadline = (args.end_time - 300) if args.end_time is not None else 0.0
    t0    = time.time()
    count = run_trm_pass(
        tasks,
        model_path,
        args.output,
        output_dir=args.output_dir,
        n_workers=n_workers,
        time_per_task=args.time_per_task,
        deadline=_deadline,
    )
    print(f"[trm] Finished: {count} tasks in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
