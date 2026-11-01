#!/usr/bin/env python3
"""
patch_disc.py  —  Session 60: Apply Duck Harness / discussion insights to jeni_llm_fast.py
Changes:
  1. Add _TRM_MAX_INPUT_TOKENS and _TRM_DEPTH_BUDGET_SEC constants
  2. Add get_connected_components to PythonREPL.globals
  3. Add check_demo_success() after PythonREPL class
  4. Add _extract_world_model() helper
  5. Replace recursive_trm_loop() with Duck-Harness-aware version:
       - world_model state carry-forward
       - context eviction at _TRM_MAX_INPUT_TOKENS
       - max_new_tokens 600 -> 1024
       - decode only newly generated tokens (shape-safe)
       - per-depth try/except isolation
       - wall-clock depth budget (yield, don't hang)
       - check_demo_success for partial diagnostics
       - feedback includes partial pass count
"""

from pathlib import Path

SRC = Path("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py")
text = SRC.read_text()

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1 — TRM constants (insert after "class PythonREPL" header comment block)
# ──────────────────────────────────────────────────────────────────────────────
OLD_CONST_ANCHOR = '# ── System prompts ─────────────────────────────────────────────────────────────'
NEW_CONSTS = '''\
# ── TRM budget constants (Duck Harness) ───────────────────────────────────────
_TRM_MAX_INPUT_TOKENS: int = 28_000   # evict context above this to prevent OOM
_TRM_DEPTH_BUDGET_SEC: float = 90.0   # wall-clock per depth; yield instead of hang

# ── System prompts ─────────────────────────────────────────────────────────────'''

assert OLD_CONST_ANCHOR in text, "ANCHOR for constants not found"
text = text.replace(OLD_CONST_ANCHOR, NEW_CONSTS, 1)

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2 — add get_connected_components to PythonREPL.globals
# ──────────────────────────────────────────────────────────────────────────────
OLD_REPL_GLOBALS = '''\
        self.globals = {
            "np": _np,
            "ndimage": _ndimage,
            "flood_fill": ARCDSL.flood_fill,
            "get_objects": ARCDSL.get_objects,
            "crop_to_content": ARCDSL.crop_to_content,
        }'''

NEW_REPL_GLOBALS = '''\
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
        }'''

assert OLD_REPL_GLOBALS in text, "ANCHOR for PythonREPL.globals not found"
text = text.replace(OLD_REPL_GLOBALS, NEW_REPL_GLOBALS, 1)

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 3 — add check_demo_success + _extract_world_model after verify_program_robustness
# ──────────────────────────────────────────────────────────────────────────────
OLD_VPR_END = '''\
    return True, "OK"


def recursive_trm_loop('''

NEW_HELPERS = '''\
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


def recursive_trm_loop('''

assert OLD_VPR_END in text, "ANCHOR for verify_program_robustness end not found"
text = text.replace(OLD_VPR_END, NEW_HELPERS, 1)

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 4 — replace recursive_trm_loop body with Duck-Harness version
# ──────────────────────────────────────────────────────────────────────────────
OLD_TRM_LOOP = '''\
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

        # 2. Validate against all training pairs + D4 symmetry
        all_passed, verify_msg = verify_program_robustness(
            repl, current_code, task_data["train"]
        )

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
            feedback_message = f"Program verification failed: {verify_msg}"

        print(
            f"[TRM] {task_id} depth={depth} all_passed={all_passed} "
            f"programs={len(successful_programs)}",
            flush=True,
        )

    # 4. Build submission candidates
    attempt_1 = successful_programs[0] if len(successful_programs) >= 1 else fallback
    attempt_2 = successful_programs[1] if len(successful_programs) >= 2 else attempt_1
    return {"attempt_1": attempt_1, "attempt_2": attempt_2}'''

NEW_TRM_LOOP = '''\
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
                _fb = f"World model: {world_model}\n{_fb}"
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
    return {"attempt_1": attempt_1, "attempt_2": attempt_2}'''

assert OLD_TRM_LOOP in text, "ANCHOR for recursive_trm_loop body not found"
text = text.replace(OLD_TRM_LOOP, NEW_TRM_LOOP, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Write patched file
# ──────────────────────────────────────────────────────────────────────────────
SRC.write_text(text)
print(f"[patch_disc] Wrote {SRC} ({len(text.splitlines())} lines)")
print("[patch_disc] All 4 patches applied successfully.")
