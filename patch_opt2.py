#!/usr/bin/env python3
"""
patch_opt2.py — Apply user's Session 58 optimizations to jeni_llm_fast.py
Patches:
  1. Add ARCDSL class before PythonREPL (DSL primitives)
  2. Enrich PythonREPL.globals with ndimage + ARCDSL methods
  3. Add verify_program_robustness() after _trm_extract_code
  4. Update recursive_trm_loop validation to use verify_program_robustness
"""
import re, pathlib, sys

src = pathlib.Path("jeni_llm_fast.py").read_text()

# ─── Patch 1: Insert ARCDSL class before PythonREPL ──────────────────────────
ARCDSL_CLASS = '''
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


'''

ANCHOR_REPL = "class PythonREPL:"
if "class ARCDSL:" not in src:
    src = src.replace(ANCHOR_REPL, ARCDSL_CLASS + ANCHOR_REPL)
    print("Patch 1 OK: ARCDSL class inserted")
else:
    print("Patch 1 SKIP: ARCDSL already present")

# ─── Patch 2: Enrich PythonREPL.globals ───────────────────────────────────────
OLD_GLOBALS = '''    def __init__(self):
        self.globals = {"np": __import__("numpy")}
        try:
            self.globals["scipy"] = __import__("scipy")
        except ImportError:
            pass'''

NEW_GLOBALS = '''    def __init__(self):
        import numpy as _np
        try:
            from scipy import ndimage as _ndimage
        except ImportError:
            _ndimage = None
        self.globals = {
            "np": _np,
            "ndimage": _ndimage,
            "flood_fill": ARCDSL.flood_fill,
            "get_objects": ARCDSL.get_objects,
            "crop_to_content": ARCDSL.crop_to_content,
        }'''

if OLD_GLOBALS in src:
    src = src.replace(OLD_GLOBALS, NEW_GLOBALS)
    print("Patch 2 OK: PythonREPL.globals enriched with ARCDSL primitives")
else:
    print("Patch 2 SKIP: globals already updated or pattern not found")

# ─── Patch 3: Insert verify_program_robustness after _trm_extract_code ────────
VERIFY_FN = '''

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

'''

ANCHOR_VERIFY = "\ndef recursive_trm_loop("
if "def verify_program_robustness" not in src:
    src = src.replace(ANCHOR_VERIFY, VERIFY_FN + "\ndef recursive_trm_loop(")
    print("Patch 3 OK: verify_program_robustness inserted")
else:
    print("Patch 3 SKIP: verify_program_robustness already present")

# ─── Patch 4: Use verify_program_robustness inside recursive_trm_loop ─────────
OLD_LOOP = """        # 2. Validate against all training pairs in REPL
        all_passed = True
        failed_idx = -1
        observed_out = None

        for idx, pair in enumerate(task_data[\"train\"]):
            test_res = repl.execute_code(current_code, pair[\"input\"])
            if test_res is None or test_res != pair[\"output\"]:
                all_passed = False
                failed_idx = idx
                observed_out = test_res
                break

        # 3. Evaluate recursion state
        if all_passed:"""

NEW_LOOP = """        # 2. Validate against all training pairs + D4 symmetry
        all_passed, verify_msg = verify_program_robustness(
            repl, current_code, task_data[\"train\"]
        )

        # 3. Evaluate recursion state
        if all_passed:"""

if OLD_LOOP in src:
    src = src.replace(OLD_LOOP, NEW_LOOP)
    print("Patch 4 OK: recursive_trm_loop now uses verify_program_robustness")
else:
    print("Patch 4 SKIP: loop pattern not found — may already be updated")

# ─── Patch 5: Fix feedback_message to use verify_msg ─────────────────────────
OLD_FB = """        else:
            if observed_out is None:
                feedback_message = (
                    f\"SyntaxError or execution crash on Example {failed_idx + 1}.\"
                )
            else:
                feedback_message = (
                    f\"Logic mismatch on Example {failed_idx + 1}: \"
                    f\"generated output shape/values did not match expected.\"
                )"""

NEW_FB = """        else:
            feedback_message = f\"Program verification failed: {verify_msg}\""""

if OLD_FB in src:
    src = src.replace(OLD_FB, NEW_FB)
    print("Patch 5 OK: feedback_message now uses verify_msg")
else:
    print("Patch 5 SKIP: feedback pattern not found")

# ─── Write patched file ────────────────────────────────────────────────────────
pathlib.Path("jeni_llm_fast.py").write_text(src)
print(f"\nPatched file written: jeni_llm_fast.py ({len(src.splitlines())} lines)")
