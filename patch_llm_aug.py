from pathlib import Path

src = Path("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py").read_text()

# ── 1. Add `import random` to imports ─────────────────────────────────────────
OLD1 = "import torch\nimport torch.multiprocessing as mp"
NEW1 = "import random\nimport torch\nimport torch.multiprocessing as mp"
assert OLD1 in src, "PATCH1 anchor not found"
src = src.replace(OLD1, NEW1, 1)

# ── 2. Add spatial transform helpers + fallback logic right after parse_grid ──
TRANSFORM_HELPERS = '''

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

'''

# Insert after the parse_grid function (before the _SYS_DIRECT constant)
ANCHOR2 = "# ── System prompts ─────────────────────────────────────────────────────────"
assert ANCHOR2 in src, "PATCH2 anchor not found"
src = src.replace(ANCHOR2, TRANSFORM_HELPERS + ANCHOR2, 1)

# ── 3. Insert transform retry block right before the hard fallback ─────────────
#    Anchor: the "# ── Hard fallback: test input" comment
RETRY_BLOCK = '''
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
                                    max_new_tokens=512,
                                    do_sample=False,
                                    temperature=None,
                                    top_p=None,
                                    pad_token_id=tokenizer.eos_token_id,
                                )
                            _aug_gen  = tokenizer.decode(_aug_out[0][_aug_ids.shape[-1]:], skip_special_tokens=True)
                            _aug_grid = parse_grid(_aug_gen, _aug_shape)
                            if _aug_grid is not None:
                                _restored = _inv(_aug_grid)
                                if not _is_valid_result(attempt_1, inp):
                                    attempt_1 = _restored
                                elif not _is_valid_result(attempt_2, inp):
                                    attempt_2 = _restored
                        except Exception as _tex:
                            print(f"[GPU{rank}] {task_id} transform {_tname} err: {_tex}", flush=True)
                            continue

'''

OLD3 = "                # ── Hard fallback: test input ─────────────────────────────────"
assert OLD3 in src, "PATCH3 anchor not found"
src = src.replace(OLD3, RETRY_BLOCK + OLD3, 1)

Path("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py").write_text(src)
print("jeni_llm_fast.py patched OK")
print(f"Lines: {src.count(chr(10))}")
