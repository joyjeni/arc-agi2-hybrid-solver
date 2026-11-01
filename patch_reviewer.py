#!/usr/bin/env python3
"""
patch_reviewer.py — Session 59 comprehensive reviewer fixes
============================================================
Applies all pending fixes to jeni_llm_fast.py AND notebook cells:

jeni_llm_fast.py:
  1a. HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE env vars in worker_inference
  1b. random.seed(42 + rank) for reproducible per-worker inference
  1c. Add deadline: float = 0.0 param to worker_inference
  1d. Strategy 1 → QwenFormatter (SFT training format), max_new_tokens=32*31, parse_grid None
  1e. Strategy 2 → QwenFormatter sampled, max_new_tokens=32*31, parse_grid None
  1f. Strategies 3,4 max_new_tokens=512 → 32*31, parse_grid ref_shape → None
  1g. force_style max_new_tokens + parse_grid fix
  1h. transform-retry parse_grid(_aug_gen, _aug_shape) → parse_grid(_aug_gen, None)
  1i. After results[task_id] = task_preds: incremental save + deadline check
  1j. run_llm_fast_pass: add deadline param, pass to workers
  1k. main(): pass deadline = end_time - 300 to run_llm_fast_pass
  1l. _trm_format_prompt: add Vedic + Japanese/Chinese solving strategies

Notebook cells:
  2a. Cell 2: 12*3600-900 → 12*3600-45*60  (15-min → 45-min margin)
  2b. Cell 9: add HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE to env dict
  2c. Cell 12 (DSA): signal.alarm(20) guard around solver.solve(ctx)
  2d. Cell 15: set HF_HUB_OFFLINE before subprocess call
  2e. Cell 16: wrap numpy arrays with .tolist(); trim predictions to len(test)
"""
import json
import sys
from pathlib import Path

LLM_FILE = Path("/home/sandbox/arc-jeni-2026/jeni_llm_fast.py")
NB_FILE  = Path("/home/sandbox/arc-jeni-2026/notebooks/arc_jeni_submission.ipynb")

errors = []

def patch(src: str, old: str, new: str, label: str) -> str:
    if old not in src:
        errors.append(f"WARN: {label} — pattern not found")
        return src
    count = src.count(old)
    result = src.replace(old, new, 1)
    print(f"  OK [{label}] (found {count}x, replaced 1)")
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: jeni_llm_fast.py
# ═══════════════════════════════════════════════════════════════════════════════
src = LLM_FILE.read_text()
print(f"\n=== Patching jeni_llm_fast.py ({len(src)} chars) ===")

# ── 1a. HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE ─────────────────────────────────
src = patch(src,
    '    os.environ["TOKENIZERS_PARALLELISM"] = "false"\n\n    results: dict = {}',
    '    os.environ["TOKENIZERS_PARALLELISM"] = "false"\n'
    '    os.environ["HF_HUB_OFFLINE"] = "1"\n'
    '    os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
    '\n    results: dict = {}',
    "1a: HF_HUB_OFFLINE+TRANSFORMERS_OFFLINE"
)

# ── 1b. random.seed for reproducibility ───────────────────────────────────────
src = patch(src,
    '    results: dict = {}\n\n    try:',
    '    results: dict = {}\n'
    '    random.seed(42 + rank)  # reproducible per-worker inference\n'
    '\n    try:',
    "1b: random.seed(42+rank)"
)

# ── 1c. deadline param in worker_inference ────────────────────────────────────
src = patch(src,
    'def worker_inference(\n'
    '    rank: int,\n'
    '    task_items: list,\n'
    '    model_path: str,\n'
    '    worker_out_file: str,\n'
    ') -> None:',
    'def worker_inference(\n'
    '    rank: int,\n'
    '    task_items: list,\n'
    '    model_path: str,\n'
    '    worker_out_file: str,\n'
    '    deadline: float = 0.0,\n'
    ') -> None:',
    "1c: deadline param"
)

# ── 1d. Strategy 1: QwenFormatter, max_new_tokens=32*31, parse_grid(None) ──────
src = patch(src,
    '                # ── Strategy 1: greedy NVARC format → attempt_1 ─────────────────\n'
    '                # Uses the exact SFT training format: compact multi-turn, no system msg\n'
    '                try:\n'
    '                    _style_ret = make_prompt(task_view, "nvarc")\n'
    '                    if _style_ret[0] == "__NVARC_MULTITURN__":\n'
    '                        msgs = _style_ret[1]\n'
    '                    else:\n'
    '                        sys_m, usr_m = _style_ret\n'
    '                        msgs = [\n'
    '                            {"role": "system", "content": sys_m},\n'
    '                            {"role": "user",   "content": usr_m},\n'
    '                        ]\n'
    '                    ids = _apply_chat(tokenizer, msgs, enable_thinking=False).to("cuda:0")\n'
    '                    with torch.no_grad():\n'
    '                        out = model.generate(\n'
    '                            ids,\n'
    '                            max_new_tokens=512,\n'
    '                            do_sample=False,\n'
    '                            temperature=None,\n'
    '                            top_p=None,\n'
    '                            pad_token_id=tokenizer.eos_token_id,\n'
    '                        )\n'
    '                    gen = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)\n'
    '                    attempt_1 = parse_grid(gen, ref_shape)\n'
    '                except Exception as exc:\n'
    '                    print(f"[GPU{rank}] {task_id} direct err: {exc}", flush=True)',

    '                # ── Strategy 1: QwenFormatter SFT format → attempt_1 (greedy) ──────\n'
    '                # Model SFT-trained on QwenFormatter format — bypass apply_chat_template.\n'
    '                try:\n'
    '                    from arc_loader import QwenFormatter as _QF1  # noqa: PLC0415\n'
    '                    _qf1 = _QF1(tokenizer)\n'
    '                    _prompt1 = _qf1.fmt_train(task_view["train"]) + _qf1.fmt_query([panel])\n'
    '                    ids = tokenizer(_prompt1, return_tensors="pt").input_ids.to("cuda:0")\n'
    '                    with torch.no_grad():\n'
    '                        out = model.generate(\n'
    '                            ids,\n'
    '                            max_new_tokens=32 * 31,\n'
    '                            do_sample=False,\n'
    '                            temperature=None,\n'
    '                            top_p=None,\n'
    '                            pad_token_id=tokenizer.eos_token_id,\n'
    '                        )\n'
    '                    gen = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)\n'
    '                    attempt_1 = parse_grid(gen, None)\n'
    '                except Exception as exc:\n'
    '                    print(f"[GPU{rank}] {task_id} direct err: {exc}", flush=True)',
    "1d: Strategy 1 QwenFormatter+max_new_tokens+parse_grid"
)

# ── 1e. Strategy 2: QwenFormatter sampled ─────────────────────────────────────
src = patch(src,
    '                # ── Strategy 2: NVARC sampled T=0.8 → attempt_2 ──────────────\n'
    '                # SFT model trained exclusively on nvarc format — use same format\n'
    '                # with sampling for diversity vs greedy attempt_1\n'
    '                try:\n'
    '                    _style_ret2 = make_prompt(task_view, "nvarc")\n'
    '                    if _style_ret2[0] == "__NVARC_MULTITURN__":\n'
    '                        msgs2 = _style_ret2[1]\n'
    '                    else:\n'
    '                        sys_m2, usr_m2 = _style_ret2\n'
    '                        msgs2 = [\n'
    '                            {"role": "system", "content": sys_m2},\n'
    '                            {"role": "user",   "content": usr_m2},\n'
    '                        ]\n'
    '                    ids2 = _apply_chat(tokenizer, msgs2, enable_thinking=False).to("cuda:0")\n'
    '                    with torch.no_grad():\n'
    '                        out2 = model.generate(\n'
    '                            ids2,\n'
    '                            max_new_tokens=512,\n'
    '                            do_sample=True,\n'
    '                            temperature=0.8,\n'
    '                            top_p=0.95,\n'
    '                            pad_token_id=tokenizer.eos_token_id,\n'
    '                        )\n'
    '                    gen2 = tokenizer.decode(out2[0][ids2.shape[-1]:], skip_special_tokens=True)\n'
    '                    attempt_2 = parse_grid(gen2, ref_shape)\n'
    '                except Exception as exc:\n'
    '                    print(f"[GPU{rank}] {task_id} nvarc_s2 err: {exc}", flush=True)',

    '                # ── Strategy 2: QwenFormatter sampled T=0.8 → attempt_2 diversity ─\n'
    '                try:\n'
    '                    from arc_loader import QwenFormatter as _QF2  # noqa: PLC0415\n'
    '                    _qf2 = _QF2(tokenizer)\n'
    '                    _prompt2 = _qf2.fmt_train(task_view["train"]) + _qf2.fmt_query([panel])\n'
    '                    ids2 = tokenizer(_prompt2, return_tensors="pt").input_ids.to("cuda:0")\n'
    '                    with torch.no_grad():\n'
    '                        out2 = model.generate(\n'
    '                            ids2,\n'
    '                            max_new_tokens=32 * 31,\n'
    '                            do_sample=True,\n'
    '                            temperature=0.8,\n'
    '                            top_p=0.95,\n'
    '                            pad_token_id=tokenizer.eos_token_id,\n'
    '                        )\n'
    '                    gen2 = tokenizer.decode(out2[0][ids2.shape[-1]:], skip_special_tokens=True)\n'
    '                    attempt_2 = parse_grid(gen2, None)\n'
    '                except Exception as exc:\n'
    '                    print(f"[GPU{rank}] {task_id} nvarc_s2 err: {exc}", flush=True)',
    "1e: Strategy 2 QwenFormatter+max_new_tokens+parse_grid"
)

# ── 1f. Strategy 3: max_new_tokens + parse_grid ───────────────────────────────
src = patch(src,
    '                        with torch.no_grad():\n'
    '                            out3 = model.generate(\n'
    '                                ids3,\n'
    '                                max_new_tokens=512,\n'
    '                                do_sample=True,\n'
    '                                temperature=0.4,\n'
    '                                top_p=0.9,\n'
    '                                pad_token_id=tokenizer.eos_token_id,\n'
    '                            )\n'
    '                        gen3 = tokenizer.decode(out3[0][ids3.shape[-1]:], skip_special_tokens=True)\n'
    '                        fb = parse_grid(gen3, ref_shape)',

    '                        with torch.no_grad():\n'
    '                            out3 = model.generate(\n'
    '                                ids3,\n'
    '                                max_new_tokens=32 * 31,\n'
    '                                do_sample=True,\n'
    '                                temperature=0.4,\n'
    '                                top_p=0.9,\n'
    '                                pad_token_id=tokenizer.eos_token_id,\n'
    '                            )\n'
    '                        gen3 = tokenizer.decode(out3[0][ids3.shape[-1]:], skip_special_tokens=True)\n'
    '                        fb = parse_grid(gen3, None)',
    "1f: Strategy 3 max_new_tokens+parse_grid"
)

# ── 1g. Strategy 4: max_new_tokens + parse_grid ───────────────────────────────
src = patch(src,
    '                        with torch.no_grad():\n'
    '                            out4 = model.generate(\n'
    '                                ids4,\n'
    '                                max_new_tokens=512,\n'
    '                                do_sample=True,\n'
    '                                temperature=0.4,\n'
    '                                top_p=0.9,\n'
    '                                pad_token_id=tokenizer.eos_token_id,\n'
    '                            )\n'
    '                        gen4 = tokenizer.decode(out4[0][ids4.shape[-1]:], skip_special_tokens=True)\n'
    '                        fb4 = parse_grid(gen4, ref_shape)',

    '                        with torch.no_grad():\n'
    '                            out4 = model.generate(\n'
    '                                ids4,\n'
    '                                max_new_tokens=32 * 31,\n'
    '                                do_sample=True,\n'
    '                                temperature=0.4,\n'
    '                                top_p=0.9,\n'
    '                                pad_token_id=tokenizer.eos_token_id,\n'
    '                            )\n'
    '                        gen4 = tokenizer.decode(out4[0][ids4.shape[-1]:], skip_special_tokens=True)\n'
    '                        fb4 = parse_grid(gen4, None)',
    "1g: Strategy 4 max_new_tokens+parse_grid"
)

# ── 1h. force_style: max_new_tokens + parse_grid ──────────────────────────────
src = patch(src,
    '                        _fs_ids = _apply_chat(tokenizer, _fs_msgs, enable_thinking=False).to("cuda:0")\n'
    '                        with torch.no_grad():\n'
    '                            _fs_out = model.generate(_fs_ids, max_new_tokens=512,\n'
    '                                do_sample=False, temperature=None, top_p=None,\n'
    '                                pad_token_id=tokenizer.eos_token_id)\n'
    '                        _fs_gen  = tokenizer.decode(_fs_out[0][_fs_ids.shape[-1]:], skip_special_tokens=True)\n'
    '                        _fs_grid = parse_grid(_fs_gen, ref_shape)',

    '                        _fs_ids = _apply_chat(tokenizer, _fs_msgs, enable_thinking=False).to("cuda:0")\n'
    '                        with torch.no_grad():\n'
    '                            _fs_out = model.generate(_fs_ids, max_new_tokens=32 * 31,\n'
    '                                do_sample=False, temperature=None, top_p=None,\n'
    '                                pad_token_id=tokenizer.eos_token_id)\n'
    '                        _fs_gen  = tokenizer.decode(_fs_out[0][_fs_ids.shape[-1]:], skip_special_tokens=True)\n'
    '                        _fs_grid = parse_grid(_fs_gen, None)',
    "1h: force_style max_new_tokens+parse_grid"
)

# ── 1i. transform-retry: max_new_tokens + parse_grid(_aug_gen, None) ──────────
src = patch(src,
    '                            with torch.no_grad():\n'
    '                                _aug_out = model.generate(\n'
    '                                    _aug_ids,\n'
    '                                    max_new_tokens=512,\n'
    '                                    do_sample=False,\n'
    '                                    temperature=None,\n'
    '                                    top_p=None,\n'
    '                                    pad_token_id=tokenizer.eos_token_id,\n'
    '                                )\n'
    '                            _aug_gen  = tokenizer.decode(_aug_out[0][_aug_ids.shape[-1]:], skip_special_tokens=True)\n'
    '                            _aug_grid = parse_grid(_aug_gen, _aug_shape)',

    '                            with torch.no_grad():\n'
    '                                _aug_out = model.generate(\n'
    '                                    _aug_ids,\n'
    '                                    max_new_tokens=32 * 31,\n'
    '                                    do_sample=False,\n'
    '                                    temperature=None,\n'
    '                                    top_p=None,\n'
    '                                    pad_token_id=tokenizer.eos_token_id,\n'
    '                                )\n'
    '                            _aug_gen  = tokenizer.decode(_aug_out[0][_aug_ids.shape[-1]:], skip_special_tokens=True)\n'
    '                            _aug_grid = parse_grid(_aug_gen, None)',
    "1i: transform-retry max_new_tokens+parse_grid(None)"
)

# ── 1j. incremental save + deadline check after results[task_id] = task_preds ─
src = patch(src,
    '            results[task_id] = task_preds\n'
    '\n'
    '            if (step + 1) % 10 == 0 or (step + 1) == len(task_items):',

    '            results[task_id] = task_preds\n'
    '\n'
    '            # Incremental save: crash recovery — write after every task\n'
    '            try:\n'
    '                with open(worker_out_file, "w") as _cf:\n'
    '                    json.dump(results, _cf)\n'
    '            except Exception:\n'
    '                pass\n'
    '            # Hard deadline guard\n'
    '            if deadline > 0.0 and time.time() > deadline:\n'
    '                print(f"[GPU{rank}] Deadline reached after {step+1}/{len(task_items)} tasks.", flush=True)\n'
    '                break\n'
    '\n'
    '            if (step + 1) % 10 == 0 or (step + 1) == len(task_items):',
    "1j: incremental save + deadline break"
)

# ── 1k. run_llm_fast_pass: add deadline param ─────────────────────────────────
src = patch(src,
    'def run_llm_fast_pass(\n'
    '    tasks: dict,\n'
    '    model_path: str,\n'
    '    output_path: str,\n'
    '    n_workers: int = 4,\n'
    ') -> int:',

    'def run_llm_fast_pass(\n'
    '    tasks: dict,\n'
    '    model_path: str,\n'
    '    output_path: str,\n'
    '    n_workers: int = 4,\n'
    '    deadline: float = 0.0,\n'
    ') -> int:',
    "1k: run_llm_fast_pass deadline param"
)

# ── 1l. Pass deadline to each worker process ──────────────────────────────────
src = patch(src,
    '        p = ctx.Process(\n'
    '            target=worker_inference,\n'
    '            args=(rank, chunks[rank], model_path, worker_files[rank]),\n'
    '            daemon=False,\n'
    '        )',

    '        p = ctx.Process(\n'
    '            target=worker_inference,\n'
    '            args=(rank, chunks[rank], model_path, worker_files[rank], deadline),\n'
    '            daemon=False,\n'
    '        )',
    "1l: pass deadline to worker Process args"
)

# ── 1m. main(): compute deadline and pass to run_llm_fast_pass ────────────────
src = patch(src,
    '    count = run_llm_fast_pass(tasks, model_path, args.output, n_workers=n_workers)',

    '    _deadline = (args.end_time - 300) if args.end_time is not None else 0.0\n'
    '    count = run_llm_fast_pass(tasks, model_path, args.output, n_workers=n_workers, deadline=_deadline)',
    "1m: main() deadline=end_time-300"
)

# ── 1n. _trm_format_prompt: add Vedic + Japanese/Chinese solving strategies ────
src = patch(src,
    '    prompt = (\n'
    '        "<|user|>\\nYou are a high-level recursive program synthesizer solving an ARC visual puzzle.\\n"\n'
    '        "Write a Python function named `transform(grid: list[list[int]]) -> list[list[int]]` using numpy.\\n\\n"\n'
    '    )',

    '    prompt = (\n'
    '        "<|user|>\\nYou are a high-level recursive program synthesizer solving an ARC visual puzzle.\\n"\n'
    '        "Write a Python function named `transform(grid: list[list[int]]) -> list[list[int]]` using numpy.\\n\\n"\n'
    '        "Solving strategies (apply the most relevant one):\\n"\n'
    '        "• Vedic Anurupyena (proportionality): detect ratio-based scaling / tiling rules.\\n"\n'
    '        "• Vedic Sunyam Samyasamuccaye (symmetry): find mirror / palindrome / rotation structure.\\n"\n'
    '        "• Vedic Vilokanam (direct observation): spot the simplest single-rule transform.\\n"\n'
    '        "• Japanese Kaizen (iterative steps): look for a rule applied cell-by-cell or row-by-row.\\n"\n'
    '        "• Japanese Kanban (pull-based): each output cell is determined by its colored neighbor.\\n"\n'
    '        "• Chinese Nine-Palace (九宫格): map 3×3 block spatial relationships and symmetry axes.\\n"\n'
    '        "• Nature-inspired: detect flood-fill propagation, cellular automata, or fractal self-similarity.\\n\\n"\n'
    '    )',
    "1n: Vedic+Japanese+Chinese strategies in TRM prompt"
)

# ── Verify no stale max_new_tokens=512 remain in worker_inference ─────────────
# Count remaining occurrences in the relevant region
worker_start = src.find("def worker_inference(")
worker_end   = src.find("\ndef run_llm_fast_pass(")
worker_block = src[worker_start:worker_end] if worker_start > 0 and worker_end > 0 else ""
remaining_512 = worker_block.count("max_new_tokens=512")
if remaining_512:
    errors.append(f"WARN: {remaining_512} occurrences of max_new_tokens=512 still in worker_inference")
else:
    print("  OK [check] no max_new_tokens=512 remain in worker_inference")

remaining_ref_shape = worker_block.count("parse_grid(gen, ref_shape)") + \
                      worker_block.count("parse_grid(gen2, ref_shape)") + \
                      worker_block.count("parse_grid(gen3, ref_shape)") + \
                      worker_block.count("parse_grid(gen4, ref_shape)") + \
                      worker_block.count("parse_grid(_fs_gen, ref_shape)") + \
                      worker_block.count("parse_grid(_aug_gen, _aug_shape)")
if remaining_ref_shape:
    errors.append(f"WARN: {remaining_ref_shape} stale parse_grid(ref_shape/_aug_shape) remain")
else:
    print("  OK [check] no stale parse_grid(ref_shape) remain in worker_inference")

LLM_FILE.write_text(src)
print(f"  → Written {len(src)} chars to {LLM_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: Notebook cells
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== Patching notebook ({NB_FILE}) ===")
nb = json.loads(NB_FILE.read_text())
cells = nb["cells"]

def patch_cell(cells, idx, old, new, label):
    src = "".join(cells[idx]["source"])
    if old not in src:
        errors.append(f"WARN: {label} — pattern not found in cell {idx+1}")
        return
    src = src.replace(old, new, 1)
    cells[idx]["source"] = [src]
    print(f"  OK [{label}] cell {idx+1}")

# ── 2a. Cell 2: 45-min margin ─────────────────────────────────────────────────
patch_cell(cells, 1,
    "global_end_time = T0 + (12 * 3600 - 900 if RERUN else 32 * 60)",
    "global_end_time = T0 + (12 * 3600 - 45 * 60 if RERUN else 32 * 60)",
    "2a: Cell 2 45-min margin"
)

# ── 2b. Cell 9: HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE ────────────────────────
patch_cell(cells, 8,
    '    ARC_MODEL_PATH="/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",\n)',
    '    ARC_MODEL_PATH="/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",\n'
    '    HF_HUB_OFFLINE="1",\n'
    '    TRANSFORMERS_OFFLINE="1",\n'
    ')',
    "2b: Cell 9 HF_HUB_OFFLINE"
)

# ── 2c. Cell 12 (DSA): signal.alarm(20) guard around solver.solve(ctx) ─────────
# The solver loop is at the very end of the giant cell. Need to add signal import
# and wrap the solve call.
c12_src = "".join(cells[11]["source"])

# First check if signal is already imported in the cell
if "import signal" not in c12_src:
    # Add signal import near the top of the cell's runner function block
    c12_src = c12_src.replace(
        "def _jeni_run_dsa_solvers(",
        "import signal as _signal_mod\n\ndef _jeni_run_dsa_solvers(",
        1
    )
    print("  OK [2c-pre] added import signal to cell 12")
else:
    print("  OK [2c-pre] signal already imported in cell 12")

# Wrap solver.solve(ctx) with alarm
OLD_SOLVE = (
    '        for solver in solvers:\n'
    '            try:\n'
    '                res = solver.solve(ctx)\n'
    '            except Exception:\n'
    '                continue'
)
NEW_SOLVE = (
    '        for solver in solvers:\n'
    '            try:\n'
    '                _signal_mod.signal(_signal_mod.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("DSA solver timeout")))\n'
    '                _signal_mod.alarm(20)\n'
    '                res = solver.solve(ctx)\n'
    '                _signal_mod.alarm(0)\n'
    '            except (TimeoutError, Exception):\n'
    '                _signal_mod.alarm(0)\n'
    '                continue'
)
if OLD_SOLVE in c12_src:
    c12_src = c12_src.replace(OLD_SOLVE, NEW_SOLVE, 1)
    print("  OK [2c: Cell 12 signal.alarm(20) guard]")
else:
    errors.append("WARN: 2c: Cell 12 solver loop pattern not found")

cells[11]["source"] = [c12_src]

# ── 2d. Cell 15: HF_HUB_OFFLINE before subprocess call ───────────────────────
patch_cell(cells, 14,
    'print(f"[llm-runner] Launching LLM fast pass …", flush=True)\n'
    '_rc = subprocess.call(_cmd)',

    'print(f"[llm-runner] Launching LLM fast pass …", flush=True)\n'
    'os.environ["HF_HUB_OFFLINE"] = "1"\n'
    'os.environ["TRANSFORMERS_OFFLINE"] = "1"\n'
    '_rc = subprocess.call(_cmd)',
    "2d: Cell 15 HF_HUB_OFFLINE before subprocess"
)

# ── 2e. Cell 16: numpy tolist + trim to len(test) ─────────────────────────────
# Patch 1: wrap DSA attempt_1 predictions with tolist() guard
patch_cell(cells, 15,
    '        dsa_pred = json.loads(_dsa_path.read_text())\n'
    '        merged   = 0\n'
    '        added    = 0\n'
    '        for tid, dsa_outputs in dsa_pred.items():',

    '        dsa_pred = json.loads(_dsa_path.read_text())\n'
    '        merged   = 0\n'
    '        added    = 0\n'
    '        # Ensure all DSA predictions are plain Python lists (not numpy arrays)\n'
    '        def _to_list(g):\n'
    '            if hasattr(g, "tolist"): return g.tolist()\n'
    '            if isinstance(g, list): return [[int(c) for c in r] for r in g]\n'
    '            return g\n'
    '        for tid, dsa_outputs in dsa_pred.items():',
    "2e-i: Cell 16 numpy tolist helper"
)

# Patch 2: apply _to_list when writing DSA predictions + trim to len(test)
patch_cell(cells, 15,
    '                nvarc_a1 = nvarc_entry.get("attempt_1", [])\n'
    '                nvarc_a2 = nvarc_entry.get("attempt_2", [])\n'
    '                dsa_a1   = dsa_out.get("attempt_1", [])',

    '                nvarc_a1 = nvarc_entry.get("attempt_1", [])\n'
    '                nvarc_a2 = nvarc_entry.get("attempt_2", [])\n'
    '                dsa_a1   = _to_list(dsa_out.get("attempt_1", []))',
    "2e-ii: Cell 16 dsa_a1 tolist"
)

# Patch 3: trim each task's predictions to len(test) after DSA merge
patch_cell(cells, 15,
    '        _sub_path.write_text(json.dumps(sub))\n'
    '        print(f"[jeni-dsa] Merge done — added {added} new tasks, updated attempt_2 in {merged} outputs")',

    '        # Trim over-length predictions to match test panel count\n'
    '        _challenges_for_trim = {}\n'
    '        try:\n'
    '            _td = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"\n'
    '            if not Path(_td).is_dir(): _td = "/kaggle/input/arc-prize-2026-arc-agi-2"\n'
    '            _cp = Path(_td) / ("arc-agi_test_challenges.json" if RERUN else "arc-agi_evaluation_challenges.json")\n'
    '            if _cp.exists(): _challenges_for_trim = json.loads(_cp.read_text())\n'
    '        except Exception: pass\n'
    '        if _challenges_for_trim:\n'
    '            for _tid in list(sub.keys()):\n'
    '                _n_test = len(_challenges_for_trim.get(_tid, {}).get("test", [sub[_tid]]))\n'
    '                sub[_tid] = sub[_tid][:_n_test]\n'
    '        _sub_path.write_text(json.dumps(sub))\n'
    '        print(f"[jeni-dsa] Merge done — added {added} new tasks, updated attempt_2 in {merged} outputs")',
    "2e-iii: Cell 16 trim to len(test)"
)

NB_FILE.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"  → Written notebook to {NB_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Summary ===")
if errors:
    for e in errors:
        print(e)
else:
    print("All patches applied successfully — no warnings.")
sys.exit(1 if errors else 0)
